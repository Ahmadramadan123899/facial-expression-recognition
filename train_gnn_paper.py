#! /usr/bin/python3
# Author : Ahmad Ramadan
"""
train_gnn_paper.py  —  Main training script for GCNTransformerPaper.

Usage (example):
  python3 train_gnn_paper.py \
      --graph_type g_topologique \
      --chemin_patches /sandbox1/aramadan/patches \
      --num_epochs 120 \
      --lr 1e-4 \
      --lambda_center 0.001

All other args use paper-aligned defaults (see argparse section at bottom).
"""

import argparse
import copy
import gc
import json
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy + random in each DataLoader worker for reproducibility.
    torch.initial_seed() already gives a unique deterministic value per worker
    (derived from the main seed), so worker_id offset is redundant but harmless.
    """
    seed = int(torch.initial_seed()) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# Local imports
import sys, importlib

from dataloader.build_dataset_paper import (
    convertion_graphe, convertir_liste_patches_en_tenseur,
    convertir_liste_videos_en_tenseur, create_graph,
    data_augmentation_equilibration_classe, extract_datas_video_labels,
    make_dirs, save_pickle, load_pickle,
    traitements_video_par_video, build_flip_map_for_subset
)
from dataloader.lazy_dataset import LazyFERDataset, scan_samples
from models.gnn_paper import CenterLoss, GCNTransformerPaper
from executor.trainer_paper import train_model_gnn_paper
from results import save_results, save_results_all_folds
from datasets import get_adapter
from utils import *



def chemin(chemin_depot_global, nom_dataset, type_graph, filename, mode_test):
    chemin_landmark_expr = os.path.join(chemin_depot_global, 'landmark', nom_dataset)
    # Each dataset gets its own results subtree so runs never overwrite each other
    mode_dir = 'test' if mode_test else 'train'
    base_res  = os.path.join(chemin_depot_global, 'resultats', mode_dir,
                              nom_dataset, type_graph)
    all_results_path  = os.path.join(base_res, 'results.txt')
    best_results_path = os.path.join(base_res, 'sort_results.txt')
    out_path          = os.path.join(base_res, filename)
    return all_results_path, best_results_path, chemin_landmark_expr, out_path



def main():
    args = parser.parse_args()

    # Global seed control (fixes ±11pt run variance from non-determinism)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f'[seed] Global seed set to {seed}  (cudnn.deterministic=True)')

    if args.graph_type not in ['g_complet', 'g_yeux_bouche', 'g_bouche', 'g_topologique']:
        raise NameError('graph_type invalide.')

    _tgconv_mode = getattr(args, 'tgconv_mode', 'learnable_A')
    _temporal_edges_mode = getattr(args, 'temporal_edges', 'none')
    if _tgconv_mode == 'temporal_edges' and _temporal_edges_mode == 'none':
        raise ValueError(
            "--tgconv_mode='temporal_edges' requires temporal edges to be present. "
            "Pass --temporal_edges construct_only (recommended: SGConv stays spatial-only "
            "and TGConvOnEdges gets the temporal edges), or combined / separated_parallel / "
            "separated_sequential."
        )

    print('Type du graphe = {}'.format(args.graph_type))
    print('Dataset        = {}'.format(args.nom_dataset))

    # Dataset adapter
    adapter = get_adapter(args.nom_dataset)(args)

    # Patches
    use_patches = args.chemin_patches is not None
    chemin_patches_dataset = None
    if use_patches:
        chemin_patches_dataset = os.path.join(args.chemin_patches, args.nom_dataset)
        # When the encoder runs on-the-fly (partial/full unfreeze), raw frames are
        # loaded at __getitem__ time — no pre-extracted patch folder is needed.
        on_the_fly_mode = args.unfreeze_resnet in ('partial', 'full', 'frozen')
        if not os.path.exists(chemin_patches_dataset) and not on_the_fly_mode:
            print(f'[warn] Patches folder not found: {chemin_patches_dataset}')
            print('       Running in coordinate-only mode.')
            use_patches = False

    if use_patches:
        print('Patch mode : ON  (coordinates + ResNet-34 FER+ appearance features)')
    else:
        print('Patch mode : OFF (coordinates only)')

    # Legacy --global_appearance_mode → six-flag translation
    # Four legacy values have a direct six-flag equivalent and are silently
    # translated. The other three ('roi_align', 'combined',
    # 'local_plus_multiscale') have none — they keep running through the
    # model's original single-mode code path, untouched by this task.
    def _all_new_flags_default(a):
        return (a.use_fragnet_landmark == 'off' and a.use_broadcast == 'off' and
                a.use_fragnet_grid == 'off' and a.use_local_encoder == 'off' and
                a.use_multiscale_roi == 'off' and a.fragnet_grid_backbone == 'resnet18')

    if args.global_appearance_mode != 'none_all' and _all_new_flags_default(args):
        _legacy_translatable = {
            'fragnet':              'use_fragnet_landmark',
            'broadcast':             'use_broadcast',
            'none':                  'use_local_encoder',
            'roi_align_multiscale':  'use_multiscale_roi',
        }
        if args.global_appearance_mode in _legacy_translatable:
            setattr(args, _legacy_translatable[args.global_appearance_mode], 'on')
            print(f"[Legacy] Translated --global_appearance_mode="
                  f"{args.global_appearance_mode!r} to new flag scheme.")
            args.global_appearance_mode = 'none_all'
        # else: 'roi_align' / 'combined' / 'local_plus_multiscale' — left as-is,
        # handled by the model's legacy global_appearance_mode code path below.
    elif args.global_appearance_mode != 'none_all' and not _all_new_flags_default(args):
        raise ValueError(
            "Cannot use both --global_appearance_mode and the new per-component "
            "--use_* flags. Use one or the other."
        )

    # True whenever the model should be built via the new six-flag scheme
    # (either flags were passed explicitly, or a legacy mode got translated
    # into them above). False only for the three untranslatable legacy modes.
    use_new_flag_scheme = (args.global_appearance_mode == 'none_all')

    # Architectural component ablation flags
    use_coord_encoder = args.use_coord_encoder == 'on'
    use_gcblocks      = args.use_gcblocks == 'on'
    use_transformers  = args.use_transformers == 'on'

    active_components = []
    if use_new_flag_scheme:
        if use_coord_encoder:                    active_components.append('coord_encoder')
        if args.use_fragnet_landmark == 'on':     active_components.append('fragnet_landmark')
        if args.use_broadcast        == 'on':     active_components.append('broadcast')
        if args.use_fragnet_grid     == 'on':     active_components.append('fragnet_grid')
        if args.use_local_encoder    == 'on':     active_components.append('local_encoder')
        if args.use_multiscale_roi   == 'on':     active_components.append('multiscale_roi')
        if not active_components:
            raise ValueError("At least one component must be ON. All six are off.")
        print(f"Active components ({len(active_components)}/6): {', '.join(active_components)}")

    if not use_coord_encoder and not use_patches:
        raise ValueError(
            "Cannot disable the coord encoder (--use_coord_encoder off) when there "
            "is no appearance signal: --chemin_patches is not set (or its patches "
            "folder is missing) and no in-model encoder is configured. The model "
            "would have no input features at all."
        )

    if not use_coord_encoder and not use_gcblocks and not use_transformers:
        print('WARNING: All three ablated components (coord encoder, GCBlocks, transformers) are OFF.')
        print('         Model reduces to appearance encoder + linear projection + pool + classifier.')

    print('\nArchitectural components:')
    print('  Coord encoder (R, C, T)  : {}'.format('ON' if use_coord_encoder else 'OFF'))
    print('  GCBlocks (x{})            : {}'.format(args.num_layers, 'ON' if use_gcblocks else 'OFF'))
    print('  Transformers (x{})        : {}'.format(args.num_layers_trans, 'ON' if use_transformers else 'OFF'))
    if use_new_flag_scheme:
        print('  Appearance mode           : multi-component ({}/6: {})'.format(
            len(active_components), ', '.join(active_components)))
    else:
        print('  Appearance mode           : {} (legacy)'.format(args.global_appearance_mode))
    print('  Temporal edges            : {}'.format(getattr(args, 'temporal_edges', 'none')))
    if getattr(args, 'upscale_patches_to_224', 'off') == 'on':
        print('Local patch upsampling: ON  (49×49 → 224×224 before ResNet encoder)')
    else:
        print('Local patch upsampling: OFF (49×49 patches fed directly to ResNet — mismatch with FER+ pretraining)')
    print('  Mixed precision (AMP)     : {}'.format('ON' if getattr(args, 'amp', 'off') == 'on' else 'OFF'))
    print('  Encoder BN stats frozen   : {}'.format('ON' if getattr(args, 'freeze_encoder_bn', 'off') == 'on' else 'OFF'))

    filename       = time.strftime('%Y%m%d-%H%M%S')
    use_test_paths = args.mode_debug

    (all_results_path, best_results_path,
     chemin_landmark_expr, out_path) = chemin(
        args.chemin_depot_global, args.nom_dataset,
        args.graph_type, filename, use_test_paths)

    make_dirs(out_path)
    with open(os.path.join(out_path, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load data
    # Labels pkl is per-dataset (e.g. labels/CK+_labels.pkl, labels/Oulu_labels.pkl)
    chemin_labels = adapter.labels_pkl_path()

    # Number of classes from adapter (reliable, not inferred from data)
    nb_classes = adapter.num_classes

    if not args.lazy_loading:
        # EAGER PATH
        # Original pipeline: load everything into RAM, slice per fold.
        # Works for CK+/Oulu/eNTERFACE05/AFEW.

        (all_videos_landmark, all_videos_patches, nb_frames_par_video,
         liste_labels, loaded_samples) = extract_datas_video_labels(
                chemin_landmark_expr,
                chemin_labels,
                path_patches=chemin_patches_dataset if use_patches else None
            )
        liste_labels = np.array(liste_labels)

        # Filter to graph-specific joints
        (nb_joints, all_videos_landmark, all_videos_patches,
         connections_par_frames, kept_indices) = convertion_graphe(
            all_videos_landmark,
            args.graph_type,
            all_videos_patches=all_videos_patches if use_patches else None
        )

        print('nb_joints    = {}'.format(nb_joints))
        print('kept_indices : {} landmarks  (first 5: {})'.format(
            len(kept_indices), kept_indices[:5]))

        # Flip map for batch augmentation
        flip_map = build_flip_map_for_subset(kept_indices)
        if flip_map is not None:
            print('Flip map     : computed ({} entries)'.format(len(flip_map)))
        else:
            print('Flip map     : None (some flip partners absent in this subset)')

        # Label normalisation — ensures labels are 0-indexed regardless of dataset
        if set(liste_labels) != set(range(nb_classes)):
            classes_sorted = sorted(set(liste_labels))
            correspondance = {c: i for i, c in enumerate(classes_sorted)}
            liste_labels   = np.array([correspondance[l] for l in liste_labels])

        # Temporal alignment
        nb_frames_par_video_preproc, processed_video_landmarks, processed_video_patches = \
            traitements_video_par_video(
                all_videos_landmark,
                nb_frames_par_video,
                args.mode_calcul_nb_frames,
                args.nb_frames,
                liste_videos_patches=all_videos_patches if use_patches else None
            )

        # Free pre-temporal-alignment dicts (saves ~10 GB on large datasets)
        del all_videos_landmark
        if all_videos_patches is not None:
            del all_videos_patches
        gc.collect()

        # Convert to tensors
        all_videos_tensor = convertir_liste_videos_en_tenseur(processed_video_landmarks)
        if use_patches and processed_video_patches is not None:
            all_patches_tensor = convertir_liste_patches_en_tenseur(processed_video_patches)
        else:
            all_patches_tensor = None

        del processed_video_landmarks
        if processed_video_patches is not None:
            del processed_video_patches
        gc.collect()

        label_remap = None  # already applied above

    else:
        # LAZY PATH
        # Streaming pipeline: keep only sample manifests in RAM, load
        # one (landmark, patches) pair per __getitem__ call. Required
        # for CAER and the AFEW+CAER joint setup.

        print('Lazy loading : ON  (streaming PyG Dataset, '
              '{} workers)'.format(args.num_workers))

        # Pull the (subject, video, label) catalog from the adapter.
        all_samples = adapter.get_all_samples()

        # Drop samples with missing/broken landmark pkls.
        loaded_samples, n_dropped = scan_samples(all_samples, adapter,
                                                  min_valid_frames=2)
        if n_dropped:
            print(f'  [info] {n_dropped} samples skipped during scan, '
                  f'{len(loaded_samples)} kept.')

        # Build label_remap if labels are sparse (matches eager path's
        # `correspondance` step). We do NOT mutate loaded_samples' labels —
        # the dataset applies the remap at __getitem__ time.
        labels_set = set(s[2] for s in loaded_samples)
        if labels_set != set(range(nb_classes)):
            classes_sorted = sorted(labels_set)
            label_remap    = {c: i for i, c in enumerate(classes_sorted)}
        else:
            label_remap = None

        liste_labels = np.array([s[2] for s in loaded_samples])
        if label_remap is not None:
            liste_labels = np.array([label_remap[l] for l in liste_labels])

        # Probe one LazyFERDataset to compute geometry — kept_indices,
        # connections_par_frames, nb_joints are deterministic given graph_type.
        probe_ds = LazyFERDataset(
            samples       = loaded_samples[:1],   # cheap probe
            adapter       = adapter,
            graph_type    = args.graph_type,
            use_patches   = use_patches,
            nb_frames     = args.nb_frames,
            label_remap   = label_remap,
        )
        nb_joints              = probe_ds.nb_joints
        kept_indices           = probe_ds.kept_indices
        connections_par_frames = probe_ds.connections

        print('nb_joints    = {}'.format(nb_joints))
        print('kept_indices : {} landmarks  (first 5: {})'.format(
            len(kept_indices), kept_indices[:5]))

        flip_map = build_flip_map_for_subset(kept_indices)
        if flip_map is not None:
            print('Flip map     : computed ({} entries)'.format(len(flip_map)))
        else:
            print('Flip map     : None (some flip partners absent in this subset)')

        # The lazy path doesn't build a master tensor.
        nb_frames_par_video_preproc = args.nb_frames
        all_videos_tensor  = None
        all_patches_tensor = None

    # Build folds via adapter (identical for both paths)
    sample_to_idx  = {(s, v): i for i, (s, v, _) in enumerate(loaded_samples)}

    adapter_folds  = adapter.get_folds()
    folds = []
    for fold in adapter_folds:
        train_idx = np.array(
            [sample_to_idx[(s, v)] for s, v, _ in fold['train']
             if (s, v) in sample_to_idx],
            dtype=np.int64,
        )
        test_idx = np.array(
            [sample_to_idx[(s, v)] for s, v, _ in fold['test']
             if (s, v) in sample_to_idx],
            dtype=np.int64,
        )
        test2_idx = np.array(
            [sample_to_idx[(s, v)] for s, v, _ in fold.get('test_secondary', [])
             if (s, v) in sample_to_idx],
            dtype=np.int64,
        )
        folds.append((train_idx, test_idx, test2_idx))

    # Subject IDs derived from loaded_samples (used for overlap assertion)
    subject_ids = np.array([s for s, v, _ in loaded_samples])

    print('\nNombre de sujets uniques : {}  →  {}-fold identity-independent CV'.format(
        len(set(subject_ids)), args.nb_cross_val))
    print('Distribution labels : {}'.format(np.unique(liste_labels, return_counts=True)))

    all_acc, all_prec, all_rec, all_cm = [], [], [], []
    all_acc_final, all_cm_final         = [], []
    all_uar_final                       = []

    device     = torch.device(args.device)
    seq_length = nb_frames_par_video_preproc
    nb_train_last, nb_val_last = 0, 0

    for fold_idx, (fold_train_idx, fold_test_idx, fold_test2_idx) in enumerate(folds):
        # Fresh deterministic generator per fold — isolates each fold's randomness
        # so fold k's training does not depend on how many random draws fold k-1 made.
        # Both eager (shuffle) and lazy (WRS) paths use this same fold_gen seed.
        fold_gen = torch.Generator()
        fold_gen.manual_seed(args.seed + fold_idx * 1000)

        print('\n#-------------------------- Fold {} / {} --------------------------#\n'.format(
            fold_idx + 1, args.nb_cross_val))

        train_subjects = set(subject_ids[fold_train_idx])
        test_subjects  = set(subject_ids[fold_test_idx])
        assert len(train_subjects & test_subjects) == 0, 'FUITE DE DONNÉES !'

        print('Sujets test  ({:2d}) : {}'.format(len(test_subjects), sorted(test_subjects)))
        print('Sujets train ({:2d}) : {} sujets'.format(len(train_subjects), len(train_subjects)))

        test2_loader    = None   # only set in lazy path when fold has test_secondary
        val2_loader_fold = None  # only set in lazy path for joint AFEW+CAER folds

        if not args.lazy_loading:
            # EAGER PATH (fold body)
            # Split
            foldX_train_raw = all_videos_tensor[:, :, fold_train_idx]
            foldY_train_raw = liste_labels[fold_train_idx]
            foldX_test      = all_videos_tensor[:, :, fold_test_idx]
            foldY_test      = torch.tensor(liste_labels[fold_test_idx], dtype=torch.int64)

            if use_patches and all_patches_tensor is not None:
                foldP_train_raw = all_patches_tensor[:, :, fold_train_idx]
                foldP_test      = all_patches_tensor[:, :, fold_test_idx]
            else:
                foldP_train_raw = None
                foldP_test      = None

            print('\nDistribution labels:')
            print('En train : {}'.format(np.unique(foldY_train_raw, return_counts=True)))
            print('En test  : {}'.format(np.unique(liste_labels[fold_test_idx], return_counts=True)))

            # Augmentation (train only)
            if use_patches and foldP_train_raw is not None:
                foldX_train_aug, foldY_train_aug, foldP_train_aug = \
                    data_augmentation_equilibration_classe(
                        foldX_train_raw, foldY_train_raw,
                        nb_joints, nb_frames_par_video_preproc,
                        sorted(list(set(foldY_train_raw))),
                        args.facteur, P=foldP_train_raw
                    )
            else:
                foldX_train_aug, foldY_train_aug = \
                    data_augmentation_equilibration_classe(
                        foldX_train_raw, foldY_train_raw,
                        nb_joints, nb_frames_par_video_preproc,
                        sorted(list(set(foldY_train_raw))),
                        args.facteur
                    )
                foldP_train_aug = None

            trainX = foldX_train_aug
            trainY = foldY_train_aug if isinstance(foldY_train_aug, torch.Tensor) \
                     else torch.tensor(foldY_train_aug, dtype=torch.int64)
            testX  = foldX_test
            testY  = foldY_test
            trainP = foldP_train_aug
            testP  = foldP_test

            nb_train      = len(trainY)
            nb_val        = len(testY)
            nb_train_last = nb_train
            nb_val_last   = nb_val

            print('\ntrainX: {}, trainY: {} | testX: {}, testY: {}'.format(
                trainX.shape, trainY.shape, testX.shape, testY.shape))

            # Build PyG graphs
            _temporal_edges_mode = getattr(args, 'temporal_edges', 'none')
            data_list_train = [
                create_graph(
                    trainX[:, :, idx], trainY[idx],
                    nb_frames_par_video_preproc, nb_joints,
                    connections_par_frames,
                    patch_tensor=trainP[:, :, idx] if trainP is not None else None,
                    temporal_edges_mode=_temporal_edges_mode,
                )
                for idx in tqdm(range(len(trainY)), total=len(trainY))
            ]
            data_list_test = [
                create_graph(
                    testX[:, :, idx], testY[idx],
                    nb_frames_par_video_preproc, nb_joints,
                    connections_par_frames,
                    patch_tensor=testP[:, :, idx] if testP is not None else None,
                    temporal_edges_mode=_temporal_edges_mode,
                )
                for idx in tqdm(range(len(testY)), total=len(testY))
            ]

            train_loader = GeoDataLoader(data_list_train, batch_size=args.batch_size,
                                         shuffle=True, generator=fold_gen)
            test_loader  = GeoDataLoader(data_list_test,  batch_size=args.batch_size,
                                         shuffle=False)
            # Eager path does not currently build a val loader. Add one here
            # later if you want early stopping on small datasets too.
            val_loader = None

        else:
            # LAZY PATH (fold body)
            # No bulk oversampling — class balancing via WeightedRandomSampler.
            train_samples = [loaded_samples[i] for i in fold_train_idx]
            test_samples  = [loaded_samples[i] for i in fold_test_idx]
            # Apply label_remap to labels we will use for class weighting.
            train_labels_int = np.array(
                [(label_remap[s[2]] if label_remap is not None else s[2])
                 for s in train_samples], dtype=np.int64)
            test_labels_int  = np.array(
                [(label_remap[s[2]] if label_remap is not None else s[2])
                 for s in test_samples], dtype=np.int64)

            # Stratified val split (optional)
            # Carve --val_split fraction off the training set for early-stopping
            # signal. Stratify by class so minority classes are represented.
            # Split is done on SAMPLE indices, not subjects — the lazy path's
            # subject overlap assertion above already guarantees train/test
            # subject disjointness, and the val slice is drawn from train so
            # subject leakage to test cannot occur.
            #
            # NOTE: for in-the-wild datasets (AFEW/CAER) each "subject" is a
            # single clip, so a per-sample stratified split is the same as a
            # per-subject one. For CK+/Oulu/eNTERFACE05, val is taken from the
            # same subjects as train — which is fine because the test fold is
            # already identity-disjoint.
            #
            # DOMAIN-MATCHED VAL (AFEW+CAER joint fold only):
            #   When the test fold is pure AFEW (Fold 1: test=AFEW_val_*) but
            #   the train fold mixes AFEW+CAER, drawing val uniformly from
            #   the joint train set produces a val pool that is ~88% CAER /
            #   12% AFEW. The model that is best at predicting that mixed val
            #   is NOT the model that is best at predicting pure-AFEW test —
            #   they are different distributions, and the val→test gap
            #   becomes huge (~20 points was observed empirically).
            #   So we restrict the val POOL to samples from the same domain
            #   as the test fold, and stratify within that pool. The
            #   training set is the FULL train fold (val samples removed) —
            #   we only restrict where val is drawn from, not what's trained.
            val_samples       = []
            val_labels_int    = np.array([], dtype=np.int64)
            if args.val_split > 0.0:
                # Guard against degenerate cases (very small folds, single-class)
                if len(train_samples) < 20 or len(set(train_labels_int)) < 2:
                    print('[val] Skipping val split (too few samples or single class)')
                else:
                    # Detect val pool
                    # Sniff the test subject prefix to determine the eval domain.
                    # For non-joint datasets (no AFEW_/CAER_ prefixes) this
                    # condition is false and val is drawn from full train.
                    test_prefix = None
                    if len(test_samples) > 0:
                        first_test_subj = test_samples[0][0]
                        if first_test_subj.startswith('AFEW_'):
                            test_prefix = 'AFEW_'
                        elif first_test_subj.startswith('CAER_'):
                            test_prefix = 'CAER_'

                    # Filter to candidate val indices that match the test domain.
                    if test_prefix is not None:
                        val_pool_idx = np.array(
                            [i for i, s in enumerate(train_samples)
                             if s[0].startswith(test_prefix)],
                            dtype=np.int64,
                        )
                    else:
                        val_pool_idx = np.arange(len(train_samples), dtype=np.int64)

                    # If the candidate pool is too small / single-class, fall
                    # back to full train.
                    pool_labels = train_labels_int[val_pool_idx]
                    if (len(val_pool_idx) < 20 or
                            len(set(pool_labels.tolist())) < 2):
                        if test_prefix is not None:
                            print(f'[val] Domain-matched pool too small for '
                                  f'prefix {test_prefix} '
                                  f'({len(val_pool_idx)} samples); '
                                  f'falling back to full train.')
                        val_pool_idx = np.arange(len(train_samples), dtype=np.int64)
                        pool_labels  = train_labels_int

                    if test_prefix is not None:
                        print(f'[val] Val pool restricted to domain '
                              f'{test_prefix!r}: '
                              f'{len(val_pool_idx)} / {len(train_samples)} '
                              f'candidate train samples')

                    # Stratified split WITHIN the val pool.
                    try:
                        sss = StratifiedShuffleSplit(
                            n_splits=1,
                            test_size=args.val_split,
                            random_state=args.seed,
                        )
                        # Split returns indices into val_pool_idx.
                        _, va_local = next(sss.split(
                            np.zeros(len(val_pool_idx)), pool_labels))
                        va_idx = val_pool_idx[va_local]   # absolute train idx

                        # Train keeps everything NOT in va_idx (the val pool's
                        # non-selected samples go back into train, so this
                        # only removes the actual val samples — same train
                        # count as the original uniform split: ~10% smaller).
                        va_mask = np.zeros(len(train_samples), dtype=bool)
                        va_mask[va_idx] = True

                        val_samples      = [train_samples[i] for i in va_idx]
                        val_labels_int   = train_labels_int[va_idx]
                        train_samples    = [s for i, s in enumerate(train_samples)
                                            if not va_mask[i]]
                        train_labels_int = train_labels_int[~va_mask]
                    except ValueError as e:
                        # Stratified split fails if any class has fewer than
                        # 2 samples — fall back to no val split.
                        print(f'[val] Stratified split failed ({e}); '
                              f'continuing without val.')
                        val_samples    = []
                        val_labels_int = np.array([], dtype=np.int64)

            # Secondary val (CAER) — only for joint AFEW+CAER fold
            # Carved from CAER_train_* samples remaining in train_samples after
            # the AFEW val split above. Actual CAER_val_* files are excluded
            # because their landmarks may not have been extracted yet.
            val2_samples    = []
            val2_labels_int = np.array([], dtype=np.int64)
            if len(fold_test2_idx) > 0 and args.val_split > 0.0:
                caer_pool_idx = np.array(
                    [i for i, s in enumerate(train_samples)
                     if s[0].startswith('CAER_train_')],
                    dtype=np.int64,
                )
                caer_pool_labels = train_labels_int[caer_pool_idx]
                if len(caer_pool_idx) >= 20 and len(set(caer_pool_labels.tolist())) >= 2:
                    try:
                        sss2 = StratifiedShuffleSplit(
                            n_splits=1, test_size=args.val_split,
                            random_state=args.seed + 1,
                        )
                        _, va2_local = next(sss2.split(
                            np.zeros(len(caer_pool_idx)), caer_pool_labels))
                        va2_idx  = caer_pool_idx[va2_local]
                        va2_mask = np.zeros(len(train_samples), dtype=bool)
                        va2_mask[va2_idx] = True
                        val2_samples    = [train_samples[i] for i in va2_idx]
                        val2_labels_int = train_labels_int[va2_idx]
                        train_samples    = [s for i, s in enumerate(train_samples)
                                            if not va2_mask[i]]
                        train_labels_int = train_labels_int[~va2_mask]
                        print(f'[val2] CAER val carved: {len(val2_samples)} samples '
                              f'from CAER_train pool ({len(caer_pool_idx)} candidates)')
                    except ValueError as e:
                        print(f'[val2] CAER val split failed ({e}); skipping.')

            print('\nDistribution labels:')
            print('En train : {}'.format(np.unique(train_labels_int, return_counts=True)))
            if len(val_samples) > 0:
                print('En val   (AFEW): {}'.format(np.unique(val_labels_int, return_counts=True)))
            if len(val2_samples) > 0:
                print('En val2  (CAER): {}'.format(np.unique(val2_labels_int, return_counts=True)))
            print('En test  : {}'.format(np.unique(test_labels_int,  return_counts=True)))

            # Patch loading mode
            # 'cached'     : load 512-d feature pickles (existing default)
            # 'on_the_fly' : load raw frames + extract 49×49 patches at __getitem__
            # Auto-select based on encoder_mode unless --use_cached_patches forces.
            if args.unfreeze_resnet in ('partial', 'full', 'frozen'):
                patch_mode = 'on_the_fly'
            else:
                patch_mode = 'cached'
            if args.use_cached_patches:
                patch_mode = 'cached'
                if args.unfreeze_resnet in ('partial', 'full'):
                    print('[warn] --use_cached_patches with --unfreeze_resnet=partial/full '
                          'is meaningless (encoder cannot see raw patches when '
                          'cached features are passed). Disabling encoder.')
            print(f'Patch loading: {patch_mode}')

            # Task 16: gate on_the_fly_landmarks to on-the-fly patch mode
            if getattr(args, 'on_the_fly_landmarks', False) and patch_mode != 'on_the_fly':
                raise ValueError(
                    '--on_the_fly_landmarks requires on-the-fly patch loading '
                    '(use --unfreeze_resnet=frozen, partial, or full to enable it). '
                    'Re-detecting landmarks makes no sense with cached features: '
                    'the cached features were extracted from different landmarks, '
                    'so geometry and appearance would still disagree.'
                )
            if getattr(args, 'on_the_fly_landmarks', False):
                print('Landmark mode: on_the_fly  '
                      '(2D-FAN re-detection per frame — geometry/appearance consistent)')

            # Datasets
            _otf_lm = getattr(args, 'on_the_fly_landmarks', False)

            # LazyFERDataset only needs to know WHICH raw materials to load
            # (per-landmark 49x49 patches and/or full 224x224 frames) — not
            # which model-level component consumes them. In multi-component
            # mode that's derived from the six flags and mapped onto the
            # dataset's existing three material categories, so lazy_dataset.py
            # itself needs no changes:
            #   patches AND full_frames -> 'local_plus_multiscale' (both, one FAN pass)
            #   full_frames only        -> 'broadcast'
            #   patches only            -> 'none'
            if use_new_flag_scheme:
                _needs_patches     = (args.use_fragnet_landmark == 'on' or
                                       args.use_local_encoder == 'on')
                _needs_full_frames = (args.use_fragnet_landmark == 'on' or
                                       args.use_broadcast == 'on' or
                                       args.use_fragnet_grid == 'on' or
                                       args.use_multiscale_roi == 'on')
                if _needs_patches and _needs_full_frames:
                    _gam = 'local_plus_multiscale'
                elif _needs_full_frames:
                    _gam = 'broadcast'
                else:
                    _gam = 'none'
            else:
                _gam = args.global_appearance_mode
            _tem = getattr(args, 'temporal_edges', 'none')

            train_ds = LazyFERDataset(
                samples                = train_samples,
                adapter                = adapter,
                graph_type             = args.graph_type,
                use_patches            = use_patches,
                nb_frames              = args.nb_frames,
                label_remap            = label_remap,
                sampling_strategy      = args.sampling_strategy,
                patch_mode             = patch_mode,
                already_cropped        = getattr(args, 'already_cropped', False),
                base_dir               = args.chemin_depot_global,
                training               = True,
                flip_map               = flip_map,
                on_the_fly_landmarks   = _otf_lm,
                global_appearance_mode = _gam,
                temporal_edges_mode    = _tem,
            )
            test_ds = LazyFERDataset(
                samples                = test_samples,
                adapter                = adapter,
                graph_type             = args.graph_type,
                use_patches            = use_patches,
                nb_frames              = args.nb_frames,
                label_remap            = label_remap,
                sampling_strategy      = args.sampling_strategy,
                patch_mode             = patch_mode,
                already_cropped        = getattr(args, 'already_cropped', False),
                base_dir               = args.chemin_depot_global,
                training               = False,
                flip_map               = flip_map,
                on_the_fly_landmarks   = _otf_lm,
                global_appearance_mode = _gam,
                temporal_edges_mode    = _tem,
            )
            val_ds = None
            if len(val_samples) > 0:
                val_ds = LazyFERDataset(
                    samples                = val_samples,
                    adapter                = adapter,
                    graph_type             = args.graph_type,
                    use_patches            = use_patches,
                    nb_frames              = args.nb_frames,
                    label_remap            = label_remap,
                    sampling_strategy      = args.sampling_strategy,
                    patch_mode             = patch_mode,
                    already_cropped        = getattr(args, 'already_cropped', False),
                    base_dir               = args.chemin_depot_global,
                    training               = False,
                    flip_map               = flip_map,
                    on_the_fly_landmarks   = _otf_lm,
                    global_appearance_mode = _gam,
                    temporal_edges_mode    = _tem,
                )

            # WeightedRandomSampler for class balancing (replaces bulk oversampler).
            # num_samples matches the eager path's augmented dataset size:
            # majority_class_count × nb_classes (same target as
            # data_augmentation_equilibration_classe with the same fold).
            class_counts = np.bincount(train_labels_int, minlength=nb_classes)
            class_weights_rs = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights_rs[train_labels_int]
            num_samples_rs = int(class_counts.max()) * nb_classes
            sampler = torch.utils.data.WeightedRandomSampler(
                weights      = torch.tensor(sample_weights, dtype=torch.double),
                num_samples  = num_samples_rs,
                replacement  = True,
                generator    = fold_gen,
            )

            _pin = device.type == 'cuda' and args.num_workers > 0

            # Task 23: GPU FAN in DataLoader workers requires 'spawn' start method.
            # CUDA contexts are not fork-safe; spawn avoids inheriting the parent's
            # GPU context.  Only needed when on_the_fly_landmarks is active — that
            # is the only code path where FAN runs inside worker processes.
            _spawn_ctx = None
            if _otf_lm and args.num_workers > 0:
                _spawn_ctx = 'spawn'
                print(f'[Task 23] DataLoader multiprocessing_context=spawn '
                      f'(GPU FAN in {args.num_workers} worker(s))')
                if device.type == 'cuda':
                    try:
                        import torch as _t
                        free_gb = _t.cuda.mem_get_info()[0] / 1024 ** 3
                        fan_gb  = args.num_workers * 0.05   # ~50 MB per FAN instance
                        if fan_gb > free_gb * 0.2:
                            print(f'[warn] {args.num_workers} FAN worker(s) × ~50 MB = '
                                  f'{fan_gb:.1f} GB; free GPU mem ≈ {free_gb:.1f} GB. '
                                  f'Consider reducing --num_workers.')
                    except Exception:
                        pass

            _dl_common = dict(
                num_workers           = args.num_workers,
                persistent_workers    = (args.num_workers > 0),
                worker_init_fn        = _worker_init_fn if args.num_workers > 0 else None,
                pin_memory            = _pin,
                **(dict(multiprocessing_context=_spawn_ctx) if _spawn_ctx else {}),
            )

            train_loader = GeoDataLoader(
                train_ds,
                batch_size = args.batch_size,
                sampler    = sampler,
                **_dl_common,
            )
            test_loader = GeoDataLoader(
                test_ds,
                batch_size = args.batch_size,
                shuffle    = False,
                **_dl_common,
            )
            test2_loader = None
            if len(fold_test2_idx) > 0:
                test2_samples = [loaded_samples[i] for i in fold_test2_idx]
                test2_ds = LazyFERDataset(
                    samples                = test2_samples,
                    adapter                = adapter,
                    graph_type             = args.graph_type,
                    use_patches            = use_patches,
                    nb_frames              = args.nb_frames,
                    label_remap            = label_remap,
                    sampling_strategy      = args.sampling_strategy,
                    patch_mode             = patch_mode,
                    already_cropped        = getattr(args, 'already_cropped', False),
                    base_dir               = args.chemin_depot_global,
                    training               = False,
                    flip_map               = flip_map,
                    on_the_fly_landmarks   = _otf_lm,
                    global_appearance_mode = _gam,
                    temporal_edges_mode    = _tem,
                )
                test2_loader = GeoDataLoader(
                    test2_ds,
                    batch_size = args.batch_size,
                    shuffle    = False,
                    **_dl_common,
                )
            val_loader = None
            if val_ds is not None:
                val_loader = GeoDataLoader(
                    val_ds,
                    batch_size = args.batch_size,
                    shuffle    = False,
                    **_dl_common,
                )

            val2_loader_fold = None
            if len(val2_samples) > 0:
                val2_ds = LazyFERDataset(
                    samples                = val2_samples,
                    adapter                = adapter,
                    graph_type             = args.graph_type,
                    use_patches            = use_patches,
                    nb_frames              = args.nb_frames,
                    label_remap            = label_remap,
                    sampling_strategy      = args.sampling_strategy,
                    patch_mode             = patch_mode,
                    already_cropped        = getattr(args, 'already_cropped', False),
                    base_dir               = args.chemin_depot_global,
                    training               = False,
                    flip_map               = flip_map,
                    on_the_fly_landmarks   = _otf_lm,
                    global_appearance_mode = _gam,
                    temporal_edges_mode    = _tem,
                )
                val2_loader_fold = GeoDataLoader(
                    val2_ds,
                    batch_size = args.batch_size,
                    shuffle    = False,
                    **_dl_common,
                )

            # Variables expected by the rest of the fold body
            trainY = torch.tensor(train_labels_int, dtype=torch.int64)
            nb_train_last = len(train_samples)
            nb_val_last   = len(test_samples)
            if val_loader is not None:
                is_joint = args.nom_dataset == 'AFEW+CAER'
                val_label  = 'AFEW-val' if is_joint else 'val'
                val2_label = 'CAER-val' if is_joint else 'val2'
                print('Train: {} | {}: {} | {}: {} | Test: {} samples'.format(
                    nb_train_last,
                    val_label,  len(val_samples),
                    val2_label, len(val2_samples) if val2_loader_fold else 0,
                    nb_val_last))
            else:
                print('Train: {} samples | Test: {} samples'.format(
                    nb_train_last, nb_val_last))


        dataloaders = {'train': train_loader}
        if val_loader is not None:
            dataloaders['val'] = val_loader
        out_dim     = nb_classes
        print('Le dataset possède {} classes'.format(out_dim))

        # Model
        # Decide encoder mode:
        #   - --unfreeze_resnet partial → backbone layer4 trainable (use --patch_backbone)
        #   - --unfreeze_resnet full    → all backbone params trainable
        #   - --unfreeze_resnet frozen  → backbone in-model but eval()+no_grad (sanity)
        #   - --unfreeze_resnet none    → no in-model encoder; cached features expected
        # When the encoder is active, the lazy dataset MUST be set to
        # patch_mode='on_the_fly'. We enforce this below in the dataset build.
        # Multi-component kwargs are only meaningful (non-None) when
        # use_new_flag_scheme; otherwise leave them None so the model falls
        # through to its legacy global_appearance_mode code path unchanged.
        _component_kwargs = dict(
            use_fragnet_landmark  = None,
            use_broadcast         = None,
            use_fragnet_grid      = None,
            use_local_encoder     = None,
            use_multiscale_roi    = None,
            fragnet_grid_backbone = args.fragnet_grid_backbone,
        )
        if use_new_flag_scheme:
            _component_kwargs.update(
                use_fragnet_landmark = (args.use_fragnet_landmark == 'on'),
                use_broadcast        = (args.use_broadcast        == 'on'),
                use_fragnet_grid     = (args.use_fragnet_grid     == 'on'),
                use_local_encoder    = (args.use_local_encoder    == 'on'),
                use_multiscale_roi   = (args.use_multiscale_roi   == 'on'),
            )

        model = GCNTransformerPaper(
            nb_joint               = nb_joints,
            seq_length             = seq_length,
            n_heads                = args.n_heads,
            dim_feedforward        = args.dim_feedforward,
            num_gc_blocks          = args.num_layers,
            num_trans_layers       = args.num_layers_trans,
            out_dim                = out_dim,
            kept_indices           = kept_indices,
            dropout                = args.dropout_transformers,
            use_patches            = use_patches,
            encoder_mode           = args.unfreeze_resnet,
            encoder_ckpt           = args.encoder_ckpt,
            encoder_backbone       = args.patch_backbone,
            encoder_imagenet_norm  = args.encoder_imagenet_norm,
            upscale_patches_to_224 = (getattr(args, 'upscale_patches_to_224', 'off') == 'on'),
            freeze_encoder_bn      = (getattr(args, 'freeze_encoder_bn', 'off') == 'on'),
            global_appearance_mode = args.global_appearance_mode if not use_new_flag_scheme else 'none',
            temporal_edges_mode    = getattr(args, 'temporal_edges', 'none'),
            tgconv_mode            = getattr(args, 'tgconv_mode', 'learnable_A'),
            use_coord_encoder      = use_coord_encoder,
            use_gcblocks           = use_gcblocks,
            use_transformers       = use_transformers,
            **_component_kwargs,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if use_new_flag_scheme:
            print('\nActive components ({}/6):'.format(len(active_components)))
            print('  Coordinate encoder    : {:<3} ({}-d per landmark)'.format(
                'ON' if use_coord_encoder else 'OFF', 768))
            print('  Fragnet-landmark      : {:<3} (512-d per landmark)'.format(
                'ON' if args.use_fragnet_landmark == 'on' else 'OFF'))
            print('  Broadcast global      : {:<3} (512-d, broadcast)'.format(
                'ON' if args.use_broadcast == 'on' else 'OFF'))
            print('  Fragnet-grid          : {:<3} (512-d, {} backbone{})'.format(
                'ON' if args.use_fragnet_grid == 'on' else 'OFF',
                args.fragnet_grid_backbone,
                ', shared' if (args.use_fragnet_grid == 'on'
                               and args.fragnet_grid_backbone == 'resnet18') else ''))
            print('  Local encoder         : {:<3} (512-d per landmark)'.format(
                'ON' if args.use_local_encoder == 'on' else 'OFF'))
            print('  Multi-scale ROI-Align : {:<3} (896-d per landmark)'.format(
                'ON' if args.use_multiscale_roi == 'on' else 'OFF'))
        print('FUSION_DIM = {} ({} geometry + {} appearance)'.format(
            model.FUSION_DIM,
            3 * 256 if use_coord_encoder else 0,
            model.APPEARANCE_DIM))
        print('Total Trainable Parameters : {}\n'.format(total_params))
        if args.unfreeze_resnet != 'none' and use_patches and model.patch_encoder is not None:
            enc_params = sum(p.numel() for p in model.patch_encoder.trainable_parameters())
            head_params = total_params - enc_params
            print('  Encoder trainable : {:,}'.format(enc_params))
            print('  Head trainable    : {:,}'.format(head_params))
        elif args.unfreeze_resnet != 'none' and use_patches:
            # Multi-component mode where no active component needs the shared
            # FER+ ResNet-18 (e.g. fragnet_grid_backbone=resnet50 alone) —
            # patch_encoder is legitimately None; all trainable params are
            # "head" params (see encoder_head_param_groups()).
            print('  Encoder trainable : 0 (no component uses the shared ResNet-18 encoder)')
            print('  Head trainable    : {:,}'.format(total_params))
        print('Les labels sont : {}'.format(torch.unique(trainY)))

        # Center Loss
        feat_dim    = 256   # output dim of model.pre_logit
        center_loss = CenterLoss(nb_classes, feat_dim).to(device)

        # Loss + Optimizers
        if args.class_weight or args.use_class_weights:
            # Compute balanced class weights over the classes ACTUALLY PRESENT
            # in this fold's training labels. Classes absent from trainY are
            # given weight 1.0 (they get no gradient anyway). Using the full
            # `classes=np.arange(nb_classes)` keeps the weight vector aligned
            # with CrossEntropyLoss's `weight` param shape.
            tr_y      = trainY.numpy()
            present   = np.unique(tr_y)
            try:
                w_present = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=present,
                    y=tr_y,
                )
            except ValueError:
                # Falls back to all-ones if sklearn can't compute (very small folds)
                w_present = np.ones_like(present, dtype=np.float32)

            weights_arr = np.ones(nb_classes, dtype=np.float32)
            for c, w in zip(present, w_present):
                weights_arr[c] = w
            mean_present = weights_arr[present].mean()
            if mean_present > 0:
                weights_arr = weights_arr / mean_present
            weights   = torch.tensor(weights_arr, dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=weights.to(device), reduction='mean')
            print('Class weights : {} (normalised, present={})\n'.format(
                np.round(weights.numpy(), 3), present.tolist()))
        else:
            print('Not using class weight !\n')
            criterion = nn.CrossEntropyLoss()

        # Optimizer (single or two-LR depending on encoder_mode)
        # When the encoder is unfrozen, the head and encoder receive different
        # learning rates: the head trains from scratch at args.lr; the encoder
        # is fine-tuned at args.encoder_lr (typically 10× smaller). Common
        # transfer-learning recipe.
        if args.unfreeze_resnet in ('partial', 'full') and use_patches:
            encoder_params, head_params = model.encoder_head_param_groups()
            optimizer_model = torch.optim.Adam(
                [
                    {'params': head_params,    'lr': args.lr},
                    {'params': encoder_params, 'lr': args.encoder_lr},
                ],
                weight_decay=args.weight_decay,
            )
            print('Optimizer    : two-LR Adam — head_lr={:.1e}, encoder_lr={:.1e}'.format(
                args.lr, args.encoder_lr))
        else:
            optimizer_model = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_center = torch.optim.Adam(
            center_loss.parameters(), lr=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_model, T_max=args.num_epochs, eta_min=1e-6)

        # Train
        trained_model, trained_model2, history = train_model_gnn_paper(
            model, criterion, center_loss, args.lambda_center,
            optimizer_model, optimizer_center,
            args.apply_scheduler, scheduler,
            dataloaders, device, args.num_epochs, args.print_every,
            T=seq_length, P=nb_joints, flip_map=flip_map,
            best_metric=args.best_metric,
            val2_loader=val2_loader_fold if args.lazy_loading else None,
            use_amp=(getattr(args, 'amp', 'off') == 'on'),
        )

        # Test (evaluated ONCE — paper protocol)
        # Save model BEFORE the test pass
        # We save here (not after test) so that a crash during the test pass
        # doesn't lose the best-val weights restored by the trainer. This is
        # the only state we cannot recover without re-running training.
        if args.saved_model:
            model_path = os.path.join(out_path, 'models')
            make_dirs(model_path)
            torch.save(trained_model.state_dict(),
                       os.path.join(model_path, 'fold{}_afew_best.pth'.format(fold_idx + 1)))
            if trained_model2 is not None:
                torch.save(trained_model2.state_dict(),
                           os.path.join(model_path, 'fold{}_caer_best.pth'.format(fold_idx + 1)))

        # Shut down persistent DataLoader workers from training so their
        # FAN-on-CUDA instances are released before test workers start up.
        # Without this, 8 training workers × ~394 MB stay alive and leave no
        # GPU memory for the test workers to initialize their own FAN models.
        del dataloaders
        if val_loader is not None:
            del val_loader
        if val2_loader_fold is not None:
            del val2_loader_fold
        torch.cuda.empty_cache()

        trained_model.eval()
        center_loss.eval()
        all_preds, all_labels_list = [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                bs    = batch.y.shape[0]

                # Dispatch patch_features vs patches by tensor dim
                # (same logic as the training/val forward passes).
                patch_feat  = None
                patches     = None
                full_frames = getattr(batch, 'full_frames', None)
                if hasattr(batch, 'patch_features') and batch.patch_features is not None:
                    if batch.patch_features.dim() == 2:
                        patch_feat = batch.patch_features
                    else:
                        patches    = batch.patch_features

                logits, _ = trained_model(
                    batch.x, batch.edge_index, batch.batch, bs,
                    patch_features=patch_feat,
                    patches=patches,
                    full_frames=full_frames,
                    edge_index_temporal=getattr(batch, 'edge_index_temporal', None),
                )
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels_list.append(batch.y.cpu())

        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels_list)

        fold_acc = accuracy_score(labels.numpy(), preds.numpy())
        fold_uar = balanced_accuracy_score(labels.numpy(), preds.numpy())
        all_acc_final.append(fold_acc)
        all_uar_final.append(fold_uar)

        print('\n================ TEST RESULTS (Fold {}/{}) ================'.format(
            fold_idx + 1, args.nb_cross_val))
        print('Sujets testés       : {}'.format(sorted(test_subjects)))
        print('Test WAR (Accuracy) : {:.2f}%'.format(fold_acc * 100))
        print('Test UAR (Balanced) : {:.2f}%'.format(fold_uar * 100))
        print('Precision (weighted): {:.2f}%'.format(
            precision_score(labels.numpy(), preds.numpy(), average='weighted') * 100))
        print('Recall    (weighted): {:.2f}%'.format(
            recall_score(labels.numpy(), preds.numpy(), average='weighted') * 100))
        print('\nPer-class breakdown:')
        print(classification_report(
            labels.numpy(), preds.numpy(),
            labels=list(range(nb_classes)),
            target_names=adapter.emotion_labels,
            zero_division=0
        ))
        print('============================================================\n')

        # Secondary test (CAER test — uses CAER-val-best checkpoint)
        if test2_loader is not None:
            caer_eval_model = trained_model2 if trained_model2 is not None else trained_model
            caer_eval_model.eval()
            all_preds2, all_labels2 = [], []
            with torch.no_grad():
                for batch in test2_loader:
                    batch = batch.to(device)
                    bs2 = batch.y.shape[0]
                    pf2, pa2 = None, None
                    ff2 = getattr(batch, 'full_frames', None)
                    if hasattr(batch, 'patch_features') and batch.patch_features is not None:
                        if batch.patch_features.dim() == 2:
                            pf2 = batch.patch_features
                        else:
                            pa2 = batch.patch_features
                    logits2, _ = caer_eval_model(
                        batch.x, batch.edge_index, batch.batch, bs2,
                        patch_features=pf2, patches=pa2,
                        full_frames=ff2,
                        edge_index_temporal=getattr(batch, 'edge_index_temporal', None),
                    )
                    all_preds2.append(logits2.argmax(dim=1).cpu())
                    all_labels2.append(batch.y.cpu())

            preds2  = torch.cat(all_preds2)
            labels2 = torch.cat(all_labels2)
            acc2 = accuracy_score(labels2.numpy(), preds2.numpy())
            uar2 = balanced_accuracy_score(labels2.numpy(), preds2.numpy())
            print('======= SECONDARY TEST RESULTS (CAER-test, same model) =======')
            print('Test WAR (Accuracy) : {:.2f}%'.format(acc2 * 100))
            print('Test UAR (Balanced) : {:.2f}%'.format(uar2 * 100))
            print('\nPer-class breakdown:')
            print(classification_report(
                labels2.numpy(), preds2.numpy(),
                labels=list(range(nb_classes)),
                target_names=adapter.emotion_labels,
                zero_division=0
            ))
            print('==============================================================\n')

        # Task 26: multi-checkpoint evaluation table
        # Cheap cross-evaluation (~4 forward passes) that reveals whether the
        # dual-checkpoint design was necessary or whether one optimum covers both
        # domains. Only runs for joint AFEW+CAER folds with val and CAER test.
        if (args.lazy_loading and val_loader is not None and test2_loader is not None):
            _ckpts = []   # list of (label, model_to_eval)

            # 1. Last-epoch
            if 'last_epoch_state_dict' in history:
                _m_last = copy.deepcopy(trained_model)
                _m_last.load_state_dict(history['last_epoch_state_dict'])
                _ckpts.append(('Last epoch ({})'.format(history.get('last_epoch', '?')), _m_last))

            # 2. AFEW-best
            _ckpts.append(('AFEW-best (ep {})'.format(
                history.get('best_epoch', '?')), trained_model))

            # 3. CAER-best
            if trained_model2 is not None:
                _ckpts.append(('CAER-best (ep {})'.format(
                    history.get('best_epoch2', '?')), trained_model2))

            # 4. Joint-best
            if 'joint_best_state_dict' in history:
                _m_joint = copy.deepcopy(trained_model)
                _m_joint.load_state_dict(history['joint_best_state_dict'])
                _ckpts.append(('Joint-best (ep {})'.format(
                    history.get('best_epoch_joint', '?')), _m_joint))

            if _ckpts:
                print('\n===== TASK 26: MULTI-CHECKPOINT EVALUATION =====')
                print('{:<26} {:>14} {:>14}'.format(
                    'Checkpoint', 'AFEW val acc', 'CAER test acc'))
                print('-' * 56)
                for _ckpt_name, _ckpt_model in _ckpts:
                    _ckpt_model.eval()
                    # AFEW val
                    _av_preds, _av_labels = [], []
                    with torch.no_grad():
                        for _vb in val_loader:
                            _vb = _vb.to(device)
                            _vbs = _vb.y.shape[0]
                            _vpf, _vpa = None, None
                            _vff = getattr(_vb, 'full_frames', None)
                            if hasattr(_vb, 'patch_features') and _vb.patch_features is not None:
                                if _vb.patch_features.dim() == 2:
                                    _vpf = _vb.patch_features
                                else:
                                    _vpa = _vb.patch_features
                            _vlogits, _ = _ckpt_model(
                                _vb.x, _vb.edge_index, _vb.batch, _vbs,
                                patch_features=_vpf, patches=_vpa,
                                full_frames=_vff,
                                edge_index_temporal=getattr(_vb, 'edge_index_temporal', None))
                            _av_preds.append(_vlogits.argmax(dim=1).cpu())
                            _av_labels.append(_vb.y.cpu())
                    _av_acc = accuracy_score(
                        torch.cat(_av_labels).numpy(), torch.cat(_av_preds).numpy())
                    # CAER test
                    _ct_preds, _ct_labels = [], []
                    with torch.no_grad():
                        for _tb in test2_loader:
                            _tb = _tb.to(device)
                            _tbs = _tb.y.shape[0]
                            _tpf, _tpa = None, None
                            _tff = getattr(_tb, 'full_frames', None)
                            if hasattr(_tb, 'patch_features') and _tb.patch_features is not None:
                                if _tb.patch_features.dim() == 2:
                                    _tpf = _tb.patch_features
                                else:
                                    _tpa = _tb.patch_features
                            _tlogits, _ = _ckpt_model(
                                _tb.x, _tb.edge_index, _tb.batch, _tbs,
                                patch_features=_tpf, patches=_tpa,
                                full_frames=_tff,
                                edge_index_temporal=getattr(_tb, 'edge_index_temporal', None))
                            _ct_preds.append(_tlogits.argmax(dim=1).cpu())
                            _ct_labels.append(_tb.y.cpu())
                    _ct_acc = accuracy_score(
                        torch.cat(_ct_labels).numpy(), torch.cat(_ct_preds).numpy())
                    print('{:<26} {:>13.2f}% {:>13.2f}%'.format(
                        _ckpt_name, _av_acc * 100, _ct_acc * 100))
                print('=' * 56 + '\n')

        all_acc.append(fold_acc)
        all_prec.append(round(precision_score(labels.numpy(), preds.numpy(),
                                               average='weighted'), 2) * 100)
        all_rec.append(round(recall_score(labels.numpy(), preds.numpy(),
                                           average='weighted'), 2) * 100)

        cm       = save_results(history, fold_idx + 1, labels, preds, out_path)
        cm_final = confusion_matrix(labels.numpy(), preds.numpy(),
                                    labels=list(range(nb_classes)))
        all_cm.append(cm)
        all_cm_final.append(cm_final)

    # Final results
    all_results = [all_acc, all_prec, all_rec, all_cm, all_acc_final, all_cm_final]
    save_results_all_folds(
        all_results, 'GCNTransformerPaper', out_path,
        all_results_path, best_results_path,
        seq_length, nb_joints, nb_train_last, nb_val_last, nb_val_last
    )

    print('\n#############################################')
    print('FINAL RESULTS — {}-fold identity-independent CV'.format(args.nb_cross_val))
    print('#############################################')
    print('Per-fold WAR : {}'.format(
        ['{:.2f}%'.format(a * 100) for a in all_acc_final]))
    print('Per-fold UAR : {}'.format(
        ['{:.2f}%'.format(u * 100) for u in all_uar_final]))
    print('Mean WAR      : {:.2f}%'.format(np.mean(all_acc_final) * 100))
    print('Std  WAR      : {:.2f}%'.format(np.std(all_acc_final)  * 100))
    print('Mean UAR      : {:.2f}%'.format(np.mean(all_uar_final) * 100))
    print('Std  UAR      : {:.2f}%'.format(np.std(all_uar_final)  * 100))
    print('Mean Precision: {:.2f}%'.format(np.mean(all_prec)))
    print('Mean Recall   : {:.2f}%'.format(np.mean(all_rec)))
    print('#############################################\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--chemin_depot_global', type=str, default='/sandbox1/aramadan')
    parser.add_argument('--nom_dataset',         type=str, default='CK+',
                        help='Dataset name: CK+, Oulu, AFEW, ... '
                             'Must match a registered adapter in datasets/__init__.py')
    parser.add_argument('--chemin_patches',      type=str, default=None)

    # Data
    parser.add_argument('--mode_debug',            action='store_false')
    parser.add_argument('--mode_calcul_nb_frames', action='store_true', default=False)
    parser.add_argument('--nb_frames',    type=int,   default=16)   # paper uses 16
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--nb_cross_val', type=int,   default=10)

    # Legacy
    parser.add_argument('--pourcentage_test',             type=float, default=0.8)
    parser.add_argument('--nb_instances_par_classe_test', type=int,   default=5)
    parser.add_argument('--mode_test_pourcentage',        action='store_true', default=False)

    # Model
    parser.add_argument('--batch_size',       type=int,   default=8)
    parser.add_argument('--num_layers',       type=int,   default=2,
                        help='Number of GC blocks in main processor')
    parser.add_argument('--num_layers_trans', type=int,   default=2,
                        help='Number of spatial+temporal transformer pairs')
    parser.add_argument('--n_heads',          type=int,   default=8)
    parser.add_argument('--dim_feedforward',  type=int,   default=1024)
    parser.add_argument('--dropout_transformers', type=float, default=0.1)
    parser.add_argument('--graph_type',       type=str,   default='g_topologique',
                        choices=['g_complet','g_yeux_bouche','g_bouche','g_topologique'])

    # Training (paper values)
    parser.add_argument('--lr',             type=float, default=1e-4,
                        help='Initial lr. Paper: 1e-4')
    parser.add_argument('--weight_decay',   type=float, default=1e-4)
    parser.add_argument('--num_epochs',     type=int,   default=120)
    parser.add_argument('--lambda_center',  type=float, default=0.001,
                        help='Weight of center loss. Paper: 0.001')
    parser.add_argument('--facteur',        type=float, default=10e4)
    parser.add_argument('--apply_scheduler',action='store_false',
                        help='Pass to DISABLE CosineAnnealingLR (enabled by default)')
    parser.add_argument('--class_weight',   action='store_true', default=False)
    parser.add_argument('--device',         type=str,   default='cuda:0',
                        choices=['cuda:0','cpu'])
    parser.add_argument('--print_every',    type=int,   default=10)
    parser.add_argument('--saved_model',    action='store_true', default=True)

    # Lazy data pipeline (for large datasets like CAER / AFEW+CAER)
    parser.add_argument('--lazy_loading',   action='store_true', default=False,
                        help='Stream samples from disk via LazyFERDataset '
                             'instead of pre-loading everything into RAM. '
                             'Required for CAER and the AFEW+CAER joint setup.')
    parser.add_argument('--num_workers',    type=int,   default=4,
                        help='DataLoader workers (only used with --lazy_loading).')

    # Validation / early stopping (lazy path only)
    parser.add_argument('--val_split',      type=float, default=0.0,
                        help='Fraction of TRAIN to carve off as val for '
                             'best-checkpoint selection. 0 disables (paper '
                             'default). Typical: 0.1. Stratified by class. '
                             'Lazy path only.')
    parser.add_argument('--best_metric',    type=str,   default='val_acc',
                        choices=['val_acc', 'val_loss'],
                        help='Metric to pick the best checkpoint from. '
                             'Only used when --val_split > 0.')

    # Frame sampling strategy (lazy path only)
    parser.add_argument('--sampling_strategy', type=str, default='top_k',
                        choices=['top_k', 'uniform'],
                        help="'top_k' (default, paper) picks the K frames "
                             'with the highest movement scores. '
                             "'uniform' picks K frames evenly spaced across "
                             'the clip — recommended for in-the-wild '
                             'datasets (AFEW, CAER) where high-motion '
                             'frames are often head turns or scene cuts.')

    # Alias for class weighting (clearer name; --class_weight still works)
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                        help='Use balanced class weights in CrossEntropyLoss. '
                             'Alias for --class_weight with a clearer name.')

    # In-model ResNet-34 patch encoder (end-to-end / partial fine-tune)
    # The paper trains the local-appearance encoder end-to-end (initialized
    # from FER+) — not the cached-feature pipeline the codebase ships with.
    # When enabled, the lazy dataset switches to on-the-fly patch extraction.
    parser.add_argument('--unfreeze_resnet', type=str, default='none',
                        choices=['none', 'frozen', 'partial', 'full'],
                        help="'none' (default) uses cached ResNet-34 features. "
                             "'partial' unfreezes ResNet-34 layer4 (paper-like, "
                             "lower memory). 'full' unfreezes all of ResNet-34. "
                             "'frozen' puts the encoder in the model but with "
                             "no gradient (sanity test only). Modes other than "
                             "'none' require --lazy_loading and switch to "
                             "on-the-fly patch extraction.")
    parser.add_argument('--encoder_lr', type=float, default=1e-5,
                        help='Learning rate for the unfrozen ResNet-34 '
                             'encoder. Typically 10× lower than --lr. '
                             'Only used when --unfreeze_resnet is partial/full.')
    parser.add_argument('--patch_backbone', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34'],
                        help="Backbone architecture for the in-model patch encoder. "
                             "'resnet34' (default): ResNet-34 pretrained on FER+. "
                             "'resnet18': ResNet-18 pretrained on MS-Celeb-1M "
                             "(10M face images — better face representation). "
                             "Both output 512-d; nothing else changes. "
                             "Pair with the matching --encoder_ckpt.")
    parser.add_argument('--encoder_ckpt', type=str,
                        default='/sandbox1/aramadan/resnet34-epoch.12-val_loss.0.494'
                                '-val_acc.0.846-val_f1.0.843.ckpt',
                        help='Path to the pretrained backbone checkpoint. '
                             'Default: FER+ ResNet-34. For --patch_backbone resnet18 '
                             'pass the Emotion-FAN Resnet18_FER+_pytorch.pth.tar')
    parser.add_argument('--encoder_imagenet_norm', type=lambda x: x.lower() != 'false',
                        default=None, metavar='BOOL',
                        help='Apply ImageNet mean/std normalisation inside the patch '
                             'encoder. Default: auto (True for resnet34, False for '
                             'resnet18). Emotion-FAN Resnet18_FER+_pytorch.pth.tar was '
                             'trained WITHOUT ImageNet norm — passing True with that '
                             'checkpoint corrupts every BN layer. Override with '
                             '--encoder_imagenet_norm false or true.')
    parser.add_argument('--amp', type=str, default='off', choices=['on', 'off'],
                        help='Mixed-precision training (autocast fp16 + GradScaler) in the '
                             'training loop. Needed when the in-model ResNet encoder runs on '
                             'upscaled 224x224 patches: bs*T*P images through a full-precision '
                             'ResNet-18 forward+backward can exceed 16GB even at batch_size=1. '
                             'Default off preserves current behavior.')
    parser.add_argument('--upscale_patches_to_224', type=str, default='off', choices=['on', 'off'],
                        help='Upscale local 49x49 landmark patches to 224x224 before feeding '
                             'the ResNet-18 encoder. The FER+ checkpoint was trained on 224x224 '
                             'face crops; feeding raw 49x49 patches causes ResNet layer4 to '
                             'collapse to 2x2 spatial dim, wasting the second half of the '
                             'network. Default off preserves current behavior for backwards '
                             'compatibility.')
    parser.add_argument('--freeze_encoder_bn', type=str, default='off', choices=['on', 'off'],
                        help='Freeze BatchNorm running stats in the in-model ResNet encoder '
                             '(conv/affine params stay trainable; running_mean/var stay at the '
                             'pretrained checkpoint values instead of being overwritten each step). '
                             'Needed at small batch_size where bs*T*P patches all come from the '
                             'same video per step — an unrepresentative, highly correlated batch '
                             'for BN statistics. Default off preserves current behavior.')
    parser.add_argument('--use_cached_patches', action='store_true', default=False,
                        help='Force cached patch features even when '
                             '--unfreeze_resnet is set. Used to keep CK+/Oulu/'
                             'eNTERFACE05 fast when the encoder is not the '
                             'bottleneck. With --unfreeze_resnet=partial/full '
                             'this flag is meaningless and prints a warning.')
    parser.add_argument('--already_cropped', action='store_true', default=False,
                        help='Skip face bbox lookup; treat input frames as '
                             'already face-cropped (resize to 240×240). Use '
                             'with --nom_dataset=Oulu_Acropped.')

    # On-the-fly landmark re-detection (Tasks 12-16)
    # Re-runs 2D-FAN per frame on the 224×224 augmented image so that the
    # geometry branch and the appearance branch share the same landmark source.
    # Only valid with --unfreeze_resnet=frozen/partial/full (which enables
    # patch_mode='on_the_fly').  Substantially slower than cached mode —
    # confirm it is tolerable on AFEW before using on CAER.
    parser.add_argument('--on_the_fly_landmarks', action='store_true', default=False,
                        help='Re-detect landmarks with 2D-FAN on the augmented '
                             '224×224 image at __getitem__ time so geometry and '
                             'appearance branches share the same landmark source. '
                             'Requires --unfreeze_resnet=frozen/partial/full '
                             '(i.e. on-the-fly patch loading). '
                             'Applied consistently to train, val, test, and '
                             'secondary splits. Substantially slower than cached '
                             'mode — test on AFEW before enabling for CAER.')

    # Global appearance encoder (Approach 1: broadcast)
    # Instead of 51 per-landmark 49×49 patches, encode the full 224×224 face
    # once per frame and broadcast the 512-d feature to all P landmark slots.
    # Requires --unfreeze_resnet (on-the-fly patch loading) and
    # --on_the_fly_landmarks for geometry/appearance consistency.
    parser.add_argument('--temporal_edges', type=str, default='none',
                        choices=['none', 'combined',
                                 'separated_parallel', 'separated_sequential',
                                 'construct_only'],
                        help="Controls temporal-edge construction and routing. "
                             "'none' (default): spatial-only edges, no temporal edges produced. "
                             "'combined': spatial + same-joint t→t+1 edges merged into edge_index. "
                             "'separated_parallel': two parallel SGConv branches in GCBlock. "
                             "'separated_sequential': spatial SGConv then temporal SGConv in GCBlock. "
                             "'construct_only': produces edge_index_temporal (like separated_parallel) "
                             "but GCBlock ignores it — only TGConvOnEdges consumes it. "
                             "Use with --tgconv_mode temporal_edges for clean separation: "
                             "SGConv sees spatial edges only, TGConvOnEdges sees temporal edges only.")
    parser.add_argument('--global_appearance_mode', type=str, default='none_all',
                        choices=['none_all', 'none', 'broadcast', 'roi_align', 'combined',
                                 'roi_align_multiscale', 'local_plus_multiscale',
                                 'fragnet'],
                        help="DEPRECATED single-mode shortcut, kept for backwards "
                             "compatibility. 'none_all' (default): the flag is inactive — "
                             "use the six --use_* component flags below instead. "
                             "Four legacy values auto-translate to the new flag scheme at "
                             "startup (see --use_fragnet_landmark etc.): "
                             "'fragnet'->--use_fragnet_landmark on, "
                             "'broadcast'->--use_broadcast on, "
                             "'none'->--use_local_encoder on, "
                             "'roi_align_multiscale'->--use_multiscale_roi on. "
                             "The remaining three values ('roi_align', 'combined', "
                             "'local_plus_multiscale') have no six-flag equivalent and "
                             "keep running through the original single-mode code path: "
                             "'roi_align': one 224×224 forward pass per frame, bilinearly "
                             "sample a distinct 512-d feature at each landmark's "
                             "image-space coordinates. "
                             "'combined': ROI-Align (512) concat pooled broadcast (512) "
                             "= 1024-d appearance; input_proj grows to 1792→512. "
                             "'local_plus_multiscale': local 49×49 patches (512) concat "
                             "multi-scale ROI-Align (896) = 1408-d appearance; "
                             "input_proj grows to 2176→512. "
                             "Cannot be combined with the six --use_* flags — passing "
                             "both raises an error. Requires --unfreeze_resnet != none.")

    # Modular multi-component appearance fusion (six independent flags)
    # Each component is toggled ON/OFF independently; outputs concatenate at
    # the fusion layer (see GCNTransformerPaper / GridBasedFragNet). Passing
    # any of these overrides --global_appearance_mode entirely.
    parser.add_argument('--use_fragnet_landmark', type=str, default='off', choices=['on', 'off'],
                        help='Landmark-guided FragNet (dual-pathway with lateral fusion). '
                             'Per-landmark 512-d features. Requires patches + full_frames.')
    parser.add_argument('--use_broadcast', type=str, default='off', choices=['on', 'off'],
                        help='Broadcast a single global 512-d feature to all P landmarks per frame. '
                             'Requires full_frames.')
    parser.add_argument('--use_fragnet_grid', type=str, default='off', choices=['on', 'off'],
                        help='Grid-based FragNet (fixed vertical strips laterally fused with a '
                             'ResNet feature pyramid, no landmark guidance). Requires full_frames.')
    parser.add_argument('--use_local_encoder', type=str, default='off', choices=['on', 'off'],
                        help='Local 49x49 landmark patches through the ResNet patch encoder. '
                             'Requires patches.')
    parser.add_argument('--use_multiscale_roi', type=str, default='off', choices=['on', 'off'],
                        help='Multi-scale ROI-Align sampling at landmark positions '
                             '(layer2+layer3+layer4). Requires full_frames.')
    parser.add_argument('--fragnet_grid_backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='Backbone for grid-based FragNet only. resnet18 shares weights '
                             'with the other components (no extra params); resnet50 is a '
                             'separate ImageNet-pretrained encoder (~25M extra params, no '
                             'FER+ checkpoint exists at this depth).')

    # TGConv variant
    parser.add_argument('--tgconv_mode', type=str, default='learnable_A',
                        choices=['learnable_A', 'temporal_edges'],
                        help="Controls the joint-mixing path inside TemporalConvBlock. "
                             "'learnable_A' (default): current behaviour — unconstrained "
                             "51×51 learnable adjacency matrix. "
                             "'temporal_edges': replaces the learnable A with proper "
                             "temporal graph convolution (PyG SGConv, K=2) on same-joint "
                             "t→t+1 edges. Requires --temporal_edges != none.")

    # Whole-component ablation
    parser.add_argument('--use_coord_encoder', type=str, default='on',
                        choices=['on', 'off'],
                        help="Whether the coordinate encoder (R, C, T branches) is "
                             "active. When off, the model uses only appearance "
                             "features (V), and the fusion input has no geometry "
                             "contribution. Requires patches to be available.")
    parser.add_argument('--use_gcblocks', type=str, default='on',
                        choices=['on', 'off'],
                        help="Whether the GCBlock stack is active. When off, the "
                             "model skips graph convolution and passes the fused "
                             "features straight to the transformer stack.")
    parser.add_argument('--use_transformers', type=str, default='on',
                        choices=['on', 'off'],
                        help="Whether the transformer stack is active. When off, "
                             "the model skips transformer encoding and passes "
                             "features directly to the classification head.")

    main()