#! /usr/bin/python3
# Author : Ahmad Ramadan
"""
train_gnn_paper.py, Main training script for GCNTransformerPaper.

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
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.utils import class_weight
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Local imports ──────────────────────────────────────────────────────────
import sys, importlib

# We import from the paper files placed in the same src/ directory
from dataloader.build_dataset_paper import (
    convertion_graphe, convertir_liste_patches_en_tenseur,
    convertir_liste_videos_en_tenseur, create_graph,
    data_augmentation_equilibration_classe, extract_datas_video_labels,
    joints_connections, make_dirs, save_pickle, load_pickle,
    traitements_video_par_video
)
from models.gnn_paper import CenterLoss, GCNTransformerPaper
from executor.trainer_paper import train_model_gnn_paper
from results import save_results, save_results_all_folds
from utils import *


# ─────────────────────────────────────────────────────────────────────────────

def chemin(chemin_depot_global, nom_dataset, type_graph, filename, mode_test):
    chemin_landmark_expr = os.path.join(chemin_depot_global, 'landmark', nom_dataset)
    if mode_test:
        all_results_path  = os.path.join(chemin_depot_global, 'resultats/test',  type_graph, 'results_test.txt')
        best_results_path = os.path.join(chemin_depot_global, 'resultats/test',  type_graph, 'sort_results_test.txt')
        chemin_labels     = os.path.join(chemin_depot_global, 'labels/labels_test.pkl')
        out_path          = os.path.join(chemin_depot_global, 'resultats/test',  type_graph, filename)
    else:
        all_results_path  = os.path.join(chemin_depot_global, 'resultats/train', type_graph, 'results.txt')
        best_results_path = os.path.join(chemin_depot_global, 'resultats/train', type_graph, 'sort_results.txt')
        chemin_labels     = os.path.join(chemin_depot_global, 'labels/labels.pkl')
        out_path          = os.path.join(chemin_depot_global, 'resultats/train', type_graph, filename)
    return all_results_path, best_results_path, chemin_landmark_expr, chemin_labels, out_path


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parser.parse_args()

    if args.graph_type not in ['g_complet', 'g_yeux_bouche', 'g_bouche', 'g_topologique']:
        raise NameError('graph_type invalide.')

    print('Type du graphe = {}'.format(args.graph_type))

    # ── Patches ─────────────────────────────────────────────────────────────
    use_patches = args.chemin_patches is not None
    chemin_patches_dataset = None
    if use_patches:
        chemin_patches_dataset = os.path.join(args.chemin_patches, args.nom_dataset)
        if not os.path.exists(chemin_patches_dataset):
            print(f'[warn] Patches folder not found: {chemin_patches_dataset}')
            print('       Running in coordinate-only mode.')
            use_patches = False

    if use_patches:
        print('Patch mode : ON  (coordinates + ResNet-34 FER+ appearance features)')
    else:
        print('Patch mode : OFF (coordinates only)')

    filename        = time.strftime('%Y%m%d-%H%M%S')
    use_test_paths  = args.mode_debug

    (all_results_path, best_results_path,
     chemin_landmark_expr, chemin_labels, out_path) = chemin(
        args.chemin_depot_global, args.nom_dataset,
        args.graph_type, filename, use_test_paths)

    make_dirs(out_path)
    with open(os.path.join(out_path, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # ── Load data ────────────────────────────────────────────────────────────
    all_videos_landmark, all_videos_patches, nb_frames_par_video, liste_labels = \
        extract_datas_video_labels(
            chemin_landmark_expr,
            chemin_labels,
            path_patches=chemin_patches_dataset if use_patches else None
        )
    liste_labels = np.array(liste_labels)

    # ── Filter to graph-specific joints (returns 5 values now) ───────────────
    nb_joints, all_videos_landmark, all_videos_patches, connections_par_frames, kept_indices = \
        convertion_graphe(
            all_videos_landmark,
            args.graph_type,
            all_videos_patches=all_videos_patches if use_patches else None
        )

    print('nb_joints    = {}'.format(nb_joints))
    print('kept_indices : {} landmarks  (first 5: {})'.format(
        len(kept_indices), kept_indices[:5]))

    nb_classes = len(set(liste_labels))

    # Label normalisation
    if set(liste_labels) != set(range(nb_classes)):
        classes_intermediaire = sorted(list(set(liste_labels)))
        correspondance        = {classes_intermediaire[i]: i for i in range(nb_classes)}
        liste_labels          = np.array([correspondance[l] for l in liste_labels])

    # ── Temporal alignment ────────────────────────────────────────────────────
    nb_frames_par_video_preproc, processed_video_landmarks, processed_video_patches = \
        traitements_video_par_video(
            all_videos_landmark,
            nb_frames_par_video,
            args.mode_calcul_nb_frames,
            args.nb_frames,
            liste_videos_patches=all_videos_patches if use_patches else None
        )

    # ── Convert to tensors ────────────────────────────────────────────────────
    all_videos_tensor = convertir_liste_videos_en_tenseur(processed_video_landmarks)
    if use_patches and processed_video_patches is not None:
        all_patches_tensor = convertir_liste_patches_en_tenseur(processed_video_patches)
    else:
        all_patches_tensor = None

    # ── Subject IDs for identity-independent GroupKFold ───────────────────────
    chemin_subject_ids = args.chemin_subject_ids or \
        os.path.join(args.chemin_depot_global, 'labels', 'subject_ids.pkl')
    subject_ids = np.array(load_pickle(chemin_subject_ids))

    assert len(subject_ids) == len(liste_labels), (
        'subject_ids ({}) et liste_labels ({}) taille différente!'.format(
            len(subject_ids), len(liste_labels)))

    print('\nNombre de sujets uniques : {}  →  {}-fold identity-independent CV'.format(
        len(set(subject_ids)), args.nb_cross_val))
    print('Distribution labels : {}'.format(np.unique(liste_labels, return_counts=True)))

    # ── 10-fold GroupKFold ────────────────────────────────────────────────────
    gkf   = GroupKFold(n_splits=args.nb_cross_val)
    folds = list(gkf.split(X=np.arange(len(liste_labels)),
                            y=liste_labels, groups=subject_ids))

    all_acc, all_prec, all_rec, all_cm = [], [], [], []
    all_acc_final, all_cm_final         = [], []

    gen_alea   = torch.Generator()
    gen_alea.manual_seed(args.seed)
    device     = torch.device(args.device)
    seq_length = nb_frames_par_video_preproc
    nb_train_last, nb_val_last = 0, 0

    for fold_idx, (fold_train_idx, fold_test_idx) in enumerate(folds):

        print('\n#-------------------------- Fold {} / {} --------------------------#\n'.format(
            fold_idx + 1, args.nb_cross_val))

        train_subjects = set(subject_ids[fold_train_idx])
        test_subjects  = set(subject_ids[fold_test_idx])
        assert len(train_subjects & test_subjects) == 0, 'FUITE DE DONNÉES !'

        print('Sujets test  ({:2d}) : {}'.format(len(test_subjects), sorted(test_subjects)))
        print('Sujets train ({:2d}) : {} sujets'.format(len(train_subjects), len(train_subjects)))

        # ── Split ───────────────────────────────────────────────────────────
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

        # ── Augmentation (train only) ────────────────────────────────────────
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

        # ── Build PyG graphs ─────────────────────────────────────────────────
        data_list_train = [
            create_graph(
                trainX[:, :, idx], trainY[idx],
                nb_frames_par_video_preproc, nb_joints,
                connections_par_frames,
                patch_tensor=trainP[:, :, idx] if trainP is not None else None
            )
            for idx in tqdm(range(len(trainY)), total=len(trainY))
        ]
        data_list_test = [
            create_graph(
                testX[:, :, idx], testY[idx],
                nb_frames_par_video_preproc, nb_joints,
                connections_par_frames,
                patch_tensor=testP[:, :, idx] if testP is not None else None
            )
            for idx in tqdm(range(len(testY)), total=len(testY))
        ]

        train_loader = GeoDataLoader(data_list_train, batch_size=args.batch_size,
                                     shuffle=True, generator=gen_alea)
        test_loader  = GeoDataLoader(data_list_test,  batch_size=args.batch_size,
                                     shuffle=False)

        dataloaders = {'train': train_loader}
        out_dim     = nb_classes
        print('Le dataset possède {} classes'.format(out_dim))

        # ── Model ─────────────────────────────────────────────────────────────
        model = GCNTransformerPaper(
            nb_joint         = nb_joints,
            seq_length       = seq_length,
            n_heads          = args.n_heads,
            dim_feedforward  = args.dim_feedforward,
            num_gc_blocks    = args.num_layers,
            num_trans_layers = args.num_layers_trans,
            out_dim          = out_dim,
            kept_indices     = kept_indices,
            dropout          = args.dropout_transformers,
            use_patches      = use_patches
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('\nTotal Trainable Parameters : {}\n'.format(total_params))
        print('Les labels sont : {}'.format(torch.unique(trainY)))

        # ── Center Loss ────────────────────────────────────────────────────────
        feat_dim    = 256   # output dim of model.pre_logit
        center_loss = CenterLoss(nb_classes, feat_dim).to(device)

        # ── Loss + Optimizers ─────────────────────────────────────────────────
        if args.class_weight:
            weights   = torch.tensor(
                class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(liste_labels),
                    y=trainY.numpy()
                ), dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=weights.to(device), reduction='mean')
            print('Class weights : {}\n'.format(np.round(weights.numpy(), 3)))
        else:
            print('Not using class weight !\n')
            criterion = nn.CrossEntropyLoss()

        optimizer_model  = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_center = torch.optim.Adam(
            center_loss.parameters(), lr=0.5)   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_model, T_max=args.num_epochs, eta_min=1e-6)

        # ── Train ─────────────────────────────────────────────────────────────
        trained_model, history = train_model_gnn_paper(
            model, criterion, center_loss, args.lambda_center,
            optimizer_model, optimizer_center,
            args.apply_scheduler, scheduler,
            dataloaders, device, args.num_epochs, args.print_every
        )

        # ── Test (evaluated ONCE — paper protocol) ────────────────────────────
        trained_model.eval()
        center_loss.eval()
        all_preds, all_labels_list = [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                bs    = batch.y.shape[0]

                patch_feat = None
                if hasattr(batch, 'patch_features') and batch.patch_features is not None:
                    patch_feat = batch.patch_features

                logits, _ = trained_model(
                    batch.x, batch.edge_index, batch.batch, bs,
                    patch_features=patch_feat
                )
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels_list.append(batch.y.cpu())

        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels_list)

        # ── Save model ────────────────────────────────────────────────────────
        if args.saved_model:
            model_path = os.path.join(out_path, 'models')
            make_dirs(model_path)
            torch.save(trained_model.state_dict(),
                       os.path.join(model_path, 'fold{}_weights.pth'.format(fold_idx + 1)))

        fold_acc = accuracy_score(labels.numpy(), preds.numpy())
        all_acc_final.append(fold_acc)

        print('\n================ TEST RESULTS (Fold {}/{}) ================'.format(
            fold_idx + 1, args.nb_cross_val))
        print('Sujets testés       : {}'.format(sorted(test_subjects)))
        print('Test Accuracy       : {:.2f}%'.format(fold_acc * 100))
        print('Precision (weighted): {:.2f}%'.format(
            precision_score(labels.numpy(), preds.numpy(), average='weighted') * 100))
        print('Recall    (weighted): {:.2f}%'.format(
            recall_score(labels.numpy(), preds.numpy(), average='weighted') * 100))
        print('============================================================\n')

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

    # ── Final results ──────────────────────────────────────────────────────────
    all_results = [all_acc, all_prec, all_rec, all_cm, all_acc_final, all_cm_final]
    save_results_all_folds(
        all_results, 'GCNTransformerPaper', out_path,
        all_results_path, best_results_path,
        seq_length, nb_joints, nb_train_last, nb_val_last, nb_val_last
    )

    print('\n#############################################')
    print('FINAL RESULTS — {}-fold identity-independent CV'.format(args.nb_cross_val))
    print('#############################################')
    print('Per-fold accuracies : {}'.format(
        ['{:.2f}%'.format(a * 100) for a in all_acc_final]))
    print('Mean Accuracy : {:.2f}%'.format(np.mean(all_acc_final) * 100))
    print('Std  Accuracy : {:.2f}%'.format(np.std(all_acc_final)  * 100))
    print('Mean Precision: {:.2f}%'.format(np.mean(all_prec)))
    print('Mean Recall   : {:.2f}%'.format(np.mean(all_rec)))
    print('#############################################\n')


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--chemin_depot_global', type=str, default='/sandbox1/aramadan')
    parser.add_argument('--nom_dataset',         type=str, default='CK+')
    parser.add_argument('--chemin_patches',      type=str, default=None)
    parser.add_argument('--chemin_subject_ids',  type=str, default=None)

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

    main()
