#!/usr/bin/env python3
"""
datasets/afew_caer_joint.py
Joint AFEW + CAER adapter — paper Section 4.3.4 protocol.

    "We follow the same setting in [3], and form a joint in-the-wild
     data set with AFEW to train and evaluate our proposed framework
     for in-the-wild video-based FER."

Training set:  AFEW-train ∪ CAER-train
Evaluation  :  Fold 1 → AFEW val   (paper Table 5, headline 51.17%)
               Fold 2 → CAER test  (paper Table 6, headline 77.06%)

Both folds reuse the same combined training set. The training loop will
therefore train the model TWICE (once per fold). If you want to skip
the second training and reuse the first model, see the note in
train_gnn_paper.py.

Disk layout
-----------
This adapter unifies the two datasets under a single virtual dataset
name `AFEW+CAER`. The first time get_all_samples() is called, it:

  1. Writes a merged labels pkl at:
        <base>/labels/AFEW+CAER_labels.pkl
  2. Creates symlinks pointing the merged dataset's landmark/ and
     patches/ folders to the *source* per-subject directories. This
     keeps zero data duplication and lets extract_datas_video_labels
     find everything under a single root.

After preparation, every (subject, video, label) tuple is tagged with
a source prefix in `subject` so the AFEW vs CAER membership stays
recoverable in get_folds():

    'AFEW_train_000046280'   (AFEW Train sample)
    'AFEW_val_000149120'     (AFEW Val sample)
    'CAER_train_0001'        (CAER train sample)
    'CAER_test_0001'         (CAER test sample)
    'CAER_val_0001'          (CAER validation — excluded from training)

The AFEW and CAER label spaces are identical here (same 7 emotion names
in the same order), so no remapping is needed:
    0=Anger, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

Prerequisites
-------------
You must have already run, for BOTH datasets independently:
    make organize_afew  / make organize_caer
    make landmarks dataset=AFEW   / make landmarks dataset=CAER
    make patches   dataset=AFEW   / make patches   dataset=CAER

Then training is:
    make train dataset='AFEW+CAER' graph_type=g_topologique \\
         PARAM_CONST="--mode_debug --nb_cross_val 1"
"""

import os
from typing import List, Tuple, Dict

from .base   import DatasetAdapter
from .afew   import AFEWAdapter
from .caer   import CAERAdapter
from utils   import load_pickle, save_pickle, make_dirs


# Canonical emotion names: map any known alias to its canonical lowercase form.
# Add entries here if new dataset adapters use different spellings.
_EMOTION_ALIASES: Dict[str, str] = {
    'angry': 'anger',   # AFEW uses 'Angry'; CAER uses 'Anger' — both → 'anger'
}


def _normalize_emotion(name: str) -> str:
    """Lowercase + strip + apply known aliases."""
    n = name.strip().lower()
    return _EMOTION_ALIASES.get(n, n)


def _normalized_label_map(label_map: Dict[str, int]) -> Dict[str, int]:
    return {_normalize_emotion(k): v for k, v in label_map.items()}


class AFEWCAERJointAdapter(DatasetAdapter):

    num_classes    = 7
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy',
                      'Sad', 'Surprise', 'Neutral']
    dataset_name   = 'AFEW+CAER'
    cv_strategy    = 'fixed_split'

    # Make sure these match exactly between the two source adapters.
    # AFEW.LABEL_MAP and CAER.LABEL_MAP both use this order — verified
    # at __init__ time below.

    def __init__(self, args):
        super().__init__(args)
        self._afew = AFEWAdapter(args)
        self._caer = CAERAdapter(args)

        # Sanity check: both adapters must assign the same integer to each emotion
        # after name normalization ('Angry'/'Anger' are aliases; both → 'anger').
        # Comparing only sorted(values()) would miss a swap like
        #   AFEW: Angry→0, Neutral→6   vs   CAER: Neutral→0, Anger→6
        # — both have value-set {0..6} but produce silently scrambled joint labels.
        afew_norm = _normalized_label_map(self._afew.LABEL_MAP)
        caer_norm = _normalized_label_map(self._caer.LABEL_MAP)
        if afew_norm != caer_norm:
            raise RuntimeError(
                'AFEW and CAER label mappings disagree after name normalization. '
                'The joint adapter requires an identical emotion→int assignment.\n'
                f'  AFEW (normalized): {afew_norm}\n'
                f'  CAER (normalized): {caer_norm}'
            )
        print(f'[AFEW+CAER] Verified emotion→label mapping: {afew_norm}')

        self._prepared = False   # symlinks + merged labels pkl
        # Prepare the joint disk layout NOW so downstream loaders (which read
        # from disk via labels_pkl_path() before get_all_samples() is called)
        # find the merged labels pkl and the symlinked landmark/patches dirs.
        self._prepare_joint_layout()


    # Path overrides

    def _src_and_key(self, subject: str):
        """Return (source_adapter, stripped_subject) for a prefixed subject."""
        if subject.startswith('AFEW_'):
            return self._afew, subject[len('AFEW_'):]
        if subject.startswith('CAER_'):
            return self._caer, subject[len('CAER_'):]
        return self, subject   # fallback (should not happen)

    def frames_dir(self, subject: str, video: str) -> str:
        src, key = self._src_and_key(subject)
        return src.frames_dir(key, video)

    def landmark_pkl(self, subject: str, video: str) -> str:
        src, key = self._src_and_key(subject)
        return src.landmark_pkl(key, video)

    def patches_pkl(self, subject: str, video: str) -> str:
        src, key = self._src_and_key(subject)
        return src.patches_pkl(key, video)

    def bbox_pkl(self, subject: str, video: str) -> str:
        src, key = self._src_and_key(subject)
        return src.bbox_pkl(key, video)

    # Joint disk layout preparation

    def _prepare_joint_layout(self):
        """
        Create the unified AFEW+CAER directory tree by symlinking each
        source subject into a single merged folder, and write a merged
        labels pkl.
        """
        if self._prepared:
            return

        base = self.base_dir
        merged_lm   = os.path.join(base, 'landmark', self.dataset_name)
        merged_pat  = os.path.join(base, 'patches',  self.dataset_name)
        labels_dir  = os.path.join(base, 'labels')
        make_dirs(merged_lm)
        make_dirs(merged_pat)
        make_dirs(labels_dir)

        merged_labels = {}

        for src_name, src in (('AFEW', self._afew), ('CAER', self._caer)):
            src_lm_root  = os.path.join(base, 'landmark', src.dataset_name)
            src_pat_root = os.path.join(base, 'patches',  src.dataset_name)
            src_labels   = load_pickle(src.labels_pkl_path())

            for subject_id, videos in src_labels.items():
                merged_subj = f'{src_name}_{subject_id}'

                # Symlink landmark folder for this subject
                src_lm_subj = os.path.join(src_lm_root,  subject_id)
                dst_lm_subj = os.path.join(merged_lm,    merged_subj)
                if os.path.isdir(src_lm_subj) and not os.path.lexists(dst_lm_subj):
                    os.symlink(src_lm_subj, dst_lm_subj)

                # Symlink patches folder for this subject
                src_pat_subj = os.path.join(src_pat_root, subject_id)
                dst_pat_subj = os.path.join(merged_pat,   merged_subj)
                if os.path.isdir(src_pat_subj) and not os.path.lexists(dst_pat_subj):
                    os.symlink(src_pat_subj, dst_pat_subj)

                # Carry labels forward under the new subject key
                merged_labels[merged_subj] = dict(videos)

        save_pickle(self.labels_pkl_path(), merged_labels)
        print(f'[AFEW+CAER] Joint layout ready: {len(merged_labels)} subjects.')
        self._prepared = True

    # Interface

    def get_all_samples(self) -> List[Tuple[str, str, int]]:
        self._prepare_joint_layout()

        samples = []
        n_afew_train = n_afew_val = 0
        n_caer_train = n_caer_val = n_caer_test = 0

        # AFEW: subject was {train,val}_<videoID>; merged_subj prefixes with 'AFEW_'.
        for subject, video, label in self._afew.get_all_samples():
            merged_subj = f'AFEW_{subject}'
            samples.append((merged_subj, video, label))
            if subject.startswith('train_'): n_afew_train += 1
            elif subject.startswith('val_'): n_afew_val   += 1

        # CAER: subject was {train,val,test}_<videoID>; merged_subj prefixes 'CAER_'.
        for subject, video, label in self._caer.get_all_samples():
            merged_subj = f'CAER_{subject}'
            samples.append((merged_subj, video, label))
            if   subject.startswith('train_'): n_caer_train += 1
            elif subject.startswith('val_'):   n_caer_val   += 1
            elif subject.startswith('test_'):  n_caer_test  += 1

        print(f'[AFEW+CAER] Total samples: {len(samples)}\n'
              f'             AFEW  → train: {n_afew_train}, val: {n_afew_val}\n'
              f'             CAER  → train: {n_caer_train}, '
              f'val: {n_caer_val}, test: {n_caer_test}')
        return samples

    def get_folds(self) -> List[Dict]:
        """
        One fold: train once on AFEW-train ∪ CAER-train, then evaluate on both
        held-out test splits from the same checkpoint — no retraining for CAER.

            Train            : AFEW-train ∪ CAER-train (joint)
            Primary eval     : AFEW val        → fold['test']
                               paper Table 5 target  51.17%
            Secondary eval   : CAER test       → fold['test_secondary']
                               paper Table 6 target  77.06%

        CAER val is excluded from both training and evaluation.
        The best checkpoint is selected on the primary (AFEW val) split; the
        same weights are then evaluated on CAER test without any further tuning.
        """
        samples = self.get_all_samples()
        if not samples:
            raise RuntimeError('No AFEW+CAER samples found.')

        # Fold 1 train: AFEW-train ∪ CAER-train (joint)
        joint_train = [s for s in samples
                       if s[0].startswith('AFEW_train_')
                       or s[0].startswith('CAER_train_')]

        afew_test = [s for s in samples if s[0].startswith('AFEW_val_')]
        caer_test = [s for s in samples if s[0].startswith('CAER_test_')]

        if not joint_train:
            raise RuntimeError('AFEW+CAER: no joint training samples.')
        if not afew_test:
            raise RuntimeError('AFEW+CAER: no AFEW val samples.')
        if not caer_test:
            raise RuntimeError('AFEW+CAER: no CAER test samples.')

        print(f'[AFEW+CAER] Train (AFEW + CAER joint): {len(joint_train)} samples')
        print(f'[AFEW+CAER]   Primary   eval (AFEW val) : {len(afew_test)}')
        print(f'[AFEW+CAER]   Secondary eval (CAER test): {len(caer_test)}')

        # Single fold: train once on joint data, evaluate on both test sets.
        # test_secondary is evaluated on the same saved checkpoint — no retraining.
        return [
            {
                'fold_id':        1,
                'train':          joint_train,
                'test':           afew_test,   # primary   → AFEW val  (→51.17% paper)
                'test_secondary': caer_test,   # secondary → CAER test (→77.06% paper)
            }
        ]