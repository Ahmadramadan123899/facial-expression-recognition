#!/usr/bin/env python3
"""
datasets/dfew.py
DFEW (Dynamic Facial Expression in the Wild) adapter — 5-fold CV protocol.

DFEW is annotated by 10 raters per clip with single-class labels. The
official protocol keeps only the 12,059 single-labeled clips (where one
emotion received a clear majority of votes) and reports UAR/WAR averaged
over 5 non-overlapping folds (fd1–fd5).

The official fold CSVs (single_trainset_{1..5}.csv, single_testset_{1..5}.csv)
are distributed separately from the dataset itself via Baidu Disk
(code: 8azt) or Google Drive — see https://github.com/CV-AC/DFEW.
This adapter supports two modes:
  - If `/sandbox1/aramadan/dfew_folds/single_trainset_<k>.csv` and
    `single_testset_<k>.csv` exist, they are loaded and used as-is.
  - Otherwise we derive 5 stratified folds locally with
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
    The 5-fold *average* UAR/WAR is then directly comparable to the
    paper — only individual fd1/fd2/... numbers will differ.

Data layout assumed on this server:
  Frames: /data/fox-data/datasets/emotions/DFEW_original/Clip/clip_224x224_16f/
          {00001..16372}/{1..16}.jpg
          → 224×224 face-cropped, temporally normalized to exactly 16 frames.
  Annotation: /data/fox-data/datasets/emotions/DFEW_original/Annotation/
              annotation.xlsx
              Columns: ['1happy','2sad','3neutral','4angry','5surprise',
                       '6disgust','7fear','order','label']
              `label` is 1-indexed (1=Happy ... 7=Fear); `label=0` means
              the clip is multi-labeled / ambiguous and is excluded.

`subject` is the zero-padded clip ID ('00001'). `video` is set equal to
`subject` (rather than empty) so the preprocessing script's path
convention `landmark/{dataset}/{subject}/{video}_data.pkl` produces a
sane filename `landmark/DFEW/00001/00001_data.pkl`. Inside this adapter
we override `frames_dir` so the path is determined by `subject` alone.

The frames_dir / landmark_pkl / patches_pkl paths are overridden to use
the actual on-server locations rather than the project's standard
{base}/frames/DFEW/... layout, because we don't want to duplicate
60 GB of already-prepared frames.

Use with --already_cropped (frames are already 224×224 face crops, so the
on-the-fly patch extractor just resizes to 240×240 and skips bbox lookup).
"""

import os
from typing import List, Tuple, Dict

import numpy as np

from .base import DatasetAdapter


# Server-side paths for DFEW. Constants so they can be overridden in tests.
DFEW_FRAMES_ROOT = ('/data/fox-data/datasets/emotions/'
                    'DFEW_original/Clip/clip_224x224_16f')
DFEW_ANNOTATION  = ('/data/fox-data/datasets/emotions/'
                    'DFEW_original/Annotation/annotation.xlsx')


class DFEWAdapter(DatasetAdapter):

    num_classes    = 7
    # Order matches the xlsx label encoding (1-indexed → 0-indexed):
    #   1happy → 0, 2sad → 1, 3neutral → 2, 4angry → 3,
    #   5surprise → 4, 6disgust → 5, 7fear → 6
    emotion_labels = ['Happy', 'Sad', 'Neutral', 'Angry',
                      'Surprise', 'Disgust', 'Fear']
    dataset_name   = 'DFEW'
    cv_strategy    = 'fixed_split'   # we ship 5 folds via get_folds()

    NB_FRAMES_ON_DISK = 16     # DFEW is pre-temporally-normalized to 16 frames

    # Path overrides
    # Frames are NOT copied to /sandbox1/aramadan/frames/DFEW/. They live
    # at DFEW_FRAMES_ROOT directly. Override frames_dir to point there.
    # Landmarks and patches still go under /sandbox1/aramadan/ following the
    # project's standard {subject}/{video}_*.pkl layout (with video=subject
    # here, so the filename becomes "{clip_id}/{clip_id}_data.pkl").

    def frames_dir(self, subject: str, video: str = '') -> str:
        # `video` is unused for DFEW — frames are one folder per clip.
        return os.path.join(DFEW_FRAMES_ROOT, subject)

    # Interface

    def _load_annotation(self) -> List[Tuple[str, str, int]]:
        """
        Returns the list of (subject, video, label_0idx) for every clip
        with label != 0 (single-labeled subset).
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            raise ImportError(
                'DFEW adapter needs pandas. pip install pandas openpyxl')

        if not os.path.exists(DFEW_ANNOTATION):
            raise FileNotFoundError(
                f'DFEW annotation file not found at {DFEW_ANNOTATION}. '
                f'Adjust DFEW_ANNOTATION in datasets/dfew.py if needed.')

        df = pd.read_excel(DFEW_ANNOTATION)
        # Drop multi-labeled / ambiguous clips
        df = df[df['label'] != 0].copy()
        df['label_0idx'] = df['label'].astype(int) - 1   # 1..7 → 0..6

        samples = []
        for _, row in df.iterrows():
            clip_id = int(row['order'])
            subject = f'{clip_id:05d}'              # zero-pad to 5 digits
            # video=subject so per-clip pickles land at
            # landmark/DFEW/{subject}/{subject}_data.pkl (preprocessing layout)
            samples.append((subject, subject, int(row['label_0idx'])))
        return samples

    def get_all_samples(self) -> List[Tuple[str, str, int]]:
        """
        Returns the 12,059-element list of single-labeled DFEW clips.
        Filters out clips whose frames folder doesn't exist (defensive —
        should never trigger).
        """
        if not os.path.isdir(DFEW_FRAMES_ROOT):
            raise FileNotFoundError(
                f'DFEW frames root not found at {DFEW_FRAMES_ROOT}. '
                f'Adjust DFEW_FRAMES_ROOT in datasets/dfew.py if needed.')

        annotated = self._load_annotation()

        samples  = []
        n_missing = 0
        for subject, video, label in annotated:
            frames_path = self.frames_dir(subject, video)
            if not os.path.isdir(frames_path):
                n_missing += 1
                continue
            samples.append((subject, video, label))

        if n_missing > 0:
            print(f'[DFEW] {n_missing} annotated clips have no frames '
                  f'folder under {DFEW_FRAMES_ROOT} (skipped).')

        # Class distribution sanity check
        labels = np.array([s[2] for s in samples])
        bincount = np.bincount(labels, minlength=self.num_classes)
        print(f'[DFEW] Found {len(samples)} single-labeled clips. '
              f'Class counts: {dict(zip(self.emotion_labels, bincount.tolist()))}')

        return samples

    # Folds

    def _try_load_official_folds(self) -> List[Dict]:
        """
        If single_trainset_<k>.csv and single_testset_<k>.csv exist under
        {base_dir}/dfew_folds/, load them. Returns [] if not present.

        Expected CSV format (one row per clip):
            <clip_id>,<label_1idx>
        e.g.
            12,2
            45,4
        clip_id is 1..16372, label is 1..7. Header optional.
        """
        folds_dir = os.path.join(self.base_dir, 'dfew_folds')
        if not os.path.isdir(folds_dir):
            return []

        try:
            import pandas as pd  # type: ignore
        except ImportError:
            return []

        official_folds = []
        for k in range(1, 6):
            train_csv = os.path.join(folds_dir, f'single_trainset_{k}.csv')
            test_csv  = os.path.join(folds_dir, f'single_testset_{k}.csv')
            if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
                return []   # fall back to local stratification

            def _parse_csv(path):
                # Be tolerant of header presence and column order.
                df = pd.read_csv(path, header=None)
                if df.iloc[0, 0] in ('order', 'clip_id', 'id'):
                    df = df.iloc[1:].reset_index(drop=True)
                df.columns = ['clip_id', 'label_1idx'][:df.shape[1]]
                out = []
                for _, row in df.iterrows():
                    cid = int(row['clip_id'])
                    lab = int(row['label_1idx']) - 1
                    subj = f'{cid:05d}'
                    out.append((subj, subj, lab))   # video=subject
                return out

            official_folds.append({
                'fold_id': k,
                'train':   _parse_csv(train_csv),
                'test':    _parse_csv(test_csv),
            })

        print(f'[DFEW] Loaded official 5 folds from {folds_dir}')
        return official_folds

    def get_folds(self) -> List[Dict]:
        # Prefer official fold CSVs if user has dropped them in
        official = self._try_load_official_folds()
        if official:
            return official

        # Else: generate 5 stratified folds locally
        from sklearn.model_selection import StratifiedKFold  # type: ignore

        samples = self.get_all_samples()
        if not samples:
            raise RuntimeError('No DFEW samples found.')

        labels = np.array([s[2] for s in samples])
        skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        folds = []
        for k, (train_idx, test_idx) in enumerate(
                skf.split(np.zeros(len(samples)), labels), start=1):
            folds.append({
                'fold_id': k,
                'train':   [samples[i] for i in train_idx],
                'test':    [samples[i] for i in test_idx],
            })
        print(f'[DFEW] Generated 5 local stratified folds '
              f'(random_state=42). For paper-comparable per-fold numbers, '
              f'drop the official CSVs into {self.base_dir}/dfew_folds/.')
        return folds