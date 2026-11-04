#!/usr/bin/env python3
"""
datasets/ckplus.py
CK+ adapter — wraps the existing CSV-based label discovery and
10-fold identity-independent GroupKFold logic that was previously
hardcoded in train_gnn_paper.py and preprocessing_exp_faciale.py.
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from .base import DatasetAdapter


class CKPlusAdapter(DatasetAdapter):

    num_classes    = 7
    emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear',
                      'Happiness', 'Sadness', 'Surprise']
    dataset_name   = 'CK+'
    cv_strategy    = 'group_kfold'

    def __init__(self, args):
        super().__init__(args)
        self._csv_path = os.path.join(args.chemin_depot_global, 'emotion_327.csv')

    # Internal helpers

    def _load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self._csv_path, sep=';')
        return df

    # Interface

    def get_all_samples(self) -> List[Tuple[str, str, int]]:
        """
        Reads emotion_327.csv and returns every (subject, video, label)
        triple whose frame folder exists on disk.
        """
        df      = self._load_csv()
        samples = []
        frames_root = os.path.join(self.args.chemin_depot_global,
                                   'frames', self.dataset_name)

        for _, row in df.iterrows():
            subject = str(row['subject'])
            video   = str(row['file']).zfill(3)
            label   = int(row['emotion'])

            frame_dir = os.path.join(frames_root, subject, video)
            if os.path.exists(frame_dir):
                samples.append((subject, video, label))
            else:
                print(f'[CK+] Warning: frames not found — {subject}/{video}')

        return samples

    def get_folds(self) -> List[Dict]:
        """10-fold identity-independent GroupKFold."""
        samples = self.get_all_samples()
        if not samples:
            raise RuntimeError('No CK+ samples found. Check frames directory and CSV.')

        subjects = np.array([s[0] for s in samples])
        labels   = np.array([s[2] for s in samples])

        nb_folds  = getattr(self.args, 'nb_cross_val', 10)
        gkf       = GroupKFold(n_splits=nb_folds)

        X_dummy   = np.zeros(len(samples))

        folds = []
        for fold_id, (train_idx, test_idx) in enumerate(
                gkf.split(X_dummy, labels, groups=subjects), start=1):
            folds.append({
                'fold_id': fold_id,
                'train':   [samples[i] for i in train_idx],
                'test':    [samples[i] for i in test_idx],
            })

        return folds