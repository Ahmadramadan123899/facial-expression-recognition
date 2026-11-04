#!/usr/bin/env python3
"""
datasets/oulu.py
Oulu-CASIA adapter — VIS strong illumination subset.
Labels are derived from emotion subfolder names (no CSV needed).
Uses 10-fold identity-independent GroupKFold on 80 subjects.

Expected frames structure (produced by organize_oulu.py):
    /sandbox1/aramadan/frames/Oulu/{subject}/{emotion}/FirstF_*.jpeg
    e.g. frames/Oulu/P001/Anger/FirstF_000.jpeg
"""

import os
from typing import List, Tuple, Dict

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold
from .base import DatasetAdapter


class OuluAdapter(DatasetAdapter):

    num_classes    = 6
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    dataset_name   = 'Oulu'
    cv_strategy    = 'group_kfold'

    # Maps emotion folder name → integer label (alphabetical order = paper order)
    LABEL_MAP = {
        'Anger':    0,
        'Disgust':  1,
        'Fear':     2,
        'Happiness':3,
        'Sadness':  4,
        'Surprise': 5,
    }

    # Interface

    def get_all_samples(self) -> List[Tuple[str, str, int]]:
        """
        Walks frames/Oulu/{subject}/{emotion}/ and returns every
        (subject, emotion, label) triple that has at least one frame.

        Note: for Oulu, video_id == emotion name (e.g. 'Anger').
        This is consistent — each subject has exactly one sequence per emotion.
        """
        frames_root = os.path.join(self.args.chemin_depot_global,
                                   'frames', self.dataset_name)

        if not os.path.exists(frames_root):
            raise FileNotFoundError(
                f'Oulu frames not found at {frames_root}. '
                f'Run organize_oulu.py first.')

        samples = []
        subjects = sorted(os.listdir(frames_root))

        for subject in subjects:
            subject_dir = os.path.join(frames_root, subject)
            if not os.path.isdir(subject_dir):
                continue

            for emotion in sorted(os.listdir(subject_dir)):
                if emotion not in self.LABEL_MAP:
                    continue

                emotion_dir = os.path.join(subject_dir, emotion)
                frames = [f for f in os.listdir(emotion_dir)
                          if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

                if len(frames) == 0:
                    print(f'[Oulu] Warning: no frames in {subject}/{emotion}')
                    continue

                label = self.LABEL_MAP[emotion]
                samples.append((subject, emotion, label))

        print(f'[Oulu] Found {len(samples)} sequences across {len(subjects)} subjects.')
        return samples

    def get_folds(self) -> List[Dict]:
        """10-fold identity-independent GroupKFold (same strategy as CK+)."""
        samples = self.get_all_samples()
        if not samples:
            raise RuntimeError('No Oulu samples found. Check frames directory.')

        subjects = np.array([s[0] for s in samples])
        labels   = np.array([s[2] for s in samples])

        nb_folds = getattr(self.args, 'nb_cross_val', 10)
        gkf      = GroupKFold(n_splits=nb_folds)
        #gkf = StratifiedGroupKFold(n_splits=nb_folds, shuffle=True, random_state=42)
        X_dummy  = np.zeros(len(samples))

        folds = []
        for fold_id, (train_idx, test_idx) in enumerate(
                gkf.split(X_dummy, labels, groups=subjects), start=1):
            folds.append({
                'fold_id': fold_id,
                'train':   [samples[i] for i in train_idx],
                'test':    [samples[i] for i in test_idx],
            })

        return folds