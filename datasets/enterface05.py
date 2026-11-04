#!/usr/bin/env python3
"""
datasets/enterface05.py
eNTERFACE05 adapter — 44 subjects × 6 emotions × 5 sentences = ~1320 sequences.

Paper protocol (Section 4.1):
    "we employ the same 5-fold identity-independent cross-validation strategy
     as theirs to perform the evaluation on eNTERFACE05."

Frames layout produced by organize_enterface05.py:
    <base>/frames/eNTERFACE05/subject_NN/<emotion>_<sentence>/frame_NNNN.jpg
"""

import os
from typing import List, Tuple, Dict

import numpy as np
from sklearn.model_selection import GroupKFold
from .base import DatasetAdapter


class ENTERFACE05Adapter(DatasetAdapter):

    num_classes    = 6
    emotion_labels = ['Anger', 'Disgust', 'Fear',
                      'Happiness', 'Sadness', 'Surprise']
    dataset_name   = 'eNTERFACE05'
    cv_strategy    = 'group_kfold'

    LABEL_MAP = {
        'anger':     0,
        'disgust':   1,
        'fear':      2,
        'happiness': 3,
        'sadness':   4,
        'surprise':  5,
    }

    # Interface

    def get_all_samples(self) -> List[Tuple[str, str, int]]:
        """
        Walks frames/eNTERFACE05/<subject>/<video_id>/ and returns every
        (subject, video_id, label) triple that has at least one frame.

        video_id format: "<emotion>_<sentence>" (e.g. "anger_1").
        """
        frames_root = os.path.join(self.args.chemin_depot_global,
                                   'frames', self.dataset_name)
        if not os.path.exists(frames_root):
            raise FileNotFoundError(
                f'eNTERFACE05 frames not found at {frames_root}. '
                f'Run organize_enterface05.py first.')

        samples = []
        for subject in sorted(os.listdir(frames_root)):
            subj_dir = os.path.join(frames_root, subject)
            if not os.path.isdir(subj_dir):
                continue

            for video_id in sorted(os.listdir(subj_dir)):
                vid_dir = os.path.join(subj_dir, video_id)
                if not os.path.isdir(vid_dir):
                    continue

                # Recover the emotion from the video_id prefix.
                # Split on the LAST underscore so emotion stays intact even
                # though some emotion names contain no underscore (all do
                # actually, but rsplit is safer than split).
                emotion = video_id.rsplit('_', 1)[0]
                if emotion not in self.LABEL_MAP:
                    continue

                frames = [f for f in os.listdir(vid_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not frames:
                    print(f'[eNTER] Warning: no frames in {subject}/{video_id}')
                    continue

                samples.append((subject, video_id, self.LABEL_MAP[emotion]))

        n_subjects = len({s[0] for s in samples})
        print(f'[eNTER] Found {len(samples)} sequences '
              f'across {n_subjects} subjects.')
        return samples

    def get_folds(self) -> List[Dict]:
        """5-fold identity-independent GroupKFold (paper protocol)."""
        samples = self.get_all_samples()
        if not samples:
            raise RuntimeError('No eNTERFACE05 samples found.')

        subjects = np.array([s[0] for s in samples])
        labels   = np.array([s[2] for s in samples])

        # Paper specifies 5; allow override via --nb_cross_val for ablations.
        nb_folds = getattr(self.args, 'nb_cross_val', 5)
        gkf      = GroupKFold(n_splits=nb_folds)
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