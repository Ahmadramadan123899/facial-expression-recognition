#!/usr/bin/env python3
"""
datasets/caer.py
CAER (Context-Aware Emotion Recognition) adapter — fixed split protocol.

Paper protocol (Section 4.1):
    "The database has been divided into a training split, a validation
     split, and a testing split with 70%, 10%, and 20% samples, respectively.
     We follow the same setting in [3]…"

We train on the official `train` split and evaluate on the official `test`
split (which has labels — unlike AFEW). The `validation` split is loaded
into samples but excluded from the train/test fold, so it doesn't affect
training nor evaluation. If you want to use it (e.g. for tuning), you can
add it to `train` in get_folds() — but for paper-comparable numbers, leave
it out.

Frames layout produced by organize_caer.py:
    <base>/frames/CAER/<split>_<videoID>/<emotion>/frame_NNNN.jpg
        <split> ∈ {train, val, test}
"""

import os
from typing import List, Tuple, Dict

from .base import DatasetAdapter


class CAERAdapter(DatasetAdapter):

    num_classes    = 7
    # Paper's display order (matches AFEW): An, Di, Fe, Ha, Sa, Su, Ne
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy',
                      'Sad', 'Surprise', 'Neutral']
    dataset_name   = 'CAER'
    cv_strategy    = 'fixed_split'

    LABEL_MAP = {
        'Anger':    0,
        'Disgust':  1,
        'Fear':     2,
        'Happy':    3,
        'Sad':      4,
        'Surprise': 5,
        'Neutral':  6,
    }

    # Interface

    def get_all_samples(self) -> List[Tuple[str, str, int]]:
        """
        Walks frames/CAER/<split>_<videoID>/<emotion>/ and returns every
        (subject, emotion, label) triple that has at least one frame.
        """
        frames_root = os.path.join(self.args.chemin_depot_global,
                                   'frames', self.dataset_name)
        if not os.path.exists(frames_root):
            raise FileNotFoundError(
                f'CAER frames not found at {frames_root}. '
                f'Run organize_caer.py first.')

        samples = []
        n_train = n_val = n_test = n_other = 0

        for subject in sorted(os.listdir(frames_root)):
            subj_dir = os.path.join(frames_root, subject)
            if not os.path.isdir(subj_dir):
                continue

            if   subject.startswith('train_'): n_train += 1
            elif subject.startswith('val_'):   n_val   += 1
            elif subject.startswith('test_'):  n_test  += 1
            else:
                n_other += 1
                print(f'[CAER] Warning: subject "{subject}" has no '
                      f'train_/val_/test_ prefix — skipped.')
                continue

            for emotion in sorted(os.listdir(subj_dir)):
                if emotion not in self.LABEL_MAP:
                    continue
                emo_dir = os.path.join(subj_dir, emotion)
                if not os.path.isdir(emo_dir):
                    continue

                frames = [f for f in os.listdir(emo_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not frames:
                    print(f'[CAER] Warning: no frames in {subject}/{emotion}')
                    continue

                samples.append((subject, emotion, self.LABEL_MAP[emotion]))

        print(f'[CAER] Found {len(samples)} sequences '
              f'(train: {n_train}, val: {n_val}, test: {n_test}'
              + (f', skipped: {n_other}' if n_other else '') + ').')
        return samples

    def get_folds(self) -> List[Dict]:
        """
        Single 'fold' representing the official CAER train/test split.
        The 'validation' split is excluded from both train and test —
        if you want to include it in training, add a `or s[0].startswith('val_')`
        clause to the `train` filter below.
        """
        samples = self.get_all_samples()
        if not samples:
            raise RuntimeError('No CAER samples found.')

        train = [s for s in samples if s[0].startswith('train_')]
        test  = [s for s in samples if s[0].startswith('test_')]

        if not train:
            raise RuntimeError('CAER: no training samples found.')
        if not test:
            raise RuntimeError('CAER: no test samples found.')

        print(f'[CAER] Single-fold split → train: {len(train)} | test: {len(test)}')

        return [{
            'fold_id': 1,
            'train':   train,
            'test':    test,
        }]