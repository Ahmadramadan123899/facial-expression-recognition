#!/usr/bin/env python3
"""
datasets/afew.py
AFEW (Acted Facial Emotion in the Wild) adapter — EmotiW protocol.

Uses the official Train/Val split (no cross-validation). Test labels are
not publicly released, so — like the paper (Section 4.1) and prior work —
we report results on the validation set, treating it as the test fold.

Expected frames structure (produced by organize_afew.py):
    <base>/frames/AFEW/{train|val}_<videoID>/<emotion>/frame_NNNN.jpg

Examples:
    frames/AFEW/train_000046280/Angry/frame_0001.jpg
    frames/AFEW/val_000149120/Angry/frame_0001.jpg

The (subject, video) pair stored in samples is therefore
(`<split>_<videoID>`, `<emotion>`) — same convention as Oulu, where each
"subject" has a single "video" matching its emotion folder. AFEW has no
real subject identities (clips come from movies), so videoID is used as
the unique subject key. The split prefix means the data-leak assertion
in train_gnn_paper.py is satisfied automatically.
"""

import os
from typing import List, Tuple, Dict

from .base import DatasetAdapter


class AFEWAdapter(DatasetAdapter):

    num_classes    = 7
    # Order follows the paper (An, Di, Fe, Ha, Sa, Su, Ne)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy',
                      'Sad', 'Surprise', 'Neutral']
    dataset_name   = 'AFEW'
    cv_strategy    = 'fixed_split'

    LABEL_MAP = {
        'Angry':    0,
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
        Walks frames/AFEW/<split>_<videoID>/<emotion>/ and returns every
        (subject, emotion, label) triple that has at least one frame.
        """
        frames_root = os.path.join(self.args.chemin_depot_global,
                                   'frames', self.dataset_name)

        if not os.path.exists(frames_root):
            raise FileNotFoundError(
                f'AFEW frames not found at {frames_root}. '
                f'Run organize_afew.py first.')

        samples  = []
        subjects = sorted(os.listdir(frames_root))

        n_train = n_val = n_other = 0
        for subject in subjects:
            subject_dir = os.path.join(frames_root, subject)
            if not os.path.isdir(subject_dir):
                continue

            if subject.startswith('train_'):
                n_split = 'train'; n_train += 1
            elif subject.startswith('val_'):
                n_split = 'val';   n_val   += 1
            else:
                n_other += 1
                print(f'[AFEW] Warning: subject "{subject}" has no train_/val_ '
                      f'prefix — skipped.')
                continue

            # AFEW: each "subject" (video clip) has exactly one emotion folder.
            for emotion in sorted(os.listdir(subject_dir)):
                if emotion not in self.LABEL_MAP:
                    continue

                emotion_dir = os.path.join(subject_dir, emotion)
                if not os.path.isdir(emotion_dir):
                    continue

                frames = [f for f in os.listdir(emotion_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                if len(frames) == 0:
                    print(f'[AFEW] Warning: no frames in {subject}/{emotion}')
                    continue

                label = self.LABEL_MAP[emotion]
                samples.append((subject, emotion, label))

        print(f'[AFEW] Found {len(samples)} sequences '
              f'(train: {n_train}, val: {n_val}'
              + (f', skipped: {n_other}' if n_other else '') + ').')
        return samples

    def get_folds(self) -> List[Dict]:
        """
        Single 'fold' representing the official AFEW Train/Val split.
        Not cross-validation — paper protocol uses the fixed EmotiW split.

        The training script's loop will execute exactly once: training on
        Train and evaluating on Val (treated here as the 'test' fold).
        """
        samples = self.get_all_samples()
        if not samples:
            raise RuntimeError('No AFEW samples found.')

        train = [s for s in samples if s[0].startswith('train_')]
        test  = [s for s in samples if s[0].startswith('val_')]

        if not train:
            raise RuntimeError(
                'AFEW: no training samples found. Did organize_afew.py run '
                'on the Train split?')
        if not test:
            raise RuntimeError(
                'AFEW: no validation samples found. Did organize_afew.py run '
                'on the Val split?')

        print(f'[AFEW] Single-fold split → train: {len(train)} | val: {len(test)}')

        return [{
            'fold_id': 1,
            'train':   train,
            'test':    test,   # AFEW val set, used as the test fold
        }]