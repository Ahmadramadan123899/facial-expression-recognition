#!/usr/bin/env python3
"""
datasets/ferv39k.py
FERV39k adapter — official All_scenes train/test protocol.

FERV39k provides three evaluation setups under setups/:
  - 22_scenes : per-scenario train/test (used for ablation in the paper)
  - 4_scenes  : per-major-category train/test
  - All_scenes: pooled train/test across all scenarios (HEADLINE numbers)

The STAR-Former paper reports its 42.12% UAR / 51.84% WAR on the All_scenes
setup, so this adapter uses train_All.csv / test_All.csv.

Data layout assumed on this server:
  Frames: /data/fox-data/datasets/emotions/FERV39K/samples_face_cropped/
          {Scenario}/{Emotion}/{ClipID}/{NN}.jpg
          → already face-cropped (square frames). Frame names are 2-digit
          zero-padded ('00.jpg', '01.jpg', ...).
  Setups: /data/fox-data/datasets/emotions/FERV39K/setups/All_scenes/
          {train,test}_All.csv
          Format: one row per clip, slash-separated, NO comma:
              Scenario/Emotion/ClipID/Emotion
          e.g.   "Argue/Happy/0107/Happy"
          (yes, Emotion repeats; the 4th field IS the label.)

`subject` = Scenario (folder under samples_face_cropped/)
`video`   = "{Emotion}_{ClipID}"  (composite key so we can recover the
            path: {scenario}/{emotion}/{clip_id}/)

frames_dir / landmark_pkl / patches_pkl are overridden to point to the
actual data locations.

Use with --already_cropped (frames are already face crops, so the
on-the-fly patch extractor resizes to 240×240 directly).
"""

import csv
import os
from typing import List, Tuple, Dict

from .base import DatasetAdapter


FERV39K_FRAMES_ROOT = ('/data/fox-data/datasets/emotions/'
                       'FERV39K/samples_face_cropped')
FERV39K_SETUPS_ROOT = ('/data/fox-data/datasets/emotions/'
                       'FERV39K/setups/All_scenes')


class FERV39kAdapter(DatasetAdapter):

    num_classes    = 7
    # Order chosen to match DFEW / STAR-Former Table 2b reporting:
    #   Happy, Sad, Neutral, Angry, Surprise, Disgust, Fear
    emotion_labels = ['Happy', 'Sad', 'Neutral', 'Angry',
                      'Surprise', 'Disgust', 'Fear']
    dataset_name   = 'FERV39k'
    cv_strategy    = 'fixed_split'   # single official train/test split

    LABEL_MAP = {
        'Happy':    0,
        'Sad':      1,
        'Neutral':  2,
        'Angry':    3,
        'Surprise': 4,
        'Disgust':  5,
        'Fear':     6,
    }

    # Path overrides

    def frames_dir(self, subject: str, video: str) -> str:
        # subject = Scenario (e.g. "Argue")
        # video   = "{Emotion}_{ClipID}" (e.g. "Happy_0107")
        emotion, clip_id = video.split('_', 1)
        return os.path.join(FERV39K_FRAMES_ROOT, subject, emotion, clip_id)

    def landmark_pkl(self, subject: str, video: str) -> str:
        return (f'{self.base_dir}/landmark/{self.dataset_name}/'
                f'{subject}/{video}_data.pkl')

    def patches_pkl(self, subject: str, video: str) -> str:
        return (f'{self.base_dir}/patches/{self.dataset_name}/'
                f'{subject}/{video}_patches.pkl')

    # CSV parsing

    def _parse_csv(self, csv_path: str) -> List[Tuple[str, str, int]]:
        """
        Parse one FERV39k setup CSV. Each row is a single string of the form
            "Scenario/Emotion/ClipID/Emotion"
        Returns list of (subject, video, label_0idx) where subject=Scenario
        and video="Emotion_ClipID".
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f'FERV39k setup CSV not found at {csv_path}.')

        out = []
        skipped_lines  = 0
        skipped_labels = 0
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # CSV reader handles quoting, but FERV39k uses plain text.
                # Each row is one field that itself contains slashes.
                parts = line.split('/')
                if len(parts) != 4:
                    skipped_lines += 1
                    continue
                scenario, emotion1, clip_id, emotion2 = parts
                # Defensive: use the 4th field as the label (matches docs),
                # warn if it disagrees with the 2nd.
                if emotion1 != emotion2:
                    print(f'[FERV39k] Inconsistent label in row '
                          f'"{line}": {emotion1} vs {emotion2} — using {emotion2}')
                label_name = emotion2
                if label_name not in self.LABEL_MAP:
                    skipped_labels += 1
                    continue
                label = self.LABEL_MAP[label_name]
                video = f'{emotion1}_{clip_id}'
                out.append((scenario, video, label))

        if skipped_lines:
            print(f'[FERV39k] {skipped_lines} malformed rows in {csv_path}')
        if skipped_labels:
            print(f'[FERV39k] {skipped_labels} unknown labels in {csv_path}')
        return out

    # Interface

    def get_all_samples(self) -> List[Tuple[str, str, int]]:
        train_csv = os.path.join(FERV39K_SETUPS_ROOT, 'train_All.csv')
        test_csv  = os.path.join(FERV39K_SETUPS_ROOT, 'test_All.csv')

        train = self._parse_csv(train_csv)
        test  = self._parse_csv(test_csv)
        all_samples = train + test

        # Filter to samples whose frames folder exists
        valid = []
        n_missing = 0
        for subj, vid, lab in all_samples:
            if os.path.isdir(self.frames_dir(subj, vid)):
                valid.append((subj, vid, lab))
            else:
                n_missing += 1
        if n_missing:
            print(f'[FERV39k] {n_missing} samples have no frames folder '
                  f'under {FERV39K_FRAMES_ROOT} (skipped).')

        print(f'[FERV39k] Total: {len(valid)} clips '
              f'(train: {len(train)}, test: {len(test)})')
        return valid

    def get_folds(self) -> List[Dict]:
        """Single train/test fold from train_All.csv / test_All.csv."""
        train_csv = os.path.join(FERV39K_SETUPS_ROOT, 'train_All.csv')
        test_csv  = os.path.join(FERV39K_SETUPS_ROOT, 'test_All.csv')

        train_all = self._parse_csv(train_csv)
        test_all  = self._parse_csv(test_csv)

        # Filter both lists to samples whose frames folder exists
        def _filter(samples, name):
            kept = []
            n_missing = 0
            for s in samples:
                if os.path.isdir(self.frames_dir(s[0], s[1])):
                    kept.append(s)
                else:
                    n_missing += 1
            if n_missing:
                print(f'[FERV39k] {n_missing} {name} samples have no '
                      f'frames folder (skipped).')
            return kept

        train = _filter(train_all, 'train')
        test  = _filter(test_all,  'test')

        if not train or not test:
            raise RuntimeError('FERV39k: empty train or test fold.')

        print(f'[FERV39k] Official All_scenes split → '
              f'train: {len(train)} | test: {len(test)}')

        return [{
            'fold_id': 1,
            'train':   train,
            'test':    test,
        }]
    