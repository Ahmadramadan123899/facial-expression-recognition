#!/usr/bin/env python3
# Author : Ahmad Ramadan
"""
organize_enterface05.py  —  STEP 0 for eNTERFACE05.

Walks the eNTERFACE05 distribution:
    <root>/subject N/<emotion>/sentence M/sN_em_M.avi

…and extracts JPEG frames into the layout the rest of the pipeline expects:
    <out_root>/subject_NN/<emotion>_<sentence>/frame_NNNN.jpg

Where:
    subject_NN              — zero-padded subject id (e.g. subject_01, subject_44)
    <emotion>_<sentence>    — anger_1, happiness_3, …
                              (one of: anger, disgust, fear, happiness, sadness, surprise)

The (subject, video_id) convention matches CK+/Oulu — the adapter uses
subject_NN as the GroupKFold key for identity-independent CV.

Usage:
    python3 organize_enterface05.py \
        --enterface_root '/sandbox1/aramadan/eNTERFACE05/enterface database' \
        --out_root        /sandbox1/aramadan/frames/eNTERFACE05 \
        --downsample 32
"""

import argparse
import os
import re
import sys

import cv2
from tqdm import tqdm


EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']


def _uniform_indices(n_total: int, n_keep):
    if n_keep is None or n_keep >= n_total:
        return list(range(n_total))
    if n_keep <= 0:
        return []
    step = (n_total - 1) / (n_keep - 1) if n_keep > 1 else 0
    return sorted({int(round(i * step)) for i in range(n_keep)})


def extract_frames(video_path: str, out_dir: str, downsample=None) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total <= 0:
        n_total = None

    keep = (set(_uniform_indices(n_total, downsample))
            if (n_total is not None and downsample is not None) else None)

    os.makedirs(out_dir, exist_ok=True)

    frame_idx = out_idx = n_written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (keep is None) or (frame_idx in keep):
            cv2.imwrite(
                os.path.join(out_dir, f'frame_{out_idx:04d}.jpg'),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            n_written += 1
            out_idx   += 1
        frame_idx += 1
    cap.release()

    # Fallback when the AVI didn't report a frame count
    if keep is None and downsample is not None and n_written > downsample:
        files = sorted(os.listdir(out_dir))
        keep_idx = set(_uniform_indices(len(files), downsample))
        kept = []
        for i, f in enumerate(files):
            full = os.path.join(out_dir, f)
            if i in keep_idx:
                kept.append(full)
            else:
                os.remove(full)
        for new_idx, full in enumerate(sorted(kept)):
            target = os.path.join(out_dir, f'frame_{new_idx:04d}.jpg')
            if full != target:
                os.rename(full, target)
        n_written = len(kept)

    return n_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enterface_root', type=str,
                        default='/sandbox1/aramadan/eNTERFACE05/enterface database',
                        help='Root containing "subject N" sub-folders.')
    parser.add_argument('--out_root', type=str,
                        default='/sandbox1/aramadan/frames/eNTERFACE05')
    parser.add_argument('--downsample', type=int, default=None,
                        help='Recommended: 32. Default: keep all frames.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.enterface_root):
        sys.exit(f'eNTERFACE05 root not found: {args.enterface_root}')
    os.makedirs(args.out_root, exist_ok=True)

    subj_dirs = sorted(d for d in os.listdir(args.enterface_root)
                       if d.startswith('subject ')
                       and os.path.isdir(os.path.join(args.enterface_root, d)))
    print(f'Found {len(subj_dirs)} subject folders.')

    n_done = n_skip = n_fail = 0
    for subj_name in subj_dirs:
        m = re.match(r'subject\s+(\d+)', subj_name)
        if not m:
            continue
        subj_id   = f'subject_{int(m.group(1)):02d}'
        subj_path = os.path.join(args.enterface_root, subj_name)

        for emotion in EMOTIONS:
            emo_path = os.path.join(subj_path, emotion)
            if not os.path.isdir(emo_path):
                continue

            sent_dirs = sorted(d for d in os.listdir(emo_path)
                               if d.startswith('sentence ')
                               and os.path.isdir(os.path.join(emo_path, d)))

            for sent_name in tqdm(sent_dirs,
                                  desc=f'{subj_id}/{emotion}', leave=False):
                ms = re.match(r'sentence\s+(\d+)', sent_name)
                if not ms:
                    continue
                sent_num  = int(ms.group(1))
                sent_path = os.path.join(emo_path, sent_name)

                avis = [f for f in os.listdir(sent_path)
                        if f.lower().endswith('.avi')]
                if not avis:
                    continue
                video_in = os.path.join(sent_path, avis[0])

                video_id = f'{emotion}_{sent_num}'
                out_dir  = os.path.join(args.out_root, subj_id, video_id)

                if (not args.overwrite
                        and os.path.isdir(out_dir)
                        and any(f.endswith('.jpg') for f in os.listdir(out_dir))):
                    n_skip += 1
                    continue

                n = extract_frames(video_in, out_dir, downsample=args.downsample)
                if n > 0:
                    n_done += 1
                else:
                    print(f'  [fail] {video_in}')
                    n_fail += 1

    print(f'\nDone. Extracted: {n_done} | Skipped: {n_skip} | Failed: {n_fail}')


if __name__ == '__main__':
    main()