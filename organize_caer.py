#!/usr/bin/env python3
# Author : Ahmad Ramadan
"""
organize_caer.py  —  STEP 0 for the CAER dataset.

Walks the CAER distribution:
    <root>/{train,validation,test}/<emotion>/NNNN.avi

…and extracts JPEG frames into the layout the rest of the pipeline expects:
    <out_root>/<split>_<videoID>/<emotion>/frame_NNNN.jpg

Where:
    <split>     ∈ {train, val, test}   (note: 'validation' shortens to 'val')
    <videoID>   = AVI basename without extension (e.g. 0001)
    <emotion>   ∈ {Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise}

The split prefix on the subject name keeps train/val/test identifiable in
the labels pkl, so the adapter recovers the official split without any
extra manifest. Same convention as organize_afew.py.

Heads-up: CAER is large (~13k videos). Frame extraction at downsample=32
plus subsequent 2D-FAN landmarking and ResNet-34 patch extraction will
take many hours. Consider running splits separately, or a single emotion
at a time, if you want to checkpoint progress.

Usage:
    python3 organize_caer.py \
        --caer_root /sandbox1/aramadan/CAER \
        --out_root  /sandbox1/aramadan/frames/CAER \
        --downsample 32

Restrict to specific splits / emotions: (opptional )
    python3 organize_caer.py --splits train --emotions Happy Sad
"""

import argparse
import os
import sys

import cv2
from tqdm import tqdm


EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Disk folder name → prefix used downstream.
SPLITS = {
    'train':      'train',
    'validation': 'val',
    'test':       'test',
}


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
    parser.add_argument('--caer_root', type=str,
                        default='/sandbox1/aramadan/CAER',
                        help='Root containing train/, validation/, test/.')
    parser.add_argument('--out_root', type=str,
                        default='/sandbox1/aramadan/frames/CAER')
    parser.add_argument('--downsample', type=int, default=None,
                        help='Recommended: 32. Default: keep all frames.')
    parser.add_argument('--splits', nargs='+', default=list(SPLITS.keys()),
                        choices=list(SPLITS.keys()))
    parser.add_argument('--emotions', nargs='+', default=EMOTIONS,
                        choices=EMOTIONS,
                        help='Restrict to a subset of emotions (default: all 7).')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.caer_root):
        sys.exit(f'CAER root not found: {args.caer_root}')
    os.makedirs(args.out_root, exist_ok=True)

    n_done = n_skip = n_fail = 0
    for split in args.splits:
        prefix = SPLITS[split]
        for emotion in args.emotions:
            emo_dir = os.path.join(args.caer_root, split, emotion)
            if not os.path.isdir(emo_dir):
                print(f'[warn] missing folder, skipping: {emo_dir}')
                continue

            videos = sorted(f for f in os.listdir(emo_dir)
                            if f.lower().endswith('.avi'))
            for video_name in tqdm(videos,
                                   desc=f'{split}/{emotion}', leave=False):
                vid_id   = os.path.splitext(video_name)[0]
                subject  = f'{prefix}_{vid_id}'
                video_in = os.path.join(emo_dir, video_name)
                out_dir  = os.path.join(args.out_root, subject, emotion)

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