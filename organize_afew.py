#!/usr/bin/env python3
# Author : Ahmad Ramadan
"""
organize_afew.py  —  STEP 0 for the AFEW dataset.

Extracts JPEG frames from every .avi clip in the AFEW5.0 distribution
into the layout expected by the rest of the pipeline:

    <out_root>/<split>_<videoID>/<emotion>/frame_NNNN.jpg

Where:
    <split>     ∈ {train, val}
    <videoID>   = AVI basename without extension (e.g. 000046280)
    <emotion>   ∈ {Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise}

This naming choice means the 'subject' is unique per video (AFEW has no
real subject identities — clips come from various movies / TV episodes)
and the train/val membership is derivable from the subject prefix, so
the existing data-leak assertion in train_gnn_paper.py is automatically
satisfied.

Layout produced is consistent with Oulu's frames/<subject>/<emotion>/
convention, so preprocessing_exp_faciale.py and extract_patches.py work
unchanged.

Usage:
    python3 organize_afew.py \
        --afew_root /data/fox-data/datasets/emotions/AFEW5.0 \
        --out_root  /sandbox1/aramadan/frames/AFEW

Add --downsample 32 to keep only ~32 uniformly spaced frames per video
(useful since AFEW clips have ~120 frames each — full extraction is
heavy on disk and 2D-FAN runtime in the next step).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
from tqdm import tqdm


EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
SPLITS   = {'Train': 'train', 'Val': 'val'}   # disk name → prefix used downstream


def _uniform_indices(n_total: int, n_keep: int):
    """Return n_keep indices uniformly spaced over [0, n_total)."""
    if n_keep is None or n_keep >= n_total:
        return list(range(n_total))
    if n_keep <= 0:
        return []
    # Same scheme as numpy.linspace then floor — keeps endpoints.
    step = (n_total - 1) / (n_keep - 1) if n_keep > 1 else 0
    return sorted({int(round(i * step)) for i in range(n_keep)})


def _count_frames_ffprobe(video_path: str) -> int:
    """Return true frame count via ffprobe (0 on failure)."""
    try:
        out = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-count_packets', '-show_entries', 'stream=nb_read_packets',
             '-of', 'csv=p=0', video_path],
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip())
    except Exception:
        return 0


def extract_frames(video_path: str, out_dir: str, downsample: int = None) -> int:
    """
    Decode all frames of `video_path` and write them as JPEGs into `out_dir`.

    Uses ffmpeg instead of OpenCV because many AFEW5.0 AVIs use a codec
    (XVID/MPEG-4 with B-frames) that causes OpenCV's sequential cap.read()
    to silently stop after only 2-13 frames even though the clip has 60-120.
    ffmpeg decodes every frame correctly for all codec variants.

    If `downsample` is set, only that many uniformly spaced frames are kept.
    Returns the number of frames actually written.
    """
    if not shutil.which('ffmpeg'):
        raise RuntimeError(
            'ffmpeg not found. Install it: sudo apt-get install ffmpeg')

    os.makedirs(out_dir, exist_ok=True)

    # Extract all frames to a temp dir via ffmpeg
    with tempfile.TemporaryDirectory() as tmp:
        ret = subprocess.run(
            ['ffmpeg', '-y', '-i', video_path,
             '-q:v', '2',           # JPEG quality ~95%
             os.path.join(tmp, 'frame_%04d.jpg')],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ret.returncode != 0:
            return 0

        all_frames = sorted(f for f in os.listdir(tmp) if f.endswith('.jpg'))
        n_total = len(all_frames)
        if n_total == 0:
            return 0

        keep_idx = set(_uniform_indices(n_total, downsample)) \
            if downsample is not None else set(range(n_total))

        n_written = 0
        for src_idx, fname in enumerate(all_frames):
            if src_idx not in keep_idx:
                continue
            dst = os.path.join(out_dir, f'frame_{n_written:04d}.jpg')
            shutil.copy2(os.path.join(tmp, fname), dst)
            n_written += 1

    return n_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--afew_root', type=str,
                        default='/data/fox-data/datasets/emotions/AFEW5.0',
                        help='Root of the AFEW5.0 distribution containing '
                             'Train/<emotion>/*.avi and Val/<emotion>/*.avi.')
    parser.add_argument('--out_root', type=str,
                        default='/sandbox1/aramadan/frames/AFEW',
                        help='Output root. Frames will be written to '
                             '<out_root>/<split>_<videoID>/<emotion>/.')
    parser.add_argument('--downsample', type=int, default=None,
                        help='If set, keep only this many uniformly-spaced '
                             'frames per video (recommended: 32). '
                             'Default: keep ALL frames (heavy on disk).')
    parser.add_argument('--splits', nargs='+', default=['Train', 'Val'],
                        choices=list(SPLITS.keys()))
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-extract videos that already have frames.')
    args = parser.parse_args()

    if not os.path.isdir(args.afew_root):
        sys.exit(f'AFEW root not found: {args.afew_root}')

    os.makedirs(args.out_root, exist_ok=True)

    n_done = n_skip = n_fail = 0

    for split in args.splits:
        prefix = SPLITS[split]
        for emotion in EMOTIONS:
            emo_dir = os.path.join(args.afew_root, split, emotion)
            if not os.path.isdir(emo_dir):
                print(f'[warn] missing folder, skipping: {emo_dir}')
                continue

            videos = sorted(f for f in os.listdir(emo_dir) if f.endswith('.avi'))
            for video_name in tqdm(videos, desc=f'{split}/{emotion}', leave=False):
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
                    print(f'  [fail] could not decode: {video_in}')
                    n_fail += 1

    print(f'\nDone. Extracted: {n_done} | Skipped (already done): {n_skip} '
          f'| Failed: {n_fail}')


if __name__ == '__main__':
    main()