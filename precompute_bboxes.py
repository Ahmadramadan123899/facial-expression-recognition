#!/usr/bin/env python3
# Author : Ahmad Ramadan
"""
precompute_bboxes.py
One-time precomputation of the face bounding box on every frame.

When we move ResNet-34 patch encoding inside the model (--unfreeze_resnet
in train_gnn_paper.py), the lazy dataset needs to crop 240×240 face regions
from the original frames on the fly. The crop logic in extract_patches.py /
preprocessing_exp_faciale.py runs 2D-FAN on the full frame to find the
face bbox; that's expensive per-step.

This script runs 2D-FAN once per frame and caches the resulting bbox as
`bboxes/{dataset}/{subject}/{video}_bbox.pkl` = {frame_filename: (x1,y1,x2,y2)}.

Compatible with the same DatasetAdapter interface as extract_patches.py.

Usage:
    python3 precompute_bboxes.py --nom_dataset AFEW
    python3 precompute_bboxes.py --nom_dataset CAER

For datasets that are already face-cropped (Oulu_Acropped), this script
does NOT need to be run — the lazy dataset uses --already_cropped to
resize directly to 240×240.

Resume:
    python3 precompute_bboxes.py --nom_dataset CAER --resume
"""

import argparse
import os
import pickle
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

import face_alignment
from datasets import get_adapter


def build_fan(device):
    try:
        ltype = face_alignment.LandmarksType.TWO_D
    except AttributeError:
        ltype = face_alignment.LandmarksType._2D
    return face_alignment.FaceAlignment(
        ltype,
        flip_input=False,
        device='cuda' if device.type == 'cuda' else 'cpu',
    )


def compute_bbox(image_bgr: np.ndarray, fa) -> Tuple[int, int, int, int]:
    """Reproduce extract_patches.py's landmark_based_crop_240 bbox computation."""
    h, w = image_bgr.shape[:2]
    full_rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    preds_full = fa.get_landmarks(full_rgb)

    if preds_full is None or len(preds_full) == 0:
        cx, cy = w // 2, h // 2
        side   = int(min(w, h) * 0.60)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        return (x1, y1, x2, y2)

    lms_full = preds_full[0]
    x_min, y_min = lms_full.min(axis=0)
    x_max, y_max = lms_full.max(axis=0)
    fw, fh       = x_max - x_min, y_max - y_min
    margin       = int(max(fw, fh) * 0.25)

    x1 = int(max(0, x_min - margin))
    y1 = int(max(0, y_min - margin))
    x2 = int(min(w, x_max + margin))
    y2 = int(min(h, y_max + margin))
    return (x1, y1, x2, y2)


def main():
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    adapter = get_adapter(args.nom_dataset)(args)

    fa = build_fan(device)
    print(f'2D-FAN loaded on {device}')

    dir_frames = os.path.join(args.chemin_depot_global, 'frames', args.nom_dataset)
    dir_bboxes = os.path.join(args.chemin_depot_global, 'bboxes', args.nom_dataset)
    os.makedirs(dir_bboxes, exist_ok=True)

    samples = adapter.get_all_samples()
    print(f'Samples to process: {len(samples)}')

    done_count = 0
    skip_count = 0
    for subject, video, _ in tqdm(samples, desc='videos'):
        subj_dir = os.path.join(dir_bboxes, subject)
        os.makedirs(subj_dir, exist_ok=True)
        out_path = os.path.join(subj_dir, video + '_bbox.pkl')

        if args.resume and os.path.exists(out_path):
            skip_count += 1
            continue

        frames_dir = adapter.frames_dir(subject, video)
        if not os.path.isdir(frames_dir):
            print(f'[skip] no frames dir: {frames_dir}')
            continue

        frame_files = sorted(f for f in os.listdir(frames_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png')))

        bbox_dict = {}
        for fname in frame_files:
            img = cv2.imread(os.path.join(frames_dir, fname))
            if img is None:
                continue
            bbox_dict[fname] = compute_bbox(img, fa)

        if bbox_dict:
            with open(out_path, 'wb') as fh:
                pickle.dump(bbox_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
            done_count += 1

    print(f'\nDone. New: {done_count}, Skipped (already cached): {skip_count}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chemin_depot_global', default='/sandbox1/aramadan')
    parser.add_argument('--nom_dataset',         required=True,
                        help='CK+ | Oulu | AFEW | CAER | eNTERFACE05 | AFEW+CAER')
    parser.add_argument('--resume',              action='store_true',
                        help='Skip videos whose bbox pickle already exists.')
    main()