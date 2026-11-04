#!/usr/bin/env python3
# Author : Ahmad Ramadan

"""
preprocessing_exp_faciale.py  —  STEP 2 + STEP 3
  - STEP 2: Crops a 240x240 face region before landmark extraction
  - STEP 3: Uses 2D-FAN (face-alignment library) instead of MediaPipe
            → detects 68 landmarks (standard facial landmarks)

Install first:
    pip install face-alignment

Usage:
    python3 preprocessing_exp_faciale.py --face_landmarks_extraction

    # Resume an interrupted run (skips already-extracted videos):
    python3 preprocessing_exp_faciale.py --face_landmarks_extraction --resume
"""

import argparse
import numpy as np
from tqdm import tqdm
import cv2
import os
import pickle as pkl
from typing import *
import face_alignment   # pip install face-alignment
import torch

from utils import *
from datasets import get_adapter


# 2D-FAN landmark detector (loaded once, reused for all videos)

def build_fan_detector():
    """Load 2D-FAN detector. Uses GPU if available."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=device
    )
    print(f"2D-FAN detector loaded on {device}")
    return fa


# Resume helper

def is_already_extracted(saved_face_landmarks: str, subject_id: str,
                         video_id: str) -> bool:
    """
    Return True if a non-empty landmarks pickle already exists for this video.
    A pickle that contains only empty-dict frames (all failures) is still
    considered done — re-running won't improve it.
    """
    pkl_path = os.path.join(saved_face_landmarks, subject_id,
                            video_id + '_data.pkl')
    if not os.path.isfile(pkl_path):
        return False
    try:
        with open(pkl_path, 'rb') as f:
            data = pkl.load(f)
        # Make sure it's a non-empty dict (not a corrupt/empty file)
        return isinstance(data, dict) and len(data) > 0
    except Exception:
        # Corrupt pickle — treat as missing so it gets reprocessed
        return False


# Face crop helper (STEP 2)

def landmark_based_crop_240(image, fa, already_cropped: bool = False):
    """
    Crop a 240×240 face region defined by 2D-FAN landmarks themselves.

    Returns
    -------
    crop      : np.ndarray (240, 240, 3) BGR — face crop ready for downstream
    lms_240   : np.ndarray (68, 2) — 2D-FAN landmarks in 240×240 crop space
                (None if 2D-FAN failed on the full image)
    """
    TARGET = 240
    h, w   = image.shape[:2]

    if already_cropped:
        crop     = cv2.resize(image, (TARGET, TARGET))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        preds    = fa.get_landmarks(crop_rgb)
        lms_240  = preds[0] if (preds is not None and len(preds) > 0) else None
        return crop, lms_240

    # Run 2D-FAN on full image to find the face
    full_rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preds_full = fa.get_landmarks(full_rgb)

    if preds_full is None or len(preds_full) == 0:
        # 2D-FAN failed on full image — fallback to centre crop, no landmarks
        cx, cy = w // 2, h // 2
        side   = int(min(w, h) * 0.60)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        crop = cv2.resize(image[y1:y2, x1:x2], (TARGET, TARGET))
        return crop, None

    lms_full = preds_full[0]   # (68, 2) in original image space

    # Compute crop box from landmark extents
    x_min, y_min = lms_full.min(axis=0)
    x_max, y_max = lms_full.max(axis=0)
    fw           = x_max - x_min
    fh           = y_max - y_min
    margin       = int(max(fw, fh) * 0.25)   # 25% margin

    x1 = int(max(0, x_min - margin))
    y1 = int(max(0, y_min - margin))
    x2 = int(min(w, x_max + margin))
    y2 = int(min(h, y_max + margin))

    # Crop and resize to 240×240
    crop_raw       = image[y1:y2, x1:x2]
    crop_h, crop_w = crop_raw.shape[:2]
    crop           = cv2.resize(crop_raw, (TARGET, TARGET))

    # Re-run 2D-FAN on the 240×240 crop for clean landmarks
    crop_rgb   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    preds_crop = fa.get_landmarks(crop_rgb)
    if preds_crop is not None and len(preds_crop) > 0:
        lms_240 = preds_crop[0]
    else:
        # Fallback: rescale the original-image landmarks into 240×240 space
        sx = TARGET / crop_w
        sy = TARGET / crop_h
        lms_240 = np.zeros_like(lms_full)
        lms_240[:, 0] = (lms_full[:, 0] - x1) * sx
        lms_240[:, 1] = (lms_full[:, 1] - y1) * sy

    return crop, lms_240


def face_landmarks_extraction(subject_id, video_id, frames_dir, filenames,
                              saved_img, saved_face_landmarks, fa,
                              already_cropped: bool = False):
    """Updated to use landmark-based cropping."""
    valid_files = sorted([f for f in filenames
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    all_frame_landmarks = {}

    for frame_id, fname in tqdm(enumerate(valid_files), total=len(valid_files),
                                desc=f"Video {video_id}"):
        img_path = os.path.join(frames_dir, fname)
        image    = cv2.imread(img_path)
        if image is None:
            continue

        # Landmark-based crop + landmarks in one shot
        crop, lms = landmark_based_crop_240(image, fa,
                                            already_cropped=already_cropped)

        if lms is None:
            all_frame_landmarks[frame_id + 1] = {}   # mark failed
            continue

        # Store absolute coordinates in 240×240 crop
        coords = np.zeros(68 * 2, dtype=np.float32)
        for i, (lx, ly) in enumerate(lms):
            coords[i * 2]     = lx
            coords[i * 2 + 1] = ly
        all_frame_landmarks[frame_id + 1] = coords

        # Annotated image for visual debugging
        ann = crop.copy()
        for (lx, ly) in lms.astype(int):
            cv2.circle(ann, (lx, ly), 2, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(saved_img, f'frame_{frame_id}.jpg'), ann)

    make_dirs(os.path.join(saved_face_landmarks, subject_id))
    save_pickle(
        os.path.join(saved_face_landmarks, subject_id, video_id + '_data.pkl'),
        all_frame_landmarks
    )




def main():
    args = parser.parse_args()

    # Instantiate the right adapter for the chosen dataset
    adapter = get_adapter(args.nom_dataset)(args)

    chemin_landmark        = adapter.landmark_dir()
    chemin_label_out       = os.path.join(args.chemin_depot_global, 'labels')
    chemin_frames_annotees = os.path.join(
        args.chemin_depot_global, 'frames_annotees', args.nom_dataset)

    make_dirs(chemin_landmark)
    make_dirs(chemin_label_out)

    print(f'Dataset      : {args.nom_dataset}')
    print(f'Landmark dir : {chemin_landmark}')
    if args.resume:
        print('Resume mode  : ON  (already-extracted videos will be skipped)')

    # Discover all (subject, video, label) triples
    samples = adapter.get_all_samples()
    print(f'Samples found: {len(samples)}')

    resultats = {}

    # Load 2D-FAN detector once (only if actually extracting landmarks)
    fa = build_fan_detector() if args.face_landmarks_extraction else None

    skipped = 0
    processed = 0

    for subject_name, video_name, label in samples:

        frames_dir = adapter.frames_dir(subject_name, video_name)

        if subject_name not in resultats:
            resultats[subject_name] = {}
        resultats[subject_name][video_name] = label

        chemin_frame_annotee = os.path.join(
            chemin_frames_annotees, subject_name, video_name)
        make_dirs(chemin_frame_annotee)

        if args.face_landmarks_extraction:

            # Resume: skip videos whose pickle already exists
            if args.resume and is_already_extracted(
                    chemin_landmark, subject_name, video_name):
                print(f'[SKIP] {subject_name} / {video_name}  (already done)')
                skipped += 1
                continue

            print(f'Processing: {subject_name} / {video_name}')
            face_landmarks_extraction(
                subject_name, video_name,
                frames_dir,
                sorted(os.listdir(frames_dir)),
                chemin_frame_annotee,
                chemin_landmark,
                fa,
                already_cropped=args.already_cropped,
            )
            processed += 1

    if args.face_landmarks_extraction:
        print(f'\nDone. Processed: {processed}  |  Skipped (already done): {skipped}')

    # Save labels dict
    output_pkl = os.path.join(chemin_label_out, f'{args.nom_dataset}_labels.pkl')
    print(f'Saving labels to {output_pkl}')
    save_pickle(output_pkl, resultats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chemin_depot_global',       type=str,
                        default='/sandbox1/aramadan')
    parser.add_argument('--nom_dataset',               type=str, default='CK+',
                        help='Dataset name: CK+, Oulu, AFEW, ...')
    parser.add_argument('--face_landmarks_extraction', action='store_true')
    parser.add_argument('--already_cropped',           action='store_true',
                        help='Set for datasets whose frames are already '
                             'face-cropped (e.g. Oulu VL_Acropped). Skips '
                             'Haar detection and resizes directly to 240x240.')
    parser.add_argument('--resume',                    action='store_true',
                        help='Skip videos whose landmark pickle already exists '
                             '— use this to continue an interrupted run.')
    parser.add_argument('--mode_test',                 action='store_true')
    main()