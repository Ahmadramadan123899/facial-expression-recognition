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
"""

import argparse
import numpy as np
from tqdm import tqdm
import cv2
import os
import pickle as pkl
from typing import *
import pandas as pd

import face_alignment   # pip install face-alignment
import torch

from utils import *


# ─────────────────────────────────────────────────────────────────────────────
# 2D-FAN landmark detector (loaded once, reused for all videos)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Face crop helper (STEP 2)
# ─────────────────────────────────────────────────────────────────────────────

def crop_face_240(image):
    """
    Detects the face bounding box and crops a 240x240 region.
    Falls back to center crop if no face is detected.

    Returns
    -------
    crop        : np.ndarray (240, 240, 3) BGR
    (x1, y1)    : top-left corner of crop in original image coordinates
    scale_x     : crop_width  / 240
    scale_y     : crop_height / 240
    """
    TARGET = 240
    h, w   = image.shape[:2]

    # Use OpenCV Haar cascade (no extra install needed)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector     = cv2.CascadeClassifier(cascade_path)
    gray         = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces        = detector.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=3, minSize=(60, 60))

    if len(faces) > 0:
        # Pick the largest detected face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        # Add 20% margin
        margin = int(max(fw, fh) * 0.20)
        x1 = max(0,     x  - margin)
        y1 = max(0,     y  - margin)
        x2 = min(w - 1, x + fw + margin)
        y2 = min(h - 1, y + fh + margin)
    else:
        # Fallback: centre crop (60% of image)
        cx, cy = w // 2, h // 2
        side   = int(min(w, h) * 0.60)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w - 1, x1 + side)
        y2 = min(h - 1, y1 + side)

    crop_w  = x2 - x1
    crop_h  = y2 - y1
    crop    = image[y1:y2, x1:x2]
    crop    = cv2.resize(crop, (TARGET, TARGET))

    return crop, x1, y1, crop_w / TARGET, crop_h / TARGET


# ─────────────────────────────────────────────────────────────────────────────
# Landmark extraction with 2D-FAN (STEP 3)
# ─────────────────────────────────────────────────────────────────────────────

def face_landmarks_extraction(subject_id, video_id, frames_dir, filenames: List[str],
                               saved_img: str, saved_face_landmarks: str, fa):
    """
    For each frame:
      1. Crop 240×240 face region  (STEP 2)
      2. Run 2D-FAN to get 68 landmarks on the crop  (STEP 3)
      3. Normalise coordinates relative to nose tip and inter-eye distance(removed)

    Saves dict {frame_id (1-based): np.array(136,)}  — 68 landmarks x 2 coords
    Coordinates are in [0, 1] range normalised the same way as the paper.
    """

    valid_files = sorted([f for f in filenames
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    all_frame_landmarks = {}

    # We need the neutral-frame scale for R normalisation (paper Eq. 1)
    # Use the first frame to determine normalisation references
    nose_pos     = None   # index of nose tip in 68-lm set: landmark 30
    le_indices   = list(range(36, 42))   # left eye  (68-point standard)
    re_indices   = list(range(42, 48))   # right eye

    for frame_id, fname in tqdm(enumerate(valid_files), total=len(valid_files),
                                 desc=f"Video {video_id}"):
        img_path = os.path.join(frames_dir, fname)
        image    = cv2.imread(img_path)
        if image is None:
            continue

        # ── STEP 2: crop 240×240 ─────────────────────────────────────────
        crop, ox, oy, sx, sy = crop_face_240(image)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # ── STEP 3: 2D-FAN ───────────────────────────────────────────────
        preds = fa.get_landmarks(crop_rgb)
        if preds is None or len(preds) == 0:
            # No face detected — store empty
            all_frame_landmarks[frame_id + 1] = {}
            continue

        lms = preds[0]   # (68, 2)  pixel coords in 240×240 crop

        # ── Normalise relative to nose tip and inter-eye distance ─────────
        # Paper: R_i = (x_i - x_nose) / (2 * S)
        # S = distance between left and right eye centres
        nose_xy  = lms[30]                          # index 30 = nose tip
        le_centre = lms[le_indices].mean(axis=0)
        re_centre = lms[re_indices].mean(axis=0)
        scale     = np.linalg.norm(le_centre - re_centre)
        if scale < 1e-6:
            scale = 1.0

        coords = np.zeros(68 * 2, dtype=np.float32)
        for i, (lx, ly) in enumerate(lms):
            #normilization removed because im a lready normilizing in the model by calculating the R coordinate 
            #norm_x = (lx - nose_xy[0]) / (2.0 * scale)
            #norm_y = (ly - nose_xy[1]) / (2.0 * scale)
            #coords[i * 2]     = norm_x
            #coords[i * 2 + 1] = norm_y
            # save absolute coordinates in the 240x240 crop
            coords[i * 2]     = lx
            coords[i * 2 + 1] = ly

        all_frame_landmarks[frame_id + 1] = coords   # (136,)

        # ── Save annotated image ──────────────────────────────────────────
        ann = crop.copy()
        for (lx, ly) in lms.astype(int):
            cv2.circle(ann, (lx, ly), 2, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(saved_img, f'frame_{frame_id}.jpg'), ann)

    make_dirs(os.path.join(saved_face_landmarks, subject_id))
    save_pickle(
        os.path.join(saved_face_landmarks, subject_id, video_id + '_data.pkl'),
        all_frame_landmarks
    )


# ─────────────────────────────────────────────────────────────────────────────

def chemin(chemin_base='/sandbox1/aramadan', nom_dossier_dataset='CK+'):
    chemin_frames          = os.path.join(chemin_base, 'frames',          nom_dossier_dataset)
    chemin_csv_labels      = os.path.join(chemin_base, 'emotion_327.csv')
    chemin_frames_annotees = os.path.join(chemin_base, 'frames_annotees', nom_dossier_dataset)
    chemin_landmark        = os.path.join(chemin_base, 'landmark',        nom_dossier_dataset)
    chemin_label_output    = os.path.join(chemin_base, 'labels')
    return (chemin_frames, chemin_frames_annotees, chemin_landmark,
            chemin_csv_labels, chemin_label_output)


def main():
    args = parser.parse_args()

    (chemin_frames, chemin_frames_annotees, chemin_landmark,
     chemin_csv, chemin_label_out) = chemin(args.chemin_depot_global, args.nom_dataset)

    make_dirs(chemin_landmark)
    make_dirs(chemin_label_out)

    print(f"Loading labels from {chemin_csv}...")
    df = pd.read_csv(chemin_csv, sep=';')

    resultats = {}

    # Load 2D-FAN detector once
    fa = build_fan_detector() if args.face_landmarks_extraction else None

    for _, row in df.iterrows():
        subject_name = str(row['subject'])
        video_name   = str(row['file']).zfill(3)
        label        = int(row['emotion'])

        chemin_frame_sujet_video = os.path.join(chemin_frames, subject_name, video_name)

        if os.path.exists(chemin_frame_sujet_video):
            if subject_name not in resultats:
                resultats[subject_name] = {}
            resultats[subject_name][video_name] = label

            chemin_frame_annotee_sujet_video = os.path.join(
                chemin_frames_annotees, subject_name, video_name)
            make_dirs(chemin_frame_annotee_sujet_video)

            if args.face_landmarks_extraction:
                print(f"Processing: {subject_name} / {video_name}")
                face_landmarks_extraction(
                    subject_name, video_name,
                    chemin_frame_sujet_video,
                    sorted(os.listdir(chemin_frame_sujet_video)),
                    chemin_frame_annotee_sujet_video,
                    chemin_landmark,
                    fa
                )
        else:
            print(f"Warning: Folder not found for {subject_name}/{video_name}")

    output_pkl = os.path.join(chemin_label_out, 'labels.pkl')
    print(f"Saving combined labels to {output_pkl}")
    save_pickle(output_pkl, resultats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chemin_depot_global', type=str, default='/sandbox1/aramadan')
    parser.add_argument('--nom_dataset',         type=str, default='CK+')
    parser.add_argument('--face_landmarks_extraction', action='store_true')
    parser.add_argument('--mode_test',           action='store_true')
    main()
