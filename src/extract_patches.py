#!/usr/bin/env python3
# Author : Ahmad Ramadan
"""
Since we now use 2D-FAN (68 landmarks) instead of MediaPipe (478 landmarks),
we extract 68 patches instead of 478.

Output format unchanged:
    dict { frame_id (int, 1-based) :
           np.ndarray(shape=(68, 512), dtype=np.float16) }

Usage:
    python3 extract_patches.py
"""

import argparse
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

import face_alignment   # pip install face-alignment

PATCH_SIZE = 49
HALF       = PATCH_SIZE // 2
BATCH_SIZE = 256
NB_LM      = 68   # 2D-FAN gives 68 landmarks (was 478 for MediaPipe)

_transform = T.Compose([
    T.ToPILImage(),#Converts the raw NumPy array into a PIL Image format 
    #so torchvision transforms can be applied.
    T.Resize((PATCH_SIZE, PATCH_SIZE)),
    #Although the patches are already 49*49,
    #this ensures the dimensions are strictly enforced.
    T.Grayscale(num_output_channels=3),
    #ResNet-34 expects a 3-channel (RGB) input.
    #Since the patches are grayscale, this copies the single channel three times (R=G=B) 
    #to "fake" an RGB image while keeping the visual information gray.
    T.ToTensor(),
    #Converts the pixel values from the range $[0, 255]$ (integers) to $[0.0, 1.0]$
    #  (floating point).
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
    #standardizes the data using the ImageNet means and standard deviations. 
    #This centers the data around zero,
    #which helps the ResNet model process the features more effectively.
])


# ─────────────────────────────────────────────────────────────────────────────
# Face crop (STEP 2 — must match preprocessing_exp_faciale.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def crop_face_240(image):
    TARGET   = 240
    h, w     = image.shape[:2]
    cascade  = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces    = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=3, minSize=(60, 60))

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        margin = int(max(fw, fh) * 0.20)
        x1 = max(0,     x  - margin)
        y1 = max(0,     y  - margin)
        x2 = min(w - 1, x + fw + margin)
        y2 = min(h - 1, y + fh + margin)
    else:
        cx, cy = w // 2, h // 2
        side   = int(min(w, h) * 0.60)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w - 1, x1 + side)
        y2 = min(h - 1, y1 + side)

    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, (TARGET, TARGET))
    return crop


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────

def build_encoder(device):
    """ResNet-34 pretrained on FER+."""
    net    = models.resnet34(pretrained=False)
    net.fc = nn.Linear(512, 7)   # match FER+ checkpoint shape first

    ckpt_path = '/sandbox1/aramadan/resnet34-epoch.12-val_loss.0.494-val_acc.0.846-val_f1.0.843.ckpt'
    cp        = torch.load(ckpt_path, map_location='cpu')
    state     = cp['state_dict']
    state     = {k.replace('model.model.', ''): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)

    net.fc = nn.Identity()   # → 512-dim features
    net    = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def build_fan(device):
    """2D-FAN landmark detector."""
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device='cuda' if device.type == 'cuda' else 'cpu'
    )
    return fa


# ─────────────────────────────────────────────────────────────────────────────
# Per-video patch extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_video(frames_dir, filenames, encoder, fa, device):
    """
    For each frame:
      1. Crop 240×240 face region
      2. Run 2D-FAN to get 68 landmarks
      3. Crop 49×49 patch around each landmark
      4. Run ResNet-34 FER+ encoder

    Returns
    -------
    dict { frame_id (int, 1-based) : np.ndarray(68, 512) float16 }
    """
    valid  = sorted([f for f in filenames
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    result = {}

    for frame_id, fname in enumerate(valid, start=1):
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue

        # ── STEP 2: crop 240×240 ────────────────────────────────────────
        crop     = crop_face_240(img)
        gray_240 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # ── STEP 3: 2D-FAN landmarks ─────────────────────────────────────
        preds = fa.get_landmarks(crop_rgb)
        if preds is None or len(preds) == 0:
            continue
        lms = preds[0]   # (68, 2)

        # ── Extract 49×49 patches ────────────────────────────────────────
        crops = []
        for (lx, ly) in lms:
            cx = int(lx)
            cy = int(ly)
            x1 = max(0, cx - HALF)
            y1 = max(0, cy - HALF)
            x2 = min(240, cx + HALF + 1)
            y2 = min(240, cy + HALF + 1)

            patch = gray_240[y1:y2, x1:x2]
            if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                pad = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                pad[:patch.shape[0], :patch.shape[1]] = patch
                patch = pad
            crops.append(patch)

        # ── ResNet-34 FER+ forward ───────────────────────────────────────
        feats = []
        for i in range(0, len(crops), BATCH_SIZE):
            batch = torch.stack(
                [_transform(c) for c in crops[i:i + BATCH_SIZE]]
            ).to(device) #It stacks these individual 3*49*49 tensors on top of each other to create one 4D tensor.
            with torch.no_grad():
                f = encoder(batch).cpu().numpy()
            feats.append(f.astype(np.float16))

        result[frame_id] = np.concatenate(feats, axis=0)   # (68, 512) float16

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    encoder = build_encoder(device)
    print("ResNet-34 FER+ loaded (pretrained, frozen).")

    fa = build_fan(device)
    print("2D-FAN detector loaded.")

    dir_frames   = os.path.join(args.chemin_depot_global, 'frames',   args.nom_dataset)
    dir_landmark = os.path.join(args.chemin_depot_global, 'landmark', args.nom_dataset)
    dir_patches  = os.path.join(args.chemin_depot_global, 'patches',  args.nom_dataset)
    os.makedirs(dir_patches, exist_ok=True)

    subjects = sorted(os.listdir(dir_landmark))
    print(f"Subjects found : {len(subjects)}")
    print(f"Output dir     : {dir_patches}\n")

    done = 0
    for subj in subjects:
        subj_lm  = os.path.join(dir_landmark, subj)
        subj_fr  = os.path.join(dir_frames,   subj)
        subj_pat = os.path.join(dir_patches,  subj)
        os.makedirs(subj_pat, exist_ok=True)

        pkl_files = sorted(f for f in os.listdir(subj_lm)
                           if f.endswith('_data.pkl'))

        for pkl_f in tqdm(pkl_files, desc=subj, leave=False):
            vid_id   = pkl_f.replace('_data.pkl', '')
            out_path = os.path.join(subj_pat, vid_id + '_patches.pkl')

            if os.path.exists(out_path):
                done += 1
                continue

            fr_dir = os.path.join(subj_fr, vid_id)
            if not os.path.exists(fr_dir):
                print(f"  [skip] no frames dir: {fr_dir}")
                continue

            feats = extract_video(fr_dir, sorted(os.listdir(fr_dir)),
                                  encoder, fa, device)

            if not feats:
                print(f"  [skip] 2D-FAN failed: {subj}/{vid_id}")
                continue

            with open(out_path, 'wb') as fh:
                pickle.dump(feats, fh, protocol=pickle.HIGHEST_PROTOCOL)
            done += 1

    print(f"\nDone. Extracted patches for {done} videos.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chemin_depot_global', default='/sandbox1/aramadan')
    parser.add_argument('--nom_dataset',         default='CK+')
    main()
