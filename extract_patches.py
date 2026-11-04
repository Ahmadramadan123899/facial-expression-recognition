#!/usr/bin/env python3
# Author : Ahmad Ramadan
"""
extract_patches.py  —  STEP 2 update
Adds 240×240 face crop before patch extraction, matching the updated
preprocessing_exp_faciale.py.

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
    T.ToPILImage(),
    T.Resize((PATCH_SIZE, PATCH_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


# Must match preprocessing_exp_faciale.py's crop exactly (STEP 2)
def landmark_based_crop_240(image, fa, already_cropped: bool = False):
    """
    Same as in preprocessing_exp_faciale.py — must stay identical for
    geometry/appearance alignment.
    """
    TARGET = 240
    h, w   = image.shape[:2]

    if already_cropped:
        crop     = cv2.resize(image, (TARGET, TARGET))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        preds    = fa.get_landmarks(crop_rgb)
        lms_240  = preds[0] if (preds is not None and len(preds) > 0) else None
        return crop, lms_240

    full_rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preds_full = fa.get_landmarks(full_rgb)

    if preds_full is None or len(preds_full) == 0:
        cx, cy = w // 2, h // 2
        side   = int(min(w, h) * 0.60)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        crop = cv2.resize(image[y1:y2, x1:x2], (TARGET, TARGET))
        return crop, None

    lms_full = preds_full[0]
    x_min, y_min = lms_full.min(axis=0)
    x_max, y_max = lms_full.max(axis=0)
    fw, fh       = x_max - x_min, y_max - y_min
    margin       = int(max(fw, fh) * 0.25)

    x1 = int(max(0, x_min - margin))
    y1 = int(max(0, y_min - margin))
    x2 = int(min(w, x_max + margin))
    y2 = int(min(h, y_max + margin))

    crop_raw = image[y1:y2, x1:x2]
    crop_h, crop_w = crop_raw.shape[:2]
    crop     = cv2.resize(crop_raw, (TARGET, TARGET))

    crop_rgb   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    preds_crop = fa.get_landmarks(crop_rgb)
    if preds_crop is not None and len(preds_crop) > 0:
        lms_240 = preds_crop[0]
    else:
        sx, sy   = TARGET / crop_w, TARGET / crop_h
        lms_240  = np.zeros_like(lms_full)
        lms_240[:, 0] = (lms_full[:, 0] - x1) * sx
        lms_240[:, 1] = (lms_full[:, 1] - y1) * sy

    return crop, lms_240

# Encoders

_RESNET34_DEFAULT_CKPT = (
    '/sandbox1/aramadan/'
    'resnet34-epoch.12-val_loss.0.494-val_acc.0.846-val_f1.0.843.ckpt'
)
_RESNET18_DEFAULT_CKPT = (
    '/sandbox1/aramadan/Resnet18_FER+_pytorch.pth'
)


def build_encoder(device, backbone: str = 'resnet34', ckpt_path: str = None):
    """ResNet-34 or ResNet-18 pretrained on FER+, frozen feature extractor."""
    if backbone == 'resnet34':
        net = models.resnet34(pretrained=False)
        net.fc = nn.Linear(512, 7)
        path  = ckpt_path or _RESNET34_DEFAULT_CKPT
        cp    = torch.load(path, map_location='cpu', weights_only=False)
        state = cp.get('state_dict', cp)
        state = {k.replace('model.model.', ''): v for k, v in state.items()}
        net.load_state_dict(state, strict=True)

    elif backbone == 'resnet18':
        net  = models.resnet18(pretrained=False)
        path = ckpt_path or _RESNET18_DEFAULT_CKPT
        cp   = torch.load(path, map_location='cpu', weights_only=False)
        state = cp.get('state_dict', cp)
        # Strip DataParallel prefix (Emotion-FAN checkpoint is multi-GPU)
        cleaned = {}
        for k, v in state.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            k = (k.replace('model.model.', '')
                  .replace('model.', '')
                  .replace('backbone.', ''))
            cleaned[k] = v
        cleaned = {k: v for k, v in cleaned.items() if not k.startswith('fc.')}
        missing, _ = net.load_state_dict(cleaned, strict=False)
        backbone_missing = [k for k in missing if not k.startswith('fc.')]
        if backbone_missing:
            print(f'[Warning] Missing backbone keys: {backbone_missing[:3]}')

    else:
        raise ValueError(f'Unknown backbone: {backbone!r}. Choose resnet34 or resnet18.')

    net.fc = nn.Identity()
    net    = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False
    print(f'[encoder] {backbone} loaded from {path} (frozen, fc→Identity)')
    return net


def build_fan(device):
    """2D-FAN landmark detector."""
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device='cuda' if device.type == 'cuda' else 'cpu'
    )
    return fa


# Per-video patch extraction

def extract_video(frames_dir, filenames, encoder, fa, device,
                  already_cropped: bool = False):
    """
    Per frame: crop to 240×240 (or resize if already_cropped), run 2D-FAN
    for landmarks, crop 49×49 patches, encode with ResNet-34 FER+.

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

        # Landmark-based crop (same logic as preprocessing)
        crop, lms = landmark_based_crop_240(img, fa, already_cropped=already_cropped)
        if lms is None:
            continue
        gray_240 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Extract 49×49 patches
        # Use actual image dimensions (always 240×240 after crop_face_240,
        # but we read them dynamically so this is robust to any edge case).
        # 2D-FAN can return landmarks slightly outside [0, H/W] on some
        # images — clamp landmark coords first, then compute patch bounds
        # from x1/y1 (not from cx/cy) so the slice is always ≤ PATCH_SIZE.
        H, W = gray_240.shape[:2]
        crops = []
        for (lx, ly) in lms:
            cx = int(np.clip(round(lx), 0, W - 1))
            cy = int(np.clip(round(ly), 0, H - 1))

            x1 = max(0, cx - HALF)
            y1 = max(0, cy - HALF)
            x2 = min(W, x1 + PATCH_SIZE)   # derived from x1, not cx
            y2 = min(H, y1 + PATCH_SIZE)   # derived from y1, not cy

            patch = gray_240[y1:y2, x1:x2]

            # Pad to PATCH_SIZE × PATCH_SIZE if at image border
            ph, pw = patch.shape[:2]
            if ph != PATCH_SIZE or pw != PATCH_SIZE:
                pad = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                pad[:ph, :pw] = patch          # always safe: ph≤PS, pw≤PS
                patch = pad

            crops.append(patch)

        # ResNet-34 FER+ forward
        feats = []
        for i in range(0, len(crops), BATCH_SIZE):
            batch = torch.stack(
                [_transform(c) for c in crops[i:i + BATCH_SIZE]]
            ).to(device)
            with torch.no_grad():
                f = encoder(batch).cpu().numpy()
            feats.append(f.astype(np.float16))

        result[frame_id] = np.concatenate(feats, axis=0)   # (68, 512) float16

    return result


def main():
    args   = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    encoder = build_encoder(device, backbone=args.backbone,
                            ckpt_path=args.encoder_ckpt or None)

    fa = build_fan(device)
    print("2D-FAN detector loaded.")

    dir_frames   = os.path.join(args.chemin_depot_global, 'frames',   args.nom_dataset)
    dir_landmark = os.path.join(args.chemin_depot_global, 'landmark', args.nom_dataset)
    patches_root = args.out_dir or os.path.join(args.chemin_depot_global, 'patches')
    dir_patches  = os.path.join(patches_root, args.nom_dataset)
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
                                  encoder, fa, device,
                                  already_cropped=args.already_cropped)

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
    parser.add_argument('--already_cropped',     action='store_true',
                        help='Set for datasets whose frames are already '
                             'face-cropped (e.g. Oulu VL_Acropped).')
    parser.add_argument('--backbone',     default='resnet34',
                        choices=['resnet34', 'resnet18'],
                        help='Backbone architecture (default: resnet34).')
    parser.add_argument('--encoder_ckpt', default='',
                        help='Path to backbone checkpoint. '
                             'Defaults to the built-in path for each backbone.')
    parser.add_argument('--out_dir',      default='',
                        help='Root output directory for patches '
                             '(default: {chemin_depot_global}/patches). '
                             'Use e.g. /sandbox1/aramadan/patches_resnet18 '
                             'to keep ResNet-34 patches intact.')
    main()