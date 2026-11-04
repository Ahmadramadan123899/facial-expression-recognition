#!/usr/bin/env python3
# Author : Ahmad Ramadan
"""
dataloader/patch_extractor.py
On-the-fly 49x49 patch extraction for end-to-end training with an unfrozen
ResNet-34 encoder.

Design
------
The 240x240 face crop logic in extract_patches.py / preprocessing_exp_faciale.py
relies on running 2D-FAN on the full input frame to find the face bounding
box. That's expensive — re-running 2D-FAN per frame per epoch would dominate
training time.

We pay this cost once via a side-cache: `bboxes/{dataset}/{subject}/{video}_bbox.pkl`
mapping {frame_filename: (x1, y1, x2, y2)}. The cache is built by
`precompute_bboxes.py` (separate script, run once per dataset before training).

At training time:
  1. Read the cached bbox for (subject, video, frame_filename).
  2. Load the raw frame from frames/{dataset}/{subject}/{video}/{frame_filename}.
  3. Crop + resize to 240x240 using the cached bbox.
  4. Use the cached 240x240 landmarks (from landmark/{dataset}/{subject}/{video}_data.pkl)
     to crop 49x49 patches around each kept landmark.
  5. Return uint8 (P, 3, 49, 49) BGR patches.

If the bbox cache is missing for a frame, we fall back to a centre-crop
(matches extract_patches.py's FAN-failure fallback) and emit a warning
once per video. The whole pipeline degrades gracefully but you should
run precompute_bboxes.py first for best quality.

For datasets where input frames are already face-cropped (Oulu_Acropped),
no bbox is needed — we just resize to 240×240.
"""

import os
import pickle
from typing import Optional, Tuple, Dict

import cv2
import numpy as np


PATCH_SIZE = 49
HALF       = PATCH_SIZE // 2
TARGET     = 240


# Per-worker memo for already-loaded bbox dicts. Keyed by absolute pickle path.
# Bounded to _BBOX_CACHE_MAX entries (LRU-style eviction when full) to prevent
# unbounded growth on large datasets like CAER (13k+ videos × ~1 KB/dict).
_BBOX_DICTS: Dict[str, dict] = {}
_BBOX_CACHE_MAX = 4000   # comfortably covers CK+ (327) and Oulu; evicts for CAER
# Warned-once set so we don't spam logs for missing bboxes.
_BBOX_WARNED = set()


def _load_bbox_dict(pkl_path: str) -> Optional[dict]:
    """Read and cache the per-video bbox pickle. Returns None if missing."""
    if pkl_path in _BBOX_DICTS:
        return _BBOX_DICTS[pkl_path]
    if not os.path.exists(pkl_path):
        _BBOX_DICTS[pkl_path] = None
        return None
    try:
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        d = None
    # Evict oldest entry when the cache is full (simple FIFO eviction).
    if len(_BBOX_DICTS) >= _BBOX_CACHE_MAX:
        oldest = next(iter(_BBOX_DICTS))
        del _BBOX_DICTS[oldest]
    _BBOX_DICTS[pkl_path] = d
    return d


def _centre_crop_bbox(h: int, w: int) -> Tuple[int, int, int, int]:
    """FAN-failure fallback — matches extract_patches.py's centre crop."""
    cx, cy = w // 2, h // 2
    side   = int(min(w, h) * 0.60)
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    return (x1, y1, x2, y2)


def load_and_crop_to_240(frame_path: str,
                         bbox_dict: Optional[dict],
                         frame_filename: str,
                         already_cropped: bool = False) -> Optional[np.ndarray]:
    """
    Load a frame and crop it to 240×240 using the cached bbox.

    Returns np.ndarray (240, 240, 3) uint8 BGR, or None if frame is unreadable.
    """
    img = cv2.imread(frame_path)
    if img is None:
        return None
    h, w = img.shape[:2]

    if already_cropped:
        return cv2.resize(img, (TARGET, TARGET))

    if bbox_dict is not None and frame_filename in bbox_dict:
        bbox = bbox_dict[frame_filename]
    else:
        vid_key = os.path.dirname(frame_path)
        if vid_key not in _BBOX_WARNED:
            _BBOX_WARNED.add(vid_key)
            print(f'[patch_extractor] No bbox cache for {vid_key} — '
                  f'falling back to centre crop. Run precompute_bboxes.py '
                  f'for cleaner training.')
        bbox = _centre_crop_bbox(h, w)

    x1, y1, x2, y2 = bbox
    return cv2.resize(img[y1:y2, x1:x2], (TARGET, TARGET))

def extract_patches_from_crop(crop_bgr: np.ndarray,
                              landmarks: np.ndarray,
                              kept_indices: list = None) -> np.ndarray:
    """
    Given a BGR face crop and the corresponding landmark array in the same
    coordinate space, extract 49×49 RGB patches around each kept landmark.

    Article §4.2: patches are cropped from the original (color) facial image.

    Returns np.ndarray (P, 3, 49, 49) uint8, RGB channel order.
    """
    if kept_indices is None:
        kept_indices = list(range(landmarks.shape[0]))

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    out = np.zeros((len(kept_indices), 3, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    for out_i, lm_idx in enumerate(kept_indices):
        lx, ly = landmarks[lm_idx]
        cx = int(np.clip(round(lx), 0, W - 1))
        cy = int(np.clip(round(ly), 0, H - 1))

        x1 = max(0, cx - HALF)
        y1 = max(0, cy - HALF)
        x2 = min(W, x1 + PATCH_SIZE)
        y2 = min(H, y1 + PATCH_SIZE)

        patch = rgb[y1:y2, x1:x2, :]  # (h, w, 3)

        ph, pw = patch.shape[:2]
        if ph != PATCH_SIZE or pw != PATCH_SIZE:
            pad = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
            pad[:ph, :pw, :] = patch
            patch = pad

        out[out_i] = patch.transpose(2, 0, 1)  # (3, 49, 49)
    return out

def bbox_pkl_path(base_dir: str, dataset_name: str,
                  subject: str, video: str) -> str:
    """Standard location for the per-video bbox cache."""
    return os.path.join(base_dir, 'bboxes', dataset_name,
                        subject, video + '_bbox.pkl')