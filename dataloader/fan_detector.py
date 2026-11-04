#!/usr/bin/env python3
# Author : Ahmad Ramadan
"""
dataloader/fan_detector.py
Worker-safe, lazily-initialized 2D-FAN landmark detector wrapper.

FAN is NOT fork-safe: calling face_alignment.FaceAlignment() in the parent
process and then forking DataLoader workers causes CUDA context / threading
errors. This module initializes the singleton on the first detect() call,
which only ever happens inside worker processes (from LazyFERDataset.__getitem__).

Task 23: GPU inference is enabled by default (falls back to CPU if CUDA is
unavailable). Requires DataLoader to use multiprocessing_context='spawn'
when --on_the_fly_landmarks is True and num_workers > 0 (set automatically in
train_gnn_paper.py). Spawn avoids CUDA context inheritance from the parent.

Task 25: detect_batch() batches all per-sample frame calls into one GPU forward
pass for ~1.5-2× additional speedup over per-frame detect(). Used by
lazy_dataset.py when on_the_fly_landmarks=True.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# One FaceAlignment instance per OS process.  Workers are spawned (not forked)
# before this is set, so each gets its own None — no sharing.
_fan = None


def _get_fan():
    """Lazily construct 2D-FAN on first call in this worker process.

    Task 23: initializes on GPU when CUDA is available (requires spawn
    multiprocessing context to be safe). Falls back to CPU otherwise.
    """
    global _fan
    if _fan is None:
        try:
            import face_alignment  # pip install face-alignment
        except ImportError as exc:
            raise ImportError(
                "face-alignment is not installed. "
                "Install it with:  pip install face-alignment"
            ) from exc
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Verified to match preprocessing_exp_faciale.py / extract_patches.py exactly:
        #   LandmarksType.TWO_D ✓ | flip_input=False ✓ | face_detector: SFD default ✓
        #   BGR→RGB conversion before get_landmarks ✓ (done in detect() below)
        # Task 23: device is now 'cuda' by default (falls back to 'cpu' if unavailable).
        _fan = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=device,
        )
        logger.debug(
            "2D-FAN detector initialized on %s (pid %d)", device.upper(), os.getpid()
        )
    return _fan


def detect(image_bgr: np.ndarray):
    """
    Detect 68 facial landmarks in a BGR image using 2D-FAN.

    Uses the same model version as preprocessing_exp_faciale.py and
    extract_patches.py (face_alignment.FaceAlignment TWO_D, flip_input=False).

    Parameters
    ----------
    image_bgr : np.ndarray  (H, W, 3) uint8, BGR channel order

    Returns
    -------
    np.ndarray (68, 2) float32
        Landmark (x, y) coordinates in the input image's pixel space.
    None
        If no face was detected or FAN raised an internal exception.
    """
    import cv2

    fa = _get_fan()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        preds = fa.get_landmarks(image_rgb)
    except Exception as exc:
        logger.debug("FAN raised an exception: %s", exc)
        return None
    if preds is None or len(preds) == 0:
        return None
    return preds[0].astype(np.float32)  # (68, 2)


def detect_batch(images_bgr: list) -> list:
    """
    Detect 68 facial landmarks in a batch of BGR images using one FAN call.

    Task 25: batches all per-sample frame calls into a single GPU forward
    pass for ~1.5-2× additional speedup over calling detect() per frame.

    Parameters
    ----------
    images_bgr : list of np.ndarray  each (H, W, 3) uint8, BGR channel order

    Returns
    -------
    list of np.ndarray (68, 2) float32 | None
        One entry per input image. None when no face was detected in that frame.
    """
    import cv2

    if not images_bgr:
        return []

    fa = _get_fan()

    batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_bgr]

    try:
        # face-alignment >= 1.3: get_landmarks_from_batch accepts a list of RGB
        # numpy arrays (H, W, 3). Returns a list of length N where each element
        # is either None (no face) or a list/array of (68, 2) landmark arrays
        # (one per detected face). We always take the first (highest-score) face.
        results = fa.get_landmarks_from_batch(batch_rgb)
    except Exception as exc:
        logger.debug(
            "FAN batch failed (%s); falling back to per-frame detect for this sample", exc)
        return [detect(img) for img in images_bgr]

    if results is None:
        return [None] * len(images_bgr)

    out = []
    for r in results:
        if r is None or (isinstance(r, (list, tuple)) and len(r) == 0):
            out.append(None)
        elif isinstance(r, list):
            # list of (68, 2) arrays — take first (highest-score) face
            out.append(np.asarray(r[0]).astype(np.float32))
        elif isinstance(r, np.ndarray):
            if r.ndim == 3:
                # (K, 68, 2) — take first face
                out.append(r[0].astype(np.float32))
            else:
                # (68, 2) directly
                out.append(r.astype(np.float32))
        else:
            out.append(None)

    # Defensive: pad if results is shorter than input (shouldn't happen)
    while len(out) < len(images_bgr):
        out.append(None)

    return out
