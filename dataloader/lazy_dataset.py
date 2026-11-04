#! /usr/bin/python3
# Author : Ahmad Ramadan
"""
dataloader/lazy_dataset.py
Streaming PyG Dataset for the Graph-Tran pipeline.

The eager loader in build_dataset_paper.py loads EVERY landmark dict and
EVERY patches dict for the whole dataset into memory, then converts the
whole list to a giant tensor, then slices it per fold. That works for
CK+ / Oulu (~500 samples, ~50 MB total) but doesn't scale to CAER (~13k
samples, ~12 GB tensor) — Python's dict overhead and slice-copy memory
blows past 60 GB RAM.

This class is the opt-in alternative used when --lazy_loading is set on
train_gnn_paper.py. It:

  * Keeps only a list of (subject_id, video_id, label) tuples in memory
    (a few hundred KB even for 13k samples).
  * In __getitem__, opens the two pkl files for that sample, applies the
    same per-video preprocessing the eager pipeline does (frame filtering,
    temporal alignment, graph-joint filtering), and returns a single PyG
    Data object.
  * Workers in DataLoader(num_workers=N) parallelize the disk reads so the
    GPU never starves.

What this loses vs the eager pipeline:
  * The bulk class-balancing oversampler (`data_augmentation_equilibration_classe`)
    is replaced by PyG-level WeightedRandomSampler (configured by the trainer).
    Same end effect, no RAM cost.
  * Bulk coordinate / patch noise augmentation (already redundant with the
    per-batch augment_batch in trainer_paper.py) is dropped. trainer_paper
    handles all paper-spec augmentation on the collated batch.

What's preserved:
  * Frame-skip filtering (drops videos with <2 valid frames)
  * Temporal alignment to nb_frame_voulu frames (top-k sampler / spline
    extender — identical to traitements_video_par_video)
  * Graph topology selection (g_complet / g_topologique / ...)
  * Cross-component edges, bidirectional edges, edge replication per frame
  * Per-batch augmentation (coordinate noise, flip, crop, patch noise)
    via trainer_paper.augment_batch
"""

import logging
import os
import pickle
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# FAN fallback tracking (module-level, one counter per worker process)
# Mirrors the _BBOX_WARNED pattern in patch_extractor.py.
# Each worker process has its own copy of these (they fork before first use),
# so counts are per-process. Warnings appear inline in training logs.

_FAN_FALLBACK_WARNED: set = set()   # once-per-video warning gate
_FAN_FALLBACK_COUNT: list = [0]     # [total_fallback_frames]; list for mutation


def _log_fan_fallback(subject: str, video: str, frame_id: int) -> None:
    """Increment the per-process fallback counter; log once per video at DEBUG."""
    _FAN_FALLBACK_COUNT[0] += 1
    vid_key = f'{subject}/{video}'
    if vid_key not in _FAN_FALLBACK_WARNED:
        _FAN_FALLBACK_WARNED.add(vid_key)
        logger.debug(
            'FAN failed for %s frame %d — cached fallback. '
            'Worker fallback total so far: %d',
            vid_key, frame_id, _FAN_FALLBACK_COUNT[0],
        )


def get_fan_fallback_count() -> int:
    """Return the total FAN fallback frame count for this process.
    Call from the main process (num_workers=0) to get the epoch total,
    or from each worker's context for per-worker diagnostics."""
    return _FAN_FALLBACK_COUNT[0]


def reset_fan_fallback_count() -> None:
    """Reset the per-process fallback counter (call once per epoch)."""
    _FAN_FALLBACK_COUNT[0] = 0

from dataloader.build_dataset_paper import (
    augmentation_frames_video,
    convertir_dictionnaire_video_en_tenseur_video,
    create_graph,
    creer_graphe_complet_exp_faciale,
    creer_graphe_version1,
    creer_graphe_version2,
    creer_graphe_yeux_nez_bouche,
    reduction_frames_video,
    reduction_patches_video,
    augmentation_patches_video,
)
from dataloader.MovementExtender import MovementExtender
from dataloader.MovementSampler import MovementSampler
from dataloader.patch_extractor import (
    extract_patches_from_crop,
    load_and_crop_to_240,
    bbox_pkl_path,
    _load_bbox_dict,
)


# Per-graph-type joint subset (mirrors convertion_graphe's branches).
# We compute kept_indices ONCE from the first valid sample at scan time.

_GRAPH_BUILDERS = {
    'g_complet'    : creer_graphe_complet_exp_faciale,
    'g_bouche'     : creer_graphe_version1,
    'g_yeux_bouche': creer_graphe_version2,
    'g_topologique': creer_graphe_yeux_nez_bouche,
}


def _filter_landmarks_dict(lm_dict):
    """Drop frames where 2D-FAN failed (stored as empty dict by preprocessing)."""
    return {k: v for k, v in lm_dict.items()
            if not (isinstance(v, dict) and len(v) == 0)}


def _align_patches_to_landmarks(patches_dict, landmark_keys):
    """
    Keep only patches whose frame_id is in landmark_keys.
    Pad missing keys with zeros (same convention as extract_datas_video_labels).
    """
    if patches_dict is None:
        return None
    aligned = {k: v for k, v in patches_dict.items() if k in landmark_keys}
    for k in landmark_keys:
        if k not in aligned:
            aligned[k] = np.zeros((68, 512), dtype=np.float16)
    return aligned


# Streaming PyG-style Dataset

class LazyFERDataset(Dataset):
    """
    Streaming dataset for FER. Loads one (landmarks, patches) sample at a
    time from disk and returns a PyG `Data` object.

    Parameters
    ----------
    samples       : list of (subject_id, video_id, label_int) tuples
    adapter       : DatasetAdapter instance — used for landmark_pkl/patches_pkl paths
    graph_type    : one of 'g_complet', 'g_topologique', 'g_yeux_bouche', 'g_bouche'
    use_patches   : bool — load patches?
    nb_frames     : int — target frames per sequence (paper: 16)
    label_remap   : dict[int,int] | None — optional sparse → dense label mapping
    """

    def __init__(self,
                 samples: List[Tuple[str, str, int]],
                 adapter,
                 graph_type: str,
                 use_patches: bool,
                 nb_frames: int = 16,
                 label_remap: Optional[dict] = None,
                 sampling_strategy: str = 'top_k',
                 # Patch-loading mode
                 #   'cached'    : load 512-d feature pickles (current default)
                 #   'on_the_fly': load raw frames + landmarks, extract 49×49
                 #                  patches per __getitem__. Required for
                 #                  in-model encoder training (--unfreeze_resnet).
                 patch_mode: str = 'cached',
                 already_cropped: bool = False,
                 base_dir: Optional[str] = None,
                 dataset_name_for_paths: Optional[str] = None,
                 training: bool = False,
                 flip_map: Optional[list] = None,
                 # On-the-fly landmark re-detection (Tasks 12-15)
                 #   When True, re-runs 2D-FAN per frame on the augmented
                 #   224×224 image so that geometry and appearance share the
                 #   same landmark source. Only valid with patch_mode='on_the_fly'.
                 on_the_fly_landmarks: bool = False,
                 # Global appearance mode
                 #   'none'      : use per-landmark 49×49 patches (default).
                 #   'broadcast' : encode the full 224×224 face once per frame,
                 #                 broadcast 512-d feature to all P landmark slots.
                 #   'roi_align' : same full-frame encoding; bilinearly sample a
                 #                 distinct 512-d feature at each landmark position.
                 #                 Both modes require patch_mode='on_the_fly'.
                 #   'local_plus_multiscale' / 'fragnet' : BOTH per-landmark 49×49
                 #                 patches AND full 224×224 frames are collected
                 #                 (needs both patches_dict and full_frames_dict
                 #                 populated).
                 global_appearance_mode: str = 'none',
                 # Temporal-edge graph construction
                 #   'none'               : spatial-only (original behaviour).
                 #   'combined'           : spatial + temporal merged into edge_index.
                 #   'separated_parallel' : spatial in edge_index,
                 #                          temporal in edge_index_temporal.
                 #   'separated_sequential': same split as separated_parallel.
                 temporal_edges_mode: str = 'none'):

        if graph_type not in _GRAPH_BUILDERS:
            raise ValueError(f"Unknown graph_type '{graph_type}'. "
                             f"Choices: {list(_GRAPH_BUILDERS.keys())}")

        if sampling_strategy not in ('top_k', 'uniform'):
            raise ValueError(f"Unknown sampling_strategy '{sampling_strategy}'. "
                             f"Choices: ('top_k', 'uniform')")

        if patch_mode not in ('cached', 'on_the_fly'):
            raise ValueError(f"Unknown patch_mode '{patch_mode}'. "
                             f"Choices: ('cached', 'on_the_fly')")

        if on_the_fly_landmarks and patch_mode != 'on_the_fly':
            raise ValueError(
                "on_the_fly_landmarks=True requires patch_mode='on_the_fly'. "
                "FAN re-detection runs on the raw frame loaded for patch extraction; "
                "it has no effect (and makes no sense) with cached features."
            )

        if on_the_fly_landmarks and not use_patches:
            raise ValueError(
                "on_the_fly_landmarks=True requires use_patches=True: "
                "FAN re-detection is triggered inside the patch-loading path; "
                "without patches, no raw frames are loaded and FAN never runs."
            )

        if global_appearance_mode not in ('none', 'broadcast', 'roi_align', 'combined',
                                          'roi_align_multiscale', 'local_plus_multiscale',
                                          'fragnet'):
            raise ValueError(
                f"Unknown global_appearance_mode {global_appearance_mode!r}. "
                "Choices: ('none', 'broadcast', 'roi_align', 'combined', "
                "'roi_align_multiscale', 'local_plus_multiscale', 'fragnet')"
            )

        self.samples               = list(samples)
        self.adapter               = adapter
        self.graph_type            = graph_type
        self.use_patches           = use_patches
        self.nb_frames             = nb_frames
        self.label_remap           = label_remap
        self.sampling_strategy     = sampling_strategy
        self.patch_mode            = patch_mode
        self.already_cropped       = already_cropped
        self.base_dir              = base_dir
        self.dataset_name_for_paths = dataset_name_for_paths or (
            adapter.dataset_name if hasattr(adapter, 'dataset_name') else None
        )
        self.training               = training
        self.flip_map               = flip_map
        self.on_the_fly_landmarks   = on_the_fly_landmarks
        self.global_appearance_mode = global_appearance_mode
        self.temporal_edges_mode    = temporal_edges_mode
        self._builder               = _GRAPH_BUILDERS[graph_type]

        # Created lazily inside workers (not picklable across fork-then-pickle in
        # all torch versions, depending on how they're set up). We construct them
        # in each worker on first __getitem__.
        self._extender     = None
        self._sampler      = None

        # Probe one sample to compute kept_indices and connections
        # These are deterministic given the graph_type, so we materialize them
        # once at construction and cache them for every __getitem__ call.
        kept_indices, connections, nb_joints = self._probe_graph_geometry()
        self.kept_indices = kept_indices
        self.connections  = connections
        self.nb_joints    = nb_joints

    # Probe

    def _probe_graph_geometry(self):
        """
        Open landmark pkls in order until we find one with at least one valid
        frame, then run the chosen graph-builder on that frame to determine
        nb_joints, kept_indices, and the per-frame edge list. These don't
        depend on the data, only on the graph_type, so caching is safe.
        """
        for (subject, video, _) in self.samples:
            lm_path = self.adapter.landmark_pkl(subject, video)
            if not os.path.exists(lm_path):
                continue
            with open(lm_path, 'rb') as f:
                lm_dict = pickle.load(f)
            lm_dict = _filter_landmarks_dict(lm_dict)
            if not lm_dict:
                continue

            first_frame  = next(iter(sorted(lm_dict.keys())))
            coords       = torch.reshape(
                torch.tensor(lm_dict[first_frame], dtype=torch.float), (-1, 2))
            n_joints, _, edges, kept = self._builder(coords)
            return kept, edges, n_joints

        raise RuntimeError("LazyFERDataset: no usable sample to probe graph "
                           "geometry. Check that landmark pkls exist.")

    # Worker-local helpers

    def _ensure_helpers(self):
        if self._extender is None:
            self._extender = MovementExtender(extension_method='spline_interpolation')
        # Only build the top-k MovementSampler when we'll actually use it.
        # Uniform sampling bypasses it entirely (see _uniform_reduction).
        if self._sampler is None and self.sampling_strategy == 'top_k':
            self._sampler  = MovementSampler().from_preset('top_k')

    @staticmethod
    def _uniform_reduction(lm_filtered, nb_target):
        """
        Uniform temporal subsampling: pick nb_target frames evenly spaced
        across the available valid frames. Returns the same dict shape as
        reduction_frames_video (keys 1..nb_target) plus the original frame
        indices (0-based positions into the sorted key list) for patch alignment.
        """
        keys = sorted(lm_filtered.keys())
        n    = len(keys)
        # Evenly spaced integer indices in [0, n-1]
        sel  = np.linspace(0, n - 1, nb_target).round().astype(int)
        # De-duplicate in order (rare for n >> nb_target, but defensive)
        seen, sel_uniq = set(), []
        for s in sel:
            if s not in seen:
                seen.add(int(s)); sel_uniq.append(int(s))
        # If de-dup shrank us, pad by repeating the last index
        while len(sel_uniq) < nb_target:
            sel_uniq.append(sel_uniq[-1])
        out = {i + 1: lm_filtered[keys[idx]]
               for i, idx in enumerate(sel_uniq)}
        return out, sel_uniq

    # PyTorch Dataset interface

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        self._ensure_helpers()
        subject, video, label = self.samples[idx]
        if self.label_remap is not None:
            label = self.label_remap[label]

        # Load landmarks
        lm_path = self.adapter.landmark_pkl(subject, video)
        with open(lm_path, 'rb') as f:
            lm_dict = pickle.load(f)
        lm_dict = _filter_landmarks_dict(lm_dict)

        if len(lm_dict) < 2:
            # Should not happen (scan_samples filters these out), but defensive.
            raise RuntimeError(f"Sample {subject}/{video} has <2 valid frames; "
                               f"should have been filtered by scan_samples().")

        # Per-sample augmentation (§4.2): sample ONCE, apply to all frames
        # Training: random horizontal flip + random 224×224 crop offset in {0..16}.
        # Eval:     no flip, center crop (dx=dy=8 → symmetric 8px margin).
        if self.training:
            do_flip = (torch.rand(1).item() < 0.5)
            dx = int(torch.randint(0, 17, (1,)).item())  # {0, …, 16}
            dy = int(torch.randint(0, 17, (1,)).item())
        else:
            do_flip = False
            dx, dy  = 8, 8  # center crop

        # Transform all 68 landmark coords to 224×224 space (same offsets for
        # every frame so geometry is temporally consistent).
        lm_dict_aug = {}
        for k, coords_flat in lm_dict.items():
            lm_arr = np.asarray(coords_flat, dtype=np.float32).reshape(-1, 2)
            if do_flip:
                lm_arr[:, 0] = 239.0 - lm_arr[:, 0]  # cv2.flip maps x → 239-x on 240-wide image
            lm_arr[:, 0] -= dx
            lm_arr[:, 1] -= dy
            lm_dict_aug[k] = lm_arr  # (68, 2) in 224×224 space

        # Load patches / full frames
        # patches_dict  : {frame_id: ndarray} — per-landmark patches or features.
        #                  None when not needed.
        # full_frames_dict : {frame_id: (3,224,224) uint8 RGB} — only populated
        #                  in broadcast mode.
        patches_dict     = None
        full_frames_dict = None

        if self.global_appearance_mode in ('local_plus_multiscale', 'fragnet'):
            # Both modes need per-landmark patches AND full frames:
            # 'local_plus_multiscale' concatenates local + multi-scale features;
            # 'fragnet' feeds patches through the fragment CNN, laterally fused
            # with pyramid windows sampled from the full frames. Both come from
            # the same raw 224×224 crop, so extract them together in one FAN
            # pass instead of calling both extraction helpers (which would
            # double file I/O and run FAN twice per frame).
            if self.patch_mode != 'on_the_fly':
                raise ValueError(
                    f"global_appearance_mode={self.global_appearance_mode!r} requires "
                    "patch_mode='on_the_fly' (needs raw frames to build both "
                    "patches and full_frames)."
                )
            if self.on_the_fly_landmarks:
                patches_dict, lm_dict_aug, full_frames_dict = self._extract_patches_on_the_fly(
                    subject, video, lm_dict_aug, do_flip, dx, dy,
                    also_collect_full_frames=True)
            else:
                patches_dict, full_frames_dict = self._extract_patches_on_the_fly(
                    subject, video, lm_dict_aug, do_flip, dx, dy,
                    also_collect_full_frames=True)

        elif self.global_appearance_mode in ('broadcast', 'roi_align', 'combined',
                                             'roi_align_multiscale'):
            # Global appearance modes: collect full 224×224 frames (skip per-
            # landmark patch extraction). FAN is still run for landmark geometry
            # when on_the_fly_landmarks is active. Both broadcast and roi_align
            # need the same full-frame tensor; roi_align additionally uses the
            # landmark positions to sample the feature map (handled in the model).
            if self.patch_mode == 'on_the_fly':
                full_frames_dict, lm_dict_aug = self._extract_broadcast_frames(
                    subject, video, lm_dict_aug, do_flip, dx, dy)
            # patches_dict stays None — model uses full_frames instead.

        elif self.use_patches:
            if self.patch_mode == 'cached':
                pat_path = self.adapter.patches_pkl(subject, video)
                if os.path.exists(pat_path):
                    with open(pat_path, 'rb') as f:
                        raw_patches = pickle.load(f)
                    patches_dict = _align_patches_to_landmarks(
                        raw_patches, set(lm_dict.keys()))
                else:
                    patches_dict = {k: np.zeros((68, 512), dtype=np.float16)
                                    for k in lm_dict.keys()}
            elif self.patch_mode == 'on_the_fly':
                if self.on_the_fly_landmarks:
                    # Task 13-14: run 2D-FAN on the 224×224 augmented image.
                    # _extract_patches_on_the_fly returns (patches, detected_lm)
                    # where detected_lm[k] is (68,2) in 224-space, falling back
                    # to lm_dict_aug[k] on FAN failure (Task 15).
                    patches_dict, lm_dict_aug = self._extract_patches_on_the_fly(
                        subject, video, lm_dict_aug, do_flip, dx, dy)
                else:
                    patches_dict = self._extract_patches_on_the_fly(
                        subject, video, lm_dict_aug, do_flip, dx, dy)

        # Filter to graph-specific joints (mirrors convertion_graphe)
        kept = self.kept_indices
        lm_filtered = {}
        if self.on_the_fly_landmarks:
            # lm_dict_aug[k] is already (P, 2): _extract_patches_on_the_fly
            # applied [kept] reduction and flip_map per-frame at the point
            # where the landmark source (FAN-detected vs cached fallback) was known.
            # Applying _builder or flip_map here would double-apply the transform.
            for k, lm_arr in lm_dict_aug.items():
                lm_filtered[k] = torch.tensor(lm_arr, dtype=torch.float).reshape(-1)
        else:
            for k, lm_arr in lm_dict_aug.items():
                coords      = torch.tensor(lm_arr, dtype=torch.float)  # (68, 2)
                _, lf, _, _ = self._builder(coords)
                lf_2d = lf.reshape(-1, 2)                              # (P, 2)
                if do_flip and self.flip_map is not None:
                    lf_2d = lf_2d[self.flip_map]                       # semantic reorder within kept subset
                lm_filtered[k] = lf_2d.reshape(-1)

        if patches_dict is not None:
            if self.on_the_fly_landmarks:
                # patches_dict[k] is already (P, 3, 49, 49) from _extract_patches_on_the_fly,
                # with [kept] and flip_map applied per-frame for fallback frames only.
                patches_filtered = patches_dict
            else:
                # patches_dict[k] shape is (68, 512) OR (68, 3, 49, 49).
                # [kept] filters to P joints; flip_map then reorders to match coord slots.
                if do_flip and self.flip_map is not None:
                    patches_filtered = {k: patches_dict[k][kept][self.flip_map]
                                        for k in lm_dict_aug.keys()}
                else:
                    patches_filtered = {k: patches_dict[k][kept] for k in lm_dict_aug.keys()}
        else:
            patches_filtered = None

        # Temporal alignment to self.nb_frames
        nb_valid = len(lm_filtered)
        if nb_valid == self.nb_frames:
            aligned_lm, aligned_pat = lm_filtered, patches_filtered
            aligned_full = full_frames_dict
        elif nb_valid < self.nb_frames:
            aligned_lm = augmentation_frames_video(
                lm_filtered, self.nb_frames, self._extender)
            aligned_pat = (augmentation_patches_video(patches_filtered, self.nb_frames)
                           if patches_filtered is not None else None)
            aligned_full = (augmentation_patches_video(full_frames_dict, self.nb_frames)
                            if full_frames_dict is not None else None)
        else:
            if self.sampling_strategy == 'uniform':
                aligned_lm, indices = self._uniform_reduction(
                    lm_filtered, self.nb_frames)
            else:
                aligned_lm, indices = reduction_frames_video(
                    lm_filtered, self.nb_frames, self._sampler)
            aligned_pat = (reduction_patches_video(patches_filtered, indices)
                           if patches_filtered is not None and indices is not None
                           else patches_filtered)
            aligned_full = (reduction_patches_video(full_frames_dict, indices)
                            if full_frames_dict is not None and indices is not None
                            else full_frames_dict)

        # Build (T*P, 2) coord tensor and patch tensor
        # Both lm and patch dicts MUST share the same sorted key order so the
        # T dimension aligns. We assert this defensively.
        lm_keys  = sorted(aligned_lm.keys())
        lm_t = convertir_dictionnaire_video_en_tenseur_video(aligned_lm)   # (P,2,T)
        lm_t = lm_t.permute(2, 0, 1).reshape(-1, 2, 1)                     # (T*P,2,1)
        lm_t = lm_t.squeeze(-1)                                            # (T*P,2)

        pat_t = None
        if aligned_pat is not None:
            # Stack frames in sorted key order — must match lm_keys exactly.
            pat_keys = sorted(aligned_pat.keys())
            if pat_keys != lm_keys:
                raise RuntimeError(
                    f'Landmark/patch key mismatch for {subject}/{video}: '
                    f'lm_keys={lm_keys[:4]}…  pat_keys={pat_keys[:4]}…')
            pat_chunks = [aligned_pat[k] for k in pat_keys]
            sample0 = pat_chunks[0]
            if sample0.ndim == 2:
                # Cached features: (P, 512) per frame  ->  (T*P, 512) float16
                pat_t = torch.cat(
                    [torch.tensor(c, dtype=torch.float16).unsqueeze(0)
                     for c in pat_chunks], dim=0
                )                                                            # (T,P,512)
                pat_t = pat_t.reshape(-1, 512)                               # (T*P,512)
            elif sample0.ndim == 4:
                # Raw patches: (P, 3, 49, 49) per frame  ->  (T*P, 3, 49, 49) uint8
                # np.ascontiguousarray ensures torch.from_numpy gets a C-contiguous
                # buffer — avoids a silent copy inside cat for non-contiguous slices.
                pat_t = torch.cat(
                    [torch.from_numpy(np.ascontiguousarray(c)).unsqueeze(0)
                     for c in pat_chunks],
                    dim=0,
                )                                                            # (T,P,3,49,49)
                pat_t = pat_t.reshape(-1, 3, 49, 49)                         # (T*P,3,49,49)
            else:
                raise RuntimeError(f'Unexpected patch tensor ndim={sample0.ndim}')

        # Full-frame tensor (broadcast mode)
        full_frames_t = None
        if aligned_full is not None:
            full_chunks = [aligned_full[k] for k in lm_keys]   # list of (3,224,224) uint8
            full_frames_t = torch.from_numpy(
                np.ascontiguousarray(np.stack(full_chunks, axis=0))
            )                                                    # (T, 3, 224, 224)

        data = create_graph(
            lm_t, int(label),
            self.nb_frames, self.nb_joints,
            self.connections,
            patch_tensor=pat_t,
            temporal_edges_mode=self.temporal_edges_mode,
        )
        if full_frames_t is not None:
            data.full_frames = full_frames_t
        return data

    def _extract_patches_on_the_fly(self, subject: str, video: str,
                                     lm_dict_aug: dict,
                                     do_flip: bool, dx: int, dy: int,
                                     also_collect_full_frames: bool = False):
        """
        For each frame_id in lm_dict_aug, load the 240×240 face crop, apply
        flip and 224×224 crop, then extract 49×49 RGB patches.

        also_collect_full_frames : when True (used by
            global_appearance_mode='local_plus_multiscale'), additionally
            stashes the augmented 224×224 RGB crop into a full_frames_dict
            (same format as _extract_broadcast_frames) using the SAME crop
            already loaded for patch extraction — avoids a second disk read
            and a second FAN pass. Appends full_frames_dict as the last
            element of the return tuple.

        When self.on_the_fly_landmarks is False (default):
            Landmarks in lm_dict_aug (already in 224-space) are used directly.
            Returns  {frame_id: (68, 3, 49, 49) uint8}
            or       ({frame_id: (68, 3, 49, 49) uint8}, full_frames_dict)
                     when also_collect_full_frames=True.

        When self.on_the_fly_landmarks is True (Tasks 13-15, 17):
            After building the 224×224 augmented image, runs 2D-FAN on it.
            For FAN-success frames: patches and landmarks are reduced to P-space
            (self.kept_indices) with NO flip_map — FAN already ran on the
            flipped image so its landmark ordering is geometrically consistent.
            For FAN-failure frames: cached coords (coord-only flip, no reorder)
            are used; flip_map IS applied to fix semantic left↔right ordering.
            Both branches return P-space arrays so __getitem__ can use results
            directly without any further [kept] / flip_map transforms.
            Returns  ({frame_id: (P, 3, 49, 49) uint8},
                      {frame_id: (P, 2) float32 in 224-space})
            or       ({frame_id: (P, 3, 49, 49) uint8},
                      {frame_id: (P, 2) float32 in 224-space},
                      full_frames_dict)
                     when also_collect_full_frames=True.

        Frame-id → filename mapping mirrors extract_patches.py:
            valid_files = sorted(*.jpg/png/jpeg in the frames dir)
            frame_id = enumerate(valid_files, start=1)
        """
        if self.base_dir is None or self.dataset_name_for_paths is None:
            raise RuntimeError(
                'LazyFERDataset: on_the_fly patch_mode requires base_dir '
                'and dataset_name_for_paths. Pass them at construction.')

        frames_dir = self.adapter.frames_dir(subject, video)
        try:
            all_files = sorted([f for f in os.listdir(frames_dir)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        except FileNotFoundError:
            if self.on_the_fly_landmarks:
                P = len(self.kept_indices)
                out_patches = {k: np.zeros((P, 3, 49, 49), dtype=np.uint8)
                               for k in lm_dict_aug}
                out_lm = {}
                for k, lm_f in lm_dict_aug.items():
                    lm_kept = lm_f[self.kept_indices]
                    if do_flip and self.flip_map is not None:
                        lm_kept = lm_kept[self.flip_map]
                    out_lm[k] = lm_kept
                if also_collect_full_frames:
                    full_frames_dict = {k: np.zeros((3, 224, 224), dtype=np.uint8)
                                        for k in lm_dict_aug}
                    return out_patches, out_lm, full_frames_dict
                return out_patches, out_lm
            out_patches = {k: np.zeros((68, 3, 49, 49), dtype=np.uint8) for k in lm_dict_aug}
            if also_collect_full_frames:
                full_frames_dict = {k: np.zeros((3, 224, 224), dtype=np.uint8)
                                    for k in lm_dict_aug}
                return out_patches, full_frames_dict
            return out_patches

        bbox_path = (self.adapter.bbox_pkl(subject, video)
                     if hasattr(self.adapter, 'bbox_pkl')
                     else bbox_pkl_path(self.base_dir, self.dataset_name_for_paths,
                                        subject, video))
        bbox_dict = _load_bbox_dict(bbox_path) if not self.already_cropped else None

        out_patches = {}
        out_lm      = {}   # populated only when on_the_fly_landmarks is True
        full_frames_dict = {} if also_collect_full_frames else None

        if self.on_the_fly_landmarks:
            # Task 25: batch all FAN calls for this sample's frames into one GPU call.
            # Phase 1 — load every crop (flip + 224-crop applied); record valid frame IDs.
            from dataloader.fan_detector import detect_batch as _fan_detect_batch  # noqa: PLC0415

            frame_crops: dict = {}  # k -> crop_224 (ndarray) or None
            valid_ks: list = []     # frame IDs with a usable crop, in insertion order

            for k, lm_fallback in lm_dict_aug.items():
                file_idx = k - 1
                if file_idx < 0 or file_idx >= len(all_files):
                    frame_crops[k] = None
                    continue
                fname      = all_files[file_idx]
                frame_path = os.path.join(frames_dir, fname)
                crop_240   = load_and_crop_to_240(
                    frame_path, bbox_dict, fname,
                    already_cropped=self.already_cropped,
                )
                if crop_240 is None:
                    frame_crops[k] = None
                    continue
                if do_flip:
                    crop_240 = cv2.flip(crop_240, 1)
                frame_crops[k] = crop_240[dy:dy + 224, dx:dx + 224]
                valid_ks.append(k)

            # Phase 2 — one batched FAN call for all valid crops.
            if valid_ks:
                batch_results = _fan_detect_batch([frame_crops[k] for k in valid_ks])
                detected_map  = dict(zip(valid_ks, batch_results))
            else:
                detected_map = {}

            # Phase 3 — emit per-frame patches and landmarks.
            P = len(self.kept_indices)
            for k, lm_fallback in lm_dict_aug.items():
                crop_224 = frame_crops.get(k)
                if crop_224 is None:
                    out_patches[k] = np.zeros((P, 3, 49, 49), dtype=np.uint8)
                    lm_kept = lm_fallback[self.kept_indices]
                    if do_flip and self.flip_map is not None:
                        lm_kept = lm_kept[self.flip_map]
                    out_lm[k] = lm_kept
                    if also_collect_full_frames:
                        full_frames_dict[k] = np.zeros((3, 224, 224), dtype=np.uint8)
                    continue

                detected = detected_map.get(k)
                if detected is not None:
                    # FAN success: reduce to P-space; NO flip_map — FAN ran on the
                    # already-flipped image so its landmark ordering is naturally
                    # consistent with the flipped geometry.
                    lm_kept        = detected[self.kept_indices]                       # (P, 2)
                    out_patches[k] = extract_patches_from_crop(
                        crop_224, detected, kept_indices=self.kept_indices)            # (P,3,49,49)
                    out_lm[k]      = lm_kept
                else:
                    # FAN failed: cached coords (coord-only flip, no index reorder) —
                    # reduce to P-space then apply flip_map to fix semantic ordering.
                    _log_fan_fallback(subject, video, k)
                    lm_kept        = lm_fallback[self.kept_indices]                    # (P, 2)
                    out_patches[k] = extract_patches_from_crop(
                        crop_224, lm_fallback, kept_indices=self.kept_indices)         # (P,3,49,49)
                    if do_flip and self.flip_map is not None:
                        lm_kept        = lm_kept[self.flip_map]
                        out_patches[k] = out_patches[k][self.flip_map]
                    out_lm[k] = lm_kept

                if also_collect_full_frames:
                    # Reuse the same crop_224 already loaded above — RGB, CHW.
                    rgb = cv2.cvtColor(crop_224, cv2.COLOR_BGR2RGB)
                    full_frames_dict[k] = np.ascontiguousarray(
                        np.transpose(rgb, (2, 0, 1)))                # (3, 224, 224)

        else:
            # Non-OTF-landmarks path: extract all 68 patches using cached landmarks.
            # Graph-joint filtering is applied in __getitem__.
            for k, lm_fallback in lm_dict_aug.items():
                file_idx = k - 1
                if file_idx < 0 or file_idx >= len(all_files):
                    out_patches[k] = np.zeros((68, 3, 49, 49), dtype=np.uint8)
                    if also_collect_full_frames:
                        full_frames_dict[k] = np.zeros((3, 224, 224), dtype=np.uint8)
                    continue
                fname      = all_files[file_idx]
                frame_path = os.path.join(frames_dir, fname)
                crop_240   = load_and_crop_to_240(
                    frame_path, bbox_dict, fname,
                    already_cropped=self.already_cropped,
                )
                if crop_240 is None:
                    out_patches[k] = np.zeros((68, 3, 49, 49), dtype=np.uint8)
                    if also_collect_full_frames:
                        full_frames_dict[k] = np.zeros((3, 224, 224), dtype=np.uint8)
                    continue
                if do_flip:
                    crop_240 = cv2.flip(crop_240, 1)
                crop_224       = crop_240[dy:dy + 224, dx:dx + 224]
                out_patches[k] = extract_patches_from_crop(
                    crop_224, lm_fallback, kept_indices=list(range(68)))
                if also_collect_full_frames:
                    rgb = cv2.cvtColor(crop_224, cv2.COLOR_BGR2RGB)
                    full_frames_dict[k] = np.ascontiguousarray(
                        np.transpose(rgb, (2, 0, 1)))                # (3, 224, 224)

        if self.on_the_fly_landmarks:
            if also_collect_full_frames:
                return out_patches, out_lm, full_frames_dict
            return out_patches, out_lm
        if also_collect_full_frames:
            return out_patches, full_frames_dict
        return out_patches

    def _extract_broadcast_frames(self, subject: str, video: str,
                                   lm_dict_aug: dict,
                                   do_flip: bool, dx: int, dy: int):
        """
        Broadcast-mode frame collector (Approach 1).

        Loads the 240×240 face crop per frame, applies flip + 224×224 crop,
        and returns the full RGB frame — no per-landmark patch extraction.

        If self.on_the_fly_landmarks is True, also runs batched 2D-FAN to
        update landmark coordinates (same as _extract_patches_on_the_fly but
        without the patch extraction step).

        Returns
        -------
        full_frames_dict : {frame_id: (3, 224, 224) uint8 RGB ndarray}
        lm_dict_aug      : updated with FAN-detected P-space landmarks when
                           on_the_fly_landmarks=True, else unchanged.
        """
        if self.base_dir is None or self.dataset_name_for_paths is None:
            raise RuntimeError(
                'LazyFERDataset: broadcast mode requires base_dir and '
                'dataset_name_for_paths. Pass them at construction.')

        frames_dir = self.adapter.frames_dir(subject, video)
        P = len(self.kept_indices)

        try:
            all_files = sorted([f for f in os.listdir(frames_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        except FileNotFoundError:
            full_frames_dict = {k: np.zeros((3, 224, 224), dtype=np.uint8)
                                for k in lm_dict_aug}
            if self.on_the_fly_landmarks:
                updated_lm = {}
                for k, lm_f in lm_dict_aug.items():
                    lm_kept = lm_f[self.kept_indices]
                    if do_flip and self.flip_map is not None:
                        lm_kept = lm_kept[self.flip_map]
                    updated_lm[k] = lm_kept
                return full_frames_dict, updated_lm
            return full_frames_dict, lm_dict_aug

        bbox_path = (self.adapter.bbox_pkl(subject, video)
                     if hasattr(self.adapter, 'bbox_pkl')
                     else bbox_pkl_path(self.base_dir, self.dataset_name_for_paths,
                                        subject, video))
        bbox_dict = _load_bbox_dict(bbox_path) if not self.already_cropped else None

        full_frames_dict = {}

        if self.on_the_fly_landmarks:
            from dataloader.fan_detector import detect_batch as _fan_detect_batch

            # Phase 1 — load + augment crops for all frames
            frame_crops: dict = {}
            valid_ks: list = []
            for k in lm_dict_aug:
                file_idx = k - 1
                if file_idx < 0 or file_idx >= len(all_files):
                    frame_crops[k] = None
                    continue
                fname      = all_files[file_idx]
                frame_path = os.path.join(frames_dir, fname)
                crop_240   = load_and_crop_to_240(
                    frame_path, bbox_dict, fname,
                    already_cropped=self.already_cropped)
                if crop_240 is None:
                    frame_crops[k] = None
                    continue
                if do_flip:
                    crop_240 = cv2.flip(crop_240, 1)
                frame_crops[k] = crop_240[dy:dy + 224, dx:dx + 224]
                valid_ks.append(k)

            # Phase 2 — batched FAN
            if valid_ks:
                batch_results = _fan_detect_batch([frame_crops[k] for k in valid_ks])
                detected_map  = dict(zip(valid_ks, batch_results))
            else:
                detected_map = {}

            # Phase 3 — emit full frames + updated landmarks (no patch extraction)
            updated_lm: dict = {}
            for k, lm_fallback in lm_dict_aug.items():
                crop_224 = frame_crops.get(k)
                if crop_224 is None:
                    full_frames_dict[k] = np.zeros((3, 224, 224), dtype=np.uint8)
                    lm_kept = lm_fallback[self.kept_indices]
                    if do_flip and self.flip_map is not None:
                        lm_kept = lm_kept[self.flip_map]
                    updated_lm[k] = lm_kept
                    continue

                # Convert BGR crop to RGB full frame
                rgb = cv2.cvtColor(crop_224, cv2.COLOR_BGR2RGB)
                full_frames_dict[k] = np.ascontiguousarray(
                    np.transpose(rgb, (2, 0, 1)))                # (3, 224, 224)

                detected = detected_map.get(k)
                if detected is not None:
                    # FAN success: use detected landmarks, no flip_map needed
                    # (FAN ran on the already-flipped image)
                    updated_lm[k] = detected[self.kept_indices]  # (P, 2)
                else:
                    # FAN failure: fall back to cached coords + flip_map
                    _log_fan_fallback(subject, video, k)
                    lm_kept = lm_fallback[self.kept_indices]
                    if do_flip and self.flip_map is not None:
                        lm_kept = lm_kept[self.flip_map]
                    updated_lm[k] = lm_kept

            return full_frames_dict, updated_lm

        else:
            # Non-OTF landmarks: just collect full frames using cached coords
            for k in lm_dict_aug:
                file_idx = k - 1
                if file_idx < 0 or file_idx >= len(all_files):
                    full_frames_dict[k] = np.zeros((3, 224, 224), dtype=np.uint8)
                    continue
                fname      = all_files[file_idx]
                frame_path = os.path.join(frames_dir, fname)
                crop_240   = load_and_crop_to_240(
                    frame_path, bbox_dict, fname,
                    already_cropped=self.already_cropped)
                if crop_240 is None:
                    full_frames_dict[k] = np.zeros((3, 224, 224), dtype=np.uint8)
                    continue
                if do_flip:
                    crop_240 = cv2.flip(crop_240, 1)
                crop_224 = crop_240[dy:dy + 224, dx:dx + 224]
                rgb = cv2.cvtColor(crop_224, cv2.COLOR_BGR2RGB)
                full_frames_dict[k] = np.ascontiguousarray(
                    np.transpose(rgb, (2, 0, 1)))                # (3, 224, 224)

            return full_frames_dict, lm_dict_aug


# Helper: scan samples on disk and drop any with <2 valid landmark frames

def scan_samples(samples, adapter, min_valid_frames: int = 2):
    """
    Walk the given (subject, video, label) triples and keep only those whose
    landmark pkl exists and yields >= min_valid_frames after filtering 2D-FAN
    failures. Mirrors the loaded_samples logic in extract_datas_video_labels
    but without loading patches or building tensors.

    Returns
    -------
    kept    : list of (subject, video, label) tuples
    dropped : int — count of skipped samples (for logging)
    """
    kept, dropped = [], 0
    for (subject, video, label) in samples:
        lm_path = adapter.landmark_pkl(subject, video)
        if not os.path.exists(lm_path):
            dropped += 1
            continue
        try:
            with open(lm_path, 'rb') as f:
                lm_dict = pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            print(f"  [skip] corrupt landmark pkl: {subject}/{video}")
            dropped += 1
            continue
        lm_dict = _filter_landmarks_dict(lm_dict)
        if len(lm_dict) < min_valid_frames:
            dropped += 1
            continue
        kept.append((subject, video, label))
    return kept, dropped