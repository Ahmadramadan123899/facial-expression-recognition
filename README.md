# Geometry-Guided Facial Expression Recognition from Videos

This repo contains the facial expression recognition system I built during my
internship at CRIStAL. It's built on top of a reproduction of Zhao et al.'s
Graph-Tran paper, which I started as my Master 1 research project. The full
context (methodology, experiments, results) is in the internship report
(`Report/internship.pdf`). This README just covers dataset access, code
layout, and how to run training and evaluation.

## Datasets

You need permission from the original authors to use these datasets. The
paths below point to the internal server. If you're outside the lab, get
the data from the official source linked in each section.

### CK+

Extended Cohn-Kanade. Lab-recorded, posed and spontaneous expressions from
123 subjects, labeled at the peak (apex) frame. 7 classes in this codebase's
split: Anger, Contempt, Disgust, Fear, Happiness, Sadness, Surprise. 327
usable sequences. It's small and clean, so it's a good sanity check, but it
doesn't tell you much about in-the-wild performance.

- Internal path: `/sandbox1/aramadan/frames/CK+/`
- Official source: http://www.jeffcohn.net/Resources/ (upon request)
- Split: 10-fold identity-independent cross-validation (`GroupKFold` on
  subject ID)

### Oulu-CASIA

Lab-recorded, 6 classes (Anger, Disgust, Fear, Happiness, Sadness, Surprise),
480 sequences from 80 subjects, shot under both strong and dark lighting.
The codebase calls this dataset `Oulu` (`--nom_dataset Oulu`). There's also a
pre-cropped variant, `Oulu_Acropped`, for frames that already come
face-cropped.

- Internal path: `/sandbox1/aramadan/frames/Oulu-CASIA/`
- Official source: https://www.oulu.fi/en/university/faculties-and-units/faculty-information-technology-and-electrical-engineering/center-machine-vision-and-signal-analysis (upon request)
- Split: 10-fold identity-independent cross-validation

### eNTERFACE05

An audio-visual emotion dataset, 6 classes. Subjects reacted to spoken
scenarios instead of posing on command, so it's a bit closer to spontaneous
expression than CK+ or Oulu, though it's still recorded in a lab. 1,166+
usable sequences.

- Internal path: `/sandbox1/aramadan/frames/eNTERFACE05/`
- Official source: https://enterface.net/enterface05/emotion.html
- Split: 5-fold identity-independent cross-validation

### AFEW

Acted Facial Expressions in the Wild. Clips from movies and TV shows, 7
classes. This one is actually in-the-wild: occlusion, camera motion, faces
not facing the camera, scene cuts. 773 train / 383 val sequences,
distributed through the EmotiW challenge. There's no stable subject identity
across clips, so for cross-validation each clip is treated as its own
"subject".

- Internal path: `/sandbox1/aramadan/frames/AFEW/`
- Official source: distributed under the EmotiW challenge; request access via
  https://sites.google.com/view/emotiw2020
- Split: fixed Train/Val split (EmotiW protocol)

### CAER

Context-Aware Emotion Recognition. 7 classes, ~13,000 short clips from TV
shows, similar to AFEW but about 10x bigger. The original CAER-Net baseline
uses both the face and the background context; this codebase only uses face
landmarks, which is one reason its CAER results don't match the paper's
baseline (see the report for the full discussion).

- Internal path: `/sandbox1/aramadan/frames/CAER/`
- Official source: https://caer-dataset.github.io/
- Split: fixed Train/Test split (~9,100 train / ~2,600 test)

Note: raw videos for CK+, Oulu, AFEW, and FER39K already exist on the fox
GPU server, so no need to hunt for them online if you're using that server.

## Report & results

- `Report/internship.pdf` is the full internship report: methodology,
  architecture, ablations, and discussion of every result below.
  Authoritative reference for anything this README only summarizes in
  passing.
- `Results/<dataset>/` holds the training artifacts for the runs cited in
  the report, one folder per dataset (`AFEW`, `Oulu`, `ck+`, `enterface05`):
  - `commandline_args.txt`: exact flags used for that run.
  - `models/fold*_*.pth`: per-fold checkpoints.
  - `plots/`: per-fold accuracy/loss/confusion-matrix curves.
  - `GCNTransformerPaper_{accuracy,precision,recall}.png`: aggregate
    across-fold plots.

## Code layout

### Data pipeline

- `dataloader/lazy_dataset.py`: streaming PyG dataset (`LazyFERDataset`).
  Loads one (landmark, patch) sample at a time from disk instead of loading
  everything into RAM up front. Needed for CAER-scale data (~13k sequences),
  where loading it all at once would run out of memory.
- `dataloader/build_dataset_paper.py`: the eager pipeline. Loads everything
  into RAM up front, builds the spatial graph connectivity for the chosen
  `graph_type` (`g_topologique`, etc.), and applies bulk class-balancing
  augmentation. Used for CK+, Oulu, and eNTERFACE05.
- `dataloader/fan_detector.py`: wraps the 2D-FAN landmark detector so it's
  safe to use with multiple DataLoader workers. FAN itself isn't fork-safe,
  so this wrapper waits and initializes it the first time each worker
  process calls it.
- `dataloader/patch_extractor.py`: crops 49×49 patches on the fly from raw
  frames, using a precomputed face-bbox cache. Used for end-to-end encoder
  fine-tuning.
- `dataloader/MovementExtender.py`: extends videos shorter than the target
  frame count using cubic spline interpolation.
- `dataloader/MovementSampler.py`: picks frames by motion score (keeps the
  top-K highest-motion frames).
- `dataloader/constants.py`: groups the 68 2D-FAN landmarks by facial
  region (jaw, brows, eyes, nose, mouth), used to build graph subsets and
  normalization clusters.
- `datasets/*.py`: one adapter per dataset (`ckplus.py`, `oulu.py`,
  `oulu_acropped.py`, `enterface05.py`, `afew.py`, `caer.py`,
  `afew_caer_joint.py`), all implementing the `DatasetAdapter` interface from
  `datasets/base.py`. `datasets/__init__.py` is the registry that maps
  dataset name strings to adapter classes. `dfew.py` and `ferv39k.py` are
  registered but weren't used in the reported experiments.

### Model

- `models/gnn_paper.py`: the entire architecture. `GCNTransformerPaper`
  (top-level model), `PatchEncoder` (ResNet-18/34 wrapper for local patches),
  `CoordBranch` (the R/C/T geometry branches), `GCBlock`, `ComponentAttention`,
  `FrameAttention`, `SpatialTransformerBlock`, `TemporalTransformerBlock`,
  `TGConv`/`TGConvOnEdges` (two temporal-mixing variants, see the report for
  the comparison), `FragmentPathway` and `GridBasedFragNet` (the
  FragNet-inspired appearance modes, see the report for the comparison), and
  `CenterLoss`.

### Training

- `train_gnn_paper.py`: the main training script. Every CLI flag is defined
  here (bottom of the file), so it's the final word on what a given run
  actually does.
- `executor/trainer_paper.py`: the training loop. Loss (cross-entropy +
  center loss), per-batch augmentation (coordinate noise, patch noise), and
  checkpoint selection.

### Auxiliary

- `preprocessing_exp_faciale.py`: runs 2D-FAN landmark extraction. Crops a
  240×240 face region per frame and saves a landmark pickle per video.
- `extract_patches.py`: offline appearance feature extraction. Crops a
  49×49 patch at each landmark and runs it through the FER+ pretrained
  ResNet, saving a feature cache for the fast/frozen-encoder training path.
- `precompute_bboxes.py`: precomputes and caches a face bounding box per
  frame. Needed for on-the-fly patch extraction (`--unfreeze_resnet`
  `partial`/`full`).
- `organize_afew.py`, `organize_caer.py`, `organize_enterface05.py`:
  convert each dataset's raw layout (`.avi` clips in dataset-specific
  folders) into the per-frame JPEG layout the rest of the pipeline expects.
- `results.py`, `utils.py`: result-saving/plotting helpers and general
  pickle/directory utilities used across the pipeline.
- `Makefile`: a one-command interface wrapping all the scripts above (see
  "Running the code" below).

## Running the code

The pipeline runs in five stages: organize raw clips into frames (AFEW/CAER
only), extract landmarks, optionally precompute bounding boxes, optionally
extract cached appearance features, then train. Everything goes through the
`Makefile`.

### Environment setup

Runs on Python 3, using the conda environment referenced in the `Makefile`
(`/sandbox1/aramadan/envs/gnn_env`). Check `requirements.txt` for exact
versions. Key dependencies:

- PyTorch
- PyTorch Geometric (`torch_geometric`)
- torchvision
- OpenCV (`opencv-python`)
- `face-alignment` (2D-FAN)
- scikit-learn
- NumPy, pandas, matplotlib, seaborn, networkx, tqdm

### Data preparation

You need permission from the original authors to get the raw datasets (see
Datasets above). Once you have frames in the expected layout, run:

```bash
make organize_afew                 # AFEW: .avi -> per-frame JPEGs
make organize_caer                 # CAER: .avi -> per-frame JPEGs
make organize_enterface05          # eNTERFACE05: .avi -> per-frame JPEGs
make landmarks dataset=CK+          # 2D-FAN landmark extraction
make bboxes    dataset=AFEW         # only needed for on-the-fly patch mode
make patches   dataset=CK+          # only needed for cached-feature mode
```

`organize_afew`, `organize_caer`, and `organize_enterface05` share two
options: `downsample=` (uniformly subsamples frames per video, e.g.
`downsample=--downsample 32`; default keeps every frame) and
`overwrite=--overwrite` (re-extracts videos that already have frames). CK+
and Oulu already ship as frames, so they never need an `organize_*` step.

Once frames exist, `make landmarks` is the one step every training mode
needs. See "Which preprocessing steps do you need?" below for the full
breakdown before jumping to `make bboxes` or `make patches`.

### Which preprocessing steps do you need?

There are three ways to run training, and each needs different prep.
Getting it wrong doesn't always fail loudly: skip `make bboxes` before an
on-the-fly run and it crashes on the first batch, but run `make patches`
when you didn't need it and you just waste GPU time for nothing.

| Mode | Trigger | `make landmarks` | `make bboxes` | `make patches` |
|---|---|---|---|---|
| **Cached-patch** (fast path) | `unfreeze_resnet` left empty (`none`) | required | not needed | required |
| **On-the-fly encoder** | `unfreeze_resnet` = `frozen`/`partial`/`full` + `lazy_loading` | required | required | not needed |
| **On-the-fly landmarks** | above, plus `--on_the_fly_landmarks` | required | required | not needed |

A few things that aren't obvious from the flag names:

- **`make landmarks` is required in all three modes. No exceptions.** Even
  in on-the-fly-landmarks mode, where 2D-FAN re-detects landmarks live on
  every augmented frame, the cached landmark file is still read as a
  fallback whenever FAN fails to find a face (`dataloader/lazy_dataset.py`).
  Skipping `make landmarks` doesn't just skip an optimization, it breaks
  that fallback.
- **Cached mode does NOT need `make bboxes`.** `extract_patches.py` (the
  `make patches` step) re-runs 2D-FAN itself to crop the 240×240 face
  region per frame; it doesn't use the bbox cache at all.
- **On-the-fly modes do NOT need `make patches`.** Raw frames are loaded at
  train time, and patches are cropped on the fly using the `make bboxes`
  cache instead of a pre-extracted feature file. Running `make patches`
  anyway isn't wrong, it's just wasted disk I/O in these modes.
- **`--already_cropped` datasets (`Oulu_Acropped`) skip `make bboxes`
  entirely**, even in on-the-fly mode. The loader just treats the whole
  frame as the face crop; no bounding box lookup happens.
- **`--chemin_patches` (Makefile: `chemin_patches`) is more than just "where
  cached patches live."** It's the on/off switch for whether the model
  loads any appearance-patch signal at all (`use_patches = chemin_patches
  is not None`). In on-the-fly modes the directory it points to doesn't
  even need to exist; the flag just needs to be *set* to something.
  That's why every example below still passes
  `chemin_patches=/sandbox1/aramadan/patches`, even for on-the-fly runs.

**Worked example: cached-patch mode (CK+, fast path)**
```bash
make landmarks dataset=CK+
make patches   dataset=CK+ backbone=resnet18 \
    encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth \
    out_dir=/sandbox1/aramadan/patches_resnet18
make train dataset=CK+ graph_type=g_topologique chemin_patches=/sandbox1/aramadan/patches_resnet18
```

**Worked example: on-the-fly encoder mode (AFEW, no landmark re-detection)**
```bash
make landmarks dataset=AFEW
make bboxes    dataset=AFEW
make train dataset='AFEW' unfreeze_resnet=partial lazy_loading=--lazy_loading \
    patch_backbone=resnet18 encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth \
    PARAM_CONST="--use_local_encoder on --nb_cross_val 1 --seed 42"
```

**Worked example: on-the-fly landmarks mode (AFEW, geometry/appearance consistent)**
```bash
make landmarks dataset=AFEW
make bboxes    dataset=AFEW
make train dataset='AFEW' unfreeze_resnet=partial lazy_loading=--lazy_loading \
    patch_backbone=resnet18 encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth \
    PARAM_CONST="--use_local_encoder on --on_the_fly_landmarks --nb_cross_val 1 --seed 42"
```

### Basic training command

```bash
make train dataset='AFEW' graph_type=g_topologique num_epochs_paper=30 \
    unfreeze_resnet=full \
    patch_backbone=resnet18 \
    encoder_ckpt=/path/to/Resnet18_FER+_pytorch.pth \
    lazy_loading=--lazy_loading \
    val_split=0.0 \
    encoder_lr=1e-5 \
    PARAM_CONST="--num_workers 4 --nb_cross_val 1 \
                 --sampling_strategy uniform \
                 --use_class_weights \
                 --on_the_fly_landmarks \
                 --seed 42"
```

`dataset` and `graph_type` pick the data and the graph topology.
`unfreeze_resnet`, `patch_backbone`, `encoder_ckpt`, `encoder_lr`, and
`lazy_loading` configure the appearance encoder (see the report for the
encoder-unfreezing comparison); leave `unfreeze_resnet` empty for the fast
cached-feature path. `val_split` sets how much of the training set is held
out for checkpoint selection. `PARAM_CONST` is a catch-all passed straight
through to `train_gnn_paper.py`; here it sets worker count, fold count,
sampling strategy, class weighting, on-the-fly landmark re-detection, and
the seed.

## Command-line flags

This isn't the auto-generated argparse help. It's grouped and explained in
plain language instead. If it ever drifts out of sync, the argparse block
at the bottom of `train_gnn_paper.py` is the source of truth.

**Paths**
- `--chemin_depot_global`: root data directory (Makefile: `CHEMINS_REPERTOIRE_DONNEES`, default `/sandbox1/aramadan`). Everything else (`frames/`, `landmark/`, `patches/`, `resultats/`) is resolved relative to this.
- `--chemin_patches`: turns appearance-patch loading on or off. Leave it unset (`None`) and the model gets no patch signal at all. See "Which preprocessing steps do you need?" above; in on-the-fly modes you still need to set this to *something*, even though the path itself doesn't have to exist.

**Dataset selection**
- `--nom_dataset {CK+, Oulu, Oulu_Acropped, AFEW, eNTERFACE05, CAER, AFEW+CAER, DFEW, FERV39k}`: which dataset to train on. Must match a key in `datasets/__init__.py`'s registry. (The `Makefile`'s `dataset=` variable maps to this flag.)
- `--val_split`: how much of the training set to hold out for validation (used to pick the best checkpoint). Default `0.0` matches the paper's protocol: no held-out split, just train for a fixed number of epochs. Lazy path only.
- `--best_metric {val_acc, val_loss}`: which validation metric selects the best checkpoint. Only used when `--val_split > 0`.

**Model architecture: appearance components**
- `--global_appearance_mode {none_all, none, broadcast, roi_align, combined, roi_align_multiscale, local_plus_multiscale, fragnet}`: the old single-flag way of picking an appearance mode. Default `none_all` (off). Prefer the per-component `--use_*` flags below instead; three of the legacy values (`roi_align`, `combined`, `local_plus_multiscale`) don't have a six-flag equivalent yet and still go through the old code path.
- `--use_local_encoder {on, off}`: local 49×49 patches through the ResNet patch encoder.
- `--use_broadcast {on, off}`: global 512-d feature broadcast to all landmarks.
- `--use_multiscale_roi {on, off}`: multi-scale ROI-Align (layer2+layer3+layer4).
- `--use_fragnet_landmark {on, off}`: FragNet dual-pathway with lateral fusion, landmark-guided.
- `--use_fragnet_grid {on, off}`: grid-based FragNet (fixed vertical strips, no landmark guidance).
- `--fragnet_grid_backbone {resnet18, resnet50}`: backbone used only by grid-based FragNet. `resnet18` shares weights with the other components; `resnet50` is a separate encoder trained from ImageNet.
- `--upscale_patches_to_224 {on, off}`: scales local patches up to 224×224 before feeding them to the ResNet, matching the resolution the FER+ checkpoint was pretrained at. See the report for the full comparison of patch resolutions.

**Model architecture: geometry and structure**
- `--use_coord_encoder {on, off}`: the R/C/T coordinate branches.
- `--use_gcblocks {on, off}`: the GCBlock stack (spatial graph convolution + temporal conv).
- `--use_transformers {on, off}`: the final spatial/temporal transformer stack.
- `--graph_type {g_complet, g_topologique, g_yeux_bouche, g_bouche}`: landmark graph topology. `g_topologique` (anatomical connectivity) is the default used for all reported experiments.
- `--temporal_edges {none, combined, separated_parallel, separated_sequential, construct_only}`: how temporal edges enter the spatial-temporal graph (see the report for the comparison).
- `--tgconv_mode {learnable_A, temporal_edges}`: whether the temporal-mixing step uses the original learnable joint-adjacency matrix, or actual graph convolution on temporal edges (see the report for the comparison).

**Encoder configuration**
- `--patch_backbone {resnet18, resnet34}`: backbone for the local patch encoder.
- `--encoder_ckpt`: path to the pretrained backbone checkpoint.
- `--unfreeze_resnet {none, frozen, partial, full}`: `none` uses cached features (no in-model encoder); `frozen` puts the encoder in the model with no gradient; `partial` unfreezes `layer4` only; `full` unfreezes the whole backbone.
- `--encoder_lr`: separate learning rate for the encoder when partial/full unfreezing is active (typically 1e-5, ~10× lower than the head LR).
- `--freeze_encoder_bn {on, off}`: keeps the encoder's BatchNorm running stats frozen. Useful at small batch sizes, where every patch in a batch can come from the same video.
- `--amp {on, off}`: turns on mixed-precision training. Mainly needed together with `--upscale_patches_to_224`, where full-precision activations can run out of GPU memory even at batch size 1.
- `--encoder_imagenet_norm {true, false}`: whether the in-model patch encoder applies ImageNet mean/std normalization. **Careful, the default is broken.** The docstring says it auto-selects `False` for `resnet18`, but the actual code (`PatchEncoder.__init__`) always resolves `None` to `True`, regardless of backbone. The resnet18 checkpoint in use (`Resnet18_FER+_pytorch.pth`, from Open-Debin/Emotion-FAN) was trained without ImageNet normalization, just `ToTensor()`, values in `[0,1]`. Leaving this at its default silently corrupts every BatchNorm layer for `--patch_backbone resnet18`. Always pass `--encoder_imagenet_norm false` explicitly with the resnet18 FER+ checkpoint. See the report for how much this affected results.

**Training and reproducibility**
- `--num_epochs`: training epochs (paper default 120).
- `--batch_size`: batch size (default 8). Drop to 4 for larger architectures (FragNet modes, full unfreeze).
- `--seed`: random seed; 42 for reproducibility across our reported results.
- `--nb_cross_val`: number of cross-validation folds.
- `--lambda_center`: weight for center loss (paper uses 0.001).
- `--lr` / `--weight_decay`: head learning rate and weight decay.
- `--device {cuda:0, cpu}`: training device (default `cuda:0`).
- `--print_every`: batches between progress-log prints (default 10; use `1` for short debug runs where you want per-batch visibility).
- `--dim_feedforward`: transformer feed-forward dimension (default 1024).
- `--apply_scheduler`: **inverted, easy to misread**. It's `action='store_false'`, so the cosine-annealing scheduler is **on by default**. Passing `--apply_scheduler` on the command line turns it **off**. None of this README's example commands pass it, so they all train with the scheduler on.
- `--class_weight`: legacy alias for `--use_class_weights` (either works; the code checks `args.class_weight or args.use_class_weights`).

**Output location: `--mode_debug` (another inverted flag)**
- `--mode_debug` is also `action='store_false'`, defaulting to `True`. It doesn't change how much data is used or how the model trains, it only decides where results get written. Flag absent (default) writes to `resultats/test/<dataset>/<graph_type>/...`. Flag passed writes to `resultats/train/...` instead. None of this README's examples pass it, so they all log under `resultats/test/`. If you want a run to count as a "real" logged run instead of a scratch check, add `--mode_debug` to `PARAM_CONST`.

**Data pipeline**
- `--on_the_fly_landmarks`: re-detects landmarks with 2D-FAN on every frame during training instead of using the cached ones, so the geometry and appearance branches see the same (augmented) input. Requires on-the-fly patch loading. The cached landmarks are still used as a fallback when FAN fails to find a face, so `make landmarks` is still required; see "Which preprocessing steps do you need?" above.
- `--lazy_loading`: streaming `LazyFERDataset` instead of the eager in-RAM pipeline; required for on-the-fly encoder/landmark training and for CAER-scale datasets.
- `--num_workers`: DataLoader workers (lazy path only).
- `--sampling_strategy {top_k, uniform}`: `top_k` (paper default) picks the frames with the most motion; `uniform` picks evenly spaced frames instead. We found `uniform` works better for in-the-wild clips, where the highest-motion frame is often a scene cut rather than the peak expression.
- `--use_class_weights`: balanced cross-entropy weighting (alias: `--class_weight`).
- `--already_cropped`: treat input frames as already face-cropped, skipping bbox lookup (use with `Oulu_Acropped`).
- `--use_cached_patches`: forces cached-feature loading even if `--unfreeze_resnet partial`/`full` is set. Combined with `partial`/`full` it's a no-op for the encoder (it can't fine-tune on features it never sees raw pixels for) and prints a warning. Mainly useful with `--unfreeze_resnet frozen`, when you want the encoder present in the model but fed pre-extracted features.

**Legacy / eager-path-only flags** (not used in any reported result, kept for backwards compatibility): `--pourcentage_test`, `--nb_instances_par_classe_test`, `--mode_test_pourcentage`, `--mode_calcul_nb_frames` (eager pipeline's per-video frame-count heuristic, no effect under `--lazy_loading`), `--saved_model` (`store_true`, default already `True`; passing it is a no-op).

Not every flag applies in every mode. If you're combining flags you haven't
seen used together, check `train_gnn_paper.py`'s argparse block or the
example commands below first.

## Example commands used for the reported results

```bash
# Baseline: local patches only, AFEW-alone, 30 epochs. Result: ~38.64% WAR, + Full end-to-end encoder unfreeze.
make train dataset='AFEW' graph_type=g_topologique num_epochs_paper=30 \
    num_layers=1 num_layers_trans=1 \
    unfreeze_resnet=full \
    patch_backbone=resnet18 \
    encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth \
    lazy_loading=--lazy_loading \
    val_split=0.0 \
    encoder_lr=1e-5 \
    PARAM_CONST="--num_workers 4 --nb_cross_val 1 \
                 --sampling_strategy uniform \
                 --use_class_weights \
                 --on_the_fly_landmarks \
                 --global_appearance_mode none \
                 --seed 42"



# Multi-scale ROI-Align-multiscale appearance mode. Result: 45.17% WAR.
make train dataset='AFEW' graph_type=g_topologique num_epochs_paper=30     num_layers=1     num_layers_trans=1 unfreeze_resnet=full     patch_backbone=resnet18   encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth lazy_loading=--lazy_loading     val_split=0.0 encoder_lr=1e-5 PARAM_CONST="--num_workers 4 --nb_cross_val 1 \
                 --sampling_strategy uniform \
                 --use_class_weights \
                 --on_the_fly_landmarks \
                 --global_appearance_mode roi_align_multiscale \
                 --temporal_edges separated_parallel \
                 --seed 42"

# Best AFEW-alone: FragNet dual-pathway with separated-parallel temporal edges.
# Result: 46.48% WAR.
make train dataset='AFEW' graph_type=g_topologique num_epochs_paper=30 num_layers=1 num_layers_trans=1 unfreeze_resnet=partial patch_backbone=resnet18 encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth lazy_loading=--lazy_loading val_split=0.0 encoder_lr=1e-5 PARAM_CONST="--num_workers 4 --nb_cross_val 1 \
                 --sampling_strategy uniform \
                 --use_class_weights \
                 --on_the_fly_landmarks \
                 --use_fragnet_landmark on \
                 --temporal_edges separated_parallel \
                 --seed 42"

# Joint AFEW+CAER training, evaluated on both (domain-matched validation).
# --global_appearance_mode none (local-patch encoder) is just a sensible
# default here, not a reproduced result. Swap it for whichever component(s)
# you want to test jointly. At least one component must be on.
make train dataset='AFEW+CAER' graph_type=g_topologique num_epochs_paper=30 num_layers=1 num_layers_trans=1 unfreeze_resnet=full patch_backbone=resnet18 encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth lazy_loading=--lazy_loading val_split=0.1 encoder_lr=1e-5 PARAM_CONST="--num_workers 4 --nb_cross_val 1 \
                 --sampling_strategy uniform \
                 --use_class_weights \
                 --on_the_fly_landmarks \
                 --global_appearance_mode none \
                 --seed 42"
```

## More examples: six-flag component scheme (`--use_*` on/off)

```bash
# local_encoder only (partial unfreeze). Result: 36.03% WAR
make train dataset='AFEW' graph_type=g_topologique num_epochs_paper=30 num_layers=1 num_layers_trans=1 unfreeze_resnet=partial batch_size=8 patch_backbone=resnet18 encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth lazy_loading=--lazy_loading val_split=0.0 encoder_lr=1e-5 PARAM_CONST="--num_workers 4 --nb_cross_val 1 \
                 --sampling_strategy uniform \
                 --use_class_weights \
                 --on_the_fly_landmarks \
                 --use_coord_encoder on \
                 --temporal_edges separated_parallel \
                 --use_fragnet_landmark off \
                 --use_broadcast off \
                 --use_fragnet_grid off \
                 --use_local_encoder on \
                 --use_multiscale_roi off \
                 --seed 42"

# All six components on. Result: 38.90% WAR / 37.32% UAR.
make train dataset='AFEW' graph_type=g_topologique num_epochs_paper=30 num_layers=1 num_layers_trans=1 unfreeze_resnet=full batch_size=2 patch_backbone=resnet18 encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth lazy_loading=--lazy_loading val_split=0.0 encoder_lr=1e-5 PARAM_CONST="--num_workers 4 --nb_cross_val 1 \
                 --sampling_strategy uniform \
                 --use_class_weights \
                 --on_the_fly_landmarks \
                 --use_coord_encoder on \
                 --temporal_edges separated_parallel \
                 --use_fragnet_landmark on \
                 --use_broadcast on \
                 --use_fragnet_grid on \
                 --use_local_encoder on \
                 --use_multiscale_roi on \
                 --seed 42"

# fragnet_landmark + broadcast + fragnet_grid + multiscale_roi (no local_encoder). Result: 44.39% WAR / 41.62% UAR.
make train dataset='AFEW' graph_type=g_topologique num_epochs_paper=30 num_layers=1 num_layers_trans=1 unfreeze_resnet=full batch_size=2 patch_backbone=resnet18 encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth lazy_loading=--lazy_loading val_split=0.0 encoder_lr=1e-5 PARAM_CONST="--num_workers 4 --nb_cross_val 1 \
                 --sampling_strategy uniform \
                 --use_class_weights \
                 --on_the_fly_landmarks \
                 --use_coord_encoder on \
                 --temporal_edges separated_parallel \
                 --use_fragnet_landmark on \
                 --use_broadcast on \
                 --use_fragnet_grid on \
                 --use_local_encoder off \
                 --use_multiscale_roi on \
                 --seed 42"

# fragnet_landmark + fragnet_grid + multiscale_roi (no local_encoder, no broadcast). Result: 42.56% WAR / 40.11% UAR.
make train dataset='AFEW' graph_type=g_topologique num_epochs_paper=30 num_layers=1 num_layers_trans=1 unfreeze_resnet=full batch_size=2 patch_backbone=resnet18 encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth lazy_loading=--lazy_loading val_split=0.0 encoder_lr=1e-5 PARAM_CONST="--num_workers 4 --nb_cross_val 1 \
                 --sampling_strategy uniform \
                 --use_class_weights \
                 --on_the_fly_landmarks \
                 --use_coord_encoder on \
                 --temporal_edges separated_parallel \
                 --use_fragnet_landmark on \
                 --use_broadcast off \
                 --use_fragnet_grid on \
                 --use_local_encoder off \
                 --use_multiscale_roi on \
                 --seed 42"

# fragnet_landmark + multiscale_roi only. Result: 43.60% WAR / 41.16% UAR.
make train dataset='AFEW' graph_type=g_topologique num_epochs_paper=30 num_layers=1 num_layers_trans=1 unfreeze_resnet=full batch_size=2 patch_backbone=resnet18 encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth lazy_loading=--lazy_loading val_split=0.0 encoder_lr=1e-5 PARAM_CONST="--num_workers 4 --nb_cross_val 1 \
                 --sampling_strategy uniform \
                 --use_class_weights \
                 --on_the_fly_landmarks \
                 --use_coord_encoder on \
                 --temporal_edges separated_parallel \
                 --use_fragnet_landmark on \
                 --use_broadcast off \
                 --use_fragnet_grid off \
                 --use_local_encoder off \
                 --use_multiscale_roi on \
                 --seed 42"

# fragnet_grid + multiscale_roi only. Result: 42.04% WAR / 39.27% UAR.
make train dataset='AFEW' graph_type=g_topologique num_epochs_paper=30 num_layers=1 num_layers_trans=1 unfreeze_resnet=full batch_size=4 patch_backbone=resnet18 encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth lazy_loading=--lazy_loading val_split=0.0 encoder_lr=1e-5 PARAM_CONST="--num_workers 4 --nb_cross_val 1 \
                 --sampling_strategy uniform \
                 --use_class_weights \
                 --on_the_fly_landmarks \
                 --use_coord_encoder on \
                 --temporal_edges separated_parallel \
                 --fragnet_grid_backbone resnet18 \
                 --use_fragnet_landmark off \
                 --use_broadcast off \
                 --use_fragnet_grid on \
                 --use_local_encoder off \
                 --use_multiscale_roi on \
                 --seed 42"
```

## ResNet-18 checkpoint source

**Direct link:**

https://onedrive.live.com/?id=2D84EC2FF5DA9C11%213027&cid=2D84EC2FF5DA9C11&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaEdjMnZVdjdJUXRsMVB0N0ZoUFhyX0tvZmQ1P2U9M012UEZY

**GitHub repository:**

https://github.com/Open-Debin/Emotion-FAN
