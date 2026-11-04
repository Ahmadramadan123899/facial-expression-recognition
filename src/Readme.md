Hi! This is the repository for my facial expression GNN project.
Please use your respective dev- branches for edits.

# Facial Expression Recognition — Graph-Tran Reimplementation

A reimplementation of the **Graph-Tran** framework from:

> Zhao et al., *"Spatial-Temporal Graphs Plus Transformers for Geometry-Guided Facial Expression Recognition"*, IEEE Transactions on Affective Computing, Vol. 14, No. 4, 2023.

Evaluated on the **CK+ (Extended Cohn-Kanade)** dataset using 10-fold identity-independent cross-validation.

**Best result achieved: 92.95% ± 4.95% mean accuracy** on CK+ with the `g_topologique` graph configuration.  (also 92.98% on g_yeux_bouche...)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Directory Structure](#directory-structure)
- [Key Files](#key-files)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Differences from the Original Paper](#differences-from-the-original-paper)

---

## Project Overview

This project implements a geometry-guided facial expression recognition system that models facial videos as spatial-temporal graphs. Instead of using raw appearance (CNN on full face images), the method builds a graph from facial landmarks and local patches centred at those landmarks, then learns discriminative features using:

- **Graph Convolutional Networks (GCN)** for local spatial feature aggregation
- **Transformers** for global non-local dependency across joints and frames
- **Spatial component attention** to focus on the most expressive facial regions
- **Temporal frame attention** to focus on the most informative frames in a video
- **Center Loss** combined with Cross-Entropy for more compact class representations

Three coordinate representations are computed from the raw landmarks and each processed by an independent GCN branch:

- **R (Relative position)** — position of each landmark relative to the nose tip, normalised by the inter-eye distance (paper Eq. 1–2)
- **C (Component offset)** — offset of each landmark from its facial component centre (mouth, eye, brow, nose), describing component self-deformation (paper Eq. 3)
- **T (Temporal motion)** — frame-to-frame displacement, capturing facial dynamics (paper Eq. 4)

---

## Architecture

```
Input video (16 frames)
    │
    ▼
[Preprocessing]
    ├── 2D-FAN landmark detector → 68 landmarks per frame
    ├── 240×240 face crop (Haar cascade)
    └── Coordinate normalisation: R = (x_i - x_nose) / (2 * S)

[Graph Construction]
    ├── Landmark subset selection (g_topologique: 51 joints)
    ├── Edge definition (2D-FAN standard connectivity)
    └── 49×49 patch extraction at each landmark

[Coordinate Encoder]  ×3 independent branches
    └── CoordBranch: GCN(2→64→128→256) + Temporal Conv (k=5)
        ├── Branch R → 256-dim
        ├── Branch C → 256-dim
        └── Branch T → 256-dim
        → concat: g = 768-dim

[Patch Encoder]
    └── ResNet-34 pre-trained on FER+ → 512-dim per landmark

[Early Fusion]
    └── u = [g, v] = 1280-dim → Linear(1280→512)

[Main Processor]  ×2 blocks
    ├── Spatial GCN (512→512)
    ├── Component Attention (7 facial clusters)
    ├── Temporal Conv (k=5)
    ├── Frame Attention
    ├── Spatial Transformer (8 heads)
    └── Temporal Transformer (8 heads)

[Classifier]
    └── GlobalMeanPool → FC(512→256) → FC(256→7)

Loss = CrossEntropy + 0.001 × CenterLoss
```

---

## Dataset

**Extended Cohn-Kanade (CK+)**

- 327 labelled video sequences from 118 subjects
- 7 emotion classes: Anger, Contempt, Disgust, Fear, Happiness, Sadness, Surprise
- Each video starts from a neutral face and ends at the peak expression
- Evaluation: **10-fold identity-independent cross-validation** using `GroupKFold`
  - No subject appears in both train and test within any fold
  - No validation set (paper protocol) — test evaluated once after training

| Class | Count |
|---|---|
| Happiness (0) | 69 |
| Fear (1) | 25 |
| Surprise (2) | 83 |
| Anger (3) | 45 |
| Disgust (4) | 59 |
| Sadness (5) | 28 |
| Contempt (6) | 18 |

---

## Directory Structure

```
STGFormer/src/
│
├── Makefile                          # Build commands for all experiments
│
├── train_gnn_paper.py                # Main training script (paper-aligned model)
│
│
├── dataloader/
│   ├── build_dataset_paper.py        # Data loading, graph construction, augmentation
│   ├── constants.py                  # Landmark index subsets (2D-FAN)
│   ├── MovementExtender.py           # Spline interpolation for short videos
│   └── MovementSampler.py            # Frame sampling for long videos
│
├── models/
│   └──gnn_paper.py                  # Graph-Tran full architecture (2D-FAN version)
│
├── executor/
│   └──trainer_paper.py              # Training loop with center loss 
│
├── preprocessing_exp_faciale.py      # Step 1: Extract 2D-FAN landmarks from video frames
├── extract_patches.py                # Step 2: Extract ResNet-34 FER+ patch features
│
├── results.py                        # Save per-fold results and confusion matrices
└── utils.py                          # Pickle I/O and directory utilities
```

**Data directories** (on the server):

```
/sandbox1/aramadan/
├── frames/CK+/{subject}/{video}/     # Raw video frames (PNG/JPG)
├── landmark/CK+/{subject}/           # Extracted landmarks (*.pkl)
│   └── {video}_data.pkl              # dict {frame_id: np.array(136,)} — 68 lm × 2 coords
├── patches/CK+/{subject}/            # Extracted patch features (*.pkl)
│   └── {video}_patches.pkl           # dict {frame_id: np.array(68, 512)} — float16
└── labels/
    ├── labels.pkl                    # dict {subject: {video: label}}
    └── subject_ids.pkl               # np.array of subject IDs for GroupKFold
└── resultats/
    └── train/
        └── {graph_type}/
            ├── {timestamp}/
            │   ├── commandline_args.txt      # All hyperparameters used for this run
            │   ├── fold_1_results.pkl        # Per-fold predictions and history
            │   ├── fold_2_results.pkl
            │   └── ...
            ├── results.txt                   # All runs appended — accuracy, precision, recall
            └── sort_results.txt              # Sorted by accuracy (best run at top)
```

---

## Key Files

### `preprocessing_exp_faciale.py`

Extracts facial landmarks from raw video frames. For each frame it:

1. Crops a 240×240 face region using the OpenCV Haar cascade detector (with 20% margin)
2. Runs the **2D-FAN** detector (Bulat & Tzimiropoulos, ICCV 2017) to get 68 standard landmarks
3. Normalises coordinates: `R_i = (x_i - x_nose) / (2 * S)` where S is the inter-eye distance
4. Saves a dict `{frame_id (1-based): np.array(136,)}` per video as a pickle file

Run with:
```bash
python3 preprocessing_exp_faciale.py --face_landmarks_extraction
```

---

### `extract_patches.py`

Extracts local appearance features at each landmark position. For each frame it:

1. Crops 240×240 face region (same Haar cascade as preprocessing)
2. Runs 2D-FAN to get 68 landmarks on the crop
3. Crops a 49×49 grayscale patch centred at each landmark
4. Passes all patches through a **ResNet-34 pre-trained on FER+** (frozen, fc replaced by Identity)
5. Saves a dict `{frame_id: np.array(68, 512, dtype=float16)}` per video

Run with:
```bash
python3 extract_patches.py
```

The ResNet-34 FER+ checkpoint is at:
```
/sandbox1/aramadan/resnet34-epoch.12-val_loss.0.494-val_acc.0.846-val_f1.0.843.ckpt
```

---

### `dataloader/build_dataset_paper.py`

The core data pipeline for the paper-aligned model. Key functions:

**`extract_datas_video_labels()`** — loads all landmark dicts, patch dicts, and labels from disk.

**`convertion_graphe()`** — filters landmarks and patches to the graph-specific joint subset (e.g. `g_topologique` keeps 51 of the 68 landmarks, discarding the jaw). Returns `(nb_joints, landmarks, patches, edges, kept_indices)`. The `kept_indices` list is passed to the model to compute correct cluster assignments.

**`creer_graphe_yeux_nez_bouche()`** (`g_topologique`) — builds the graph using landmarks 17–67 (both eyebrows, nose, both eyes, mouth — no jaw). Edges follow the standard 2D-FAN 68-point connectivity.

**`create_graph()`** — converts a video tensor into a PyTorch Geometric `Data` object. Two critical properties:
  - Coordinates are passed **as-is** (already R-normalised by preprocessing — no second normalisation)
  - Spatial edges are **replicated for all 16 frames** and made bidirectional, so the spatial GCN can operate across the full temporal sequence

**`data_augmentation_equilibration_classe()`** — oversamples minority classes to balance training. Coordinate noise is added to augmented samples; augmented patches receive small Gaussian noise (level sampled from [0, 0.05]) to simulate the paper's patch augmentation.

---

### `models/gnn_paper.py`

The full Graph-Tran architecture adapted for 2D-FAN 68-point landmarks. Key components:

**Cluster definitions** — landmarks 17–21 = left brow, 22–26 = right brow, 27–35 = nose, 36–41 = left eye, 42–47 = right eye, 48–67 = mouth, 0–16 = jaw/other. Used by component attention and the C representation.

**`CoordBranch`** — 3-layer spatial GCN (2→64→128→256) followed by a 1-D residual temporal convolution (kernel size 5). Three independent branches process R, C, and T respectively.

**`ComponentAttention`** — clusters joints into 7 facial components, computes mean cluster features, applies channel average+max pooling, then a 1×1 conv + softmax to generate per-cluster attention weights. Mapped back to individual joints and applied as a residual rescaling.

**`FrameAttention`** — spatial average pool + max pool across joints and channels, 1×1 conv + softmax over frames, applied as a residual rescaling.

**`GCBlock`** — one main processor block: Spatial GCN → Component Attention → Temporal Conv → Frame Attention.

**`SpatialTransformerBlock`** — Pre-LN multi-head self-attention across all joints within each frame (bs×T, P, C tokens).

**`TemporalTransformerBlock`** — Pre-LN multi-head self-attention across all frames for each joint (bs×P, T, C tokens).

**`CenterLoss`** — penalises the L2 distance between feature vectors and their class centres. Centres are learnable parameters with their own Adam optimizer (lr=0.5).

**Channel flow:**
```
R/C/T coords → 3 × CoordBranch → g (768-dim)
Patches → ResNet-34 FER+ GAP → v (512-dim)
u = [g, v] = 1280-dim
input_proj(1280 → 512)
2 × (GCBlock → SpatialTrans → TemporalTrans) at 512-dim
GlobalMeanPool → pre_logit(512→256) → head(256→7)
```

---

### `executor/trainer_paper.py`

Training loop implementing the paper protocol:

- No validation set — test set evaluated **once** after all 120 epochs
- Loss = `CrossEnttropyLoss + 0.001 × CenterLoss`
- Two optimizers: `Adam(model, lr=1e-4, weight_decay=1e-4)` and `Adam(center_loss.centers, lr=0.5)`
- `CosineAnnealingLR` scheduler: lr decays from 1e-4 to 1e-6 over 120 epochs
- Gradient clipping: `max_norm=5.0`

---

### `dataloader/constants.py`

Defines landmark index subsets for both the 2D-FAN 68-point standard (new pipeline) and the legacy MediaPipe 478-point format (old pipeline).

**2D-FAN subsets used:**
- `LIST_FAN_TOPOLOGIQUE` — 51 points (brows + eyes + nose + mouth, no jaw) — **best result**
- `LIST_FAN_COMPLET` — all 68 points
- `LIST_FAN_EYES_MOUTH` — 42 points (brows + eyes + mouth)
- `LIST_FAN_LIPS` — 20 points (mouth only)

---

### `Makefile`

Key targets:

```makefile
# Run the full paper-aligned model (best configuration)
make train_exp_faciale_paper graph_type=g_topologique

# Run all graph types
make train_exp_faciale_paper graph_type=g_complet
make train_exp_faciale_paper graph_type=g_yeux_bouche
make train_exp_faciale_paper graph_type=g_bouche

# Extract landmarks (run once)
make data_exp_faciale

# Run baseline model
make train_exp_faciale graph_type=g_topologique
```

**Key hyperparameters:**

| Parameter | Value | Note |
|---|---|---|
| `lr_paper` | 1e-4 | Adam learning rate |
| `num_epochs_paper` | 120 | Cosine annealing over full run |
| `lambda_center` | 0.05 | Center loss weight |
| `nb_frames_paper` | 16 | Uniform temporal sampling |
| `batch_size` | 16 | Mini-batch size |
| `n_heads` | 8 | Transformer attention heads |
| `num_layers` | 2 | Number of GC blocks |
| `num_layers_trans` | 2 | Number of transformer pairs |
| `dropout_transformers` | 0.1 | Dropout probability |
| `facteur` | 10e4 | Controls coordinate noise std during augmentation |

---

## Pipeline

The full pipeline runs in three sequential steps:

```
Step 1 — Landmark extraction
    python3 preprocessing_exp_faciale.py --face_landmarks_extraction
    → /sandbox1/aramadan/landmark/CK+/{subject}/{video}_data.pkl

Step 2 — Patch feature extraction
    python3 extract_patches.py
    → /sandbox1/aramadan/patches/CK+/{subject}/{video}_patches.pkl

Step 3 — Training and evaluation
    make train_exp_faciale_paper graph_type=g_topologique
    → Results printed per fold + final 10-fold summary
```

Steps 1 and 2 only need to be run once. Step 3 can be repeated with different hyperparameters without re-extraction.

---

## Installation

```bash
# Create environment
conda create -n gnn_env python=3.9
conda activate gnn_env

# Core dependencies
pip install torch torchvision
pip install torch-geometric
pip install face-alignment
pip install opencv-python
pip install scikit-learn pandas tqdm networkx matplotlib

# Installation
pip install -r requirements.txt
```
Note that torch-geometric sometimes needs to be installed separately depending on your CUDA version. If someone has issues, they should follow the official PyG install guide at https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html and match the torch + CUDA versions exactly.
```

The environment used in this project is at:
```
/sandbox1/aramadan/envs/gnn_env/
```


---

## Usage

### Full training run (paper-aligned, best configuration)

```bash
cd ~/Reconnaissance_Actions_Gestes-main/STGFormer/src
make train_exp_faciale_paper graph_type=g_topologique
```

### Custom hyperparameters

```bash
/sandbox1/aramadan/envs/gnn_env/bin/python3 train_gnn_paper.py \
    --lr 1e-4 \
    --chemin_depot_global /sandbox1/aramadan \
    --nom_dataset CK+ \
    --graph_type g_topologique \
    --mode_debug \
    --nb_cross_val 10 \
    --nb_frames 16 \
    --batch_size 8 \
    --n_heads 8 \
    --num_layers_trans 2 \
    --num_layers 2 \
    --num_epochs 120 \
    --lambda_center 0.05 \
    --chemin_patches /sandbox1/aramadan/patches
```

---

## Output and Results

After each training run, results are automatically saved to:
```
/sandbox1/aramadan/resultats/train/{graph_type}/{timestamp}/
```
Each run folder contains the commandline arguments used (`commandline_args.txt`) and per-fold result files generated by `results.py`. The aggregated `results.txt` file accumulates across all runs of the same graph type, and `sort_results.txt` keeps them ranked by mean accuracy for easy comparison.

## Results

### 10-fold Identity-Independent Cross-Validation on CK+

**Best configuration: `g_topologique`, dropout=0.1, λ=0.001, lr=1e-4, 120 epochs**

| Fold | Subjects tested | Accuracy |
|---|---|---|
| 1 | S011, S014, S029, S079, ... | 96.97% |
| 2 | S056, S070, S071, S078, ... | 93.94% |
| 3 | S022, S026, S052, S055, ... | 96.97% |
| 4 | S032, S034, S045, S068, ... | 96.97% |
| 5 | S035, S046, S053, S058, ... | 93.94% |
| 6 | S037, S059, S064, S083, ... | 87.88% |
| 7 | S005, S044, S051, S057, ... | 87.88% |
| 8 | S054, S061, S062, S077, ... | 93.75% |
| 9 | S010, S065, S074, S081, ... | 87.50% |
| 10 | S028, S042, S050, S060, ... | 93.75% |
| **Mean** | | **92.95%** |
| **Std** | | **3.64%** |

## Experiment Results - g_topologique

The folder `g_topologique_results/` contains the full output of its best run:

| File | Description |
|---|---|
| `commandline_args.txt` | Exact hyperparameters used for the run |
| `GCNTransformerPaper_accuracy.png` | Per-fold and mean accuracy curve across training |
| `GCNTransformerPaper_precision.png` | Weighted precision per fold |
| `GCNTransformerPaper_recall.png` | Weighted recall per fold |
| `sum_confusion_matrix_final_test.png` | Aggregated confusion matrix over all 10 test folds |
| `plots/` | Additional per-fold training loss and accuracy plots |

**Final result: Mean Accuracy = 92.95%% ± 3.64%** on CK+ with 10-fold identity-independent cross-validation.

### Ablation — Graph type comparison

| Graph type | Joints | Mean Accuracy | Std |
|---|---|---|---|
| `g_yeux_bouche` | 42 | **92.98%** | 4.77% |
| `g_topologique` | 51 | **92.95%** | 3.64% |
| `g_complet` | 68 | 92.36% | 5.29% |
| `g_bouche` | 20 | 92.36% | 4.00% |

### Progression across development stages

| Model | Mean Acc | Note |
|---|---|---|
| Coordinates only (baseline) | 89.26% | Simple GCN + Transformer |
| + ResNet-34 FER+ patches  | 92.98% | Full paper architecture |
| Paper (Graph-Tran original) | 98.78% | Online spatial augmentation |

---

## Differences from the Original Paper

The original paper reports **98.78%** on CK+. The gap to our 92.98% comes from one primary difference:

**Online image augmentation** — the paper applies random horizontal flips and random 224×224 crops to the raw 240×240 face images during every training epoch, then re-extracts landmarks and patches from the augmented image on the fly. This provides much richer data augmentation and effectively doubles the number of unique training samples seen per epoch.

In our implementation, landmark and patch features are **pre-extracted offline** and stored on disk for efficiency. This means spatial augmentation (flips, crops) cannot be applied retroactively to the saved 512-dim feature vectors. The only augmentation available at training time is coordinate noise (Gaussian) and small feature noise on augmented patch copies.

Other minor differences:
- We use **ResNet-34** for the patch encoder; the paper uses **ResNet-18** (similar capacity, negligible impact)
- We use a **single GPU**; the paper used two RTX 3090s (batch dynamics differ slightly)
- The paper's `batch_size=8` was originally intended; some runs used `batch_size=16` in this project

---

## References

- Zhao, R., Liu, T., Huang, Z., Lun, D.P.K., & Lam, K.M. (2023). Spatial-Temporal Graphs Plus Transformers for Geometry-Guided Facial Expression Recognition. *IEEE Transactions on Affective Computing*, 14(4), 2751–2767.
- Bulat, A., & Tzimiropoulos, G. (2017). How far are we from solving the 2D & 3D face alignment problem? *ICCV 2017*.
- Barsoum, E., Zhang, C., Ferrer, C.C., & Zhang, Z. (2016). Training deep networks for facial expression recognition with crowdsourced label distribution. *ACM ICMI 2016*.
- Lucey, P., Cohn, J.F., Kanade, T., et al. (2010). The Extended Cohn-Kanade Dataset (CK+). *CVPR Workshops 2010*.
