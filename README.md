# Facial Expression Recognition from Videos: M1 research project

> This branch is the **first version** of the project, from my M1 research project.
> The later, bigger version from my internship (5 datasets, new ideas, better results)
> is on the [`main`](../../tree/main) branch.

A deep learning model that watches a short video of a face and says which of 7 emotions it shows.
It follows **68 points on the face** (eye corners, mouth, eyebrows...) and how they move over time,
plus a small image patch around each point.

The model is Graph-Tran, from a 2023 paper by Zhao et al. The authors never published their code,
so I rebuilt it from the paper alone, in PyTorch, and tested it on the CK+ dataset.
Done at the **CRIStAL lab (Université de Lille)**, supervised by Tanmoy Mondal and Deise Santana Maia.
The full report is in [`Report/research_project.pdf`](Report/research_project.pdf).

**In short:** Python, PyTorch, PyTorch Geometric, graph neural networks (GCN), transformers,
ResNet-34, OpenCV, 2D-FAN face points.

## Result

**92.98% accuracy on CK+** (± 4.69% between folds), against 98.78% in the paper.

The test is strict: the data is split in 10 parts by person, so the model is always tested on
people it has never seen during training.

### Which face points matter most?

| Points used | Number of points | Accuracy |
|---|---|---|
| Eyes, eyebrows and mouth | 42 | **92.98%** |
| Eyes, eyebrows, nose and mouth (no jaw) | 51 | 92.95% |
| All 68 points | 68 | 92.36% |
| Mouth only | 20 | 88.00% |

The eyes and eyebrows carry a lot of the signal: with the mouth alone, accuracy drops by 5 points.
Adding the jaw does not help.

### How the accuracy went up during the project

| Version | Accuracy |
|---|---|
| Face points only (MediaPipe detector) | 90.5% |
| + image patches around each point | 88.38% |
| Full paper model (MediaPipe, graph fixed) | 92.35% |
| Full paper model, 2D-FAN detector, edges fixed | **92.98%** |
| Original paper | 98.78% |

## Why it's below the paper

The paper changes every training image a little at each epoch (random flips and crops) and then
finds the face points and patches again on the changed image. To save time, I extracted the patches
once and stored them, so this kind of change was not possible anymore. This explains most of the gap.
(I added it during the internship: see the `main` branch.)

Also, each test fold has only about 33 videos, so one mistake costs 3% on that fold.

## Two bugs I found and fixed

1. **Only the first frame had graph edges.** The connections between face points were created for
   frame 1 only, so frames 2 to 16 had none: the graph convolution was blind to 94% of the video.
   Fixing it was the biggest single improvement.
2. **The positions were normalized twice.** The preprocessing already centered and scaled the points,
   and the graph step did it again, which distorted them.

## How it works

1. **Find the face** in each frame and crop it (240×240).
2. **Find 68 face points** with the 2D-FAN detector.
3. **Cut a 49×49 patch** around each point and turn it into features with a ResNet-34 trained on faces (FER+).
4. **Build a graph**: face points are nodes, points of the same face part are connected,
   and the same structure is repeated over 16 frames.
5. **The model** reads the point positions in three ways (position relative to the nose, position
   inside its face part, movement since the last frame), mixes them with the patch features, then uses
   graph convolutions, attention over face parts and frames, and transformers to predict the emotion.

## Data

No data is in this repo. CK+ is available on request from its authors
([jeffcohn.net/Resources](http://www.jeffcohn.net/Resources/)).

## Running it

The code was written for the lab's GPU server, so the paths in `src/Makefile` point there.
Change them to your own folders first.

```bash
cd src
pip install -r requirements.txt

make data_exp_faciale                                   # find the face points
python extract_patches.py                               # patch features
make train_exp_faciale_paper graph_type=g_yeux_bouche   # train + 10-fold test
```

[`docs/LAB_GUIDE.md`](docs/LAB_GUIDE.md) is the detailed guide I wrote for the lab: every file, the
full pipeline, and all the settings.

## Files

```
src/train_gnn_paper.py            training script
src/models/gnn_paper.py           the model
src/executor/trainer_paper.py     training loop (cross-entropy + center loss)
src/dataloader/                   building the graphs, frame sampling
src/preprocessing_exp_faciale.py  face points extraction
src/extract_patches.py            patch features extraction
g_*_results/                      plots and settings for each set of face points
Report/research_project.pdf       research project report
docs/LAB_GUIDE.md                 detailed lab guide
```

## Credits

- Original method: R. Zhao, T. Liu, Z. Huang, D.P.K. Lun, K.M. Lam, *Spatial-Temporal Graphs Plus
  Transformers for Geometry-Guided Facial Expression Recognition*, IEEE Transactions on Affective
  Computing, 14(4), 2023.
- Face point detector: 2D-FAN, by Bulat and Tzimiropoulos (ICCV 2017).
- CK+ dataset: Lucey et al. (CVPR Workshops 2010).
