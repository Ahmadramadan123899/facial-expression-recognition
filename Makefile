# Author : Ahmad Ramadan
# ======================================================================================================================================
# Définition des variables
# ======================================================================================================================================
VENV_DIR = /sandbox1/aramadan/envs/gnn_env/bin/python3
.PHONY = data_exp_faciale extract_patches train_exp_faciale_paper clean_result

# Chemins
CHEMINS_REPERTOIRE_DONNEES = /sandbox1/aramadan
chemin_patches              = /sandbox1/aramadan/patches
# For ResNet-18 cached features use: chemin_patches=/sandbox1/aramadan/patches_resnet18
RESULT_PATH                 = /sandbox1/aramadan/resultats

# ── Dataset — override on command line: make train dataset=Oulu ──────────────
dataset          = CK+
already_cropped  =           # set to --already_cropped for pre-cropped datasets (e.g. Oulu)

# Paramètres d'entraînement
PARAM_CONST       = --mode_debug --nb_cross_val 10
batch_size        = 8
n_heads           = 8
num_layers_trans  = 2
num_layers        = 2
graph_type        = g_topologique
facteur           = 10e4
class_weight_flag =

# Paramètres paper-alignés
lambda_center    = 0.001
nb_frames_paper  = 16
lr_paper         = 1e-4
num_epochs_paper = 120
dropout_transformers=0.1
weight_decay=1e-4

# ── Patch encoder (on-the-fly ResNet fine-tuning) ──────────────────────────
# unfreeze_resnet: leave empty for cached-patch mode (default), or set to:
#   partial  → only layer4 trainable  (recommended, lower memory)
#   full     → all backbone params trainable
#   frozen   → backbone in model but no gradient (sanity check)
# patch_backbone: resnet34 (FER+ pretrained, default) or resnet18 (ImageNet)
# encoder_ckpt:   path to the backbone checkpoint file
# lazy_loading:   set to --lazy_loading when using on-the-fly mode
# encoder_lr:     learning rate for unfrozen backbone layers (default 1e-5)
# val_split:      fraction of train set held out for validation (default 0.1)
unfreeze_resnet  =
patch_backbone   = resnet34
encoder_ckpt     = /sandbox1/aramadan/resnet34-epoch.12-val_loss.0.494-val_acc.0.846-val_f1.0.843.ckpt
lazy_loading     =
already_cropped  =           # set to --already_cropped for pre-cropped datasets (e.g. Oulu_Acropped)
encoder_lr       = 1e-5
val_split        = 0.1

# ======================================================================================================================================
# Commandes
# ======================================================================================================================================

### STEP 1 — Extract 2D-FAN landmarks
### Usage: make landmarks dataset=Oulu
###        make landmarks dataset=CK+
landmarks:
	$(VENV_DIR) preprocessing_exp_faciale.py \
		--chemin_depot_global $(CHEMINS_REPERTOIRE_DONNEES) \
		--nom_dataset "$(dataset)" \
		--face_landmarks_extraction \
		--resume \
		$(already_cropped)

### STEP 1b — Precompute face bounding boxes (needed for on-the-fly ResNet mode)
### Skip for already-cropped datasets (Oulu_Acropped) — use --already_cropped there.
### Always run on CPU (CUDA_VISIBLE_DEVICES="") to avoid competing with training jobs.
### Usage: make bboxes dataset=CK+
###        make bboxes dataset=eNTERFACE05
bboxes:
	CUDA_VISIBLE_DEVICES="" $(VENV_DIR) precompute_bboxes.py \
		--chemin_depot_global $(CHEMINS_REPERTOIRE_DONNEES) \
		--nom_dataset "$(dataset)" \
		--resume

### STEP 2 — Extract patch features (ResNet-34 or ResNet-18)
### ResNet-34 FER+ (default, keeps existing patches):
###   make patches dataset=CK+
###   make patches dataset=Oulu already_cropped=--already_cropped
### ResNet-18 FER+ (saves to patches_resnet18/, does not overwrite ResNet-34):
###   make patches dataset=CK+ backbone=resnet18 \
###       encoder_ckpt=/sandbox1/aramadan/Resnet18_FER+_pytorch.pth \
###       out_dir=/sandbox1/aramadan/patches_resnet18
backbone   =        # leave empty for default (resnet34)
out_dir    =        # leave empty for default ({chemin_depot_global}/patches)
patches:
	$(VENV_DIR) extract_patches.py \
		--chemin_depot_global $(CHEMINS_REPERTOIRE_DONNEES) \
		--nom_dataset "$(dataset)" \
		$(if $(already_cropped),--already_cropped) \
		$(if $(backbone),--backbone $(backbone)) \
		$(if $(encoder_ckpt),--encoder_ckpt "$(encoder_ckpt)") \
		$(if $(out_dir),--out_dir "$(out_dir)")

### STEP 3 — Train Graph-Tran
###
### Cached-patch mode (default, fast):
###   make train dataset=CK+ graph_type=g_topologique
###   make train dataset=AFEW graph_type=g_topologique
###
### On-the-fly ResNet fine-tuning (needs --lazy_loading):
###   make train dataset=AFEW unfreeze_resnet=partial lazy_loading=--lazy_loading
###   make train dataset=AFEW unfreeze_resnet=partial lazy_loading=--lazy_loading \
###              patch_backbone=resnet18 encoder_ckpt=/sandbox1/aramadan/resnet18_msceleb.pth
###
### Joint AFEW+CAER (train once, evaluate on both):
###   make train dataset='AFEW+CAER' unfreeze_resnet=partial lazy_loading=--lazy_loading \
###              PARAM_CONST="--nb_cross_val 1"
train:
	$(VENV_DIR) train_gnn_paper.py \
		--lr $(lr_paper) \
		--chemin_depot_global $(CHEMINS_REPERTOIRE_DONNEES) \
		--nom_dataset "$(dataset)" \
		--graph_type $(graph_type) \
		$(PARAM_CONST) \
		--nb_frames $(nb_frames_paper) \
		--batch_size $(batch_size) \
		--n_heads $(n_heads) \
		--num_layers_trans $(num_layers_trans) \
		--num_layers $(num_layers) \
		--num_epochs $(num_epochs_paper) \
		--facteur $(facteur) \
		--lambda_center $(lambda_center) \
		--chemin_patches $(chemin_patches) \
		--dropout_transformers $(dropout_transformers) \
		--weight_decay $(weight_decay) \
		--patch_backbone $(patch_backbone) \
		--encoder_ckpt "$(encoder_ckpt)" \
		--encoder_lr $(encoder_lr) \
		--val_split $(val_split) \
		$(if $(unfreeze_resnet),--unfreeze_resnet $(unfreeze_resnet)) \
		$(lazy_loading) \
		$(if $(already_cropped),--already_cropped) \
		$(class_weight_flag)

# Legacy target name kept so existing commands still work
train_exp_faciale_paper: train
data_exp_faciale: landmarks

### NETTOYAGE
clean_result:
	rm -i -rf $(RESULT_PATH)/*


### STEP 0 — AFEW/CAER/eNTERFACE05: extract JPEG frames from .avi files
### (CK+/Oulu ship as frames already — no organize_* step needed for them)
### downsample/overwrite are shared across all three organize_* targets below.
### Usage: make organize_afew         [downsample=32]
###        make organize_caer         [downsample=32]
###        make organize_enterface05  [downsample=32]
###        make organize_afew overwrite=--overwrite   # re-extract existing videos
afew_root       = /data/fox-data/datasets/emotions/AFEW5.0
caer_root       = /sandbox1/aramadan/CAER
enterface_root  = /sandbox1/aramadan/eNTERFACE05/enterface database
downsample =                # leave empty to keep all frames, e.g. downsample=--downsample 32
overwrite  =                # leave empty to skip already-extracted videos, or set to --overwrite

organize_afew:
	$(VENV_DIR) organize_afew.py \
	    --afew_root $(afew_root) \
	    --out_root  $(CHEMINS_REPERTOIRE_DONNEES)/frames/AFEW \
	    $(downsample) \
	    $(overwrite)

organize_caer:
	$(VENV_DIR) organize_caer.py \
	    --caer_root $(caer_root) \
	    --out_root  $(CHEMINS_REPERTOIRE_DONNEES)/frames/CAER \
	    $(downsample) \
	    $(overwrite)

organize_enterface05:
	$(VENV_DIR) organize_enterface05.py \
	    --enterface_root "$(enterface_root)" \
	    --out_root  $(CHEMINS_REPERTOIRE_DONNEES)/frames/eNTERFACE05 \
	    $(downsample) \
	    $(overwrite)
