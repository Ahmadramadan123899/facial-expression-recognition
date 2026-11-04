# Author : Ahmad Ramadan
# ======================================================================================================================================
# Définition des variables
# ======================================================================================================================================
VENV_DIR = /sandbox1/aramadan/envs/gnn_env/bin/python3
.PHONY = data_exp_faciale train_exp_faciale_paper clean_result

# Chemins
CHEMINS_REPERTOIRE_DONNEES = /sandbox1/aramadan
chemin_patches              = /sandbox1/aramadan/patches
DATASET_NORMAL              = "CK+"
RESULT_PATH                 = /sandbox1/aramadan/resultats

# Paramètres d'entraînement
PARAM_CONST       = --mode_debug --nb_cross_val 10 --pourcentage_test 0.8
batch_size        = 16
n_heads           = 8
num_layers_trans  = 2
num_layers        = 2
graph_type        = g_topologique
facteur           = 10e4
class_weight_flag =

# Paramètres paper-alignés
lambda_center    = 0.05
nb_frames_paper  = 16
lr_paper         = 1e-4
num_epochs_paper = 120

# ======================================================================================================================================
# Commandes
# ======================================================================================================================================

### PREPROCESSING — extract 2D-FAN landmarks
data_exp_faciale:
	$(VENV_DIR) preprocessing_exp_faciale.py \
		--chemin_depot_global $(CHEMINS_REPERTOIRE_DONNEES) \
		--nom_dataset $(DATASET_NORMAL)

### PAPER-ALIGNED MODEL (full Graph-Tran architecture)
train_exp_faciale_paper:
	$(VENV_DIR) train_gnn_paper.py \
		--lr $(lr_paper) \
		--chemin_depot_global $(CHEMINS_REPERTOIRE_DONNEES) \
		--nom_dataset $(DATASET_NORMAL) \
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
		$(class_weight_flag)

### NETTOYAGE
clean_result:
	rm -i -rf $(RESULT_PATH)/*