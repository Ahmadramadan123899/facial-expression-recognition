#! /usr/bin/python3
# Author : Ahmad Ramadan

#   convertion_graphe() returns 5 values instead of 4: 
# (nb_joints, landmarks, patches, edges, kept_indices)
#   The kept_indices are needed by GCNTransformerPaper to compute
#   cluster assignments for component attention and R/C normalisation.

import os
import torch
import numpy as np
import pickle as pkl

from tqdm import tqdm
from typing import *
import statistics
from dataloader.MovementExtender import *
from dataloader.MovementSampler import *
from dataloader.constants import *

from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt


# Gestion des fichiers

def load_pickle(filename: str):
    with open(filename, 'rb') as handle:
        data = pkl.load(handle)
    return data


def save_pickle(filename: str, data) -> None:
    with open(filename, 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)


def make_dirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# Extraction des données

def extract_datas_video_labels(path_data, path_labels, path_patches=None,
                                min_valid_frames=2):
    """
    Load landmarks (and optional patches) for every (subject, video) listed
    in path_labels. Drops videos that have fewer than `min_valid_frames`
    valid frames after filtering 2D-FAN failures.

    Returns
    -------
    all_videos_landmark : list of dict {frame_id: np.array(136,)}
    all_videos_patches  : list of dict {frame_id: np.array(68, 512)} or list of None
    nb_frames_par_video : list[int]
    labels_liste        : list[int]
    loaded_samples      : list of (subject, video, label)
        Same order as the lists above. Train script uses this to build the
        index mapping so skipped videos don't cause off-by-one errors.
    """
    labels = load_pickle(path_labels)

    labels_liste        = []
    all_videos_landmark = []
    all_videos_patches  = []
    nb_frames_par_video = []
    loaded_samples      = []
    skipped             = 0

    for sujet_id in labels.keys():
        for video_id in labels[sujet_id].keys():
            lm_path = os.path.join(path_data, sujet_id, video_id + '_data.pkl')
            if not os.path.exists(lm_path):
                print(f"  [skip] no landmark pkl for {sujet_id}/{video_id}")
                skipped += 1
                continue

            dictionnaire_video = load_pickle(lm_path)

            # Filter frames where 2D-FAN failed (stored as empty dict by
            # preprocessing_exp_faciale.py).
            dictionnaire_video = {
                k: v for k, v in dictionnaire_video.items()
                if not (isinstance(v, dict) and len(v) == 0)
            }

            # Drop the entire video if it has too few valid frames.
            # Spline interpolation in MovementExtender needs >= 2 frames.
            if len(dictionnaire_video) < min_valid_frames:
                print(f"  [skip] {sujet_id}/{video_id} -- only "
                      f"{len(dictionnaire_video)} valid frames "
                      f"(< {min_valid_frames})")
                skipped += 1
                continue

            # Load matching patches, aligned to valid landmark frames
            patches_for_video = None
            if path_patches is not None:
                patch_path = os.path.join(path_patches, sujet_id,
                                          video_id + '_patches.pkl')
                if os.path.exists(patch_path):
                    raw_patches = load_pickle(patch_path)
                    # Keep only patches whose frame_id has a valid landmark
                    patches_for_video = {
                        k: v for k, v in raw_patches.items()
                        if k in dictionnaire_video
                    }
                    # If patch extraction failed on more frames than
                    # landmark extraction did, pad missing keys with zeros
                    # so per-frame indexing in convertion_graphe stays safe.
                    for k in dictionnaire_video.keys():
                        if k not in patches_for_video:
                            patches_for_video[k] = np.zeros(
                                (68, 512), dtype=np.float16)
                else:
                    patches_for_video = {
                        k: np.zeros((68, 512), dtype=np.float16)
                        for k in dictionnaire_video.keys()
                    }
                    print(f"  [warn] patches missing for "
                          f"{sujet_id}/{video_id}, using zeros")

            # Commit this video to the output lists (keep them in sync)
            label = labels[sujet_id][video_id]
            labels_liste.append(label)
            all_videos_landmark.append(dictionnaire_video)
            all_videos_patches.append(patches_for_video)
            nb_frames_par_video.append(int(len(dictionnaire_video.keys())))
            loaded_samples.append((sujet_id, video_id, label))

    if skipped > 0:
        print(f"  [info] {skipped} videos skipped, "
              f"{len(loaded_samples)} loaded.")

    return (all_videos_landmark, all_videos_patches, nb_frames_par_video,
            labels_liste, loaded_samples)



def convertir_dictionnaire_video_en_tenseur_video(landmarks_video):
    idx = 0
    for key in sorted(landmarks_video.keys()):
        landmark_tensor = torch.reshape(torch.tensor(landmarks_video[key], dtype=torch.float), (-1, 2))
        landmark_tensor = landmark_tensor.unsqueeze(-1)
        if idx == 0:
            landmarks_tensor = landmark_tensor
            idx += 1
        else:
            landmarks_tensor = torch.cat([landmarks_tensor, landmark_tensor], 2)
    return landmarks_tensor


def convertir_dictionnaire_patches_en_tenseur(patches_video):
    frames  = sorted(patches_video.keys())
    tensors = []
    for k in frames:
        t = torch.tensor(patches_video[k], dtype=torch.float16)
        tensors.append(t.unsqueeze(-1))
    return torch.cat(tensors, dim=2)


def convertir_liste_videos_en_tenseur(liste_videos):
    # Pop-as-we-go: convert each dict to a tensor chunk, then immediately
    # drop the dict reference so its memory can be reclaimed.
    chunks = []
    while liste_videos:
        video = liste_videos.pop(0)
        video_tenseur = convertir_dictionnaire_video_en_tenseur_video(video)
        video_tenseur = video_tenseur.permute(2, 0, 1).reshape(-1, 2, 1)
        chunks.append(video_tenseur)
        del video
    return torch.cat(chunks, dim=2)


def convertir_liste_patches_en_tenseur(liste_patches):
    # Pop-as-we-go: convert each dict to a tensor chunk, then immediately
    # drop the dict reference. Prevents holding the full dict-form tensor
    # AND the tensor-form tensor in memory simultaneously.
    chunks = []
    while liste_patches:
        patches = liste_patches.pop(0)
        pat_t = convertir_dictionnaire_patches_en_tenseur(patches)
        pat_t = pat_t.permute(2, 0, 1).reshape(-1, 512, 1)
        chunks.append(pat_t)
        del patches
    return torch.cat(chunks, dim=2)
# Frame-count equalisation

def reduction_frames_video(landmarks_video, nb_frames_atteindre, mouv_sampler):
    nb_frames_video = len(landmarks_video.keys())
    if nb_frames_video == nb_frames_atteindre:
        return landmarks_video, None
    elif nb_frames_atteindre > nb_frames_video:
        raise NameError("On veut faire une réduction alors que c'est une augmentation")
    else:
        landmarks_tensor  = convertir_dictionnaire_video_en_tenseur_video(landmarks_video)
        indices_gardees   = mouv_sampler(landmarks_tensor, nb_frames_atteindre)
        landmark_video_final = {}
        for i, indice in enumerate(indices_gardees):
            landmark_video_final[i + 1] = landmarks_tensor[:, :, indice].flatten().numpy()
        return landmark_video_final, indices_gardees


def reduction_patches_video(patches_video, indices_gardees):
    keys   = sorted(patches_video.keys())
    result = {}
    for new_key, orig_idx in enumerate(indices_gardees, start=1):
        # orig_idx may be a 0-d torch.Tensor from MovementSampler — convert
        # to plain int so Python list indexing doesn't rely on __index__.
        result[new_key] = patches_video[keys[int(orig_idx)]]
    return result


def augmentation_frames_video(landmarks_video, nb_frames_atteindre, mouv_ext):
    nb_frames_video = len(landmarks_video.keys())
    if nb_frames_video == nb_frames_atteindre:
        return landmarks_video
    elif nb_frames_atteindre < nb_frames_video:
        raise NameError("On veut faire une augmentation alors qu'il faut une réduction")
    else:
        landmarks_tensor   = convertir_dictionnaire_video_en_tenseur_video(landmarks_video)
        extended_landmarks = mouv_ext(landmarks_tensor.to(dtype=torch.float32), nb_frames_atteindre)
        landmark_video_final = {}
        for i in range(extended_landmarks.shape[2]):
            landmark_video_final[i + 1] = extended_landmarks[:, :, i].flatten().numpy()
        return landmark_video_final


def augmentation_patches_video(patches_video, nb_frames_atteindre):
    nb_orig = len(patches_video)
    if nb_orig == nb_frames_atteindre:
        return patches_video
    # Rebuild with sequential keys 1..nb_orig so the padded indices below
    # can't collide with original (potentially non-sequential) frame IDs.
    # This mirrors augmentation_frames_video's output convention.
    keys   = sorted(patches_video.keys())
    result = {i + 1: patches_video[k] for i, k in enumerate(keys)}
    last_patches = result[nb_orig]
    for i in range(nb_orig + 1, nb_frames_atteindre + 1):
        result[i] = last_patches
    return result


def traitements_video_par_video(liste_videos_landmark, nb_frames_par_video,
                                 mode_frame_calculee, nb_frame_final=10,
                                 liste_videos_patches=None):
    if mode_frame_calculee:
        nb_frame_voulu = int(statistics.median(nb_frames_par_video))
    else:
        nb_frame_voulu = nb_frame_final

    extender = MovementExtender(extension_method='spline_interpolation')
    sampler  = MovementSampler().from_preset('top_k')

    use_patches = (liste_videos_patches is not None and
                   liste_videos_patches[0] is not None)

    for i, landmark_video in enumerate(liste_videos_landmark):
        nb_frames = nb_frames_par_video[i]
        if nb_frames == nb_frame_voulu:
            continue
        elif nb_frames < nb_frame_voulu:
            liste_videos_landmark[i] = augmentation_frames_video(
                landmark_video, nb_frame_voulu, extender)
            if use_patches:
                liste_videos_patches[i] = augmentation_patches_video(
                    liste_videos_patches[i], nb_frame_voulu)
        else:
            liste_videos_landmark[i], indices = reduction_frames_video(
                landmark_video, nb_frame_voulu, sampler)
            if use_patches and indices is not None:
                liste_videos_patches[i] = reduction_patches_video(
                    liste_videos_patches[i], indices)

    return (nb_frame_voulu, liste_videos_landmark,
            liste_videos_patches if use_patches else None)


# Data augmentation (class balancing)

def preparations_partitions_test(liste_labels, nb_instance_classe=5, pourcentage_test=0.8,
                                  rgn=None, is_mode_pourcentage=False):
    if is_mode_pourcentage:
        if rgn is None:
            raise NameError("Il nous faut une graine aléatoire")
        from dataloader.build_dataset import partitions
        partitions_train_test, indice_test = partitions(1, len(liste_labels), rgn, pourcentage_test, True)
        partition_train = partitions_train_test[:indice_test]
        partition_test  = partitions_train_test[indice_test:]
    else:
        ens_labels    = list(set(liste_labels))
        min_classe    = min([len(np.where(liste_labels == i)[0]) for i in ens_labels])
        instance_test = [np.where(liste_labels == i)[0][:nb_instance_classe] for i in ens_labels]
        if min_classe <= nb_instance_classe:
            partition_train, partition_test = preparations_partitions_test(
                liste_labels, min_classe - 1, pourcentage_test, rgn, is_mode_pourcentage)
        else:
            partition_test  = np.concatenate(instance_test)
            partition_train = list(set(np.arange(len(liste_labels))) - set(partition_test))
    return partition_train, partition_test


def partitions(nb_partition, nb_elements, rgn, pourcentage=1, is_test=False):
    partition = np.arange(nb_elements, dtype=int)
    rgn.shuffle(partition)
    if is_test:
        indice_decoupe = round(nb_elements * pourcentage)
    else:
        indice_decoupe = [(nb_elements * i) // nb_partition for i in range(nb_partition + 1)]
    return (partition, indice_decoupe)


def augmentation_image_bruit(X, nb_joint, nb_frames, facteur_ecart_type=None):
    """
    Paper-correct coordinate noise (Section 4.2):
      "Gaussian-distributed random noise with the noise level uniformly
       sampled from the range [0, 5]" — absolute pixel units on 240×240 crop.

    facteur_ecart_type is kept for API compatibility but is no longer used.
    """
    std = torch.empty(1).uniform_(0.0, 5.0).item()
    noise = torch.randn_like(X) * std
    return X + noise


# Spatial augmentation helpers (paper Section 4.2)

# 68-point 2D-FAN / dlib horizontal-flip landmark mapping.
# FLIP_68[i] = j means: when the image is horizontally flipped, the landmark
# at canonical position i appears at position j in the flipped image.
FLIP_68 = [
    16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  # jaw
    26, 25, 24, 23, 22, 21, 20, 19, 18, 17,                                 # eyebrows
    27, 28, 29, 30,                                                          # nose bridge
    35, 34, 33, 32, 31,                                                      # nose base
    45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40,                        # eyes L↔R
    54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,                        # outer mouth
    64, 63, 62, 61, 60, 67, 66, 65,                                         # inner mouth
]
assert len(FLIP_68) == 68


def build_flip_map_for_subset(kept_indices):
    """
    Build the per-subset flip mapping for horizontal augmentation.
    Returns None if any flip partner falls outside the kept subset
    (e.g. g_bouche only keeps mouth landmarks — some partners may be missing).
    """
    kept     = list(kept_indices)
    kept_set = set(kept)
    idx_map  = {orig: new for new, orig in enumerate(kept)}

    flip_map = []
    for orig_idx in kept:
        target_orig = FLIP_68[orig_idx]
        if target_orig not in kept_set:
            return None
        flip_map.append(idx_map[target_orig])

    return flip_map


# STEP 1 — Patch noise during augmentation

def data_augmentation_equilibration_classe(X, Y, nb_joint, nb_frames, liste_classe,
                                            facteur, P=None):
    """
    Balances classes by oversampling minority classes with coordinate noise.
    STEP 1 CHANGE: augmented patches now get small Gaussian noise (paper adds
    noise level uniformly sampled from [0, 5] to patches during training).
    """
    use_patches = (P is not None)
    X_final = torch.clone(X)
    Y_final = Y.copy()
    if use_patches:
        P_final = torch.clone(P)

    train_par_classe   = {c: X.permute(2, 0, 1)[Y == c, :, :] for c in liste_classe}
    nb_classe          = {c: len(np.where(Y == c)[0]) for c in liste_classe}
    nb_max             = max(nb_classe.values())

    if use_patches:
        patches_par_classe = {c: P.permute(2, 0, 1)[Y == c, :, :] for c in liste_classe}

    for classe in liste_classe:
        src_coords  = train_par_classe[classe]
        n_src       = len(src_coords)
        if use_patches:
            src_patches = patches_par_classe[classe]

        aug_count = 0
        while nb_classe[classe] < nb_max:
            i = aug_count % n_src
            aug_count += 1
            if nb_classe[classe] >= nb_max:
                break
            nb_classe[classe] += 1

            # Coordinate augmentation (add noise) — unchanged
            x_new = augmentation_image_bruit(src_coords[i], nb_joint, nb_frames, facteur).unsqueeze(-1)
            X_final = torch.cat([X_final, x_new], dim=2)
            Y_final = np.append(Y_final, classe)

            if use_patches:
                # Feature-level patch noise (calibrated to pixel U[0, 5])
                # Paper adds pixel std=5/255 noise to raw 49×49 patches BEFORE
                # the ResNet-34 encoder. Since only 512-d features are stored,
                # we apply a calibrated proxy: pixel std=5 → after ToTensor
                # (÷255) → ~0.020 → after ImageNet Normalize (÷0.225) → ~0.087
                # in normalised input space → ResNet-34 BatchNorm attenuates
                # to ~0.02 in 512-d feature space.
                p_raw       = src_patches[i % len(src_patches)].float()
                noise_level = torch.empty(1).uniform_(0.0, 0.02).item()
                noise       = torch.randn_like(p_raw) * noise_level
                p_new       = (p_raw + noise).half().unsqueeze(-1)
                P_final     = torch.cat([P_final, p_new], dim=2)

    if use_patches:
        return (X_final, Y_final, P_final)
    else:
        return (X_final, Y_final)

# Graph construction helpers

def convertion_graphe(all_videos_landmark, graph_type, all_videos_patches=None):
    """
    Filters landmarks (and patches) to graph-specific joints.

    Returns
    -------
    (nb_joints, final_landmarks, final_patches, edges, kept_indices)
    *** kept_indices is NEW — needed by GCNTransformerPaper ***
    final_patches is None when all_videos_patches is None.
    """
    final_all_videos_landmark = []
    final_all_videos_patches  = []
    use_patches = (all_videos_patches is not None and
                   len(all_videos_patches) > 0 and
                   all_videos_patches[0] is not None)

    patch_iter = all_videos_patches if use_patches else [None] * len(all_videos_landmark)

    kept_indices_global = None   # captured from first video/frame

    for landmarks, patches in zip(all_videos_landmark, patch_iter):
        video_landmark = {}
        video_patches  = {} if use_patches else None

        for key in landmarks.keys():
            val = landmarks[key]

            # Defensive skip — should never trigger after extract_datas_
            # filtering, but kept as a safety net for any malformed entry.
            if isinstance(val, dict) or (hasattr(val, '__len__') and len(val) == 0):
                continue

            landmarks_tensor = torch.reshape(
                torch.tensor(val, dtype=torch.float), (-1, 2))

            if str(graph_type) == 'g_complet':
                nb_joints, landmarks_final, edges, kept = creer_graphe_complet_exp_faciale(landmarks_tensor)
            elif str(graph_type) == 'g_bouche':
                nb_joints, landmarks_final, edges, kept = creer_graphe_version1(landmarks_tensor)
            elif str(graph_type) == 'g_yeux_bouche':
                nb_joints, landmarks_final, edges, kept = creer_graphe_version2(landmarks_tensor)
            elif str(graph_type) == 'g_topologique':
                nb_joints, landmarks_final, edges, kept = creer_graphe_yeux_nez_bouche(landmarks_tensor)
            else:
                raise NameError("graph_type doit valoir ['g_complet','g_yeux_bouche','g_bouche','g_topologique']")

            # Capture kept_indices once (same for all videos/frames)
            if kept_indices_global is None:
                kept_indices_global = kept

            landmarks_final     = torch.reshape(landmarks_final, (-1,))
            video_landmark[key] = landmarks_final

            if use_patches and patches and key in patches:
                video_patches[key] = patches[key][kept]   # (nb_joints, 512)

        final_all_videos_landmark.append(video_landmark)
        if use_patches:
            final_all_videos_patches.append(video_patches)

    return (nb_joints,
            final_all_videos_landmark,
            final_all_videos_patches if use_patches else None,
            edges,
            kept_indices_global)   # ← NEW 5th return value


def build_temporal_edges(P: int, T: int) -> list:
    """
    Build temporal edges connecting same-joint pairs across consecutive frames.
    Node index convention: node_id = t * P + i  (T-major, matches create_graph).
    Returns list of (u, v) tuples, both directions included for undirected graph.
    """
    edges = []
    for t in range(T - 1):
        for i in range(P):
            u = t * P + i
            v = (t + 1) * P + i
            edges.append((u, v))
            edges.append((v, u))
    return edges


def create_graph(video_tensor, label, nb_frames, nb_joints, connections,
                 patch_tensor=None, temporal_edges_mode: str = 'none'):
    """
    Paper-aligned graph creation. Per-frame spatial edges are replicated for
    every T frame and made bidirectional.

    temporal_edges_mode controls how (or whether) temporal edges are added:
      'none'               - spatial-only (original behaviour).
      'combined'           - spatial + temporal edges merged into edge_index.
      'separated_parallel' - spatial in edge_index, temporal in edge_index_temporal.
      'separated_sequential' - same split as separated_parallel.
      'construct_only'     - same edge layout as separated_parallel (produces
                             edge_index_temporal) but GCBlock ignores them in its
                             SGConv path. Use with --tgconv_mode temporal_edges so
                             TGConvOnEdges is the sole consumer of temporal edges.

    x shape in the resulting Data: (T*P, 2)  — T-major (matches model's
    coords = x.view(bs, T, P, 2) in forward()).
    edge_index shape: (2, 2*T*E) where E = per-frame spatial edges.
    """
    T = nb_frames
    P = nb_joints

    # 1. Coordinates: use as-is
    x = video_tensor.float()   # (T*P, 2)

    # 2. Expand per-frame edges to cover all T frames, bidirectional
    if isinstance(connections, list):
        per_frame = connections
    else:
        arr = connections.tolist() if hasattr(connections, 'tolist') else list(connections)
        per_frame = arr

    spatial_edges = []
    for t in range(T):
        offset = t * P
        for (u, v) in per_frame:
            spatial_edges.append([u + offset, v + offset])   # forward
            spatial_edges.append([v + offset, u + offset])   # reverse (undirected GCN)

    y = torch.tensor([label], dtype=torch.long)

    if temporal_edges_mode == 'combined':
        # Merge spatial + temporal edges into a single edge_index.
        temporal_raw = build_temporal_edges(P, T)
        all_edges = spatial_edges + [[u, v] for (u, v) in temporal_raw]
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        if patch_tensor is not None:
            return Data(x=x, patch_features=patch_tensor, edge_index=edge_index, y=y)
        else:
            return Data(x=x, edge_index=edge_index, y=y)

    elif temporal_edges_mode in ('separated_parallel', 'separated_sequential', 'construct_only'):
        # Keep spatial and temporal as separate tensors.  PyG's Batch uses any
        # attribute whose name contains 'edge_index' and applies the per-graph
        # node offset automatically (via Data.__inc__), so both tensors are
        # correctly renumbered when DataLoader collates multiple samples.
        edge_index_spatial = torch.tensor(spatial_edges, dtype=torch.long).t().contiguous()
        temporal_raw = build_temporal_edges(P, T)
        edge_index_temporal = torch.tensor(
            [[u, v] for (u, v) in temporal_raw], dtype=torch.long
        ).t().contiguous()
        if patch_tensor is not None:
            return Data(x=x, patch_features=patch_tensor,
                        edge_index=edge_index_spatial,
                        edge_index_temporal=edge_index_temporal, y=y)
        else:
            return Data(x=x, edge_index=edge_index_spatial,
                        edge_index_temporal=edge_index_temporal, y=y)

    else:
        # 'none': original spatial-only behaviour.
        edge_index = torch.tensor(spatial_edges, dtype=torch.long).t().contiguous()
        if patch_tensor is not None:
            return Data(x=x, patch_features=patch_tensor, edge_index=edge_index, y=y)
        else:
            return Data(x=x, edge_index=edge_index, y=y)

# 2D-FAN 68-landmark standard connectivity
# Standard dlib/IBUG 68-point facial landmark edges
_FAN_EDGES = [
    # Jaw (0-16)
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),
    (8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),
    # Left eyebrow (17-21)
    (17,18),(18,19),(19,20),(20,21),
    # Right eyebrow (22-26)
    (22,23),(23,24),(24,25),(25,26),
    # Nose bridge (27-30)
    (27,28),(28,29),(29,30),
    # Nose bottom (30-35)
    (30,31),(31,32),(32,33),(33,34),(34,35),(30,35),
    # Left eye (36-41)
    (36,37),(37,38),(38,39),(39,40),(40,41),(41,36),
    # Right eye (42-47)
    (42,43),(43,44),(44,45),(45,46),(46,47),(47,42),
    # Outer mouth (48-59)
    (48,49),(49,50),(50,51),(51,52),(52,53),(53,54),
    (54,55),(55,56),(56,57),(57,58),(58,59),(59,48),
    # Inner mouth (60-67)
    (60,61),(61,62),(62,63),(63,64),(64,65),(65,66),(66,67),(67,60),
    # Connect eyebrows to nose
    (21,27),(22,27),
    # Connect outer/inner mouth
    (48,60),(54,64),
]

# Long-range cross-component edges — link semantically related landmark
# groups that are spatially distant. Helps the spatial GCN exchange
# information between components without relying solely on transformers.
_CROSS_EDGES = [
    # Left eyebrow ↔ left eye
    (19, 37), (21, 36),
    # Right eyebrow ↔ right eye
    (24, 44), (22, 42),
    # Nose tip ↔ mouth centre
    (30, 51), (30, 62),
    # Eyes ↔ mouth corners
    (36, 48), (45, 54),
]


def from_nodes_create_list_edges_fan(list_nodes):
    """
    Build edges for a subset of the 68 2D-FAN landmarks.
    Returns standard mesh edges + cross-component edges
    where both endpoints are in the subset.
    """
    node_set = set(list_nodes)
    standard = [(e1, e2) for (e1, e2) in _FAN_EDGES
                if e1 in node_set and e2 in node_set]
    cross    = [(e1, e2) for (e1, e2) in _CROSS_EDGES
                if e1 in node_set and e2 in node_set]
    return standard + cross


def create_graph_from_list_graph(coord_noeud, list_noeud):
    """Returns (filtered_coords, filtered_edges, kept_indices)."""
    noeuds = set()
    for n1, n2 in list_noeud:
        noeuds.update([n1, n2])
    noeuds_ordonnees = sorted(list(noeuds))
    coincidence      = {noeud: i for i, noeud in enumerate(noeuds_ordonnees)}
    new_coord_noeuds = coord_noeud[noeuds_ordonnees]
    new_edges        = [(coincidence[n1], coincidence[n2]) for (n1, n2) in list_noeud]
    return (new_coord_noeuds, new_edges, noeuds_ordonnees)


# Graph type constructors (2D-FAN 68-landmark versions)

def creer_graphe_complet_exp_faciale(coord_noeud):
    """g_complet — all 68 landmarks."""
    nodes = list(range(len(coord_noeud)))   # 0..67
    edges = from_nodes_create_list_edges_fan(nodes)
    final_nodes, final_edges, kept = create_graph_from_list_graph(coord_noeud, edges)
    return (len(final_nodes), final_nodes, final_edges, kept)


def creer_graphe_version1(coord_noeud):
    """g_bouche — mouth only (landmarks 48-67)."""
    nodes = list(range(48, 68))
    edges = from_nodes_create_list_edges_fan(nodes)
    final_nodes, final_edges, kept = create_graph_from_list_graph(coord_noeud, edges)
    return (len(final_nodes), final_nodes, final_edges, kept)


def creer_graphe_version2(coord_noeud):
    """g_yeux_bouche — eyes + brows + mouth."""
    nodes = (list(range(17, 27)) +   # both eyebrows
             list(range(36, 48)) +   # both eyes
             list(range(48, 68)))    # mouth
    edges = from_nodes_create_list_edges_fan(nodes)
    final_nodes, final_edges, kept = create_graph_from_list_graph(coord_noeud, edges)
    return (len(final_nodes), final_nodes, final_edges, kept)


def creer_graphe_yeux_nez_bouche(coord_noeud):
    """g_topologique — eyes + brows + nose + mouth (no jaw)."""
    nodes = (list(range(17, 27)) +   # both eyebrows
             list(range(27, 36)) +   # nose
             list(range(36, 48)) +   # both eyes
             list(range(48, 68)))    # mouth
    edges = from_nodes_create_list_edges_fan(nodes)
    final_nodes, final_edges, kept = create_graph_from_list_graph(coord_noeud, edges)
    return (len(final_nodes), final_nodes, final_edges, kept)


# Debug

def show_pyg_graph(data, nom_graphe):
    import networkx as nx
    import matplotlib.pyplot as plt
    name_graph = nom_graphe + ".png"
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(data.x)))
    edges = data.edge_index.T
    G.add_edges_from(edges)
    pos = data.x.numpy()
    plt.figure(figsize=(6, 6))
    plt.clf()
    nx.draw(G, with_labels=False, pos=pos, alpha=0.5, node_size=5)
    if name_graph in os.listdir():
        os.remove(name_graph)
    plt.savefig(name_graph)
