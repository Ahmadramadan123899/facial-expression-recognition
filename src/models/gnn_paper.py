#! /usr/bin/python3
# Author : Ahmad Ramadan

"""
gnn_paper.py,  Graph-Tran for 2D-FAN 68-landmark inputs

Landmark layout (2D-FAN / dlib 68-point standard):
  0-16  : jaw
  17-21 : left eyebrow
  22-26 : right eyebrow
  27-35 : nose bridge + tip (30 = nose tip)
  36-41 : left eye
  42-47 : right eye
  48-67 : mouth

Paper channel flow (Section 4.2):
  3 x CoordBranch(2→64→128→256) → g = 768-dim
  ResNet appearance               → v = 512-dim
  Early fusion u = [g, v]         → 1280-dim
  input_proj(1280→512)
  GCBlocks + Transformers at 512 channels
  pre_logit(512→256) → head(256→nc)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


# ─────────────────────────────────────────────────────────────────────────────
# 2D-FAN 68-point cluster assignments
# ─────────────────────────────────────────────────────────────────────────────

_JAW        = set(range(0,  17))   # cluster 6 (other / contour)
_LEFT_BROW  = set(range(17, 22))   # cluster 0
_RIGHT_BROW = set(range(22, 27))   # cluster 1
_NOSE       = set(range(27, 36))   # cluster 4
_LEFT_EYE   = set(range(36, 42))   # cluster 2
_RIGHT_EYE  = set(range(42, 48))   # cluster 3
_MOUTH      = set(range(48, 68))   # cluster 5

_NOSE_TIP   = 30   # 2D-FAN: landmark 30 = nose tip


def _get_cluster(idx: int) -> int:
    if idx in _LEFT_BROW:  return 0
    if idx in _RIGHT_BROW: return 1
    if idx in _LEFT_EYE:   return 2
    if idx in _RIGHT_EYE:  return 3
    if idx in _NOSE:       return 4
    if idx in _MOUTH:      return 5
    return 6   # jaw / other


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class TemporalConvBlock(nn.Module):
    """1-D residual conv along temporal axis, per joint. Kernel size = 5 (paper)."""
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size,padding=kernel_size // 2, bias=False)
        self.bn   = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x : (bs, P, C, T)
        bs, P, C, T = x.shape
        xf = x.view(bs * P, C, T)
        xf = self.drop(self.relu(self.bn(self.conv(xf)))) + xf
        return xf.view(bs, P, C, T)


class ComponentAttention(nn.Module):
    """
    Spatial component attention (paper Fig. 5).
    Clusters joints → avg+max pool per cluster → 1x1 conv → softmax → rescale.
    """
    def __init__(self, channels: int, num_clusters: int = 7):
        super().__init__()
        self.num_clusters = num_clusters
        self.conv = nn.Conv1d(2, 1, kernel_size=1, bias=False) #expect(batch_size, 2, length)

    def forward(self, x: Tensor, cluster_ids: Tensor) -> Tensor:
        # x : (bs, P, C, T)
        #C NB OF FEATURE CHANNELS PER JOINT
        bs, P, C, T = x.shape
        nc = self.num_clusters

        # Cluster-level features: (bs, nc, C, T)
        cm = torch.zeros(bs, nc, C, T, device=x.device)
        for c in range(nc):
            mask = (cluster_ids == c)
            if mask.sum() > 0:
                cm[:, c] = x[:, mask].mean(dim=1)

        # CAP / CMP over channel dim (cm : (bs, nc, C, T)→ (bs, nc, T) )
        cap = cm.mean(dim=2)
        cmp = cm.max(dim=2).values

        # Reshape for conv: (bs*T, 2, nc) → (bs*T, 1, nc) flattin batch and time 
        cap_t = cap.permute(0, 2, 1).reshape(bs * T, nc)
        cmp_t = cmp.permute(0, 2, 1).reshape(bs * T, nc)
        #stacking (bs*T, 2, nc)
        
        attn  = F.softmax(
            self.conv(torch.stack([cap_t, cmp_t], dim=1)), dim=-1
        ).squeeze(1).view(bs, T, nc)   #(bs*T, 1, nc) then (bs, T, nc) 
        #So now for each frame of each video, we have one attention weight per cluster.


        # Map cluster scores back to joints: (bs, P, 1, T)
        #attn[:, :, cluster_ids] : (bs, T, P)
        ja = attn[:, :, cluster_ids].permute(0, 2, 1).unsqueeze(2) #to match x : (bs, P, C, T) shape
        return x * ja + x   # residual


class FrameAttention(nn.Module):
    """
    Temporal frame attention (Fig. 6).
    SAP+SMP across spatial/channel dims → 1x1 conv → softmax → rescale frames.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x : (bs, P, C, T)
        bs, P, C, T = x.shape
        xf   = x.reshape(bs, P * C, T)
        sap  = xf.mean(dim=1)           # (bs, T)
        smp  = xf.max(dim=1).values     # (bs, T)
        #after stacking (bs, 2, T)
        attn = F.softmax(
            self.conv(torch.stack([sap, smp], dim=1)), dim=-1
        )   # (bs, 1, T)
        return x * attn.unsqueeze(1) + x   # residual


class CoordBranch(nn.Module):
    """
    One branch of the triple coordinate encoder.
    Paper: 3 spatial GCN layers with channels [64, 128, 256] + temporal conv (k=5).
    """
    def __init__(self, dropout: float = 0.1, kernel_size: int = 5):
        super().__init__()
        self.conv1 = GCNConv(2,   64)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = GCNConv(64,  128)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = GCNConv(128, 256)
        self.bn3   = nn.BatchNorm1d(256)
        self.temp  = TemporalConvBlock(256, kernel_size, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x_flat: Tensor, edge_index: Tensor,
                bs: int, T: int, P: int) -> Tensor:
        # x_flat : (bs*T*P, 2)
        x = self.drop(F.relu(self.bn1(self.conv1(x_flat,   edge_index))))
        x = self.drop(F.relu(self.bn2(self.conv2(x,        edge_index))))
        x = self.drop(F.relu(self.bn3(self.conv3(x,        edge_index))))
        # (bs*T*P, 256) → (bs, P, 256, T)
        x = x.view(bs, T, P, 256).permute(0, 2, 3, 1).contiguous()
        return self.temp(x)   # (bs, P, 256, T)


class GCBlock(nn.Module):
    """
    Main processor GC block (paper Fig. 4):
      Spatial GCN → Component Attention → Temporal Conv → Frame Attention
    """
    def __init__(self, channels: int, cluster_ids: Tensor,
                 dropout: float = 0.1, kernel_size: int = 5):
        super().__init__()
        self.sgcn  = GCNConv(channels, channels)
        self.sbn   = nn.BatchNorm1d(channels)
        self.cattn = ComponentAttention(channels, 7)
        self.tconv = TemporalConvBlock(channels, kernel_size, dropout)
        self.fattn = FrameAttention(channels)
        self.drop  = nn.Dropout(dropout)
        self.register_buffer('cluster_ids', cluster_ids)

    def forward(self, x: Tensor, edge_index: Tensor,
                bs: int, T: int, P: int) -> Tensor:
        # Spatial GCN
        xf = x.permute(0, 3, 1, 2).reshape(bs * T * P, -1)
        xf = self.drop(F.relu(self.sbn(self.sgcn(xf, edge_index))))
        x  = xf.view(bs, T, P, -1).permute(0, 2, 3, 1).contiguous()
        # Attention
        x = self.cattn(x, self.cluster_ids)
        x = self.tconv(x)
        x = self.fattn(x)
        return x


class SpatialTransformerBlock(nn.Module):
    """
    Spatial transformer: each frame → all joints attend to each other (paper Eq. 9-10).
    Uses standard Pre-LN (LayerNorm before attention) for stability.
    """
    def __init__(self, d_model: int, n_heads: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        bs, P, C, T = x.shape
        # (bs*T, P, C)
        q = x.permute(0, 3, 1, 2).reshape(bs * T, P, C)
        #attn_output, attn_weights = attn(Q, K, V)
        a, _ = self.attn(self.norm1(q), self.norm1(q), self.norm1(q))
        q    = q + self.drop(a)
        q    = q + self.ff(self.norm2(q))
        return q.view(bs, T, P, C).permute(0, 2, 3, 1).contiguous()


class TemporalTransformerBlock(nn.Module):
    """
    Temporal transformer: each joint → all frames attend to each other (paper Eq. 11-12).
    """
    def __init__(self, d_model: int, n_heads: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        bs, P, C, T = x.shape
        # (bs*P, T, C)
        q = x.permute(0, 1, 3, 2).reshape(bs * P, T, C)
        a, _ = self.attn(self.norm1(q), self.norm1(q), self.norm1(q))
        q    = q + self.drop(a)
        q    = q + self.ff(self.norm2(q))
        return q.view(bs, P, T, C).permute(0, 1, 3, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Center Loss
# ─────────────────────────────────────────────────────────────────────────────

class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        return ((features - self.centers[labels]) ** 2).sum(dim=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Full Model — 2D-FAN 
# ─────────────────────────────────────────────────────────────────────────────

class GCNTransformerPaper(nn.Module):
    """
    Graph-Tran for 2D-FAN 68-landmark inputs.

    Channel flow (paper Section 4.2):
      3 CoordBranches: each 2→64→128→256  → g = 768-dim
      ResNet appearance (512-dim)          → v = 512-dim
      Early fusion u = [g, v]              → 1280-dim
      input_proj(1280 → 512)
      num_gc_blocks x (GCBlock → SpatialTrans → TemporalTrans) at 512-dim
      GlobalMean over joints+frames
      pre_logit(512→256) → head(256→nc)
    """

    COORD_OUT     = 256
    PATCH_DIM     = 512
    FUSION_DIM    = 3 * 256 + 512   # 1280
    MAIN_CHANNELS = 512

    def __init__(self,
                 nb_joint        : int,
                 seq_length      : int,          #nb franes 16
                 n_heads         : int   = 8,
                 dim_feedforward : int   = 1024, # Dimension of the hidden layer in the Transformer feed-forward network
                 num_gc_blocks   : int   = 2,
                 num_trans_layers: int   = 2,
                 out_dim         : int   = 7,
                 kept_indices    : list  = None, # Indices of landmarks to retain if filtering a subset of facial points
                 dropout         : float = 0.1,
                 use_patches     : bool  = True):# Flag to include local appearance patches (visual features) with coordinates
        super().__init__()

        self.nb_joint      = nb_joint
        self.seq_length    = seq_length
        self.use_patches   = use_patches
        self.num_gc_blocks = num_gc_blocks

        # ── Cluster assignments (2D-FAN indices) ──────────────────────
        kept_list   = list(kept_indices) if kept_indices is not None \
                      else list(range(nb_joint)) #kept land
        cluster_ids = torch.tensor(
            [_get_cluster(k) for k in kept_list], dtype=torch.long
        )
        self.register_buffer('cluster_ids', cluster_ids)

        # Nose tip index in the kept-landmark array
        self.nose_tip_pos = kept_list.index(_NOSE_TIP) if _NOSE_TIP in kept_list else None

        # Left/right eye indices for scale computation
        le_pos = [i for i, k in enumerate(kept_list) if k in _LEFT_EYE]
        re_pos = [i for i, k in enumerate(kept_list) if k in _RIGHT_EYE]
        if le_pos:
            self.register_buffer('le_idx', torch.tensor(le_pos, dtype=torch.long))
        else:
            self.le_idx = None
        if re_pos:
            self.register_buffer('re_idx', torch.tensor(re_pos, dtype=torch.long))
        else:
            self.re_idx = None

        # ── Triple coordinate encoder ──────────────────────────────────
        self.branch_R = CoordBranch(dropout)
        self.branch_C = CoordBranch(dropout)
        self.branch_T = CoordBranch(dropout)

        # ── Input projection 1280 → 512 ───────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(self.FUSION_DIM, self.MAIN_CHANNELS),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── Main processor ─────────────────────────────────────────────
        self.gc_blocks = nn.ModuleList([
            GCBlock(self.MAIN_CHANNELS, cluster_ids, dropout)
            for _ in range(num_gc_blocks)
        ])
        self.spatial_trans = nn.ModuleList([
            SpatialTransformerBlock(self.MAIN_CHANNELS, n_heads,
                                    dim_feedforward, dropout)
            for _ in range(num_trans_layers)
        ])
        self.temporal_trans = nn.ModuleList([
            TemporalTransformerBlock(self.MAIN_CHANNELS, n_heads,
                                     dim_feedforward, dropout)
            for _ in range(num_trans_layers)
        ])

        # Learnable positional embedding (paper Eq. 7, applied once)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, nb_joint, self.MAIN_CHANNELS)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02) #intialises it with small random value 

        # ── Classifier ─────────────────────────────────────────────────
        self.pre_logit = nn.Sequential(
            nn.Linear(self.MAIN_CHANNELS, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head = nn.Linear(256, out_dim)

    # ── Coordinate transforms (paper Eqs. 1-4) ────────────────────────────

    def _compute_R(self, coords: Tensor) -> Tensor:
        """R_i = (x_i - x_nose) / (2 * S), S = inter-eye distance (Eq. 1-2)."""
        # coords : (bs, T, P, 2)
        if self.nose_tip_pos is not None:
            center = coords[:, :, self.nose_tip_pos:self.nose_tip_pos + 1, :]
        else:
            center = coords.mean(dim=2, keepdim=True)
        R = coords - center

        if self.le_idx is not None and self.re_idx is not None:
            le    = coords[:, :, self.le_idx, :].mean(dim=2)    # (bs, T, 2)
            re    = coords[:, :, self.re_idx, :].mean(dim=2)
            scale = (le - re).norm(dim=-1, keepdim=True).unsqueeze(2)
        else:
            scale = R.norm(dim=-1, keepdim=True).mean(dim=2, keepdim=True)
        return R / (2.0 * scale.clamp(min=1e-6))

    def _compute_C(self, R: Tensor) -> Tensor:
        """C_i = R_i - mean(R_cluster_i)  (Eq. 3)."""
        C = R.clone()
        for c in range(7):
            mask = (self.cluster_ids == c)
            if mask.sum() > 0:
                mu = R[:, :, mask, :].mean(dim=2, keepdim=True)
                C[:, :, mask, :] = R[:, :, mask, :] - mu
        return C

    def _compute_T(self, R: Tensor) -> Tensor:
        """tau_t = R_t - R_{t-1}  (Eq. 4)."""
        tau = torch.zeros_like(R)
        tau[:, 1:] = R[:, 1:] - R[:, :-1]
        return tau

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: Tensor, edge_index: Tensor,
                batch_vec: Tensor, bs: int,
                patch_features: Optional[Tensor] = None):
        """
        x              : (bs*T*P, 2)
        edge_index     : (2, E)  — batched PyG edge index
        batch_vec      : PyG batch vector (unused, kept for API compat.)
        bs             : actual batch size
        patch_features : (bs*T*P, 512) float16  — ResNet-34 FER+ features
        """
        P = self.nb_joint
        T = self.seq_length

        # 1. Coordinate representations R, C, T
        coords = x.view(bs, T, P, 2)
        R      = self._compute_R(coords)
        C      = self._compute_C(R)
        Tau    = self._compute_T(R)

        # 2. Triple coordinate encoder (each branch: 2→64→128→256)
        gR = self.branch_R(R.reshape(bs*T*P, 2),   edge_index, bs, T, P) #(bs, P, 256, T)
        gC = self.branch_C(C.reshape(bs*T*P, 2),   edge_index, bs, T, P)
        gT = self.branch_T(Tau.reshape(bs*T*P, 2), edge_index, bs, T, P)
        g  = torch.cat([gR, gC, gT], dim=2)   # (bs, P, 768, T)

        # 3. Appearance features
        if self.use_patches and patch_features is not None:
            v = patch_features.float().view(bs, T, P, 512).permute(0, 2, 3, 1)
        else:
            v = torch.zeros(bs, P, 512, T, device=x.device)

        # 4. Early fusion u = [g, v] → 1280-dim
        u = torch.cat([g, v], dim=2)   # (bs, P, 1280, T)

        # 5. Project 1280 → 512
        bs_, P_, Cin_, T_ = u.shape
        u = self.input_proj(
            u.permute(0, 1, 3, 2).reshape(-1, Cin_)
        ).reshape(bs_, P_, T_, self.MAIN_CHANNELS).permute(0, 1, 3, 2).contiguous()
        # u : (bs, P, 512, T)

        # 6. Main processor: iterative GCBlock + Spatial/Temporal Transformer
        pos_added = False
        for i in range(self.num_gc_blocks):
            u = self.gc_blocks[i](u, edge_index, bs, T, P)

            # Add learnable positional embedding ONCE before first transformer (Eq. 7)
            if not pos_added:
                # pos_embed : (1, P, 512) — broadcast over bs and T
                u = u + self.pos_embed.unsqueeze(-1)   # (bs, P, 512, T)
                pos_added = True

            ti = min(i, len(self.spatial_trans) - 1)#if we have fewer transformers then gcblocks
            u  = self.spatial_trans[ti](u)
            u  = self.temporal_trans[ti](u)

        # 7. Global mean pooling over joints and frames
        feat = u.mean(dim=[1, 3])   # (bs, 512) single 512-dim video representation

        # 8. Classification
        pre    = self.pre_logit(feat)   # (bs, 256)
        logits = self.head(pre)          # (bs, nc)
        return logits, pre
