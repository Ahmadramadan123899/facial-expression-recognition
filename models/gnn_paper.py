#! /usr/bin/python3
"""
gnn_paper.py  —  Graph-Tran for 2D-FAN 68-landmark inputs
Zhao et al., IEEE TAFFC 2023.

Landmark layout (2D-FAN / dlib 68-point standard):
  0-16  : jaw
  17-21 : left eyebrow
  22-26 : right eyebrow
  27-35 : nose bridge + tip (30 = nose tip)
  36-41 : left eye
  42-47 : right eye
  48-67 : mouth

Paper channel flow (Section 4.2):
  3 × CoordBranch(2→64→128→256) → g = 768-dim
  ResNet appearance               → v = 512-dim
  Early fusion u = [g, v]         → 1280-dim
  input_proj(1280→512)
  GCBlocks + Transformers at 512 channels
  pre_logit(512→256) → head(256→nc)
"""

from typing import Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch import Tensor
from torch_geometric.nn import SGConv as _PyGSGConv


# In-model ResNet-18 patch encoder (for end-to-end / partial fine-tune)

class PatchEncoder(nn.Module):
    """
    ResNet patch encoder — wraps ResNet-18 or ResNet-34 and turns
    (N, 3, 49, 49) uint8 patches into (N, 512) feature vectors.
    Both architectures output 512-d from layer4, so nothing downstream
    changes when switching backbones.

    Three modes:
      - 'frozen'  : eval()+no-grad.
      - 'partial' : unfreeze layer4 only. Stages 1-3 stay frozen.
      - 'full'    : all parameters trainable.

    backbone options:
      - 'resnet34' (default) : FER+ pretrained checkpoint (PyTorch Lightning .ckpt).
        Key stripping: 'model.model.' / 'model.' prefixes removed.
      - 'resnet18'           : Emotion-FAN FER+ pretrained checkpoint (.pth.tar).
        Raw state dict, DataParallel 'module.' prefix stripped, then
        'features.N.' keys remapped to standard torchvision names.

    Input expectation:
      - patches : (N, 3, 49, 49) uint8 OR float in [0, 1]. ImageNet
        normalisation is applied inside the module by default for both
        backbones. Pass apply_imagenet_norm=False to disable.
    """

    # ImageNet stats
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 mode: str = 'partial',
                 backbone: str = 'resnet34',
                 apply_imagenet_norm: Optional[bool] = None,
                 upscale_to_224: bool = False,
                 freeze_bn_stats: bool = False):
        super().__init__()
        if mode not in ('frozen', 'partial', 'full'):
            raise ValueError(f"mode must be frozen|partial|full, got {mode!r}")
        if backbone not in ('resnet18', 'resnet34'):
            raise ValueError(f"backbone must be resnet18|resnet34, got {backbone!r}")
        # Although Emotion-FAN's FER+ training used ToTensor-only ([0,1] inputs),
        # empirical results on CK+ (small dataset, full unfreezing) show that
        # applying ImageNet normalization is consistently better. Reason: with
        # only ~300 sequences and full BN unfreezing, the model needs stable
        # initial activations — ImageNet norm provides that even if the FER+
        # checkpoint BN stats were fit to [0,1]. BN adapts within a few epochs
        # but the first-epoch dynamics dominate on small datasets.
        # Pass apply_imagenet_norm=False explicitly to disable.
        if apply_imagenet_norm is None:
            apply_imagenet_norm = True
        self.apply_imagenet_norm = apply_imagenet_norm
        self.mode     = mode
        self.backbone = backbone
        # Upscale 49x49 landmark patches to 224x224 before the ResNet forward.
        # The FER+ checkpoint was trained on 224x224 crops; native 49x49 input
        # collapses layer4 to 2x2 spatial dim. No-op on inputs already 224x224
        # (forward_feature_map / forward_feature_pyramid, which only ever see
        # full 224x224 frames).
        self.upscale_to_224 = upscale_to_224
        # Freeze BatchNorm running stats (keep conv/affine params trainable).
        # At bs*T*P patches per step where bs=1, all patches come from a
        # single video — a highly correlated, non-representative "batch" for
        # BN running-stat estimation. Freezing to the checkpoint's pretrained
        # stats (fit on real 224x224 face crops — a good match once patches
        # are upscaled) avoids that instability without needing a larger
        # batch (which doesn't fit in memory at this resolution regardless).
        self.freeze_bn_stats = freeze_bn_stats

        # Build the selected backbone (no pretrained weights — load below).
        if backbone == 'resnet18':
            net = tvm.resnet18(weights=None)
        else:
            # resnet34: add the FER+ classification head so the checkpoint
            # loads without key mismatches, then replace with Identity.
            net = tvm.resnet34(weights=None)
            net.fc = nn.Linear(512, 7)

        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            # Support two checkpoint formats:
            #   'state_dict' key → ResNet-34 FER+ (PyTorch Lightning .ckpt)
            #   raw dict       → ResNet-18 FER+ Emotion-FAN .pth.tar
            if 'state_dict' in cp:
                state = cp['state_dict']
            else:
                state = cp

            # Emotion-FAN ResNet-18 uses a features.N. naming convention
            # instead of standard torchvision names. Remap back.
            _DAN_MAP = {
                'features.0.': 'conv1.',
                'features.1.': 'bn1.',
                'features.4.': 'layer1.',
                'features.5.': 'layer2.',
                'features.6.': 'layer3.',
                'features.7.': 'layer4.',
            }
            cleaned = {}
            for k, v in state.items():
                # Strip DataParallel prefix (multi-GPU checkpoints, e.g. Emotion-FAN)
                if k.startswith('module.'):
                    k = k[len('module.'):]
                remapped = False
                for dan_pfx, tv_pfx in _DAN_MAP.items():
                    if k.startswith(dan_pfx):
                        k = tv_pfx + k[len(dan_pfx):]
                        remapped = True
                        break
                if not remapped:
                    # Strip ResNet-34 FER+ checkpoint wrapper prefixes
                    k = k.replace('model.model.', '')
                    k = k.replace('model.', '')
                cleaned[k] = v

            # Drop fc.* entirely — size varies (7 / 1000 / 87k classes) and
            # we replace fc with Identity anyway. Also drop the extra DAN 'bn'.
            cleaned = {k: v for k, v in cleaned.items()
                       if not k.startswith('fc.') and not k.startswith('bn.')}

            missing, _ = net.load_state_dict(cleaned, strict=False)
            # Warn only on true backbone misses (conv/bn/layer keys).
            backbone_missing = [k for k in missing if not k.startswith('fc.')]
            if backbone_missing:
                print(f'[PatchEncoder] Missing backbone keys: '
                      f'{len(backbone_missing)} (first 3: {backbone_missing[:3]})')
            print(f'[PatchEncoder] Loaded {backbone} checkpoint from {checkpoint_path}')
        else:
            print(f'[PatchEncoder] No checkpoint at {checkpoint_path}; '
                  f'using random init (NOT recommended).')

        net.fc = nn.Identity()  # 512-dim features (same for resnet18/34)
        self.net = net

        # Register normalisation buffers (move with .to(device))
        self.register_buffer('imagenet_mean', self._MEAN)
        self.register_buffer('imagenet_std',  self._STD)

        self._apply_freeze_policy()

    def _apply_freeze_policy(self):
        if self.mode == 'frozen':
            for p in self.net.parameters():
                p.requires_grad = False
        elif self.mode == 'partial':
            for p in self.net.parameters():
                p.requires_grad = False
            for p in self.net.layer4.parameters():
                p.requires_grad = True
        elif self.mode == 'full':
            for p in self.net.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode == 'frozen':
            self.net.eval()
        elif self.freeze_bn_stats and mode:
            # Conv/affine params still receive gradients (unaffected by
            # train()/eval() — only which stats normalise the activations
            # changes); running_mean/running_var stay at the checkpoint's
            # pretrained values instead of being overwritten by this batch.
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def trainable_parameters(self):
        """Yield only the parameters that require_grad — for the optimizer."""
        for p in self.net.parameters():
            if p.requires_grad:
                yield p

    def forward(self, patches: Tensor) -> Tensor:
        """
        patches : (N, 3, 49, 49) uint8 or float
        returns : (N, 512) float
        """
        if patches.dtype == torch.uint8:
            x = patches.float() / 255.0
        else:
            x = patches.float()
            # If caller already divided by 255 they're in [0, 1]; else assume so.
        if self.upscale_to_224 and x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        if self.apply_imagenet_norm:
            x = (x - self.imagenet_mean) / self.imagenet_std

        if self.mode == 'frozen':
            with torch.no_grad():
                feats = self.net(x)
        else:
            feats = self.net(x)
        return feats

    def forward_feature_map(self, x: Tensor) -> Tensor:
        """
        Run ResNet forward up through layer4, stopping before global avg-pool.

        x      : (N, 3, 224, 224) uint8 or float in [0, 1]
        returns: (N, 512, 7, 7)  — spatial feature map at stride-32
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if self.apply_imagenet_norm:
            x = (x - self.imagenet_mean) / self.imagenet_std

        def _run(x):
            x = self.net.conv1(x)
            x = self.net.bn1(x)
            x = self.net.relu(x)
            x = self.net.maxpool(x)
            x = self.net.layer1(x)
            x = self.net.layer2(x)
            x = self.net.layer3(x)
            x = self.net.layer4(x)
            return x   # (N, 512, 7, 7)

        if self.mode == 'frozen':
            with torch.no_grad():
                return _run(x)
        else:
            return _run(x)

    def forward_feature_pyramid(self, x: Tensor) -> dict:
        """
        Run ResNet forward and return feature maps at three scales.

        x      : (N, 3, 224, 224) uint8 or float in [0, 1]
        returns: dict with keys 'layer2', 'layer3', 'layer4'
                   layer2: (N, 128, 28, 28)  stride-8,  mid-level texture
                   layer3: (N, 256, 14, 14)  stride-16, anatomical features
                   layer4: (N, 512,  7,  7)  stride-32, semantic context
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if self.apply_imagenet_norm:
            x = (x - self.imagenet_mean) / self.imagenet_std

        def _run(x):
            x = self.net.conv1(x)
            x = self.net.bn1(x)
            x = self.net.relu(x)
            x = self.net.maxpool(x)
            x = self.net.layer1(x)
            x = self.net.layer2(x)
            feat_l2 = x   # (N, 128, 28, 28)
            x = self.net.layer3(x)
            feat_l3 = x   # (N, 256, 14, 14)
            x = self.net.layer4(x)
            feat_l4 = x   # (N, 512, 7, 7)
            return {'layer2': feat_l2, 'layer3': feat_l3, 'layer4': feat_l4}

        if self.mode == 'frozen':
            with torch.no_grad():
                return _run(x)
        else:
            return _run(x)


# FragNet-style dual-pathway lateral fusion (He & Schomaker, IEEE T-PAMI 2020)

def sample_window(feat_map: Tensor, landmarks_norm: Tensor, window_size: int = 3) -> Tensor:
    """
    Bilinearly samples a window_sizexwindow_size block of features from
    feat_map, centered at each landmark's normalized position, with grid
    spacing equal to one feature-map cell — i.e. the window covers the
    landmark's own cell plus its immediate neighbours at that pyramid scale.
    Scale-invariant: the same normalized landmark position samples the
    corresponding window regardless of feat_map's spatial resolution.

    feat_map       : (N, C, H, W)          — N = bs*T
    landmarks_norm : (N, P, 2) in [-1, +1] — (x, y) order
    returns        : (N, P, C, window_size, window_size)
    """
    N, C, H, W = feat_map.shape
    _, P, _ = landmarks_norm.shape
    ws = window_size

    step_x = 2.0 / W
    step_y = 2.0 / H
    offsets = torch.arange(-(ws // 2), ws // 2 + 1,
                           device=feat_map.device, dtype=feat_map.dtype)
    grid_dx = offsets * step_x
    grid_dy = offsets * step_y
    # (ws, ws, 2): local offset grid, row = dy, col = dx (matches grid_sample's
    # (N, H_out, W_out, 2) convention with last dim (x, y)).
    dxdy = torch.stack(torch.meshgrid(grid_dx, grid_dy, indexing='xy'), dim=-1)

    centers = landmarks_norm.to(feat_map.dtype).view(N, P, 1, 1, 2)
    grid = centers + dxdy.view(1, 1, ws, ws, 2)   # (N, P, ws, ws, 2)
    grid = grid.clamp(-1.0, 1.0)

    # grid_sample expects (N, H_out, W_out, 2) — merge P into H_out.
    grid_flat = grid.reshape(N, P * ws, ws, 2)
    sampled = F.grid_sample(feat_map, grid_flat, mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    # sampled: (N, C, P*ws, ws) → (N, P, C, ws, ws)
    return sampled.view(N, C, P, ws, ws).permute(0, 2, 1, 3, 4).contiguous()


class FragmentPathway(nn.Module):
    """
    FragNet-style local pathway adapted for per-landmark facial appearance:
    a small CNN — shared across all P landmarks — processes each landmark's
    raw 49×49 patch, laterally fused at three stages with a 3×3 window
    cropped from the corresponding scale of the global ResNet feature
    pyramid (paper Eq. 4: F_i = [M_i, A_i]).
    """
    def __init__(self, projection_dim: int = 512):
        super().__init__()
        # Stage 0 (pre-processing, no lateral fusion).
        self.conv0 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn0   = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(2)
        # Stage 1 — fused with a feat_l2 window (128 ch) after this stage.
        self.conv1 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2)
        # Stage 2 — input carries stage-1's 128-ch A_1 concat; fused with
        # a feat_l3 window (256 ch) after this stage.
        self.conv2 = nn.Conv2d(128 + 128, 256, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2)
        # Stage 3 — input carries stage-2's 256-ch A_2 concat; fused with
        # a feat_l4 window (512 ch) after this stage.
        self.conv3 = nn.Conv2d(256 + 256, 512, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2)
        # Head: M_3 (512 ch) concat A_3 (512 ch) → 1024 → projection_dim.
        self.head = nn.Linear(512 + 512, projection_dim)

    def forward(self, patches: Tensor, A1: Tensor, A2: Tensor, A3: Tensor) -> Tensor:
        """
        patches : (N, 3, 49, 49)   N = bs*T*P
        A1      : (N, 128, 3, 3)   cropped from feat_l2
        A2      : (N, 256, 3, 3)   cropped from feat_l3
        A3      : (N, 512, 3, 3)   cropped from feat_l4
        returns : (N, projection_dim)
        """
        x  = self.pool0(F.relu(self.bn0(self.conv0(patches))))       # (N, 64, 24, 24)

        M1 = self.pool1(F.relu(self.bn1(self.conv1(x))))             # (N, 128, 12, 12)
        A1_up = F.interpolate(A1, size=M1.shape[-2:], mode='bilinear', align_corners=False)
        F1 = torch.cat([M1, A1_up], dim=1)                           # (N, 256, 12, 12)

        M2 = self.pool2(F.relu(self.bn2(self.conv2(F1))))            # (N, 256, 6, 6)
        A2_up = F.interpolate(A2, size=M2.shape[-2:], mode='bilinear', align_corners=False)
        F2 = torch.cat([M2, A2_up], dim=1)                           # (N, 512, 6, 6)

        M3 = self.pool3(F.relu(self.bn3(self.conv3(F2))))            # (N, 512, 3, 3)
        F3 = torch.cat([M3, A3], dim=1)                              # (N, 1024, 3, 3)

        pooled = F3.mean(dim=(2, 3))                                 # (N, 1024)
        return self.head(pooled)                                     # (N, projection_dim)


# Grid-based FragNet (He & Schomaker 2020, writer-ID FragNet-ResNet50 —
# adapted for faces). No landmark guidance: fragments are a fixed spatial
# partition of the frame (vertical strips), laterally fused with a ResNet
# feature pyramid at each stage, in contrast to FragmentPathway above (which
# is landmark-guided).

class _ResNet50Pyramid(nn.Module):
    """
    Standalone ImageNet-pretrained ResNet-50 backbone exposing layer1..layer4
    feature maps. Used only by GridBasedFragNet(backbone_variant='resnet50') —
    no FER+ checkpoint exists at this depth, so this is always a separate,
    from-ImageNet encoder (not shared with PatchEncoder).
    """
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self):
        super().__init__()
        try:
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        except Exception as e:
            print(f'[GridBasedFragNet] Could not download ImageNet ResNet-50 '
                  f'weights ({e}); using random init.')
            net = tvm.resnet50(weights=None)
        self.conv1   = net.conv1
        self.bn1     = net.bn1
        self.relu    = net.relu
        self.maxpool = net.maxpool
        self.layer1  = net.layer1
        self.layer2  = net.layer2
        self.layer3  = net.layer3
        self.layer4  = net.layer4
        self.register_buffer('imagenet_mean', self._MEAN)
        self.register_buffer('imagenet_std',  self._STD)

    def forward(self, x: Tensor) -> list:
        """x : (N, 3, 224, 224) uint8 or float in [0, 1] -> [f1, f2, f3, f4]."""
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        x = (x - self.imagenet_mean) / self.imagenet_std
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); f1 = x   # (N,  256, 56, 56)
        x = self.layer2(x); f2 = x   # (N,  512, 28, 28)
        x = self.layer3(x); f3 = x   # (N, 1024, 14, 14)
        x = self.layer4(x); f4 = x   # (N, 2048,  7,  7)
        return [f1, f2, f3, f4]


class GridBasedFragNet(nn.Module):
    """
    Grid-based FragNet: the frame is split into NUM_FRAGMENTS fixed vertical
    strips (no landmark guidance, unlike FragmentPathway). A shared per-
    fragment CNN processes each strip, laterally fused at each stage with a
    crop of the corresponding scale from a ResNet feature pyramid — adapted
    from frag_net_resnet_50.py (He & Schomaker, writer identification).

    backbone_variant='resnet18': reuses the model's existing PatchEncoder
      (3-scale pyramid: layer2/layer3/layer4 = 128/256/512 ch). No extra
      backbone parameters.
    backbone_variant='resnet50': owns a separate ImageNet-pretrained
      ResNet-50 (4-scale pyramid: layer1..layer4 = 256/512/1024/2048 ch).
      Adds ~25M parameters.

    forward() returns (N, NUM_FRAGMENTS, projection_dim) per-fragment
    features — pooling to a single per-frame vector and broadcasting to
    per-landmark shape is the caller's job (GCNTransformerPaper.forward),
    matching how the 'broadcast' component treats global context.
    """
    NUM_FRAGMENTS = 5

    def __init__(self,
                 backbone_encoder: Optional[nn.Module] = None,
                 backbone_variant: str = 'resnet18',
                 projection_dim: int = 512):
        super().__init__()
        if backbone_variant not in ('resnet18', 'resnet50'):
            raise ValueError(f"backbone_variant must be resnet18|resnet50, got {backbone_variant!r}")
        self.backbone_variant = backbone_variant
        self.stage0 = self._conv_block(3, 64)

        if backbone_variant == 'resnet18':
            self.backbone_encoder = backbone_encoder   # shared PatchEncoder — not owned
            self.own_backbone     = None
            pyramid_channels = [128, 256, 512]          # layer2, layer3, layer4
            fuse_out_widths  = [128, 256, 512]
        else:
            self.backbone_encoder = None
            self.own_backbone     = _ResNet50Pyramid()
            pyramid_channels = [256, 512, 1024, 2048]   # layer1..layer4
            fuse_out_widths  = [128, 256, 512, 512]

        self.fuse_convs = nn.ModuleList()
        in_ch = 64
        for lateral_ch, out_ch in zip(pyramid_channels, fuse_out_widths):
            self.fuse_convs.append(self._conv_block(in_ch + lateral_ch, out_ch))
            in_ch = out_ch

        # ceil_mode=True: fragment crops at deep pyramid levels can be as
        # narrow as 1px wide (e.g. a 7-wide layer4 map split 5 ways) — plain
        # floor-mode pooling would collapse that to width 0 and crash.
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_ch, projection_dim)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _split_n(feat_map: Tensor, n: int, dim: int = 3) -> list:
        """Split feat_map into exactly n chunks along dim (sizes differ by
        at most 1) — unlike torch.chunk, which can silently return FEWER
        than n chunks when the dim size isn't evenly divisible."""
        size = feat_map.shape[dim]
        base, rem = divmod(size, n)
        sizes = [base + 1 if i < rem else base for i in range(n)]
        return list(torch.split(feat_map, sizes, dim=dim))

    def _pyramid(self, full_frames: Tensor) -> list:
        if self.backbone_variant == 'resnet18':
            feats = self.backbone_encoder.forward_feature_pyramid(full_frames)
            return [feats['layer2'], feats['layer3'], feats['layer4']]
        return self.own_backbone(full_frames)   # [layer1, layer2, layer3, layer4]

    def forward(self, full_frames: Tensor, precomputed_pyramid: Optional[list] = None) -> Tensor:
        """
        full_frames        : (N, 3, 224, 224)  N = bs*T
        precomputed_pyramid: optional [layer2, layer3, layer4] already computed
                              by the caller (resnet18 variant only) — avoids a
                              redundant ResNet-18 forward when other components
                              (fragnet_landmark / multiscale_roi) already ran it.
        returns             : (N, NUM_FRAGMENTS, projection_dim)
        """
        pyramid = precomputed_pyramid if precomputed_pyramid is not None else self._pyramid(full_frames)

        # Stage 0 trains from scratch (no pretrained checkpoint), so only
        # scale to [0, 1] here — the pyramid backbone normalizes internally.
        x_in = full_frames.float() / 255.0 if full_frames.dtype == torch.uint8 else full_frames.float()
        frags = self._split_n(x_in, self.NUM_FRAGMENTS)                 # 5 x (N,3,224,~45)
        m = [self.pool(self.stage0(f)) for f in frags]                  # 5 x (N,64,112,~22)

        for lateral_feat, conv in zip(pyramid, self.fuse_convs):
            lateral_frags = self._split_n(lateral_feat, self.NUM_FRAGMENTS)
            new_m = []
            for mi, li in zip(m, lateral_frags):
                mi_up = F.interpolate(mi, size=li.shape[-2:], mode='bilinear', align_corners=False)
                fused = torch.cat([mi_up, li], dim=1)
                new_m.append(self.pool(conv(fused)))
            m = new_m

        pooled  = [self.gap(mi).flatten(1) for mi in m]                 # 5 x (N, C_last)
        stacked = torch.stack(pooled, dim=1)                            # (N, NUM_FRAGMENTS, C_last)
        return self.head(stacked)                                       # (N, NUM_FRAGMENTS, projection_dim)


# 2D-FAN 68-point cluster assignments

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


# Building blocks

class TGConv(nn.Module):
    def __init__(self, channels: int, num_joints: int, kernel_size: int = 5):
        super().__init__()
        #Creates a learnable adjacency matrix. It starts as an Identity matrix
        #(each joint only looks at itself)
        #but evolves during training to learn which joints should influence each other.
        self.A    = nn.Parameter(torch.eye(num_joints))
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn   = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        bs, P, C, T = x.shape
        #Normalizes the learnable adjacency matrix so each row sums to 1.
        #This ensures the "influence" of neighboring joints is scaled correctly.
        A     = torch.softmax(self.A, dim=-1)
        #This is the Graph Convolution step.
        #It multiplies the adjacency matrix A with the input x.
        #It mixes information across the Joint dimension (j to i),
        #essentially aggregating features from "neighboring" joints.
        x_agg = torch.einsum('ij,bjct->bict', A, x)
        xf    = x_agg.reshape(bs * P, C, T)
        xf    = self.bn(self.conv(xf))
        return xf.reshape(bs, P, C, T)


class TGConvOnEdges(nn.Module):
    """Temporal graph convolution using external same-joint temporal edges.

    Replaces the learnable joint-mixing matrix A of the original TGConv with
    actual temporal graph convolution via PyG's SGConv on temporal edges.
    Temporal edges connect joint i at frame t to joint i at frame t+1 (same
    joint across consecutive frames), provided externally via edge_index_temporal.

    Node indexing: T-major (node = t*P + i), consistent with the dataloader's
    build_temporal_edges and the model's (bs, T, P, C) reshape convention.

    The Conv1d temporal convolution along the time dimension is preserved from
    the original TGConv — only the joint-mixing path is replaced.
    """
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        # bias=False: bn_g provides the learnable per-channel shift (beta)
        self.sgcn  = _PyGSGConv(channels, channels, K=2, bias=False)
        self.bn_g  = nn.BatchNorm1d(channels)
        self.conv  = nn.Conv1d(channels, channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn    = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor, edge_index_temporal: Tensor) -> Tensor:
        # x : (bs, P, C, T)
        # edge_index_temporal : (2, E_t) — T-major node indices (batched)
        bs, P, C, T = x.shape

        # Temporal graph conv: aggregate same-joint features across frames.
        # Reshape to (bs*T*P, C) using T-major order to match edge_index convention.
        x_flat = x.permute(0, 3, 1, 2).reshape(bs * T * P, C)  # (bs,P,C,T)→(bs,T,P,C)→flat
        x_flat = self.bn_g(self.sgcn(x_flat, edge_index_temporal))
        # Reshape back to (bs, P, C, T)
        x_agg = x_flat.reshape(bs, T, P, C).permute(0, 2, 3, 1).contiguous()

        # Temporal Conv1d (preserved from original TGConv)
        xf = x_agg.reshape(bs * P, C, T)
        xf = self.bn(self.conv(xf))
        return xf.reshape(bs, P, C, T)


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, num_joints: int,
                 kernel_size: int = 5, dropout: float = 0.1,
                 tgconv_mode: str = 'learnable_A'):
        super().__init__()
        self.tgconv_mode = tgconv_mode
        self.conv1   = nn.Conv1d(channels, channels, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.bn1     = nn.BatchNorm1d(channels)
        if tgconv_mode == 'learnable_A':
            self.tg_conv = TGConv(channels, num_joints, kernel_size)
        elif tgconv_mode == 'temporal_edges':
            self.tg_conv = TGConvOnEdges(channels, kernel_size)
        else:
            raise ValueError(f"Unknown tgconv_mode: {tgconv_mode!r}")
        self.conv2   = nn.Conv1d(channels, channels, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.bn2     = nn.BatchNorm1d(channels)
        self.relu    = nn.ReLU()
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index_temporal: Optional[Tensor] = None) -> Tensor:
        bs, P, C, T = x.shape
        residual = x
        xf = x.view(bs * P, C, T)
        xf = self.relu(self.bn1(self.conv1(xf))).view(bs, P, C, T)
        if self.tgconv_mode == 'learnable_A':
            xf = self.relu(self.tg_conv(xf))
        else:
            assert edge_index_temporal is not None, \
                "edge_index_temporal required for tgconv_mode='temporal_edges'"
            xf = self.relu(self.tg_conv(xf, edge_index_temporal))
        xf = xf.view(bs * P, C, T)
        xf = self.relu(self.bn2(self.conv2(xf))).view(bs, P, C, T)
        return self.drop(xf) + residual
    
class SGConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden: tuple = None):
        super().__init__()
        mid1, mid2 = hidden if hidden else (out_channels, out_channels)
        self.lin1     = nn.Linear(in_channels, mid1, bias=False)
        self.bn1      = nn.BatchNorm1d(mid1)
        self.sgcn     = _PyGSGConv(mid1, mid2, K=2)
        self.bn_sg    = nn.BatchNorm1d(mid2)
        self.lin2     = nn.Linear(mid2, out_channels, bias=False)
        self.bn2      = nn.BatchNorm1d(out_channels)
        self.relu     = nn.ReLU()
        self.residual = nn.Linear(in_channels, out_channels, bias=False) \
                        if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        residual = self.residual(x)
        x = self.relu(self.bn1(self.lin1(x)))
        x = self.relu(self.bn_sg(self.sgcn(x, edge_index)))
        x = self.relu(self.bn2(self.lin2(x)))
        return x + residual
    
class ComponentAttention(nn.Module):
    """
    Clusters joints -> avg+max pool per cluster -> 1x1 conv -> softmax -> rescale.
    """
    def __init__(self, channels: int, num_clusters: int = 7):
        super().__init__()
        self.num_clusters = num_clusters
        self.conv = nn.Conv1d(2, 1, kernel_size=1, bias=False)

    def forward(self, x: Tensor, cluster_ids: Tensor) -> Tensor:
        # x : (bs, P, C, T)
        bs, P, C, T = x.shape
        nc = self.num_clusters

        # Cluster-level features: (bs, nc, C, T)
        cm = torch.zeros(bs, nc, C, T, device=x.device)
        for c in range(nc):
            mask = (cluster_ids == c)
            if mask.sum() > 0:
                cm[:, c] = x[:, mask].mean(dim=1)

        # CAP / CMP over channel dim → (bs, nc, T)
        cap = cm.mean(dim=2)
        cmp = cm.max(dim=2).values

        # Reshape for conv: (bs*T, 2, nc) → (bs*T, 1, nc)
        cap_t = cap.permute(0, 2, 1).reshape(bs * T, nc)
        cmp_t = cmp.permute(0, 2, 1).reshape(bs * T, nc)
        #(bs*T, 2, nc) → conv → (bs*T, 1, nc) → softmax → (bs*T, 1, nc) → squeeze → (bs*T, nc) → view → (bs, T, nc)
        attn  = F.softmax(
            self.conv(torch.stack([cap_t, cmp_t], dim=1)), dim=-1
        ).squeeze(1).view(bs, T, nc)   # (bs, T, nc)

        # Map cluster scores back to joints: (bs, P, 1, T)
        ja = attn[:, :, cluster_ids].permute(0, 2, 1).unsqueeze(2) #Each joint gets the attention score of its cluster:
        return x * ja + x   # residual ,# (bs, P, C, T) * (bs, P, 1, T) 


class FrameAttention(nn.Module):
    """
    Temporal frame attention (paper Fig. 6).
    SAP+SMP across spatial/channel dims -> 1x1 conv -> softmax -> rescale frames.
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
        attn = F.softmax(
            self.conv(torch.stack([sap, smp], dim=1)), dim=-1
        )   # (bs, 1, T)
        return x * attn.unsqueeze(1) + x   # residual


class CoordBranch(nn.Module):
    """
    One branch of the triple coordinate encoder.
    Paper: 1 spatial GCN block with channels [64, 128, 256] + temporal conv (k=5).
    """
    def __init__(self, num_joints: int, dropout: float = 0.1, kernel_size: int = 5,
                 tgconv_mode: str = 'learnable_A'):
        super().__init__()
        self.sgconv = SGConv(2, 256, hidden=(64, 128))
        self.temp   = TemporalConvBlock(256, num_joints, kernel_size, dropout,
                                        tgconv_mode=tgconv_mode)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x_flat: Tensor, edge_index: Tensor,
                bs: int, T: int, P: int,
                edge_index_temporal: Optional[Tensor] = None) -> Tensor:
        # x_flat : (bs*T*P, 2)
        x = self.drop(self.sgconv(x_flat, edge_index))
        # (bs*T*P, 256) → (bs, P, 256, T)
        x = x.view(bs, T, P, 256).permute(0, 2, 3, 1).contiguous()
        return self.temp(x, edge_index_temporal)   # (bs, P, 256, T)


class GCBlock(nn.Module):
    """
    Main processor GC block (paper Fig. 4):
      Spatial GCN -> Component Attention -> Temporal Conv -> Frame Attention

    temporal_edges_mode controls how temporal edges are used in message passing:
      'none'               – spatial-only (original behaviour).
      'combined'           – edge_index already contains merged spatial+temporal
                             edges; a single sgconv handles them uniformly.
      'separated_parallel' – two independent SGConv branches (spatial + temporal)
                             whose outputs are summed before downstream layers.
      'separated_sequential'– spatial SGConv first, then temporal SGConv applied
                             sequentially on the output.
    """
    def __init__(self, channels: int, num_joints: int, cluster_ids: Tensor,
             dropout: float = 0.1, kernel_size: int = 5,
             temporal_edges_mode: str = 'none',
             tgconv_mode: str = 'learnable_A'):
        super().__init__()
        self.temporal_edges_mode = temporal_edges_mode
        self.sgconv = SGConv(512, 512)
        if temporal_edges_mode in ('separated_parallel', 'separated_sequential'):
            self.sgconv_temporal = SGConv(512, 512)
        # 'construct_only': edges are produced by the dataloader and consumed
        # by TGConvOnEdges, but GCBlock's SGConv path stays spatial-only.
        self.cattn = ComponentAttention(channels, 7)
        self.tconv = TemporalConvBlock(channels, num_joints, kernel_size, dropout,
                                       tgconv_mode=tgconv_mode)
        self.fattn = FrameAttention(channels)
        self.drop  = nn.Dropout(dropout)
        self.register_buffer('cluster_ids', cluster_ids)

    def forward(self, x: Tensor, edge_index: Tensor,
                bs: int, T: int, P: int,
                edge_index_temporal: Optional[Tensor] = None) -> Tensor:
        residual = x
        xf = x.permute(0, 3, 1, 2).reshape(bs * T * P, -1)

        if self.temporal_edges_mode == 'separated_parallel':
            # Two independent branches; sum outputs.
            xf = self.drop(
                self.sgconv(xf, edge_index) + self.sgconv_temporal(xf, edge_index_temporal)
            )
        elif self.temporal_edges_mode == 'separated_sequential':
            # Spatial first, then temporal on the result.
            xf = self.sgconv(xf, edge_index)
            xf = self.drop(self.sgconv_temporal(xf, edge_index_temporal))
        else:
            # 'none' and 'combined': single sgconv (combined edge_index or spatial-only).
            xf = self.drop(self.sgconv(xf, edge_index))

        x  = xf.view(bs, T, P, -1).permute(0, 2, 3, 1).contiguous()
        x = self.cattn(x, self.cluster_ids)
        x = self.tconv(x, edge_index_temporal) + residual
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


# Center Loss

class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        return ((features - self.centers[labels]) ** 2).sum(dim=1).mean()


# Full Paper Model — 2D-FAN version

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
                 seq_length      : int,
                 n_heads         : int   = 8,
                 dim_feedforward : int   = 1024,
                 num_gc_blocks   : int   = 1,
                 num_trans_layers: int   = 1,
                 out_dim         : int   = 7,
                 kept_indices    : list  = None,
                 dropout         : float = 0.1,
                 use_patches     : bool  = True,
                 # In-model patch encoder (end-to-end / partial fine-tune)
                 encoder_mode    : str             = 'none',   # 'none' | 'frozen' | 'partial' | 'full'
                 encoder_ckpt    : Optional[str]   = None,
                 encoder_backbone: str             = 'resnet34',  # 'resnet18' | 'resnet34'
                 encoder_imagenet_norm: Optional[bool] = None,   # None = auto based on backbone
                 upscale_patches_to_224: bool = False,
                 freeze_encoder_bn: bool = False,
                 # Global appearance mode (legacy single-mode shortcut)
                 global_appearance_mode: str = 'none',  # 'none' | 'broadcast' | 'roi_align' | ...
                 # Six independent, concatenable appearance components
                 # Passing ANY of these five as non-None switches the model into
                 # "multi-component mode": global_appearance_mode is then ignored
                 # entirely and each component is toggled independently, with
                 # outputs concatenated at the fusion layer. Leaving all five as
                 # None (the default) preserves the exact legacy single-mode
                 # behaviour driven by global_appearance_mode above — this is
                 # what lets old call sites (e.g. smoke_fragnet.py) keep working
                 # unmodified.
                 use_fragnet_landmark: Optional[bool] = None,   # component 1: FragmentPathway
                 use_broadcast       : Optional[bool] = None,   # component 2
                 use_fragnet_grid    : Optional[bool] = None,   # component 3: GridBasedFragNet
                 use_local_encoder   : Optional[bool] = None,   # component 5
                 use_multiscale_roi  : Optional[bool] = None,   # component 6
                 fragnet_grid_backbone: str = 'resnet18',        # 'resnet18' | 'resnet50' — component 3 only
                 # Spatial-temporal graph construction
                 temporal_edges_mode: str = 'none',   # 'none'|'combined'|'separated_parallel'|'separated_sequential'
                 tgconv_mode: str = 'learnable_A',    # 'learnable_A'|'temporal_edges'
                 # Whole-component ablation
                 use_coord_encoder: bool = True,
                 use_gcblocks     : bool = True,
                 use_transformers : bool = True):
        super().__init__()

        if not use_coord_encoder and not use_patches:
            raise ValueError(
                "use_coord_encoder=False requires use_patches=True — otherwise "
                "the model has no input features at all."
            )

        self.nb_joint               = nb_joint
        self.seq_length             = seq_length
        self.use_patches            = use_patches
        self.num_gc_blocks          = num_gc_blocks
        self.encoder_mode           = encoder_mode
        self.global_appearance_mode = global_appearance_mode
        self.temporal_edges_mode    = temporal_edges_mode
        self.tgconv_mode            = tgconv_mode
        self.use_coord_encoder      = use_coord_encoder
        self.use_gcblocks           = use_gcblocks
        self.use_transformers       = use_transformers

        # Multi-component appearance mode
        # Active iff any of the five new flags was explicitly passed (not
        # left at its None default). See the __init__ signature comment.
        self._multi_component_mode = any(
            v is not None for v in (use_fragnet_landmark, use_broadcast,
                                     use_fragnet_grid, use_local_encoder,
                                     use_multiscale_roi)
        )
        self.use_fragnet_landmark  = bool(use_fragnet_landmark)
        self.use_broadcast         = bool(use_broadcast)
        self.use_fragnet_grid      = bool(use_fragnet_grid)
        self.use_local_encoder     = bool(use_local_encoder)
        self.use_multiscale_roi    = bool(use_multiscale_roi)
        self.fragnet_grid_backbone = fragnet_grid_backbone

        if self._multi_component_mode and not use_coord_encoder and not any([
                self.use_fragnet_landmark, self.use_broadcast, self.use_fragnet_grid,
                self.use_local_encoder, self.use_multiscale_roi]):
            raise ValueError(
                "Multi-component mode: at least one of the six components "
                "(coord encoder + five appearance flags) must be on."
            )

        # In-model patch encoder (optional)
        # encoder_mode='none'    : no encoder; forward expects pre-computed
        #                          512-d patch_features (cached pipeline).
        # encoder_mode='frozen'  : encoder in model, eval()+no_grad.
        # encoder_mode='partial' : layer4 trainable, rest frozen.
        # encoder_mode='full'    : all encoder params trainable.
        # encoder_backbone       : 'resnet34' (FER+) or 'resnet18' (MS-Celeb-1M)
        #
        # In multi-component mode, only build it when some active component
        # actually consumes it — keeps parameter counts clean when e.g. only
        # fragnet_grid(resnet50) is on (Step 4/9 of the task spec).
        if self._multi_component_mode:
            _needs_shared_encoder = (
                self.use_fragnet_landmark or self.use_broadcast or
                self.use_local_encoder or self.use_multiscale_roi or
                (self.use_fragnet_grid and fragnet_grid_backbone == 'resnet18')
            )
        else:
            _needs_shared_encoder = True   # legacy: unconditional on use_patches, as before

        if encoder_mode != 'none' and use_patches and _needs_shared_encoder:
            self.patch_encoder = PatchEncoder(
                checkpoint_path=encoder_ckpt,
                mode=encoder_mode,
                backbone=encoder_backbone,
                apply_imagenet_norm=encoder_imagenet_norm,
                upscale_to_224=upscale_patches_to_224,
                freeze_bn_stats=freeze_encoder_bn,
            )
        else:
            self.patch_encoder = None

        # FragNet-style dual-pathway fragment CNN (optional; component 1)
        # Shared across all P landmarks; laterally fused with the encoder's
        # feature pyramid at three stages. Parameters land in "head_params"
        # (not "encoder_params") in encoder_head_param_groups() since it's
        # a separate module from self.patch_encoder.
        _want_fragnet_landmark = (self.use_fragnet_landmark if self._multi_component_mode
                                   else global_appearance_mode == 'fragnet')
        if _want_fragnet_landmark:
            self.fragment_pathway = FragmentPathway(projection_dim=512)
        else:
            self.fragment_pathway = None

        # Grid-based FragNet (optional; component 3 — multi-component mode only)
        if self._multi_component_mode and self.use_fragnet_grid:
            self.fragnet_grid = GridBasedFragNet(
                backbone_encoder=(self.patch_encoder if fragnet_grid_backbone == 'resnet18' else None),
                backbone_variant=fragnet_grid_backbone,
                projection_dim=512,
            )
        else:
            self.fragnet_grid = None

        # Cluster assignments (2D-FAN indices)
        kept_list   = list(kept_indices) if kept_indices is not None \
                      else list(range(nb_joint))
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

        # Triple coordinate encoder (optional — whole-component ablation)
        if self.use_coord_encoder:
            self.branch_R = CoordBranch(nb_joint, dropout, tgconv_mode=tgconv_mode)
            self.branch_C = CoordBranch(nb_joint, dropout, tgconv_mode=tgconv_mode)
            self.branch_T = CoordBranch(nb_joint, dropout, tgconv_mode=tgconv_mode)
        else:
            self.branch_R = self.branch_C = self.branch_T = None

        if self._multi_component_mode:
            self.APPEARANCE_DIM = (
                (512 if self.use_fragnet_landmark else 0) +
                (512 if self.use_broadcast        else 0) +
                (512 if self.use_fragnet_grid     else 0) +
                (512 if self.use_local_encoder    else 0) +
                (896 if self.use_multiscale_roi   else 0)
            )
        elif global_appearance_mode == 'roi_align_multiscale':
            self.APPEARANCE_DIM = 128 + 256 + 512        # 896
        elif global_appearance_mode == 'combined':
            self.APPEARANCE_DIM = 1024
        elif global_appearance_mode == 'local_plus_multiscale':
            self.APPEARANCE_DIM = 512 + 896   # local patches + multi-scale ROI-Align
        else:
            self.APPEARANCE_DIM = 512

        self.FUSION_DIM = (3 * 256 + self.APPEARANCE_DIM) if self.use_coord_encoder \
                          else self.APPEARANCE_DIM

        # Input projection FUSION_DIM → 512
        self.input_proj = nn.Sequential(
            nn.Linear(self.FUSION_DIM, self.MAIN_CHANNELS),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Main processor (GCBlocks / transformers — whole-component ablation)
        if self.use_gcblocks:
            self.gc_blocks = nn.ModuleList([
                GCBlock(self.MAIN_CHANNELS, nb_joint, cluster_ids, dropout,
                        temporal_edges_mode=temporal_edges_mode,
                        tgconv_mode=tgconv_mode)
                for _ in range(num_gc_blocks)
            ])
        else:
            self.gc_blocks = nn.ModuleList()

        if self.use_transformers:
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
        else:
            self.spatial_trans  = nn.ModuleList()
            self.temporal_trans = nn.ModuleList()

        # Learnable positional embedding (paper Eq. 7, applied once)
        #self.pos_embed = nn.Parameter(torch.zeros(1, nb_joint, self.MAIN_CHANNELS))
        self.pos_embed = nn.Parameter(torch.zeros(1, nb_joint, self.MAIN_CHANNELS, seq_length))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Classifier
        self.pre_logit = nn.Sequential(
            nn.Linear(self.MAIN_CHANNELS, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head = nn.Linear(256, out_dim)

    # Coordinate transforms (paper Eqs. 1-4)

    def encoder_head_param_groups(self):
        """
        Split trainable params into (encoder_params, head_params) lists, so
        the trainer can apply a different LR to each.
        Returns (encoder_params, head_params).
        If self.patch_encoder is None or 'frozen', encoder_params is empty.
        """
        encoder_params, head_params = [], []
        enc_param_ids = set()
        if self.patch_encoder is not None and self.encoder_mode != 'frozen':
            for p in self.patch_encoder.trainable_parameters():
                encoder_params.append(p)
                enc_param_ids.add(id(p))
        for p in self.parameters():
            if p.requires_grad and id(p) not in enc_param_ids:
                head_params.append(p)
        return encoder_params, head_params

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

    # Forward

    def forward(self, x: Tensor, edge_index: Tensor,
                batch_vec: Tensor, bs: int,
                patch_features: Optional[Tensor] = None,
                patches: Optional[Tensor] = None,
                full_frames: Optional[Tensor] = None,
                edge_index_temporal: Optional[Tensor] = None):
        """
        x                   : (bs*T*P, 2)
        edge_index          : (2, E)  — batched PyG edge index (spatial, or combined)
        batch_vec           : PyG batch vector (unused, kept for API compat.)
        bs                  : actual batch size
        patch_features      : (bs*T*P, 512) — pre-computed ResNet-34 features
        patches             : (bs*T*P, 3, 49, 49) — raw patches for in-model encoder
        full_frames         : (bs*T, 3, 224, 224) — full face crops
        edge_index_temporal : (2, E_t) — temporal edges (separated_* modes only)
        """
        P = self.nb_joint
        T = self.seq_length

        # 1. Coordinate representations R, C, T
        # coords is needed unconditionally: the ROI-Align / multiscale / fragnet
        # appearance branches below sample at these image-space positions even
        # when the geometry encoder itself is ablated off.
        coords = x.view(bs, T, P, 2)

        if self.use_coord_encoder:
            R      = self._compute_R(coords)
            C      = self._compute_C(R)
            Tau    = self._compute_T(R)

            # 2. Triple coordinate encoder (each branch: 2→64→128→256)
            gR = self.branch_R(R.reshape(bs*T*P, 2),   edge_index, bs, T, P,
                               edge_index_temporal=edge_index_temporal)
            gC = self.branch_C(C.reshape(bs*T*P, 2),   edge_index, bs, T, P,
                               edge_index_temporal=edge_index_temporal)
            gT = self.branch_T(Tau.reshape(bs*T*P, 2), edge_index, bs, T, P,
                               edge_index_temporal=edge_index_temporal)
            g  = torch.cat([gR, gC, gT], dim=2)   # (bs, P, 768, T)

        # 3. Appearance features
        if self._multi_component_mode:
            v = self._forward_multi_component(full_frames, patches, coords, bs, T, P)
        elif self.use_patches:
            if (self.global_appearance_mode == 'combined'
                    and full_frames is not None
                    and self.patch_encoder is not None):
                # Combined mode: single encoder forward → both ROI-Align and
                # global-pooled broadcast from the same 7×7 feature map.
                # No second encoder pass — pooling the map is a free operation.
                fmap = self.patch_encoder.forward_feature_map(
                    full_frames.view(bs * T, 3, 224, 224))  # (bs*T, 512, 7, 7)

                # ROI-Align path (identical to roi_align mode)
                coords_flat = coords.view(bs * T, P, 2).float()
                coords_norm = (coords_flat / 224.0) * 2.0 - 1.0
                coords_norm = coords_norm.clamp(-1.0, 1.0)
                grid    = coords_norm.unsqueeze(1)                              # (bs*T, 1, P, 2)
                sampled = F.grid_sample(
                    fmap, grid,
                    mode='bilinear', padding_mode='zeros', align_corners=False,
                )
                sampled = sampled.squeeze(2)                                    # (bs*T, 512, P)
                v_roi   = sampled.view(bs, T, 512, P).permute(0, 3, 2, 1).contiguous()
                # v_roi: (bs, P, 512, T)

                # Broadcast path: global avg-pool over the (7,7) spatial dims
                v_broadcast_flat = fmap.mean(dim=(2, 3))                        # (bs*T, 512)
                v_broadcast = v_broadcast_flat.view(bs, T, 512)
                v_broadcast = v_broadcast.unsqueeze(2).expand(-1, -1, P, -1)   # (bs, T, P, 512)
                v_broadcast = v_broadcast.permute(0, 2, 3, 1).contiguous()
                # v_broadcast: (bs, P, 512, T)

                v = torch.cat([v_roi, v_broadcast], dim=2)                     # (bs, P, 1024, T)

            elif (self.global_appearance_mode == 'roi_align_multiscale'
                    and full_frames is not None
                    and self.patch_encoder is not None):
                # Multi-scale ROI-Align: one encoder forward produces feature maps
                # at three pyramid levels; bilinearly sample each landmark's feature
                # at all three scales and concatenate → 896-d per landmark.
                # grid_sample with normalized coords is scale-invariant: (112,112)
                # maps to (0,0) regardless of the feature map's spatial size.
                full_flat = full_frames.view(bs * T, 3, 224, 224)
                feats = self.patch_encoder.forward_feature_pyramid(full_flat)

                coords_flat = coords.view(bs * T, P, 2).float()    # (bs*T, P, 2)
                coords_norm = (coords_flat / 224.0) * 2.0 - 1.0
                coords_norm = coords_norm.clamp(-1.0, 1.0)
                grid = coords_norm.unsqueeze(1)                     # (bs*T, 1, P, 2)

                v_l2 = F.grid_sample(feats['layer2'], grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=False)
                v_l3 = F.grid_sample(feats['layer3'], grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=False)
                v_l4 = F.grid_sample(feats['layer4'], grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=False)
                # Shapes: (bs*T, C, 1, P) — squeeze height dim
                v_l2 = v_l2.squeeze(2)   # (bs*T, 128, P)
                v_l3 = v_l3.squeeze(2)   # (bs*T, 256, P)
                v_l4 = v_l4.squeeze(2)   # (bs*T, 512, P)
                v_concat = torch.cat([v_l2, v_l3, v_l4], dim=1)    # (bs*T, 896, P)

                v = v_concat.view(bs, T, 896, P).permute(0, 3, 2, 1).contiguous()
                # v: (bs, P, 896, T)

            elif (self.global_appearance_mode == 'broadcast'
                    and full_frames is not None
                    and self.patch_encoder is not None):
                # Broadcast mode: encode once per frame, broadcast to all landmarks.
                # full_frames: (bs*T, 3, 224, 224) — collated by PyG DataLoader
                feats_per_frame = self.patch_encoder(full_frames)          # (bs*T, 512)
                feats_per_frame = feats_per_frame.view(bs, T, 512)
                v = feats_per_frame.unsqueeze(2).expand(-1, -1, P, -1)    # (bs, T, P, 512)
                v = v.permute(0, 2, 3, 1).contiguous()                    # (bs, P, 512, T)
            elif (self.global_appearance_mode == 'roi_align'
                    and full_frames is not None
                    and self.patch_encoder is not None):
                # ROI-Align mode: encode once per frame to get a 7×7 feature map,
                # then bilinearly sample a distinct 512-d feature at each landmark's
                # image-space coordinates. Preserves both global context (full-face
                # receptive field) and per-landmark distinction (different spatial
                # positions sample different cells of the feature map).
                #
                # Coordinate convention: coords is in 224-pixel space (pre-R-norm).
                # grid_sample with align_corners=False maps pixel p to
                # normalised coord (p/224)*2 - 1, so a landmark at the image centre
                # (112,112) maps to (0,0), which bilinearly averages the 4 centre
                # cells of the 7×7 map. This is verified by smoke_global_roi_align.py.
                fmap = self.patch_encoder.forward_feature_map(full_frames)  # (bs*T, 512, 7, 7)

                # coords: (bs, T, P, 2) in 224-pixel space — computed above.
                coords_flat = coords.view(bs * T, P, 2).float()              # (bs*T, P, 2)
                coords_norm = (coords_flat / 224.0) * 2.0 - 1.0             # → [-1, +1]
                # Clamp: landmarks can drift slightly outside [0,223] after augmentation
                coords_norm = coords_norm.clamp(-1.0, 1.0)
                # grid_sample expects (N, H_out, W_out, 2) with (x,y) last.
                # One "output pixel" per landmark → H_out=1, W_out=P.
                grid = coords_norm.unsqueeze(1)                              # (bs*T, 1, P, 2)

                # Bilinearly sample the feature map at each landmark position.
                sampled = F.grid_sample(
                    fmap, grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False,
                )                                                            # (bs*T, 512, 1, P)
                sampled = sampled.squeeze(2)                                 # (bs*T, 512, P)
                sampled = sampled.view(bs, T, 512, P)
                v = sampled.permute(0, 3, 2, 1).contiguous()                # (bs, P, 512, T)
            elif (self.global_appearance_mode == 'local_plus_multiscale'
                    and full_frames is not None
                    and patches is not None
                    and self.patch_encoder is not None):
                # Local + multi-scale: both appearance pathways run through the
                # SAME PatchEncoder instance (no separate encoder is created) —
                # once on 49×49 patches, once on 224×224 frames up to layer4.
                # Local pathway (identical to the 'none'-mode code path).
                v_local_flat = self.patch_encoder(patches)                   # (bs*T*P, 512)
                v_local = v_local_flat.view(bs, T, P, 512).permute(0, 2, 3, 1).contiguous()
                # v_local: (bs, P, 512, T)

                # Multi-scale pathway (identical to the roi_align_multiscale path).
                full_flat = full_frames.view(bs * T, 3, 224, 224)
                feats = self.patch_encoder.forward_feature_pyramid(full_flat)

                coords_flat = coords.view(bs * T, P, 2).float()
                coords_norm = (coords_flat / 224.0) * 2.0 - 1.0
                coords_norm = coords_norm.clamp(-1.0, 1.0)
                grid = coords_norm.unsqueeze(1)

                v_l2 = F.grid_sample(feats['layer2'], grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=False).squeeze(2)
                v_l3 = F.grid_sample(feats['layer3'], grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=False).squeeze(2)
                v_l4 = F.grid_sample(feats['layer4'], grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=False).squeeze(2)
                v_multi_concat = torch.cat([v_l2, v_l3, v_l4], dim=1)        # (bs*T, 896, P)
                v_multi = v_multi_concat.view(bs, T, 896, P).permute(0, 3, 2, 1).contiguous()
                # v_multi: (bs, P, 896, T)

                v = torch.cat([v_local, v_multi], dim=2)                     # (bs, P, 1408, T)
            elif (self.global_appearance_mode == 'fragnet'
                    and full_frames is not None
                    and patches is not None
                    and self.patch_encoder is not None
                    and self.fragment_pathway is not None):
                # FragNet-style dual pathway (He & Schomaker 2020): a small
                # shared CNN processes each landmark's raw 49×49 patch,
                # laterally fused at three stages with windows cropped from
                # the SAME encoder's feature pyramid (no separate encoder —
                # identical pyramid computation to roi_align_multiscale).
                full_flat = full_frames.view(bs * T, 3, 224, 224)
                feats = self.patch_encoder.forward_feature_pyramid(full_flat)

                coords_flat = coords.view(bs * T, P, 2).float()
                coords_norm = (coords_flat / 224.0) * 2.0 - 1.0
                coords_norm = coords_norm.clamp(-1.0, 1.0)

                A1 = sample_window(feats['layer2'], coords_norm, window_size=3)   # (bs*T, P, 128, 3, 3)
                A2 = sample_window(feats['layer3'], coords_norm, window_size=3)   # (bs*T, P, 256, 3, 3)
                A3 = sample_window(feats['layer4'], coords_norm, window_size=3)   # (bs*T, P, 512, 3, 3)

                A1 = A1.reshape(bs * T * P, 128, 3, 3)
                A2 = A2.reshape(bs * T * P, 256, 3, 3)
                A3 = A3.reshape(bs * T * P, 512, 3, 3)

                # FragmentPathway trains from scratch (no pretrained checkpoint),
                # so only scale to [0,1] — no ImageNet normalisation needed.
                patches_in = (patches.float() / 255.0 if patches.dtype == torch.uint8
                             else patches.float())
                v_frag = self.fragment_pathway(patches_in, A1, A2, A3)           # (bs*T*P, 512)
                v = v_frag.view(bs, T, P, 512).permute(0, 2, 3, 1).contiguous()  # (bs, P, 512, T)
            elif patches is not None and self.patch_encoder is not None:
                # Local-patch mode: encode each landmark patch individually.
                # patches : (bs*T*P, 3, 49, 49)  -> (bs*T*P, 512)
                feats = self.patch_encoder(patches)
                v = feats.view(bs, T, P, 512).permute(0, 2, 3, 1)
            elif patch_features is not None:
                v = patch_features.float().view(bs, T, P, 512).permute(0, 2, 3, 1)
            else:
                v = torch.zeros(bs, P, 512, T, device=x.device)
        else:
            v = torch.zeros(bs, P, 512, T, device=x.device)

        # 4. Early fusion u = [g, v] → FUSION_DIM
        # (v can be None only in multi-component mode with every appearance
        # component off — legal iff use_coord_encoder is on, enforced in __init__.)
        if self.use_coord_encoder and v is not None:
            u = torch.cat([g, v], dim=2)   # (bs, P, FUSION_DIM, T)
        elif self.use_coord_encoder:
            u = g                          # appearance-free: geometry only
        else:
            u = v                          # geometry-free: appearance only

        # 5. Project 1280 → 512
        bs_, P_, Cin_, T_ = u.shape
        u = self.input_proj(
            u.permute(0, 1, 3, 2).reshape(-1, Cin_)
        ).reshape(bs_, P_, T_, self.MAIN_CHANNELS).permute(0, 1, 3, 2).contiguous()
        # u : (bs, P, 512, T)

        # 6. Main processor: iterative GCBlock + Spatial/Temporal Transformer
        # (each sub-step is individually skippable via the whole-component
        # ablation flags; skipped steps pass u through unchanged, so pooling
        # and the classifier below need no special-casing.)
        pos_added = False
        for i in range(self.num_gc_blocks):
            if self.use_gcblocks:
                u = self.gc_blocks[i](u, edge_index, bs, T, P,
                                      edge_index_temporal=edge_index_temporal)

            # Add learnable positional embedding ONCE before first transformer (Eq. 7)
            if not pos_added:
                # pos_embed : (1, P, 512) — broadcast over bs and T
                #u = u + self.pos_embed.unsqueeze(-1)   # (bs, P, 512, T)
                u = u + self.pos_embed #(1, P, 512, T)
                pos_added = True

            if self.use_transformers:
                ti = min(i, len(self.spatial_trans) - 1)
                u  = self.spatial_trans[ti](u)
                u  = self.temporal_trans[ti](u)

        # 7. Global mean pooling over joints and frames
        feat = u.mean(dim=[1, 3])   # (bs, 512)

        # 8. Classification
        pre    = self.pre_logit(feat)   # (bs, 256)
        logits = self.head(pre)          # (bs, nc)
        return logits, pre

    # Multi-component appearance fusion

    def _forward_multi_component(self, full_frames: Optional[Tensor],
                                  patches: Optional[Tensor],
                                  coords: Tensor, bs: int, T: int, P: int) -> Optional[Tensor]:
        """
        Runs each active appearance component and concatenates their outputs
        in the fixed order (fragnet_landmark, broadcast, fragnet_grid,
        local_encoder, multiscale_roi) matching APPEARANCE_DIM in __init__.

        Encoder-reuse optimisation (task spec Step 7): fragnet_landmark,
        multiscale_roi, and fragnet_grid(resnet18) all sample the SAME
        ResNet-18 feature pyramid — computed once here and shared. broadcast
        reuses layer4's spatial mean when the pyramid is already being
        computed for another component, else it runs its own single forward.
        local_encoder always needs its own forward (different input shape:
        49×49 patches, not 224×224 frames).

        Returns None iff every appearance component is off (legal only when
        use_coord_encoder is on — enforced in __init__).
        """
        parts = []

        full_flat = full_frames.view(bs * T, 3, 224, 224) if full_frames is not None else None
        coords_flat = coords.view(bs * T, P, 2).float()
        coords_norm = ((coords_flat / 224.0) * 2.0 - 1.0).clamp(-1.0, 1.0)
        grid = coords_norm.unsqueeze(1)   # (bs*T, 1, P, 2)

        needs_pyramid = (self.use_fragnet_landmark or self.use_multiscale_roi or
                         (self.use_fragnet_grid and self.fragnet_grid_backbone == 'resnet18'))
        pyramid = None
        if needs_pyramid:
            pyramid = self.patch_encoder.forward_feature_pyramid(full_flat)

        # Component 1: landmark-guided FragNet (dual-pathway lateral fusion).
        if self.use_fragnet_landmark:
            A1 = sample_window(pyramid['layer2'], coords_norm, window_size=3).reshape(bs * T * P, 128, 3, 3)
            A2 = sample_window(pyramid['layer3'], coords_norm, window_size=3).reshape(bs * T * P, 256, 3, 3)
            A3 = sample_window(pyramid['layer4'], coords_norm, window_size=3).reshape(bs * T * P, 512, 3, 3)
            patches_in = patches.float() / 255.0 if patches.dtype == torch.uint8 else patches.float()
            v_frag = self.fragment_pathway(patches_in, A1, A2, A3)   # (bs*T*P, 512)
            parts.append(v_frag.view(bs, T, P, 512).permute(0, 2, 3, 1).contiguous())

        # Component 2: broadcast global features.
        if self.use_broadcast:
            if needs_pyramid:
                v_bc_flat = pyramid['layer4'].mean(dim=(2, 3))     # (bs*T, 512) — free, no extra forward
            else:
                v_bc_flat = self.patch_encoder(full_flat)          # (bs*T, 512) — sole encoder forward
            v_bc = v_bc_flat.view(bs, T, 512).unsqueeze(2).expand(-1, -1, P, -1)
            parts.append(v_bc.permute(0, 2, 3, 1).contiguous())

        # Component 3: grid-based FragNet (no landmark guidance).
        if self.use_fragnet_grid:
            if self.fragnet_grid_backbone == 'resnet18':
                grid_pyramid = [pyramid['layer2'], pyramid['layer3'], pyramid['layer4']]
                v_grid_frag = self.fragnet_grid(full_flat, precomputed_pyramid=grid_pyramid)
            else:
                v_grid_frag = self.fragnet_grid(full_flat)         # separate ResNet-50 forward
            pooled = v_grid_frag.mean(dim=1)                       # (bs*T, 512) — mean over fragments
            v_grid = pooled.view(bs, T, 512).unsqueeze(2).expand(-1, -1, P, -1)
            parts.append(v_grid.permute(0, 2, 3, 1).contiguous())

        # Component 5: local 49×49 landmark patches.
        if self.use_local_encoder:
            v_local_flat = self.patch_encoder(patches)             # (bs*T*P, 512)
            parts.append(v_local_flat.view(bs, T, P, 512).permute(0, 2, 3, 1).contiguous())

        # Component 6: multi-scale ROI-Align.
        if self.use_multiscale_roi:
            v_l2 = F.grid_sample(pyramid['layer2'], grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=False).squeeze(2)
            v_l3 = F.grid_sample(pyramid['layer3'], grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=False).squeeze(2)
            v_l4 = F.grid_sample(pyramid['layer4'], grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=False).squeeze(2)
            v_multi = torch.cat([v_l2, v_l3, v_l4], dim=1)         # (bs*T, 896, P)
            parts.append(v_multi.view(bs, T, 896, P).permute(0, 3, 2, 1).contiguous())

        if not parts:
            return None
        return torch.cat(parts, dim=2)   # (bs, P, APPEARANCE_DIM, T)