#! /usr/bin/python3
# Author : Ahmad Ramadan

"""
trainer_paper.py  —  Training loop for GCNTransformerPaper.

Key differences vs trainer.py:
  - Model forward returns (logits, features) for center loss.
  - Two optimizers: one for model, one for center loss centres.
  - Loss = CrossEntropy + lambda * CenterLoss.
  - Paper protocol: fixed epochs, CosineAnnealingLR, no early stopping.

Augmentation (paper Section 4.2):
  - Horizontal flip, random 224×224 crop offset: handled per-sample in
    LazyFERDataset.__getitem__ (applied identically to image + landmarks).
  - Gaussian noise on coordinates and patches: applied here in augment_batch
    after collation, as paper specifies noise is added last.
"""

import time
import copy
import statistics

import numpy as np
import torch


# Per-batch data augmentation (paper Section 4.2) — noise only
# Flip and random 224-crop are handled per-sample in LazyFERDataset.__getitem__.

def augment_batch(data, T: int, P: int, flip_map=None):
    """
    Add Gaussian noise to coordinates and raw patches after collation.
    Flip and 224-crop are already baked into the samples by LazyFERDataset.

    Noise spec (§4.2):
      - Coord noise: std ~ U[0, 5] absolute pixels, independent per sample.
      - Patch pixel noise: std ~ U[0, 5/255] in [0,1] space, per sample.

    data.x              : (bs*T*P, 2)
    data.patch_features : (bs*T*P, 3, 49, 49) uint8, or None
    flip_map is accepted for API compatibility but unused.
    """
    bs = data.num_graphs

    # Coordinate noise
    x = data.x.float().reshape(bs, T, P, 2).clone()
    for i in range(bs):
        coord_std = torch.empty(1).uniform_(0.0, 5.0).item()
        x[i] = x[i] + torch.randn_like(x[i]) * coord_std
    data.x = x.reshape(bs * T * P, 2).contiguous().to(data.x.dtype)

    # Patch pixel noise (raw on-the-fly patches only)
    if hasattr(data, 'patch_features') and data.patch_features is not None:
        pf = data.patch_features
        if pf.dim() == 4 and pf.shape[1:] == (3, 49, 49):
            pff = pf.float() / 255.0   # [0, 1]
            for i in range(bs):
                s, e = i * T * P, (i + 1) * T * P
                pixel_std_01 = torch.empty(1).uniform_(0.0, 5.0 / 255.0).item()
                pff[s:e] = pff[s:e] + torch.randn(e - s, 3, 49, 49) * pixel_std_01
            data.patch_features = pff.clamp(0.0, 1.0)

    return data


# Training loop

def train_model_gnn_paper(model, criterion, center_loss, lambda_center,
                           optimizer_model, optimizer_center,
                           apply_scheduler, scheduler,
                           dataloaders, device, num_epochs, print_every,
                           T=16, P=None, flip_map=None,
                           best_metric='val_acc',
                           val2_loader=None,
                           use_amp=False):
    """
    Parameters
    ----------
    model            : GCNTransformerPaper
    criterion        : nn.CrossEntropyLoss
    center_loss      : CenterLoss (with its own parameters / optimizer)
    lambda_center    : float  weight for center loss (paper: 0.001)
    optimizer_model  : Adam for model parameters
    optimizer_center : Adam for center_loss.centers  (lr=0.5 typical)
    apply_scheduler  : bool
    scheduler        : CosineAnnealingLR
    dataloaders      : dict with "train" key and optional "val" key.
                       If "val" present, the model state with the best
                       `best_metric` over training is restored before
                       returning.
    device           : torch.device
    num_epochs       : int (paper: 120)
    print_every      : int
    T                : int  number of frames per sequence (default 16)
    P                : int  number of landmarks/joints (required for augment_batch)
    flip_map         : list[int] | None  horizontal-flip landmark mapping
    best_metric      : 'val_acc' (default) or 'val_loss' — used to pick the
                       best checkpoint when dataloaders['val'] is present.
    use_amp          : bool — mixed-precision (autocast fp16 + GradScaler)
                       forward/backward. Needed when the in-model ResNet
                       encoder runs on upscaled 224×224 patches (bs*T*P
                       images through a full-precision ResNet-18 forward+
                       backward can exceed 16GB even at bs=1).
    """
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    if use_amp:
        print('Mixed precision (AMP): ON (autocast fp16 + GradScaler)')

    print('Model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % (
        type(model).__name__, type(optimizer_model).__name__,
        optimizer_model.param_groups[0]['lr'], num_epochs, device))

    if P is None:
        raise ValueError('P (number of joints) must be provided to train_model_gnn_paper '
                         'for per-batch spatial augmentation.')

    use_val  = 'val' in dataloaders and dataloaders['val'] is not None
    use_val2 = val2_loader is not None
    if use_val and best_metric not in ('val_acc', 'val_loss'):
        raise ValueError("best_metric must be 'val_acc' or 'val_loss'")
    if use_val:
        suffix = ' + secondary CAER val' if use_val2 else ''
        print(f'Validation : ON  (best checkpoint selected by {best_metric}{suffix})')
    else:
        print('Validation : OFF (final-epoch model used for test)')

    since = time.time()

    history = {
        'loss'                       : [],
        'acc'                        : [],
        'val_loss'                   : [],
        'val_acc'                    : [],
        'val2_acc'                   : [],   # CAER val (secondary), nan when unused
        'tps_par_moy_batch_par_epoch': [],
        'tps_par_epoch'              : [],
    }

    # Use the sampler's num_samples when present (lazy path with WeightedRandomSampler
    # draws more samples per epoch than the raw dataset size).
    _train_sampler = getattr(dataloaders['train'], 'sampler', None)
    if hasattr(_train_sampler, 'num_samples'):
        nb_train = _train_sampler.num_samples
    else:
        nb_train = len(dataloaders['train'].dataset)
    nb_val   = len(dataloaders['val'].dataset) if use_val else 0

    # Best-checkpoint tracking — primary (AFEW val)
    best_score      = -float('inf') if best_metric == 'val_acc' else float('inf')
    best_epoch      = 0
    best_state_dict = None

    # Best-checkpoint tracking — secondary (CAER val, always by acc)
    best_score2      = -float('inf')
    best_epoch2      = 0
    best_state_dict2 = None

    # Task 26: best-checkpoint tracking — joint (average of AFEW val + CAER val).
    # Only populated when both use_val and use_val2 are True.
    best_score_joint      = -float('inf')
    best_epoch_joint      = 0
    best_state_dict_joint = None

    for epoch in range(1, num_epochs + 1):

        loss_train_tmp         = []
        running_corrects_train = 0

        model.train(True)
        center_loss.train(True)
        debut_epoch = time.time()
        batch_times = []

        for data in dataloaders['train']:
            debut_batch = time.time()

            # Paper augmentation: applied directly to the collated Batch
            # data is already a Batch object from GeoDataLoader, so
            # data.num_graphs exists and augment_batch works correctly.
            # This avoids the per-sample unpack/repack loop entirely.
            data = augment_batch(data, T, P, flip_map=flip_map)

            data = data.to(device)
            bs   = data.y.shape[0]

            optimizer_model.zero_grad()
            optimizer_center.zero_grad()

            # Forward
            # data.patch_features may carry one of two shapes:
            #   - (bs*T*P, 512)         → pre-computed features (cached pipeline)
            #   - (bs*T*P, 3, 49, 49)   → raw patches (in-model encoder)
            # Dispatch to the right kwarg based on dim.
            patch_feat  = None
            patches     = None
            full_frames = getattr(data, 'full_frames', None)
            if hasattr(data, 'patch_features') and data.patch_features is not None:
                if data.patch_features.dim() == 2:
                    patch_feat = data.patch_features
                else:
                    patches    = data.patch_features

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, features = model(
                    data.x, data.edge_index, data.batch, bs,
                    patch_features=patch_feat,
                    patches=patches,
                    full_frames=full_frames,
                    edge_index_temporal=getattr(data, 'edge_index_temporal', None),
                )

                # Loss
                ce_loss = criterion(logits, data.y)
                cl_loss = center_loss(features, data.y)
                loss    = ce_loss + lambda_center * cl_loss

            scaler.scale(loss).backward()

            # Clip gradients for stability (model + center params) — must
            # unscale first so the max_norm threshold applies to real-scale grads.
            scaler.unscale_(optimizer_model)
            scaler.unscale_(optimizer_center)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(center_loss.parameters(), max_norm=1.0)

            scaler.step(optimizer_model)
            scaler.step(optimizer_center)
            scaler.update()

            preds = torch.argmax(logits, dim=1)
            loss_train_tmp.append(loss.item())
            running_corrects_train += torch.sum(preds == data.y.data).item()
            batch_times.append(time.time() - debut_batch)

        fin_epoch = time.time()
        tps_epoch = fin_epoch - debut_epoch
        moy_batch = statistics.mean(batch_times)

        history['loss'].append(np.average(loss_train_tmp))
        history['acc'].append((running_corrects_train / nb_train) * 100)
        history['tps_par_moy_batch_par_epoch'].append(moy_batch)
        history['tps_par_epoch'].append(tps_epoch)

        # Validation pass (no augmentation, no center-loss update)
        val_loss_avg = float('nan')
        val_acc_pct  = float('nan')
        if use_val:
            model.eval()
            center_loss.eval()
            val_loss_sum    = 0.0
            val_corrects    = 0
            val_n           = 0
            with torch.no_grad():
                for vdata in dataloaders['val']:
                    vdata   = vdata.to(device)
                    vbs     = vdata.y.shape[0]
                    vpf  = None
                    vpat = None
                    vff  = getattr(vdata, 'full_frames', None)
                    if (hasattr(vdata, 'patch_features')
                            and vdata.patch_features is not None):
                        if vdata.patch_features.dim() == 2:
                            vpf  = vdata.patch_features
                        else:
                            vpat = vdata.patch_features
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        vlogits, vfeats = model(
                            vdata.x, vdata.edge_index, vdata.batch, vbs,
                            patch_features=vpf,
                            patches=vpat,
                            full_frames=vff,
                            edge_index_temporal=getattr(vdata, 'edge_index_temporal', None),
                        )
                    v_ce = criterion(vlogits, vdata.y)
                    v_cl = center_loss(vfeats, vdata.y)
                    v_l  = v_ce + lambda_center * v_cl
                    val_loss_sum += v_l.item() * vbs
                    val_corrects += (vlogits.argmax(dim=1) == vdata.y).sum().item()
                    val_n        += vbs
            val_loss_avg = val_loss_sum / max(val_n, 1)
            val_acc_pct  = 100.0 * val_corrects / max(val_n, 1)

            # Track best checkpoint — primary
            score    = val_acc_pct if best_metric == 'val_acc' else -val_loss_avg
            if score > best_score:
                best_score      = score
                best_epoch      = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc_pct)

        # Secondary validation pass (CAER val)
        val2_acc_pct = float('nan')
        if use_val2:
            model.eval()
            center_loss.eval()
            val2_corrects = 0
            val2_n        = 0
            with torch.no_grad():
                for vdata in val2_loader:
                    vdata   = vdata.to(device)
                    vbs     = vdata.y.shape[0]
                    vpf, vpat = None, None
                    vff2 = getattr(vdata, 'full_frames', None)
                    if (hasattr(vdata, 'patch_features')
                            and vdata.patch_features is not None):
                        if vdata.patch_features.dim() == 2:
                            vpf  = vdata.patch_features
                        else:
                            vpat = vdata.patch_features
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        vlogits2, _ = model(
                            vdata.x, vdata.edge_index, vdata.batch, vbs,
                            patch_features=vpf, patches=vpat,
                            full_frames=vff2,
                            edge_index_temporal=getattr(vdata, 'edge_index_temporal', None),
                        )
                    val2_corrects += (vlogits2.argmax(dim=1) == vdata.y).sum().item()
                    val2_n        += vbs
            val2_acc_pct = 100.0 * val2_corrects / max(val2_n, 1)
            if val2_acc_pct > best_score2:
                best_score2      = val2_acc_pct
                best_epoch2      = epoch
                best_state_dict2 = copy.deepcopy(model.state_dict())
            # Task 26: track joint-best when both val domains are active.
            if use_val and not np.isnan(val_acc_pct):
                joint_score = (val_acc_pct + val2_acc_pct) / 2.0
                if joint_score > best_score_joint:
                    best_score_joint      = joint_score
                    best_epoch_joint      = epoch
                    best_state_dict_joint = copy.deepcopy(model.state_dict())
        history['val2_acc'].append(val2_acc_pct)

        if apply_scheduler:
            scheduler.step()

        if epoch % print_every == 0:
            msg = (
                'Epoch {}/{}, LR: {:.8f} | train loss: {:.4f} '
                '(ce={:.4f} cl={:.4f}) | train acc: {:.3f}%'
            ).format(
                epoch, num_epochs,
                optimizer_model.param_groups[0]['lr'],
                history['loss'][-1],
                ce_loss.item(), cl_loss.item(),
                history['acc'][-1]
            )
            if use_val:
                msg += ' | val loss: {:.4f} | val acc: {:.3f}%'.format(
                    val_loss_avg, val_acc_pct)
                if best_epoch == epoch:
                    msg += '  ← best'
            if use_val2:
                msg += ' | caer_val: {:.3f}%'.format(val2_acc_pct)
                if best_epoch2 == epoch:
                    msg += '  ← best'
            print(msg)

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Task 26: save last-epoch state dict BEFORE best-checkpoint restoration.
    history['last_epoch_state_dict'] = copy.deepcopy(model.state_dict())
    history['last_epoch']            = num_epochs

    # Task 26: store joint-best metadata in history.
    if best_state_dict_joint is not None:
        history['joint_best_state_dict'] = best_state_dict_joint
        history['best_epoch_joint']      = best_epoch_joint
        history['best_score_joint']      = best_score_joint
        print('Joint-best checkpoint at epoch {} (joint_val_avg = {:.4f}%)'.format(
            best_epoch_joint, best_score_joint))

    # Build secondary model (CAER-best) before restoring primary weights
    model2 = None
    if use_val2 and best_state_dict2 is not None:
        print('Restoring CAER-best checkpoint from epoch {} (caer_val_acc = {:.4f}%)'.format(
            best_epoch2, best_score2))
        model2 = copy.deepcopy(model)
        model2.load_state_dict(best_state_dict2)
        history['best_epoch2'] = best_epoch2
        history['best_score2'] = best_score2

    # Restore primary (AFEW) best-val checkpoint
    if use_val and best_state_dict is not None:
        print('Restoring AFEW-best checkpoint from epoch {} ({} = {:.4f})'.format(
            best_epoch, best_metric,
            best_score if best_metric == 'val_acc' else -best_score))
        model.load_state_dict(best_state_dict)
        history['best_epoch'] = best_epoch
        history['best_score'] = best_score if best_metric == 'val_acc' else -best_score

    return model, model2, history