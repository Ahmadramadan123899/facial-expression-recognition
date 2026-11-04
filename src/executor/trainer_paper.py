#! /usr/bin/python3
# Author : Ahmad Ramadan

"""
trainer_paper.py - Training loop for GCNTransformerPaper.
keyy points:
  - Model forward returns (logits, features) for center loss.
  - Two optimizers: one for model, one for center loss centres.
  - Loss = CrossEntropy + lambda * CenterLoss.
  - Paper protocol: fixed epochs, CosineAnnealingLR, no early stopping.
"""

import time
import statistics

import numpy as np
import torch

def train_model_gnn_paper(model, criterion, center_loss, lambda_center,
                           optimizer_model, optimizer_center,
                           apply_scheduler, scheduler,
                           dataloaders, device, num_epochs, print_every):
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
    dataloaders      : dict with "train" key only  (no val set)
    device           : torch.device
    num_epochs       : int (paper: 120)
    print_every      : int
    """

    print('Model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % (
        type(model).__name__, type(optimizer_model).__name__,
        optimizer_model.param_groups[0]['lr'], num_epochs, device))

    since = time.time()

    history = {
        'loss'                       : [],
        'acc'                        : [],
        'tps_par_moy_batch_par_epoch': [],
        'tps_par_epoch'              : [],
    }

    nb_train = len(dataloaders['train'].dataset)

    for epoch in range(1, num_epochs + 1):

        loss_train_tmp         = []
        running_corrects_train = 0

        model.train(True)
        center_loss.train(True)
        debut_epoch = time.time()
        batch_times = []

        for data in dataloaders['train']:
            debut_batch = time.time()
            data        = data.to(device)
            bs          = data.y.shape[0]

            optimizer_model.zero_grad()
            optimizer_center.zero_grad()

            # ── Forward ──────────────────────────────────────────────
            patch_feat = None
            if hasattr(data, 'patch_features') and data.patch_features is not None:
                patch_feat = data.patch_features

            logits, features = model(
                data.x, data.edge_index, data.batch, bs,
                patch_features=patch_feat
            )

            # ── Loss ─────────────────────────────────────────────────
            ce_loss = criterion(logits, data.y)
            cl_loss = center_loss(features, data.y)
            loss    = ce_loss + lambda_center * cl_loss

            loss.backward()

            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer_model.step()
            optimizer_center.step()

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

        if apply_scheduler:
            scheduler.step()

        if epoch % print_every == 0:
            print(
                'Epoch {}/{}, LR: {:.8f} | train loss: {:.4f} '
                '(ce={:.4f} cl={:.4f}) | train acc: {:.3f}%'.format(
                    epoch, num_epochs,
                    optimizer_model.param_groups[0]['lr'],
                    history['loss'][-1],
                    ce_loss.item(), cl_loss.item(),
                    history['acc'][-1]
                )
            )

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model, history
