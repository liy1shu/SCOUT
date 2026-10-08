"""Offline evaluation: query NLL before and after one inner-loop step."""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from scout.models.losses import GMMLoss
from scout.training.common import batch_to_device, reconstruction_loss


def evaluate_metrics(model, dataset, device, sigma=0.1, num_workers=2,
                     recon=False) -> Dict[str, float]:
    """Returns pre/post-adaptation NLL and inner-loop diagnostics.

    ``post`` adapts q0 with a single inner step on the sample's history (the
    same thing the trainer does), ``pre`` uses q0 directly.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    was_training = model.training
    model.eval()
    loss_fn = GMMLoss(sigma=sigma)

    pre, post, pre_in, post_in, gnorms, recons = [], [], [], [], [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        b = batch_to_device(batch, device)
        q0 = model.q0

        with torch.no_grad():
            w, m = model.predict(q0, b["eval_pcd"])
            pre.append(loss_fn(w, m, b["eval_action"]).item())
            if recon:
                r, _ = reconstruction_loss(model, b["eval_pcd"], b["eval_outcome"])
                recons.append(r.item())

        if b["hist_pcds"].shape[0] > 0:
            with torch.enable_grad():
                q_post, in_loss, g = model.inner_update(
                    q0, b["hist_pcds"], b["hist_actions"], b["hist_outcomes"], create_graph=False)
            q_post = torch.nan_to_num(q_post.detach(), nan=0.0)
            with torch.no_grad():
                H = b["hist_pcds"].shape[0]
                z_pred = model.dyn(q_post.unsqueeze(0).expand(H, -1), b["hist_actions"], b["hist_pcds"])
                z_tgt = model.encode_outcome(b["hist_pcds"], b["hist_outcomes"])
                post_in.append(F.mse_loss(z_pred, z_tgt).item())
            pre_in.append(in_loss.item())
            gnorms.append(g.item())
        else:
            q_post = q0.detach()

        with torch.no_grad():
            w, m = model.predict(q_post, b["eval_pcd"])
            post.append(loss_fn(w, m, b["eval_action"]).item())

    if was_training:
        model.train()

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "pre_nll": mean(pre),
        "post_nll": mean(post),
        "pre_inner_loss": mean(pre_in),
        "post_inner_loss": mean(post_in),
        "grad_norm": mean(gnorms),
        "recon_loss": mean(recons),
    }
