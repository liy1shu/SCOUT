"""Meta-training loop (Algorithm 1 in the paper).

For every training sample:

  inner:  q_k = q0 - alpha * grad_q  mean_t || f(q, a_t, o_t) - E(o_{t+1}) ||^2
          (skipped with probability ``history_dropout`` or when H = 0)
  outer:  L = NLL_GMM(a*_query | GMM(q_k, o_query))
            + outer_dyn_weight * || f(q_k, a*, o_query) - E(o*_{t+1}) ||^2
            + recon_weight     * flow-reconstruction loss          (door)

Gradients are accumulated over ``effective_batch_size`` samples, clipped, and
applied with Adam/AdamW to all network weights and q0.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from scout.configs import TaskConfig
from scout.models.losses import GMMLoss
from scout.models.scout import SCOUT
from scout.training.common import batch_to_device, reconstruction_loss
from scout.training.evaluate import evaluate_metrics


def _make_optimizer(model: SCOUT, cfg):
    params = model.param_groups()
    if cfg.optimizer == "adam":
        return optim.Adam(params, lr=cfg.outer_lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return optim.AdamW(params, lr=cfg.outer_lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer {cfg.optimizer}")


class _NullLogger:
    def log(self, *a, **k):
        pass

    def finish(self):
        pass


def _init_wandb(task_cfg: TaskConfig, save_dir, project, run_name, entity, enabled):
    if not enabled:
        return _NullLogger()
    try:
        import wandb
    except Exception as e:   # not installed, or broken install
        print(f"[wandb] disabled ({type(e).__name__}: {e})")
        return _NullLogger()
    wandb.init(project=project, name=run_name or f"scout-{task_cfg.name}", entity=entity,
               config={**task_cfg.to_dict(), "save_dir": save_dir})
    return wandb


def run_meta_training(model: SCOUT, train_set, val_set, task_cfg: TaskConfig,
                      save_dir: str, wandb_project: str = "scout",
                      wandb_run: Optional[str] = None, wandb_entity: Optional[str] = None,
                      use_wandb: bool = True, resume: Optional[str] = None):
    cfg = task_cfg.train
    mcfg = task_cfg.model
    device = model.device
    os.makedirs(save_dir, exist_ok=True)

    use_recon = cfg.recon_weight != 0.0
    if use_recon:
        assert mcfg.outcome_type == "flow" and model.decoder is not None, \
            "The reconstruction loss needs a flow outcome and the flow decoder."

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    optimizer = _make_optimizer(model, cfg)
    loss_fn = GMMLoss(sigma=cfg.sigma)
    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_checkpoint_dict(ckpt)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed from {resume} at epoch {start_epoch}")

    logger = _init_wandb(task_cfg, save_dir, wandb_project, wandb_run, wandb_entity, use_wandb)

    print(f"Meta-training '{task_cfg.name}' on {device} | train={len(train_set)} val={len(val_set)}")
    print(model.summary())
    print(f"  effective batch {cfg.effective_batch_size} | history dropout {cfg.history_dropout} "
          f"| recon {cfg.recon_weight} | outer-dyn {cfg.outer_dyn_weight}")

    best_post, best_gap = float("inf"), float("inf")
    params = model.param_groups()
    step = 0

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=cfg.num_workers)

        sums = dict(nll=0.0, dyn=0.0, recon=0.0, inner=0.0, gnorm=0.0)
        n_hist, n_samples, in_batch = 0, 0, 0
        batch_sums = dict(nll=0.0, dyn=0.0, recon=0.0, inner=0.0, gnorm=0.0)
        batch_hist = 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for batch in pbar:
            b = batch_to_device(batch, device)
            H = b["hist_pcds"].shape[0]
            use_history = H > 0 and (np.random.rand() >= cfg.history_dropout)

            # ── inner loop ───────────────────────────────────────────────────
            if use_history:
                q_k, inner_loss, gnorm = model.inner_update(
                    model.q0, b["hist_pcds"], b["hist_actions"], b["hist_outcomes"],
                    create_graph=True)
                q_k = torch.nan_to_num(q_k, nan=0.0)
                batch_sums["inner"] += inner_loss.item()
                batch_sums["gnorm"] += gnorm.item()
                batch_hist += 1
            else:
                q_k = model.q0

            # ── outer loop ───────────────────────────────────────────────────
            weights, means = model.predict(q_k, b["eval_pcd"])
            nll = loss_fn(weights, means, b["eval_action"])
            total = nll

            if cfg.outer_dyn_weight != 0.0:
                z_pred = model.dyn(q_k.unsqueeze(0), b["eval_action"], b["eval_pcd"])
                if mcfg.detach_outcome_target:
                    with torch.no_grad():
                        z_tgt = model.encode_outcome(b["eval_pcd"], b["eval_outcome"])
                else:
                    z_tgt = model.encode_outcome(b["eval_pcd"], b["eval_outcome"])
                dyn_loss = nn.functional.mse_loss(z_pred, z_tgt)
                total = total + cfg.outer_dyn_weight * dyn_loss
                batch_sums["dyn"] += dyn_loss.item()

            if use_recon:
                recon, _ = reconstruction_loss(
                    model, b["eval_pcd"], b["eval_outcome"],
                    b["hist_pcds"], b["hist_outcomes"], on_history=cfg.recon_on_history)
                total = total + cfg.recon_weight * recon
                batch_sums["recon"] += recon.item()

            (total / cfg.effective_batch_size).backward()
            batch_sums["nll"] += nll.item()
            in_batch += 1
            n_samples += 1

            # ── optimiser step every effective_batch_size samples ────────────
            if in_batch == cfg.effective_batch_size:
                nn.utils.clip_grad_norm_(params, max_norm=cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                for k in sums:
                    sums[k] += batch_sums[k]
                n_hist += batch_hist

                if step % cfg.log_every == 0:
                    log = {
                        "batch/nll": batch_sums["nll"] / in_batch,
                        "batch/outer_dyn": batch_sums["dyn"] / in_batch,
                        "batch/recon": batch_sums["recon"] / in_batch,
                    }
                    if batch_hist:
                        log["batch/inner_loss"] = batch_sums["inner"] / batch_hist
                        log["batch/grad_norm"] = batch_sums["gnorm"] / batch_hist
                    logger.log(log)
                pbar.set_postfix(nll=f"{batch_sums['nll'] / in_batch:.3f}")

                batch_sums = dict.fromkeys(batch_sums, 0.0)
                batch_hist, in_batch = 0, 0
                step += 1

        if in_batch > 0:   # tail flush
            nn.utils.clip_grad_norm_(params, max_norm=cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            for k in sums:
                sums[k] += batch_sums[k]
            n_hist += batch_hist

        train = {
            "train/nll": sums["nll"] / max(n_samples, 1),
            "train/outer_dyn": sums["dyn"] / max(n_samples, 1),
            "train/recon": sums["recon"] / max(n_samples, 1),
            "train/inner_loss": sums["inner"] / max(n_hist, 1),
            "train/grad_norm": sums["gnorm"] / max(n_hist, 1),
            "train/epoch_time_s": time.time() - t0,
        }

        # ── validation ───────────────────────────────────────────────────────
        val = evaluate_metrics(model, val_set, device, sigma=cfg.sigma,
                               num_workers=min(cfg.num_workers, 2), recon=use_recon)
        gap = val["post_nll"] - val["pre_nll"]
        print(f"Epoch {epoch + 1} | train NLL {train['train/nll']:.4f} "
              f"| inner {train['train/inner_loss']:.4f} | recon {train['train/recon']:.4f}")
        print(f"          | val NLL pre {val['pre_nll']:.4f} post {val['post_nll']:.4f} "
              f"(delta {gap:+.4f}) | val inner pre {val['pre_inner_loss']:.4f} "
              f"post {val['post_inner_loss']:.4f}")

        logger.log({"epoch": epoch + 1, **train, **{f"val/{k}": v for k, v in val.items()},
                    "val/post_minus_pre": gap})

        # ── checkpoints ──────────────────────────────────────────────────────
        ckpt = model.checkpoint_dict(epoch + 1, optimizer, task_config=task_cfg.to_dict(),
                                     val_metrics=val)
        torch.save(ckpt, os.path.join(save_dir, "latest.pt"))
        if val["post_nll"] < best_post:
            best_post = val["post_nll"]
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))
            print(f"  [*] best.pt saved (post NLL {best_post:.4f})")
        if gap < best_gap:
            best_gap = gap
            torch.save(ckpt, os.path.join(save_dir, "best_gap.pt"))

    print("Training complete.")
    logger.finish()
