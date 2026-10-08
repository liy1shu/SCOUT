"""The SCOUT model bundle: policy + dynamics + outcome encoder + belief latent q0.

This module owns everything that touches the belief latent:

* :meth:`SCOUT.inner_update` — one inner-loop step (differentiable, used in
  meta-training with ``create_graph=True``).
* :meth:`SCOUT.adapt` — k inner steps at test time (no meta-gradient).
* :meth:`SCOUT.predict` — the policy forward ``(w, mu) = GMM(q, o)``.
* checkpoint I/O (also loads the released weights, which carry no embedded config).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scout.configs import ModelConfig
from scout.models.dynamics import DynamicsModel
from scout.models.gmm_head import GMMHead
from scout.models.outcome_encoder import FlowDecoder, FlowOutcomeEncoder, PointCloudOutcomeEncoder


class SCOUT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        common = dict(d_model=cfg.d_model, nhead=cfg.nhead, dropout=cfg.dropout,
                      use_checkpoint=cfg.use_checkpoint)

        self.gmm = GMMHead(latent_dim=cfg.latent_dim, k=cfg.k_components,
                           action_dim=cfg.action_dim, num_layers=cfg.gmm_layers, **common)
        self.dyn = DynamicsModel(latent_dim=cfg.latent_dim, action_dim=cfg.action_dim,
                                 num_layers=cfg.dyn_layers, **common)
        enc_cls = FlowOutcomeEncoder if cfg.outcome_type == "flow" else PointCloudOutcomeEncoder
        self.outcome_enc = enc_cls(latent_dim=cfg.latent_dim, num_layers=cfg.outcome_layers, **common)
        self.decoder = FlowDecoder(latent_dim=cfg.latent_dim, hidden=cfg.d_model) \
            if cfg.use_flow_decoder else None
        self.q0 = nn.Parameter(torch.randn(cfg.latent_dim) * 0.1)

    # ── convenience ──────────────────────────────────────────────────────────
    @property
    def device(self):
        return self.q0.device

    def param_groups(self):
        return list(self.parameters())

    def summary(self) -> str:
        def n(m):
            return sum(p.numel() for p in m.parameters()) / 1e6
        lines = [f"  GMM        params: {n(self.gmm):.2f}M",
                 f"  DYN        params: {n(self.dyn):.2f}M",
                 f"  OutcomeEnc params: {n(self.outcome_enc):.2f}M"]
        if self.decoder is not None:
            lines.append(f"  Decoder    params: {n(self.decoder):.2f}M")
        return "\n".join(lines)

    # ── forward pieces ───────────────────────────────────────────────────────
    def predict(self, q: torch.Tensor, pcd: torch.Tensor):
        """q [B, D] or [D]; pcd [B, N, 3] -> (weights [B, M], means [B, M, A])."""
        if q.dim() == 1:
            q = q.unsqueeze(0)
        return self.gmm(q, pcd)

    def encode_outcome(self, pcds: torch.Tensor, outcomes: torch.Tensor):
        """outcomes = post-action clouds (pcd) or per-point flows (flow)."""
        return self.outcome_enc(pcds, outcomes)

    def inner_update(self, q_var: torch.Tensor, hist_pcds, hist_actions, hist_outcomes,
                     create_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One gradient step on q against the latent dynamics loss.

        L_inner(q) = mean_t || f(q, a_t, o_t) - E(o_{t+1}) ||^2

        Returns ``(q_k, inner_loss, grad_norm)``.  The gradient is normalised
        only when its norm exceeds 1, so weak signals keep their magnitude.
        """
        H = hist_pcds.shape[0]
        q_diff = q_var.detach().requires_grad_(True) if self.cfg.detach_q else q_var
        q_exp = q_diff.unsqueeze(0).expand(H, -1)

        z_pred = self.dyn(q_exp, hist_actions, hist_pcds)
        if self.cfg.detach_outcome_target:
            with torch.no_grad():
                z_target = self.encode_outcome(hist_pcds, hist_outcomes)
        else:
            z_target = self.encode_outcome(hist_pcds, hist_outcomes)
        inner_loss = F.mse_loss(z_pred, z_target)

        q_grad = torch.autograd.grad(inner_loss, q_diff, create_graph=create_graph)[0]
        q_grad = torch.nan_to_num(q_grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = q_grad.norm() + 1e-8
        if grad_norm > 1.0:
            q_grad = q_grad / grad_norm

        q_k = q_var - self.cfg.ttt_alpha * q_grad
        return q_k, inner_loss, grad_norm

    def adapt(self, hist_pcds, hist_actions, hist_outcomes, n_steps: int,
              q_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Test-time adaptation: ``n_steps`` inner updates from q0 (or q_init).

        Each step starts from a fresh detached leaf; there is no meta-gradient
        to carry across steps at inference.  Returns a detached q.
        """
        q = (self.q0 if q_init is None else q_init).detach().clone()
        if n_steps <= 0 or hist_pcds.shape[0] == 0:
            return q
        with torch.enable_grad():
            for _ in range(n_steps):
                q_in = q.detach().requires_grad_(True)
                q, _, _ = self.inner_update(q_in, hist_pcds, hist_actions, hist_outcomes,
                                            create_graph=False)
                q = torch.nan_to_num(q.detach(), nan=0.0)
        return q

    # ── checkpoints ──────────────────────────────────────────────────────────
    def checkpoint_dict(self, epoch: int, optimizer=None, **extra) -> dict:
        ckpt = {
            "epoch": epoch,
            "gmm": self.gmm.state_dict(),
            "dyn": self.dyn.state_dict(),
            "outcome_enc": self.outcome_enc.state_dict(),
            "q0": self.q0.detach().cpu(),
            "model_config": asdict(self.cfg),
        }
        if self.decoder is not None:
            ckpt["decoder"] = self.decoder.state_dict()
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        ckpt.update(extra)
        return ckpt

    def load_checkpoint_dict(self, ckpt: dict, strict: bool = True):
        self.gmm.load_state_dict(ckpt["gmm"], strict=strict)
        self.dyn.load_state_dict(ckpt["dyn"], strict=strict)
        self.outcome_enc.load_state_dict(ckpt["outcome_enc"], strict=strict)
        if self.decoder is not None and "decoder" in ckpt:
            self.decoder.load_state_dict(ckpt["decoder"], strict=strict)
        with torch.no_grad():
            self.q0.copy_(ckpt["q0"].to(self.q0.device))
        return self

    @classmethod
    def from_checkpoint(cls, path: str, cfg: Optional[ModelConfig] = None,
                        device: str = "cpu", prefer_embedded: bool = False) -> "SCOUT":
        """Load a checkpoint.

        Checkpoints written by ``scripts/train.py`` embed their ``ModelConfig``;
        the released weights do not, so ``cfg`` (e.g. ``get_task_config(task).model``)
        is required for them.  If both exist, ``cfg`` wins unless
        ``prefer_embedded`` is set.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if "model_config" in ckpt and (cfg is None or prefer_embedded):
            cfg = ModelConfig(**ckpt["model_config"])
        elif cfg is None:
            raise ValueError(f"'{path}' has no embedded model config: pass cfg=get_task_config(task).model.")
        model = cls(cfg).to(device)
        model.load_checkpoint_dict(ckpt)
        model.eval()
        print(f"Loaded checkpoint '{path}' (epoch {ckpt.get('epoch', '?')}) -> {device}")
        return model
