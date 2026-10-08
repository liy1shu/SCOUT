"""Isotropic Gaussian-mixture negative log-likelihood (the outer-loop loss)."""

import torch
import torch.nn as nn


class GMMLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = float(sigma)

    def set_sigma(self, sigma):
        self.sigma = float(sigma)

    def forward(self, weights, means, gt_action, reduction="mean"):
        """weights [B, M], means [B, M, D], gt_action [B, D]."""
        sq_dist = ((gt_action.unsqueeze(1) - means) ** 2).sum(dim=-1)
        exponent = torch.log(weights + 1e-8) - sq_dist / (2.0 * self.sigma ** 2)
        per_sample = -torch.logsumexp(exponent, dim=1)
        if reduction == "mean":
            return per_sample.mean()
        if reduction == "none":
            return per_sample
        raise ValueError(f"Unknown reduction: {reduction}")
