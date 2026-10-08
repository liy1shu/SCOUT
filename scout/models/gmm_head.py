"""Policy network: FiLM-conditioned per-point transformer with a GMM head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scout.models.attention import FiLMTransformerBlock


class GMMHead(nn.Module):
    """(q, point cloud) -> mixture ``(weights [B, N*K], means [B, N*K, action_dim])``.

    Every one of the N points proposes K mixture components.  The point cloud
    goes through a FiLM transformer conditioned on q; the per-point features
    are concatenated with a broadcast q for the weight/mean heads, and a
    zero-initialised q-only residual path lets q nudge the output directly.

    Means are predicted as offsets from the proposing point.  With a 6-D
    action the last three dims are an L2-normalised direction.
    """

    def __init__(self, latent_dim=64, k=4, action_dim=3, d_model=128,
                 num_layers=4, nhead=4, dropout=0.1, use_checkpoint=True):
        super().__init__()
        assert action_dim in (3, 6)
        self.k = k
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.embed = nn.Linear(3, d_model)
        self.blocks = nn.ModuleList([
            FiLMTransformerBlock(d_model, nhead, cond_dim=latent_dim,
                                 dropout=dropout, use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        self.weight_head = nn.Sequential(
            nn.Linear(d_model + latent_dim, 64), nn.GELU(), nn.Linear(64, k))
        self.mean_head = nn.Sequential(
            nn.Linear(d_model + latent_dim, 128), nn.GELU(), nn.Linear(128, k * action_dim))

        self.q_residual_weight = nn.Linear(latent_dim, k)
        self.q_residual_mean = nn.Linear(latent_dim, k * action_dim)
        for lin in (self.q_residual_weight, self.q_residual_mean):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, q, pcd):
        B, N, _ = pcd.shape

        x = self.embed(pcd)
        for blk in self.blocks:
            x = blk(x, q)
        x = self.final_ln(x)

        feats = torch.cat([x, q.unsqueeze(1).expand(B, N, -1)], dim=-1)
        w_logits = self.weight_head(feats) + self.q_residual_weight(q).unsqueeze(1)
        m_raw = (self.mean_head(feats).view(B, N, self.k, self.action_dim)
                 + self.q_residual_mean(q).view(B, 1, self.k, self.action_dim))

        weights = F.softmax(w_logits.view(B, -1), dim=1)
        pos = pcd.unsqueeze(2) + m_raw[..., 0:3]
        if self.action_dim == 6:
            direction = F.normalize(m_raw[..., 3:6], p=2, dim=-1)
            means = torch.cat([pos, direction], dim=-1)
        else:
            means = pos
        return weights, means.view(B, -1, self.action_dim)
