"""Forward dynamics model f(q, a_t, o_t) -> z_pred."""

import torch
import torch.nn as nn

from scout.models.attention import FiLMTransformerBlock


class DynamicsModel(nn.Module):
    """Per-point FiLM transformer over the pre-action point cloud.

    The action enters through the per-point input features
    ``[p, a_pos - p (, a_dir)]`` and q through FiLM.  Per-point features are
    mean-pooled and projected to the outcome latent space, where the target is
    ``OutcomeEncoder(o_{t+1})``.
    """

    def __init__(self, latent_dim=64, action_dim=3, d_model=128, num_layers=3,
                 nhead=4, dropout=0.1, use_checkpoint=True):
        super().__init__()
        assert action_dim in (3, 6)
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.embed = nn.Linear(3 + action_dim, d_model)   # 6 or 9 inputs per point
        self.blocks = nn.ModuleList([
            FiLMTransformerBlock(d_model, nhead, cond_dim=latent_dim,
                                 dropout=dropout, use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.latent_head = nn.Sequential(nn.Linear(d_model, latent_dim), nn.LayerNorm(latent_dim))

    def _backbone(self, q, action, pcd):
        B, N, _ = pcd.shape
        rel_pos = action[:, :3].unsqueeze(1) - pcd
        feats = [pcd, rel_pos]
        if self.action_dim == 6:
            feats.append(action[:, 3:].unsqueeze(1).expand(B, N, 3))
        x = self.embed(torch.cat(feats, dim=-1))
        for blk in self.blocks:
            x = blk(x, q)
        return self.final_ln(x)

    def forward(self, q, action, pcd):
        return self.latent_head(self._backbone(q, action, pcd).mean(dim=1))
