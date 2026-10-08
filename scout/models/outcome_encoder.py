"""Outcome encoders E(o_{t+1}) -> z_target and the auxiliary flow decoder.

Both encoders share the call signature ``encoder(pcd_before, outcome)`` so the
rest of the code base is agnostic to the outcome representation:

* :class:`PointCloudOutcomeEncoder` (ISE tasks) — ``outcome`` is the post-action
  point cloud; ``pcd_before`` is ignored.
* :class:`FlowOutcomeEncoder` (ambiguous door) — ``outcome`` is the per-point
  flow; the post-action cloud is ``pcd_before + flow``.

Both end in ``LayerNorm(latent_dim)`` so the latent has a near-constant norm
and cannot collapse to a vanishing vector (collapse can still show up as a
constant *direction*; see the reconstruction loss).
"""

import torch
import torch.nn as nn

from scout.models.attention import FlowContentAttnBlock, SelfAttnBlock


class PointCloudOutcomeEncoder(nn.Module):
    """Transformer over the post-action point cloud -> latent [B, latent_dim]."""

    def __init__(self, d_model=128, nhead=4, num_layers=2, latent_dim=64,
                 dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.embed = nn.Linear(3, d_model)
        self.blocks = nn.ModuleList([
            SelfAttnBlock(d_model, nhead, dropout=dropout, use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(nn.Linear(d_model, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, pcd_before, pcd_after):
        x = self.embed(pcd_after)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_ln(x)
        return self.proj(x.mean(dim=1))


class FlowOutcomeEncoder(nn.Module):
    """(pcd_before, flow) -> latent [B, latent_dim].

    Position (post-action cloud) and flow are embedded separately.  The first
    block routes attention on position + flow but carries flow only; the
    remaining blocks are plain self-attention.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=2, latent_dim=64,
                 dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.flow_embed = nn.Linear(3, d_model)
        self.pcd_embed = nn.Linear(3, d_model)
        self.first_block = FlowContentAttnBlock(d_model, nhead, dropout=dropout,
                                               use_checkpoint=use_checkpoint)
        self.rest_blocks = nn.ModuleList([
            SelfAttnBlock(d_model, nhead, dropout=dropout, use_checkpoint=use_checkpoint)
            for _ in range(num_layers - 1)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(nn.Linear(d_model, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, pcd_before, flow):
        flow_feat = self.flow_embed(flow)
        pcd_feat = self.pcd_embed(pcd_before + flow)
        x = self.first_block(flow_feat + pcd_feat, flow_feat)
        for blk in self.rest_blocks:
            x = blk(x)
        x = self.final_ln(x)
        return self.proj(x.mean(dim=1))


class FlowDecoder(nn.Module):
    """Point-wise MLP: (point [3], z [latent_dim]) -> flow at that point [3].

    Only used by the reconstruction loss in the outer loop; it never enters
    the inner-loop adaptation.
    """

    def __init__(self, latent_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, pcd, z):
        B, N, _ = pcd.shape
        z_bcast = z.unsqueeze(1).expand(B, N, -1)
        return self.net(torch.cat([pcd, z_bcast], dim=-1))
