"""Transformer blocks with *manual* attention.

Attention is implemented with explicit matmul + softmax rather than
``nn.MultiheadAttention`` / ``F.scaled_dot_product_attention``: the fused CUDA
kernels those dispatch to do not implement double backward, and the TTT inner
loop differentiates through ``torch.autograd.grad(..., create_graph=True)``.

All blocks optionally wrap their forward in ``torch.utils.checkpoint`` while
training (``use_reentrant=False`` is required for double backward).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _ffn(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model * 4, d_model),
        nn.Dropout(dropout),
    )


def _attend(q, k, v, scale, attn_dropout):
    """q, k, v: [B, heads, N, d_head] -> [B, N, heads*d_head]."""
    B, Hh, N, Dh = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    attn = attn_dropout(attn)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2).contiguous().view(B, N, Hh * Dh)


class _ManualAttnMixin:
    def _split_heads(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.nhead, self.d_head).transpose(1, 2)


class SelfAttnBlock(nn.Module, _ManualAttnMixin):
    """Pre-norm self-attention + FFN, no conditioning."""

    def __init__(self, d_model, nhead, dropout=0.1, use_checkpoint=False):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.use_checkpoint = use_checkpoint

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.attn_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, dropout)

    def _attn(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.nhead, self.d_head).permute(2, 0, 3, 1, 4)
        out = _attend(qkv[0], qkv[1], qkv[2], self.scale, self.attn_dropout)
        return self.out_dropout(self.attn_out(out))

    def _forward_impl(self, x):
        x = x + self._attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class FiLMTransformerBlock(nn.Module, _ManualAttnMixin):
    """Pre-norm self-attention + FFN with FiLM modulation of the FFN output.

    The conditioning vector (the belief latent q) produces (gamma, beta) and
    the FFN branch becomes ``(1 + gamma) * h + beta``.  FiLM is initialised
    near identity (small gamma, zero beta) but *not* at zero, so q-conditioning
    is live from the first epoch.
    """

    def __init__(self, d_model, nhead, cond_dim, dropout=0.1, use_checkpoint=False):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.use_checkpoint = use_checkpoint

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.attn_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, dropout)

        self.film = nn.Linear(cond_dim, 2 * d_model)
        nn.init.xavier_normal_(self.film.weight, gain=0.5)
        nn.init.zeros_(self.film.bias)

    def _attn(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.nhead, self.d_head).permute(2, 0, 3, 1, 4)
        out = _attend(qkv[0], qkv[1], qkv[2], self.scale, self.attn_dropout)
        return self.out_dropout(self.attn_out(out))

    def _forward_impl(self, x, cond):
        x = x + self._attn(self.ln1(x))
        h = self.ffn(self.ln2(x))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)
        return x + h

    def forward(self, x, cond):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, cond, use_reentrant=False)
        return self._forward_impl(x, cond)


class FlowContentAttnBlock(nn.Module, _ManualAttnMixin):
    """Attention where Q/K see (flow + position) but V sees flow only.

    Used as the first block of the flow outcome encoder: geometry decides
    *which* tokens attend to each other, while only the motion is carried
    through the network.  Static points (zero flow) all map to the same
    value vector, so the pooled output cannot depend on geometry alone.
    """

    def __init__(self, d_model, nhead, dropout=0.1, use_checkpoint=False):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.use_checkpoint = use_checkpoint

        self.ln1_qk = nn.LayerNorm(d_model)
        self.ln1_v = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, dropout)

    def _attn(self, x_qk, x_v):
        q = self._split_heads(self.q_proj(x_qk))
        k = self._split_heads(self.k_proj(x_qk))
        v = self._split_heads(self.v_proj(x_v))
        out = _attend(q, k, v, self.scale, self.attn_dropout)
        return self.out_dropout(self.attn_out(out))

    def _forward_impl(self, x_qk, x_v):
        x = x_v + self._attn(self.ln1_qk(x_qk), self.ln1_v(x_v))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward(self, x_qk, x_v):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x_qk, x_v, use_reentrant=False)
        return self._forward_impl(x_qk, x_v)
