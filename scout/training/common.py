"""Small helpers shared by the trainer and the evaluator."""

import torch


def batch_to_device(batch: dict, device) -> dict:
    """DataLoader(batch_size=1) adds a leading dim; drop it from the history
    tensors, keep it on the query tensors (the model expects [1, N, 3])."""
    out = {}
    for k, v in batch.items():
        v = v.float().to(device)
        if k.startswith("hist_"):
            v = v.squeeze(0)
        out[k] = v
    return out


def reconstruction_loss(model, pcd, flow, hist_pcds=None, hist_flows=None,
                        on_history=False):
    """Flow reconstruction from the outcome latent (door only).

    ``Dec(pcd + flow, E(pcd, flow)) ~ flow``.  Averaged 50/50 with the same
    term on the history outcomes when ``on_history`` is set.
    """
    z = model.encode_outcome(pcd, flow)
    loss = torch.nn.functional.mse_loss(model.decoder(pcd + flow, z), flow)
    if on_history and hist_pcds is not None and hist_pcds.shape[0] > 0:
        z_h = model.encode_outcome(hist_pcds, hist_flows)
        loss_h = torch.nn.functional.mse_loss(model.decoder(hist_pcds + hist_flows, z_h), hist_flows)
        loss = 0.5 * loss + 0.5 * loss_h
    return loss, z
