"""CPU smoke tests: shapes, double backward through the inner loop, a trainer
step on synthetic data, and checkpoint round-trips (with and without an
embedded model config, the latter being the format of the released weights).  Run with ``python -m pytest tests`` or ``python tests/test_models.py``.
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scout.configs import TaskConfig, get_task_config  # noqa: E402
from scout.models.scout import SCOUT  # noqa: E402
from scout.training.evaluate import evaluate_metrics  # noqa: E402
from scout.training.trainer import run_meta_training  # noqa: E402

N, H = 64, 3


def _small(cfg: TaskConfig) -> TaskConfig:
    return cfg.override(**{"model.d_model": 32, "model.nhead": 2, "model.gmm_layers": 1,
                           "model.dyn_layers": 1, "model.outcome_layers": 2, "model.latent_dim": 16,
                           "model.use_checkpoint": True})


class _SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, action_dim, n=6, with_history=True):
        self.action_dim, self.n, self.with_history = action_dim, n, with_history

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        h = H if (self.with_history and i % 2 == 0) else 0
        return {
            "eval_pcd": torch.randn(N, 3), "eval_action": torch.randn(self.action_dim),
            "eval_outcome": torch.randn(N, 3) * 0.1,
            "hist_pcds": torch.randn(h, N, 3), "hist_actions": torch.randn(h, self.action_dim),
            "hist_outcomes": torch.randn(h, N, 3) * 0.1,
        }


def _check_task(name):
    cfg = _small(get_task_config(name))
    model = SCOUT(cfg.model)
    A = cfg.model.action_dim
    pcd = torch.randn(1, N, 3)

    w, m = model.predict(model.q0, pcd)
    assert w.shape == (1, N * cfg.model.k_components)
    assert m.shape == (1, N * cfg.model.k_components, A)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)
    if A == 6:
        assert torch.allclose(m[..., 3:].norm(dim=-1), torch.ones(1, N * cfg.model.k_components), atol=1e-5)

    hp, ha, ho = torch.randn(H, N, 3), torch.randn(H, A), torch.randn(H, N, 3)
    # Inner update with create_graph=True must support double backward (train mode => checkpointing on).
    model.train()
    q_k, inner, g = model.inner_update(model.q0, hp, ha, ho, create_graph=True)
    assert q_k.shape == model.q0.shape and torch.isfinite(inner)
    w, m = model.predict(q_k, pcd)
    loss = -torch.log(w).mean() + m.pow(2).mean()
    loss.backward()
    assert model.q0.grad is not None and torch.isfinite(model.q0.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    # Test-time adaptation moves q and is deterministic in eval mode.
    model.eval()
    q1 = model.adapt(hp, ha, ho, n_steps=2)
    q2 = model.adapt(hp, ha, ho, n_steps=2)
    assert torch.allclose(q1, q2) and not torch.allclose(q1, model.q0)
    assert torch.allclose(model.adapt(hp, ha, ho, n_steps=0), model.q0)

    # Checkpoint round trip, with and without an embedded config.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save(model.checkpoint_dict(epoch=3), path)
        m2 = SCOUT.from_checkpoint(path)
        assert m2.cfg == model.cfg
        for p1, p2 in zip(model.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)
        no_cfg = model.checkpoint_dict(epoch=3)
        no_cfg.pop("model_config")
        torch.save(no_cfg, path)
        m3 = SCOUT.from_checkpoint(path, cfg=cfg.model)
        assert torch.equal(m3.q0, model.q0)
    print(f"  {name}: ok")
    return cfg


def test_all_tasks_forward_and_double_backward():
    for name in ("pick_bar", "faucet", "door"):
        _check_task(name)


def test_trainer_runs_on_synthetic_data():
    for name in ("pick_bar", "door"):
        cfg = _small(get_task_config(name)).override(**{
            "train.epochs": 1, "train.effective_batch_size": 4, "train.num_workers": 0})
        model = SCOUT(cfg.model)
        A = cfg.model.action_dim
        with tempfile.TemporaryDirectory() as d:
            run_meta_training(model, _SyntheticDataset(A), _SyntheticDataset(A, n=3), cfg,
                              save_dir=d, use_wandb=False)
            assert os.path.exists(os.path.join(d, "best.pt"))
            metrics = evaluate_metrics(model, _SyntheticDataset(A, n=3), "cpu",
                                       recon=cfg.train.recon_weight > 0, num_workers=0)
            assert all(np.isfinite(v) for v in metrics.values())
        print(f"  trainer[{name}]: ok")


def test_metaworld_dataset_normalisation():
    from scout.data.metaworld_dataset import AttemptHistoryDataset
    rng = np.random.RandomState(0)
    s = {"observation": [rng.randn(300, 3) + 5], "history_actions": [rng.randn(3) + 5],
         "history_outcomes": [rng.randn(300, 3) + 5], "ground_truth_action": rng.randn(3) + 5,
         "ground_truth_outcome": rng.randn(300, 3) + 5, "num_history": 1, "env_name": "x"}
    ds = AttemptHistoryDataset([s, {**s, "observation": [], "history_actions": [], "history_outcomes": []}],
                               n_pts=100, augment_prob=1.0)
    assert len(ds) == 1, "episode without history must be skipped"
    item = ds[0]
    assert item["eval_pcd"].shape == (100, 3) and item["hist_pcds"].shape == (2, 100, 3)
    assert abs(item["eval_pcd"].mean().item()) < 1e-5 and item["eval_pcd"].norm(dim=-1).max() <= 1.0 + 1e-5
    print("  dataset: ok")


if __name__ == "__main__":
    test_all_tasks_forward_and_double_backward()
    test_trainer_runs_on_synthetic_data()
    test_metaworld_dataset_normalisation()
    print("all tests passed")
