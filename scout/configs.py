"""Configuration dataclasses and the per-task registry.

Every task in the paper is fully described by a :class:`TaskConfig`. The
registry below (``TASKS``) holds the exact hyper-parameters used for the
camera-ready experiments; scripts load a task with :func:`get_task_config` and
may override individual fields from the command line.

Nothing in this module imports a simulator, so it is safe to use on a
training-only machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Action parameterisation.  3 = contact / target point (ISE tasks),
    # 6 = contact point + unit pull direction (ambiguous door).
    action_dim: int = 3
    # What the outcome encoder sees.  "pcd": post-action point cloud (ISE
    # tasks).  "flow": per-point flow + post-action point cloud (door).
    outcome_type: str = "pcd"

    latent_dim: int = 64
    d_model: int = 128
    nhead: int = 4
    gmm_layers: int = 4
    dyn_layers: int = 3
    outcome_layers: int = 2
    k_components: int = 4
    dropout: float = 0.1
    # Gradient checkpointing on transformer blocks (verified bit-exact with
    # double backward).  ~2-3x memory savings on the inner loop.
    use_checkpoint: bool = True

    # Inner-loop (test-time training) step size alpha.
    ttt_alpha: float = 0.1
    # If True the inner gradient is taken w.r.t. a *detached* copy of q, so q0
    # only receives the outer gradient through the linear ``q_k = q - a*grad``
    # term (door setting).  If False q0 gets the full second-order MAML
    # curvature (ISE setting).
    detach_q: bool = False
    # Stop gradients on the outcome-encoder target inside the inner loop
    # (BYOL-style).  Off for all reported experiments.
    detach_outcome_target: bool = False
    # Auxiliary flow decoder used by the reconstruction loss (door only).
    use_flow_decoder: bool = False

    def __post_init__(self):
        assert self.action_dim in (3, 6), "action_dim must be 3 or 6"
        assert self.outcome_type in ("pcd", "flow")
        assert self.d_model % self.nhead == 0


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    # Point clouds are sub/over-sampled to this many points.
    n_pts: int = 1200
    # Normalise every point cloud / action of a sample by the centre and
    # radius of its query point cloud (ISE tasks).  The door data is used in
    # raw world coordinates.
    normalize: bool = True
    # Fraction of a single collected pickle used for training; the rest is the
    # validation split (ISE tasks).  The door data is split by object id at
    # collection time instead.
    train_ratio: float = 0.9
    split_seed: int = 0
    # False: sequential split (first train_ratio of the pickle), as in the paper;
    # True: seeded random permutation (split_seed).
    shuffle_split: bool = False
    # With this probability the ground-truth (query) interaction is appended to
    # the history during training.
    augment_prob: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    epochs: int = 100
    outer_lr: float = 1e-3
    # Gradients are accumulated over this many single-sample forward passes
    # before an optimiser step (samples have variable history length so they
    # are processed one at a time).
    effective_batch_size: int = 64
    # Probability of dropping the history (q_k = q0) for a training sample.
    history_dropout: float = 0.3
    # Isotropic GMM standard deviation.
    sigma: float = 0.1
    optimizer: str = "adam"          # "adam" | "adamw"
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Outer-loop auxiliary terms.
    recon_weight: float = 0.0        # flow reconstruction (door only)
    recon_on_history: bool = False   # also reconstruct the H history outcomes
    outer_dyn_weight: float = 0.0    # latent dynamics MSE on the query step

    num_workers: int = 4
    seed: int = 0
    log_every: int = 50              # optimiser steps between wandb batch logs


# ─────────────────────────────────────────────────────────────────────────────
# Rollout / evaluation in the simulator
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RolloutConfig:
    n_episodes: int = 400
    max_attempts: int = 14
    # Number of inner-loop steps at test time (0 = no adaptation).
    ttt_steps: int = 1
    # "argmax": execute the highest-weight proposal; "sample": draw a proposal
    # from the mixture weights; "auto": argmax when ttt_steps > 0 else sample
    # (the setting used for the ISE numbers in the paper).
    proposal_selection: str = "auto"
    max_steps_per_attempt: int = 500
    seed: int = 0

    # Door-only settings.
    trials_per_object: int = 5
    closest_point: bool = True       # snap proposals onto the target link
    success_threshold: float = 0.9   # relative joint opening counted as success


# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TaskConfig:
    name: str
    family: str                       # "metaworld" | "door"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskConfig":
        return cls(
            name=d["name"],
            family=d["family"],
            model=ModelConfig(**d["model"]),
            data=DataConfig(**d["data"]),
            train=TrainConfig(**d["train"]),
            rollout=RolloutConfig(**d["rollout"]),
        )

    def override(self, **kwargs) -> "TaskConfig":
        """Return a copy with dotted overrides, e.g. ``override(**{"train.epochs": 10})``.

        Keys without a dot are looked up in every sub-config and must be
        unambiguous.
        """
        cfg = replace(self)
        subs = {"model": replace(cfg.model), "data": replace(cfg.data),
                "train": replace(cfg.train), "rollout": replace(cfg.rollout)}
        for key, value in kwargs.items():
            if value is None:
                continue
            if "." in key:
                section, name = key.split(".", 1)
            else:
                owners = [s for s, o in subs.items()
                          if name_in(o, key)]
                if len(owners) != 1:
                    raise KeyError(f"Override '{key}' is ambiguous or unknown: {owners}")
                section, name = owners[0], key
            if not name_in(subs[section], name):
                raise KeyError(f"Unknown config field {section}.{name}")
            setattr(subs[section], name, value)
        return replace(cfg, **subs)


def name_in(obj, name: str) -> bool:
    return any(f.name == name for f in fields(obj))


# ─────────────────────────────────────────────────────────────────────────────
# Registry — hyper-parameters of the camera-ready experiments
# ─────────────────────────────────────────────────────────────────────────────
def _ise(name: str, *, k: int, history_dropout: float, epochs: int,
         num_workers: int = 4, max_steps_per_attempt: int = 500,
         n_episodes: int = 400) -> TaskConfig:
    """Shared settings of the ISE / Meta-World tasks."""
    return TaskConfig(
        name=name,
        family="metaworld",
        model=ModelConfig(action_dim=3, outcome_type="pcd", k_components=k,
                          ttt_alpha=0.1, detach_q=False, use_flow_decoder=False),
        data=DataConfig(n_pts=1200, normalize=True, train_ratio=0.9, augment_prob=0.1),
        train=TrainConfig(epochs=epochs, outer_lr=1e-3, effective_batch_size=64,
                          history_dropout=history_dropout, sigma=0.1,
                          optimizer="adam", weight_decay=0.0, grad_clip=1.0,
                          num_workers=num_workers),
        rollout=RolloutConfig(n_episodes=n_episodes, max_attempts=14, ttt_steps=1,
                              max_steps_per_attempt=max_steps_per_attempt),
    )


TASKS: Dict[str, TaskConfig] = {
    # Continuous hidden parameter (bar centre of mass): K = 4, history dropout.
    "pick_bar": _ise("pick_bar", k=4, history_dropout=0.3, epochs=2000, n_episodes=800),
    "push_bar": _ise("push_bar", k=4, history_dropout=0.3, epochs=1000, n_episodes=800),
    # Binary tasks: K = 2 and no history dropout (dropping the history lets
    # the policy collapse to a q-insensitive solution on a two-mode task).
    "faucet": _ise("faucet", k=2, history_dropout=0.0, epochs=2000,
                   num_workers=0, max_steps_per_attempt=300),
    "box": _ise("box", k=2, history_dropout=0.0, epochs=2000,
                max_steps_per_attempt=300),
    # Hidden ramp friction (continuous): K = 4, history dropout.  Needs
    # MuJoCo <= 2.3.7 (scout-brick environment, see README).
    "brick": _ise("brick", k=4, history_dropout=0.3, epochs=900,
                  max_steps_per_attempt=600, n_episodes=800),
    # Ambiguous articulated doors: 6-D action, flow outcome, reconstruction loss.
    "door": TaskConfig(
        name="door",
        family="door",
        model=ModelConfig(action_dim=6, outcome_type="flow", k_components=4,
                          ttt_alpha=0.5, detach_q=True, use_flow_decoder=True),
        data=DataConfig(n_pts=1200, normalize=False, augment_prob=0.0),
        train=TrainConfig(epochs=100, outer_lr=1e-3, effective_batch_size=64,
                          history_dropout=0.3, sigma=0.1,
                          optimizer="adamw", weight_decay=1e-4, grad_clip=5.0,
                          recon_weight=1.0, recon_on_history=True,
                          outer_dyn_weight=0.0, num_workers=4),
        rollout=RolloutConfig(n_episodes=0, max_attempts=30, ttt_steps=5,
                              proposal_selection="argmax",
                              trials_per_object=5, closest_point=True,
                              success_threshold=0.9),
    ),
}


def get_task_config(name: str) -> TaskConfig:
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(TASKS)}")
    # Return a deep copy so callers can mutate freely.
    return TaskConfig.from_dict(TASKS[name].to_dict())
