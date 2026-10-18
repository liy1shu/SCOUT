# SCOUT: Scouting the Dynamics Gap

Code for **Scouting the Dynamics Gap: Test-Time Policy Adaptation via
Action-Outcome Feedback** (CoRL 2026). Project page:
<https://liy1shu.github.io/SCOUT/>.

SCOUT couples a point-cloud policy (a FiLM-conditioned GMM head) with a
forward dynamics model through a shared belief latent `q`. At test time the
only thing that changes is `q`: it is updated by gradient descent on the
dynamics prediction error of the attempts made so far, and the policy adapts
through its conditioning on `q`. Meta-training learns the network weights and
the initial belief `q0` so that this inner update translates directly into
better actions (Algorithm 1 in the paper).

```
 history (o_t, a_t, o_t+1)  ──►  inner loop:  q_k = q_0 - α ∇_q Σ_t ‖ f(q, a_t, o_t) − E(o_t+1) ‖²
                                                           │
 query observation o  ───────────────────────────────────► GMM(q_k, o) ──► (w, μ) ──► action
```

## Contents

- [Installation](#installation)
- [Pretrained checkpoints and door assets](#pretrained-checkpoints-and-door-assets)
- [Evaluating the pretrained models](#evaluating-the-pretrained-models)
- [Reproducing the paper from scratch](#reproducing-the-paper-from-scratch)
- [Using the model in your own code](#using-the-model-in-your-own-code)
- [Repository layout](#repository-layout)
- [Notes on this release](#notes-on-this-release)

## Installation

Tested on Ubuntu 24.04 with Python 3.10, PyTorch 2.5.1 (CUDA 12.1), MuJoCo 3.3.0
(3.1–3.15 also work), gymnasium 1.4, PyBullet 3.2.7 and an NVIDIA GPU. Training needs a GPU;
the simulators run on the CPU and render headlessly.

### 1. System packages

The Meta-World tasks render point clouds headlessly through EGL:

```bash
sudo apt-get install -y libegl1 libgl1 libegl-mesa0     # EGL loader, GL, Mesa fallback (see Troubleshooting)
```

### 2. Python environment

```bash
git clone https://github.com/liy1shu/SCOUT.git && cd SCOUT
conda create -n scout python=3.10 -y
conda activate scout

# PyTorch for your CUDA driver. Install it first: the default PyPI wheel may target a newer
# CUDA than your driver supports, and torch then silently runs on the CPU.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# SCOUT and its task dependencies
#   logging   = wandb (optional; pass --no-wandb to train.py otherwise)
#   metaworld = gymnasium, mujoco, PyOpenGL, open3d, opencv, matplotlib, imageio
#   door      = pybullet, scipy (+ imageio/opencv for videos)
pip install -e ".[logging,metaworld,door]"

# Meta-World fork with the SCOUT environments and scripted experts, vendored in this repo.
# Install it *instead of* upstream metaworld (uninstall that first if present).
pip install -e third_party/metaworld
```

`pip install -e .` alone (numpy, torch, tqdm) is enough to train on collected data.
Only data collection and rollouts need the simulator extras.

### 2b. Slide Brick: a second environment with MuJoCo 2.3.7

Slide Brick only reproduces on MuJoCo ≤ 2.3.7, while the other four Meta-World tasks were
trained on MuJoCo 3 (see [Notes on this release](#notes-on-this-release)). Its collection and
rollouts therefore run in a copy of the environment above with MuJoCo pinned to 2.3.7; the
code is the same. Training needs no simulator and works in either environment.

```bash
conda create --clone scout -n scout-brick -y
conda activate scout-brick
pip install mujoco==2.3.7   # pip warns that metaworld requires mujoco>=3.0.0: expected, ignore it
python scripts/diag_render.py --env brick-slide-v2   # must print PASS
```

Under MuJoCo ≥ 3 the brick task refuses to start rather than silently produce wrong numbers.

### 3. Check the installation

```bash
python scripts/diag_render.py    # headless MuJoCo rendering for the Meta-World tasks: must print PASS
python tests/test_models.py      # model, inner loop, trainer and checkpoint round-trip on CPU (seconds)
python tests/test_door_sim.py    # PyBullet door simulation on a synthetic two-mode door (seconds)
```

Optionally, `TASKS="pick_bar box" bash scripts/smoke_test.sh` runs a tiny version of
data collection → training → rollout end to end (~3 min for all four Meta-World tasks; outputs in `./smoke`).
The door part of the smoke test needs the door assets (see below). The brick part is skipped
under MuJoCo 3; run `TASKS=brick bash scripts/smoke_test.sh` in `scout-brick`.

### Troubleshooting

* **`diag_render.py` prints FAIL / `RuntimeError: MuJoCo segmentation rendering returned
  invalid ids`.** The point clouds come from MuJoCo segmentation renders through headless EGL
  (`MUJOCO_GL=egl`, set automatically). On some headless NVIDIA datacenter GPUs (seen on H200
  with driver 570) the NVIDIA EGL driver returns corrupted images. Use Mesa's software EGL
  instead. It renders on the CPU and is fast enough for the 480×480 point-cloud renders:

  ```bash
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
  export LP_NUM_THREADS=8      # optional: cap Mesa's threads when running several jobs in parallel
  python scripts/diag_render.py   # should now print "OpenGL: Mesa | llvmpipe ..." and PASS
  ```

  Mesa may print `libEGL warning: failed to open /dev/dri/...`; that is harmless.
* **torch reports `cuda.is_available() == False`.** Your torch wheel targets a newer CUDA than
  the driver supports. Reinstall torch with the matching `--index-url` (step 2).
* **`ModuleNotFoundError: metaworld.policies.sawyer_push_test_v2_policy` or unknown env
  `pick-bar-v2`.** Upstream Meta-World is installed instead of the fork:
  `pip uninstall metaworld && pip install -e third_party/metaworld`.
* **`diag_render.py` prints FAIL in `scout-brick` for any env other than `brick-slide-v2`.**
  Expected: MuJoCo 2.3.7's segmentation rendering does not work on NVIDIA EGL (and its EGL
  backend cannot use Mesa). The brick task does not need it: its point cloud is built from depth
  renders. Run the other four tasks in `scout`.
* **CUDA allocator assert during training.** The scripts set
  `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`, because `expandable_segments` triggers an
  allocator assert inside gradient-checkpointed double backward (seen on PyTorch 2.1).
  Keep it set if you call the trainer from your own code.

## Pretrained checkpoints and door assets

* **Checkpoints** for all six tasks (`pickbar.pt`, `pushbar.pt`, `faucet.pt`, `box.pt`,
  `brick.pt`, `door.pt`): download them from [Google Drive](https://drive.google.com/drive/folders/1NK_6SHZfI5PTbkabH5GKCWxzVg7_pzfH?usp=drive_link) into `ckpts/`.
* **Ambiguous-door assets** (door task only): PartNet-Mobility doors turned into pairs of
  variants that look identical when closed but open differently (hinge side / direction).
  Either download the generated set from [Google Drive](https://drive.google.com/file/d/1T-7Hf7KgsaaH5ckRqQGsFR8GC_8-aIwV/view?usp=sharing) and extract it to
  `~/datasets/failure_history_door/raw/`, or build it yourself from
  [PartNet-Mobility](https://sapien.ucsd.edu/downloads). Download the door objects listed
  in `scripts/generate_door_assets.py` into `<pm-path>/raw/<id>/`, then:

  ```bash
  python scripts/generate_door_assets.py --pm-path ~/datasets/partnet-mobility \
      --output ~/datasets/failure_history_door      # writes <output>/raw/<id>_<link>[_handle]_<mode>/
  ```

  The train/test split used in the paper is `metadata/multimodal_door.json`. Every door script
  takes `--pm-dir` (default `~/datasets/failure_history_door/raw`) and `--metadata`.

## Evaluating the pretrained models

```bash
# Meta-World (ISE) tasks: mean attempts-to-success over 800 episodes (pick bar, push bar, slide brick)
# or 400 episodes (faucet, box); failed episodes count as 14
python scripts/rollout.py --task pick_bar --ckpt ckpts/pickbar.pt --no-video
python scripts/rollout.py --task push_bar --ckpt ckpts/pushbar.pt --no-video
python scripts/rollout.py --task faucet   --ckpt ckpts/faucet.pt  --no-video
python scripts/rollout.py --task box      --ckpt ckpts/box.pt     --no-video

# Slide Brick: in the scout-brick environment (MuJoCo 2.3.7, see Installation 2b)
conda activate scout-brick
python scripts/rollout.py --task brick    --ckpt ckpts/brick.pt   --no-video
conda activate scout

# without test-time adaptation (proposals sampled from the mixture)
python scripts/rollout.py --task pick_bar --ckpt ckpts/pickbar.pt --ttt-steps 0 --no-video

# ambiguous doors: success rate on the test objects, sweep over the number of inner steps
python scripts/rollout.py --task door --ckpt ckpts/door.pt --ttt-steps 5 1 0
```

| task | paper (mean attempts ↓) |
|---|---|
| Pick Bar | 2.83 |
| Push Bar | 4.14 |
| Turn Faucet | 1.54 |
| Open Box | 1.83 |
| Slide Brick | 3.37 |

Door: success rate 0.83 / 0.97 with 1 / 5 inner steps.

* A full ISE evaluation takes roughly 10–15 minutes per 400 episodes. Use `--n-episodes 20`
  for a quick look. With few episodes the mean is noisy, because a failed episode counts as 14.
* Drop `--no-video` to write one mp4 per episode to `results/<task>/videos_ttt<k>/`. Each frame
  shows the simulator next to the proposal weights, the predicted target and the attempt history.
* Results: `results/<task>/summary_*.pkl` holds the metrics. `episodes_*.pkl` holds every
  attempt's proposals and outcome clouds.
* Proposal selection: with `--ttt-steps > 0` the highest-weight proposal is executed. With
  `--ttt-steps 0` a proposal is sampled from the mixture (the "w/o TTT" baseline). Override with
  `--set rollout.proposal_selection=argmax|sample`.

## Reproducing the paper from scratch

Every task is a preset in `scout/configs.py` holding all camera-ready hyper-parameters:

| task | sim | action | outcome | K | α | history dropout | extras |
|---|---|---|---|---|---|---|---|
| `pick_bar` | Meta-World `pick-bar-v2` | 3-D grasp point | post-action cloud | 4 | 0.1 | 0.3 | |
| `push_bar` | Meta-World `push-test-v2` | 3-D contact point | post-action cloud | 4 | 0.1 | 0.3 | |
| `faucet` | Meta-World `faucet-{open,close}-v3` | 3-D push goal | post-action cloud | 2 | 0.1 | 0.0 | |
| `box` | Meta-World `box-{open,slide}-v2` | 3-D post-grasp target | post-action cloud | 2 | 0.1 | 0.0 | |
| `brick` | Meta-World `brick-slide-v2` | 3-D lift apex | post-action cloud | 4 | 0.1 | 0.3 | MuJoCo 2.3.7 (`scout-brick`) |
| `door` | PyBullet / PartNet-Mobility | 6-D point + pull direction | per-point flow | 4 | 0.5 | 0.3 | flow-reconstruction loss, AdamW |

Any field can be overridden on the command line with `--set section.field=value` (the section
prefix is optional when unambiguous), e.g. `--set train.epochs=500 rollout.ttt_steps=5`.

### 1. Collect offline data

The scripted experts try a random number of wrong actions before the correct one. Each
successful episode is stored with its failed attempts as the history.

```bash
python scripts/collect_data.py --task pick_bar --num-episodes 200 --output data/pick_bar.pkl
python scripts/collect_data.py --task push_bar --num-episodes 200 --output data/push_bar.pkl
python scripts/collect_data.py --task faucet   --num-episodes 200 --output data/faucet.pkl
python scripts/collect_data.py --task box      --num-episodes 200 --output data/box.pkl
python scripts/collect_data.py --task brick    --num-episodes 200 --output data/brick.pkl   # in scout-brick

# doors: parallel trajectory generation, then split into per-trajectory files
python scripts/collect_data.py --task door --output-dir data/door --num-processes 16
python scripts/prepare_door_dataset.py --raw-dir data/door/raw_pkls --out-dir data/door/train_val
```

### 2. Meta-train

```bash
python scripts/train.py --task pick_bar --data data/pick_bar.pkl
python scripts/train.py --task push_bar --data data/push_bar.pkl
python scripts/train.py --task faucet   --data data/faucet.pkl
python scripts/train.py --task box      --data data/box.pkl
python scripts/train.py --task brick    --data data/brick.pkl
python scripts/train.py --task door     --data data/door/train_val
```

* Checkpoints are written to `ckpts/<task>/`:
  * `best.pt`: lowest post-adaptation validation NLL
  * `best_gap.pt`: largest pre-minus-post improvement
  * `latest.pt`: most recent epoch

  They embed their model config.
* Each epoch prints the validation NLL *before* and *after* one inner step. A post value below
  the pre value means adaptation is working.
* ISE data is split sequentially: the first 90% of the pickle is training data, the rest
  validation. Use `--set data.shuffle_split=true` for a seeded random split instead.
* Logging uses wandb. Pass `--no-wandb` to disable it.

### 3. Evaluate

As above, with your own checkpoint, e.g.
`python scripts/rollout.py --task pick_bar --ckpt ckpts/pick_bar/best.pt --no-video`.

## Using the model in your own code

```python
from scout import SCOUT, get_task_config
from scout.inference import adapt_from_history, predict_target

cfg = get_task_config("pick_bar")
# cfg.model is needed for the released weights (no embedded config); optional for your own checkpoints
model = SCOUT.from_checkpoint("ckpts/pickbar.pt", cfg=cfg.model, device="cuda")

# eval_pcd: [N, 3] query point cloud; hist_*: lists of past attempts (clouds, 3-D actions, outcome clouds)
q = adapt_from_history(model, eval_pcd, hist_pcds, hist_actions, hist_outcomes,
                       n_steps=1, normalize=cfg.data.normalize)          # belief update
target, weights, means, _ = predict_target(model, q, eval_pcd, use_argmax=True,
                                           normalize=cfg.data.normalize)  # action
```

## Repository layout

```
scout/
  configs.py            dataclass configs + TASKS registry (all camera-ready hyper-parameters)
  cli.py                shared --task / --set argument handling for the scripts
  inference.py          adapt_from_history / predict_target, used by every rollout
  models/
    attention.py        manual-attention transformer blocks (double-backward safe), FiLM block
    outcome_encoder.py  E(o_t+1): point-cloud variant (ISE) / flow variant (door), flow decoder
    gmm_head.py         policy network π(q, o) -> (w, μ)
    dynamics.py         forward dynamics f(q, a, o) -> z
    losses.py           GMM negative log-likelihood
    scout.py            SCOUT bundle: inner_update / adapt / predict / checkpoint I/O
  data/
    metaworld_dataset.py  episodes of failed attempts -> (query, history) samples (ISE tasks)
    door_dataset.py       trajectory pickles -> (query, history) samples (doors)
    pointcloud.py         sampling / normalisation helpers
  training/
    trainer.py          meta-training loop (inner loop + outer NLL [+ recon] + grad accumulation)
    evaluate.py         offline pre/post-adaptation NLL
    common.py           shared batch / reconstruction helpers
  envs/
    metaworld/          ISE tasks: shared collect + rollout loops, per-task hooks in tasks/*.py
                        (pick_bar, push_bar, faucet, box, brick), segmented point clouds
                        (pointcloud.py), rollout videos (viz.py)
    door/               ambiguous doors: self-contained PyBullet simulation (ported from HAVE),
                        trajectory generation (collect.py), closed-loop rollout, videos (viz.py)
scripts/
  diag_render.py        headless rendering check
  smoke_test.sh         tiny end-to-end collect -> train -> rollout for every task
  collect_data.py       offline data collection (all tasks)
  generate_door_assets.py  PartNet-Mobility doors -> ambiguous door assets
  prepare_door_dataset.py  raw door pickles -> train/val layout
  train.py              meta-training
  rollout.py            closed-loop evaluation in simulation
metadata/
  multimodal_door.json             door train/test split used in the paper
  movable_links_fullset_000_full.json  movable links of the PartNet-Mobility objects (asset generation)
third_party/metaworld/  Meta-World v3.0.0 fork with the SCOUT environments and scripted experts
tests/                  CPU tests: models/trainer (test_models.py), door simulator (test_door_sim.py)
```

## Notes on this release

* Model, inner loop and trainer are a single implementation shared by all tasks; the
  differences between tasks are entirely in `scout/configs.py`.
* ISE data processing (`AttemptHistoryDataset`):
  * episodes without history are dropped;
  * every cloud is sub-sampled to 1200 points;
  * query and history are normalised by the query cloud's centre and radius;
  * with probability `augment_prob`, the ground-truth interaction is appended to the history.
* The pick-bar and push-bar environments generate their scene XML (the bar and its hidden
  centre of mass) when the env is constructed. Each env writes a private temporary file, so
  several envs can safely run in parallel.
* The door simulation is a trimmed port of the HAVE (CoRL 2025) code base and its rpad /
  FlowBot3D dependencies. Its behaviour is unchanged (same gripper, approach, pull and analytic
  flow); only unused code paths and the trimesh / plotly dependencies were dropped.
* **Slide Brick needs MuJoCo ≤ 2.3.7.** The brick is tilted up a ramp and released; it must
  slide back and come to rest inside a 2 cm height window, and the right tilt height depends on
  the hidden ramp friction (U(0.245, 0.395)). The checkpoint was trained in the original
  mujoco-py (MuJoCo 2.1) Meta-World, and from MuJoCo 3.0 the dynamics of this task change: the arm
  settles in a different joint configuration during the reset and the slide ends elsewhere. With
  the scripted expert's ground-truth tilt height, one attempt per episode, 60 episodes:

  | backend | expert success |
  |---|---|
  | original (mujoco-py, MuJoCo 2.1) | 0.90 |
  | this repo, MuJoCo 2.3.7 | 0.90 |
  | this repo, MuJoCo 3.15 | 0.07 |

  MuJoCo 2.1.5, 2.2.2 and 2.3.7 reproduce the original reset state exactly; every 3.x version
  tested (3.0.1–3.15) does not. Hence the separate `scout-brick` environment. The other four tasks
  were trained on MuJoCo 3 and stay in `scout`.
* The brick env, `brick-slide-v2`
  (`third_party/metaworld/metaworld/envs/sawyer_brick_slide_ise_v2.py`), keeps three details of the
  original Meta-World that differ from the v3 base class, scoped to this env only: the mocap weld
  uses `torquescale = 1` (v3: 5); `step` builds the observation without the extra `mj_forward`
  and observation clipping; and success means coming to rest at a height in (0.073, 0.093) m
  after being lifted above 0.14 m. Its expert
  (`metaworld/policies/sawyer_brick_slide_ise_policy.py`) is the original code. The fork's
  `slide-brick-v2` is a different, unused variant of the task.
* Brick point clouds: MuJoCo 2.3.7's segmentation render is broken on NVIDIA EGL, so the brick
  mask is taken from depth instead (pixels where a brick-only depth render matches the
  full-scene one), followed by the original 3×3 median filter. Against the original clouds at
  identical states: mean Chamfer distance 0.04 mm and max-radius difference ≤ 0.5 mm. The
  original drops one pixel row where the brick touches the ramp, which moves the centre by ~1 mm.

## Citation

```bibtex
@inproceedings{li2026scout,
  title     = {Scouting the Dynamics Gap: Test-Time Policy Adaptation via Action-Outcome Feedback},
  author    = {Li, Yishu and Geng, Liyuan and Mao, Xinyi and Li, Amber and Held, David},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```
