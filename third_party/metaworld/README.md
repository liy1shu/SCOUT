# Meta-World fork for the SCOUT ISE tasks

This is [Meta-World](https://github.com/Farama-Foundation/Metaworld) v3.0.0
(MIT license, Farama Foundation; see `LICENSE`) with the environments and
scripted experts used by the SCOUT ISE experiments. Install it in place of
upstream Meta-World:

```bash
pip install -e third_party/metaworld
```

## Additions

| gym `env_name` | environment | scripted expert |
|---|---|---|
| `pick-bar-v2` | `envs/sawyer_pick_bar_v2.py` – bar with a hidden centre of mass (`cm_offset`) | `policies/sawyer_pick_bar_v2_policy.py` |
| `push-test-v2` | `envs/sawyer_push_test_v2.py` – push the same bar | `policies/sawyer_push_test_v2_policy.py` |
| `box-open-v2` | `envs/sawyer_box_open_v2.py` – lid lifts off | `policies/sawyer_box_open_v2_policy.py` |
| `box-slide-v2` | `envs/sawyer_box_slide_v2.py` – lid slides (1-DOF joint) | `policies/sawyer_box_slide_v2_policy.py` |
| `slide-brick-v2` | `envs/sawyer_brick_slide_v2.py` – brick with hidden friction (not used by this release) | `policies/sawyer_brick_slide_v2_policy.py` |

The experts expose `get_adaptive_action(obs)`: a state machine that executes a
random number of *wrong* attempts before the correct one, with
`current_is_keyframe` / `keyframe_type` (`'hover'` before, `'lift'` after an
attempt), `target` (the attempted 3-D action), `attempt_done` (the caller
resets the object), and `external_target` (the rollout injects SCOUT's
prediction). The faucet experts (`policies/sawyer_faucet_{open,close}_v3_policy.py`)
are rewritten the same way.

## Modifications to upstream files

* `envs/sawyer_faucet_{open,close}_v3.py` load `assets_v2/sawyer_xyz/sawyer_faucet_{open,close}.xml`,
  variants that differ only in an invisible stop wall (`box_left`) and mirrored goal sites.
  The wall sits at x = ∓0.12, flush with the handle, so pushing the handle the wrong way
  barely moves it (≈0.16 rad); the two contact parameter sets (`solimp`/`solref`) also
  differ. Keep it there: the released faucet checkpoint was trained with this setting, and
  with the wall at ±0.20 the handle swings ~56° before touching it and the task loses its
  ambiguity-resolving signal.
* `sawyer_xyz_env.py`, `__init__.py` and every `envs/sawyer_*_v3.py`: a `width`/`height`
  render-size argument (default 480×480) is threaded through to the renderer;
  `max_path_length` is 1000 instead of 500; `random_init` defaults to True.
* `assets/objects/assets/xyz_base.xml`: an extra fixed camera `corner4`.
* `env_dict.py`, `envs/__init__.py`: registration of the new environments.
* New asset directories: `assets_v2/` (v2-style scenes including the box, faucet and
  push-test scenes), `envs/objects/`, `envs/push_test/` (bar scene) and `envs/textures/`.
  `asset_path_utils.full_v2_path_for` resolves into `assets_v2`.
* `envs/sawyer_{pick_bar,push_test,brick_slide}_v2.py` generate their bar/brick MJCF at
  construction time. They used to write it to one shared file in `envs/push_test/`, so envs
  created concurrently (parallel processes, or pick-bar next to push-bar, which write
  different masses) could load each other's scene. Each env instance now writes a private
  temporary file (`asset_path_utils.write_generated_xml`), removed at exit; the generated
  XML itself is unchanged.

The upstream `assets_v1/` directory of the fork is not referenced by any code
and is omitted here. `envs/sawyer_box_{open,slide}_v2.py` imported `Box` from
the legacy `gym` package; they now import it from `gymnasium` (identical API),
so the old `gym` package is no longer required.
