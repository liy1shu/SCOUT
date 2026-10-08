#!/usr/bin/env bash
# End-to-end smoke test: tiny data collection -> short training -> short rollout for
# every task, with the real simulators.  A few minutes on a GPU machine (door excluded).
#
#   bash scripts/smoke_test.sh                 # all tasks
#   TASKS="pick_bar faucet" bash scripts/smoke_test.sh
#   DOOR_PM_DIR=~/datasets/failure_history_door/raw bash scripts/smoke_test.sh
#
# Outputs go to $OUT (default ./smoke) and are safe to delete afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-./smoke}
TASKS=${TASKS:-"pick_bar push_bar faucet box door"}
DOOR_PM_DIR=${DOOR_PM_DIR:-~/datasets/failure_history_door/raw}
DOOR_META=${DOOR_META:-metadata/multimodal_door.json}
N_COLLECT=${N_COLLECT:-6}        # ISE episodes to collect per task
N_ROLLOUT=${N_ROLLOUT:-3}        # ISE episodes to roll out per setting
TRAIN_SET="train.epochs=2 train.effective_batch_size=2 train.num_workers=0"

mkdir -p "$OUT"
python -c "import metaworld.policies.sawyer_push_test_v2_policy" 2>/dev/null \
  || { echo "Meta-World fork not installed: pip install -e third_party/metaworld"; exit 1; }
echo "== unit tests =="
python tests/test_models.py
python tests/test_door_sim.py

for task in $TASKS; do
  echo; echo "================ $task ================"
  if [ "$task" != "door" ]; then
    python scripts/collect_data.py --task "$task" --num-episodes "$N_COLLECT" --output "$OUT/data/$task.pkl"
    python scripts/train.py --task "$task" --data "$OUT/data/$task.pkl" --save-dir "$OUT/ckpts" \
        --no-wandb --set $TRAIN_SET data.train_ratio=0.67
    python scripts/rollout.py --task "$task" --ckpt "$OUT/ckpts/$task/best.pt" --ttt-steps 1 0 \
        --n-episodes "$N_ROLLOUT" --out-dir "$OUT/results" --no-video --set rollout.max_attempts=4
  else
    python scripts/collect_data.py --task door --pm-dir "$DOOR_PM_DIR" --metadata "$DOOR_META" \
        --output-dir "$OUT/data/door" --num-processes 2 --max-objects 2 \
        --traj-per-obj-train 2 --traj-per-obj-test 1 --max-trials-train 4 --max-trials-test 2 --history-len 2
    python scripts/prepare_door_dataset.py --raw-dir "$OUT/data/door/raw_pkls" --out-dir "$OUT/data/door/train_val"
    python scripts/train.py --task door --data "$OUT/data/door/train_val" --save-dir "$OUT/ckpts" \
        --no-wandb --set $TRAIN_SET
    python scripts/rollout.py --task door --ckpt "$OUT/ckpts/door/best.pt" --ttt-steps 1 0 \
        --pm-dir "$DOOR_PM_DIR" --metadata "$DOOR_META" --max-objects 1 --out-dir "$OUT/results" \
        --set rollout.trials_per_object=1 rollout.max_attempts=5
  fi
done
echo; echo "smoke test finished: outputs in $OUT"
