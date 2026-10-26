#!/usr/bin/env bash
set -euo pipefail

# Run INSIDE the mtr container. Dataset and checkpoint are provided via Compose mounts.
DATA_DIR="${SCENARIONET_DIR:-/mnt/scenarionet}"
CKPT="${MTR_CKPT:-}"
if [[ -z "$CKPT" ]]; then
    CKPT="$(find /mnt/weights -name checkpoint_for_val_test.ckpt -print -quit)"
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "ERROR: No checkpoint found below /mnt/weights" >&2
    exit 2
fi
if [[ ! -f "$DATA_DIR/dataset_summary.pkl" || ! -f "$DATA_DIR/dataset_mapping.pkl" ]]; then
    echo "ERROR: Missing ScenarioNet summary/mapping in $DATA_DIR" >&2
    echo "Mount the converted ScenarioNet database into /mnt/scenarionet first." >&2
    exit 3
fi
if [[ -d "$DATA_DIR/output_MTR" ]] && compgen -G "$DATA_DIR/output_MTR/*" >/dev/null; then
    echo "ERROR: $DATA_DIR/output_MTR already contains predictions." >&2
    echo "Rename or move old outputs before rerunning; upstream script refuses overwrites." >&2
    exit 4
fi
cd /opt/valeo4cast/UniTraj/unitraj
python3 predict.py --config-name=challenge_config_noisy_val \
  "val_data_path=[$DATA_DIR]" \
  "ckpt_path=$CKPT" \
  'devices=[0]' \
  'method.eval_batch_size=8' \
  'load_num_workers=0' \
  'debug_max_nb_scenarios=2' \
  'debug_challenge=false' \
  'drop_no_future=false' \
  'overwrite_cache=false' "$@"
