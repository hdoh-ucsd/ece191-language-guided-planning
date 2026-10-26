#!/usr/bin/env bash
set -euo pipefail

# Run INSIDE the mtr container. End-to-end pretrained MTR inference on the nuScenes
# mini-val scenarios that ship with Valeo4Cast (UniTraj/unitraj/data_samples/nuscenes),
# so the input/output format can be studied without any nuPlan data.
#
# The samples are copied out of the image into the mounted /mnt/scenarionet volume so
# the preprocessing cache and the predictions survive container restarts.
SRC=/opt/valeo4cast/UniTraj/unitraj/data_samples/nuscenes
TOY="${TOY_DIR:-/mnt/scenarionet/nuscenes_toy}"

if [[ ! -f "$TOY/dataset_summary.pkl" ]]; then
    echo "Copying bundled nuScenes samples to $TOY"
    mkdir -p "$TOY"
    cp -r "$SRC"/. "$TOY"/
fi

# predict.py refuses to overwrite; keep old predictions under a timestamped name.
if [[ -d "$TOY/output_MTR" ]] && compgen -G "$TOY/output_MTR/*" >/dev/null; then
    prev="$TOY/output_MTR.prev.$(date +%s)"
    echo "Moving previous predictions to $prev"
    mv "$TOY/output_MTR" "$prev"
fi

# run_inference.sh already limits to debug_max_nb_scenarios=2; extra hydra overrides pass through.
SCENARIONET_DIR="$TOY" bash /opt/mtr-scripts/run_inference.sh "$@"

mkdir -p /mnt/outputs
python3 /opt/mtr-scripts/inspect_io.py \
    --cache "$TOY/cache_test_MTR" \
    --outputs "$TOY/output_MTR" \
    --plot /mnt/outputs/toy_forecast.png \
    | tee /mnt/outputs/toy_io_report.txt
