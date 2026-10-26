#!/usr/bin/env bash
set -euo pipefail

# Run INSIDE the nuplan-convert container (CPU only). Converts every nuPlan log found in
# $NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini into a ScenarioNet database that the mtr
# service can read via run_inference.sh (SCENARIONET_DIR=/mnt/scenarionet/<name>).
RAW="${NUPLAN_RAW_DIR:-$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini}"
OUT="${SCENARIONET_OUT:-/mnt/scenarionet/nuplan_toy}"
WORKERS="${NUM_WORKERS:-1}"

if ! compgen -G "$RAW/*.db" >/dev/null; then
    echo "ERROR: no .db logs under $RAW (fetch one with scripts/fetch_nuplan_toy.py)" >&2
    exit 2
fi
if [[ ! -f "$NUPLAN_MAPS_ROOT/nuplan-maps-v1.0.json" ]]; then
    echo "ERROR: $NUPLAN_MAPS_ROOT/nuplan-maps-v1.0.json missing" >&2
    exit 3
fi
echo "logs:"; ls -l "$RAW"/*.db
echo "maps:"; find "$NUPLAN_MAPS_ROOT" -name map.gpkg -exec ls -l {} \;

cd /opt/valeo4cast/scenarionet
python3 -m scenarionet.convert_nuplan \
    --database_path "$OUT" \
    --raw_data_path "$RAW" \
    --num_workers "$WORKERS" \
    --overwrite "$@"

echo "converted database at $OUT:"
ls -l "$OUT"
python3 - "$OUT" <<'EOF'
import pickle, sys, os
root = sys.argv[1]
summary = pickle.load(open(os.path.join(root, 'dataset_summary.pkl'), 'rb'))
mapping = pickle.load(open(os.path.join(root, 'dataset_mapping.pkl'), 'rb'))
if not summary:
    sys.exit('ERROR: conversion produced 0 scenarios. If the log has no mission goals, make sure '
             'NUPLAN_REMOVE_INVALID_GOALS is not set to true.')
first = next(iter(summary))
print(f'{len(summary)} scenarios; first: {first!r} in subdir {mapping[first]!r}')
for name, meta in summary.items():
    ns = meta.get('number_summary', {})
    print(f'  {name}: type={meta.get("scenario_type")} steps={len(meta.get("ts", []))} '
          f'objects={ns.get("num_objects")} map_features={ns.get("num_map_features")}')
EOF
