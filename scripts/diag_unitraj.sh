#!/usr/bin/env bash
# Diagnose how the editable UniTraj install exposes the `unitraj` namespace package.
set -u
cd /opt/valeo4cast/UniTraj/unitraj
echo "--- autobot package files"; ls models/autobot/ | head
echo "--- editable install hooks"
ls /usr/local/lib/python3.10/dist-packages/ | grep -iE 'editable|unitraj|scenarionet|easy-install'
for f in /usr/local/lib/python3.10/dist-packages/__editable__*.pth /usr/local/lib/python3.10/dist-packages/easy-install.pth; do
    [[ -f "$f" ]] && { echo "== $f"; head -c 800 "$f"; echo; }
done
echo "--- resolution from UniTraj/unitraj (cwd)"
python3 -c 'import unitraj; print("unitraj.__path__ =", list(unitraj.__path__))'
python3 -c 'import unitraj.models as m; print("unitraj.models.__path__ =", list(m.__path__))'
python3 -c 'import unitraj.models.autobot.autobot; print("autobot import OK")' 2>&1 | tail -1
python3 -c 'from unitraj.models.mtr.ops.knn import knn_cuda; print("knn_cuda via unitraj:", knn_cuda.__file__)' 2>&1 | tail -1
echo "--- resolution with predict.py path hack"
python3 -c 'import sys; sys.path.append("./models/mtr/ops/"); sys.path.append("./"); from models import build_model; print("build_model import OK")' 2>&1 | tail -1
