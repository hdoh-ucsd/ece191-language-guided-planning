"""Every tunable number and name used by the doPlan language re-ranking pipeline.

Spec: docs/superpowers/specs/2026-10-23-doplan-word2vec-mtr-rerank-design.md
"""
from __future__ import annotations

from dataclasses import dataclass

# Order matters: geometry.compat_table stacks its columns in this order.
INTENTS = ('turn_left', 'turn_right', 'lane_left', 'lane_right', 'stop', 'straight', 'other')
LATERAL_INTENTS = ('turn_left', 'turn_right', 'lane_left', 'lane_right')

# Evaluation logs (spec section 6). Their text is held out of word2vec and classifier training.
EVAL_LOGS = (
    '2021.10.11.02.57.41_veh-50_00352_00535',
    '2021.10.11.08.31.07_veh-50_01750_01948',
    '2021.10.11.02.57.41_veh-50_01522_02088',
    '2021.08.17.18.54.02_veh-45_00665_01065',
    '2021.08.17.17.17.01_veh-45_02314_02798',
    '2021.08.17.16.57.11_veh-08_01200_01636',
)


@dataclass(frozen=True)
class Config:
    # word2vec (spec 4.2)
    vector_size: int = 100
    window: int = 5
    min_count: int = 2
    epochs: int = 50
    seed: int = 0
    sg: int = 1              # skip-gram
    negative: int = 10       # negative samples
    # timing (spec 4.4)
    doplan_frame_s: float = 0.2   # doPlan frames are 5 per second of drive
    past_len: int = 21            # rows of gt_trajs before the prediction time; row 20 is "now"
    pred_stride_s: float = 0.5    # MTR writes a prediction every 5 steps
    # geometry thresholds (spec 4.5); metres and degrees
    turn_lat_m: float = 2.0
    turn_heading_deg: float = 30.0
    lane_lat_m: float = 1.5
    lane_heading_deg: float = 20.0
    straight_lat_m: float = 1.5
    straight_heading_deg: float = 15.0
    stop_path_m: float = 3.0
    compat_floor: float = 0.02
    # gate and fusion (spec 4.6)
    gate_min_score: float = 0.05   # modes below this normalised MTR score do not count as a fork
    gate_endpoint_m: float = 4.0
    gate_heading_deg: float = 20.0
    lam: float = 1.0
    # evaluation (spec 5)
    compliance_m: float = 1.0
    bootstrap_n: int = 1000
    n_example_plots: int = 12
