"""Put doPlan windows and MTR ego predictions on the nuPlan log clock (spec 4.4).

nuPlan's devkit (extract_sensor_tokens_as_scenario + get_sampled_sensor_tokens_in_time_window_from_db)
builds a scenario from the lidar_pc rows with timestamp >= int(anchor + extraction_offset * 1e6), in
time order, every second row (subsample ratio 0.5). ScenarioNet step i is therefore lidar frame
k0 + 2 * i, where k0 is the first frame at or after the start time. Reproducing that inclusive rule on
frame indices matches MTR's gt_trajs to the recorded ego_pose to under 1 cm (toy log, 2026-10-23);
"start + 0.1 s * i" is one frame (45 ms) off because the frame nominally at the start time sits a
fraction of a millisecond before it. The ScenarioNet "world" frame is the nuPlan frame translated so
the ego is at (0, 0) at step 0; headings are unchanged. doPlan frames are 5 per second from the first
lidar_pc timestamp of the log.
"""
from __future__ import annotations

import math
import pickle
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .text import Instruction

FRAMES_PER_STEP = 2        # ScenarioNet subsample_ratio 0.5 of the 20 Hz lidar
PRED_RE = re.compile(r'^(?P<scenario>[0-9a-f]+)___(?P<obj>.+?)___(?P<idx>\d+)___\.pt$')
MOMENT_FRAMES = FRAMES_PER_STEP * 5     # one 0.5 s bucket = 10 lidar frames
TIME_COLUMNS = ['pred_file', 'scenario_id', 'log_name', 'time_index', 'frame', 't_pred_us', 'moment']
PAIR_COLUMNS = TIME_COLUMNS + ['annotation_id', 'clip_id', 'instruction', 'first_clause', 'rule_intent']
COVERAGE_COLUMNS = ['annotation_id', 'clip_id', 'window_s', 'n_moments', 'covered_s', 'frac_covered']


class AlignmentError(RuntimeError):
    """Predictions cannot be placed on the nuPlan clock. Never skipped silently."""


@dataclass(frozen=True)
class LidarFrames:
    """The lidar_pc frames of one log in time order, with the ego_pose linked to each frame."""
    tokens: tuple[str, ...]
    ts: np.ndarray          # int64 microseconds
    ego_xy: np.ndarray      # [n, 2] world metres (nuPlan rear-axle pose)
    ego_yaw: np.ndarray     # [n] radians
    index: dict[str, int] = field(init=False, repr=False)   # token -> frame number

    def __post_init__(self) -> None:
        object.__setattr__(self, 'index', {t: i for i, t in enumerate(self.tokens)})


def load_lidar_frames(db_path: Path) -> LidarFrames:
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        rows = con.execute(
            'SELECT lower(hex(l.token)), l.timestamp, e.x, e.y, e.qw, e.qx, e.qy, e.qz '
            'FROM lidar_pc l JOIN ego_pose e ON e.token = l.ego_pose_token ORDER BY l.timestamp').fetchall()
    finally:
        con.close()
    if not rows:
        raise AlignmentError(f'{db_path.name}: no lidar_pc frames')
    tokens = tuple(r[0] for r in rows)
    yaw = np.array([math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                    for _, _, _, _, qw, qx, qy, qz in rows])
    return LidarFrames(tokens=tokens, ts=np.array([r[1] for r in rows], dtype=np.int64),
                       ego_xy=np.array([[r[2], r[3]] for r in rows], dtype=np.float64), ego_yaw=yaw)


def log_start_us(frames: LidarFrames) -> int:
    return int(frames.ts[0])


def scenario_start_frame(meta: dict, frames: LidarFrames) -> int:
    """Frame number of ScenarioNet step 0: the first lidar frame at or after anchor + offset."""
    tok = str(meta['scenario_token']).lower()
    if tok not in frames.index:
        raise AlignmentError(f'scenario token {tok} is not a lidar_pc token of log {meta.get("log_name")}')
    offset_s = float(meta['scenario_extraction_info']['extraction_offset'])
    start_us = int(frames.ts[frames.index[tok]] + offset_s * 1e6)       # same truncation as the devkit
    k0 = int(np.searchsorted(frames.ts, start_us, side='left'))         # first frame with ts >= start_us
    if k0 >= len(frames.ts):
        raise AlignmentError(f'scenario {meta.get("scenario_id")}: start lies after the end of the log')
    return k0


def prediction_frame(start_frame: int, time_index: int) -> int:
    return start_frame + FRAMES_PER_STEP * time_index


def prediction_time_us(frames: LidarFrames, start_frame: int, time_index: int) -> int:
    k = prediction_frame(start_frame, time_index)
    if k >= len(frames.ts):
        raise AlignmentError(f'time index {time_index} from frame {start_frame} runs past the end of the log')
    return int(frames.ts[k])


def moment_key(log_name: str, frame: int) -> str:
    """Identifies one 0.5 s bucket of one drive; predictions from overlapping scenarios share it."""
    return f'{log_name}:{frame // MOMENT_FRAMES}'


def window_us(t0_us: int, start_frame: int, end_frame: int, frame_s: float = 0.2) -> tuple[int, int]:
    return (t0_us + int(round(start_frame * frame_s * 1e6)),
            t0_us + int(round(end_frame * frame_s * 1e6)))


def list_ego_predictions(pt_dir: Path) -> list[tuple[str, str, int]]:
    """(file name, scenario id, time index) for every ego prediction file, sorted by name."""
    out = []
    for p in sorted(pt_dir.glob('*___ego___*___.pt')):
        m = PRED_RE.match(p.name)
        if m and m.group('obj') == 'ego':
            out.append((p.name, m.group('scenario'), int(m.group('idx'))))
    return out


def load_summary(scenarionet_dir: Path) -> dict[str, dict]:
    """scenario_id -> ScenarioNet metadata. dataset_summary.pkl is our own converter's output."""
    with open(scenarionet_dir / 'dataset_summary.pkl', 'rb') as f:
        summary = pickle.load(f)
    return {str(m['scenario_id']): m for m in summary.values()}


def prediction_times(predictions, summary: dict[str, dict], frames_by_log: dict[str, LidarFrames],
                     cfg: Config) -> pd.DataFrame:
    starts: dict[str, int] = {}
    rows = []
    for fname, sid, idx in predictions:
        if sid not in summary:
            raise AlignmentError(f'{fname}: scenario {sid} is missing from dataset_summary.pkl')
        meta = summary[sid]
        if meta['log_name'] not in frames_by_log:
            raise AlignmentError(f'{fname}: no lidar frames for log {meta["log_name"]}')
        frames = frames_by_log[meta['log_name']]
        if sid not in starts:
            starts[sid] = scenario_start_frame(meta, frames)
        k = prediction_frame(starts[sid], idx)
        t = prediction_time_us(frames, starts[sid], idx)
        rows.append({'pred_file': fname, 'scenario_id': sid, 'log_name': meta['log_name'],
                     'time_index': idx, 'frame': k, 't_pred_us': t, 'moment': moment_key(meta['log_name'], k)})
    return pd.DataFrame(rows, columns=TIME_COLUMNS)


def _in_window(pred_times: pd.DataFrame, ins: Instruction, frames_by_log, cfg: Config) -> pd.Series:
    a, b = window_us(log_start_us(frames_by_log[ins.clip_id]), ins.start_frame, ins.end_frame, cfg.doplan_frame_s)
    return (pred_times.log_name == ins.clip_id) & (pred_times.t_pred_us >= a) & (pred_times.t_pred_us <= b)


def build_pairs(instructions: list[Instruction], pred_times: pd.DataFrame, frames_by_log, cfg: Config) -> pd.DataFrame:
    """One row per (ego prediction inside the window, annotation). Logs without a .db are skipped."""
    rows = []
    for ins in instructions:
        if ins.clip_id not in frames_by_log:
            continue
        for r in pred_times[_in_window(pred_times, ins, frames_by_log, cfg)].itertuples(index=False):
            rows.append({'pred_file': r.pred_file, 'scenario_id': r.scenario_id, 'log_name': r.log_name,
                         'time_index': r.time_index, 'frame': r.frame, 't_pred_us': r.t_pred_us, 'moment': r.moment,
                         'annotation_id': ins.annotation_id, 'clip_id': ins.clip_id, 'instruction': ins.text,
                         'first_clause': ins.first_clause, 'rule_intent': ins.rule_intent})
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def coverage_table(instructions: list[Instruction], pred_times: pd.DataFrame, frames_by_log, cfg: Config) -> pd.DataFrame:
    rows = []
    for ins in instructions:
        if ins.clip_id not in frames_by_log:
            continue
        n = int(pred_times.loc[_in_window(pred_times, ins, frames_by_log, cfg), 'moment'].nunique())
        window_s = (ins.end_frame - ins.start_frame) * cfg.doplan_frame_s
        covered = min(n * cfg.pred_stride_s, window_s)
        rows.append({'annotation_id': ins.annotation_id, 'clip_id': ins.clip_id, 'window_s': window_s,
                     'n_moments': n, 'covered_s': covered,
                     'frac_covered': covered / window_s if window_s > 0 else 0.0})
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def _gt_now(path: Path, past_len: int) -> np.ndarray:
    return np.asarray(torch.load(path, map_location='cpu', weights_only=False)['gt_trajs'])[past_len - 1]


def rear_axle_to_centre_m(meta: dict) -> float:
    """nuPlan's ego_pose is the rear axle; ScenarioNet records the vehicle centre. Their displacements
    differ whenever the heading changes (up to 1.5 m on lane changes), so the check must compare centres."""
    vp = meta.get('ego_vehicle_parameters') or {}
    if 'front_length' in vp and 'rear_length' in vp:
        return (float(vp['front_length']) - float(vp['rear_length'])) / 2.0
    return 0.0


def ego_centre_xy(frames: LidarFrames, frame: int, offset_m: float) -> np.ndarray:
    yaw = float(frames.ego_yaw[frame])
    return frames.ego_xy[frame] + offset_m * np.array([math.cos(yaw), math.sin(yaw)])


def sanity_check(frames: LidarFrames, meta: dict, pt_dir: Path, cfg: Config,
                 tol_m: float = 0.05, tol_rad: float = 0.02, predictions=None) -> dict:
    """Compare one scenario's MTR ego track with the ego_pose of the lidar frames it maps to, after
    shifting the recorded rear-axle pose to the vehicle centre that ScenarioNet uses.
    `predictions` is an optional pre-listed list_ego_predictions(pt_dir) result, to avoid re-globbing per scenario."""
    sid = str(meta['scenario_id'])
    preds = list_ego_predictions(pt_dir) if predictions is None else predictions
    idxs = sorted(i for _, s, i in preds if s == sid)
    if len(idxs) < 2:
        raise AlignmentError(f'scenario {sid}: fewer than two ego predictions in {pt_dir}')
    i0, i1 = idxs[0], min(idxs[1:], key=lambda i: abs(i - 20))
    k0 = scenario_start_frame(meta, frames)
    f0, f1 = prediction_frame(k0, i0), prediction_frame(k0, i1)
    if f1 >= len(frames.ts):
        raise AlignmentError(f'scenario {sid}: prediction index {i1} runs past the end of the log')
    g0 = _gt_now(pt_dir / f'{sid}___ego___{i0}___.pt', cfg.past_len)
    g1 = _gt_now(pt_dir / f'{sid}___ego___{i1}___.pt', cfg.past_len)
    c = rear_axle_to_centre_m(meta)
    disp_err = float(np.linalg.norm((g1[:2] - g0[:2]) - (ego_centre_xy(frames, f1, c) - ego_centre_xy(frames, f0, c))))
    yaw_err = abs((float(g0[6]) - float(frames.ego_yaw[f0]) + math.pi) % (2 * math.pi) - math.pi)
    if disp_err > tol_m:
        raise AlignmentError(f'scenario {sid} ({meta.get("log_name")}): ego displacement between indices '
                             f'{i0} and {i1} differs from ego_pose by {disp_err:.3f} m (> {tol_m} m)')
    if yaw_err > tol_rad:
        raise AlignmentError(f'scenario {sid} ({meta.get("log_name")}): heading differs from ego_pose yaw '
                             f'by {yaw_err:.3f} rad (> {tol_rad})')
    return {'scenario_id': sid, 'log_name': meta.get('log_name'), 'start_frame': k0,
            'displacement_error_m': disp_err, 'heading_error_rad': float(yaw_err)}
