"""Per-scene view of the re-ranking: MTR's ego modes from a scene's start timestamp onwards, before and
after re-ranking with the doPlan instructions whose windows cover each prediction time.

A scene is (name, clip_id, start_us): a nuPlan log and a lidar timestamp in microseconds. For every
`stride_s` from the start up to `horizon_s`, the ego prediction on that log nearest to the moment (within
`tol_s`) is reported, joined with the pairs of an existing run_all.py run (its reranked.csv). Predictions
that no instruction window covers are reported with MTR's scores only. Next to the per-moment table and
plots, <scene>_paths.png draws the recorded drive with the original (MTR) and re-ranked top-1 paths on
the lane geometry, <scene>_changed.png one panel per changed (moment, instruction), and
<scene>_changed.csv the geometry of both top-1 modes at those moments (scene_paths.py).

Usage (from the project root, after run_all.py wrote --run):
  PYTHONPATH=scripts python3 -m doplan_lang.scene_report \\
      --scenarionet scenarionet_data/nuplan_scenes --nuplan nuplan_data \\
      --csv nuplan_data/doplan/release_annotationsV1.csv \\
      --run outputs/doplan_scenes --scenes outputs/doplan_scenes/scenes.csv
scenes.csv columns: scene, clip_id, start_us.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .align import (LidarFrames, list_ego_predictions, load_lidar_frames, load_summary, log_start_us,
                    prediction_times, window_us)
from .config import Config
from .evaluate import ade_fde, plot_prediction
from .lanes import LaneGraph, load_map_features, mode_manoeuvres
from .rerank import load_prediction
from .run_all import find_db
from .scene_paths import (changed_markdown, changed_moments, load_map_features, plot_changed_grid, plot_path_overview,
                          recorded_path, scene_frame)
from .text import load_annotations

MOMENT_COLUMNS = ['t_rel_s', 'dt_ms', 'pred_file', 'scenario_id', 'log_name', 'time_index', 'frame', 't_pred_us', 'moment']


@dataclass(frozen=True)
class Scene:
    name: str
    clip_id: str
    start_us: int


def read_scenes(path: Path) -> list[Scene]:
    with open(path, newline='', encoding='utf-8') as f:
        return [Scene(r['scene'].strip(), r['clip_id'].strip(), int(r['start_us'])) for r in csv.DictReader(f)]


def select_moments(times: pd.DataFrame, scene: Scene, horizon_s: float, stride_s: float, tol_s: float) -> pd.DataFrame:
    """One ego prediction per stride from the scene start: on the scene's log, nearest to
    start + k * stride (k = 0 .. horizon / stride) and within tol_s of it. Moments with no prediction
    that close are left out, so t_rel_s says which ones were found."""
    sub = times[times.log_name == scene.clip_id]
    rows = []
    if sub.empty:
        return pd.DataFrame(columns=MOMENT_COLUMNS)
    for k in range(int(round(horizon_s / stride_s)) + 1):
        t = scene.start_us + int(round(k * stride_s * 1e6))
        d = (sub.t_pred_us - t).abs()
        i = d.idxmin()
        if d[i] <= tol_s * 1e6:
            rows.append({'t_rel_s': k * stride_s, 'dt_ms': float(sub.t_pred_us[i] - t) / 1e3,
                         **{c: sub.loc[i, c] for c in MOMENT_COLUMNS[2:]}})
    return pd.DataFrame(rows, columns=MOMENT_COLUMNS)


def _mtr_only_row(pred: dict, cfg: Config, graph: LaneGraph | None = None) -> dict:
    """A reranked.csv-like row for a prediction no instruction covers: fused scores equal MTR's."""
    old = np.asarray(pred['pred_scores'], dtype=np.float64)
    fut = np.asarray(pred['gt_trajs'])[cfg.past_len:]
    per = [ade_fde(m, fut[:, :2], fut[:, -1]) for m in np.asarray(pred['pred_trajs'])]
    top = int(old.argmax())
    man, fut_man = mode_manoeuvres(pred, graph, cfg)
    row = {'annotation_id': '', 'instruction': '', 'intent': '', 'gated': False,
           'old_top1': top, 'new_top1': top, 'ade_old': per[top][0], 'ade_new': per[top][0],
           'fde_old': per[top][1], 'fde_new': per[top][1], 'min_ade': min(a for a, _ in per),
           'old_top1_label': man[top].label, 'new_top1_label': man[top].label, 'fut_label': fut_man.label,
           'old_top1_stop': man[top].stop, 'new_top1_stop': man[top].stop, 'fut_stop': fut_man.stop}
    for k in range(len(old)):
        row[f's_old_{k}'], row[f's_new_{k}'] = old[k], old[k] / old.sum()
    return row


def scene_table(moments: pd.DataFrame, reranked: pd.DataFrame, pt_dir: Path, cfg: Config, graph_for=None) -> pd.DataFrame:
    """Long table: one row per (moment, instruction covering it), or one MTR-only row per uncovered moment.
    `graph_for(scenario_id) -> LaneGraph` labels the modes of the MTR-only rows."""
    rows = []
    for m in moments.itertuples(index=False):
        hits = reranked[reranked.pred_file == m.pred_file]
        base = {c: getattr(m, c) for c in MOMENT_COLUMNS}
        if hits.empty:
            graph = graph_for(m.scenario_id) if graph_for is not None else None
            rows.append({**base, **_mtr_only_row(load_prediction(pt_dir / m.pred_file), cfg, graph)})
        else:
            for h in hits.to_dict('records'):
                rows.append({**base, **{k: v for k, v in h.items() if k not in base}})
    return pd.DataFrame(rows)


def _lab(row: dict, which: str) -> str:
    """Lane label of a top-1 or of the recorded future with a '+stop' suffix, '-' when unlabelled."""
    v = row.get(f'{which}_label')
    v = '-' if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    return v + ('+stop' if row.get(f'{which}_stop') in (True, 'True') else '')


def _scores(row: dict, prefix: str) -> list[float]:
    k = 0
    out = []
    while f'{prefix}{k}' in row and not pd.isna(row[f'{prefix}{k}']):
        out.append(float(row[f'{prefix}{k}']))
        k += 1
    return out


def _on_grid(t_s: float, every_s: float) -> bool:
    return abs(t_s / every_s - round(t_s / every_s)) < 1e-6


def write_scene(scene: Scene, table: pd.DataFrame, active: pd.DataFrame, frames: LidarFrames, summary: dict,
                out_dir: Path, pt_dir: Path, scenarionet_dir: Path, cfg: Config, plot_every_s: float) -> Path:
    """<scene>.md, <scene>.csv, <scene>_changed.csv and the figures. `summary` is load_summary() of the
    ScenarioNet database (scenario id -> metadata); `frames` are the lidar frames of the scene's log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f'{scene.name}.csv', index=False)
    frames_start_us = log_start_us(frames)
    lines = [f'# {scene.name}', '',
             f'- clip: `{scene.clip_id}`',
             f'- start timestamp: {scene.start_us} us = {(scene.start_us - frames_start_us) / 1e6:.2f} s after the '
             f'first lidar frame of the log (doPlan frame {(scene.start_us - frames_start_us) / 1e6 / cfg.doplan_frame_s:.1f})',
             f'- moments found: {table.t_rel_s.nunique()} (every {cfg.pred_stride_s} s); max |dt| to the nominal moment '
             f'{table.dt_ms.abs().max() if len(table) else float("nan"):.0f} ms', '',
             '## Instructions whose window covers the scene start', '']
    if active.empty:
        lines.append('(none: MTR scores are reported unchanged)')
    else:
        lines += ['| annotation | window (s after log start) | instruction | rule intent |', '|---|---|---|---|']
        lines += [f'| {a.annotation_id} | {a.window_start_s:.1f} to {a.window_end_s:.1f} | {a.instruction} | {a.rule_intent} |'
                  for a in active.itertuples(index=False)]
    changed = changed_moments(table, pt_dir, cfg)
    changed.to_csv(out_dir / f'{scene.name}_changed.csv', index=False)
    origin, shifts, centre_m = scene_frame(table, summary, frames)
    maps = {sid: load_map_features(scenarionet_dir, sid) for sid in shifts}
    record = recorded_path(frames, centre_m, origin,
                           int(np.searchsorted(frames.ts, table.t_pred_us.min() - 2_000_000)),
                           int(np.searchsorted(frames.ts, table.t_pred_us.max() + 6_000_000, side='right')))
    # the overview draws every changed row; with nothing changed, the original top-1 at the plotted moments
    draw = changed if not changed.empty else table[[_on_grid(t, plot_every_s) for t in table.t_rel_s]].drop_duplicates('pred_file')
    overview = plot_path_overview(scene.name, draw[['t_rel_s', 'pred_file', 'scenario_id', 'old_top1', 'new_top1']], pt_dir,
                                  shifts, maps, record, out_dir / f'{scene.name}_paths.png', cfg)
    grid = plot_changed_grid(scene.name, changed, pt_dir, shifts, maps, out_dir / f'{scene.name}_changed.png', cfg)
    lines += ['', '## Original vs re-ranked top-1 path', ''] + changed_markdown(changed, len(table), int(table.t_rel_s.nunique()))
    lines += ['', f'![{overview.stem}]({overview.name})'] + ([f'![{grid.stem}]({grid.name})'] if grid else [])
    lines += ['', '## Moments', '',
              '| t (s) | annotation | intent | gate | MTR scores | fused scores | top-1 MTR -> fused | '
              'labels MTR top-1 / fused / recorded | ADE top-1 MTR / fused (m) | minADE |',
              '|---|---|---|---|---|---|---|---|---|---|']
    plots = []
    for row in table.to_dict('records'):
        old, new = _scores(row, 's_old_'), _scores(row, 's_new_')
        old_norm = [v / sum(old) for v in old]
        lines.append(f'| {row["t_rel_s"]:.1f} | {row["annotation_id"] or "-"} | {row["intent"] or "-"} | '
                     f'{"open" if row["gated"] else "closed"} | {" ".join(f"{v:.2f}" for v in old_norm)} | '
                     f'{" ".join(f"{v:.2f}" for v in new)} | {int(row["old_top1"])} -> {int(row["new_top1"])} | '
                     f'{_lab(row, "old_top1")} / {_lab(row, "new_top1")} / {_lab(row, "fut")} | '
                     f'{row["ade_old"]:.2f} / {row["ade_new"]:.2f} | {row["min_ade"]:.2f} |')
        if _on_grid(row['t_rel_s'], plot_every_s):
            tag = row['annotation_id'] or 'mtr_only'
            title = (f'{scene.name}  t = +{row["t_rel_s"]:.1f} s  "{(row["instruction"] or "no instruction active")[:80]}"\n'
                     f'intent {row["intent"] or "-"}; gate {"open" if row["gated"] else "closed"}; {row["pred_file"]}')
            path = out_dir / f'{scene.name}_t{row["t_rel_s"]:05.1f}s_{tag}.png'
            plot_prediction(load_prediction(pt_dir / row['pred_file']), path, title, old_norm, new,
                            int(row['old_top1']), int(row['new_top1']), cfg)
            plots.append(path)
    if plots:
        lines += ['', '## Plots', ''] + [f'![{p.stem}]({p.name})' for p in plots]
    md = out_dir / f'{scene.name}.md'
    md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return md


def active_instructions(instructions, scene: Scene, t0_us: int, cfg: Config) -> pd.DataFrame:
    rows = []
    for ins in instructions:
        if ins.clip_id != scene.clip_id:
            continue
        a, b = window_us(t0_us, ins.start_frame, ins.end_frame, cfg.doplan_frame_s)
        if a <= scene.start_us <= b:
            rows.append({'annotation_id': ins.annotation_id, 'window_start_s': (a - t0_us) / 1e6,
                         'window_end_s': (b - t0_us) / 1e6, 'instruction': ins.text, 'rule_intent': ins.rule_intent})
    return pd.DataFrame(rows, columns=['annotation_id', 'window_start_s', 'window_end_s', 'instruction', 'rule_intent'])


def main(argv=None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenarionet', type=Path, required=True)
    p.add_argument('--nuplan', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True, help='doPlan release_annotationsV1.csv')
    p.add_argument('--run', type=Path, required=True, help='run_all.py output dir holding reranked.csv')
    p.add_argument('--scenes', type=Path, required=True, help='CSV with columns scene, clip_id, start_us')
    p.add_argument('--out', type=Path, default=None, help='default: <run>/scenes')
    p.add_argument('--horizon-s', type=float, default=10.0)
    p.add_argument('--tol-s', type=float, default=0.3)
    p.add_argument('--plot-every-s', type=float, default=2.0)
    a = p.parse_args(argv)
    cfg = Config()
    out = a.out or a.run / 'scenes'
    scenes = read_scenes(a.scenes)
    summary = load_summary(a.scenarionet)
    pt_dir = a.scenarionet / 'output_MTR'
    frames_by_log = {log: load_lidar_frames(find_db(a.nuplan, log))
                     for log in sorted({m['log_name'] for m in summary.values()})}
    times = prediction_times(list_ego_predictions(pt_dir), summary, frames_by_log, cfg)
    reranked = pd.read_csv(a.run / 'reranked.csv')
    instructions = load_annotations(a.csv)
    graphs: dict[str, LaneGraph] = {}

    def graph_for(sid: str) -> LaneGraph:
        if sid not in graphs:
            if len(graphs) >= 2:
                graphs.pop(next(iter(graphs)))
            graphs[sid] = LaneGraph.from_map_features(load_map_features(a.scenarionet, sid))
        return graphs[sid]

    written = []
    for sc in scenes:
        if sc.clip_id not in frames_by_log:
            raise SystemExit(f'{sc.name}: log {sc.clip_id} is not in {a.scenarionet}')
        t0 = log_start_us(frames_by_log[sc.clip_id])
        moments = select_moments(times, sc, a.horizon_s, cfg.pred_stride_s, a.tol_s)
        if moments.empty:
            print(f'{sc.name}: no ego prediction within {a.tol_s} s of any moment from {sc.start_us}; '
                  f'log spans {t0} to {int(frames_by_log[sc.clip_id].ts[-1])}')
            continue
        table = scene_table(moments, reranked, pt_dir, cfg, graph_for)
        md = write_scene(sc, table, active_instructions(instructions, sc, t0, cfg), frames_by_log[sc.clip_id], summary,
                         out / sc.name, pt_dir, a.scenarionet, cfg, a.plot_every_s)
        written.append(md)
        print(md.read_text(encoding='utf-8'))
    return written


if __name__ == '__main__':
    main()
