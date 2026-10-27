"""Original-versus-changed path views of one scene (scene_report.py writes them).

For every moment where re-ranking moved the ego top-1 to another mode, the MTR top-1 path (original) and
the re-ranked top-1 path (changed) are drawn from the same point on the recorded drive, over the lane
geometry of the ScenarioNet scenario. The table next to them gives the (lat, dpsi, path) features of
both modes and of the recorded future, so that a change can be read as "lane change suppressed" or
"slowest mode picked" rather than as a bare ADE difference.

Scene frame: nuPlan metres, origin at the ego centre at the scene's first moment, axes unrotated.
A scenario's own frame is nuPlan translated so that its ego centre at step 0 is (0, 0) (align.py), so
everything of scenario s is shifted by ego_centre(start frame of s) - origin.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .align import LidarFrames, ego_centre_xy, rear_axle_to_centre_m, scenario_start_frame
from .config import Config
from .geometry import future_features, mode_features
from .lanes import load_map_features  # noqa: F401  (re-exported for scene_report and tests)
from .rerank import load_prediction

CHANGED_COLUMNS = ['t_rel_s', 'pred_file', 'scenario_id', 'annotation_id', 'instruction', 'intent', 'old_top1', 'new_top1',
                   'old_lat', 'old_dpsi', 'old_path', 'new_lat', 'new_dpsi', 'new_path',
                   'rec_lat', 'rec_dpsi', 'rec_path', 'rec_steps', 'ade_old', 'ade_new', 'fde_old', 'fde_new', 'closer',
                   'old_top1_label', 'new_top1_label', 'fut_label']
# ScenarioNet map feature type prefix -> line style; features of other types are not drawn.
MAP_STYLE = {
    'LANE_SURFACE': {'color': '0.86', 'lw': 0.6, 'ls': '-'},
    'ROAD_LINE': {'color': '0.65', 'lw': 0.8, 'ls': '--'},
    'ROAD_EDGE': {'color': '0.45', 'lw': 1.0, 'ls': '-'},
    'CROSSWALK': {'color': '0.8', 'lw': 0.6, 'ls': ':'},
}
ORIGINAL, CHANGED, RECORD = 'tab:blue', 'tab:red', 'black'


def scene_frame(table: pd.DataFrame, summary: dict, frames: LidarFrames) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """(scene origin in nuPlan metres, scenario id -> shift to add to that scenario's coordinates,
    rear-axle-to-centre offset of the ego in metres)."""
    first = table.sort_values('t_rel_s').iloc[0]
    c = rear_axle_to_centre_m(summary[first.scenario_id])
    origin = ego_centre_xy(frames, int(first.frame), c)
    shifts = {sid: ego_centre_xy(frames, scenario_start_frame(summary[sid], frames), rear_axle_to_centre_m(summary[sid])) - origin
              for sid in table.scenario_id.unique()}
    return origin, shifts, c


def recorded_path(frames: LidarFrames, centre_m: float, origin: np.ndarray, k_from: int, k_to: int) -> np.ndarray:
    """Ego centre of lidar frames k_from .. k_to - 1 in the scene frame."""
    k_from, k_to = max(0, k_from), min(len(frames.ts), k_to)
    yaw = frames.ego_yaw[k_from:k_to]
    return frames.ego_xy[k_from:k_to] + centre_m * np.stack([np.cos(yaw), np.sin(yaw)], 1) - origin


def changed_moments(table: pd.DataFrame, pt_dir: Path, cfg: Config) -> pd.DataFrame:
    """Rows of the scene table whose re-ranked top-1 differs from MTR's, with the (lat, dpsi, path) of
    both top-1 modes and of the recorded future. `closer` says whether the new top-1 has the lower ADE."""
    rows = []
    for r in table[table.old_top1 != table.new_top1].to_dict('records'):
        pred = load_prediction(pt_dir / r['pred_file'])
        f = mode_features(pred['pred_trajs'], pred['gt_trajs'], cfg.past_len)
        lat, dpsi, path, n = future_features(pred['gt_trajs'], cfg.past_len)
        o, nw = int(r['old_top1']), int(r['new_top1'])
        rows.append({**{k: r[k] for k in CHANGED_COLUMNS[:6]}, 'old_top1': o, 'new_top1': nw,
                     'old_lat': f[o, 0], 'old_dpsi': f[o, 1], 'old_path': f[o, 2],
                     'new_lat': f[nw, 0], 'new_dpsi': f[nw, 1], 'new_path': f[nw, 2],
                     'rec_lat': lat, 'rec_dpsi': dpsi, 'rec_path': path, 'rec_steps': n,
                     **{k: float(r[k]) for k in ('ade_old', 'ade_new', 'fde_old', 'fde_new')},
                     'closer': bool(r['ade_new'] < r['ade_old']),
                     **{k: (None if pd.isna(r.get(k)) else r.get(k)) if not isinstance(r.get(k), str) else r.get(k)
                        for k in ('old_top1_label', 'new_top1_label', 'fut_label')}})
    return pd.DataFrame(rows, columns=CHANGED_COLUMNS)


def _curves(pred: dict, shift: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    g = np.asarray(pred['gt_trajs'])
    hist, fut = g[:cfg.past_len], g[cfg.past_len:]
    return {'history': hist[hist[:, -1] > 0, :2] + shift, 'future': fut[fut[:, -1] > 0, :2] + shift,
            'modes': np.asarray(pred['pred_trajs'])[:, :, :2] + shift}


def _draw_map(ax, map_features: dict, shift: np.ndarray, xlim, ylim) -> None:
    from matplotlib.collections import LineCollection
    groups: dict[str, list] = {k: [] for k in MAP_STYLE}
    for feat in map_features.values():
        kind = next((k for k in MAP_STYLE if str(feat.get('type', '')).startswith(k)), None)
        pts = feat.get('polyline') if feat.get('polyline') is not None else feat.get('polygon')
        if kind is None or pts is None or len(pts) < 2:
            continue
        xy = np.asarray(pts, dtype=np.float64)[:, :2] + shift
        if xy[:, 0].max() < xlim[0] or xy[:, 0].min() > xlim[1] or xy[:, 1].max() < ylim[0] or xy[:, 1].min() > ylim[1]:
            continue
        groups[kind].append(np.vstack([xy, xy[:1]]) if kind == 'CROSSWALK' else xy)
    for kind, segs in groups.items():
        if segs:
            ax.add_collection(LineCollection(segs, zorder=0, **MAP_STYLE[kind]))


def _finish(ax, xs, ys, margin_m: float, maps: dict[str, dict], shifts: dict[str, np.ndarray]) -> None:
    xlim = (min(xs) - margin_m, max(xs) + margin_m)
    ylim = (min(ys) - margin_m, max(ys) + margin_m)
    for sid, feats in maps.items():
        _draw_map(ax, feats, shifts[sid], xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=7)


def plot_changed_grid(scene_name: str, changed: pd.DataFrame, pt_dir: Path, shifts: dict[str, np.ndarray],
                      maps: dict[str, dict], out_path: Path, cfg: Config, margin_m: float = 8.0) -> Path | None:
    """One panel per changed (moment, instruction): history, recorded future, original and changed top-1."""
    if changed.empty:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(changed)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.4 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis('off')
    for ax, r in zip(axes.flat, changed.itertuples(index=False)):
        cv = _curves(load_prediction(pt_dir / r.pred_file), shifts[r.scenario_id], cfg)
        old, new = cv['modes'][int(r.old_top1)], cv['modes'][int(r.new_top1)]
        ax.plot(*cv['history'].T, color=RECORD, lw=2.5, zorder=3, label='history')
        if len(cv['future']):
            ax.plot(*cv['future'].T, color=RECORD, ls='--', lw=2, zorder=3, label='recorded future')
        ax.plot(*old.T, color=ORIGINAL, lw=2.5, zorder=4, label=f'MTR top-1 (mode {int(r.old_top1)})')
        ax.plot(*new.T, color=CHANGED, lw=2.5, zorder=5, label=f're-ranked top-1 (mode {int(r.new_top1)})')
        ax.scatter(*old[-1], marker='x', s=70, color=ORIGINAL, zorder=6)
        ax.scatter(*new[-1], marker='*', s=120, color=CHANGED, zorder=6)
        ax.scatter(*cv['history'][-1], marker='o', s=30, color=RECORD, zorder=6)
        pts = np.vstack([cv['history'], cv['future'], old, new])
        _finish(ax, pts[:, 0], pts[:, 1], margin_m, maps, shifts)
        ax.set_title(f'+{r.t_rel_s:.1f} s  {r.annotation_id}  {r.intent}\n"{r.instruction[:48]}"\n'
                     f'mode {int(r.old_top1)} -> {int(r.new_top1)};  ADE {r.ade_old:.2f} -> {r.ade_new:.2f} m', fontsize=8)
        ax.legend(fontsize=6, loc='best')
    fig.suptitle(f'{scene_name}: original (blue) vs re-ranked (red) top-1 at every moment where it changed', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / nrows))     # keep the suptitle clear of the first row's titles
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_path_overview(scene_name: str, draw: pd.DataFrame, pt_dir: Path, shifts: dict[str, np.ndarray], maps: dict[str, dict],
                       record: np.ndarray, out_path: Path, cfg: Config, margin_m: float = 10.0) -> Path:
    """The recorded drive with, at each row of `draw` (t_rel_s, pred_file, scenario_id, old_top1, new_top1),
    the original top-1 and, where it differs, the re-ranked top-1, all in the scene frame."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot(*record.T, color=RECORD, lw=2.5, zorder=3, label='recorded drive')
    xs, ys = list(record[:, 0]), list(record[:, 1])
    drawn: set[tuple] = set()
    labels_done = {'orig': False, 'new': False}
    for r in draw.sort_values('t_rel_s').itertuples(index=False):
        key = (r.pred_file, int(r.old_top1), int(r.new_top1))
        if key in drawn:
            continue
        above = len(drawn) % 2 == 0          # alternate the time labels above/below the path: moments are 0.5 s apart
        drawn.add(key)
        cv = _curves(load_prediction(pt_dir / r.pred_file), shifts[r.scenario_id], cfg)
        old = cv['modes'][int(r.old_top1)]
        ax.plot(*old.T, color=ORIGINAL, lw=1.6, alpha=0.85, zorder=4, label=None if labels_done['orig'] else 'MTR top-1 (original)')
        labels_done['orig'] = True
        xs += list(old[:, 0])
        ys += list(old[:, 1])
        if int(r.new_top1) != int(r.old_top1):
            new = cv['modes'][int(r.new_top1)]
            ax.plot(*new.T, color=CHANGED, lw=1.6, alpha=0.9, zorder=5, label=None if labels_done['new'] else 're-ranked top-1 (changed)')
            labels_done['new'] = True
            xs += list(new[:, 0])
            ys += list(new[:, 1])
        ax.annotate(f'+{r.t_rel_s:.1f} s', cv['history'][-1], fontsize=7, xytext=(2, 4 if above else -10),
                    textcoords='offset points', zorder=7)
        ax.scatter(*cv['history'][-1], s=14, color=RECORD, zorder=6)
    _finish(ax, xs, ys, margin_m, maps, shifts)
    n_changed = int((draw.old_top1 != draw.new_top1).sum())
    ax.set_title(f'{scene_name}: recorded drive and the ego top-1 path before/after re-ranking '
                 f'({n_changed} changed of {len(draw)} rows drawn)', fontsize=9)
    ax.set_xlabel('east (m) from the ego at the scene start')
    ax.set_ylabel('north (m)')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def changed_markdown(changed: pd.DataFrame, n_rows: int, n_moments: int) -> list[str]:
    """Markdown lines of the "original vs re-ranked top-1" section (without the figure links)."""
    if changed.empty:
        return [f'Re-ranking left the top-1 unchanged on all {n_rows} rows ({n_moments} moments); the original path is the path.']
    closer = int(changed.closer.sum())
    lines = [f'Re-ranking changed the top-1 on {len(changed)} of {n_rows} (moment, instruction) rows '
             f'({changed.t_rel_s.nunique()} of {n_moments} moments). The changed top-1 is closer to the recorded path '
             f'(lower ADE) on {closer} rows and farther on {len(changed) - closer}.', '',
             'Features are in the ego frame at the prediction time: lat = endpoint lateral offset (m, + is left), '
             'dpsi = final heading change (deg, + is left), path = length (m) over 6 s.', '',
             '| t (s) | annotation | intent | mode orig -> new | original top-1 (lat, dpsi, path) | re-ranked top-1 (lat, dpsi, path) '
             '| recorded future (lat, dpsi, path) | lane labels orig -> new (recorded) | ADE orig -> new (m) | FDE orig -> new (m) | closer? |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in changed.itertuples(index=False):
        lines.append(f'| {r.t_rel_s:.1f} | {r.annotation_id} | {r.intent} | {int(r.old_top1)} -> {int(r.new_top1)} '
                     f'| {r.old_lat:+.1f}, {r.old_dpsi:+.0f}, {r.old_path:.0f} | {r.new_lat:+.1f}, {r.new_dpsi:+.0f}, {r.new_path:.0f} '
                     f'| {r.rec_lat:+.1f}, {r.rec_dpsi:+.0f}, {r.rec_path:.0f} '
                     f'| {r.old_top1_label or "-"} -> {r.new_top1_label or "-"} ({r.fut_label or "-"}) | {r.ade_old:.2f} -> {r.ade_new:.2f} '
                     f'| {r.fde_old:.2f} -> {r.fde_new:.2f} | {"yes" if r.closer else "no"} |')
    return lines
