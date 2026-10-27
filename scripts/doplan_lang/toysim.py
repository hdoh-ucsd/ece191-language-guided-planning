"""Toy simulator (v2 spec 4.6): execute one selected 6 s trajectory with a pure-pursuit tracker on a kinematic
bicycle among agents replayed from the log, and score the outcome: completion of the instructed manoeuvre
(lane graph), off-road steps, collision (oriented boxes), comfort and tracking error.

No re-planning and no reactive agents: the plan is fixed for 6 s and the other agents do what they did in
the log, so a collision after the ego leaves the recorded path is pessimistic. The recorded future is run
through the same tracker as the reference.

Usage:
  PYTHONPATH=scripts python3 -m doplan_lang.toysim --scenarionet scenarionet_data/nuplan_scenes \\
      --reranked outputs/doplan_scenes/reranked.csv --out outputs/doplan_scenes/toysim
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LATERAL_INTENTS, Config
from .evaluate import bootstrap_diff, json_safe
from .lanes import LaneGraph, final_speed, load_scenario, manoeuvre
from .rerank import load_prediction

EXECUTIONS = ('mtr_top1', 'reranked_top1', 'recorded')


@dataclass(frozen=True)
class AgentSet:
    positions: np.ndarray    # [k, T, 2]
    headings: np.ndarray     # [k, T]
    valid: np.ndarray        # [k, T]
    lengths: np.ndarray      # [k]
    widths: np.ndarray       # [k]

    def at(self, step: int):
        """(centres, headings, lengths, widths) of the agents valid at `step` (empty arrays past the end)."""
        if step < 0 or step >= self.positions.shape[1]:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0)
        ok = self.valid[:, step] > 0
        return self.positions[ok, step], self.headings[ok, step], self.lengths[ok], self.widths[ok]


def agents_from_tracks(tracks: dict, sdc_id) -> AgentSet:
    """Every track but the ego, as replayed boxes."""
    pos, head, valid, lengths, widths = [], [], [], [], []
    for tid, tr in tracks.items():
        if str(tid) == str(sdc_id):
            continue
        st = tr['state']
        v = np.asarray(st['valid'], dtype=np.float64).reshape(-1)
        if not v.any():
            continue
        pos.append(np.asarray(st['position'], dtype=np.float64)[:, :2])
        head.append(np.asarray(st['heading'], dtype=np.float64).reshape(-1))
        valid.append(v)
        first = int(np.argmax(v > 0))
        lengths.append(float(np.asarray(st['length']).reshape(-1)[first]))
        widths.append(float(np.asarray(st['width']).reshape(-1)[first]))
    if not pos:
        return AgentSet(np.zeros((0, 1, 2)), np.zeros((0, 1)), np.zeros((0, 1)), np.zeros(0), np.zeros(0))
    return AgentSet(np.stack(pos), np.stack(head), np.stack(valid), np.array(lengths), np.array(widths))


def plan_speeds(plan_xy: np.ndarray, start_xy, dt: float) -> np.ndarray:
    pts = np.vstack([np.asarray(start_xy, dtype=np.float64)[None], plan_xy])
    return np.maximum(np.linalg.norm(np.diff(pts, axis=0), axis=1) / dt, 0.0)


def simulate(plan_xy, start_xy, start_heading: float, start_speed: float, cfg: Config) -> dict:
    """Pure pursuit on `plan_xy` [n, 2] with a kinematic bicycle at the vehicle centre (spec 4.6).
    Returns xy [n, 2], heading [n], speed [n], kappa [n], accel [n] after each of the n steps. Plan points
    behind the ego are never targeted, so a reversing tail is not tracked; with no point ahead the ego brakes."""
    plan = np.asarray(plan_xy, dtype=np.float64)
    n = len(plan)
    speeds = plan_speeds(plan, start_xy, cfg.sim_dt)
    x, y = (float(v) for v in start_xy)
    psi, v = float(start_heading), max(0.0, float(start_speed))
    out = {k: np.zeros(n) for k in ('heading', 'speed', 'kappa', 'accel')}
    xy = np.zeros((n, 2))
    for i in range(n):
        fwd = np.array([math.cos(psi), math.sin(psi)])
        rel = plan - [x, y]
        ahead = rel @ fwd > 0.0
        dist = np.linalg.norm(rel, axis=1)
        look = float(np.clip(cfg.sim_lookahead_gain * v, cfg.sim_lookahead_min_m, cfg.sim_lookahead_max_m))
        cand = np.where(ahead & (dist >= look))[0]
        if len(cand):
            j = int(cand[0])
        elif ahead.any():
            j = int(np.where(ahead)[0][-1])
        else:
            j = None
        if j is None:
            kappa, v_target = 0.0, 0.0
        else:
            alpha = math.atan2(rel[j, 1], rel[j, 0]) - psi
            kappa = float(np.clip(2.0 * math.sin(alpha) / max(dist[j], 0.5), -cfg.sim_max_curvature, cfg.sim_max_curvature))
            v_target = float(speeds[min(i, n - 1)])
        a = float(np.clip(cfg.sim_speed_gain * (v_target - v) / cfg.sim_dt, -cfg.sim_max_accel, cfg.sim_max_accel))
        v = max(0.0, v + a * cfg.sim_dt)
        psi += v * kappa * cfg.sim_dt
        x += v * math.cos(psi) * cfg.sim_dt
        y += v * math.sin(psi) * cfg.sim_dt
        xy[i] = (x, y)
        out['heading'][i], out['speed'][i], out['kappa'][i], out['accel'][i] = psi, v, kappa, a
    return {'xy': xy, **out}


def _corners(centre, heading, length, width) -> np.ndarray:
    """[m, 4, 2] corners of oriented boxes (all arguments broadcast over m)."""
    c = np.atleast_2d(np.asarray(centre, dtype=np.float64))
    h = np.atleast_1d(np.asarray(heading, dtype=np.float64))
    length, width = np.atleast_1d(np.asarray(length, dtype=np.float64)), np.atleast_1d(np.asarray(width, dtype=np.float64))
    fwd = np.stack([np.cos(h), np.sin(h)], -1)
    lft = np.stack([-np.sin(h), np.cos(h)], -1)
    hl, hw = (length / 2)[:, None], (width / 2)[:, None]
    return np.stack([c + hl * fwd + hw * lft, c + hl * fwd - hw * lft, c - hl * fwd - hw * lft, c - hl * fwd + hw * lft], 1)


def boxes_overlap(centre1, heading1: float, length1: float, width1: float, centres2, headings2, lengths2, widths2) -> np.ndarray:
    """Separating-axis test of one oriented box against m boxes -> bool [m]."""
    m = len(np.atleast_2d(centres2))
    if m == 0:
        return np.zeros(0, dtype=bool)
    c1 = np.repeat(_corners(centre1, heading1, length1, width1), m, axis=0)       # [m, 4, 2]
    c2 = _corners(centres2, headings2, lengths2, widths2)
    h1, h2 = np.full(m, float(heading1)), np.atleast_1d(np.asarray(headings2, dtype=np.float64))
    axes = np.stack([np.stack([np.cos(h), np.sin(h)], -1) for h in (h1, h1 + np.pi / 2, h2, h2 + np.pi / 2)], 1)   # [m, 4, 2]
    p1 = np.einsum('mcd,mad->mac', c1, axes)      # [m, axes, corners]
    p2 = np.einsum('mcd,mad->mac', c2, axes)
    sep = (p1.max(-1) < p2.min(-1)) | (p2.max(-1) < p1.min(-1))
    return ~sep.any(axis=1)


def completion_of(label: str | None, final_speed_mps: float, intent: str, cfg: Config) -> float:
    """1.0 / 0.0 whether an executed path with lane label `label` performs `intent`; NaN when undefined
    (intent "other", or a lateral/straight intent with no label)."""
    if intent in LATERAL_INTENTS or intent == 'straight':
        return float(label == intent) if label is not None else math.nan
    if intent == 'stop':
        return float(final_speed_mps < cfg.stop_speed_mps)
    return math.nan


def score_execution(ex: dict, plan_xy, agents: AgentSet, t0: int, graph: LaneGraph, now_xy, now_heading: float,
                    hist_xy, cfg: Config, ego_length: float | None = None, ego_width: float | None = None) -> dict:
    """Intent-independent outcome metrics of one executed trajectory (spec 4.6); `t0` is the scenario step of
    "now". Completion is added per pair with completion_of()."""
    L, W = ego_length or cfg.sim_ego_length, ego_width or cfg.sim_ego_width
    xy, head = ex['xy'], ex['heading']
    collision, clearance = False, math.inf
    for i in range(len(xy)):
        c, h, ln, wd = agents.at(t0 + 1 + i)
        if len(c) == 0:
            continue
        if not collision and boxes_overlap(xy[i], head[i], L, W, c, h, ln, wd).any():
            collision = True
        clearance = min(clearance, float((np.linalg.norm(c - xy[i], axis=1) - np.hypot(ln, wd) / 2 - np.hypot(L, W) / 2).min()))
    offlane = float(np.mean([graph.off_lane(p, cfg.offlane_m) for p in xy])) if len(graph) else math.nan
    man = manoeuvre(graph, now_xy, now_heading, hist_xy, xy, cfg)
    plan = np.asarray(plan_xy, dtype=np.float64)
    n = min(len(plan), len(xy))
    return {'collision': bool(collision), 'min_clearance_m': clearance if math.isfinite(clearance) else math.nan,
            'offlane_fraction': offlane, 'label': man.label, 'stop': bool(man.stop),
            'final_speed_mps': final_speed(xy, cfg.sim_dt),
            'max_lat_accel': float(np.max(np.abs(ex['speed'] ** 2 * ex['kappa']))),
            'max_long_accel': float(np.max(np.abs(ex['accel']))),
            'tracking_rms_m': float(np.sqrt(np.mean(np.sum((xy[:n] - plan[:n]) ** 2, axis=1))))}


def _ego_start(pred: dict, cfg: Config):
    """(now row, history xy, start speed) of a prediction file."""
    g = np.asarray(pred['gt_trajs'])
    now = g[cfg.past_len - 1]
    hist = g[:cfg.past_len]
    hist = hist[hist[:, -1] > 0, :2]
    v0 = float(np.linalg.norm(hist[-1] - hist[-2]) / cfg.sim_dt) if len(hist) >= 2 else 0.0
    return now, hist, v0


def run(reranked: pd.DataFrame, scenarionet_dir: Path, pt_dir: Path, cfg: Config, max_pairs: int | None = None,
        seed: int = 0) -> pd.DataFrame:
    """One row per (pair, execution) for the executions in EXECUTIONS; executions are shared between pairs."""
    df = reranked if max_pairs is None or len(reranked) <= max_pairs else reranked.sample(max_pairs, random_state=seed)
    rows = []
    skipped = 0
    for sid, grp in df.groupby('scenario_id', sort=True):
        sc = load_scenario(scenarionet_dir, str(sid))
        if not sc or 'tracks' not in sc:
            skipped += len(grp)
            continue
        graph = LaneGraph.from_map_features(sc.get('map_features', {}))
        agents = agents_from_tracks(sc['tracks'], sc['metadata'].get('sdc_id'))
        vp = sc['metadata'].get('ego_vehicle_parameters') or {}
        L, W = vp.get('length'), vp.get('width')
        done: dict[tuple, dict | None] = {}
        for r in grp.itertuples(index=False):
            pred = load_prediction(pt_dir / r.pred_file)
            now, hist, v0 = _ego_start(pred, cfg)
            fut = np.asarray(pred['gt_trajs'])[cfg.past_len:]
            plans = {'mtr_top1': ('mode', int(r.old_top1)), 'reranked_top1': ('mode', int(r.new_top1))}
            if int(r.fut_valid) >= len(fut):
                plans['recorded'] = ('recorded', -1)
            for which, key in plans.items():
                k = (r.pred_file,) + key
                if k not in done:
                    plan = np.asarray(pred['pred_trajs'])[key[1], :, :2] if key[0] == 'mode' else fut[:, :2]
                    if not np.all(np.isfinite(plan)):
                        done[k] = None
                    else:
                        ex = simulate(plan, now[:2], float(now[6]), v0, cfg)
                        done[k] = {**score_execution(ex, plan, agents, int(r.time_index), graph, now[:2], float(now[6]),
                                                     hist, cfg, L, W), 'plan_mode': key[1]}
                if done[k] is None:
                    skipped += 1
                    continue
                rows.append({'pred_file': r.pred_file, 'scenario_id': sid, 'log_name': r.log_name, 'moment': r.moment,
                             'annotation_id': r.annotation_id, 'rule_intent': r.rule_intent, 'gated': bool(r.gated),
                             'time_index': int(r.time_index), 'execution': which, **done[k],
                             'completion': completion_of(done[k]['label'], done[k]['final_speed_mps'], r.rule_intent, cfg)})
    if skipped:
        print(f'toysim: skipped {skipped} executions (scenario pickle or finite plan missing)')
    return pd.DataFrame(rows)


def _paired(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df.execution == 'mtr_top1'].set_index(['pred_file', 'annotation_id'])
    b = df[df.execution == 'reranked_top1'].set_index(['pred_file', 'annotation_id'])
    return a.join(b, lsuffix='_a', rsuffix='_b', how='inner')


def summarize_toysim(df: pd.DataFrame, cfg: Config) -> dict:
    out = {'n_rows': int(len(df)), 'n_pairs': int(df.groupby(['pred_file', 'annotation_id']).ngroups) if len(df) else 0}
    for subset_name, sub in (('gated', df[df.gated] if len(df) else df), ('all', df)):
        s = {}
        for ex in EXECUTIONS:
            e = sub[sub.execution == ex] if len(sub) else sub
            comp = e.completion.dropna() if len(e) else pd.Series(dtype=float)
            s[ex] = {'n': int(len(e)), 'completion_n': int(len(comp)),
                     'completion': float(comp.mean()) if len(comp) else math.nan,
                     'collision': float(e.collision.mean()) if len(e) else math.nan,
                     'offlane': float((e.offlane_fraction > 0).mean()) if len(e) else math.nan,
                     'tracking_rms_m': float(e.tracking_rms_m.mean()) if len(e) else math.nan,
                     'lat_accel_p95': float(e.max_lat_accel.quantile(0.95)) if len(e) else math.nan}
        diffs = {}
        both = _paired(sub) if len(sub) else pd.DataFrame()
        for key, col in (('completion', 'completion'), ('collision', 'collision'), ('offlane', 'offlane_fraction')):
            if len(both):
                x, y = both[f'{col}_a'], both[f'{col}_b']
                if key == 'offlane':
                    x, y = (x > 0), (y > 0)
                ok = x.notna() & y.notna()
                diffs[key] = bootstrap_diff(x[ok].astype(float), y[ok].astype(float), cfg.bootstrap_n, cfg.seed,
                                            groups=both.moment_a[ok])
            else:
                diffs[key] = bootstrap_diff(pd.Series(dtype=float), pd.Series(dtype=float), cfg.bootstrap_n, cfg.seed)
        s['diff_reranked_minus_mtr'] = diffs
        out[subset_name] = s
    return out


def _f(x) -> str:
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.3f}'


def write_toysim_markdown(m: dict, path: Path, cfg: Config) -> None:
    lines = ['# Toy simulator: outcome of the selected trajectories', '',
             f'Pure pursuit on a kinematic bicycle (dt {cfg.sim_dt} s, |a| <= {cfg.sim_max_accel} m/s^2, |kappa| <= '
             f'{cfg.sim_max_curvature} 1/m), agents replayed from the log (not reactive: collisions after the ego leaves '
             f'the recorded path are pessimistic). Completion: the executed path performs the instructed manoeuvre by the '
             f'lane graph (stop: final speed < {cfg.stop_speed_mps} m/s); undefined for "other". Off-lane: any step '
             f'outside every lane polygon and more than {cfg.offlane_m} m from every lane centreline (riding a lane edge is '
             'not off-lane). The recorded execution is the reference: the recorded future through the same tracker.', '',
             '| subset | execution | n | completion (n) | collision | off-lane | tracking RMS (m) | lat accel p95 (m/s^2) |',
             '|---|---|---|---|---|---|---|---|']
    for subset in ('gated', 'all'):
        for ex in EXECUTIONS:
            s = m[subset][ex]
            lines.append(f"| {subset} | {ex} | {s['n']} | {_f(s['completion'])} ({s['completion_n']}) | {_f(s['collision'])} | "
                         f"{_f(s['offlane'])} | {_f(s['tracking_rms_m'])} | {_f(s['lat_accel_p95'])} |")
    lines += ['', '| subset | metric | reranked - MTR [95% CI] |', '|---|---|---|']
    for subset in ('gated', 'all'):
        for key, d in m[subset]['diff_reranked_minus_mtr'].items():
            lines.append(f"| {subset} | {key} | {_f(d['mean'])} [{_f(d['lo'])}, {_f(d['hi'])}] |")
    lines += ['', f"{m['n_rows']} executions over {m['n_pairs']} pairs; CIs resample moments."]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_execution_examples(df: pd.DataFrame, scenarionet_dir: Path, pt_dir: Path, out_dir: Path, cfg: Config, n: int) -> list[Path]:
    """The n pairs with the largest change in completion or collision between the two executions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .scene_paths import _draw_map
    if df.empty:
        return []
    both = _paired(df)
    if both.empty:
        return []
    both = both.assign(change=(both.completion_b.fillna(0) - both.completion_a.fillna(0)).abs()
                       + (both.collision_b.astype(float) - both.collision_a.astype(float)).abs()
                       + (both.plan_mode_a != both.plan_mode_b).astype(float) * 0.1)
    chosen = both.sort_values('change', ascending=False).head(n)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, (idx, r) in enumerate(chosen.iterrows()):
        pred_file, ann = idx
        sc = load_scenario(scenarionet_dir, str(r.scenario_id_a))
        pred = load_prediction(pt_dir / pred_file)
        now, hist, v0 = _ego_start(pred, cfg)
        fig, ax = plt.subplots(figsize=(7, 6))
        xs, ys = list(hist[:, 0]), list(hist[:, 1])
        ax.plot(*hist.T, 'k-', lw=2.5, zorder=4, label='history')
        fut = np.asarray(pred['gt_trajs'])[cfg.past_len:]
        fut = fut[fut[:, -1] > 0, :2]
        if len(fut):
            ax.plot(*fut.T, color='0.2', ls='--', lw=2, zorder=4, label='recorded future')
        plan_len = 0
        for which, mode, color in (('MTR top-1', int(r.plan_mode_a), 'tab:blue'), ('re-ranked top-1', int(r.plan_mode_b), 'tab:red')):
            plan = np.asarray(pred['pred_trajs'])[mode, :, :2]
            plan_len = len(plan)
            ex = simulate(plan, now[:2], float(now[6]), v0, cfg)
            ax.plot(*plan.T, color=color, ls=':', lw=1, zorder=5)
            ax.plot(*ex['xy'].T, color=color, lw=2.5, zorder=5, label=f'{which} executed (mode {mode})')
            xs += list(ex['xy'][:, 0])
            ys += list(ex['xy'][:, 1])
        if sc and 'tracks' in sc:
            agents = agents_from_tracks(sc['tracks'], sc['metadata'].get('sdc_id'))
            c, h, ln, wd = agents.at(int(r.time_index_a) + plan_len)
            for corners in _corners(c, h, ln, wd):
                ax.fill(*np.vstack([corners, corners[:1]]).T, color='0.55', alpha=0.6, lw=0, zorder=3)
        xlim, ylim = (min(xs) - 10, max(xs) + 10), (min(ys) - 10, max(ys) + 10)
        if sc:
            _draw_map(ax, sc.get('map_features', {}), np.zeros(2), xlim, ylim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{ann} "{r.rule_intent_a}" {pred_file}\ncompletion {_f(r.completion_a)} -> {_f(r.completion_b)}; '
                     f'collision {r.collision_a} -> {r.collision_b}; off-lane {_f(r.offlane_fraction_a)} -> {_f(r.offlane_fraction_b)}; '
                     f'agents (grey) at the final step', fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / f'toysim_{k:02d}_{ann}_{Path(pred_file).stem}.png'
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenarionet', type=Path, required=True)
    p.add_argument('--reranked', type=Path, required=True, help='reranked.csv of a run_all run')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-pairs', type=int, default=None, help='random sample of pairs (default: all)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    cfg = Config()
    a.out.mkdir(parents=True, exist_ok=True)
    df = run(pd.read_csv(a.reranked), a.scenarionet, a.scenarionet / 'output_MTR', cfg, a.max_pairs, a.seed)
    df.to_csv(a.out / 'toysim.csv', index=False)
    m = summarize_toysim(df, cfg)
    (a.out / 'toysim.json').write_text(json.dumps(json_safe(m), indent=2, allow_nan=False), encoding='utf-8')
    write_toysim_markdown(m, a.out / 'toysim.md', cfg)
    plot_execution_examples(df, a.scenarionet, a.scenarionet / 'output_MTR', a.out / 'plots', cfg, cfg.n_example_plots)
    print((a.out / 'toysim.md').read_text(encoding='utf-8'))
    return m


if __name__ == '__main__':
    main()
