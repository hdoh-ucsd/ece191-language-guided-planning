"""Metrics and plots for re-ranked (prediction, annotation) pairs (spec 5).

Unit = one pair. Headline numbers are over gated pairs; "all" is reported for context and
"gated_disagree" (classifier intent differs from the rule label) is where word2vec can differ
from the keyword rules. Compliance follows doPlan section V-B: among pairs with a lateral intent
whose recorded 6 s endpoint lies >= compliance_m toward the instructed side (ego frame), the
fraction whose top-1 mode endpoint also does. The population is defined by the rule label and a
full-horizon recorded future, so every variant is scored on the same pairs. Counterfactual consistency additionally requires
that the left/right-swapped instruction picks a mode whose endpoint lies on the opposite side.
Confidence intervals resample moments (cluster bootstrap), since several pairs share one instant.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LATERAL_INTENTS, Config
from .rerank import load_prediction

FUTURE_STEPS = 60   # MTR horizon in steps; compliance needs the full 6 s endpoint


def ade_fde(mode_xy, future_xy, valid) -> tuple[float, float]:
    """(ADE, FDE) over valid steps; FDE is the error at the LAST VALID step, which is step 60 only for a full future."""
    v = np.asarray(valid).astype(bool)
    if not v.any():
        return math.nan, math.nan
    err = np.linalg.norm(np.asarray(mode_xy, dtype=np.float64)[v] - np.asarray(future_xy, dtype=np.float64)[v], axis=-1)
    return float(err.mean()), float(err[-1])


def add_errors(df: pd.DataFrame, pt_dir: Path, cfg: Config) -> pd.DataFrame:
    """Per pair: ADE/FDE of the MTR top-1 and the fused top-1, and the minimum over modes."""
    cache: dict[str, list[tuple[float, float]]] = {}
    cols = {k: [] for k in ('ade_old', 'fde_old', 'ade_new', 'fde_new', 'min_ade', 'min_fde')}
    for r in df.itertuples(index=False):
        if r.pred_file not in cache:
            p = load_prediction(pt_dir / r.pred_file)
            fut = np.asarray(p['gt_trajs'])[cfg.past_len:]
            cache[r.pred_file] = [ade_fde(m, fut[:, :2], fut[:, -1]) for m in np.asarray(p['pred_trajs'])]
        per = cache[r.pred_file]
        cols['ade_old'].append(per[r.old_top1][0])
        cols['fde_old'].append(per[r.old_top1][1])
        cols['ade_new'].append(per[r.new_top1][0])
        cols['fde_new'].append(per[r.new_top1][1])
        cols['min_ade'].append(min(a for a, _ in per))
        cols['min_fde'].append(min(f for _, f in per))
    out = df.copy()
    for k, v in cols.items():
        out[k] = v
    return out


def side_sign(intent: str) -> int:
    return 1 if intent.endswith('_left') else -1 if intent.endswith('_right') else 0


def compliance_frame(df: pd.DataFrame, cfg: Config, intent_col: str = 'rule_intent') -> pd.DataFrame:
    """Pairs whose instruction (by the keyword rule label, so every variant is scored on the same pairs)
    is lateral, whose future covers the full horizon, and whose recorded endpoint moved >= compliance_m
    toward the instructed side. The caller chooses the subset (gated or not)."""
    sub = df[df[intent_col].isin(LATERAL_INTENTS) & (df.fut_valid >= FUTURE_STEPS)].copy()
    sgn = sub[intent_col].map(side_sign)
    keep = sgn * sub.fut_lat >= cfg.compliance_m
    sub, sgn = sub[keep].copy(), sgn[keep]
    sub['old_ok'] = sgn * sub.old_top1_lat >= cfg.compliance_m
    sub['new_ok'] = sgn * sub.new_top1_lat >= cfg.compliance_m
    sub['cf_ok'] = sub.new_ok & (-sgn * sub.cf_top1_lat >= cfg.compliance_m)
    return sub


def bootstrap_diff(before, after, n: int = 1000, seed: int = 0, groups=None) -> dict:
    """Paired bootstrap of mean(after - before) with a 95% percentile interval; NaN pairs dropped.

    With `groups` (one key per pair, e.g. the moment of driving), whole groups are resampled so that
    several pairs from the same instant do not count as independent evidence.
    """
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    if len(d) == 0:
        return {'mean': math.nan, 'lo': math.nan, 'hi': math.nan, 'n': 0, 'n_groups': 0}
    g = np.arange(len(d)) if groups is None else pd.factorize(np.asarray(groups)[keep])[0]
    n_groups = int(g.max()) + 1
    sums = np.bincount(g, weights=d, minlength=n_groups)
    counts = np.bincount(g, minlength=n_groups).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n, n_groups))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {'mean': float(d.mean()), 'lo': float(np.percentile(means, 2.5)),
            'hi': float(np.percentile(means, 97.5)), 'n': int(len(d)), 'n_groups': n_groups}


def _mean(series) -> float:
    return float(series.mean()) if len(series) else math.nan


def summarize(df: pd.DataFrame, cfg: Config) -> dict:
    out = {'n_pairs': int(len(df)),
           'n_predictions': int(df.pred_file.nunique()) if len(df) else 0,
           'n_moments': int(df.moment.nunique()) if len(df) else 0,
           'n_annotations': int(df.annotation_id.nunique()) if len(df) else 0,
           'gate_open_fraction': _mean(df.gated.astype(float))}
    out['pairs_per_log'] = {str(k): int(v) for k, v in df.groupby('log_name').size().items()} if len(df) else {}
    out['intent_disagrees_with_rules_fraction'] = _mean((df.intent != df.rule_intent).astype(float))
    subsets = (('gated', df[df.gated]), ('all', df),
               ('gated_disagree', df[df.gated & (df.intent != df.rule_intent)]))   # where word2vec can differ
    for name, sub in subsets:
        valid = sub[sub.fut_valid > 0]
        m = {'n_pairs': int(len(sub)), 'n_with_future': int(len(valid))}
        for key in ('ade', 'fde'):
            m[f'{key}_top1_before'] = _mean(valid[f'{key}_old'])
            m[f'{key}_top1_after'] = _mean(valid[f'{key}_new'])
            m[f'min_{key}'] = _mean(valid[f'min_{key}'])
            m[f'{key}_diff_ci'] = bootstrap_diff(valid[f'{key}_old'], valid[f'{key}_new'], cfg.bootstrap_n, cfg.seed,
                                                 groups=valid.moment)
        comp = compliance_frame(sub, cfg)
        m['compliance_n'] = int(len(comp))
        m['compliance_before'] = _mean(comp.old_ok.astype(float))
        m['compliance_after'] = _mean(comp.new_ok.astype(float))
        m['counterfactual_consistency'] = _mean(comp.cf_ok.astype(float))
        m['compliance_diff_ci'] = bootstrap_diff(comp.old_ok.astype(float), comp.new_ok.astype(float),
                                                 cfg.bootstrap_n, cfg.seed, groups=comp.moment)
        m['compliance_intent'] = 'rule_intent'
        m['n_moments'] = int(sub.moment.nunique()) if len(sub) else 0
        out[name] = m
    return out


def json_safe(x):
    """NaN -> None so metrics.json is strict JSON."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    return x


def _fmt(x) -> str:
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.3f}'


def write_markdown(metrics: dict[str, dict], path: Path, cfg: Config) -> None:
    lines = ['# doPlan language re-ranking: results', '',
             f'lambda = {cfg.lam}; gate opens when plausible modes (score >= {cfg.gate_min_score}) spread '
             f'> {cfg.gate_endpoint_m} m at the endpoint or > {cfg.gate_heading_deg} deg in heading; '
             f'compliance threshold {cfg.compliance_m} m.', '',
             '| variant | subset | pairs with future | moments | top-1 ADE before | after | diff [95% CI] | '
             'minADE | top-1 FDE before | after |',
             '|---|---|---|---|---|---|---|---|---|---|']
    comp_lines = ['| variant | subset | n | before | after | diff [95% CI] | counterfactual |',
                  '|---|---|---|---|---|---|---|']
    skipped = []
    for variant, m in metrics.items():
        for subset in ('gated', 'all', 'gated_disagree'):
            s, ci, cci = m[subset], m[subset]['ade_diff_ci'], m[subset]['compliance_diff_ci']
            if s['n_pairs'] == 0:
                skipped.append((variant, subset))
                continue
            lines.append(f"| {variant} | {subset} | {s['n_with_future']} | {s['n_moments']} | "
                         f"{_fmt(s['ade_top1_before'])} | {_fmt(s['ade_top1_after'])} | "
                         f"{_fmt(ci['mean'])} [{_fmt(ci['lo'])}, {_fmt(ci['hi'])}] | {_fmt(s['min_ade'])} | "
                         f"{_fmt(s['fde_top1_before'])} | {_fmt(s['fde_top1_after'])} |")
            comp_lines.append(f"| {variant} | {subset} | {s['compliance_n']} | {_fmt(s['compliance_before'])} | "
                              f"{_fmt(s['compliance_after'])} | "
                              f"{_fmt(cci['mean'])} [{_fmt(cci['lo'])}, {_fmt(cci['hi'])}] | "
                              f"{_fmt(s['counterfactual_consistency'])} |")
    lines += [''] + comp_lines
    if skipped:
        by_subset: dict[str, list[str]] = {}
        for variant, subset in skipped:
            by_subset.setdefault(subset, []).append(variant)
        note = '; '.join(f"{subset} is empty for: {', '.join(vs)}" for subset, vs in by_subset.items())
        if ('rules_only', 'gated_disagree') in skipped:
            note += '; for rules_only this is by construction, since its intents are the rule labels'
        lines += ['', f'({note}.)']
    first = next(iter(metrics.values()))
    lines += ['', f"{first['n_pairs']} pairs over {first['n_predictions']} ego predictions, "
                  f"{first['n_moments']} distinct 0.5 s moments and {first['n_annotations']} annotations "
                  f"(CIs resample moments); gate open on {_fmt(first['gate_open_fraction'])} of pairs; "
                  f"classifier intent differs from the rule label on "
                  f"{_fmt(first['intent_disagrees_with_rules_fraction'])} of pairs ('gated_disagree' rows).",
              '', 'compliance population: rule-labelled lateral instructions with a full 6 s recorded future, '
                  'counted identically for every variant.',
              '', 'pairs per log: ' + ', '.join(f'{k} {v}' for k, v in first['pairs_per_log'].items()),
              '', 'Counterfactual consistency is meaningful on gated pairs only: a closed gate leaves '
                  'cf_top1 == new_top1.']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_examples(df: pd.DataFrame, pt_dir: Path, out_dir: Path, cfg: Config) -> list[Path]:
    """One PNG per pair with the largest L1 change between normalised MTR scores and fused scores."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    k = sum(1 for c in df.columns if c.startswith('s_old_'))
    old = df[[f's_old_{i}' for i in range(k)]].to_numpy(dtype=np.float64)
    old = old / old.sum(axis=1, keepdims=True)
    new = df[[f's_new_{i}' for i in range(k)]].to_numpy(dtype=np.float64)
    chosen = df.assign(_change=np.abs(new - old).sum(axis=1), **{f'oldnorm_{i}': old[:, i] for i in range(k)})
    chosen = chosen.sort_values('_change', ascending=False).drop_duplicates('pred_file')
    paths = []
    for n, r in enumerate(chosen.head(cfg.n_example_plots).itertuples(index=False)):
        p = load_prediction(pt_dir / r.pred_file)
        g, modes = np.asarray(p['gt_trajs']), np.asarray(p['pred_trajs'])
        hist, fut = g[:cfg.past_len], g[cfg.past_len:]
        hv, fv = hist[:, -1] > 0, fut[:, -1] > 0
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(hist[hv, 0], hist[hv, 1], 'k-', lw=3, zorder=4, label='history')
        if fv.any():
            ax.plot(fut[fv, 0], fut[fv, 1], color='0.2', ls='--', lw=2.5, zorder=4, label='recorded future')
        for i, m in enumerate(modes):
            style = {'lw': 3} if i == r.new_top1 else {'lw': 1.0, 'alpha': 0.7}
            ax.plot(m[:, 0], m[:, 1], zorder=2, label=f'mode {i}: MTR {getattr(r, f"oldnorm_{i}"):.2f} -> fused '
                                                     f'{getattr(r, f"s_new_{i}"):.2f}', **style)
        ax.scatter(*modes[r.old_top1, -1], marker='x', s=120, c='k', zorder=5, label=f'MTR top-1 (mode {r.old_top1})')
        ax.scatter(*modes[r.new_top1, -1], marker='*', s=200, c='r', zorder=5, label=f'fused top-1 (mode {r.new_top1})')
        ax.set_title(f'"{r.instruction[:90]}"\nintent {r.intent}; gate {"open" if r.gated else "closed"}; '
                     f'{r.pred_file}', fontsize=9)
        ax.axis('equal')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / f'example_{n:02d}_{r.annotation_id}_{r.scenario_id}_{r.time_index}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def write_metrics(metrics: dict[str, dict], out_dir: Path, cfg: Config) -> None:
    """metrics.json (strict JSON) and metrics.md side by side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics.json').write_text(json.dumps(json_safe(metrics), indent=2, allow_nan=False), encoding='utf-8')
    write_markdown(metrics, out_dir / 'metrics.md', cfg)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description='Evaluate one reranked.csv (run_all.py normally does this).')
    p.add_argument('--reranked', type=Path, required=True)
    p.add_argument('--pt-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--label', default='classifier')
    a = p.parse_args(argv)
    cfg = Config()
    df = add_errors(pd.read_csv(a.reranked), a.pt_dir, cfg)
    metrics = {a.label: summarize(df, cfg)}
    write_metrics(metrics, a.out_dir, cfg)
    plot_examples(df, a.pt_dir, a.out_dir / 'plots', cfg)
    print((a.out_dir / 'metrics.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
