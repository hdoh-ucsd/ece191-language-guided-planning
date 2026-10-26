"""Chain text -> word2vec -> intent -> align -> rerank -> evaluate for one ScenarioNet database.

Usage (from the project root):
  PYTHONPATH=scripts python3 -m doplan_lang.run_all \
      --scenarionet scenarionet_data/nuplan_doplan --nuplan nuplan_data \
      --csv nuplan_data/doplan/release_annotationsV1.csv --out outputs/doplan_lang

word2vec and the intent classifier are trained once into --out and reused unless --retrain is given
or --exclude-log changes the hold-out set.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import joblib
from gensim.models import KeyedVectors

from . import intent as intent_mod
from . import train_word2vec
from .align import (build_pairs, coverage_table, list_ego_predictions, load_lidar_frames, load_summary,
                    prediction_times, sanity_check)
from .config import Config
from .evaluate import add_errors, plot_examples, summarize, write_metrics
from .intent import predict_proba
from .rerank import one_hot, rerank_pairs
from .text import load_annotations, rule_intent


def find_db(nuplan_root: Path, log_name: str) -> Path:
    hits = sorted((nuplan_root / 'nuplan-v1.1' / 'splits').glob(f'*/{log_name}.db'))
    if not hits:
        raise FileNotFoundError(f'no {log_name}.db under {nuplan_root}/nuplan-v1.1/splits/*/')
    return hits[0]


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenarionet', type=Path, required=True, help='database dir holding dataset_summary.pkl and output_MTR/')
    p.add_argument('--nuplan', type=Path, required=True, help='nuPlan data root (HOST_NUPLAN_DIR)')
    p.add_argument('--csv', type=Path, required=True, help='doPlan release_annotationsV1.csv')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--retrain', action='store_true', help='retrain word2vec and the classifier even if present')
    p.add_argument('--no-pt', action='store_true', help='do not write reranked_pt/ copies (tens of thousands of files on the real run)')
    p.add_argument('--lam', type=float, default=None, help='fusion exponent (default Config.lam)')
    p.add_argument('--exclude-log', action='append', default=None, help='logs to hold out of training; REPLACES config.EVAL_LOGS when given; repeatable')
    a = p.parse_args(argv)
    cfg = Config() if a.lam is None else replace(Config(), lam=a.lam)
    a.out.mkdir(parents=True, exist_ok=True)
    excl = [x for log in (a.exclude_log or []) for x in ('--exclude-log', log)]

    kv_path, clf_path = a.out / 'word2vec.kv', a.out / 'intent_clf.joblib'
    retrain = a.retrain or bool(a.exclude_log)   # a changed hold-out set must never reuse stale vectors
    if retrain or not kv_path.exists():
        train_word2vec.main(['--csv', str(a.csv), '--out-dir', str(a.out)] + excl)
    if retrain or not clf_path.exists():
        intent_mod.main(['--csv', str(a.csv), '--kv', str(kv_path), '--out-dir', str(a.out)] + excl)
    kv, clf = KeyedVectors.load(str(kv_path)), joblib.load(clf_path)

    instructions = load_annotations(a.csv)
    summary = load_summary(a.scenarionet)
    pt_dir = a.scenarionet / 'output_MTR'
    dbs = {log: find_db(a.nuplan, log) for log in sorted({m['log_name'] for m in summary.values()})}
    frames_by_log = {log: load_lidar_frames(db) for log, db in dbs.items()}
    preds = list_ego_predictions(pt_dir)
    n_preds = Counter(sid for _, sid, _ in preds)     # inference may have been capped to fewer scenarios
    checks = [sanity_check(frames_by_log[m['log_name']], m, pt_dir, cfg, predictions=preds)
              for sid, m in summary.items() if n_preds[sid] >= 2]
    if not checks:
        raise SystemExit(f'no scenario in {a.scenarionet} has two or more ego predictions in {pt_dir}')
    print(f'alignment check passed for {len(checks)} of {len(summary)} scenarios '
          f'({len(summary) - len(checks)} have no ego predictions); worst displacement error '
          f'{max(c["displacement_error_m"] for c in checks):.3f} m')

    times = prediction_times(preds, summary, frames_by_log, cfg)
    pairs = build_pairs(instructions, times, frames_by_log, cfg)
    pairs.to_csv(a.out / 'pairs.csv', index=False)
    cov = coverage_table(instructions, times, frames_by_log, cfg)
    cov.to_csv(a.out / 'coverage.csv', index=False)
    print(f'{len(preds)} ego predictions, {len(pairs)} pairs, {len(cov)} annotations on these logs; '
          f'median window coverage {cov.frac_covered.median():.2f}; '
          f'{int((cov.n_moments == 0).sum())} annotations without any prediction')
    if pairs.empty:
        raise SystemExit('no (prediction, annotation) pairs: output_MTR holds no ego predictions inside any window')

    variants = {'classifier': lambda c: predict_proba(clf, kv, c),
                'rules_only': lambda c: one_hot(rule_intent(c))}
    metrics = {}
    for name, fn in variants.items():
        is_main = name == 'classifier'
        df = rerank_pairs(pairs, pt_dir, cfg, fn, (None if a.no_pt else a.out / 'reranked_pt') if is_main else None)
        df = add_errors(df, pt_dir, cfg)
        df.to_csv(a.out / ('reranked.csv' if is_main else 'reranked_rules.csv'), index=False)
        metrics[name] = summarize(df, cfg)
        if is_main:
            plot_examples(df, pt_dir, a.out / 'plots', cfg)
    write_metrics(metrics, a.out, cfg)
    print((a.out / 'metrics.md').read_text(encoding='utf-8'))
    return metrics


if __name__ == '__main__':
    main()
