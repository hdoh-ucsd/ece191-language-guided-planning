"""Mean-word2vec sentence vectors and the weakly supervised intent classifier (spec 4.3).

Usage:
  PYTHONPATH=scripts python3 -m doplan_lang.intent \
      --csv nuplan_data/doplan/release_annotationsV1.csv \
      --kv outputs/doplan_lang/word2vec.kv --out-dir outputs/doplan_lang
Writes intent_clf.joblib, intent_cv.json and intent_report.md.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import EVAL_LOGS, INTENTS
from .text import Instruction, load_annotations, tokenize

_N_ZERO_ROWS = 20   # backstop: synthetic all-OOV rows labelled 'other'; predict_proba also short-circuits the zero vector


def sentence_vector(tokens, kv: KeyedVectors) -> np.ndarray:
    vecs = [kv[t] for t in tokens if t in kv.key_to_index]
    if not vecs:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def dataset(instructions: list[Instruction], kv: KeyedVectors, exclude_logs):
    """Feature matrix over first action clauses, weak rule labels, and the rows used."""
    excluded = set(exclude_logs)
    rows = [i for i in instructions if i.clip_id not in excluded]
    if not rows:
        return np.zeros((0, kv.vector_size), dtype=np.float32), np.array([], dtype=object), rows
    X = np.stack([sentence_vector(tokenize(i.first_clause), kv) for i in rows])
    y = np.array([i.rule_intent for i in rows], dtype=object)
    return X, y, rows


def _new_clf(seed: int) -> Pipeline:
    # Mean word vectors have small norms, so standardise before the (C=1.0) regulariser.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=seed))


def train_classifier(X: np.ndarray, y: np.ndarray, seed: int = 0) -> Pipeline:
    zeros = np.zeros((_N_ZERO_ROWS, X.shape[1]), dtype=X.dtype)
    clf = _new_clf(seed)
    clf.fit(np.vstack([X, zeros]), np.concatenate([y, np.array(['other'] * _N_ZERO_ROWS, dtype=object)]))
    return clf


def cross_validate(X: np.ndarray, y: np.ndarray, seed: int = 0, folds: int = 5) -> dict:
    if len(set(y)) < 2:
        raise ValueError(f'need at least two intent classes to cross-validate, got {sorted(set(y))}')
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(_new_clf(seed), X, y, cv=skf)
    labels = [k for k in INTENTS if k in set(y)]
    return {
        'n': int(len(y)),
        'accuracy': float(accuracy_score(y, pred)),
        'f1': dict(zip(labels, f1_score(y, pred, labels=labels, average=None).tolist())),
        'labels': labels,
        'confusion': confusion_matrix(y, pred, labels=labels).tolist(),
    }


def predict_proba(clf: Pipeline, kv: KeyedVectors, clause: str) -> dict[str, float]:
    v = sentence_vector(tokenize(clause), kv)
    if not np.any(v):
        return {k: (1.0 if k == 'other' else 0.0) for k in INTENTS}
    probs = clf.predict_proba(v[None])[0]
    out = {k: 0.0 for k in INTENTS}
    for cls, pr in zip(clf.classes_, probs):
        out[str(cls)] = float(pr)
    return out


def report(cv: dict, clf: Pipeline, kv: KeyedVectors, rows: list[Instruction], seed: int = 0, n_examples: int = 30) -> str:
    lines = ['# intent classifier report', '',
             f'- training clauses: {cv["n"]}', f'- 5-fold CV accuracy vs rules: {cv["accuracy"]:.3f}',
             '- per-class F1: ' + ', '.join(f'{k} {v:.2f}' for k, v in cv['f1'].items()), '',
             '## confusion (rows = rule label, cols = predicted)', '',
             '| | ' + ' | '.join(cv['labels']) + ' |', '|' + '---|' * (len(cv['labels']) + 1)]
    for lab, row in zip(cv['labels'], cv['confusion']):
        lines.append(f'| {lab} | ' + ' | '.join(str(c) for c in row) + ' |')
    others = [r for r in rows if r.rule_intent == 'other' and r.first_clause]
    preds = [(r, predict_proba(clf, kv, r.first_clause)) for r in others]
    counts = Counter(max(p, key=p.get) for _, p in preds)
    lines += ['', f'## clauses the rules could not label ({len(others)})', '',
              '- classifier argmax counts: ' + ', '.join(f'{k} {v}' for k, v in counts.most_common()), '']
    random.Random(seed).shuffle(preds)
    for r, p in preds[:n_examples]:
        top = max(p, key=p.get)
        lines.append(f'- "{r.first_clause}" -> {top} ({p[top]:.2f})')
    return '\n'.join(lines) + '\n'


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--kv', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--exclude-log', action='append', default=None,
                   help='logs to hold out of training; REPLACES config.EVAL_LOGS when given; repeatable')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    kv = KeyedVectors.load(str(a.kv))
    instructions = load_annotations(a.csv)
    excluded = tuple(a.exclude_log) if a.exclude_log else EVAL_LOGS
    X, y, rows = dataset(instructions, kv, excluded)
    cv = cross_validate(X, y, a.seed)
    clf = train_classifier(X, y, a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, a.out_dir / 'intent_clf.joblib')
    (a.out_dir / 'intent_cv.json').write_text(json.dumps(cv, indent=2), encoding='utf-8')
    (a.out_dir / 'intent_report.md').write_text(report(cv, clf, kv, rows, a.seed), encoding='utf-8')
    print(f'{cv["n"]} clauses; CV accuracy {cv["accuracy"]:.3f}; saved {a.out_dir / "intent_clf.joblib"}')


if __name__ == '__main__':
    main()
