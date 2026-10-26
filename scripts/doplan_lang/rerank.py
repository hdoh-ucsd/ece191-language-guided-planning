"""Decision-point gate and log-linear fusion of MTR mode scores with intent compatibility (spec 4.6).

new_scores = softmax( log(mtr_score) + lambda * log(compatibility) ), applied only where the
plausible modes fork (gate open). Fused scores are normalised; MTR's own scores are kept as written.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .config import INTENTS, Config
from .geometry import compat, future_features, mode_features
from .text import swap_left_right

IntentFn = Callable[[str], dict[str, float]]


def one_hot(intent: str) -> dict[str, float]:
    return {k: (1.0 if k == intent else 0.0) for k in INTENTS}


def load_prediction(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def wrap_deg(d) -> np.ndarray:
    """Wrap angle differences into [-180, 180)."""
    return (np.asarray(d, dtype=np.float64) + 180.0) % 360.0 - 180.0


def gate_open(pred_trajs, features: np.ndarray, scores, cfg: Config) -> bool:
    """True when the plausible modes disagree: endpoint spread > gate_endpoint_m or wrapped heading
    spread > gate_heading_deg among modes whose normalised MTR score is at least gate_min_score."""
    s = np.asarray(scores, dtype=np.float64)
    keep = s / s.sum() >= cfg.gate_min_score
    if keep.sum() < 2:
        return False
    ends = np.asarray(pred_trajs, dtype=np.float64)[keep, -1, :]
    dpsi = features[keep, 1]
    endpoint_spread = np.linalg.norm(ends[:, None, :] - ends[None, :, :], axis=-1).max()
    heading_spread = np.abs(wrap_deg(dpsi[:, None] - dpsi[None, :])).max()
    return bool(endpoint_spread > cfg.gate_endpoint_m or heading_spread > cfg.gate_heading_deg)


def fuse(scores, compatibility, lam: float) -> np.ndarray:
    s = np.log(np.clip(np.asarray(scores, dtype=np.float64), 1e-9, None))
    s = s + lam * np.log(np.asarray(compatibility, dtype=np.float64))
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def rerank_pairs(pairs: pd.DataFrame, pt_dir: Path, cfg: Config, intent_fn: IntentFn,
                 out_pt_dir: Path | None = None) -> pd.DataFrame:
    """Re-rank every (prediction, annotation) pair. `intent_fn(clause) -> {intent: prob}`.

    Adds per pair: p_<intent>, intent (argmax), gated, lat_k/dpsi_k/path_k, s_old_k/s_new_k,
    old_top1/new_top1/cf_top1 and their endpoint lateral offsets, and the recorded future's
    fut_lat/fut_dpsi/fut_path/fut_valid. `s_old_k` are MTR's scores as written
    (they need not sum to 1); `s_new_k` are normalised. With out_pt_dir, writes a copy of each prediction
    with pred_scores replaced (original kept as pred_scores_mtr).
    """
    if out_pt_dir is not None:
        out_pt_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, tuple] = {}
    rows = []
    for r in pairs.itertuples(index=False):
        if r.pred_file not in cache:
            pred = load_prediction(pt_dir / r.pred_file)
            feats = mode_features(pred['pred_trajs'], pred['gt_trajs'], cfg.past_len)
            cache[r.pred_file] = (pred, feats, future_features(pred['gt_trajs'], cfg.past_len),
                                  gate_open(pred['pred_trajs'], feats, pred['pred_scores'], cfg))
        pred, feats, fut, gated = cache[r.pred_file]
        probs = intent_fn(r.first_clause)
        probs_cf = intent_fn(swap_left_right(r.first_clause))
        old = np.asarray(pred['pred_scores'], dtype=np.float64)
        new = fuse(old, compat(feats, probs, cfg), cfg.lam) if gated else old / old.sum()
        new_cf = fuse(old, compat(feats, probs_cf, cfg), cfg.lam) if gated else old / old.sum()

        row = r._asdict()
        row.update({f'p_{k}': probs[k] for k in INTENTS})
        row['intent'] = max(probs, key=probs.get)
        row['gated'] = gated
        for k in range(len(old)):
            row[f'lat_{k}'], row[f'dpsi_{k}'], row[f'path_{k}'] = feats[k]
            row[f's_old_{k}'], row[f's_new_{k}'] = old[k], new[k]
        row['old_top1'], row['new_top1'], row['cf_top1'] = int(old.argmax()), int(new.argmax()), int(new_cf.argmax())
        row['old_top1_lat'] = feats[row['old_top1'], 0]
        row['new_top1_lat'] = feats[row['new_top1'], 0]
        row['cf_top1_lat'] = feats[row['cf_top1'], 0]
        row['fut_lat'], row['fut_dpsi'], row['fut_path'], row['fut_valid'] = fut
        rows.append(row)

        if out_pt_dir is not None:
            out = dict(pred)
            out['pred_scores_mtr'] = np.asarray(pred['pred_scores'])
            out['pred_scores'] = new.astype(np.float32)
            out['annotation_id'] = r.annotation_id
            out['instruction'] = r.instruction
            torch.save(out, out_pt_dir / f'{Path(r.pred_file).stem}_{r.annotation_id}.pt')
    return pd.DataFrame(rows)
