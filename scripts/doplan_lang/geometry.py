"""Per-mode trajectory features in the ego frame and intent compatibility (spec 4.5).

Ego frame at prediction time: origin at the ego position, x axis along its heading, so a
positive lateral offset means "to the left". Features per mode:
  lat   endpoint lateral offset (m)
  dpsi  final heading relative to the current heading (deg), from the last displacement
        between waypoints at least 0.5 m apart (0 if the mode barely moves)
  path  cumulative path length (m) from the ego position through all waypoints
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from .config import INTENTS, Config


def sigmoid(x) -> np.ndarray:
    return expit(np.asarray(x, dtype=np.float64))   # scipy's logistic: no overflow warnings for huge inputs


def to_ego_frame(xy, origin_xy, heading: float) -> np.ndarray:
    d = np.asarray(xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64)
    c, s = np.cos(heading), np.sin(heading)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)


def traj_features(xy_ego, min_sep_m: float = 0.5) -> tuple[float, float, float]:
    """(lat, dpsi, path) of a trajectory given in the ego frame; the path starts at the ego (origin)."""
    xy = np.asarray(xy_ego, dtype=np.float64)
    if len(xy) == 0:
        return 0.0, 0.0, 0.0
    xy = np.vstack([np.zeros((1, 2)), xy])       # the first waypoint is 0.1 s ahead; count the way to it
    lat = float(xy[-1, 1])
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    dpsi = 0.0
    for j in range(len(xy) - 2, -1, -1):
        d = xy[-1] - xy[j]
        if np.linalg.norm(d) >= min_sep_m:
            dpsi = float(np.degrees(np.arctan2(d[1], d[0])))
            break
    return lat, dpsi, path


def mode_features(pred_trajs, gt_trajs, past_len: int = 21) -> np.ndarray:
    """[modes, 3] = (lat, dpsi, path) for each predicted mode (pred_trajs in the same frame as gt_trajs)."""
    now = np.asarray(gt_trajs)[past_len - 1]
    origin, heading = now[:2], float(now[6])
    return np.array([traj_features(to_ego_frame(m, origin, heading)) for m in np.asarray(pred_trajs)], dtype=np.float64)


def future_features(gt_trajs, past_len: int = 21) -> tuple[float, float, float, int]:
    """(lat, dpsi, path, n_valid) of the recorded future, valid steps only."""
    g = np.asarray(gt_trajs)
    now, fut = g[past_len - 1], g[past_len:]
    valid = fut[:, -1] > 0
    lat, dpsi, path = traj_features(to_ego_frame(fut[valid, :2], now[:2], float(now[6])))
    return lat, dpsi, path, int(valid.sum())


def compat_table(features: np.ndarray, cfg: Config) -> np.ndarray:
    """[modes, len(INTENTS)] compatibility in [floor, 1], columns in INTENTS order."""
    lat, dpsi, path = features[:, 0], features[:, 1], features[:, 2]

    def above(x, threshold, scale):
        return sigmoid((x - threshold) / scale)

    not_turning_lane = 1 - above(np.abs(dpsi), cfg.geom_lane_heading_deg, 10.0)
    table = np.stack([
        above(lat, cfg.turn_lat_m, 1.0) * above(dpsi, cfg.turn_heading_deg, 10.0),        # turn_left
        above(-lat, cfg.turn_lat_m, 1.0) * above(-dpsi, cfg.turn_heading_deg, 10.0),      # turn_right
        above(lat, cfg.lane_lat_m, 0.75) * not_turning_lane,                                # lane_left
        above(-lat, cfg.lane_lat_m, 0.75) * not_turning_lane,                               # lane_right
        1 - above(path, cfg.stop_path_m, 1.5),                                              # stop
        (1 - above(np.abs(lat), cfg.straight_lat_m, 0.75))
        * (1 - above(np.abs(dpsi), cfg.straight_heading_deg, 10.0)),                        # straight
        np.ones_like(lat),                                                                  # other
    ], axis=1)
    assert table.shape[1] == len(INTENTS)
    return np.maximum(table, cfg.compat_floor)


def compat_table_from_labels(labels, stops, cfg: Config) -> np.ndarray:
    """Label table (v2 spec 4.2): the label column is 1, the other lateral/straight columns the floor, `stop`
    follows the stop flag, `other` is 1. A mode without a label (off every lane, reversing, or no route) gets the
    floor on every lateral/straight column: with a map available it cannot be shown to do any of them."""
    table = np.full((len(labels), len(INTENTS)), cfg.compat_floor)
    i_stop, i_other = INTENTS.index('stop'), INTENTS.index('other')
    for k, (lab, stop) in enumerate(zip(labels, stops)):
        if lab is not None:
            table[k, INTENTS.index(lab)] = 1.0
        if stop:
            table[k, i_stop] = 1.0
        table[k, i_other] = 1.0
    return table


def compat(features: np.ndarray, probs: dict[str, float], cfg: Config, labels=None, stops=None) -> np.ndarray:
    """Expected compatibility of each mode under the intent distribution `probs`: the lane-label table when
    labels are given (the scenario has a map) and cfg.compat_source == 'map', else the v1 geometric table."""
    p = np.array([probs.get(k, 0.0) for k in INTENTS], dtype=np.float64)
    if labels is not None and cfg.compat_source == 'map':
        table = compat_table_from_labels(labels, stops if stops is not None else [False] * len(labels), cfg)
    else:
        table = compat_table(features, cfg)
    return table @ p
