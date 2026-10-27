"""Lane graph of one ScenarioNet scenario: what a trajectory does, read off the map (v2 spec 4.1).

A trajectory is described by the lane it starts on, the lane it ends on and the cheapest route
between them over the lane graph (exit moves, left and right neighbour moves). The heading change
along that route and the net number of lateral moves give the label: turn_left, turn_right,
lane_left, lane_right or straight. Stopping is read from the trajectory's net speed over its last
0.5 s. A None label means the start or the end is on no lane, or no route links them within the
cost bound; callers fall back to the v1 geometric table. Coordinates are the scenario frame.
"""
from __future__ import annotations

import heapq
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from .config import Config

LANE_PREFIX = 'LANE_SURFACE'
LABELS = ('turn_left', 'turn_right', 'lane_left', 'lane_right', 'straight')
STEP_S = 0.1                       # ScenarioNet and MTR step


class _MapUnpickler(pickle.Unpickler):
    """ScenarioNet pickles its ScenarioDescription (a dict subclass from metadrive); read it as a dict."""

    def find_class(self, module, name):
        if module.startswith(('metadrive', 'scenarionet')):
            return dict
        return super().find_class(module, name)


def load_scenario(scenarionet_dir: Path, scenario_id: str) -> dict:
    """The ScenarioNet pickle of one scenario as plain dicts ({} when the file is not there)."""
    hits = sorted(Path(scenarionet_dir).glob(f'*/sd_*_{scenario_id}.pkl'))
    if not hits:
        return {}
    with open(hits[0], 'rb') as f:
        return dict(_MapUnpickler(f).load())


def load_map_features(scenarionet_dir: Path, scenario_id: str) -> dict:
    return dict(load_scenario(scenarionet_dir, scenario_id).get('map_features', {}))


def wrap_deg(d):
    return (np.asarray(d, dtype=np.float64) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Lane:
    lane_id: str
    xy: np.ndarray            # [n, 2]
    s: np.ndarray             # [n] cumulative arc length
    heading: np.ndarray       # [n] radians, direction of the segment ahead
    polygon: MplPath | None
    exits: tuple[str, ...]
    entries: tuple[str, ...]
    left: tuple[str, ...]
    right: tuple[str, ...]

    def project(self, points) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(s, distance, heading) of the nearest centreline point for each of `points` [m, 2]."""
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        a, ab = self.xy[:-1], np.diff(self.xy, axis=0)
        ab2 = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12)
        t = np.clip(np.einsum('mij,ij->mi', p[:, None, :] - a[None], ab) / ab2, 0.0, 1.0)      # [m, n-1]
        q = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(q - p[:, None, :], axis=-1)
        i = d.argmin(axis=1)
        rows = np.arange(len(p))
        s = self.s[i] + t[rows, i] * (self.s[i + 1] - self.s[i])
        return s, d[rows, i], self.heading[i]


@dataclass(frozen=True)
class LanePose:
    lane_id: str
    s: float
    dist: float
    heading: float


@dataclass(frozen=True)
class Route:
    moves: str               # 'F' exit, 'L' left neighbour, 'R' right neighbour
    dpsi_deg: float          # lane heading at the end minus at the start, wrapped


@dataclass(frozen=True)
class Manoeuvre:
    label: str | None
    stop: bool
    start: LanePose | None
    end: LanePose | None
    final_speed_mps: float


NO_MANOEUVRE = Manoeuvre(None, False, None, None, 0.0)


class LaneGraph:
    def __init__(self, lanes: dict[str, Lane]):
        self.lanes = lanes
        self.ids = list(lanes)
        pts = [lane.xy for lane in lanes.values()]
        self._pts = np.vstack(pts) if pts else np.zeros((0, 2))
        self._owner = (np.concatenate([np.full(len(lane.xy), i) for i, lane in enumerate(lanes.values())])
                       if pts else np.zeros(0, int))
        self._tree = cKDTree(self._pts) if len(self._pts) else None

    @classmethod
    def from_map_features(cls, map_features: dict) -> 'LaneGraph':
        raw = {str(k): v for k, v in map_features.items()
               if str(v.get('type', '')).startswith(LANE_PREFIX) and v.get('polyline') is not None and len(v['polyline']) >= 2}
        lanes: dict[str, Lane] = {}
        for lid, v in raw.items():
            xy = np.asarray(v['polyline'], dtype=np.float64)[:, :2]
            seg = np.diff(xy, axis=0)
            heading = np.arctan2(seg[:, 1], seg[:, 0])
            poly = v.get('polygon')
            links = {k: tuple(str(x) for x in (v.get(k) if v.get(k) is not None else []) if str(x) in raw)
                     for k in ('exit_lanes', 'entry_lanes', 'left_neighbor', 'right_neighbor')}
            lanes[lid] = Lane(lid, xy, np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))]),
                              np.append(heading, heading[-1]),
                              MplPath(np.asarray(poly, dtype=np.float64)[:, :2]) if poly is not None and len(poly) >= 3 else None,
                              links['exit_lanes'], links['entry_lanes'], links['left_neighbor'], links['right_neighbor'])
        return cls(lanes)

    def __len__(self) -> int:
        return len(self.lanes)

    def candidates(self, point, radius_m: float) -> list[str]:
        """Lanes with a centreline point within radius_m of `point`, nearest first."""
        if self._tree is None:
            return []
        idx = self._tree.query_ball_point(np.asarray(point, dtype=np.float64), radius_m, return_sorted=True)
        return [self.ids[i] for i in dict.fromkeys(int(self._owner[j]) for j in idx)]

    def contains(self, point) -> bool:
        """True when some lane polygon contains `point`."""
        p = np.asarray(point, dtype=np.float64)
        return any(self.lanes[c].polygon is not None and self.lanes[c].polygon.contains_point(p)
                   for c in self.candidates(p, 6.0))

    def centreline_distance(self, point, radius_m: float = 6.0) -> float:
        """Distance from `point` to the nearest lane centreline (inf when none lies within radius_m)."""
        p = np.asarray(point, dtype=np.float64)
        cands = self.candidates(p, radius_m)
        return min((float(self.lanes[c].project(p)[1][0]) for c in cands), default=float('inf'))

    def off_lane(self, point, margin_m: float) -> bool:
        """Outside every lane polygon AND farther than margin_m from every centreline (edge-riding is on-lane)."""
        return not self.contains(point) and self.centreline_distance(point) > margin_m

    def assign(self, point, tail=None, heading: float | None = None, cfg: Config = Config()) -> LanePose | None:
        """The lane under `point` (spec 4.1). Candidates are the lanes with a centreline within lane_assign_m;
        lanes heading more than lane_heading_deg away from `heading` (radians; when not given, the direction
        of travel of the `tail` points, if they moved at least 1 m) are dropped; the rest are scored by the
        mean centreline distance of the tail points (the point's own distance without a tail) with a bonus
        of lane_inside_bonus_m for a polygon that contains the point. The tail is what disambiguates
        overlapping intersection connectors: it asks which lane the trajectory has been following."""
        p = np.asarray(point, dtype=np.float64)
        if not np.all(np.isfinite(p)):
            return None
        cands = self.candidates(p, cfg.lane_assign_m)
        if not cands:
            return None
        tail = None if tail is None else np.asarray(tail, dtype=np.float64)
        if tail is not None and (len(tail) < 2 or not np.all(np.isfinite(tail))):
            tail = None
        if heading is None and tail is not None and np.linalg.norm(tail[-1] - tail[0]) >= 1.0:
            heading = float(np.arctan2(*(tail[-1] - tail[0])[::-1]))
        poses = {c: tuple(float(x[0]) for x in self.lanes[c].project(p)) for c in cands}
        if heading is not None:
            cands = [c for c in cands if abs(float(wrap_deg(np.degrees(poses[c][2] - heading)))) < cfg.lane_heading_deg]
            if not cands:
                return None

        def score(c: str) -> float:
            base = float(self.lanes[c].project(tail)[1].mean()) if tail is not None else poses[c][1]
            inside = self.lanes[c].polygon is not None and self.lanes[c].polygon.contains_point(p)
            return base - (cfg.lane_inside_bonus_m if inside else 0.0)

        best = min(cands, key=score)
        s, d, h = poses[best]
        return LanePose(best, s, d, h)

    def route(self, a: LanePose | None, b: LanePose | None, cfg: Config = Config()) -> Route | None:
        """Cheapest route from lane a to lane b over exit (F, cost 1) and neighbour (L/R, lane_lateral_cost)
        moves within lane_route_max_cost; None when there is none."""
        if a is None or b is None or a.lane_id not in self.lanes or b.lane_id not in self.lanes:
            return None
        best = {a.lane_id: 0.0}
        heap = [(0.0, a.lane_id, '')]
        while heap:
            cost, lid, moves = heapq.heappop(heap)
            if lid == b.lane_id:
                return Route(moves, float(wrap_deg(np.degrees(b.heading - a.heading))))
            if cost > best.get(lid, np.inf):
                continue
            lane = self.lanes[lid]
            for nxt, mv, c in ([(x, 'F', 1.0) for x in lane.exits] + [(x, 'L', cfg.lane_lateral_cost) for x in lane.left]
                               + [(x, 'R', cfg.lane_lateral_cost) for x in lane.right]):
                nc = cost + c
                if nc <= cfg.lane_route_max_cost and nc < best.get(nxt, np.inf):
                    best[nxt] = nc
                    heapq.heappush(heap, (nc, nxt, moves + mv))
        return None


def label(route: Route | None, cfg: Config = Config()) -> str | None:
    """turn by the lane heading change, else lane change by the net lateral moves, else straight."""
    if route is None:
        return None
    if abs(route.dpsi_deg) >= cfg.turn_heading_deg:
        return 'turn_left' if route.dpsi_deg > 0 else 'turn_right'
    net = route.moves.count('L') - route.moves.count('R')
    return 'lane_left' if net >= 1 else 'lane_right' if net <= -1 else 'straight'


def final_speed(traj, dt: float = STEP_S, window_s: float = 0.5) -> float:
    """Net displacement over the last window_s divided by window_s (a creeping tail reads as slow)."""
    xy = np.asarray(traj, dtype=np.float64)
    if len(xy) < 2:
        return 0.0
    k = min(max(1, int(round(window_s / dt))), len(xy) - 1)
    return float(np.linalg.norm(xy[-1] - xy[-1 - k]) / (k * dt))


def manoeuvre(graph: LaneGraph, now_xy, now_heading: float, hist_xy, traj_xy, cfg: Config = Config()) -> Manoeuvre:
    """Label of a trajectory that starts at the ego pose `now` (spec 4.1). `hist_xy` are the last history
    points (may be empty); `traj_xy` [n, 2] the future points, 0.1 s apart."""
    traj = np.asarray(traj_xy, dtype=np.float64)
    if len(traj) == 0 or not np.all(np.isfinite(traj)):
        return NO_MANOEUVRE
    n_tail = max(2, int(round(cfg.lane_tail_s / STEP_S)))
    hist = np.asarray(hist_xy, dtype=np.float64) if hist_xy is not None else np.zeros((0, 2))
    start = graph.assign(now_xy, tail=hist[-n_tail:] if len(hist) >= 2 else None, heading=now_heading, cfg=cfg)
    end = graph.assign(traj[-1], tail=traj[-n_tail:], cfg=cfg)
    v = final_speed(traj)
    lab = label(graph.route(start, end, cfg), cfg) if start is not None and end is not None else None
    return Manoeuvre(lab, v < cfg.stop_speed_mps, start, end, v)


def mode_manoeuvres(pred: dict, graph: LaneGraph | None, cfg: Config) -> tuple[list[Manoeuvre], Manoeuvre]:
    """(one Manoeuvre per mode, Manoeuvre of the recorded future) of one prediction file."""
    modes = np.asarray(pred['pred_trajs'])[:, :, :2]
    if graph is None or len(graph) == 0:
        return [NO_MANOEUVRE] * len(modes), NO_MANOEUVRE
    g = np.asarray(pred['gt_trajs'])
    now = g[cfg.past_len - 1]
    hist = g[:cfg.past_len]
    hist = hist[hist[:, -1] > 0, :2]
    fut = g[cfg.past_len:]
    fut = fut[fut[:, -1] > 0, :2]
    out = [manoeuvre(graph, now[:2], float(now[6]), hist, m, cfg) for m in modes]
    return out, (manoeuvre(graph, now[:2], float(now[6]), hist, fut, cfg) if len(fut) else NO_MANOEUVRE)
