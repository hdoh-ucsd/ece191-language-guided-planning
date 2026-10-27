# Lane-grounded metric and toy simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Judge what an MTR trajectory does by the lane graph of its ScenarioNet scenario (compatibility and compliance), then execute the selected trajectory in a minimal simulator with replayed agents and score completion, off-road, collision, comfort and tracking.

**Architecture:** One new module `lanes.py` owns the map (lane graph, assignment, routing, labels, stop detection). `geometry.py` turns labels into a compatibility table; `rerank.py` records labels per mode; `evaluate.py` adds lane compliance; `run_all.py` loads graphs lazily and adds the geometry ablation variant. A separate `toysim.py` executes trajectories with a pure-pursuit tracker on a kinematic bicycle among log-replayed agents and reports outcome metrics. Nothing in the Docker images changes.

**Tech Stack:** Python 3.12, numpy, scipy (`cKDTree`), matplotlib (`path.Path`), pandas, torch (CPU, to read `.pt`), pytest. Existing package `scripts/doplan_lang/` (import as `doplan_lang`, `PYTHONPATH=scripts`).

**Spec:** `docs/superpowers/specs/2026-10-26-lane-grounded-metric-and-toysim-design.md` (sections cited as "spec 4.1").

**Version control:** the user has not asked for commits; the commit steps below are listed for a worker who is told to commit. Otherwise leave the tree uncommitted.

**Conventions:** run from `/root/valeo_mtr_docker`; test command `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests -q`; data-gated tests skip when `scenarionet_data/` is absent.

---

## File structure

| Path | Responsibility |
|---|---|
| `scripts/doplan_lang/config.py` | new lane-graph and simulator knobs; `geom_lane_heading_deg` renamed from the v1 `lane_heading_deg` |
| `scripts/doplan_lang/lanes.py` (new) | `load_scenario`, `load_map_features`, `Lane`, `LaneGraph` (assign, route, contains), `label`, `final_speed`, `manoeuvre`, `mode_manoeuvres` |
| `scripts/doplan_lang/geometry.py` | `compat_table_from_labels`; `compat` takes labels and stops |
| `scripts/doplan_lang/rerank.py` | `rerank_pairs(..., graph_for=None)`: label columns, map compatibility |
| `scripts/doplan_lang/evaluate.py` | `lane_compliance_frame`, lane compliance in `summarize` and `write_markdown` |
| `scripts/doplan_lang/run_all.py` | `--compat`, lazy graphs, `classifier_geometry` variant |
| `scripts/doplan_lang/scene_paths.py`, `scene_report.py` | import `load_map_features` from `lanes`; label columns in the tables |
| `scripts/doplan_lang/toysim.py` (new) | kinematic bicycle, pure pursuit, replayed agents, SAT collision, off-road, completion, report, CLI |
| `scripts/doplan_lang/tests/conftest.py` | `synthetic_map_features()` fixture helper |
| `scripts/doplan_lang/tests/test_lanes.py`, `test_toysim.py` (new); `test_geometry.py`, `test_rerank.py`, `test_evaluate.py`, `test_scene_report.py`, `test_config.py` (modify) | tests |

---

### Task 1: Config knobs

**Files:**
- Modify: `scripts/doplan_lang/config.py`
- Modify: `scripts/doplan_lang/geometry.py` (one rename)
- Test: `scripts/doplan_lang/tests/test_config.py`

- [ ] **Step 1: Write the failing test** (append to `test_config.py`)

```python
def test_v2_knobs_present():
    from doplan_lang.config import Config
    cfg = Config()
    assert cfg.compat_source == 'map' and cfg.geom_lane_heading_deg == 20.0 and cfg.lane_heading_deg == 90.0
    assert cfg.stop_speed_mps == 1.0 and cfg.sim_dt == 0.1 and cfg.sim_max_curvature == 0.2
```

- [ ] **Step 2: Run it**: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_config.py -q` → FAIL (`geom_lane_heading_deg`).

- [ ] **Step 3: Implement.** In `config.py` replace `lane_heading_deg: float = 20.0` by `geom_lane_heading_deg: float = 20.0` and append to `Config`:

```python
    # lane graph (v2 spec 4.1, 5)
    compat_source: str = 'map'          # 'map' (lane labels) or 'geometry' (v1 sigmoid table)
    lane_assign_m: float = 3.0          # max centreline distance for a lane assignment
    lane_heading_deg: float = 90.0      # max |ego heading - lane heading| for the ego's current lane
    lane_tail_s: float = 2.0            # trajectory tail that disambiguates overlapping lanes
    lane_lateral_cost: float = 1.5      # route cost of a neighbour move (an exit move costs 1)
    lane_route_max_cost: float = 8.0
    stop_speed_mps: float = 1.0         # net speed over the last 0.5 s below which a trajectory "stops"
    # toy simulator (v2 spec 4.6)
    sim_dt: float = 0.1
    sim_max_accel: float = 4.0
    sim_max_curvature: float = 0.2
    sim_lookahead_gain: float = 0.8
    sim_lookahead_min_m: float = 3.0
    sim_lookahead_max_m: float = 10.0
    sim_speed_gain: float = 1.0
    sim_ego_length: float = 5.176
    sim_ego_width: float = 2.297
```

In `geometry.py` `compat_table`: `cfg.lane_heading_deg` → `cfg.geom_lane_heading_deg`. Grep for other uses: `grep -rn lane_heading_deg scripts/ docs/` and fix any (the v1 spec text stays).

- [ ] **Step 4: Run the whole suite** → all pass.

---

### Task 2: Lane graph (`lanes.py`)

**Files:**
- Create: `scripts/doplan_lang/lanes.py`
- Modify: `scripts/doplan_lang/tests/conftest.py` (synthetic map helper)
- Test: `scripts/doplan_lang/tests/test_lanes.py`

- [ ] **Step 1: Synthetic map helper** (append to `conftest.py`)

```python
def _lane(xy, exits=(), entries=(), left=(), right=(), kind='LANE_SURFACE_STREET', half_width=1.75):
    xy = np.asarray(xy, dtype=np.float64)
    seg = np.diff(xy, axis=0)
    n = np.stack([-seg[:, 1], seg[:, 0]], 1) / np.linalg.norm(seg, axis=1)[:, None]
    n = np.vstack([n, n[-1:]])
    polygon = np.vstack([xy + half_width * n, (xy - half_width * n)[::-1]])
    return {'type': kind, 'polyline': xy, 'polygon': polygon, 'exit_lanes': list(exits), 'entry_lanes': list(entries),
            'left_neighbor': list(left), 'right_neighbor': list(right)}


def _arc_xy(centre, radius, deg_from, deg_to, n=40):
    th = np.radians(np.linspace(deg_from, deg_to, n))
    return np.stack([centre[0] + radius * np.cos(th), centre[1] + radius * np.sin(th)], 1)


def synthetic_map_features():
    """Three straight lanes A0 (y=+3.5), A1 (y=0), A2 (y=-3.5) from x=0 to 60 continuing into B0..B2 to
    x=120; a right-turn connector C (A2 -> south, radius 10) and a left-turn connector D (A0 -> north)."""
    xs = np.linspace(0, 60, 61)
    f = {}
    for i, y in enumerate((3.5, 0.0, -3.5)):
        left, right = ([f'A{i - 1}'] if i > 0 else []), ([f'A{i + 1}'] if i < 2 else [])
        f[f'A{i}'] = _lane(np.stack([xs, np.full_like(xs, y)], 1), exits=[f'B{i}'] + (['D'] if i == 0 else ['C'] if i == 2 else []),
                           left=left, right=right)
        f[f'B{i}'] = _lane(np.stack([xs + 60, np.full_like(xs, y)], 1), entries=[f'A{i}'],
                           left=[f'B{i - 1}'] if i > 0 else [], right=[f'B{i + 1}'] if i < 2 else [])
    f['C'] = _lane(_arc_xy((60.0, -13.5), 10.0, 90, 0), entries=['A2'], kind='LANE_SURFACE_UNSTRUCTURE')
    f['D'] = _lane(_arc_xy((60.0, 13.5), 10.0, -90, 0), entries=['A0'], kind='LANE_SURFACE_UNSTRUCTURE')
    return f
```

- [ ] **Step 2: Failing tests** (`test_lanes.py`)

```python
import numpy as np

from doplan_lang.config import Config
from doplan_lang.lanes import LaneGraph, LanePose, final_speed, label, manoeuvre, wrap_deg

from .conftest import synthetic_map_features

CFG = Config()


def graph():
    return LaneGraph.from_map_features(synthetic_map_features())


def test_build_filters_and_links():
    g = graph()
    assert set(g.lanes) == {'A0', 'A1', 'A2', 'B0', 'B1', 'B2', 'C', 'D'}
    assert g.lanes['A1'].left == ('A0',) and g.lanes['A1'].right == ('A2',) and g.lanes['A2'].exits == ('B2', 'C')
    assert g.lanes['A1'].s[-1] == 60.0 and abs(g.lanes['A1'].heading[0]) < 1e-9
    assert len(LaneGraph.from_map_features({})) == 0 and LaneGraph.from_map_features({}).assign((0, 0)) is None


def test_assign_uses_polygon_then_tail():
    g = graph()
    p = g.assign((30.0, 0.4))
    assert p.lane_id == 'A1' and abs(p.s - 30.0) < 1e-6 and abs(p.dist - 0.4) < 1e-6
    assert g.assign((30.0, 40.0)) is None                                     # nothing within 3 m
    # x = 62, y = -3.6 lies inside both B2 and the connector C; the tail decides
    straight_tail = np.stack([np.linspace(58, 62, 20), np.full(20, -3.5)], 1)
    arc_tail = np.stack([60 + 10 * np.cos(np.radians(np.linspace(90, 78, 20))), -13.5 + 10 * np.sin(np.radians(np.linspace(90, 78, 20)))], 1)
    assert g.assign((62.0, -3.6), tail=straight_tail).lane_id == 'B2'
    assert g.assign(arc_tail[-1], tail=arc_tail).lane_id == 'C'
    assert g.assign((30.0, 0.4), heading=np.pi).lane_id is None or g.assign((30.0, 0.4), heading=np.pi) is None   # wrong way


def test_routes_and_labels():
    g = graph()
    a1, a0, a2 = g.assign((30, 0)), g.assign((30, 3.5)), g.assign((30, -3.5))
    b1, c, d = g.assign((90, 0)), g.assign((69.9, -12.5)), g.assign((69.9, 12.5))
    assert c.lane_id == 'C' and d.lane_id == 'D'
    assert label(g.route(a1, a1)) == 'straight' and g.route(a1, a1).moves == ''
    assert label(g.route(a1, b1)) == 'straight' and g.route(a1, b1).moves == 'F'
    assert label(g.route(a1, a0)) == 'lane_left' and label(g.route(a1, a2)) == 'lane_right'
    assert label(g.route(a2, c)) == 'turn_right' and label(g.route(a1, c)) == 'turn_right'   # heading beats the R move
    assert label(g.route(a0, d)) == 'turn_left'
    assert g.route(a1, LanePose('zzz', 0, 0, 0)) is None and label(None) is None
    assert g.route(b1, a1) is None                                   # no backwards edges
    far = Config(lane_route_max_cost=0.5)
    assert g.route(a1, b1, far) is None


def test_final_speed_and_stop_on_creeping_tail():
    xy = np.stack([np.linspace(0, 20, 60), np.zeros(60)], 1)
    assert abs(final_speed(xy) - (20 / 59 * 5 / 0.5)) < 1e-6
    creep = xy.copy(); creep[-10:] = [[20.0, 0], [20.4, 0], [20.1, 0], [20.2, 0], [20.1, 0], [20.2, 0], [20.1, 0], [20.2, 0], [20.1, 0], [20.0, 0]]
    assert final_speed(creep) < 1.0
    g = graph()
    m = manoeuvre(g, (10.0, 0.0), 0.0, np.stack([np.linspace(0, 10, 20), np.zeros(20)], 1), creep + [10.0, 0.0], CFG)
    assert m.label == 'straight' and m.stop and m.start.lane_id == 'A1' and m.end.lane_id == 'A1'


def test_manoeuvre_labels_turn_and_lane_change():
    g = graph()
    hist = np.stack([np.linspace(40, 50, 20), np.full(20, -3.5)], 1)
    turn = np.vstack([np.stack([np.linspace(50.5, 60, 20), np.full(20, -3.5)], 1),
                      np.stack([60 + 10 * np.cos(np.radians(np.linspace(88, 2, 40))), -13.5 + 10 * np.sin(np.radians(np.linspace(88, 2, 40)))], 1)])
    assert manoeuvre(g, (50.0, -3.5), 0.0, hist, turn, CFG).label == 'turn_right'
    change = np.stack([np.linspace(50.5, 80, 60), np.linspace(-3.5, 0.0, 60)], 1)
    assert manoeuvre(g, (50.0, -3.5), 0.0, hist, change, CFG).label == 'lane_left'
    drift = np.stack([np.linspace(50.5, 80, 60), np.full(60, -2.2)], 1)      # 1.3 m left but still in A2/B2
    assert manoeuvre(g, (50.0, -3.5), 0.0, hist, drift, CFG).label == 'straight'
    off = np.stack([np.linspace(50.5, 80, 60), np.linspace(-3.5, -30, 60)], 1)
    assert manoeuvre(g, (50.0, -3.5), 0.0, hist, off, CFG).label is None
    assert manoeuvre(g, (50.0, -3.5), 0.0, hist, np.full((60, 2), np.nan), CFG).label is None
    assert wrap_deg(190.0) == -170.0
```

- [ ] **Step 3: Run** → FAIL (`No module named doplan_lang.lanes`).

- [ ] **Step 4: Implement `lanes.py`**

```python
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
        self._owner = np.concatenate([np.full(len(lane.xy), i) for i, lane in enumerate(lanes.values())]) if pts else np.zeros(0, int)
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

    def assign(self, point, tail=None, heading: float | None = None, cfg: Config = Config()) -> LanePose | None:
        """The lane under `point` (spec 4.1): polygon containment first, then the lane whose centreline the
        `tail` points follow best; `heading` (radians) rejects lanes heading the other way."""
        p = np.asarray(point, dtype=np.float64)
        if not np.all(np.isfinite(p)):
            return None
        cands = self.candidates(p, cfg.lane_assign_m)
        if not cands:
            return None
        inside = [c for c in cands if self.lanes[c].polygon is not None and self.lanes[c].polygon.contains_point(p)]
        if inside:
            cands = inside
        poses = {c: tuple(float(x[0]) for x in self.lanes[c].project(p)) for c in cands}
        if heading is not None:
            cands = [c for c in cands if abs(float(wrap_deg(np.degrees(poses[c][2] - heading)))) <= cfg.lane_heading_deg]
            if not cands:
                return None
        tail = None if tail is None else np.asarray(tail, dtype=np.float64)
        if tail is not None and len(tail) >= 2 and np.all(np.isfinite(tail)):
            best = min(cands, key=lambda c: float(self.lanes[c].project(tail)[1].mean()))
        else:
            best = min(cands, key=lambda c: poses[c][1])
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
```

- [ ] **Step 5: Run** `test_lanes.py` → PASS. Note in `test_assign_uses_polygon_then_tail` the wrong-way assertion reduces to `is None`; simplify it to `assert g.assign((30.0, 0.4), heading=np.pi) is None`.

- [ ] **Step 6: Move the unpickler.** In `scene_paths.py` delete `_MapUnpickler` and `load_map_features`, add `from .lanes import load_map_features`; in `test_scene_report.py` the import stays `from doplan_lang.scene_paths import ... load_map_features` (re-exported). Run the suite → PASS.

---

### Task 3: Compatibility from labels (`geometry.py`)

**Files:**
- Modify: `scripts/doplan_lang/geometry.py`
- Test: `scripts/doplan_lang/tests/test_geometry.py`

- [ ] **Step 1: Failing test** (append)

```python
def test_compat_table_from_labels():
    import numpy as np
    from doplan_lang.config import INTENTS, Config
    from doplan_lang.geometry import compat, compat_table_from_labels
    cfg = Config()
    geometric = np.full((3, len(INTENTS)), 0.5)
    t = compat_table_from_labels(['lane_right', None, 'straight'], [False, True, True], geometric, cfg)
    i = {k: INTENTS.index(k) for k in INTENTS}
    assert t[0, i['lane_right']] == 1.0 and t[0, i['lane_left']] == cfg.compat_floor and t[0, i['other']] == 1.0
    assert t[0, i['stop']] == cfg.compat_floor and t[2, i['stop']] == 1.0 and t[2, i['straight']] == 1.0
    assert np.all(t[1, [i['turn_left'], i['lane_right'], i['straight']]] == 0.5) and t[1, i['stop']] == 1.0   # None: geometric row + stop flag
    feats = np.zeros((3, 3))
    c_map = compat(feats, {'lane_right': 1.0}, cfg, labels=['lane_right', None, 'straight'], stops=[False, True, True])
    assert c_map[0] == 1.0 and c_map[2] == cfg.compat_floor
    c_geo = compat(feats, {'lane_right': 1.0}, Config(compat_source='geometry'), labels=['lane_right', None, 'straight'], stops=[False, True, True])
    assert c_geo[0] == c_geo[2]
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (replace `compat`, add the new function):

```python
def compat_table_from_labels(labels, stops, geometric: np.ndarray, cfg: Config) -> np.ndarray:
    """Label table (spec 4.2): the label column is 1, the other lateral/straight columns the floor, `stop`
    follows the stop flag, `other` is 1. A mode without a label keeps its geometric row (plus the stop flag)."""
    table = np.array(geometric, dtype=np.float64, copy=True)
    i_stop, i_other = INTENTS.index('stop'), INTENTS.index('other')
    for k, (lab, stop) in enumerate(zip(labels, stops)):
        if lab is not None:
            table[k] = cfg.compat_floor
            table[k, INTENTS.index(lab)] = 1.0
            table[k, i_other] = 1.0
        if stop:
            table[k, i_stop] = 1.0
    return table


def compat(features: np.ndarray, probs: dict[str, float], cfg: Config, labels=None, stops=None) -> np.ndarray:
    """Expected compatibility of each mode under the intent distribution `probs`; lane labels when given and
    cfg.compat_source == 'map', else the v1 geometric table."""
    p = np.array([probs.get(k, 0.0) for k in INTENTS], dtype=np.float64)
    table = compat_table(features, cfg)
    if labels is not None and cfg.compat_source == 'map':
        table = compat_table_from_labels(labels, stops if stops is not None else [False] * len(labels), table, cfg)
    return table @ p
```

- [ ] **Step 4: Run the suite** → PASS.

---

### Task 4: Labels in re-ranking (`rerank.py`)

**Files:**
- Modify: `scripts/doplan_lang/rerank.py`
- Test: `scripts/doplan_lang/tests/test_rerank.py`

- [ ] **Step 1: Failing test** (append; the fake prediction's three modes are left arc, right arc, straight from the origin heading 0; place it on the synthetic map at (30, 0) in lane A1 so the right arc (radius 20) ends at (20, -20) + (30, 0) = (50, -20): off every lane → None; the left arc ends at (50, 20): off; the straight mode ends at (60, 0) on A1/B1. Use origin (30, -3.5) in A2 instead so the straight mode is 'straight' and a 2 m drift mode can be tested through a hand-built prediction.)

```python
def test_rerank_with_lane_graph_labels(tmp_path):
    from doplan_lang.lanes import LaneGraph
    from .conftest import synthetic_map_features
    pt = tmp_path / 'pt'; pt.mkdir()
    # hand-built prediction in lane A2 (y = -3.5), heading east: mode 0 drifts 2 m left but stays in A2's
    # polygon's neighbour region? no: A2 spans y in [-5.25, -1.75]; a 2 m drift ends at y = -1.5, inside A1.
    # mode 0: lane change into A1 (ends y = 0); mode 1: drift to y = -2.2 (still A2); mode 2: straight.
    xs = 30.5 + np.arange(1, 61) * 0.5
    modes = np.stack([np.stack([xs, np.linspace(-3.5, 0.0, 60)], 1), np.stack([xs, np.linspace(-3.5, -2.2, 60)], 1),
                      np.stack([xs, np.full(60, -3.5)], 1)]).astype(np.float32)
    gt = np.zeros((81, 10), dtype=np.float32)
    gt[:, 0] = np.concatenate([30.5 - np.arange(21, 0, -1) * 0.5, xs]); gt[:, 1] = -3.5; gt[:, 9] = 1.0
    torch.save({'scenario_id': 'S', 'pred_trajs': modes, 'pred_scores': np.array([0.3, 0.4, 0.3], dtype=np.float32),
                'past_det_scores': np.float32(1.0), 'object_id': 'ego', 'object_type': 1, 'gt_trajs': gt,
                'track_index_to_predict': 0, 'current_time_index': 0}, pt / 'S___ego___20___.pt')
    pairs = pd.DataFrame([dict(zip(PAIR_COLUMNS, ['S___ego___20___.pt', 'S', 'logA', 20, 40, 1000, 'logA:0',
                                                  'a1', 'logA', 'switch one lane to the left', 'switch one lane to the left', 'lane_left']))])
    graph = LaneGraph.from_map_features(synthetic_map_features())
    df = rerank_pairs(pairs, pt, Config(), lambda c: one_hot(rule_intent(c)), graph_for=lambda sid: graph)
    r = df.iloc[0]
    assert (r.label_0, r.label_1, r.label_2) == ('lane_left', 'straight', 'straight') and r.fut_label == 'straight'
    assert r.lane_start == 'A2' and r.n_unlabelled == 0 and not r.stop_2
    assert r.old_top1 == 1 and r.new_top1 == 0 and r.new_top1_label == 'lane_left' and r.old_top1_label == 'straight'
    # without a graph: v1 behaviour, the 1.3 m drift mode is not a lane change and nothing is labelled
    df0 = rerank_pairs(pairs, pt, Config(), lambda c: one_hot(rule_intent(c)))
    assert df0.iloc[0].label_0 is None or pd.isna(df0.iloc[0].label_0)
    assert df0.iloc[0].n_unlabelled == 3
```

The fake prediction must pass the gate: the endpoints spread 3.5 m laterally, under the 4 m gate... Set mode 0 to end at y = +0.5 (4 m spread) so the gate opens: `np.linspace(-3.5, 0.5, 60)`; A1's polygon spans [-1.75, 1.75], so y = 0.5 is in A1. Use that.

- [ ] **Step 2: Run** → FAIL (`graph_for`). **Step 3: Implement.** In `rerank.py`:

```python
from .lanes import LaneGraph, mode_manoeuvres
```

and replace `rerank_pairs` by:

```python
def rerank_pairs(pairs: pd.DataFrame, pt_dir: Path, cfg: Config, intent_fn: IntentFn,
                 out_pt_dir: Path | None = None, graph_for=None) -> pd.DataFrame:
    """Re-rank every (prediction, annotation) pair. `intent_fn(clause) -> {intent: prob}`;
    `graph_for(scenario_id) -> LaneGraph` enables the lane labels (spec 4.3); None keeps v1 behaviour.

    Adds per pair: p_<intent>, intent (argmax), gated, lat_k/dpsi_k/path_k, label_k/stop_k, s_old_k/s_new_k,
    old_top1/new_top1/cf_top1 with their endpoint lateral offsets and labels, fut_lat/fut_dpsi/fut_path/fut_valid,
    fut_label/fut_stop, lane_start and n_unlabelled. `s_old_k` are MTR's scores as written; `s_new_k` are
    normalised. With out_pt_dir, writes a copy of each prediction with pred_scores replaced.
    """
    if out_pt_dir is not None:
        out_pt_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, tuple] = {}
    rows: dict[int, dict] = {}
    order = pairs.reset_index(drop=True)
    seq = order.sort_values(['scenario_id', 'pred_file'], kind='stable') if graph_for is not None else order
    for idx, r in zip(seq.index, seq.itertuples(index=False)):
        if r.pred_file not in cache:
            pred = load_prediction(pt_dir / r.pred_file)
            feats = mode_features(pred['pred_trajs'], pred['gt_trajs'], cfg.past_len)
            man, fut_man = mode_manoeuvres(pred, graph_for(r.scenario_id) if graph_for is not None else None, cfg)
            cache[r.pred_file] = (pred, feats, future_features(pred['gt_trajs'], cfg.past_len),
                                  gate_open(pred['pred_trajs'], feats, pred['pred_scores'], cfg), man, fut_man)
        pred, feats, fut, gated, man, fut_man = cache[r.pred_file]
        labels = [m.label for m in man] if graph_for is not None else None
        stops = [m.stop for m in man] if graph_for is not None else None
        probs = intent_fn(r.first_clause)
        probs_cf = intent_fn(swap_left_right(r.first_clause))
        old = np.asarray(pred['pred_scores'], dtype=np.float64)
        new = fuse(old, compat(feats, probs, cfg, labels, stops), cfg.lam) if gated else old / old.sum()
        new_cf = fuse(old, compat(feats, probs_cf, cfg, labels, stops), cfg.lam) if gated else old / old.sum()

        row = r._asdict()
        row.update({f'p_{k}': probs[k] for k in INTENTS})
        row['intent'] = max(probs, key=probs.get)
        row['gated'] = gated
        for k in range(len(old)):
            row[f'lat_{k}'], row[f'dpsi_{k}'], row[f'path_{k}'] = feats[k]
            row[f'label_{k}'], row[f'stop_{k}'] = man[k].label, man[k].stop
            row[f's_old_{k}'], row[f's_new_{k}'] = old[k], new[k]
        row['old_top1'], row['new_top1'], row['cf_top1'] = int(old.argmax()), int(new.argmax()), int(new_cf.argmax())
        for which in ('old', 'new', 'cf'):
            row[f'{which}_top1_lat'] = feats[row[f'{which}_top1'], 0]
            row[f'{which}_top1_label'] = man[row[f'{which}_top1']].label
        row['fut_lat'], row['fut_dpsi'], row['fut_path'], row['fut_valid'] = fut
        row['fut_label'], row['fut_stop'] = fut_man.label, fut_man.stop
        row['lane_start'] = man[0].start.lane_id if man[0].start is not None else None
        row['n_unlabelled'] = sum(m.label is None for m in man)
        rows[idx] = row

        if out_pt_dir is not None:
            out = dict(pred)
            out['pred_scores_mtr'] = np.asarray(pred['pred_scores'])
            out['pred_scores'] = new.astype(np.float32)
            out['annotation_id'] = r.annotation_id
            out['instruction'] = r.instruction
            torch.save(out, out_pt_dir / f'{Path(r.pred_file).stem}_{r.annotation_id}.pt')
    return pd.DataFrame([rows[i] for i in range(len(order))])
```

- [ ] **Step 4: Run the suite** → PASS (existing re-rank tests pass `graph_for=None`).

---

### Task 5: Lane compliance (`evaluate.py`)

**Files:**
- Modify: `scripts/doplan_lang/evaluate.py`
- Test: `scripts/doplan_lang/tests/test_evaluate.py`

- [ ] **Step 1: Failing test** (append)

```python
def test_lane_compliance_frame_and_summary():
    import numpy as np
    import pandas as pd
    from doplan_lang.config import Config
    from doplan_lang.evaluate import lane_compliance_frame, mirror_intent, summarize
    base = dict(pred_file='p', moment='m', log_name='l', annotation_id='a', gated=True, intent='lane_left', fut_valid=60,
                ade_old=1.0, ade_new=1.0, fde_old=1.0, fde_new=1.0, min_ade=1.0, min_fde=1.0, fut_lat=2.0,
                old_top1_lat=0.0, new_top1_lat=2.0, cf_top1_lat=-2.0, label_0='straight', label_1='lane_left', label_2=None)
    rows = [dict(base, rule_intent='lane_left', fut_label='lane_left', old_top1_label='straight', new_top1_label='lane_left', cf_top1_label='lane_right'),
            dict(base, rule_intent='lane_left', fut_label='straight', old_top1_label='straight', new_top1_label='lane_left', cf_top1_label='lane_right'),  # driver did not: excluded
            dict(base, rule_intent='turn_right', fut_label='turn_right', old_top1_label='turn_right', new_top1_label='straight', cf_top1_label='straight'),
            dict(base, rule_intent='stop', fut_label='straight', old_top1_label='straight', new_top1_label='straight', cf_top1_label='straight')]  # not lateral
    df = pd.DataFrame(rows)
    comp = lane_compliance_frame(df, Config())
    assert len(comp) == 2 and list(comp.old_ok) == [False, True] and list(comp.new_ok) == [True, False] and list(comp.cf_ok) == [True, False]
    assert mirror_intent('lane_left') == 'lane_right' and mirror_intent('turn_right') == 'turn_left'
    m = summarize(df, Config(bootstrap_n=10))
    assert m['gated']['lane_compliance_n'] == 2 and m['gated']['lane_compliance_before'] == 0.5 and m['gated']['lane_compliance_after'] == 0.5
    assert abs(m['label_coverage'] - 2 / 3) < 1e-9 and m['fut_label_counts']['straight'] == 2
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** Add after `side_sign`:

```python
def mirror_intent(intent: str) -> str:
    return intent.replace('_left', '_X').replace('_right', '_left').replace('_X', '_right')


def lane_compliance_frame(df: pd.DataFrame, cfg: Config, intent_col: str = 'rule_intent') -> pd.DataFrame:
    """Lane compliance (spec 4.4): pairs with a lateral rule intent, a full future and a recorded future whose
    lane label IS that intent; success when the top-1's lane label is the intent."""
    if 'fut_label' not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + ['old_ok', 'new_ok', 'cf_ok'])
    sub = df[df[intent_col].isin(LATERAL_INTENTS) & (df.fut_valid >= FUTURE_STEPS) & (df.fut_label == df[intent_col])].copy()
    sub['old_ok'] = sub.old_top1_label == sub[intent_col]
    sub['new_ok'] = sub.new_top1_label == sub[intent_col]
    sub['cf_ok'] = sub.new_ok & (sub.cf_top1_label == sub[intent_col].map(mirror_intent))
    return sub
```

In `summarize`, after `out['intent_disagrees_with_rules_fraction'] = ...` add:

```python
    label_cols = [c for c in df.columns if c.startswith('label_')]
    out['label_coverage'] = _mean(df[label_cols].notna().to_numpy().ravel().astype(float)) if label_cols and len(df) else math.nan
    out['fut_label_counts'] = {str(k): int(v) for k, v in df.fut_label.value_counts(dropna=False).items()} if 'fut_label' in df.columns and len(df) else {}
```

and inside the subset loop, after the v1 compliance block:

```python
        lc = lane_compliance_frame(sub, cfg)
        m['lane_compliance_n'] = int(len(lc))
        m['lane_compliance_before'] = _mean(lc.old_ok.astype(float))
        m['lane_compliance_after'] = _mean(lc.new_ok.astype(float))
        m['lane_counterfactual_consistency'] = _mean(lc.cf_ok.astype(float))
        m['lane_compliance_diff_ci'] = bootstrap_diff(lc.old_ok.astype(float), lc.new_ok.astype(float),
                                                      cfg.bootstrap_n, cfg.seed, groups=lc.moment)
```

`fut_label_counts` keys may be `'nan'` for missing labels; that is fine (JSON-safe strings). In `write_markdown`: a third table

```python
    lane_lines = ['| variant | subset | n | before | after | diff [95% CI] | counterfactual |',
                  '|---|---|---|---|---|---|---|']
    ...  # inside the loop, next to comp_lines.append:
            lci = s['lane_compliance_diff_ci']
            lane_lines.append(f"| {variant} | {subset} | {s['lane_compliance_n']} | {_fmt(s['lane_compliance_before'])} | "
                              f"{_fmt(s['lane_compliance_after'])} | {_fmt(lci['mean'])} [{_fmt(lci['lo'])}, {_fmt(lci['hi'])}] | "
                              f"{_fmt(s['lane_counterfactual_consistency'])} |")
    ...
    lines += ['', 'Lane compliance (population: lateral instructions whose recorded future, by the lane graph, '
                  'performs the instructed manoeuvre within 6 s; success: the top-1 does too):', ''] + lane_lines
    lines += ['', f"lane labels cover {_fmt(first['label_coverage'])} of (pair, mode) rows; recorded-future labels: "
                  + ', '.join(f'{k} {v}' for k, v in first['fut_label_counts'].items())]
```

and the header line gains `f'compatibility from {cfg.compat_source}'`.

- [ ] **Step 4: Run the suite** → PASS.

---

### Task 6: Variants and graphs in `run_all.py`; labels in the scene report

**Files:**
- Modify: `scripts/doplan_lang/run_all.py`, `scripts/doplan_lang/scene_report.py`, `scripts/doplan_lang/scene_paths.py`
- Test: `scripts/doplan_lang/tests/test_scene_report.py` (label column), `test_integration.py` (coverage)

- [ ] **Step 1: run_all.** Add `from .lanes import LaneGraph, load_map_features`, the argument `p.add_argument('--compat', choices=('map', 'geometry'), default=None, help='compatibility source for every variant (default Config.compat_source)')`, apply `cfg = replace(cfg, compat_source=a.compat)` when given, and replace the variants block:

```python
    graphs: dict[str, LaneGraph] = {}

    def graph_for(sid: str) -> LaneGraph:
        if sid not in graphs:
            if len(graphs) >= 2:
                graphs.pop(next(iter(graphs)))
            graphs[sid] = LaneGraph.from_map_features(load_map_features(a.scenarionet, sid))
        return graphs[sid]

    clf_fn = lambda c: predict_proba(clf, kv, c)   # noqa: E731
    variants = {'classifier': (clf_fn, cfg, 'reranked.csv'),
                'rules_only': (lambda c: one_hot(rule_intent(c)), cfg, 'reranked_rules.csv')}
    if cfg.compat_source == 'map':
        variants['classifier_geometry'] = (clf_fn, replace(cfg, compat_source='geometry'), 'reranked_geometry.csv')
    metrics = {}
    for name, (fn, vcfg, fname) in variants.items():
        is_main = name == 'classifier'
        df = rerank_pairs(pairs, pt_dir, vcfg, fn, (None if a.no_pt else a.out / 'reranked_pt') if is_main else None,
                          graph_for=graph_for)
        df = add_errors(df, pt_dir, vcfg)
        df.to_csv(a.out / fname, index=False)
        metrics[name] = summarize(df, vcfg)
        if is_main:
            print(f'lane labels cover {metrics[name]["label_coverage"]:.3f} of (pair, mode) rows; '
                  f'{int(df.n_unlabelled.sum())} unlabelled modes over {len(df)} pairs')
            plot_examples(df, pt_dir, a.out / 'plots', vcfg)
```

- [ ] **Step 2: scene report.** In `scene_report.py` the moments table gets a column `labels (MTR top-1 / fused / recorded)` built as

```python
def _lab(row, which):
    v = row.get(f'{which}_label')
    v = '-' if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    return v + ('+stop' if row.get(f'{which}_stop') in (True, 'True') else '')
```

printed as `f'{_lab(row, "old_top1")} / {_lab(row, "new_top1")} / {_lab(row, "fut")}'`. `_mtr_only_row` gains the parameter `graph` and fills `old_top1_label/new_top1_label/fut_label/old_top1_stop/new_top1_stop/fut_stop` from `mode_manoeuvres(pred, graph, cfg)`; `scene_table(moments, reranked, pt_dir, cfg, graph_for=None)` passes `graph_for(m.scenario_id)` when given; `main` builds `graph_for` exactly as in run_all (copy the four lines). `rerank.py` does not write `old_top1_stop`; add `row[f'{which}_top1_stop'] = man[row[f'{which}_top1']].stop` to the loop in Task 4 so the columns exist in reranked.csv. In `scene_paths.changed_moments` copy `old_top1_label`, `new_top1_label`, `fut_label` from the row when present (else None) and add them to `CHANGED_COLUMNS`; `changed_markdown` adds a column `labels orig -> new (recorded)`.

- [ ] **Step 3: Tests.** In `test_scene_report.py::test_scene_table_and_report` add to the reranked row `'old_top1_label': 'straight', 'new_top1_label': 'turn_left', 'fut_label': 'turn_left', 'old_top1_stop': False, 'new_top1_stop': False, 'fut_stop': False` and assert `'| straight / turn_left / turn_left |' in text`. In `test_integration.py` add `assert metrics['classifier']['label_coverage'] > 0.8 and 'classifier_geometry' in metrics` (measured 0.86 on the toy log; the rest end off-lane or reversing).

- [ ] **Step 4: Run the suite** → PASS (the integration test needs the toy data; it runs on this machine).

---

### Task 7: Toy simulator (`toysim.py`)

**Files:**
- Create: `scripts/doplan_lang/toysim.py`
- Test: `scripts/doplan_lang/tests/test_toysim.py`

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pandas as pd

from doplan_lang.config import Config
from doplan_lang.lanes import LaneGraph
from doplan_lang.toysim import AgentSet, boxes_overlap, score_execution, simulate

from .conftest import synthetic_map_features

CFG = Config()


def test_tracker_follows_straight_and_arc():
    plan = np.stack([np.arange(1, 61) * 0.8, np.zeros(60)], 1)                     # 8 m/s east
    ex = simulate(plan, (0.0, 0.0), 0.0, 8.0, CFG)
    assert ex['xy'].shape == (60, 2) and np.sqrt(np.mean(np.sum((ex['xy'] - plan) ** 2, axis=1))) < 0.3
    th = np.linspace(0, np.pi / 2, 61)[1:]                                          # quarter circle, radius 20, 5.2 m/s
    arc = np.stack([20 * np.sin(th), 20 * (1 - np.cos(th))], 1)
    ex = simulate(arc, (0.0, 0.0), 0.0, 5.2, CFG)
    assert np.sqrt(np.mean(np.sum((ex['xy'] - arc) ** 2, axis=1))) < 0.5 and abs(ex['heading'][-1] - np.pi / 2) < 0.15
    assert np.all(np.abs(ex['kappa']) <= CFG.sim_max_curvature + 1e-9) and np.all(np.abs(ex['accel']) <= CFG.sim_max_accel + 1e-9)


def test_tracker_does_not_reverse_on_creeping_tail():
    plan = np.stack([np.concatenate([np.linspace(0.5, 20, 50), [20.4, 20.1, 20.2, 20.1, 20.2, 20.1, 20.2, 20.1, 20.0, 19.9]]), np.zeros(60)], 1)
    ex = simulate(plan, (0.0, 0.0), 0.0, 4.0, CFG)
    assert np.all(np.diff(ex['xy'][:, 0]) >= -1e-9) and ex['speed'][-1] < 1.5


def test_boxes_overlap():
    c2 = np.array([[3.0, 0.0], [10.0, 0.0], [0.0, 2.0]])
    hit = boxes_overlap((0.0, 0.0), 0.0, 5.0, 2.0, c2, np.zeros(3), np.full(3, 4.0), np.full(3, 2.0))
    assert list(hit) == [True, False, True]
    rot = boxes_overlap((0.0, 0.0), 0.0, 5.0, 2.0, np.array([[4.0, 0.0]]), np.array([np.pi / 2]), np.array([4.0]), np.array([1.0]))
    assert list(rot) == [True]                                                     # rotated box reaches x = 3.5


def test_score_execution_on_synthetic_map():
    g = LaneGraph.from_map_features(synthetic_map_features())
    hist = np.stack([np.linspace(40, 50, 20), np.full(20, -3.5)], 1)
    change = np.stack([np.linspace(50.5, 80, 60), np.linspace(-3.5, 0.0, 60)], 1)
    ex = simulate(change, (50.0, -3.5), 0.0, 5.0, CFG)
    agents = AgentSet(positions=np.zeros((1, 100, 2)) + [200.0, 200.0], headings=np.zeros((1, 100)), valid=np.ones((1, 100)),
                      lengths=np.array([4.0]), widths=np.array([2.0]))
    s = score_execution(ex, change, agents, 20, g, 'lane_left', (50.0, -3.5), 0.0, hist, CFG)
    assert s['completion'] == 1.0 and not s['collision'] and s['offroad_fraction'] == 0.0 and s['tracking_rms_m'] < 0.5
    s2 = score_execution(ex, change, agents, 20, g, 'lane_right', (50.0, -3.5), 0.0, hist, CFG)
    assert s2['completion'] == 0.0
    blocker = AgentSet(positions=np.zeros((1, 100, 2)) + [60.0, -2.5], headings=np.zeros((1, 100)), valid=np.ones((1, 100)),
                       lengths=np.array([4.0]), widths=np.array([2.0]))
    assert score_execution(ex, change, blocker, 20, g, 'lane_left', (50.0, -3.5), 0.0, hist, CFG)['collision']
    off = np.stack([np.linspace(50.5, 80, 60), np.linspace(-3.5, -30, 60)], 1)
    s3 = score_execution(simulate(off, (50.0, -3.5), 0.0, 5.0, CFG), off, agents, 20, g, 'other', (50.0, -3.5), 0.0, hist, CFG)
    assert s3['offroad_fraction'] > 0.5 and np.isnan(s3['completion'])
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement `toysim.py`**

```python
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


def agents_from_tracks(tracks: dict, sdc_id: str) -> AgentSet:
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
    Returns xy [n, 2], heading [n], speed [n], kappa [n], accel [n] after each of the n steps."""
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
    c, h = np.atleast_2d(np.asarray(centre, dtype=np.float64)), np.atleast_1d(np.asarray(heading, dtype=np.float64))
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


def score_execution(ex: dict, plan_xy, agents: AgentSet, t0: int, graph: LaneGraph, intent: str, now_xy, now_heading: float,
                    hist_xy, cfg: Config, ego_length: float | None = None, ego_width: float | None = None) -> dict:
    """Outcome metrics of one executed trajectory (spec 4.6); `t0` is the scenario step of "now"."""
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
    offroad = float(np.mean([not graph.contains(p) for p in xy])) if len(graph) else math.nan
    man = manoeuvre(graph, now_xy, now_heading, hist_xy, xy, cfg)
    if intent in LATERAL_INTENTS or intent == 'straight':
        completion = float(man.label == intent) if man.label is not None else math.nan
    elif intent == 'stop':
        completion = float(final_speed(xy, cfg.sim_dt) < cfg.stop_speed_mps)
    else:
        completion = math.nan
    plan = np.asarray(plan_xy, dtype=np.float64)
    n = min(len(plan), len(xy))
    return {'collision': bool(collision), 'min_clearance_m': clearance if math.isfinite(clearance) else math.nan,
            'offroad_fraction': offroad, 'completion': completion, 'label': man.label, 'stop': bool(man.stop),
            'final_speed_mps': man.final_speed_mps,
            'max_lat_accel': float(np.max(np.abs(ex['speed'] ** 2 * ex['kappa']))), 'max_long_accel': float(np.max(np.abs(ex['accel']))),
            'tracking_rms_m': float(np.sqrt(np.mean(np.sum((xy[:n] - plan[:n]) ** 2, axis=1))))}
```

Then the driver, summary, markdown, plots and CLI (same file):

```python
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
        done: dict[tuple, dict] = {}
        for r in grp.itertuples(index=False):
            pred = load_prediction(pt_dir / r.pred_file)
            g = np.asarray(pred['gt_trajs'])
            now = g[cfg.past_len - 1]
            hist = g[:cfg.past_len]
            hist = hist[hist[:, -1] > 0, :2]
            v0 = float(np.linalg.norm(hist[-1] - hist[-2]) / cfg.sim_dt) if len(hist) >= 2 else 0.0
            fut = g[cfg.past_len:]
            plans = {'mtr_top1': ('mode', int(r.old_top1)), 'reranked_top1': ('mode', int(r.new_top1))}
            if int(r.fut_valid) >= len(fut):
                plans['recorded'] = ('recorded', -1)
            for which, key in plans.items():
                k = (r.pred_file,) + key
                if k not in done:
                    plan = np.asarray(pred['pred_trajs'])[key[1], :, :2] if key[0] == 'mode' else fut[:, :2]
                    if not np.all(np.isfinite(plan)):
                        done[k] = None
                        continue
                    ex = simulate(plan, now[:2], float(now[6]), v0, cfg)
                    done[k] = {**score_execution(ex, plan, agents, int(r.time_index), graph, r.rule_intent, now[:2], float(now[6]),
                                                 hist, cfg, L, W), 'plan_mode': key[1]}
                if done[k] is None:
                    skipped += 1
                    continue
                rows.append({'pred_file': r.pred_file, 'scenario_id': sid, 'log_name': r.log_name, 'moment': r.moment,
                             'annotation_id': r.annotation_id, 'rule_intent': r.rule_intent, 'gated': bool(r.gated),
                             'time_index': int(r.time_index), 'execution': which, **done[k]})
    if skipped:
        print(f'toysim: skipped {skipped} executions (scenario pickle or finite plan missing)')
    return pd.DataFrame(rows)


def summarize_toysim(df: pd.DataFrame, cfg: Config) -> dict:
    out = {'n_rows': int(len(df)), 'n_pairs': int(df.groupby(['pred_file', 'annotation_id']).ngroups) if len(df) else 0}
    for subset_name, sub in (('gated', df[df.gated]), ('all', df)):
        s = {}
        for ex in EXECUTIONS:
            e = sub[sub.execution == ex]
            comp = e.completion.dropna()
            s[ex] = {'n': int(len(e)), 'completion_n': int(len(comp)), 'completion': float(comp.mean()) if len(comp) else math.nan,
                     'collision': float(e.collision.mean()) if len(e) else math.nan,
                     'offroad': float((e.offroad_fraction > 0).mean()) if len(e) else math.nan,
                     'tracking_rms_m': float(e.tracking_rms_m.mean()) if len(e) else math.nan,
                     'lat_accel_p95': float(e.max_lat_accel.quantile(0.95)) if len(e) else math.nan}
        a = sub[sub.execution == 'mtr_top1'].set_index(['pred_file', 'annotation_id'])
        b = sub[sub.execution == 'reranked_top1'].set_index(['pred_file', 'annotation_id'])
        both = a.join(b, lsuffix='_a', rsuffix='_b', how='inner')
        diffs = {}
        for key, col in (('completion', 'completion'), ('collision', 'collision'), ('offroad', 'offroad_fraction')):
            x, y = both[f'{col}_a'], both[f'{col}_b']
            if key == 'offroad':
                x, y = (x > 0), (y > 0)
            ok = x.notna() & y.notna()
            diffs[key] = bootstrap_diff(x[ok].astype(float), y[ok].astype(float), cfg.bootstrap_n, cfg.seed, groups=both.moment_a[ok])
        s['diff_reranked_minus_mtr'] = diffs
        out[subset_name] = s
    return out


def write_toysim_markdown(m: dict, path: Path, cfg: Config) -> None:
    lines = ['# Toy simulator: outcome of the selected trajectories', '',
             f'Pure pursuit on a kinematic bicycle (dt {cfg.sim_dt} s, |a| <= {cfg.sim_max_accel} m/s^2, |kappa| <= '
             f'{cfg.sim_max_curvature} 1/m), agents replayed from the log (not reactive: collisions after the ego leaves '
             f'the recorded path are pessimistic). Completion: the executed path performs the instructed manoeuvre by the '
             f'lane graph (stop: final speed < {cfg.stop_speed_mps} m/s). Off-road: any step outside every lane polygon.', '',
             '| subset | execution | n | completion (n) | collision | off-road | tracking RMS (m) | lat accel p95 (m/s^2) |',
             '|---|---|---|---|---|---|---|---|']
    f = lambda x: 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.3f}'   # noqa: E731
    for subset in ('gated', 'all'):
        for ex in EXECUTIONS:
            s = m[subset][ex]
            lines.append(f"| {subset} | {ex} | {s['n']} | {f(s['completion'])} ({s['completion_n']}) | {f(s['collision'])} | "
                         f"{f(s['offroad'])} | {f(s['tracking_rms_m'])} | {f(s['lat_accel_p95'])} |")
    lines += ['', '| subset | metric | reranked - MTR [95% CI] |', '|---|---|---|']
    for subset in ('gated', 'all'):
        for key, d in m[subset]['diff_reranked_minus_mtr'].items():
            lines.append(f"| {subset} | {key} | {f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}] |")
    lines += ['', f"{m['n_rows']} executions over {m['n_pairs']} pairs; CIs resample moments."]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_execution_examples(df: pd.DataFrame, scenarionet_dir: Path, pt_dir: Path, out_dir: Path, cfg: Config, n: int) -> list[Path]:
    """The n pairs with the largest change in completion or collision between the two executions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .scene_paths import _draw_map
    a = df[df.execution == 'mtr_top1'].set_index(['pred_file', 'annotation_id'])
    b = df[df.execution == 'reranked_top1'].set_index(['pred_file', 'annotation_id'])
    both = a.join(b, lsuffix='_a', rsuffix='_b', how='inner')
    both['change'] = (both.completion_b.fillna(0) - both.completion_a.fillna(0)).abs() + (both.collision_b.astype(float) - both.collision_a.astype(float)).abs()
    chosen = both.sort_values('change', ascending=False).head(n)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, (idx, r) in enumerate(chosen.iterrows()):
        pred_file, ann = idx
        sc = load_scenario(scenarionet_dir, str(r.scenario_id_a))
        pred = load_prediction(pt_dir / pred_file)
        g = np.asarray(pred['gt_trajs'])
        now = g[cfg.past_len - 1]
        hist = g[:cfg.past_len]
        hist = hist[hist[:, -1] > 0, :2]
        v0 = float(np.linalg.norm(hist[-1] - hist[-2]) / cfg.sim_dt) if len(hist) >= 2 else 0.0
        fig, ax = plt.subplots(figsize=(7, 6))
        xs, ys = list(hist[:, 0]), list(hist[:, 1])
        ax.plot(*hist.T, 'k-', lw=2.5, label='history')
        fut = g[cfg.past_len:]
        fut = fut[fut[:, -1] > 0, :2]
        if len(fut):
            ax.plot(*fut.T, color='0.2', ls='--', lw=2, label='recorded future')
        for which, mode, color in (('MTR top-1', int(r.plan_mode_a), 'tab:blue'), ('re-ranked top-1', int(r.plan_mode_b), 'tab:red')):
            plan = np.asarray(pred['pred_trajs'])[mode, :, :2]
            ex = simulate(plan, now[:2], float(now[6]), v0, cfg)
            ax.plot(*plan.T, color=color, ls=':', lw=1)
            ax.plot(*ex['xy'].T, color=color, lw=2.5, label=f'{which} executed (mode {mode})')
            xs += list(ex['xy'][:, 0]); ys += list(ex['xy'][:, 1])
        agents = agents_from_tracks(sc['tracks'], sc['metadata'].get('sdc_id')) if sc else None
        if agents is not None:
            c, h, ln, wd = agents.at(int(r.time_index_a) + len(plan))
            for corners in _corners(c, h, ln, wd):
                ax.fill(*np.vstack([corners, corners[:1]]).T, color='0.6', alpha=0.5, lw=0)
        xlim, ylim = (min(xs) - 10, max(xs) + 10), (min(ys) - 10, max(ys) + 10)
        if sc:
            _draw_map(ax, sc.get('map_features', {}), np.zeros(2), xlim, ylim)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f'{ann} "{r.rule_intent_a}" {pred_file}\ncompletion {r.completion_a} -> {r.completion_b}; collision {r.collision_a} -> {r.collision_b}; '
                     f'agents at the final step', fontsize=8)
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
    p.add_argument('--max-pairs', type=int, default=None)
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
```

- [ ] **Step 4: Run the suite** → PASS. Add a data-gated test: the recorded future of `3e1942159626505c___ego___40___.pt` executed on the scene-7 scenario has no collision and off-road fraction 0 (skip when `scenarionet_data/nuplan_scenes` is absent).

---

### Task 8: Data runs and README

- [ ] **Step 1: scene run.** Move the old scene outputs' CSVs aside is not needed (overwritten). Run:

```bash
PYTHONPATH=scripts python3 -m doplan_lang.run_all --scenarionet scenarionet_data/nuplan_scenes --nuplan nuplan_data \
    --csv nuplan_data/doplan/release_annotationsV1.csv --out outputs/doplan_scenes --no-pt \
    --exclude-log 2021.06.09.12.39.51_veh-26_05332_05540 --exclude-log 2021.07.16.21.17.55_veh-26_03772_03842 \
    > logs/doplan_scenes_run_v2.log 2>&1
PYTHONPATH=scripts python3 -m doplan_lang.scene_report --scenarionet scenarionet_data/nuplan_scenes --nuplan nuplan_data \
    --csv nuplan_data/doplan/release_annotationsV1.csv --run outputs/doplan_scenes --scenes outputs/doplan_scenes/scenes.csv \
    > logs/doplan_scenes_report_v2.log 2>&1
PYTHONPATH=scripts python3 -m doplan_lang.toysim --scenarionet scenarionet_data/nuplan_scenes \
    --reranked outputs/doplan_scenes/reranked.csv --out outputs/doplan_scenes/toysim > logs/doplan_scenes_toysim.log 2>&1
```

Expected: the 8.5 to 10 s rows no longer change the top-1 to the stopping mode under "take a right"; the recorded execution reports collision 0 and off-road 0.

- [ ] **Step 2: six-log run** (background, minutes):

```bash
PYTHONPATH=scripts python3 -m doplan_lang.run_all --scenarionet scenarionet_data/nuplan_doplan --nuplan nuplan_data \
    --csv nuplan_data/doplan/release_annotationsV1.csv --out outputs/doplan_lang --no-pt > logs/doplan_lang_run_v2.log 2>&1
PYTHONPATH=scripts python3 -m doplan_lang.toysim --scenarionet scenarionet_data/nuplan_doplan \
    --reranked outputs/doplan_lang/reranked.csv --out outputs/doplan_lang/toysim > logs/doplan_lang_toysim.log 2>&1
```

- [ ] **Step 3: README.** Section 2 tables: rows for `lanes.py` and `toysim.py`; test count. Section 4.2: a new paragraph "Lane-grounded re-run (2026-10-26)" with the lane compliance table and the toysim table, keeping the v1 numbers (now the `classifier_geometry` variant). Section 4.3: the scene-7 reading updated with the labels and the simulator outcome. Section 5.7: the `toysim` command and the `--compat geometry` switch. Section 6: `toysim.md` in the document list.

---

## Self-review

- Spec coverage: 4.1 Task 2; 4.2 Task 3; 4.3 Task 4; 4.4 Task 5; 4.5 and 4.7 Task 6; 4.6 Task 7; 5 Task 1; 6 Task 8; 7 covered by the None/empty paths in Tasks 2, 4, 7; 8 the test steps.
- Deviation from the spec, recorded there: a mode without a label still gets the stop flag (4.2), since the stop test is more robust than v1's path-length column.
- Names used consistently: `LaneGraph.assign/route/contains`, `label`, `final_speed`, `manoeuvre`, `mode_manoeuvres`, `compat_table_from_labels(labels, stops, cfg)`, `compat(features, probs, cfg, labels, stops)`, `rerank_pairs(..., graph_for, manoeuvre_cache)`, `lane_compliance_frame`, `mirror_intent`, `LaneGraph.off_lane`, `AgentSet.at`, `agents_from_tracks`, `simulate`, `boxes_overlap`, `completion_of`, `score_execution` (intent-independent), `run`, `summarize_toysim`, `write_toysim_markdown`, `plot_execution_examples`.

Execution notes (2026-10-27): all tasks done; three defects found on the scene run and fixed before the data runs: polygon containment as a hard filter mislabelled intersection endpoints (now a score bonus plus a direction-of-travel test), the geometric fallback re-admitted the reversing-tail artefact when a map exists (unlabelled modes now get the floor), and completion was cached per execution although it depends on the instruction (now derived per pair). "Off-road" became "off-lane" with a 2 m centreline margin because nuPlan lane polygons leave unmapped strips between lanes.
