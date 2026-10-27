# Lane-grounded manoeuvre labels, lane compliance and a toy simulator: design

Date: 2026-10-26. Follows the v1 design (`2026-10-23-doplan-word2vec-mtr-rerank-design.md`), whose
section numbers are referenced as "v1 4.5" etc. Approved direction (conversation of 2026-10-26):
map-grounded metric first, then a toy simulator built on the same lane model.

## 1. Goal

Two things the v1 evaluation cannot do, both shown by the runs of 2026-10-24 and 2026-10-26:

1. Say what a trajectory *does*. v1 describes a mode by its endpoint lateral offset and final
   heading in the ego frame (v1 4.5). A stopping mode that creeps back 0.8 m reads as a 175 deg
   turn; a 2 m drift reads as a lane change whether or not there is a lane there. Scene 7 of the
   scene run "improved" mostly through that artefact.
2. Say whether a worse ADE/FDE is acceptable. ADE compares with one recorded driver at one
   instant, so it cannot separate premature execution (the driver was still going straight) from
   an equally valid path or a timing offset of a second.

This iteration (a) replaces the ego-frame thresholds by the lane graph of the ScenarioNet
scenario as the judge of what a trajectory does, for both the compatibility table and the
compliance metric, and (b) adds a minimal simulator that executes a selected trajectory with a
tracking controller among log-replayed agents and scores the outcome: completion of the
instructed manoeuvre, off-road, collision, comfort, tracking feasibility. Nothing in the Docker
images changes; MTR is not re-run.

Not goals: reactive agents, re-planning with MTR in the loop, stage tracking, traffic lights.

## 2. Inputs and verified facts (2026-10-26)

- ScenarioNet pickle per scenario (`<db>/<subdir>/sd_nuplan_v1.1_<id>.pkl`), readable on the host
  with the tolerant unpickler already in `scene_paths.py`. `map_features` of scene 7: 369
  `LANE_SURFACE_STREET`, 461 `LANE_SURFACE_UNSTRUCTURE` (intersection connectors in the nuPlan
  converter), 370 road lines, 46 road edges, 35 crosswalks. Every lane has `polyline` (N, 2)
  centreline at about 0.25 m spacing, `polygon` (M, 2), and `entry_lanes`, `exit_lanes`,
  `left_neighbor`, `right_neighbor` as lists of string ids. All 1160 neighbour references and
  909 of 922 exit references resolve to lanes in the file; the rest leave the map window.
  Lanes carry no width.
- `tracks`: per agent `state.position` (T, 3), `heading` (T,), `velocity` (T, 2), `valid` (T,),
  `length`/`width`/`height` (T, 1), T = 201 at 10 Hz; `metadata.sdc_id` names the ego;
  `metadata.ego_vehicle_parameters`: wheel_base 3.089, length 5.176, width 2.297,
  rear_length 1.127, front_length 4.049 (Pacifica).
- Map, tracks and predictions share the scenario frame (nuPlan metres minus the ego centre at
  step 0; verified to under 1 mm on 80 predictions). `gt_trajs[past_len - 1]` is "now" with the
  heading in column 6; `pred_trajs` is (5, 60, 2).
- Host libraries: numpy, scipy (`cKDTree`), matplotlib (`path.Path` for point-in-polygon),
  pandas, torch (CPU). No shapely; none will be added.

## 3. Architecture

```
scenario pickle ──► lanes.LaneGraph ──┬─► rerank: per-mode label + stop flag ──► geometry.compat_table_from_labels
                                      │            (fallback: v1 geometric table where a mode has no label)
                                      ├─► evaluate: lane compliance (population and success by label)
                                      └─► toysim: tracker on a kinematic bicycle + replayed agents
                                                  ──► completion / off-road / collision / comfort / tracking
```

`lanes.py` is the one place that knows the map. `rerank.py`, `evaluate.py` and `toysim.py`
consume `Manoeuvre` records; `scene_report.py` prints the labels. `run_all.py` loads one graph
per scenario, lazily, and reports the v1 geometric compatibility as an ablation variant.

## 4. Component definitions

### 4.1 Lane graph (`lanes.py`)

`load_map_features(scenarionet_dir, scenario_id)` moves here from `scene_paths.py` (which
re-imports it).

`LaneGraph.from_map_features(map_features)`: lanes are the features whose type starts with
`LANE_SURFACE` and whose polyline has at least two points. Per lane: `xy` (N, 2) float64,
cumulative arc length `s`, per-point heading (direction of the segment ahead; the last point
repeats the previous), a `matplotlib.path.Path` of the polygon or None, and `exits`, `entries`,
`left`, `right` filtered to ids present in the graph. One `cKDTree` over all polyline points
with a parallel array of (lane index, point index). An empty graph answers None to every query.

`assign(point, tail=None, heading=None) -> LanePose | None` with `LanePose(lane_id, s, dist,
heading)`:

1. Candidates: the lanes with a centreline point within `lane_assign_m` (3.0 m). None: return None.
2. Direction: `heading` when given (the ego heading, for the start lane); otherwise the direction of
   travel of the `tail` points, if they moved at least 1 m. Candidates whose lane heading at the
   projection differs by `lane_heading_deg` (90) or more are dropped; none left: return None. This
   is what keeps a crossing or oncoming connector from claiming an endpoint inside an intersection.
3. Score: the mean projection distance of the `tail` points (at least two) to the lane, else the
   point's own distance, minus `lane_inside_bonus_m` (1.0) when the lane polygon contains the
   point. Pick the minimum. The tail asks which lane the last 2 s of the trajectory followed;
   containment is a bonus and not a filter because intersection connector polygons are wide and
   overlap (found on the toy log: a hard containment filter mislabelled 15% of the modes).

`route(a, b) -> Route | None` with `Route(moves, dpsi_deg)`: Dijkstra from lane `a` to lane `b`
over moves `F` (exit lane, cost 1.0), `L` and `R` (neighbour, cost `lane_lateral_cost` 1.5),
bounded by `lane_route_max_cost` (8.0). Same lane: empty moves. `dpsi_deg` is the lane heading
at `b` minus the lane heading at `a`, wrapped to [-180, 180). The heading change is read off the
lanes, never off the trajectory, so a creeping or jittery tail cannot invent a turn.

`label(route) -> str | None`: `|dpsi| >= turn_heading_deg` (30, v1 value) gives `turn_left`
or `turn_right` by the sign; otherwise the net lateral count (`L` minus `R`) gives `lane_left`
(>= 1) or `lane_right` (<= -1); otherwise `straight`. A None route is None ("off route").

`manoeuvre(now_xy, now_heading, hist_xy, traj_xy, cfg) -> Manoeuvre(label, stop, start, end,
final_speed_mps)`: `start = assign(now, tail=last 2 s of history, heading=now_heading)`;
`end = assign(traj[-1], tail=traj[-20:])`; `final_speed` = net displacement over the last 0.5 s
of the trajectory divided by 0.5 s (net, not summed, so the creeping tail of a stopping mode
reads as slow); `stop = final_speed < stop_speed_mps` (1.0); `label = label(route(start, end))`
when both poses exist, else None. Applies unchanged to a mode, to the recorded future (valid
rows only) and to an executed simulator path.

### 4.2 Compatibility from labels (`geometry.py`)

`compat_table_from_labels(labels, stops, cfg) -> [modes, len(INTENTS)]`: for
a mode with label L, column L is 1.0 and the other four lateral/straight columns are
`compat_floor`; column `stop` is 1.0 if the mode stops else the floor; column `other` is 1.0.
A mode whose label is None (off every lane, reversing against the lane, or no route) gets the
floor on every lateral/straight column: with a map available it cannot be shown to perform any
of them, and letting the v1 row through was found to re-admit the reversing-tail artefact
(scene 7, 9.0 s). The v1 geometric table applies only when the scenario has no map at all. `compat(features,
probs, cfg, labels=None, stops=None)` uses the label table when `cfg.compat_source == 'map'` and
labels are given, else the v1 table. The v1 mixing with the intent distribution is unchanged.

### 4.3 Re-ranking (`rerank.py`)

`rerank_pairs(pairs, pt_dir, cfg, intent_fn, out_pt_dir=None, graph_for=None)`:
`graph_for(scenario_id) -> LaneGraph` (None means v1 behaviour, used by old tests). Per
prediction (cached as today) it also computes `Manoeuvre` for each mode and for the recorded
future. New columns per pair: `label_k`, `stop_k` for each mode, `fut_label`, `fut_stop`,
`old_top1_label`, `new_top1_label`, `cf_top1_label`, `lane_start` (lane id under the ego),
`n_unlabelled` (modes without a label). The gate is unchanged (v1 4.6, geometric fork test);
only the compatibility changes. Pairs are processed in `(scenario_id, pred_file)` order so each
graph is built once; `run_all` keeps the last two graphs and passes one `manoeuvre_cache`
(pred_file -> labels) to all variants, since the labels depend on neither the intent function nor
the compatibility source.

### 4.4 Lane compliance (`evaluate.py`)

Population: pairs whose rule intent is lateral, whose recorded future covers the full horizon,
and whose `fut_label` equals the rule intent (the driver did the instructed manoeuvre within
6 s, as judged by the lane graph). Success: `new_top1_label == rule_intent` (before:
`old_top1_label`). Counterfactual: success and `cf_top1_label` equals the left/right-mirrored
intent. Reported as `lane_compliance_n/before/after/diff_ci/counterfactual` next to the v1
lateral-offset compliance, which stays as is for comparison. `summarize` also reports label
coverage (fraction of modes with a label) and the distribution of `fut_label`. `metrics.md`
gains a third table, "lane compliance", and one coverage line.

### 4.5 Variants (`run_all.py`)

`classifier` (map compatibility, the new default), `classifier_geometry` (v1 compatibility,
same classifier: the ablation that isolates the map), `rules_only` (map compatibility, rule
intents). `--compat geometry` switches the default for all variants off (for the lambda sweep
reruns). A scenario without a pickle yields an empty graph: every mode is unlabelled, the v1
table applies, and the count is printed and stored in the metrics.

### 4.6 Toy simulator (`toysim.py`)

Purpose: outcome rather than agreement. The plan is one selected 6 s trajectory; a tracking
controller drives a kinematic bicycle along it among agents replayed from the log; the executed
path is scored. No re-planning, no reactive agents: the plan is fixed for 6 s and the other
agents do what they did in the log, so a collision after the ego leaves the recorded path is
pessimistic. The report says so.

Ego model: state (x, y, psi, v) at the vehicle centre, dt 0.1 s, 60 steps.
`x += v cos psi dt`, `y += v sin psi dt`, `psi += v kappa dt`, `v += a dt`, with
`|kappa| <= sim_max_curvature` (0.2 1/m, a 5 m radius) and `|a| <= sim_max_accel` (4 m/s^2).
Start: the recorded "now" position and heading, speed from the last 0.1 s of history.
Steering: pure pursuit on the plan with lookahead `clip(sim_lookahead_gain * v, 3, 10)` m
(`kappa = 2 sin(alpha) / lookahead`). Speed: P control toward the plan's speed at the same
step (plan speeds from finite differences, clipped at zero), gain 1.0 1/s, limited by
`sim_max_accel`. A plan point that lies behind the ego (dot product with the heading negative,
as in the creeping tail) is skipped, so a reversing tail is not tracked.

Replayed agents: every track other than the ego, at steps `t0 + 1 .. t0 + 60` where valid, as
oriented boxes from `length`, `width`, `heading`. The ego box is 5.176 x 2.297 around the
centre.

Per executed trajectory:
- `collision`: any step where the ego box and an agent box overlap (separating-axis test,
  vectorised over agents per step); `min_clearance_m`: the minimum over steps of the smallest
  centre-to-centre distance minus the two half-diagonals (a lower bound).
- `offlane_fraction`: steps where the ego centre lies inside no lane polygon and more than
  `offlane_m` (2.0) from every lane centreline (`LaneGraph.off_lane`). The margin matters: nuPlan
  lane polygons leave unmapped strips between adjacent lanes, and a top-1 that rides a lane edge
  is not off the road.
- `max_lat_accel`, `max_long_accel` from the executed states (`v^2 kappa`, `a`).
- `tracking_rms_m`: RMS distance between the executed and planned positions at equal steps.
  A plan the car cannot follow (a jump, a backwards step) shows here.
- `completion`: for a lateral or straight rule intent, `manoeuvre(now, history, executed).label
  == intent`; for `stop`, `final_speed < stop_speed_mps`; for `other`, not defined (NaN). It is
  the one metric that depends on the instruction, so it is derived per pair from the execution's
  label and final speed (`completion_of`); the other metrics are computed once per execution.

Three executions per pair: `mtr_top1` (mode `old_top1`), `reranked_top1` (mode `new_top1`) and
`recorded` (the recorded future, where it covers 60 steps). The recorded run is the sanity bar:
its collision and off-road rates should be near zero, and its tracking RMS states what the
controller itself costs. Executions are keyed by (pred_file, mode) and shared between pairs.

Outputs in `--out`: `toysim.csv` (one row per pair and execution), `toysim.md` with, per
execution and subset (gated, all): completion rate on lateral-and-straight pairs, collision
rate, off-lane rate (fraction of executions with any off-lane step), mean tracking RMS, 95th
percentile of lateral acceleration; before/after differences (`reranked_top1` minus `mtr_top1`)
with the v1 cluster bootstrap by moment. `cfg.n_example_plots` PNGs of the executions with the
largest completion or collision change, drawn on the lane geometry with the agents at the
final step. CLI: `python -m doplan_lang.toysim --scenarionet DIR --reranked reranked.csv --out
DIR [--max-pairs N --seed S]`.

### 4.7 Scene report (`scene_report.py`, `scene_paths.py`)

The moments table and the "changed moments" table gain the label and stop flag of the MTR
top-1, the re-ranked top-1 and the recorded future, so a change reads as `lane_left -> straight`
directly. `scene_paths.load_map_features` is imported from `lanes`.

## 5. Configuration (`config.py`)

Added to `Config`, all with the defaults above: `compat_source = 'map'`, `lane_assign_m = 3.0`,
`lane_heading_deg = 90.0` (renamed from the v1 value used by the geometric table, which becomes
`geom_lane_heading_deg`), `lane_tail_s = 2.0`, `lane_lateral_cost = 1.5`,
`lane_route_max_cost = 8.0`, `stop_speed_mps = 1.0`, `sim_dt = 0.1`, `sim_max_accel = 4.0`,
`sim_max_curvature = 0.2`, `sim_lookahead_gain = 0.8`, `sim_lookahead_min_m = 3.0`,
`sim_lookahead_max_m = 10.0`, `sim_speed_gain = 1.0`. The ego box and wheel base come from
`metadata.ego_vehicle_parameters` when present, else `sim_ego_length = 5.176`,
`sim_ego_width = 2.297`.

## 6. Data runs

1. Scene run (`scenarionet_data/nuplan_scenes`, 2 scenarios): `run_all`, `scene_report`,
   `toysim`, all into `outputs/doplan_scenes/`. Expected: the 8.5 to 10 s "turn_right" picks
   disappear (the stopping mode is labelled `straight` + stop), the 4 to 6 s lane_right cases
   remain, and the simulator reports the recorded future with no collision and no off-road step.
2. Six-log run (`scenarionet_data/nuplan_doplan`, 411 scenarios, 77,127 pairs): `run_all`
   into `outputs/doplan_lang/` (overwrites the v1 tables; the v1 numbers stay in the README
   under 4.2 and are reproduced by `classifier_geometry`), then `toysim` on the gated pairs.
   Cost: one pickle load per scenario (411 files, about 4 GB) and about 90,000 lane
   assignments; a few minutes. The simulator executes at most two modes per prediction plus the
   recorded future: about 25,000 runs of 60 steps with a vectorised collision test; minutes.
3. README: section 4 gains the lane compliance and the simulator outcome for both runs,
   section 2 lists `lanes.py` and `toysim.py`, section 5.7 shows the toysim command.

## 7. Failure handling

- No pickle for a scenario, or a pickle without lanes: empty graph, labels None, v1 table,
  counted in `n_unlabelled` and in the metrics; never silent.
- Unresolvable lane ids in exits or neighbours are dropped at graph build time.
- Non-finite coordinates in a trajectory: label None, simulator row skipped with a count.
- Simulator start speed from a history with fewer than two valid rows: zero.
- The alignment and gate code paths are untouched; their existing errors still raise.

## 8. Testing

Synthetic lane graphs built in the tests (no data needed): a straight three-lane road
(`A0` left, `A1` centre, `A2` right, 60 m long, neighbours linked) continuing into `B0..B2`,
plus a right-turn connector `C` from `A2` bending -90 deg and a left-turn connector `D` from
`A0` bending +90 deg, each with polygons 3.5 m wide.

- `lanes`: assign picks the containing lane; the tail disambiguates two overlapping connectors;
  routes and labels: `A1 -> A1` straight, `A1 -> B1` straight, `A1 -> A0` lane_left,
  `A1 -> A2` lane_right, `A2 -> C` turn_right, `A1 -> C` turn_right (heading beats the one `R`
  move), `A0 -> D` turn_left; a lane beyond the cost bound is None; an empty graph answers None;
  the creeping-tail trajectory from scene 7 (stored as a small fixture) is `stop` and not a turn.
- `geometry`: the label table has 1.0 in the label column and the floor elsewhere, `other`
  always 1, `stop` follows the flag, and a None label copies the geometric row.
- `rerank`: label columns are written; on a synthetic fork with the map, the lane_right
  instruction moves the top-1 to the mode that ends in `A2` and not to one that drifts 2 m
  without changing lane.
- `evaluate`: lane compliance population and rates on a hand-built frame, including the
  counterfactual.
- `toysim`: the tracker follows a straight plan and a 90 deg arc within 0.3 m RMS; a plan with
  a backwards tail is followed without reversing; the collision test is positive for two
  overlapping boxes and negative for boxes 1 m apart; off-road is detected outside every
  polygon; completion on the synthetic graph; the recorded future executed on the scene-7
  scenario (data-gated test) has no collision.
- Integration (data-gated, toy log): still passes and reports a label coverage above 0.8 (measured 0.86:
  the unlabelled modes end off every lane, or end travelling against the lane direction, which is
  a reversing mode and rightly has no label).

## 9. Out of scope

Reactive agents, re-planning with MTR in the loop, lane widths from the map (polygons serve),
traffic lights (`dynamic_map_states` is empty in the scene run), multi-clause instructions,
stage tracking, Singapore left-hand traffic in the lane labels (a U-turn still maps to
`turn_left` through the rules, v1 9).
