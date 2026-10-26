# doPlan word2vec + MTR mode re-ranking: design

Date: 2026-10-23. Status: approved in brainstorming, awaiting written review.

## 1. Goal

Extract word2vec information from the doPlan passenger-instruction text and use it to
re-rank the 5 trajectory modes that the pretrained Valeo4Cast MTR model predicts for the
ego vehicle on nuPlan logs. No MTR retraining. The deliverable is a host-side Python
package, a trained word2vec model plus intent classifier, a re-ranked set of predictions
for six annotated nuPlan mini-split logs, and a metrics report comparing top-1 accuracy
and directional compliance before and after re-ranking.

Decisions already taken (brainstorming 2026-10-23):

- Fusion = post-hoc re-ranking of MTR mode probabilities (no fusion head, no encoder change).
- word2vec is trained on the doPlan text only (gensim skip-gram); no pretrained vectors.
  The embedding is stored as a plain word-to-vector table so pretrained vectors can be
  swapped in later without code changes.
- The embedding reaches geometry through an intent classifier (logistic regression on the
  mean word vector) trained on weak keyword labels, not through raw cosine similarity,
  because in a corpus this small "left" and "right" are near-synonyms.
- Temporal relevance is handled by decision-point gating: re-rank only at prediction
  times where MTR's modes fork. Stage tracking across clauses is a later iteration.
- Data: the annotated Singapore and Pittsburgh mini-split logs.
- Success = both top-1 ADE/FDE and directional compliance, before vs after.

## 2. Inputs and verified facts

- doPlan annotations: `outputs/doPlan_submissions/release_annotationsV1.csv` in
  https://github.com/Mi3-Lab/doPlan (6621 rows; 5198 with text; columns annotation_id,
  clip_id, start_frame, end_frame, window_duration_sec, passenger_instruction,
  referential_label, is_blank, annotator_id). Frames are 5 per second from log start.
  Vocabulary about 1000 words, 83k tokens, median 12 words per instruction.
- MTR output per ego prediction (`<scenario>___ego___<t>___.pt`): `pred_trajs` [5, 60, 2]
  (6 s at 10 Hz, ScenarioNet world frame), `pred_scores` [5], `gt_trajs` [81, 10]
  (21 past + 60 future rows: x, y, z, l, w, h, heading, vx, vy, valid; row 20 is the
  prediction time), `current_time_index`. Predictions exist every 5 frames (0.5 s).
- ScenarioNet summary per scenario: `log_name`, `scenario_token` (= a nuPlan lidar_pc
  token), `scenario_extraction_info` {scenario_duration, extraction_offset}, `ts` at 0.1 s.
- Verified on the Pittsburgh toy log (residual under 1 cm): ScenarioNet step i is lidar
  frame k0 + 2 i, where k0 is the first lidar_pc frame with timestamp >= int(timestamp of the
  scenario token + extraction_offset * 1e6), exactly the devkit's inclusive window query with
  subsample 0.5. The simpler "token time + offset + 0.1 s * i" is one frame (45 ms) off
  because lidar timestamps jitter by a fraction of a millisecond. The ScenarioNet world frame
  is the nuPlan frame translated so the ego is at (0, 0) at step 0; headings are unchanged.
  ScenarioNet positions are the vehicle CENTRE while nuPlan's ego_pose is the rear axle; on
  the six real logs the rear-axle comparison failed 174 of 411 scenarios (up to 1.5 m on lane
  changes), the centre comparison fails none (max 3 mm).
- The converter's scenario filter (`all_scenarios`, `timestamp_threshold_s=10`,
  `expand_scenarios=false`) yields a handful of 15 to 20 s scenarios per log, so MTR
  predictions cover only part of each instruction window. Coverage is reported, not assumed.
- The doPlan paper (arXiv 2609.38028) reports that the instructed maneuver begins a median
  of 24.6 s after the evaluation point; most instruction text describes events beyond any
  single 6 s prediction. Gating exists because of this.

## 3. Architecture

Host-side package `scripts/doplan_lang/` (Python 3.12, miniconda; new dependencies:
gensim 4.4, pytest). Nothing in the Docker images changes except an optional converter
knob (section 6). Every module has one job and a plain-data interface (CSV, parquet/CSV
tables, .kv, .joblib, JSON).

| Module | Does | Reads | Writes |
|---|---|---|---|
| `text.py` | normalise, tokenise, split clauses, weak intent rules | CSV | in-memory records |
| `train_word2vec.py` | train skip-gram, report vocabulary and neighbours | text | `word2vec.kv`, `word2vec_report.md` |
| `intent.py` | sentence vector, intent classifier train/predict | `.kv`, text | `intent_clf.joblib`, `intent_report.md` |
| `align.py` | absolute times, window join, coverage | nuPlan .db, summary.pkl, output_MTR | `pairs.csv`, `coverage.csv` |
| `geometry.py` | per-mode features, intent compatibility | one prediction | arrays |
| `rerank.py` | gate + score fusion over all pairs | `pairs.csv`, .pt files, clf | `reranked.csv` |
| `evaluate.py` | metrics, plots | `reranked.csv`, .pt files | `metrics.json`, `metrics.md`, PNGs |
| `config.py` | all thresholds and paths in one dataclass | | |

All outputs go under `outputs/doplan_lang/`. A single `run_all.py` chains the modules
for one database directory and one CSV.

## 4. Component definitions

### 4.1 Text (`text.py`)

- Normalise: lowercase, strip, collapse whitespace, map "stopsign" -> "stop sign",
  "stoplight" -> "stop light", "u-turn"/"uturn" -> "u turn"; tokens = `[a-z']+`.
- Clauses: split on sentence punctuation (. ; ! ? ,) and on the connectors "then",
  "and then", "after", "after that", "once", "when", "until", "before". ("next" and
  "at the next" were dropped during planning: many instructions begin "at the next stop
  sign ..." and splitting there would strip the action.) The clause used in this version
  is the first action clause: the first clause whose rule intent is not `other`, falling
  back to the first clause. This skips leading conditions such as "when you pass the
  intersection". All clauses are kept in the record for later stage tracking.
- Weak intent rules (first match wins, applied per clause; refined 2026-10-23 after a
  review of the labels on the real corpus):
  - `straight` (checked first): "stay/keep/remain/continue in|on ... lane" means hold the
    lane, not change it, unless the clause also contains a turn verb (turn, make a, take a,
    hang a), in which case the turn rules apply.
  - `lane_left`: ("lane" or "merge" or "switch" or "move over" or "get over") and "left"
  - `lane_right`: same with "right"
  - `turn_left`: "left"/"leftmost" and not lane words, or "u turn"
  - `turn_right`: "right"/"rightmost" and not lane words (but not "right away" /
    "right now"; "make a right here" is a turn, while in "make a left right here" the
    "right here" is a place and the clause is a left turn)
  - `stop`: the verb "stop"/"stops"/"stopping" (not the nouns "stop sign" / "stop light",
    not the adjective "stopped"), "wait", "pull over", "pull up", "pull into",
    "drop ... off" (not the noun "drop off zone/area/lane"), "halt", and "park" only as a
    verb (clause start or after and/then/please/just/can/to), so "iron street park" is a place
  - `straight`: "straight", "continue", "keep going", "stay in", "go through", "follow",
    "proceed"
  - `other`: anything else (blank text is `other`)
  Rule precedence is lane > turn > stop > straight so that "merge left then stop at the
  light" gets lane_left for its first clause.

### 4.2 word2vec (`train_word2vec.py`)

- Corpus: tokenised instructions (whole instruction, not just first clause) from every
  doPlan clip except the six evaluation logs in section 6. Blank rows are dropped.
- gensim `Word2Vec(vector_size=100, window=5, min_count=2, sg=1, negative=10,
  epochs=50, seed=0, workers=1)` for reproducibility.
- Report: vocabulary size, fraction of evaluation-log tokens in vocabulary, ten nearest
  neighbours for the seed words left, right, stop, straight, lane, merge, light, sign,
  intersection, exit, and the cosine between left and right (expected high; recorded so
  the limitation is documented).

### 4.3 Intent classifier (`intent.py`)

- Sentence vector = mean of word vectors of the first clause, OOV tokens skipped. An
  all-OOV clause gives the zero vector, which maps to `other` by construction (the
  classifier is trained with such rows labelled `other`).
- `StandardScaler` followed by `sklearn.linear_model.LogisticRegression(C=1.0,
  class_weight='balanced', max_iter=2000)` (lbfgs, multinomial) over the 7 intents (mean
  word vectors have small norms, so features are standardised before the regulariser), trained on the weak labels of
  non-evaluation clips. Reports: 5-fold cross-validated accuracy and per-class F1 against
  the rules; confusion matrix; and for clauses the rules labelled `other`, the top
  classifier intent and probability, with 30 random examples so the generalisation beyond
  the rules can be eyeballed.
- Output probabilities P(intent | clause) are what re-ranking consumes. The rule label is
  also kept so the evaluation can run with rules-only as an ablation.

### 4.4 Alignment (`align.py`)

- Per log: `t0 = min(lidar_pc.timestamp)` from the .db; doPlan window =
  `[t0 + 0.2 s * start_frame, t0 + 0.2 s * end_frame]`.
- Per scenario (from `dataset_summary.pkl`): `k0 = first lidar frame with timestamp >=
  int(lidar_pc.timestamp[scenario_token] + extraction_offset * 1e6)`. Per ego prediction file:
  `t_pred = lidar_pc.timestamp[k0 + 2 * current_time_index]`.
- A (prediction, annotation) pair exists when `t_pred` lies inside the window and the
  annotation has text. Overlapping annotations give separate pairs. Several annotators
  often wrote instructions for the same window (the richest Singapore log has 57 text
  records over only 3 distinct windows), so pair counts are dominated by those windows.
- Converter scenarios overlap heavily (the six toy scenarios start within 2 s of each
  other and last 15 to 20 s), so one instant of driving yields several MTR predictions.
  Each pair carries a `moment` key: log name plus the lidar frame index of the prediction
  divided by 10 (one 0.5 s bucket), so predictions from overlapping scenarios that fall in
  the same bucket share it regardless of the scenarios' 20 Hz phase or timestamp jitter.
- `coverage.csv`: per annotation, window length, number of distinct moments with an ego
  prediction inside it, seconds covered (moments * 0.5 s), fraction covered. Printed
  summary: median fraction covered, number of annotations with zero coverage.
- Sanity check at run time, per scenario that has ego predictions (inference may have been
  capped to fewer scenarios than the database holds): the ego displacement between the
  first MTR prediction index and the one nearest 20 must match the displacement of the
  ego_pose linked to the corresponding lidar frames, shifted from the rear axle to the vehicle
  centre that ScenarioNet records ((front_length - rear_length) / 2 = 1.461 m for the Pacifica,
  from the scenario's `ego_vehicle_parameters`), within 0.05 m, and the heading in `gt_trajs` row 20 must match
  that ego_pose yaw within 0.02 rad. A failed check stops the run with the scenario named.

### 4.5 Geometry and compatibility (`geometry.py`)

Ego frame at prediction time: origin at `gt_trajs[20, :2]`, x axis along
`gt_trajs[20, 6]`. For each mode: rotate the 60 waypoints into this frame.

Features per mode:
- `lat`: y of the endpoint (left positive).
- `dpsi`: the mode's final heading relative to the current ego heading, i.e. the angle of
  the last displacement vector in the ego frame, taken from the last two waypoints at least
  0.5 m apart (0 if the mode barely moves).
- `path`: cumulative path length from the ego position through all waypoints (the first
  waypoint is 0.1 s ahead, so the segment to it counts).

Compatibility `c(mode, intent)` in (0, 1], smooth so that lambda has a meaning:
- `turn_left`: sigmoid((lat - 2) / 1) * sigmoid((dpsi - 30 deg) / 10 deg)
- `turn_right`: same with -lat, -dpsi
- `lane_left`: sigmoid((lat - 1.5) / 0.75) * (1 - sigmoid((|dpsi| - 20 deg) / 10 deg))
- `lane_right`: mirrored
- `stop`: 1 - sigmoid((path - 3) / 1.5)
- `straight`: (1 - sigmoid((|lat| - 1.5) / 0.75)) * (1 - sigmoid((|dpsi| - 15 deg) / 10 deg))
- `other`: 1 for every mode
Each is floored at 0.02 so no mode is zeroed. Compatibility with a distribution:
`c(mode) = sum_i P(intent_i) * c(mode, intent_i)`.

### 4.6 Gate and fusion (`rerank.py`)

- Gate open when, among the plausible modes (normalised MTR score at least
  `gate_min_score` = 0.05; fewer than two plausible modes means closed), the max pairwise
  endpoint distance > 4 m or the max pairwise heading difference, wrapped into
  [-180, 180) degrees, > 20 degrees. Modes MTR scores near 1e-6 (often reversing or
  U-turn modes) would otherwise open the gate on every prediction. Longitudinal spread
  (one mode brakes, another keeps speed) counts on purpose: it is the fork a `stop` intent
  acts on. Closed gate: scores unchanged (normalised), pair flagged `gated=False`.
- Fusion: `s'_k = softmax_k( log s_k + lambda * log c_k )`, lambda default 1.0.
- `reranked.csv` row per pair: prediction file, annotation_id, instruction, first clause,
  rule intent, classifier intent distribution (7 columns), gated flag, per-mode features
  (lat, dpsi, path for 5 modes), old scores, new scores, old top-1, new top-1.
- Also writes `reranked_pt/<prediction file stem>_<annotation_id>.pt` (one per pair, since
  a prediction can fall in several windows) with `pred_scores` replaced and the original
  kept as `pred_scores_mtr`, so the existing plotting scripts still work.

## 5. Evaluation (`evaluate.py`)

Unit = one (ego prediction, annotation) pair. Metrics on gated pairs (headline) and on
all pairs (context), each before and after:
- top-1 ADE and FDE (m) of the highest-scoring mode vs the recorded future over valid
  steps; minADE/minFDE over the 5 modes as the unchanged floor.
- Directional compliance, as in doPlan Section V-B: among pairs whose first-clause rule
  label is turn_left/turn_right/lane_left/lane_right (the rule label, not the variant's
  own intent, so every variant is scored on the same pairs), whose recorded future covers the
  full 6 s, and whose recorded 6 s endpoint has a lateral offset of at least 1 m toward the
  instructed side in the ego frame, the fraction where the top-1 mode's endpoint also has at
  least 1 m lateral offset toward that side. Reported per subset (gated, all).
- Counterfactual consistency: swap left/right (and leftmost/rightmost, left-turn) in the
  normalised first clause with `text.swap_left_right`, re-run the classifier and fusion, and report the fraction of those pairs where the top-1 endpoint
  side flips with the instruction.
- Ablation column: rules-only intents (one-hot) instead of classifier probabilities.
- Bootstrap 95% confidence intervals (1000 resamples) for the before/after difference in
  top-1 ADE and compliance, since the pair count may be in the low hundreds. Resampling is
  by moment (cluster bootstrap), not by pair, because several pairs share one instant of
  driving; the report states both the pair count and the moment count, shows the ADE and
  compliance CIs in the table, and lists pairs per log (one Singapore window carries 57
  annotations, so a single log can dominate the pair-weighted means). metrics.json writes
  NaN as null.
Outputs: `metrics.json`, `metrics.md` (one table), and up to 12 example PNGs (modes,
recorded future, instruction text, old and new top-1 highlighted), chosen as the pairs with
the largest score change in either direction.

## 6. Data run

Evaluation logs (all in the nuPlan mini split; text-record counts from
`nuplan_data/doplan_clip_inventory.csv`):

| log | city | length | text records | .db size |
|---|---|---|---|---|
| 2021.10.11.02.57.41_veh-50_00352_00535 | sg-one-north | 182 s | 57 | 15 MB |
| 2021.10.11.08.31.07_veh-50_01750_01948 | sg-one-north | 197 s | 18 | 13 MB |
| 2021.10.11.02.57.41_veh-50_01522_02088 | sg-one-north | 509 s | 3 | 31 MB |
| 2021.08.17.18.54.02_veh-45_00665_01065 | us-pa-pittsburgh-hazelwood | 399 s | 6 | 36 MB |
| 2021.08.17.17.17.01_veh-45_02314_02798 | us-pa-pittsburgh-hazelwood | 483 s | 3 | 54 MB |
| 2021.08.17.16.57.11_veh-08_01200_01636 | us-pa-pittsburgh-hazelwood | 435 s | 3 | 37 MB |

The fourth Singapore mini log (2021.10.11.07.12.18_veh-50_00211_00304) has only blank
annotations and is skipped. Total 90 text records; about 185 MB of logs plus the 231 MB
Singapore map (Pittsburgh map already on disk).

Steps:
1. `scripts/fetch_nuplan_toy.py --dest nuplan_data --log ... --city sg-one-north`
   (repeatable flags) into the existing layout. The Pittsburgh toy log stays where it is;
   the converter is pointed at a separate raw folder listing only the six logs, via the
   existing `NUPLAN_RAW_DIR` variable, so the toy database is untouched.
2. `docker compose run --rm -e NUPLAN_RAW_DIR=... -e SCENARIONET_OUT=/mnt/scenarionet/nuplan_doplan nuplan-convert`.
3. `docker compose run --rm -e SCENARIONET_DIR=/mnt/scenarionet/nuplan_doplan mtr bash /opt/mtr-scripts/run_inference.sh debug_max_nb_scenarios=100000`.
4. `python3 scripts/doplan_lang/run_all.py --scenarionet scenarionet_data/nuplan_doplan --nuplan nuplan_data --csv <doPlan csv> --out outputs/doplan_lang`.

Optional knob if coverage is thin: expose the converter's hardcoded
`scenario_filter.timestamp_threshold_s=10` through an environment variable in
`scripts/patch_nuplan_convert.py` (same pattern as `NUPLAN_REMOVE_INVALID_GOALS`). Not
applied in the first run; the first run reports coverage with the default.

Constraint: Docker Desktop with WSL integration must be running for steps 2 and 3. It was
not reachable from the shell on 2026-10-23.

## 7. Failure handling

- Blank instructions carry no language and are dropped when the CSV is loaded, so they
  form no pairs. An all-OOV first clause gives intent `other`, flat compatibility, and a
  fusion that is a no-op; the pair is kept and its intent column says `other`.
- Prediction with no valid future step: re-ranked, excluded from metrics.
- Scenario token missing from the .db, or alignment sanity check fails: abort with the
  scenario and log named; no silent skipping.
- Mode that barely moves (path < 0.5 m): dpsi = 0, lat from endpoint as usual.
- Missing map or log for a city: fetch script error surfaces before conversion.
- Pairs where gate is closed are reported separately so "nothing changed" cases cannot
  inflate or hide the effect.

## 8. Testing

pytest under `scripts/doplan_lang/tests/`, runnable on the host without Docker:
- `test_text.py`: normalisation, clause splitting on each connector, every rule with a
  positive and a negative example, "right now" not labelled turn_right.
- `test_geometry.py`: synthetic left arc gives lat > 0 and dpsi > 0; mirrored arc negative;
  stationary mode gives path about 0 and stop compatibility near 1; straight line gives
  straight compatibility near 1 and turn compatibilities near the floor.
- `test_align.py`: synthetic lidar timestamps, token, offset, and frames produce the
  expected pairs and coverage fractions; boundary frames inclusive.
- `test_rerank.py`: lambda = 0 leaves scores unchanged; closed gate leaves scores unchanged;
  a left-turn intent with one left mode raises that mode's score.
- `test_integration.py`: runs the full chain on `scenarionet_data/nuplan_toy` with one
  fabricated annotation covering the toy log, asserting the pipeline produces the
  expected files and that the alignment check passes on real data.

## 9. Out of scope

Known v1 limitations recorded during execution (2026-10-23), candidates for a second iteration:
- `dpsi` is the final heading from `arctan2`, which wraps at 180 degrees, so a reversing mode
  or a U-turn ending past 180 degrees reads as about -178 degrees and gets a small spurious
  `turn_right` compatibility (about 0.12 against a 0.02 floor). Tracking heading change
  cumulatively along the path, or treating endpoints with x < 0 as non-turning, would fix it.
- `stop` compatibility uses path length under 3 m over 6 s, which only fires for an ego that is
  already nearly stationary. Most doPlan stop instructions describe decelerating from speed
  (15 to 30 m of travel within the horizon), so a final-speed feature (last displacement per
  0.1 s below about 1 m/s) or path relative to constant velocity would serve better.
- The intent classifier reproduces the keyword rules almost exactly (probabilities near
  one-hot), so the rules-only ablation is expected to look identical; the `gated_disagree`
  subset in the report quantifies what word2vec adds.
- The rules map "u turn" to `turn_left`, which is right for Boston, Pittsburgh and Las Vegas
  but wrong in Singapore (left-hand traffic: a U-turn curves right). None of the 90
  evaluation instructions contains "u turn", so this run is unaffected; a city-aware rule is
  needed before Singapore U-turns matter.
- The evaluation set is 90 instructions, 67 of them turn_left, 57 from one Singapore log; the
  pair-weighted means are dominated by that log (see `pairs per log` in the report).


Retraining or fine-tuning MTR, a learned fusion head, stage tracking across clauses,
re-ranking for non-ego agents, pretrained word vectors, video or CLIP features, and the Las
Vegas or Boston logs. Each can be added behind the existing interfaces: swap the `.kv`
file, add a gate function, or point `run_all.py` at another database.
