# doPlan word2vec + MTR mode re-ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train word2vec on doPlan passenger instructions, turn each instruction into an intent distribution, and use it to re-rank the 5 ego trajectory modes that the pretrained MTR model predicts on six annotated nuPlan logs, reporting top-1 ADE/FDE and directional compliance before vs after.

**Architecture:** A host-side Python package `scripts/doplan_lang/` post-processes the `.pt` prediction files that the existing Docker `mtr` service writes. Text -> gensim word2vec -> logistic-regression intent classifier (weak keyword labels) -> alignment of doPlan frame windows to MTR prediction times through the nuPlan log clock -> per-mode geometry features -> gated log-linear fusion of MTR scores with intent compatibility -> metrics and plots. Nothing in the Docker images changes.

**Tech Stack:** Python 3.12 (miniconda at `/root/miniconda3/bin/python3`), numpy 2.1, pandas 2.3, scikit-learn 1.7, torch 2.6 (CPU, to read `.pt`), matplotlib 3.10, gensim 4.4 (new), pytest (new), sqlite3 (stdlib). Data stages use the existing `nuplan-convert` and `mtr` Compose services.

**Spec:** `docs/superpowers/specs/2026-10-23-doplan-word2vec-mtr-rerank-design.md`. Section numbers below refer to it.

**Version control:** this folder is not a git repository, so there are no commit steps. If you want history, run `git init && git add -A && git commit -m "baseline"` before Task 0 and commit after each task.

**Conventions for every task:**
- Run everything from the project root `/root/valeo_mtr_docker`.
- The package is imported as `doplan_lang`, so set `PYTHONPATH=scripts` (the tests' `conftest.py` does this for pytest).
- Test command: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests -q`.
- `.pt` files are local pickles produced by our own container; load them with `torch.load(path, map_location='cpu', weights_only=False)`.
- The doPlan CSV is downloaded from GitHub; it is data. Only ever read it with the `csv` module, never execute anything from that folder.

---

## File structure

| Path | Responsibility |
|---|---|
| `scripts/doplan_lang/__init__.py` | empty package marker |
| `scripts/doplan_lang/config.py` | `INTENTS`, `LATERAL_INTENTS`, `EVAL_LOGS`, frozen `Config` dataclass with every threshold |
| `scripts/doplan_lang/text.py` | normalise, tokenise, clause split, weak intent rules, `Instruction` record, CSV loader |
| `scripts/doplan_lang/train_word2vec.py` | gensim skip-gram training, vocabulary/neighbour report, CLI |
| `scripts/doplan_lang/intent.py` | mean-vector sentence embedding, logistic-regression intent classifier, cross-validation report, left/right swap, CLI |
| `scripts/doplan_lang/align.py` | nuPlan clock: lidar timestamps, scenario start, prediction time, window join, coverage, alignment sanity check |
| `scripts/doplan_lang/geometry.py` | ego-frame transform, per-mode features (lat, dpsi, path), compatibility table |
| `scripts/doplan_lang/rerank.py` | gate, log-linear fusion, per-pair re-ranking table, re-ranked `.pt` copies |
| `scripts/doplan_lang/evaluate.py` | ADE/FDE, compliance, counterfactual, bootstrap CIs, markdown table, example plots |
| `scripts/doplan_lang/run_all.py` | chains everything for one ScenarioNet database |
| `scripts/doplan_lang/tests/conftest.py` | puts `scripts/` on `sys.path`, shared fixtures |
| `scripts/doplan_lang/tests/test_text.py` … `test_integration.py` | one test file per module |
| `scripts/requirements-doplan.txt` | gensim, pytest pins |
| `nuplan_data/doplan/release_annotationsV1.csv` | the doPlan annotations (downloaded, data only) |
| `nuplan_data/nuplan-v1.1/splits/doplan/*.db` | hard links to the six evaluation logs for the converter |
| `scenarionet_data/nuplan_doplan/` | converted database + `output_MTR/` predictions |
| `outputs/doplan_lang/` | word2vec.kv, reports, pairs.csv, coverage.csv, reranked*.csv, reranked_pt/, metrics.json, metrics.md, plots/ |

---

### Task 0: Package skeleton, dependencies, data file, config

**Files:**
- Create: `scripts/requirements-doplan.txt`
- Create: `scripts/doplan_lang/__init__.py`
- Create: `scripts/doplan_lang/config.py`
- Create: `scripts/doplan_lang/tests/__init__.py`
- Create: `scripts/doplan_lang/tests/conftest.py`
- Create: `scripts/doplan_lang/tests/test_config.py`
- Download: `nuplan_data/doplan/release_annotationsV1.csv`

- [ ] **Step 1: Install the two new packages**

```bash
cat > scripts/requirements-doplan.txt <<'REQ'
# Host-side doPlan language re-ranking (scripts/doplan_lang). torch is needed only to read the MTR
# .pt files; install a CPU wheel if it is absent (torch>=2.0). numpy/scipy/joblib come with the pins.
gensim==4.4.0
scikit-learn==1.7.0      # intent_clf.joblib is a pickle tied to this version
pandas>=2.2
matplotlib>=3.8
scipy>=1.11
pytest>=8
REQ
python3 -m pip install -r scripts/requirements-doplan.txt
python3 -c "import gensim, pytest; print(gensim.__version__, pytest.__version__)"
```
Expected: `4.4.0 8.x`.

- [ ] **Step 2: Download the doPlan annotations into their own data folder**

```bash
mkdir -p nuplan_data/doplan
curl -sSL -o nuplan_data/doplan/release_annotationsV1.csv \
  https://raw.githubusercontent.com/Mi3-Lab/doPlan/main/outputs/doPlan_submissions/release_annotationsV1.csv
python3 -I -c "
import csv; rows=list(csv.DictReader(open('nuplan_data/doplan/release_annotationsV1.csv', encoding='utf-8')))
print(len(rows), sum(1 for r in rows if r['passenger_instruction'].strip()))"
```
Expected: `6621 5198`.

- [ ] **Step 3: Create the package and config**

```bash
mkdir -p scripts/doplan_lang/tests
touch scripts/doplan_lang/__init__.py scripts/doplan_lang/tests/__init__.py
```

`scripts/doplan_lang/config.py`:
```python
"""Every tunable number and name used by the doPlan language re-ranking pipeline.

Spec: docs/superpowers/specs/2026-10-23-doplan-word2vec-mtr-rerank-design.md
"""
from __future__ import annotations

from dataclasses import dataclass

# Order matters: geometry.compat_table stacks its columns in this order.
INTENTS = ('turn_left', 'turn_right', 'lane_left', 'lane_right', 'stop', 'straight', 'other')
LATERAL_INTENTS = ('turn_left', 'turn_right', 'lane_left', 'lane_right')

# Evaluation logs (spec section 6). Their text is held out of word2vec and classifier training.
EVAL_LOGS = (
    '2021.10.11.02.57.41_veh-50_00352_00535',
    '2021.10.11.08.31.07_veh-50_01750_01948',
    '2021.10.11.02.57.41_veh-50_01522_02088',
    '2021.08.17.18.54.02_veh-45_00665_01065',
    '2021.08.17.17.17.01_veh-45_02314_02798',
    '2021.08.17.16.57.11_veh-08_01200_01636',
)


@dataclass(frozen=True)
class Config:
    # word2vec (spec 4.2)
    vector_size: int = 100
    window: int = 5
    min_count: int = 2
    epochs: int = 50
    seed: int = 0
    sg: int = 1              # skip-gram
    negative: int = 10       # negative samples
    # timing (spec 4.4)
    doplan_frame_s: float = 0.2   # doPlan frames are 5 per second of drive
    step_s: float = 0.1           # ScenarioNet / MTR time step
    past_len: int = 21            # rows of gt_trajs before the prediction time; row 20 is "now"
    pred_stride_s: float = 0.5    # MTR writes a prediction every 5 steps
    # geometry thresholds (spec 4.5); metres and degrees
    turn_lat_m: float = 2.0
    turn_heading_deg: float = 30.0
    lane_lat_m: float = 1.5
    lane_heading_deg: float = 20.0
    straight_lat_m: float = 1.5
    straight_heading_deg: float = 15.0
    stop_path_m: float = 3.0
    compat_floor: float = 0.02
    # gate and fusion (spec 4.6)
    gate_min_score: float = 0.05   # modes below this normalised MTR score do not count as a fork
    gate_endpoint_m: float = 4.0
    gate_heading_deg: float = 20.0
    lam: float = 1.0
    # evaluation (spec 5)
    compliance_m: float = 1.0
    bootstrap_n: int = 1000
    n_example_plots: int = 12
```

`scripts/doplan_lang/tests/conftest.py`:
```python
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]      # .../scripts
PROJECT_ROOT = SCRIPTS_DIR.parent                      # .../valeo_mtr_docker
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
```

- [ ] **Step 4: Write a config test and run it**

`scripts/doplan_lang/tests/test_config.py`:
```python
from doplan_lang.config import Config, EVAL_LOGS, INTENTS, LATERAL_INTENTS


def test_intent_order_is_fixed():
    assert INTENTS == ('turn_left', 'turn_right', 'lane_left', 'lane_right', 'stop', 'straight', 'other')
    assert set(LATERAL_INTENTS) < set(INTENTS)


def test_six_eval_logs():
    assert len(EVAL_LOGS) == 6 and len(set(EVAL_LOGS)) == 6


def test_config_is_frozen_with_defaults():
    cfg = Config()
    assert cfg.lam == 1.0 and cfg.past_len == 21
    try:
        cfg.lam = 2.0
    except Exception:
        return
    raise AssertionError('Config must be frozen')
```
Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_config.py -q`
Expected: `3 passed`.

---

### Task 1: Text module (spec 4.1)

**Files:**
- Create: `scripts/doplan_lang/text.py`
- Test: `scripts/doplan_lang/tests/test_text.py`

- [ ] **Step 1: Write the failing tests**

`scripts/doplan_lang/tests/test_text.py`:
```python
import pytest

from doplan_lang.text import (first_action_clause, make_instruction, normalize, rule_intent,
                              split_clauses, tokenize)


def test_normalize_lowercases_and_maps_compounds():
    assert normalize('  Stop at the STOPSIGN, then U-turn ') == 'stop at the stop sign, then u turn'
    assert normalize('Stoplights ahead') == 'stop light ahead'


def test_tokenize_keeps_apostrophes_only():
    assert tokenize("Don't stop, go!") == ["don't", 'stop', 'go']


def test_split_clauses_on_punctuation_and_connectors():
    assert split_clauses('Turn left at the light, then stop. Once clear, go straight') == [
        'turn left at the light', 'stop', 'clear', 'go straight']


def test_split_clauses_leading_connector_gives_no_empty_clause():
    assert split_clauses('when you pass the intersection, switch to the left lane') == [
        'you pass the intersection', 'switch to the left lane']


def test_split_clauses_does_not_split_on_next():
    assert split_clauses('at the next stop sign make a right turn') == ['at the next stop sign make a right turn']


@pytest.mark.parametrize('clause,intent', [
    ('turn left at the intersection', 'turn_left'),
    ('make a right turn at the stop sign', 'turn_right'),      # turn beats stop
    ('merge one lane to the left', 'lane_left'),
    ('switch to the right lane', 'lane_right'),
    ('get over to the left lane and turn left', 'lane_left'),  # lane beats turn
    ('stop for the red light', 'stop'),
    ('stop right now', 'stop'),                                # "right now" is not a direction
    ('keep going straight', 'straight'),
    ('stay in the middle lane', 'straight'),
    ('make a u turn', 'turn_left'),
    ('exit luxor las vegas', 'other'),
    ('turn right from the left lane', 'other'),                # both sides: ambiguous
    ('', 'other'),
])
def test_rule_intent(clause, intent):
    assert rule_intent(clause) == intent


def test_first_action_clause_skips_conditions():
    assert first_action_clause(['you pass the intersection', 'switch to the left lane']) == 'switch to the left lane'
    assert first_action_clause(['exit luxor']) == 'exit luxor'
    assert first_action_clause([]) == ''


def test_make_instruction_record():
    ins = make_instruction('a1', 'log-x', '10', '60', 'Turn left, then stop.')
    assert ins.clauses == ('turn left', 'stop')
    assert ins.first_clause == 'turn left' and ins.rule_intent == 'turn_left'
    assert ins.start_frame == 10 and ins.end_frame == 60
    assert ins.tokens == ('turn', 'left', 'then', 'stop')
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_text.py -q`
Expected: `ImportError ... No module named 'doplan_lang.text'`.

- [ ] **Step 3: Implement `text.py`**

```python
"""doPlan instruction text: normalisation, tokens, clauses and weak intent rules (spec 4.1).

Rule precedence: lane change > turn > stop > straight > other. A clause naming both sides
("turn right from the left lane") is ambiguous and labelled `other`.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

_REPLACEMENTS = (
    (re.compile(r'\bstopsigns?\b'), 'stop sign'),
    (re.compile(r'\bstoplights?\b'), 'stop light'),
    (re.compile(r'\bu-?turns?\b'), 'u turn'),
)
_TOKEN = re.compile(r"[a-z']+")
_CLAUSE_SPLIT = re.compile(r"[.;!?,]+|\b(?:and then|then|after that|after|once|when|until|before)\b")

_LANE = re.compile(r"\b(?:lanes?|merge|merging|switch|move over|get over)\b")
_LEFT = re.compile(r"\bleft\b")
_RIGHT = re.compile(r"\bright\b(?!\s+(?:away|now|here)\b)")
_UTURN = re.compile(r"\bu turn\b")
_STOP = re.compile(r"\b(?:stop|stops|stopping|wait|pull over|pull into|park|drop off|halt)\b")
_STRAIGHT = re.compile(r"\b(?:straight|continue|keep going|stay in|go through|follow|proceed|procced)\b")


def normalize(text: str) -> str:
    text = (text or '').lower().strip()
    for pattern, repl in _REPLACEMENTS:
        text = pattern.sub(repl, text)
    return re.sub(r'\s+', ' ', text)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def split_clauses(text: str) -> list[str]:
    parts = _CLAUSE_SPLIT.split(normalize(text))
    return [p.strip() for p in parts if p and p.strip()]


def rule_intent(clause: str) -> str:
    c = normalize(clause)
    lane = bool(_LANE.search(c))
    left = bool(_LEFT.search(c))
    right = bool(_RIGHT.search(c))
    if lane and left and not right:
        return 'lane_left'
    if lane and right and not left:
        return 'lane_right'
    if _UTURN.search(c):
        return 'turn_left'
    if left and not right:
        return 'turn_left'
    if right and not left:
        return 'turn_right'
    if _STOP.search(c):
        return 'stop'
    if _STRAIGHT.search(c):
        return 'straight'
    return 'other'


def first_action_clause(clauses: list[str]) -> str:
    """First clause with a non-`other` rule intent, else the first clause, else ''."""
    for clause in clauses:
        if rule_intent(clause) != 'other':
            return clause
    return clauses[0] if clauses else ''


@dataclass(frozen=True)
class Instruction:
    annotation_id: str
    clip_id: str          # nuPlan log name
    start_frame: int      # doPlan frames, 5 per second from log start
    end_frame: int
    text: str
    clauses: tuple[str, ...]
    first_clause: str
    rule_intent: str
    tokens: tuple[str, ...]   # whole instruction, for word2vec


def make_instruction(annotation_id, clip_id, start_frame, end_frame, text: str) -> Instruction:
    clauses = tuple(split_clauses(text))
    first = first_action_clause(list(clauses))
    return Instruction(
        annotation_id=str(annotation_id), clip_id=str(clip_id),
        start_frame=int(float(start_frame)), end_frame=int(float(end_frame)),
        text=text, clauses=clauses, first_clause=first,
        rule_intent=rule_intent(first), tokens=tuple(tokenize(text)))


def load_annotations(csv_path: Path) -> list[Instruction]:
    """Non-blank doPlan rows as Instruction records (blank rows carry no language)."""
    out: list[Instruction] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            text = (row.get('passenger_instruction') or '').strip()
            if (row.get('is_blank') or 'false').strip().lower() == 'true' or not text:
                continue
            out.append(make_instruction(row['annotation_id'], row['clip_id'],
                                        row['start_frame'], row['end_frame'], text))
    return out
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_text.py -q`
Expected: `20 passed`.

- [ ] **Step 5: Smoke the rules on the real CSV and record the label distribution**

```bash
PYTHONPATH=scripts python3 -c "
from collections import Counter
from doplan_lang.text import load_annotations
ins = load_annotations('nuplan_data/doplan/release_annotationsV1.csv')
print(len(ins), Counter(i.rule_intent for i in ins).most_common())
print(sum(1 for i in ins if len(i.clauses) > 1), 'multi-clause')"
```
Expected: 5198 instructions; `other` should be well under half; the exact counts go into the plan's execution notes.

- [ ] **Step 6 (added after the Task 1 spec review): refine the rules that mislabel real instructions**

The spec reviewer ran the rules over the real corpus and found about 180 wrong weak labels:
"make a right here" fell to `other` (the plan excluded "right here", the spec did not);
the nouns "stop sign" / "stop light" and the place-name noun "park" counted as `stop`;
"drop me off" and "pull up" were missed; "stay in the right lane" became `lane_right`.
Replace the rule patterns and the start of `rule_intent` in `text.py` with:

```python
_LANE = re.compile(r"\b(?:lanes?|merge|merging|switch|move over|get over)\b")
# "stay in the right lane" holds the lane; it is not a lane change.
_KEEP_LANE = re.compile(r"\b(?:stay|keep|remain|continue)\s+(?:in|on)\s+(?:the\s+)?(?:\w+\s+)?lanes?\b")
_LEFT = re.compile(r"\bleft\b")
_RIGHT = re.compile(r"\bright\b(?!\s+(?:away|now)\b)")
_UTURN = re.compile(r"\bu turn\b")
# "stop" as a verb (not the nouns "stop sign" / "stop light"); "park" only as a verb.
_STOP = re.compile(r"\bstop(?:s|ped|ping)?\b(?!\s+(?:signs?|lights?)\b)"
                   r"|\b(?:wait|pull over|pull up|pull into|halt)\b"
                   r"|\bdrop(?:\s+\w+)?\s+off\b"
                   r"|(?:^|\b(?:and|then|please|just|can|to)\s+)park\b")
_STRAIGHT = re.compile(r"\b(?:straight|continue|keep going|stay in|go through|follow|proceed|procced)\b")


def rule_intent(clause: str) -> str:
    c = normalize(clause)
    if _KEEP_LANE.search(c):
        return 'straight'
    lane = bool(_LANE.search(c))
    ...  # the rest of the function is unchanged
```

Add these cases to the `test_rule_intent` parametrize list in `test_text.py`:
```python
    ('make a right here', 'turn_right'),                       # "right here" is still a turn
    ('go straight at the stop sign', 'straight'),              # "stop sign" is a noun
    ('passing the second stop light', 'other'),
    ('stop at the stop sign', 'stop'),
    ('head toward iron street park', 'other'),                 # "park" the place
    ('park here', 'stop'),                                     # "park" the verb
    ('pull in and park', 'stop'),
    ('drop me off at the aria', 'stop'),
    ('pull up to the curb', 'stop'),
    ('stay in the right lane', 'straight'),                    # holding a lane is not a lane change
    ('keep going straight and merge into the left lane', 'lane_left'),
    ('be on the far left turning lane', 'lane_left'),
```
and this test for the connectors the spec's section 8 asks to cover:
```python
@pytest.mark.parametrize('text,clauses', [
    ('turn left and then stop', ['turn left', 'stop']),
    ('stop after the light', ['stop', 'the light']),
    ('turn right after that go straight', ['turn right', 'go straight']),
    ('wait until it is clear', ['wait', 'it is clear']),
    ('stop before the crosswalk', ['stop', 'the crosswalk']),
])
def test_split_clauses_each_connector(text, clauses):
    assert split_clauses(text) == clauses
```
Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_text.py -q`
Expected: `37 passed`. Then rerun Step 5 and report the new label counts (expect `stop` to drop by roughly 150 and `turn_right` to rise by about 30).

- [ ] **Step 7 (added after the Step 6 re-review): three residual mislabels**

The re-review found about 50 remaining wrong labels: "stopped" as an adjective ("go around
the stopped car") now counts as `stop`; the noun "drop off zone" ("leave the drop off zone")
counts as `stop`; and a hold-lane clause that also contains a turn verb ("turn left at the
intersection and stay in the left lane") returns `straight`. Change `text.py`:

```python
_TURN_VERB = re.compile(r"\b(?:turn|turning|make a|take a|hang a)\b")
_STOP = re.compile(r"\bstop(?:s|ping)?\b(?!\s+(?:signs?|lights?)\b)"
                   r"|\b(?:wait|pull over|pull up|pull into|halt)\b"
                   r"|\bdrop(?:\s+\w+)?\s+off\b(?!\s+(?:zone|area|lane)s?\b)"
                   r"|(?:^|\b(?:and|then|please|just|can|to)\s+)park\b")


def rule_intent(clause: str) -> str:
    c = normalize(clause)
    keep_lane = bool(_KEEP_LANE.search(c))
    if keep_lane and not _TURN_VERB.search(c):
        return 'straight'
    lane = bool(_LANE.search(c)) and not keep_lane
```
(`_STOP` loses the `ped` form; the rest of `rule_intent` is unchanged.) Add to the
`test_rule_intent` cases:
```python
    ('turn left at the intersection and stay in the left lane', 'turn_left'),   # turn beats hold-lane
    ('go around the stopped car', 'other'),                                     # adjective, not a stop
    ('leave the drop off zone', 'other'),                                       # noun, not a stop
    ('merge into the drop off zone', 'other'),
    ('drop me off in the drop off zone', 'stop'),                               # the verb still counts
```
Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests -q`
Expected: `45 passed`. Rerun Step 5 and report the counts.

- [ ] **Step 8 (added after the Task 0-1 code-quality review): loader test, label check, small rule fixes, counterfactual helper**

In `text.py`:
- Directions include the superlatives: `_LEFT = re.compile(r"\bleft(?:most)?\b")`,
  `_RIGHT = re.compile(r"\bright(?:most)?\b(?!\s+(?:away|now)\b)")`.
- `_TURN_VERB` must not fire on "make a stop":
  `_TURN_VERB = re.compile(r"\b(?:turn|turning|(?:make|take|hang) a (?:left|right|u turn))\b")`.
- Comment the deliberate corpus typo `procced` in `_STRAIGHT` and the overlap of "stay in" with `_KEEP_LANE`.
- Docstring precedence sentence: "Precedence: hold-lane (unless a turn verb is present) > lane change > turn > stop > straight > other."
- `load_annotations(csv_path: str | Path)`; wrap the `make_instruction` call so a malformed row
  re-raises `ValueError(f'annotation {row.get("annotation_id")!r}: {exc}')`.
- Add the counterfactual helper here (it is text processing; Task 3 and Task 6 import it from `text`):
```python
_DIRECTION = re.compile(r'\b(left|right)(most)?\b')


def swap_left_right(text: str) -> str:
    """Counterfactual instruction: every left becomes right and vice versa (also leftmost/rightmost, left-turn)."""
    return _DIRECTION.sub(lambda m: {'left': 'right', 'right': 'left'}[m.group(1)] + (m.group(2) or ''),
                          normalize(text))
```

In `tests/test_text.py` add:
```python
from doplan_lang.config import INTENTS
from doplan_lang.text import load_annotations, swap_left_right


def test_load_annotations_filters_and_parses(tmp_path):
    csv_path = tmp_path / 'ann.csv'
    csv_path.write_text(
        'annotation_id,clip_id,start_frame,end_frame,window_duration_sec,passenger_instruction,referential_label,is_blank,annotator_id\n'
        'a1,logA,10,60.0,10.0,Turn left at the light.,Static Referential,false,x\n'
        'a2,logA,0,50,10.0,,Non-Referential,true,x\n'
        'a3,logA,0,50,10.0,   ,Non-Referential,false,x\n'
        'a4,logB,5,15,2.0,Stop.,Non-Referential,false,x\n')
    ins = load_annotations(csv_path)
    assert [i.annotation_id for i in ins] == ['a1', 'a4']
    assert ins[0].end_frame == 60 and ins[0].text == 'Turn left at the light.' and ins[0].rule_intent == 'turn_left'
    assert ins[1].clip_id == 'logB' and ins[1].rule_intent == 'stop'


def test_load_annotations_names_bad_row(tmp_path):
    csv_path = tmp_path / 'ann.csv'
    csv_path.write_text('annotation_id,clip_id,start_frame,end_frame,passenger_instruction,is_blank\n'
                        'bad1,logA,abc,5,Stop.,false\n')
    with pytest.raises(ValueError, match='bad1'):
        load_annotations(csv_path)


def test_rule_labels_are_known_intents():
    for clause in ['turn left', 'turn right', 'merge left', 'merge right', 'stop', 'go straight', 'exit here']:
        assert rule_intent(clause) in INTENTS


def test_swap_left_right():
    assert swap_left_right('Turn left into the right lane') == 'turn right into the left lane'
    assert swap_left_right('get in the rightmost left-turn lane') == 'get in the leftmost right-turn lane'
    assert swap_left_right('stop here') == 'stop here'
```
and these `test_rule_intent` cases:
```python
    ('switch into the leftmost lane', 'lane_left'),
    ('take a right onto the leftmost lane', 'other'),          # both sides named
    ('stay in the right lane and make a stop', 'straight'),    # "make a stop" is not a turn verb
    ('go straight and stop', 'stop'),                          # stop beats straight
    ('make a u turn into the right lane', 'lane_right'),       # lane beats turn, by design
```
Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests -q`
Expected: `54 passed`. Rerun Step 5 and report the counts.

---

### Task 2: word2vec training (spec 4.2)

**Files:**
- Create: `scripts/doplan_lang/train_word2vec.py`
- Test: `scripts/doplan_lang/tests/test_train_word2vec.py`

- [ ] **Step 1: Write the failing test**

```python
from dataclasses import replace

import numpy as np
import pytest

from doplan_lang.config import Config
from doplan_lang.text import make_instruction
from doplan_lang.train_word2vec import report, token_coverage, train, training_sentences

CORPUS = [
    'turn left at the light', 'turn right at the light', 'stop at the stop sign',
    'go straight through the intersection', 'merge into the left lane', 'merge into the right lane',
] * 20


def _instructions(clip='train'):
    return [make_instruction(f'a{i}', clip, 0, 10, t) for i, t in enumerate(CORPUS)]


def test_training_sentences_exclude_eval_logs():
    ins = _instructions('train') + _instructions('held-out')
    sents = training_sentences(ins, exclude_logs=('held-out',))
    assert len(sents) == len(CORPUS) and sents[0] == ['turn', 'left', 'at', 'the', 'light']


def test_train_small_corpus_is_deterministic_and_reports():
    cfg = replace(Config(), min_count=1, epochs=5, vector_size=16)
    sents = [t.split() for t in CORPUS]
    kv1, kv2 = train(sents, cfg), train(sents, cfg)
    assert kv1.vector_size == 16 and 'left' in kv1.key_to_index
    assert kv1.index_to_key == kv2.index_to_key and np.array_equal(kv1.vectors, kv2.vectors)
    assert token_coverage(kv1, [['left', 'zebra']]) == 0.5
    md = report(kv1, [['left', 'zebra']])
    assert 'vocabulary' in md and 'cosine(left, right)' in md and '**left**' in md


def test_train_empty_corpus_raises():
    with pytest.raises(ValueError, match='no training sentences'):
        train([], Config())
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_train_word2vec.py -q`
Expected: `ImportError`.

- [ ] **Step 3: Implement `train_word2vec.py`**

```python
"""Train skip-gram word2vec on doPlan instructions and report what it learned (spec 4.2).

Usage:
  PYTHONPATH=scripts python3 -m doplan_lang.train_word2vec \
      --csv nuplan_data/doplan/release_annotationsV1.csv --out-dir outputs/doplan_lang
Evaluation logs (config.EVAL_LOGS) are excluded from training unless --exclude-log is given.

The output is a gensim KeyedVectors file (a pickle tied to the gensim version; large models add a
sidecar .npy next to it). To swap in pretrained vectors later, convert them once with
KeyedVectors.load_word2vec_format(path, binary=...).save('word2vec.kv') and check coverage with
this module's report; pretrained vocabularies are often case-sensitive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gensim.models import KeyedVectors, Word2Vec

from .config import EVAL_LOGS, Config
from .text import Instruction, load_annotations

SEED_WORDS = ('left', 'right', 'stop', 'straight', 'lane', 'merge', 'light', 'sign', 'intersection', 'exit')


def training_sentences(instructions: list[Instruction], exclude_logs) -> list[list[str]]:
    excluded = set(exclude_logs)
    return [list(i.tokens) for i in instructions if i.clip_id not in excluded and i.tokens]


def train(sentences: list[list[str]], cfg: Config) -> KeyedVectors:
    if not sentences:
        raise ValueError('no training sentences left after excluding the held-out logs')
    model = Word2Vec(sentences=sentences, vector_size=cfg.vector_size, window=cfg.window,
                     min_count=cfg.min_count, sg=cfg.sg, negative=cfg.negative, epochs=cfg.epochs,
                     seed=cfg.seed, workers=1)   # workers=1 keeps training reproducible
    return model.wv


def token_coverage(kv: KeyedVectors, sentences: list[list[str]]) -> float:
    tokens = [t for s in sentences for t in s]
    if not tokens:
        return 0.0
    return sum(t in kv.key_to_index for t in tokens) / len(tokens)


def report(kv: KeyedVectors, eval_sentences: list[list[str]], topn: int = 10) -> str:
    lines = ['# word2vec report', '',
             f'- vocabulary: {len(kv.key_to_index)} words, {kv.vector_size} dimensions',
             f'- evaluation-log token coverage: {token_coverage(kv, eval_sentences):.3f}']
    if 'left' in kv.key_to_index and 'right' in kv.key_to_index:
        lines.append(f'- cosine(left, right): {float(kv.similarity("left", "right")):.3f} '
                     '(high is expected in a small corpus; the intent classifier compensates)')
    lines += ['', '## nearest neighbours']
    for w in SEED_WORDS:
        if w in kv.key_to_index:
            nn = ', '.join(f'{n} ({s:.2f})' for n, s in kv.most_similar(w, topn=topn))
            lines.append(f'- **{w}**: {nn}')
        else:
            lines.append(f'- **{w}**: not in vocabulary')
    return '\n'.join(lines) + '\n'


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--exclude-log', action='append', default=None,
                   help='logs to hold out of training; REPLACES config.EVAL_LOGS when given; repeatable')
    a = p.parse_args(argv)
    cfg = Config()
    instructions = load_annotations(a.csv)
    excluded = set(a.exclude_log) if a.exclude_log else set(EVAL_LOGS)
    sentences = training_sentences(instructions, excluded)
    eval_sentences = [list(i.tokens) for i in instructions if i.clip_id in excluded]
    kv = train(sentences, cfg)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    kv.save(str(a.out_dir / 'word2vec.kv'))
    (a.out_dir / 'word2vec_report.md').write_text(report(kv, eval_sentences), encoding='utf-8')
    print(f'trained on {len(sentences)} instructions; vocabulary {len(kv.key_to_index)}; '
          f'saved {a.out_dir / "word2vec.kv"}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_train_word2vec.py -q`
Expected: `3 passed`. (The empty-corpus guard, `Config.sg`/`Config.negative`, the stronger
determinism assertion, the `--exclude-log` help text, `encoding='utf-8'` and the docstring note on
pretrained vectors were added after the Task 2 code-quality review; apply them with Task 3.)

- [ ] **Step 5: Train on the real corpus and read the report**

```bash
PYTHONPATH=scripts python3 -m doplan_lang.train_word2vec \
  --csv nuplan_data/doplan/release_annotationsV1.csv --out-dir outputs/doplan_lang
cat outputs/doplan_lang/word2vec_report.md
```
Expected: 5108 training instructions (5198 minus the 90 on the six evaluation logs), vocabulary about 600 words (the corpus has 996 distinct words, 616 of them occurring at least twice), coverage of evaluation-log tokens above 0.9, and neighbours of `left` that include `right`. Observed 2026-10-23: vocabulary 616, coverage 0.993, cosine(left, right) 0.590, 5.5 s. Keep the report; it is part of the deliverable.

---

### Task 3: Intent classifier (spec 4.3)

**Files:**
- Create: `scripts/doplan_lang/intent.py`
- Test: `scripts/doplan_lang/tests/test_intent.py`

- [ ] **Step 1: Write the failing tests**

```python
from dataclasses import replace

import numpy as np

from doplan_lang.config import Config, INTENTS
from doplan_lang.intent import cross_validate, dataset, predict_proba, sentence_vector, train_classifier
from doplan_lang.text import make_instruction
from doplan_lang.train_word2vec import train

TEMPLATES = {
    'turn_left': ['turn left at the light', 'make a left at the intersection', 'take a left turn'],
    'turn_right': ['turn right at the light', 'make a right at the intersection', 'take a right turn'],
    'stop': ['stop at the stop sign', 'stop for the red light', 'wait at the light'],
    'straight': ['go straight through the intersection', 'keep going straight', 'continue straight'],
    'other': ['exit the casino', 'enter the strip', 'drive toward the tower'],
}


def _corpus():
    rows, i = [], 0
    for intent, texts in TEMPLATES.items():
        for t in texts * 8:
            rows.append(make_instruction(f'a{i}', 'train', 0, 10, t))
            i += 1
    return rows


def test_sentence_vector_mean_and_zero_for_oov():
    cfg = replace(Config(), min_count=1, epochs=5, vector_size=8)
    kv = train([t.split() for texts in TEMPLATES.values() for t in texts], cfg)
    v = sentence_vector(['turn', 'left'], kv)
    assert v.shape == (8,) and np.allclose(v, (kv['turn'] + kv['left']) / 2)
    assert not np.any(sentence_vector(['zebra'], kv))


def test_classifier_recovers_rules_and_handles_oov():
    cfg = replace(Config(), min_count=1, epochs=20, vector_size=16)
    rows = _corpus()
    kv = train([list(r.tokens) for r in rows], cfg)
    X, y, used = dataset(rows, kv, exclude_logs=())
    assert len(used) == len(rows) and set(y) == set(TEMPLATES)
    cv = cross_validate(X, y, seed=0, folds=3)
    assert cv['accuracy'] > 0.9 and set(cv['f1']) == set(TEMPLATES)
    clf = train_classifier(X, y, seed=0)
    p = predict_proba(clf, kv, 'turn left at the light')
    assert set(p) == set(INTENTS) and abs(sum(p.values()) - 1) < 1e-6 and max(p, key=p.get) == 'turn_left'
    assert predict_proba(clf, kv, 'zebra giraffe')['other'] == 1.0


```
(`swap_left_right` lives in `text.py` since Task 1 Step 8 and is tested there.)

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_intent.py -q`
Expected: `ImportError`.

- [ ] **Step 3: Implement `intent.py`**

```python
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

_N_ZERO_ROWS = 20   # synthetic all-OOV rows labelled `other`, so the zero vector maps to `other`


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


def report(cv: dict, clf, kv, rows: list[Instruction], seed: int = 0, n_examples: int = 30) -> str:
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
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_intent.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Train on the real corpus and read the report**

```bash
PYTHONPATH=scripts python3 -m doplan_lang.intent --csv nuplan_data/doplan/release_annotationsV1.csv \
  --kv outputs/doplan_lang/word2vec.kv --out-dir outputs/doplan_lang
sed -n 1,40p outputs/doplan_lang/intent_report.md
```
Expected: CV accuracy against the rules above 0.85 (the classifier is learning keyword-driven labels from embeddings; lower means the embedding is not carrying the signal and must be investigated before continuing). The "clauses the rules could not label" list shows what the embedding adds beyond the rules. Observed 2026-10-23 before the StandardScaler was added: 0.934 (per-class F1 0.87 to 0.96, `other` 0.77).

- [ ] **Step 6 (added after the first Task 3 run): one more rule gap**

The report showed "make a left right here" / "turn left right here" labelled `other` by the rules
(both direction words present). In `text.py`, after computing `left` and `right` in `rule_intent`, add:
```python
    if left and right and re.search(r'\bright here\b', c):
        right = False          # "left, right here": "right here" is a place, not a direction
```
and add to `test_rule_intent`: `('make a left right here', 'turn_left')`, `('turn left right here', 'turn_left')`.
Then rerun Step 5 of this task and record the new CV accuracy.

- [ ] **Step 7 (cleanups from the Task 3 code-quality review, applied with Task 4):** `text.py` gets a
module constant `_RIGHT_HERE = re.compile(r"\bright here\b")` used by the guard above; `intent.py`
gets `if len(set(y)) < 2: raise ValueError(f'need at least two intent classes to cross-validate, got {sorted(set(y))}')`
at the top of `cross_validate`, the comment `# backstop: synthetic all-OOV rows labelled 'other'; predict_proba also short-circuits the zero vector`
on `_N_ZERO_ROWS`, and type hints `report(cv: dict, clf: Pipeline, kv: KeyedVectors, rows: list[Instruction], seed: int = 0, n_examples: int = 30) -> str`;
`tests/test_intent.py` gets
```python
def test_cross_validate_needs_two_classes():
    import pytest
    with pytest.raises(ValueError, match='two intent classes'):
        cross_validate(np.zeros((4, 3)), np.array(['other'] * 4, dtype=object), seed=0, folds=2)
```

Observed 2026-10-23 with the StandardScaler and this rule fix: CV accuracy 0.957; F1 turn_left 0.97, turn_right 0.97, lane_left 0.93,
lane_right 0.98, stop 0.98, straight 0.95, other 0.83; the classifier assigns a non-`other` intent to only
9 of the 423 rule-`other` clauses, and its probabilities are nearly one-hot. Expect the rules-only
ablation in Task 7 to look almost identical; Task 7 therefore also reports the subset of pairs where
classifier and rules disagree.

---

### Task 4: Alignment on the nuPlan clock (spec 4.4)

**Files:**
- Create: `scripts/doplan_lang/align.py`
- Test: `scripts/doplan_lang/tests/test_align.py`

**How the time axis really works (found while executing; replaces the first draft of this task).**
nuPlan's devkit builds a scenario from the lidar_pc rows with `timestamp >= int(anchor + offset*1e6)`,
in time order, taking every second row (subsample 0.5), so ScenarioNet step `i` is lidar frame
`k0 + 2*i`, where `k0` is the first frame at or after the start time. Lidar timestamps jitter by a
fraction of a millisecond, so the frame nominally at the start time is usually just before it and is
excluded. Reproducing the inclusive query on frame indices matches MTR's `gt_trajs` to the recorded
ego_pose to under 1 cm on the toy log; the first draft (`start + 0.1 s * i`) was off by one frame
(45 ms), giving 0.12 to 0.28 m displacement errors. The ScenarioNet "world" frame is the nuPlan frame
translated so the ego sits at (0, 0) at step 0; headings are unchanged.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
import torch

from doplan_lang.align import (AlignmentError, FRAMES_PER_STEP, LidarFrames, build_pairs, coverage_table,
                               list_ego_predictions, log_start_us, prediction_frame, prediction_time_us,
                               prediction_times, sanity_check, scenario_start_frame, window_us)
from doplan_lang.config import Config
from doplan_lang.text import make_instruction

T0 = 1_000_000_000


def frames(n=2000, jitter_us=-100):
    """20 Hz lidar frames; the frame nominally at 30.0 s is `jitter_us` early, like real logs."""
    ts = T0 + np.arange(n, dtype=np.int64) * 50_000
    ts[600] += jitter_us
    tokens = tuple(f'{i:08x}' for i in range(n))
    xy = np.stack([np.arange(n) * 0.25, np.zeros(n)], 1)          # 5 m/s along +x
    return LidarFrames(tokens=tokens, ts=ts, ego_xy=xy, ego_yaw=np.zeros(n),
                       index={t: i for i, t in enumerate(tokens)})


def meta(token_frame=660, offset=-3.0):
    return {'scenario_id': 'bb', 'scenario_token': f'{token_frame:08X}', 'log_name': 'logA',
            'scenario_extraction_info': {'extraction_offset': offset, 'scenario_duration': 15.0}}


def test_scenario_start_frame_reproduces_devkit_inclusive_query():
    assert FRAMES_PER_STEP == 2
    assert scenario_start_frame(meta(), frames(jitter_us=-100)) == 601   # nominal frame is 0.1 ms early: excluded
    assert scenario_start_frame(meta(), frames(jitter_us=0)) == 600      # exactly on time: included
    assert scenario_start_frame(meta(offset=0.0), frames()) == 660
    assert log_start_us(frames()) == T0


def test_prediction_frame_and_time():
    fr = frames()
    assert prediction_frame(601, 20) == 641
    assert prediction_time_us(fr, 601, 20) == int(fr.ts[641]) == T0 + 32_050_000
    with pytest.raises(AlignmentError):
        prediction_time_us(fr, 1990, 20)                                  # runs past the end of the log


def test_unknown_token_or_start_past_end_raises():
    fr = frames()
    with pytest.raises(AlignmentError):
        scenario_start_frame({**meta(), 'scenario_token': 'zz'}, fr)
    with pytest.raises(AlignmentError):
        scenario_start_frame(meta(token_frame=1999, offset=5.0), fr)


def test_window_frames_to_microseconds():
    assert window_us(T0, 0, 430, 0.2) == (T0, T0 + 86_000_000)
    assert window_us(T0, 5, 10, 0.2) == (T0 + 1_000_000, T0 + 2_000_000)


def test_list_ego_predictions_parses_names(tmp_path):
    for n in ['bb___ego___0___.pt', 'bb___ego___25___.pt', 'bb___06a194308ebd5aeb___0___.pt', 'junk.txt']:
        (tmp_path / n).write_bytes(b'')
    assert list_ego_predictions(tmp_path) == [('bb___ego___0___.pt', 'bb', 0), ('bb___ego___25___.pt', 'bb', 25)]


def _pred_times():
    preds = [(f'bb___ego___{i}___.pt', 'bb', i) for i in (0, 5, 10, 15, 20)]   # frames 601, 611, ..., 641
    return prediction_times(preds, {'bb': meta()}, {'logA': frames()}, Config())


def test_prediction_times_table():
    pt = _pred_times()
    assert list(pt.t_pred_us) == [T0 + 30_050_000 + i * 500_000 for i in range(5)]
    assert list(pt.log_name.unique()) == ['logA']


def test_build_pairs_inclusive_window_and_other_log_ignored():
    pt = _pred_times()
    ins = [make_instruction('x1', 'logA', 150, 156, 'Turn left.'),   # 30.0 s .. 31.2 s -> indices 0, 5, 10
           make_instruction('x2', 'logA', 200, 300, 'Stop.'),        # 40 s .. 60 s -> no predictions
           make_instruction('x3', 'logB', 0, 500, 'Stop.')]          # log without predictions
    pairs = build_pairs(ins, pt, {'logA': frames()}, Config())
    assert list(pairs.time_index) == [0, 5, 10] and set(pairs.annotation_id) == {'x1'}
    assert pairs.iloc[0].first_clause == 'turn left' and pairs.iloc[0].rule_intent == 'turn_left'


def test_coverage_table():
    pt = _pred_times()
    ins = [make_instruction('x1', 'logA', 150, 156, 'Turn left.'), make_instruction('x2', 'logA', 200, 300, 'Stop.')]
    cov = coverage_table(ins, pt, {'logA': frames()}, Config()).set_index('annotation_id')
    assert cov.loc['x1', 'n_preds'] == 3 and cov.loc['x1', 'window_s'] == pytest.approx(1.2)
    assert cov.loc['x1', 'frac_covered'] == pytest.approx(1.0)
    assert cov.loc['x2', 'n_preds'] == 0 and cov.loc['x2', 'frac_covered'] == 0.0


def _write_gt(path, xy, heading=0.0):
    gt = np.zeros((81, 10), dtype=np.float32)
    gt[:, 0], gt[:, 1], gt[:, 6], gt[:, 9] = xy[0], xy[1], heading, 1.0
    torch.save({'gt_trajs': gt, 'scenario_id': 'bb', 'object_id': 'ego'}, path)


def test_sanity_check_passes_on_consistent_track_and_fails_otherwise(tmp_path):
    fr = frames()
    _write_gt(tmp_path / 'bb___ego___0___.pt', (0.0, 0.0))
    _write_gt(tmp_path / 'bb___ego___20___.pt', (10.0, 0.0))        # frames 601 -> 641 = 40 * 0.25 m
    out = sanity_check(fr, meta(), tmp_path, Config())
    assert out['displacement_error_m'] < 1e-6 and out['heading_error_rad'] < 1e-6
    _write_gt(tmp_path / 'bb___ego___20___.pt', (9.0, 0.0))         # 1 m off
    with pytest.raises(AlignmentError, match='displacement'):
        sanity_check(fr, meta(), tmp_path, Config())
    _write_gt(tmp_path / 'bb___ego___20___.pt', (10.0, 0.0), heading=0.5)
    _write_gt(tmp_path / 'bb___ego___0___.pt', (0.0, 0.0), heading=0.5)
    with pytest.raises(AlignmentError, match='heading'):
        sanity_check(fr, meta(), tmp_path, Config())
    with pytest.raises(AlignmentError, match='fewer than two'):
        sanity_check(fr, {**meta(), 'scenario_id': 'cc'}, tmp_path, Config())
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_align.py -q`
Expected: `ImportError`.

- [ ] **Step 3: Implement `align.py`**

```python
"""Put doPlan windows and MTR ego predictions on the nuPlan log clock (spec 4.4).

nuPlan's devkit (extract_sensor_tokens_as_scenario + get_sampled_sensor_tokens_in_time_window_from_db)
builds a scenario from the lidar_pc rows with timestamp >= int(anchor + extraction_offset * 1e6), in
time order, every second row (subsample ratio 0.5). ScenarioNet step i is therefore lidar frame
k0 + 2 * i, where k0 is the first frame at or after the start time. Reproducing that inclusive rule on
frame indices matches MTR's gt_trajs to the recorded ego_pose to under 1 cm (toy log, 2026-10-23);
"start + 0.1 s * i" is one frame (45 ms) off because the frame nominally at the start time sits a
fraction of a millisecond before it. The ScenarioNet "world" frame is the nuPlan frame translated so
the ego is at (0, 0) at step 0; headings are unchanged. doPlan frames are 5 per second from the first
lidar_pc timestamp of the log.
"""
from __future__ import annotations

import math
import pickle
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .text import Instruction

FRAMES_PER_STEP = 2        # ScenarioNet subsample_ratio 0.5 of the 20 Hz lidar
PRED_RE = re.compile(r'^(?P<scenario>[0-9a-f]+)___(?P<obj>.+?)___(?P<idx>\d+)___\.pt$')
TIME_COLUMNS = ['pred_file', 'scenario_id', 'log_name', 'time_index', 't_pred_us']
PAIR_COLUMNS = TIME_COLUMNS + ['annotation_id', 'clip_id', 'instruction', 'first_clause', 'rule_intent']
COVERAGE_COLUMNS = ['annotation_id', 'clip_id', 'window_s', 'n_preds', 'covered_s', 'frac_covered']


class AlignmentError(RuntimeError):
    """Predictions cannot be placed on the nuPlan clock. Never skipped silently."""


@dataclass
class LidarFrames:
    """The lidar_pc frames of one log in time order, with the ego_pose linked to each frame."""
    tokens: tuple[str, ...]
    ts: np.ndarray          # int64 microseconds
    ego_xy: np.ndarray      # [n, 2] world metres (nuPlan rear-axle pose)
    ego_yaw: np.ndarray     # [n] radians
    index: dict[str, int]   # token -> frame number


def load_lidar_frames(db_path: Path) -> LidarFrames:
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        rows = con.execute(
            'SELECT lower(hex(l.token)), l.timestamp, e.x, e.y, e.qw, e.qx, e.qy, e.qz '
            'FROM lidar_pc l JOIN ego_pose e ON e.token = l.ego_pose_token ORDER BY l.timestamp').fetchall()
    finally:
        con.close()
    if not rows:
        raise AlignmentError(f'{db_path.name}: no lidar_pc frames')
    tokens = tuple(r[0] for r in rows)
    yaw = np.array([math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                    for _, _, _, _, qw, qx, qy, qz in rows])
    return LidarFrames(tokens=tokens, ts=np.array([r[1] for r in rows], dtype=np.int64),
                       ego_xy=np.array([[r[2], r[3]] for r in rows], dtype=np.float64), ego_yaw=yaw,
                       index={t: i for i, t in enumerate(tokens)})


def log_start_us(frames: LidarFrames) -> int:
    return int(frames.ts[0])


def scenario_start_frame(meta: dict, frames: LidarFrames) -> int:
    """Frame number of ScenarioNet step 0: the first lidar frame at or after anchor + offset."""
    tok = str(meta['scenario_token']).lower()
    if tok not in frames.index:
        raise AlignmentError(f'scenario token {tok} is not a lidar_pc token of log {meta.get("log_name")}')
    offset_s = float(meta['scenario_extraction_info']['extraction_offset'])
    start_us = int(frames.ts[frames.index[tok]] + offset_s * 1e6)       # same truncation as the devkit
    k0 = int(np.searchsorted(frames.ts, start_us, side='left'))         # first frame with ts >= start_us
    if k0 >= len(frames.ts):
        raise AlignmentError(f'scenario {meta.get("scenario_id")}: start lies after the end of the log')
    return k0


def prediction_frame(start_frame: int, time_index: int) -> int:
    return start_frame + FRAMES_PER_STEP * time_index


def prediction_time_us(frames: LidarFrames, start_frame: int, time_index: int) -> int:
    k = prediction_frame(start_frame, time_index)
    if k >= len(frames.ts):
        raise AlignmentError(f'time index {time_index} from frame {start_frame} runs past the end of the log')
    return int(frames.ts[k])


def window_us(t0_us: int, start_frame: int, end_frame: int, frame_s: float = 0.2) -> tuple[int, int]:
    return (t0_us + int(round(start_frame * frame_s * 1e6)),
            t0_us + int(round(end_frame * frame_s * 1e6)))


def list_ego_predictions(pt_dir: Path) -> list[tuple[str, str, int]]:
    """(file name, scenario id, time index) for every ego prediction file, sorted by name."""
    out = []
    for p in sorted(pt_dir.glob('*___ego___*___.pt')):
        m = PRED_RE.match(p.name)
        if m and m.group('obj') == 'ego':
            out.append((p.name, m.group('scenario'), int(m.group('idx'))))
    return out


def load_summary(scenarionet_dir: Path) -> dict[str, dict]:
    """scenario_id -> ScenarioNet metadata. dataset_summary.pkl is our own converter's output."""
    with open(scenarionet_dir / 'dataset_summary.pkl', 'rb') as f:
        summary = pickle.load(f)
    return {str(m['scenario_id']): m for m in summary.values()}


def prediction_times(predictions, summary: dict[str, dict], frames_by_log: dict[str, LidarFrames],
                     cfg: Config) -> pd.DataFrame:
    starts: dict[str, int] = {}
    rows = []
    for fname, sid, idx in predictions:
        if sid not in summary:
            raise AlignmentError(f'{fname}: scenario {sid} is missing from dataset_summary.pkl')
        meta = summary[sid]
        frames = frames_by_log[meta['log_name']]
        if sid not in starts:
            starts[sid] = scenario_start_frame(meta, frames)
        rows.append({'pred_file': fname, 'scenario_id': sid, 'log_name': meta['log_name'],
                     'time_index': idx, 't_pred_us': prediction_time_us(frames, starts[sid], idx)})
    return pd.DataFrame(rows, columns=TIME_COLUMNS)


def _in_window(pred_times: pd.DataFrame, ins: Instruction, frames_by_log, cfg: Config) -> pd.Series:
    a, b = window_us(log_start_us(frames_by_log[ins.clip_id]), ins.start_frame, ins.end_frame, cfg.doplan_frame_s)
    return (pred_times.log_name == ins.clip_id) & (pred_times.t_pred_us >= a) & (pred_times.t_pred_us <= b)


def build_pairs(instructions: list[Instruction], pred_times: pd.DataFrame, frames_by_log, cfg: Config) -> pd.DataFrame:
    """One row per (ego prediction inside the window, annotation). Logs without a .db are skipped."""
    rows = []
    for ins in instructions:
        if ins.clip_id not in frames_by_log:
            continue
        for r in pred_times[_in_window(pred_times, ins, frames_by_log, cfg)].itertuples(index=False):
            rows.append({'pred_file': r.pred_file, 'scenario_id': r.scenario_id, 'log_name': r.log_name,
                         'time_index': r.time_index, 't_pred_us': r.t_pred_us,
                         'annotation_id': ins.annotation_id, 'clip_id': ins.clip_id, 'instruction': ins.text,
                         'first_clause': ins.first_clause, 'rule_intent': ins.rule_intent})
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def coverage_table(instructions: list[Instruction], pred_times: pd.DataFrame, frames_by_log, cfg: Config) -> pd.DataFrame:
    rows = []
    for ins in instructions:
        if ins.clip_id not in frames_by_log:
            continue
        n = int(_in_window(pred_times, ins, frames_by_log, cfg).sum())
        window_s = (ins.end_frame - ins.start_frame) * cfg.doplan_frame_s
        covered = min(n * cfg.pred_stride_s, window_s)
        rows.append({'annotation_id': ins.annotation_id, 'clip_id': ins.clip_id, 'window_s': window_s,
                     'n_preds': n, 'covered_s': covered,
                     'frac_covered': covered / window_s if window_s > 0 else 0.0})
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def _gt_now(path: Path, past_len: int) -> np.ndarray:
    return np.asarray(torch.load(path, map_location='cpu', weights_only=False)['gt_trajs'])[past_len - 1]


def sanity_check(frames: LidarFrames, meta: dict, pt_dir: Path, cfg: Config,
                 tol_m: float = 0.05, tol_rad: float = 0.02) -> dict:
    """Compare one scenario's MTR ego track with the ego_pose of the lidar frames it maps to."""
    sid = str(meta['scenario_id'])
    idxs = sorted(i for _, s, i in list_ego_predictions(pt_dir) if s == sid)
    if len(idxs) < 2:
        raise AlignmentError(f'scenario {sid}: fewer than two ego predictions in {pt_dir}')
    i0, i1 = idxs[0], min(idxs[1:], key=lambda i: abs(i - 20))
    k0 = scenario_start_frame(meta, frames)
    f0, f1 = prediction_frame(k0, i0), prediction_frame(k0, i1)
    if f1 >= len(frames.ts):
        raise AlignmentError(f'scenario {sid}: prediction index {i1} runs past the end of the log')
    g0 = _gt_now(pt_dir / f'{sid}___ego___{i0}___.pt', cfg.past_len)
    g1 = _gt_now(pt_dir / f'{sid}___ego___{i1}___.pt', cfg.past_len)
    disp_err = float(np.linalg.norm((g1[:2] - g0[:2]) - (frames.ego_xy[f1] - frames.ego_xy[f0])))
    yaw_err = abs((float(g0[6]) - float(frames.ego_yaw[f0]) + math.pi) % (2 * math.pi) - math.pi)
    if disp_err > tol_m:
        raise AlignmentError(f'scenario {sid} ({meta.get("log_name")}): ego displacement between indices '
                             f'{i0} and {i1} differs from ego_pose by {disp_err:.3f} m (> {tol_m} m)')
    if yaw_err > tol_rad:
        raise AlignmentError(f'scenario {sid} ({meta.get("log_name")}): heading differs from ego_pose yaw '
                             f'by {yaw_err:.3f} rad (> {tol_rad})')
    return {'scenario_id': sid, 'log_name': meta.get('log_name'), 'start_frame': k0,
            'displacement_error_m': disp_err, 'heading_error_rad': float(yaw_err)}
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_align.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Run the sanity check against the real toy data**

The toy inference was run with the default `debug_max_nb_scenarios=2`, so only 2 of the 6 toy
scenarios have ego predictions; check those two.

```bash
PYTHONPATH=scripts python3 -c "
from pathlib import Path
from doplan_lang.align import list_ego_predictions, load_lidar_frames, load_summary, sanity_check
from doplan_lang.config import Config
sn = Path('scenarionet_data/nuplan_toy'); db = Path('nuplan_data/nuplan-v1.1/splits/mini/2021.08.24.13.12.55_veh-45_00386_00472.db')
fr = load_lidar_frames(db); have = {s for _, s, _ in list_ego_predictions(sn / 'output_MTR')}
for sid, meta in load_summary(sn).items():
    print(sanity_check(fr, meta, sn / 'output_MTR', Config()) if sid in have else f'{sid}: no ego predictions')"
```
Expected: two dicts with `displacement_error_m` below 0.01 and `heading_error_rad` below 0.001, four "no ego predictions" lines, no exception.
Observed 2026-10-23: start frames 625 and 647, displacement errors 0.0005 m and 0.0043 m, heading errors about 4e-5 rad.

- [ ] **Step 6 (added after the Task 4 spec review): overlapping scenarios share moments**

The converter's scenarios overlap heavily (the six toy scenarios start within 2 s of each other and
last 15 to 20 s), so one moment of driving yields several MTR predictions and `coverage_table`
overstated coverage by counting all of them. Each pair now carries a `moment` key (log name plus the
prediction time rounded to the 0.5 s stride), coverage counts distinct moments, and Task 7's
bootstrap resamples moments rather than pairs. Changes in `align.py`:

```python
PAIR_COLUMNS = TIME_COLUMNS + ['moment', 'annotation_id', 'clip_id', 'instruction', 'first_clause', 'rule_intent']


def moment_key(log_name: str, t_pred_us: int, stride_s: float) -> str:
    """Identifies one instant of one drive; predictions from overlapping scenarios share it."""
    return f'{log_name}:{int(round(t_pred_us / (stride_s * 1e6)))}'
```
In `prediction_times`, add `'moment': moment_key(meta['log_name'], t, cfg.pred_stride_s)` to each row
(compute `t = prediction_time_us(...)` once) and add `'moment'` to `TIME_COLUMNS` after `'t_pred_us'`.
In `build_pairs`, copy `r.moment` into the pair row. In `coverage_table`, replace the count with
`n = int(pred_times.loc[_in_window(...), 'moment'].nunique())` and rename the column `n_preds` to
`n_moments` (also in `COVERAGE_COLUMNS`).

Tests: in `test_prediction_times_table` add `assert pt.moment.nunique() == 5 and pt.moment.iloc[0] == 'logA:2060'`
(T0 + 30.05 s = 1030.05 s, divided by the 0.5 s stride and rounded). Add a test that two predictions from
different scenarios at the same instant share a moment:
```python
def test_same_instant_from_two_scenarios_shares_a_moment():
    fr = frames()
    summary = {'bb': meta(), 'cc': {**meta(token_frame=670), 'scenario_id': 'cc'}}   # cc starts 10 frames (= 5 steps) later
    preds = [('bb___ego___5___.pt', 'bb', 5), ('cc___ego___0___.pt', 'cc', 0)]
    pt = prediction_times(preds, summary, {'logA': fr}, Config())
    assert pt.moment.nunique() == 1
    ins = [make_instruction('x1', 'logA', 150, 156, 'Turn left.')]
    assert len(build_pairs(ins, pt, {'logA': fr}, Config())) == 2                   # both predictions are pairs
    assert coverage_table(ins, pt, {'logA': fr}, Config()).iloc[0].n_moments == 1  # but one moment of coverage
```
and rename `n_preds` to `n_moments` in `test_coverage_table`. Run the suite: expect 72 passed, and
rerun Step 5 (unchanged numbers).

- [ ] **Step 8 (added during the real run, 2026-10-24): compare vehicle centres in `sanity_check`**

On the six real logs the check failed 174 of 411 scenarios (errors up to 1.5 m, concentrated on lane
changes and high lateral acceleration; no frame shift fixed them). Cause: nuPlan's `ego_pose` is the
rear axle, ScenarioNet records the vehicle centre, and the two displacements differ whenever the heading
changes (the toy scenarios barely turned, so they passed). `align.py` gains
`rear_axle_to_centre_m(meta)` = (front_length - rear_length) / 2 from `ego_vehicle_parameters` (1.461 m
for the Pacifica; 0.0 when absent) and `ego_centre_xy(frames, frame, offset_m)`; `sanity_check` compares
centre displacements. With that, all 411 scenarios agree to within 3 mm (median 0.1 mm). Test:
`test_sanity_check_compares_vehicle_centres_when_heading_changes` (a 90 degree turn between the two
frames: centre comparison passes, rear-axle comparison is 2 m off).

- [ ] **Step 7 (added after the Task 4 code-quality review): bucket moments by lidar frame, not by timestamp**

On real data scenario starts have arbitrary 20 Hz phases, so rounding timestamps to a 0.5 s grid split
predictions 100 ms apart into different moments and merged ones 400 ms apart (31 moments from 58 toy
predictions). Bucket by frame index instead: every bucket is exactly 10 frames wide and immune to jitter.

```python
MOMENT_FRAMES = FRAMES_PER_STEP * 5     # one 0.5 s bucket = 10 lidar frames
TIME_COLUMNS = ['pred_file', 'scenario_id', 'log_name', 'time_index', 'frame', 't_pred_us', 'moment']


def moment_key(log_name: str, frame: int) -> str:
    """Identifies one 0.5 s bucket of one drive; predictions from overlapping scenarios share it."""
    return f'{log_name}:{frame // MOMENT_FRAMES}'
```
`prediction_times` computes `k = prediction_frame(starts[sid], idx)`, stores it as `'frame': k`, and uses
`moment_key(meta['log_name'], k)`; `prediction_time_us` keeps its range check. Also:
- `prediction_times` raises `AlignmentError(f'{fname}: no lidar frames for log {log}')` when the log is
  missing from `frames_by_log` (instead of a bare KeyError).
- `LidarFrames` becomes `@dataclass(frozen=True)` with `index` derived in `__post_init__` via
  `object.__setattr__(self, 'index', {t: i for i, t in enumerate(self.tokens)})`; `index` is declared
  `index: dict[str, int] = field(init=False, repr=False)`; callers no longer pass it.
- `sanity_check(frames, meta, pt_dir, cfg, tol_m, tol_rad, predictions=None)`: when `predictions` (the
  list from `list_ego_predictions`) is given, use it instead of listing the directory again.
Tests: the fixture stops passing `index`; `test_prediction_times_table` asserts
`list(pt.frame) == [601, 611, 621, 631, 641]` and `pt.moment.iloc[0] == 'logA:60'` (601 // 10) and
`pt.moment.nunique() == 5`; replace the shared-moment test with one where the second scenario starts
2 frames later (`token_frame=663`; bb's nominal start frame 600 is jittered early and excluded, so bb
starts at 601 while cc starts at 603): the two predictions `('bb___ego___5___.pt', 'bb', 5)` (frame 611)
and `('cc___ego___5___.pt', 'cc', 5)` (frame 613) share a moment, and `('cc___ego___10___.pt', 'cc', 10)`
(frame 623) does not; add a `KeyError`-free check that a log missing from `frames_by_log` raises
`AlignmentError`. Run the suite: expect 73 passed.

---

### Task 5: Geometry features and compatibility (spec 4.5)

**Files:**
- Create: `scripts/doplan_lang/geometry.py`
- Test: `scripts/doplan_lang/tests/test_geometry.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from doplan_lang.config import Config, INTENTS
from doplan_lang.geometry import compat, compat_table, future_features, mode_features, to_ego_frame, traj_features


def arc(radius, angle_deg, n=60, left=True):
    """n waypoints along a circular arc starting at the origin heading +x."""
    th = np.linspace(0, np.radians(angle_deg), n + 1)[1:]
    x = radius * np.sin(th)
    y = radius * (1 - np.cos(th)) * (1 if left else -1)
    return np.stack([x, y], 1)


def straight(n=60, step=0.5):
    return np.stack([np.arange(1, n + 1) * step, np.zeros(n)], 1)


def gt(heading=0.0, origin=(0.0, 0.0)):
    g = np.zeros((81, 10), dtype=np.float32)
    g[:, 0], g[:, 1], g[:, 6], g[:, 9] = origin[0], origin[1], heading, 1.0
    return g


def test_to_ego_frame_rotates_and_translates():
    out = to_ego_frame(np.array([[1.0, 1.0]]), origin_xy=(1.0, 0.0), heading=np.pi / 2)
    assert np.allclose(out, [[1.0, 0.0]])          # 1 m ahead along a +y heading
    left = to_ego_frame(np.array([[-1.0, 0.0]]), origin_xy=(0.0, 0.0), heading=np.pi / 2)
    assert np.allclose(left, [[0.0, 1.0]])         # heading +y, a point at -x is 1 m to the LEFT


def test_traj_features_left_arc_and_mirror():
    lat, dpsi, path = traj_features(arc(20, 90))
    assert lat > 2 and 80 < dpsi < 100 and abs(path - 20 * np.pi / 2) < 0.5
    lat_r, dpsi_r, _ = traj_features(arc(20, 90, left=False))
    assert lat_r == pytest.approx(-lat) and dpsi_r == pytest.approx(-dpsi)


def test_traj_features_stationary():
    assert traj_features(np.zeros((60, 2))) == (0.0, 0.0, 0.0)


def test_mode_features_uses_ego_frame():
    heading, origin = np.pi / 2, (100.0, 50.0)
    c, s = np.cos(heading), np.sin(heading)
    a = arc(20, 90)
    world = np.stack([c * a[:, 0] - s * a[:, 1] + origin[0], s * a[:, 0] + c * a[:, 1] + origin[1]], 1)
    feats = mode_features(world[None], gt(heading, origin), past_len=21)
    assert feats.shape == (1, 3) and feats[0, 0] > 2 and feats[0, 1] > 80


def test_future_features_uses_valid_rows_only():
    g = gt()
    g[21:, 0] = np.arange(1, 61) * 0.5      # straight ahead, 0.5 m per step
    g[51:, 9] = 0.0                          # last 30 future rows invalid
    lat, dpsi, path, n = future_features(g, 21)
    assert n == 30 and lat == 0 and dpsi == 0 and path == pytest.approx(15.0)   # 30 steps of 0.5 m from the ego


def test_compat_table_prefers_matching_intent():
    cfg = Config()
    feats = np.array([traj_features(arc(20, 90)), traj_features(arc(20, 90, left=False)),
                      traj_features(straight()), traj_features(np.zeros((60, 2))), [4.0, 0.0, 30.0]])
    t = compat_table(feats, cfg)
    i = {k: j for j, k in enumerate(INTENTS)}
    assert t.shape == (5, 7) and (t >= cfg.compat_floor).all() and (t <= 1).all()
    assert t[0, i['turn_left']] > 0.9 and t[0, i['turn_right']] == cfg.compat_floor
    assert t[1, i['turn_right']] > 0.9
    assert t[2, i['straight']] > 0.6 and t[2, i['turn_left']] == cfg.compat_floor
    assert t[3, i['stop']] > 0.8
    assert t[4, i['lane_left']] > 0.8 and t[4, i['turn_left']] < 0.1
    assert (t[:, i['other']] == 1).all()


def test_compat_mixes_by_probability():
    cfg = Config()
    feats = np.array([[0.0, 0.0, 30.0]])
    assert compat(feats, {'other': 1.0}, cfg)[0] == 1.0
    half = compat(feats, {'straight': 0.5, 'other': 0.5}, cfg)[0]
    assert 0.5 < half < 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_geometry.py -q`
Expected: `ImportError`.

- [ ] **Step 3: Implement `geometry.py`**

```python
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
    return expit(np.asarray(x, dtype=np.float64))     # scipy's logistic: no overflow warnings for huge inputs


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

    not_turning_lane = 1 - above(np.abs(dpsi), cfg.lane_heading_deg, 10.0)
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


def compat(features: np.ndarray, probs: dict[str, float], cfg: Config) -> np.ndarray:
    """Expected compatibility of each mode under the intent distribution `probs`."""
    p = np.array([probs.get(k, 0.0) for k in INTENTS], dtype=np.float64)
    return compat_table(features, cfg) @ p
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_geometry.py -q`
Expected: `7 passed`. (`expit` and the lateral ego-frame case were added after the Task 5 code-quality review.)

(Execution note 2026-10-23: the first draft summed the path over the waypoints only, which drops the
segment from the ego to the first waypoint (up to 1.5 m at 15 m/s against a 3 m stop threshold); the
code above prepends the origin. On the toy predictions, modes with dpsi near -178 deg are reversing
or U-turn modes with near-zero MTR score; the formulas give them low turn compatibility, which is fine.)

---

### Task 6: Gate, fusion and the re-ranking table (spec 4.6)

**Files:**
- Create: `scripts/doplan_lang/rerank.py`
- Modify: `scripts/doplan_lang/tests/conftest.py` (add the fake-prediction helper)
- Test: `scripts/doplan_lang/tests/test_rerank.py`

- [ ] **Step 1: Add a fake-prediction helper to `conftest.py`**

Append to `scripts/doplan_lang/tests/conftest.py`:
```python
import numpy as np
import pytest
import torch


def _arc(radius, angle_deg, n=60, left=True):
    th = np.linspace(0, np.radians(angle_deg), n + 1)[1:]
    return np.stack([radius * np.sin(th), radius * (1 - np.cos(th)) * (1 if left else -1)], 1)


def write_fake_prediction(path, scores=(0.3, 0.3, 0.4), heading=0.0, origin=(0.0, 0.0), future='straight'):
    """Three modes (left arc, right arc, straight) around a fake ego; the recorded future is
    the mode named by `future`. Mirrors the fields predict.py writes."""
    modes = {'left': _arc(20, 90), 'right': _arc(20, 90, left=False),
             'straight': np.stack([np.arange(1, 61) * 0.5, np.zeros(60)], 1)}
    c, s = np.cos(heading), np.sin(heading)

    def to_world(xy):
        return np.stack([c * xy[:, 0] - s * xy[:, 1] + origin[0], s * xy[:, 0] + c * xy[:, 1] + origin[1]], 1)

    pred_trajs = np.stack([to_world(modes['left']), to_world(modes['right']), to_world(modes['straight'])]).astype(np.float32)
    gt = np.zeros((81, 10), dtype=np.float32)
    gt[:, 0], gt[:, 1], gt[:, 6], gt[:, 9] = origin[0], origin[1], heading, 1.0
    gt[21:, :2] = to_world(modes[future])
    torch.save({'scenario_id': 'fake', 'pred_trajs': pred_trajs, 'pred_scores': np.array(scores, dtype=np.float32),
                'past_det_scores': np.float32(1.0), 'object_id': 'ego', 'object_type': 1, 'gt_trajs': gt,
                'track_index_to_predict': 0, 'current_time_index': 0}, path)
    return path


@pytest.fixture
def fake_pred_dir(tmp_path):
    d = tmp_path / 'output_MTR'
    d.mkdir()
    write_fake_prediction(d / 'fake___ego___0___.pt')
    return d
```

- [ ] **Step 2: Write the failing tests**

`scripts/doplan_lang/tests/test_rerank.py`:
```python
import numpy as np
import pandas as pd
import pytest
import torch

from doplan_lang.align import PAIR_COLUMNS
from doplan_lang.config import Config
from doplan_lang.geometry import mode_features
from doplan_lang.rerank import fuse, gate_open, load_prediction, one_hot, rerank_pairs, wrap_deg
from doplan_lang.text import rule_intent

from .conftest import write_fake_prediction


def test_fuse_identity_cases():
    s = np.array([0.5, 0.3, 0.2])
    assert np.allclose(fuse(s, np.ones(3), lam=1.0), s)              # flat compatibility
    assert np.allclose(fuse(s * 0.5, np.array([0.1, 0.9, 0.5]), lam=0.0), s)   # lambda 0 only normalises


def test_fuse_raises_compatible_mode():
    new = fuse(np.array([0.4, 0.3, 0.3]), np.array([0.02, 1.0, 0.02]), lam=1.0)
    assert new.argmax() == 1 and abs(new.sum() - 1) < 1e-9


def test_gate_open_only_when_plausible_modes_fork():
    cfg = Config()
    scores = np.array([0.5, 0.3, 0.2])
    same = np.zeros((3, 60, 2)); same[:, :, 0] = np.arange(1, 61) * 0.5
    feats_same = np.zeros((3, 3))
    assert not gate_open(same, feats_same, scores, cfg)
    forked = same.copy(); forked[1, -1, 1] = 5.0                     # one endpoint 5 m to the left
    assert gate_open(forked, feats_same, scores, cfg)
    feats_turn = feats_same.copy(); feats_turn[1, 1] = 45.0           # one mode turning 45 degrees
    assert gate_open(same, feats_turn, scores, cfg)
    junk = np.array([0.98, 0.01, 0.01])                               # the forking modes are implausible
    assert not gate_open(forked, feats_turn, junk, cfg)
    wrap = feats_same.copy(); wrap[:, 1] = [178.0, -178.0, 180.0]      # all within 4 degrees once wrapped
    assert not gate_open(same, wrap, scores, cfg)
    assert wrap_deg(356.0) == pytest.approx(-4.0) and wrap_deg(-190.0) == pytest.approx(170.0)


def test_one_hot():
    assert one_hot('stop') == {'turn_left': 0.0, 'turn_right': 0.0, 'lane_left': 0.0, 'lane_right': 0.0,
                               'stop': 1.0, 'straight': 0.0, 'other': 0.0}


def test_rerank_pairs_moves_top1_and_writes_pt(fake_pred_dir, tmp_path):
    cfg = Config()
    pairs = pd.DataFrame([{'pred_file': 'fake___ego___0___.pt', 'scenario_id': 'fake', 'log_name': 'logA',
                           'time_index': 0, 't_pred_us': 0, 'moment': 'logA:0', 'annotation_id': 'a1', 'clip_id': 'logA',
                           'instruction': 'Turn left at the light.', 'first_clause': 'turn left at the light',
                           'rule_intent': 'turn_left'}], columns=PAIR_COLUMNS)
    out_pt = tmp_path / 'reranked_pt'
    df = rerank_pairs(pairs, fake_pred_dir, cfg, lambda c: one_hot(rule_intent(c)), out_pt)
    row = df.iloc[0]
    assert row.gated and row.old_top1 == 2 and row.new_top1 == 0 and row.cf_top1 == 1
    assert row.intent == 'turn_left' and row.p_turn_left == 1.0
    assert row.new_top1_lat > 2 and row.cf_top1_lat < -2 and row.fut_valid == 60
    assert abs(sum(row[f's_new_{k}'] for k in range(3)) - 1) < 1e-9
    saved = load_prediction(out_pt / 'fake___ego___0____a1.pt')
    assert np.allclose(saved['pred_scores_mtr'], [0.3, 0.3, 0.4]) and saved['pred_scores'].argmax() == 0
    feats = mode_features(saved['pred_trajs'], saved['gt_trajs'], cfg.past_len)
    assert np.allclose(feats[:, 0], [row.lat_0, row.lat_1, row.lat_2])


def _pair_row(**over):
    base = {'pred_file': 'fake___ego___0___.pt', 'scenario_id': 'fake', 'log_name': 'logA', 'time_index': 0,
            'frame': 0, 't_pred_us': 0, 'moment': 'logA:0', 'annotation_id': 'a2', 'clip_id': 'logA',
            'instruction': 'Exit the casino.', 'first_clause': 'exit the casino', 'rule_intent': 'other'}
    return pd.DataFrame([{**base, **over}], columns=PAIR_COLUMNS)


def test_other_intent_leaves_ranking_with_open_gate(fake_pred_dir):
    row = rerank_pairs(_pair_row(), fake_pred_dir, Config(), lambda c: one_hot(rule_intent(c))).iloc[0]
    assert row.gated and row.old_top1 == row.new_top1 == 2 and row.intent == 'other'
    assert np.allclose([row.s_new_0, row.s_new_1, row.s_new_2], [0.3, 0.3, 0.4])   # flat compatibility: normalised MTR scores


def test_closed_gate_leaves_ranking(tmp_path):
    d = tmp_path / 'output_MTR'
    d.mkdir()
    write_fake_prediction(d / 'fake___ego___0___.pt', scores=(0.02, 0.02, 0.96))    # only the straight mode is plausible
    row = rerank_pairs(_pair_row(instruction='Turn left.', first_clause='turn left', rule_intent='turn_left'),
                       d, Config(), lambda c: one_hot(rule_intent(c))).iloc[0]
    assert not row.gated and row.old_top1 == row.new_top1 == 2 and row.intent == 'turn_left'
    assert np.allclose([row.s_new_0, row.s_new_1, row.s_new_2], [0.02, 0.02, 0.96])
```

- [ ] **Step 3: Run to verify failure**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_rerank.py -q`
Expected: `ImportError`.

- [ ] **Step 4: Implement `rerank.py`**

```python
"""Decision-point gate and log-linear fusion of MTR mode scores with intent compatibility (spec 4.6).

new_scores = softmax( log(mtr_score) + lambda * log(compatibility) ), applied only where the
modes fork (gate open). Fused scores are normalised; MTR's own scores are kept as written.
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
    fut_lat/fut_dpsi/fut_path/fut_valid. With out_pt_dir, writes a copy of each prediction
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
            torch.save(out, out_pt_dir / f'{r.pred_file[:-3]}_{r.annotation_id}.pt')
    return pd.DataFrame(rows)
```
Note the output name: `fake___ego___0___.pt` minus `.pt` is `fake___ego___0___`, plus `_a1.pt` gives `fake___ego___0____a1.pt`, which is what the test expects.

(Execution note 2026-10-23: on the 58 toy predictions with a fabricated "turn left" instruction the gate
was open on all 58 but no top-1 changed: MTR's median top score there is 0.96 and only 5 predictions
have any mode more than 2 m to the left, so at lambda = 1 a compatibility ratio of at most 0.5/0.02
cannot overturn a 0.96 versus 0.01 score ratio. This is the intended conservative behaviour; Task 11
sweeps lambda to show its effect on the real pairs.)

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_rerank.py -q`
Expected: `7 passed`. (The plausible-mode gate, `wrap_deg`, `Config.gate_min_score` and the two
identity tests were added after the Task 6 spec review: on the toy data the first gate opened on all 58
predictions because modes with MTR scores near 1e-6 created the spread.)

---

### Task 7: Metrics, bootstrap CIs, markdown table, example plots (spec 5)

**Files:**
- Create: `scripts/doplan_lang/evaluate.py`
- Test: `scripts/doplan_lang/tests/test_evaluate.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
import math

import numpy as np
import pandas as pd
import pytest

from doplan_lang.align import PAIR_COLUMNS
from doplan_lang.config import Config
from doplan_lang.evaluate import (add_errors, ade_fde, bootstrap_diff, compliance_frame, plot_examples,
                                  summarize, write_markdown)
from doplan_lang.rerank import one_hot, rerank_pairs
from doplan_lang.text import rule_intent

from .conftest import write_fake_prediction


def test_ade_fde_basic_and_invalid():
    mode = np.array([[1.0, 0.0], [2.0, 0.0]])
    fut = np.array([[1.0, 1.0], [2.0, 2.0]])
    assert ade_fde(mode, fut, np.array([1, 1])) == (1.5, 2.0)
    assert ade_fde(mode, fut, np.array([1, 0])) == (1.0, 1.0)
    assert all(math.isnan(v) for v in ade_fde(mode, fut, np.array([0, 0])))


def test_bootstrap_diff_is_paired_and_drops_nan():
    r = bootstrap_diff([1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5], n=200, seed=0)
    assert r == {'mean': -0.5, 'lo': -0.5, 'hi': -0.5, 'n': 4, 'n_groups': 4}
    assert bootstrap_diff([math.nan], [1.0])['n'] == 0


def test_bootstrap_diff_resamples_groups():
    before = [0.0] * 6
    after = [1.0, 1.0, 1.0, 1.0, 1.0, -5.0]            # one bad moment with one pair, one good moment with five
    groups = ['good'] * 5 + ['bad']
    r = bootstrap_diff(before, after, n=500, seed=0, groups=groups)
    assert r['n_groups'] == 2 and r['mean'] == pytest.approx(0.0)
    assert r['lo'] == pytest.approx(-5.0) and r['hi'] == pytest.approx(1.0)   # draws are all-good or all-bad or mixed


def _pair(pred_file, ann, text):
    return {'pred_file': pred_file, 'scenario_id': 'fake', 'log_name': 'logA', 'time_index': 0, 't_pred_us': 0,
            'moment': f'logA:{pred_file}', 'annotation_id': ann, 'clip_id': 'logA', 'instruction': text,
            'first_clause': text.lower().rstrip('.'), 'rule_intent': rule_intent(text)}


def _reranked(tmp_path):
    d = tmp_path / 'output_MTR'
    d.mkdir()
    write_fake_prediction(d / 'left___ego___0___.pt', future='left')          # the driver turned left
    write_fake_prediction(d / 'straight___ego___0___.pt', future='straight')  # the driver went straight
    pairs = pd.DataFrame([_pair('left___ego___0___.pt', 'a1', 'Turn left at the light.'),
                          _pair('straight___ego___0___.pt', 'a2', 'Turn left at the light.'),
                          _pair('straight___ego___0___.pt', 'a3', 'Exit the casino.')], columns=PAIR_COLUMNS)
    cfg = Config()
    df = rerank_pairs(pairs, d, cfg, lambda c: one_hot(rule_intent(c)))
    return add_errors(df, d, cfg), d, cfg


def test_add_errors_and_compliance(tmp_path):
    df, _, cfg = _reranked(tmp_path)
    a1 = df[df.annotation_id == 'a1'].iloc[0]
    assert a1.ade_new < a1.ade_old
    assert a1.ade_new == pytest.approx(0.0, abs=1e-4) and a1.min_ade == pytest.approx(0.0, abs=1e-4)
    a3 = df[df.annotation_id == 'a3'].iloc[0]
    assert a3.old_top1 == a3.new_top1 == 2                      # intent `other` leaves the ranking alone
    comp = compliance_frame(df, cfg)
    assert list(comp.annotation_id) == ['a1']                   # a2: future went straight; a3: no lateral intent
    assert not comp.iloc[0].old_ok and comp.iloc[0].new_ok and comp.iloc[0].cf_ok


def test_summarize_markdown_and_plots(tmp_path):
    df, pt_dir, cfg = _reranked(tmp_path)
    m = summarize(df, cfg)
    assert m['n_pairs'] == 3 and m['n_predictions'] == 2 and m['n_moments'] == 2 and m['gate_open_fraction'] == 1.0
    g = m['gated']
    assert g['compliance_n'] == 1 and g['compliance_before'] == 0.0 and g['compliance_after'] == 1.0
    assert g['counterfactual_consistency'] == 1.0 and g['n_with_future'] == 3
    assert m['intent_disagrees_with_rules_fraction'] == 0.0 and m['gated_disagree']['n_pairs'] == 0
    # re-ranking fixed a1 and broke a2 by the same amount, so the mean top-1 ADE is unchanged
    assert m['all']['ade_top1_after'] == pytest.approx(m['all']['ade_top1_before'], abs=1e-3)
    json.dumps(m)                                               # everything must be JSON-serialisable
    md = tmp_path / 'metrics.md'
    write_markdown({'classifier': m}, md, cfg)
    assert '| classifier | gated |' in md.read_text()
    pngs = plot_examples(df, pt_dir, tmp_path / 'plots', cfg)
    assert len(pngs) == 3 and all(p.exists() for p in pngs)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_evaluate.py -q`
Expected: `ImportError`.

- [ ] **Step 3: Implement `evaluate.py`**

```python
"""Metrics and plots for re-ranked (prediction, annotation) pairs (spec 5).

Unit = one pair. Headline numbers are over gated pairs; "all" is reported for context.
Compliance follows doPlan section V-B: among pairs with a lateral intent whose recorded 6 s
endpoint lies >= compliance_m toward the instructed side (ego frame), the fraction whose
top-1 mode endpoint also does. Counterfactual consistency additionally requires that the
left/right-swapped instruction picks a mode whose endpoint lies on the opposite side.
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


def ade_fde(mode_xy, future_xy, valid) -> tuple[float, float]:
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


def compliance_frame(df: pd.DataFrame, cfg: Config, intent_col: str = 'intent') -> pd.DataFrame:
    """Gated pairs with a lateral intent whose recorded future moved toward the instructed side."""
    sub = df[df.gated & df[intent_col].isin(LATERAL_INTENTS) & (df.fut_valid > 0)].copy()
    sgn = sub[intent_col].map(side_sign)
    sub = sub[sgn * sub.fut_lat >= cfg.compliance_m].copy()
    sgn = sub[intent_col].map(side_sign)
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
        m['n_moments'] = int(sub.moment.nunique()) if len(sub) else 0
        out[name] = m
    return out


def _fmt(x) -> str:
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.3f}'


def write_markdown(metrics: dict[str, dict], path: Path, cfg: Config) -> None:
    lines = ['# doPlan language re-ranking: results', '',
             f'lambda = {cfg.lam}; gate opens when mode endpoints spread > {cfg.gate_endpoint_m} m '
             f'or headings spread > {cfg.gate_heading_deg} deg; compliance threshold {cfg.compliance_m} m.', '',
             '| variant | subset | pairs | top-1 ADE before | after | diff [95% CI] | minADE | '
             'top-1 FDE before | after | compliance n | before | after | counterfactual |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for variant, m in metrics.items():
        for subset in ('gated', 'all', 'gated_disagree'):
            s, ci = m[subset], m[subset]['ade_diff_ci']
            lines.append(f"| {variant} | {subset} | {s['n_with_future']} | {_fmt(s['ade_top1_before'])} | "
                         f"{_fmt(s['ade_top1_after'])} | {_fmt(ci['mean'])} [{_fmt(ci['lo'])}, {_fmt(ci['hi'])}] | "
                         f"{_fmt(s['min_ade'])} | {_fmt(s['fde_top1_before'])} | {_fmt(s['fde_top1_after'])} | "
                         f"{s['compliance_n']} | {_fmt(s['compliance_before'])} | {_fmt(s['compliance_after'])} | "
                         f"{_fmt(s['counterfactual_consistency'])} |")
    first = next(iter(metrics.values()))
    lines += ['', f"{first['n_pairs']} pairs over {first['n_predictions']} ego predictions, "
                  f"{first['n_moments']} distinct 0.5 s moments and {first['n_annotations']} annotations "
                  f"(CIs resample moments); gate open on {_fmt(first['gate_open_fraction'])} of pairs; "
                  f"classifier intent differs from the rule label on "
                  f"{_fmt(first['intent_disagrees_with_rules_fraction'])} of pairs ('gated_disagree' rows)."]
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
    chosen = df.assign(_change=np.abs(new - old).sum(axis=1)).sort_values('_change', ascending=False)
    paths = []
    for n, r in enumerate(chosen.head(cfg.n_example_plots).itertuples(index=False)):
        p = load_prediction(pt_dir / r.pred_file)
        g, modes = np.asarray(p['gt_trajs']), np.asarray(p['pred_trajs'])
        hist, fut = g[:cfg.past_len], g[cfg.past_len:]
        hv, fv = hist[:, -1] > 0, fut[:, -1] > 0
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(hist[hv, 0], hist[hv, 1], 'k-', lw=3, label='history')
        if fv.any():
            ax.plot(fut[fv, 0], fut[fv, 1], 'k--', lw=3, label='recorded future')
        for i, m in enumerate(modes):
            style = {'lw': 3} if i == r.new_top1 else {'lw': 1.2, 'alpha': 0.7}
            ax.plot(m[:, 0], m[:, 1], label=f'mode {i}: MTR {getattr(r, f"s_old_{i}"):.2f} -> fused '
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
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    write_markdown(metrics, a.out_dir / 'metrics.md', cfg)
    plot_examples(df, a.pt_dir, a.out_dir / 'plots', cfg)
    print((a.out_dir / 'metrics.md').read_text())


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_evaluate.py -q`
Expected: `5 passed`.

- [ ] **Step 5 (added after the Task 7 spec review): compliance population, CIs in the table, JSON NaN**

Findings: `compliance_frame` always filtered on `gated`, so the "all" row's compliance was really the
gated one; the compliance CI was in metrics.json but not in the table; the compliance population was
chosen by each variant's own intent, so classifier and rules-only rows counted different pairs; FDE and
`fut_lat` use the last valid step, so a short future is not a 6 s endpoint; `json.dumps` writes bare
`NaN`. Changes in `evaluate.py`:

```python
def compliance_frame(df: pd.DataFrame, cfg: Config, intent_col: str = 'rule_intent') -> pd.DataFrame:
    """Pairs whose instruction (by the keyword rule label, so every variant is scored on the same pairs)
    is lateral, whose future covers the full horizon, and whose recorded endpoint moved >= compliance_m
    toward the instructed side. The caller chooses the subset (gated or not)."""
    sub = df[df[intent_col].isin(LATERAL_INTENTS) & (df.fut_valid >= FUTURE_STEPS)].copy()
    ...  # the rest unchanged (sgn from intent_col; old_ok / new_ok / cf_ok)
```
with `FUTURE_STEPS = 60` as a module constant. In `summarize`, pass the subset as before (so "gated" is
gated-only, "all" is all) and add `m['compliance_intent'] = 'rule_intent'`. Add a top-level
`out['pairs_per_log'] = df.groupby('log_name').size().to_dict()` so a single multi-annotator window
that dominates the pairs is visible. Add
```python
def _json_safe(x):
    """NaN -> None so metrics.json is strict JSON."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return x
```
and use `json.dumps(_json_safe(metrics), indent=2)` in `main` (and in run_all). In `write_markdown`,
add a column `compliance diff [95% CI]` after `after` using `m[subset]['compliance_diff_ci']`, and a
line under the table: "compliance population: rule-labelled lateral instructions with a full 6 s
recorded future; counted on both variants identically".

Tests: `test_add_errors_and_compliance` keeps passing (the fake futures have 60 valid steps and the
rule labels match); add
```python
def test_compliance_population_uses_rule_label_and_full_future(tmp_path):
    df, _, cfg = _reranked(tmp_path)
    df.loc[df.annotation_id == 'a1', 'intent'] = 'other'          # the classifier disagrees: still counted
    assert list(compliance_frame(df, cfg).annotation_id) == ['a1']
    df.loc[df.annotation_id == 'a1', 'fut_valid'] = 59            # short future: excluded
    assert compliance_frame(df, cfg).empty


def test_json_safe_and_markdown_ci_column(tmp_path):
    from doplan_lang.evaluate import _json_safe
    assert _json_safe({'a': math.nan, 'b': [1.0, math.nan], 'c': {'d': math.nan}}) == {'a': None, 'b': [1.0, None], 'c': {'d': None}}
    df, _, cfg = _reranked(tmp_path)
    m = summarize(df, cfg)
    assert m['pairs_per_log'] == {'logA': 3} and m['gated']['compliance_intent'] == 'rule_intent'
    md = tmp_path / 'metrics.md'
    write_markdown({'classifier': m}, md, cfg)
    text = md.read_text()
    assert 'compliance diff [95% CI]' in text and 'compliance population' in text
    json.loads(json.dumps(_json_safe(m)))                          # strict round trip
```
Run the suite: expect 95 passed (+2 here, +1 from Task 8's Part A).

- [ ] **Step 6 (added after the Task 7 code-quality review): readable report, strict JSON test, plot legibility**

- `write_markdown` writes two tables instead of one 14-column table: an accuracy table
  (variant, subset, pairs with future, moments, top-1 ADE before, after, diff [95% CI], minADE, top-1 FDE
  before, after) and a compliance table (variant, subset, n, before, after, diff [95% CI], counterfactual);
  rows whose subset has `n_pairs == 0` are skipped with a note "(gated_disagree is empty by construction
  for rules_only)". The footer adds `pairs per log: <log> <count>, ...` from `pairs_per_log` and a sentence
  that counterfactual consistency is only meaningful on gated pairs (a closed gate makes `cf_top1 == new_top1`).
- The JSON test uses `json.dumps(_json_safe(m), allow_nan=False)`; `main` and `run_all` also pass
  `allow_nan=False` so a NaN that slips through fails loudly.
- `plot_examples` deduplicates by `pred_file` before taking the top `n_example_plots`, draws history and
  future with `zorder=4` (future `color='0.2', ls='--', lw=2.5`), modes at `zorder=2` with non-selected
  modes at `lw=1.0`, and labels modes with the NORMALISED MTR score (`old` row) and the fused score.
- `compliance_frame` computes `sgn` once after the filter; `ade_fde`'s docstring says FDE is the error at
  the last valid step.
Suite: expect 96 passed.

---

### Task 8: `run_all.py` and the end-to-end test on the toy log

**Files:**
- Create: `scripts/doplan_lang/run_all.py`
- Test: `scripts/doplan_lang/tests/test_integration.py`

- [ ] **Step 1: Write the failing integration test**

```python
import csv
import json

import pytest

from doplan_lang import run_all

from .conftest import PROJECT_ROOT

TOY_LOG = '2021.08.24.13.12.55_veh-45_00386_00472'
SCENARIONET = PROJECT_ROOT / 'scenarionet_data' / 'nuplan_toy'
DB = PROJECT_ROOT / 'nuplan_data' / 'nuplan-v1.1' / 'splits' / 'mini' / f'{TOY_LOG}.db'
CSV = PROJECT_ROOT / 'nuplan_data' / 'doplan' / 'release_annotationsV1.csv'
have_data = (SCENARIONET / 'output_MTR').is_dir() and DB.exists() and CSV.exists()


@pytest.mark.skipif(not have_data, reason='toy nuPlan conversion, predictions or doPlan CSV not present')
def test_full_chain_on_toy_log(tmp_path):
    # The toy log has no doPlan annotation, so fabricate one covering the whole 86 s drive and
    # append it to the real CSV (the real rows are the training corpus).
    fake_csv = tmp_path / 'annotations.csv'
    with open(CSV, newline='', encoding='utf-8') as src, open(fake_csv, 'w', newline='', encoding='utf-8') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow(row)
        writer.writerow({'annotation_id': 'fake_0001', 'clip_id': TOY_LOG, 'start_frame': '0', 'end_frame': '430',
                         'window_duration_sec': '86.0', 'passenger_instruction': 'Turn left at the intersection.',
                         'referential_label': 'Static Referential', 'is_blank': 'false', 'annotator_id': 'test'})
    out = tmp_path / 'out'
    metrics = run_all.main(['--scenarionet', str(SCENARIONET), '--nuplan', str(PROJECT_ROOT / 'nuplan_data'),
                            '--csv', str(fake_csv), '--out', str(out)])
    for name in ('word2vec.kv', 'word2vec_report.md', 'intent_clf.joblib', 'intent_report.md', 'pairs.csv',
                 'coverage.csv', 'reranked.csv', 'reranked_rules.csv', 'metrics.json', 'metrics.md'):
        assert (out / name).exists(), name
    assert metrics['classifier']['n_pairs'] == 58            # every toy ego prediction lies inside the window
    assert json.loads((out / 'metrics.json').read_text())['rules_only']['n_pairs'] == 58
    assert len(list((out / 'plots').glob('*.png'))) >= 1
    assert len(list((out / 'reranked_pt').glob('*.pt'))) == 58
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests/test_integration.py -q`
Expected: `ImportError` (not a skip; the toy data is present on this machine).

- [ ] **Step 3: Implement `run_all.py`**

```python
"""Chain text -> word2vec -> intent -> align -> rerank -> evaluate for one ScenarioNet database.

Usage (from the project root):
  PYTHONPATH=scripts python3 -m doplan_lang.run_all \
      --scenarionet scenarionet_data/nuplan_doplan --nuplan nuplan_data \
      --csv nuplan_data/doplan/release_annotationsV1.csv --out outputs/doplan_lang

word2vec and the intent classifier are trained once into --out and reused unless --retrain.
"""
from __future__ import annotations

import argparse
import json
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
from .evaluate import add_errors, plot_examples, summarize, write_markdown
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
    p.add_argument('--lam', type=float, default=None, help='fusion exponent (default Config.lam)')
    p.add_argument('--exclude-log', action='append', default=None, help='override EVAL_LOGS for training; repeatable')
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
    checks = [sanity_check(frames_by_log[m['log_name']], m, pt_dir, cfg)
              for sid, m in summary.items() if n_preds[sid] >= 2]
    if not checks:
        raise SystemExit(f'no ego predictions in {pt_dir}')
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
        df = rerank_pairs(pairs, pt_dir, cfg, fn, a.out / 'reranked_pt' if is_main else None)
        df = add_errors(df, pt_dir, cfg)
        df.to_csv(a.out / ('reranked.csv' if is_main else 'reranked_rules.csv'), index=False)
        metrics[name] = summarize(df, cfg)
        if is_main:
            plot_examples(df, pt_dir, a.out / 'plots', cfg)
    (a.out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    write_markdown(metrics, a.out / 'metrics.md', cfg)
    print((a.out / 'metrics.md').read_text())
    return metrics


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the whole test suite**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests -q`
Expected: all tests pass, including `test_full_chain_on_toy_log` (it trains word2vec on the real corpus, so allow about a minute).

- [ ] **Step 5: Run the chain by hand on the toy data and look at one plot**

```bash
PYTHONPATH=scripts python3 -m doplan_lang.run_all --scenarionet scenarionet_data/nuplan_toy \
  --nuplan nuplan_data --csv nuplan_data/doplan/release_annotationsV1.csv --out outputs/doplan_lang_toy
cat outputs/doplan_lang_toy/coverage.csv | head -3
```
Expected: the alignment check passes for 2 of 6 scenarios (the toy inference was capped at 2), then `SystemExit: no (prediction, annotation) pairs` because the toy log has no doPlan annotations. That is the correct behaviour for this log; the real run is Task 11.

(Execution notes 2026-10-23: the code as written differs from the block above in ways described
elsewhere in this plan: `sanity_check(..., predictions=preds)`, `_json_safe` with `allow_nan=False`
and utf-8 for metrics.json/.md, the retrain-on-`--exclude-log` rule in the docstring. The files are the
reference. Observed: 96 tests, integration test 13 s.)

- [ ] **Step 6 (added after the Task 8 spec review): make the integration test bite**

The 86 s fabricated window captures all 58 predictions however times are placed, so add a second
fabricated annotation with a short window and assert its pair count against the prediction times,
plus the gate and intent outcomes:
```python
        writer.writerow({'annotation_id': 'fake_0002', 'clip_id': TOY_LOG, 'start_frame': '150', 'end_frame': '200',
                         'window_duration_sec': '10.0', 'passenger_instruction': 'Stop at the light.',
                         'referential_label': 'Static Referential', 'is_blank': 'false', 'annotator_id': 'test'})
```
and after the run:
```python
    import pandas as pd
    pairs = pd.read_csv(out / 'pairs.csv')
    short = pairs[pairs.annotation_id == 'fake_0002']
    t0 = pairs.t_pred_us.min() - pairs.time_index.min() * 0  # not the log start; derive the window from the .db instead:
```
Simplest correct check: read the log start from the .db (`load_lidar_frames(DB).ts[0]`), compute the
window `[t0 + 30 s, t0 + 40 s]`, and assert `len(short) == ((pairs.t_pred_us >= lo) & (pairs.t_pred_us <= hi) & (pairs.annotation_id == 'fake_0001')).sum()`
(the long annotation holds every prediction, so filtering it by the short window gives the expected
count), and `0 < len(short) < 58`. Also assert `metrics['classifier']['gated']['n_pairs'] > 0`,
`0 < metrics['classifier']['gate_open_fraction'] < 1`, and that every `fake_0001` pair has
`intent == 'turn_left'` and every `fake_0002` pair has `intent == 'stop'` in `reranked.csv`.
Observed 2026-10-23: the short window holds 34 of the 58 predictions; 92 pairs; gate open on 0.152.

- [ ] **Step 7 (added after the final package review): small cleanups before the data run**

- `run_all.py`: add `--no-pt` (skip writing `reranked_pt/`; the real run would produce 20k to 85k files,
  up to about 1 GB per lambda); rename evaluate's `_json_safe` to `json_safe` and add
  `write_metrics(metrics, out_dir, cfg)` in `evaluate.py` (json with `allow_nan=False` + markdown) used by
  both `evaluate.main` and `run_all`; make the empty-`checks` message say "no scenario has two or more ego
  predictions in ..."; unify the `--exclude-log` help text across train_word2vec, intent and run_all
  ("logs to hold out of training; REPLACES config.EVAL_LOGS when given; repeatable").
- `evaluate.write_markdown`: the skipped-subset note names the variant(s) actually skipped.
- `config.py`: delete the unused `step_s` (alignment is frame-based); `requirements-doplan.txt` as in Task 0.
- `tests/test_integration.py`: remove the debug `print`; pass `--no-pt` in a second, cheap assertion that
  `reranked_pt/` is then absent (reuse the trained models in `out` by calling `run_all.main` again with
  `--no-pt`; it skips training because the files exist).


---

### Task 9: Fetch the six evaluation logs and the Singapore map

**Files:**
- Download into: `nuplan_data/nuplan-v1.1/splits/mini/*.db`, `nuplan_data/maps/sg-one-north/**/map.gpkg`
- Create: `nuplan_data/nuplan-v1.1/splits/doplan/*.db` (hard links)

- [ ] **Step 1: Fetch logs and map with range requests (about 420 MB; files already on disk are skipped)**

```bash
python3 scripts/fetch_nuplan_toy.py --dest nuplan_data \
  --log 2021.10.11.02.57.41_veh-50_00352_00535 \
  --log 2021.10.11.08.31.07_veh-50_01750_01948 \
  --log 2021.10.11.02.57.41_veh-50_01522_02088 \
  --log 2021.08.17.18.54.02_veh-45_00665_01065 \
  --log 2021.08.17.17.17.01_veh-45_02314_02798 \
  --log 2021.08.17.16.57.11_veh-08_01200_01636 \
  --city sg-one-north --city us-pa-pittsburgh-hazelwood
```
Expected: six `integrity=ok` lines (three with `location=sg-one-north`, three with `location=us-pa-pittsburgh-hazelwood`), the Pittsburgh map reported as `exists, skipping`, and a final `done; transferred ... MB` line. If a transfer resets mid-file, rerun the same command; finished files are skipped.

- [ ] **Step 2: Give the converter a folder that holds only these six logs**

```bash
mkdir -p nuplan_data/nuplan-v1.1/splits/doplan
for l in 2021.10.11.02.57.41_veh-50_00352_00535 2021.10.11.08.31.07_veh-50_01750_01948 \
         2021.10.11.02.57.41_veh-50_01522_02088 2021.08.17.18.54.02_veh-45_00665_01065 \
         2021.08.17.17.17.01_veh-45_02314_02798 2021.08.17.16.57.11_veh-08_01200_01636; do
  ln -f nuplan_data/nuplan-v1.1/splits/mini/$l.db nuplan_data/nuplan-v1.1/splits/doplan/$l.db
done
ls -l nuplan_data/nuplan-v1.1/splits/doplan/
```
Expected: six `.db` files, same sizes as in `splits/mini/`. (`ln -f` makes hard links; if the filesystem refuses, use `cp` instead; 185 MB.)

- [ ] **Step 3: Confirm the doPlan windows fall inside each log**

```bash
PYTHONPATH=scripts python3 -c "
from pathlib import Path
from doplan_lang.align import load_lidar_frames
from doplan_lang.config import EVAL_LOGS
from doplan_lang.text import load_annotations
ins = load_annotations('nuplan_data/doplan/release_annotationsV1.csv')
for log in EVAL_LOGS:
    fr = load_lidar_frames(Path(f'nuplan_data/nuplan-v1.1/splits/doplan/{log}.db'))
    length = (fr.ts[-1] - fr.ts[0]) / 1e6
    mine = [i for i in ins if i.clip_id == log]
    print(f'{log}: {length:.1f} s, {len(mine)} instructions, max end {max(i.end_frame for i in mine) * 0.2:.1f} s')"
```
Expected: for every log, `max end` is at most the log length (a window ending after the log would mean the frame convention is wrong; stop and investigate before converting). Observed 2026-10-23 (with an equivalent sqlite query): 182.9/178.0, 198.0/180.6, 566.0/453.0, 400.0/392.8, 483.9/453.6, 436.0/358.0 s (log length / max window end). Also confirm the hard links share inodes with `splits/mini/` (`ls -i`).

---

### Task 10: Convert and predict (Docker; needs Docker Desktop running with WSL integration)

**Files:**
- Create: `scenarionet_data/nuplan_doplan/` (converter output), `scenarionet_data/nuplan_doplan/output_MTR/` (predictions)

- [ ] **Step 1: Check Docker is reachable**

```bash
docker compose version && docker image ls --format '{{.Repository}}:{{.Tag}}' | grep -E 'nuplan-convert|valeo-mtr'
```
Expected: a Compose version and the two images `local/nuplan-convert:py39` and `local/valeo-mtr:cu128`. If `docker` is not found, start Docker Desktop on Windows and enable WSL integration for this distro, then retry. If an image is missing, `docker compose build nuplan-convert` / `docker compose build mtr` (the mtr build takes about 12 minutes).

- [ ] **Step 2: Convert the six logs (CPU, several minutes)**

```bash
docker compose run --rm \
  -e NUPLAN_RAW_DIR=/mnt/nuplan/nuplan-v1.1/splits/doplan \
  -e SCENARIONET_OUT=/mnt/scenarionet/nuplan_doplan \
  -e NUM_WORKERS=4 nuplan-convert 2>&1 | tee logs/nuplan_doplan_convert.log
```
Expected: `logs:` lists six `.db` files; `maps:` lists two `map.gpkg`; the final summary prints N scenarios (expect tens, not 0 and not thousands) with `log_name`s from both cities. Zero scenarios means `NUPLAN_REMOVE_INVALID_GOALS` was set; unset it.

- [ ] **Step 3: Run MTR on every scenario (GPU)**

```bash
docker compose run --rm -e SCENARIONET_DIR=/mnt/scenarionet/nuplan_doplan \
  mtr bash /opt/mtr-scripts/run_inference.sh debug_max_nb_scenarios=100000 2>&1 | tee logs/nuplan_doplan_infer.log
ls scenarionet_data/nuplan_doplan/output_MTR | grep -c '___ego___'
```
Expected: no error; the count of ego prediction files is roughly 30 per scenario. If `output_MTR already contains predictions`, move the old folder aside first (the upstream script refuses to overwrite).

---

### Task 11: Run the pipeline on the real data and record the results

**Files:**
- Create: `outputs/doplan_lang/*` (all pipeline outputs)
- Modify: `README.md` (new section 7)

- [ ] **Step 1: Run the whole chain**

```bash
PYTHONPATH=scripts python3 -m doplan_lang.run_all \
  --scenarionet scenarionet_data/nuplan_doplan --nuplan nuplan_data \
  --csv nuplan_data/doplan/release_annotationsV1.csv --out outputs/doplan_lang --retrain \
  2>&1 | tee logs/doplan_lang_run.log
```
Expected: `alignment check passed for N of N scenarios` with worst displacement error under 0.05 m; a pairs/coverage line; then the metrics table. Read `outputs/doplan_lang/metrics.md`, `coverage.csv`, and the PNGs under `outputs/doplan_lang/plots/`.

- [ ] **Step 1b: Lambda sweep (post-hoc, reuses the trained models)**

```bash
for lam in 2 4; do
  mkdir -p outputs/doplan_lang_lam$lam
  cp outputs/doplan_lang/word2vec.kv outputs/doplan_lang/intent_clf.joblib outputs/doplan_lang_lam$lam/
  PYTHONPATH=scripts python3 -m doplan_lang.run_all --scenarionet scenarionet_data/nuplan_doplan \
    --nuplan nuplan_data --csv nuplan_data/doplan/release_annotationsV1.csv --out outputs/doplan_lang_lam$lam --lam $lam --no-pt
done
```
(The model files are copied first so they are reused rather than retrained; `--no-pt` skips the
re-ranked `.pt` copies, which would be hundreds of MB per lambda.) Report the gated-subset top-1 ADE and
compliance for lambda 1, 2 and 4 in one table in the README section.

- [ ] **Step 2: Judge coverage**

```bash
PYTHONPATH=scripts python3 -c "
import pandas as pd; c = pd.read_csv('outputs/doplan_lang/coverage.csv')
print(c.frac_covered.describe()); print('zero-coverage annotations:', int((c.n_moments == 0).sum()), 'of', len(c))"
```
If the median `frac_covered` is below 0.3 or more than half the annotations have zero coverage, do Task 12 and rerun Tasks 10 and 11; otherwise skip Task 12.

- [ ] **Step 3: Add a README section**

Append to `README.md`:
```markdown
## 7. doPlan language re-ranking of MTR modes

`scripts/doplan_lang/` trains word2vec on the doPlan passenger instructions
(`nuplan_data/doplan/release_annotationsV1.csv`), turns each instruction into an intent
distribution with a weakly supervised classifier, aligns doPlan windows with MTR ego
predictions through the nuPlan log clock, and re-ranks MTR's 5 modes where they fork.
Design: `docs/superpowers/specs/2026-10-23-doplan-word2vec-mtr-rerank-design.md`.

```bash
python3 -m pip install -r scripts/requirements-doplan.txt
PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests -q
# after Task 9/10 of the plan (six annotated logs converted + predicted):
PYTHONPATH=scripts python3 -m doplan_lang.run_all --scenarionet scenarionet_data/nuplan_doplan \
  --nuplan nuplan_data --csv nuplan_data/doplan/release_annotationsV1.csv --out outputs/doplan_lang
```

Outputs in `outputs/doplan_lang/`: `word2vec.kv` + `word2vec_report.md`, `intent_clf.joblib` +
`intent_report.md`, `pairs.csv`, `coverage.csv`, `reranked.csv` (per pair: intent probabilities,
gate, per-mode features, MTR and fused scores, errors), `reranked_rules.csv` (rules-only
ablation), `reranked_pt/` (predictions with fused `pred_scores`), `metrics.json`,
`metrics.md`, `plots/`.
```
Then paste the table from `outputs/doplan_lang/metrics.md` under the new section with one sentence stating the pair count and the gate-open fraction, so the README carries the result.

- [ ] **Step 4: Final full test run**

Run: `PYTHONPATH=scripts python3 -m pytest scripts/doplan_lang/tests -q`
Expected: all passed.

(Execution notes 2026-10-24: conversion gave 411 scenarios (233 Pittsburgh, 178 Singapore);
inference wrote 222,359 files, 12,688 ego predictions. The first run_all failed the alignment check
on 174 scenarios until `sanity_check` compared vehicle centres (Task 4 Step 8); then 411/411 passed
within 3 mm. 77,127 pairs, median coverage 0.75, no zero-coverage annotation, so Task 12 was not
needed. Results and the lambda sweep (0.5, 1, 2, 4) are in README section 7: compliance rises with
lambda (0.695 -> 0.778/0.824/0.839/0.848) while gated top-1 ADE worsens (1.694 -> 1.788/1.890/
1.973/2.072 m); classifier and rules-only are indistinguishable. 97 tests pass.)

---

### Task 12 (only if Task 11 step 2 says coverage is thin): converter scenario-spacing knob

**Files:**
- Modify: `scripts/patch_nuplan_convert.py` (add one entry to `PATCHES['scenarionet']`)

- [ ] **Step 1: Add the patch entry**

In `scripts/patch_nuplan_convert.py`, inside the `'scenarionet': [ ... ]` list, after the `remove_invalid_goals` entry, add:
```python
        (
            'scenarionet/converter/nuplan/utils.py',
            'make the minimum spacing between scenarios configurable (NUPLAN_TIMESTAMP_THRESHOLD_S)',
            '        "scenario_filter.timestamp_threshold_s=10",  # minial scenario duration (s)\n',
            '        # valeo_mtr_docker patch: denser scenario sampling for doPlan window coverage.\n'
            '        "scenario_filter.timestamp_threshold_s={}".format(\n'
            '            os.environ.get("NUPLAN_TIMESTAMP_THRESHOLD_S", "10")),\n',
            ('import os',),
        ),
```
The `import os` requirement is already satisfied by the earlier patch in the same file.

- [ ] **Step 2: Rebuild the converter image and reconvert with denser spacing**

```bash
docker compose build nuplan-convert
rm -rf scenarionet_data/nuplan_doplan
docker compose run --rm -e NUPLAN_RAW_DIR=/mnt/nuplan/nuplan-v1.1/splits/doplan \
  -e SCENARIONET_OUT=/mnt/scenarionet/nuplan_doplan -e NUM_WORKERS=4 \
  -e NUPLAN_TIMESTAMP_THRESHOLD_S=3 nuplan-convert 2>&1 | tee logs/nuplan_doplan_convert_dense.log
```
Expected: several times more scenarios than in Task 10. Then repeat Task 10 step 3 and Task 11 (the pipeline's `--retrain` is not needed; omit it).

- [ ] **Step 3: Document the knob**

Add to the README section 7 one line: `NUPLAN_TIMESTAMP_THRESHOLD_S` (default 10 s) sets the minimum spacing between extracted scenarios of the same type; lower it to cover more of each doPlan window.
