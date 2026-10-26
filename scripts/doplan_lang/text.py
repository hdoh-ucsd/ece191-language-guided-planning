"""doPlan instruction text: normalisation, tokens, clauses and weak intent rules (spec 4.1).

Precedence: hold-lane (unless a turn verb is present) > lane change > turn > stop >
straight > other. A clause naming both sides
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
# "stay in the right lane" holds the lane; it is not a lane change.
_KEEP_LANE = re.compile(r"\b(?:stay|keep|remain|continue)\s+(?:in|on)\s+(?:the\s+)?(?:\w+\s+)?lanes?\b")
_TURN_VERB = re.compile(r"\b(?:turn|turning|(?:make|take|hang) a (?:left|right|u turn))\b")
_LEFT = re.compile(r"\bleft(?:most)?\b")
_RIGHT = re.compile(r"\bright(?:most)?\b(?!\s+(?:away|now)\b)")
_UTURN = re.compile(r"\bu turn\b")
_RIGHT_HERE = re.compile(r"\bright here\b")
# "stop" as a verb (not the nouns "stop sign" / "stop light"); "park" only as a verb.
_STOP = re.compile(r"\bstop(?:s|ping)?\b(?!\s+(?:signs?|lights?)\b)"
                   r"|\b(?:wait|pull over|pull up|pull into|halt)\b"
                   r"|\bdrop(?:\s+\w+)?\s+off\b(?!\s+(?:zone|area|lane)s?\b)"
                   r"|(?:^|\b(?:and|then|please|just|can|to)\s+)park\b")
# "stay in" overlaps _KEEP_LANE on purpose: the hold-lane check in rule_intent runs first.
# "procced" is a typo that occurs in the corpus, kept deliberately.
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
    keep_lane = bool(_KEEP_LANE.search(c))
    if keep_lane and not _TURN_VERB.search(c):
        return 'straight'
    lane = bool(_LANE.search(c)) and not keep_lane
    left = bool(_LEFT.search(c))
    right = bool(_RIGHT.search(c))
    if left and right and _RIGHT_HERE.search(c):
        right = False          # "left, right here": "right here" is a place, not a direction
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


def load_annotations(csv_path: str | Path) -> list[Instruction]:
    """Non-blank doPlan rows as Instruction records (blank rows carry no language)."""
    out: list[Instruction] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            text = (row.get('passenger_instruction') or '').strip()
            if (row.get('is_blank') or 'false').strip().lower() == 'true' or not text:
                continue
            try:
                out.append(make_instruction(row['annotation_id'], row['clip_id'],
                                            row['start_frame'], row['end_frame'], text))
            except ValueError as exc:
                raise ValueError(f'annotation {row.get("annotation_id")!r}: {exc}') from exc
    return out


_DIRECTION = re.compile(r'\b(left|right)(most)?\b')


def swap_left_right(text: str) -> str:
    """Counterfactual instruction: every left becomes right and vice versa (also leftmost/rightmost, left-turn)."""
    return _DIRECTION.sub(lambda m: {'left': 'right', 'right': 'left'}[m.group(1)] + (m.group(2) or ''),
                          normalize(text))
