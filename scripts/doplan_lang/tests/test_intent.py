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


def test_cross_validate_needs_two_classes():
    import pytest
    with pytest.raises(ValueError, match='two intent classes'):
        cross_validate(np.zeros((4, 3)), np.array(['other'] * 4, dtype=object), seed=0, folds=2)
