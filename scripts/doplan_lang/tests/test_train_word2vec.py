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
