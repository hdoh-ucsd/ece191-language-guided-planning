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
    return {'pred_file': pred_file, 'scenario_id': 'fake', 'log_name': 'logA', 'time_index': 0, 'frame': 0,
            't_pred_us': 0, 'moment': f'logA:{pred_file}', 'annotation_id': ann, 'clip_id': 'logA',
            'instruction': text, 'first_clause': text.lower().rstrip('.'), 'rule_intent': rule_intent(text)}


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
    # two pairs share straight___ego___0___.pt; plots are deduplicated to one per prediction file
    assert len(pngs) == 2 and all(p.exists() for p in pngs)


def test_compliance_population_uses_rule_label_and_full_future(tmp_path):
    df, _, cfg = _reranked(tmp_path)
    df.loc[df.annotation_id == 'a1', 'intent'] = 'other'          # the classifier disagrees: still counted
    assert list(compliance_frame(df, cfg).annotation_id) == ['a1']
    df.loc[df.annotation_id == 'a1', 'fut_valid'] = 59            # short future: excluded
    assert compliance_frame(df, cfg).empty


def testjson_safe_and_markdown_ci_column(tmp_path):
    from doplan_lang.evaluate import json_safe
    assert json_safe({'a': math.nan, 'b': [1.0, math.nan], 'c': {'d': math.nan}}) == {'a': None, 'b': [1.0, None], 'c': {'d': None}}
    df, _, cfg = _reranked(tmp_path)
    m = summarize(df, cfg)
    assert m['pairs_per_log'] == {'logA': 3} and m['gated']['compliance_intent'] == 'rule_intent'
    md = tmp_path / 'metrics.md'
    write_markdown({'classifier': m}, md, cfg)
    text = md.read_text()
    assert '| n | before | after | diff [95% CI] | counterfactual |' in text and 'pairs per log: logA 3' in text
    assert 'compliance population' in text
    json.loads(json.dumps(json_safe(m), allow_nan=False))                          # strict round trip
