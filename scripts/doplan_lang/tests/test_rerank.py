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
    wrap = feats_same.copy(); wrap[0, 1], wrap[1, 1], wrap[2, 1] = 178.0, -178.0, 180.0  # all within 4 degrees once wrapped
    assert not gate_open(same, wrap, scores, cfg)
    assert wrap_deg(356.0) == pytest.approx(-4.0) and wrap_deg(-190.0) == pytest.approx(170.0)


def test_one_hot():
    assert one_hot('stop') == {'turn_left': 0.0, 'turn_right': 0.0, 'lane_left': 0.0, 'lane_right': 0.0,
                               'stop': 1.0, 'straight': 0.0, 'other': 0.0}


def test_rerank_pairs_moves_top1_and_writes_pt(fake_pred_dir, tmp_path):
    cfg = Config()
    pairs = pd.DataFrame([{'pred_file': 'fake___ego___0___.pt', 'scenario_id': 'fake', 'log_name': 'logA',
                           'time_index': 0, 'frame': 0, 't_pred_us': 0, 'moment': 'logA:0', 'annotation_id': 'a1',
                           'clip_id': 'logA', 'instruction': 'Turn left at the light.',
                           'first_clause': 'turn left at the light', 'rule_intent': 'turn_left'}], columns=PAIR_COLUMNS)
    out_pt = tmp_path / 'reranked_pt'
    df = rerank_pairs(pairs, fake_pred_dir, cfg, lambda c: one_hot(rule_intent(c)), out_pt)
    row = df.iloc[0]
    assert row.gated and row.old_top1 == 2 and row.new_top1 == 0 and row.cf_top1 == 1
    assert row.intent == 'turn_left' and row.p_turn_left == 1.0
    assert row.new_top1_lat > 2 and row.cf_top1_lat < -2 and row.fut_valid == 60
    assert abs(sum(row[f's_new_{k}'] for k in range(3)) - 1) < 1e-9
    assert row.moment == 'logA:0' and row.frame == 0                  # pair columns are carried through
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


def test_two_annotations_on_one_prediction_share_features_and_get_two_files(fake_pred_dir, tmp_path):
    pairs = pd.concat([_pair_row(annotation_id='a1', instruction='Turn left.', first_clause='turn left', rule_intent='turn_left'),
                       _pair_row(annotation_id='a2')], ignore_index=True)
    out_pt = tmp_path / 'reranked_pt'
    df = rerank_pairs(pairs, fake_pred_dir, Config(), lambda c: one_hot(rule_intent(c)), out_pt)
    assert len(df) == 2 and df.lat_0.nunique() == 1 and df.gated.all()
    assert df.new_top1.tolist() == [0, 2]                                   # left turn moves it, `other` does not
    assert sorted(p.name for p in out_pt.glob('*.pt')) == ['fake___ego___0____a1.pt', 'fake___ego___0____a2.pt']
