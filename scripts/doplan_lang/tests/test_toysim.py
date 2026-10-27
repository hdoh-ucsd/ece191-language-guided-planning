import numpy as np
import pandas as pd
import pytest

from doplan_lang.config import Config
from doplan_lang.lanes import LaneGraph
from doplan_lang.toysim import AgentSet, boxes_overlap, completion_of, run, score_execution, simulate, summarize_toysim

from .conftest import PROJECT_ROOT, synthetic_map_features

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
    tail = [20.4, 20.1, 20.2, 20.1, 20.2, 20.1, 20.2, 20.1, 20.0, 19.9]
    plan = np.stack([np.concatenate([np.linspace(0.5, 20, 50), tail]), np.zeros(60)], 1)
    ex = simulate(plan, (0.0, 0.0), 0.0, 4.0, CFG)
    assert np.all(np.diff(ex['xy'][:, 0]) >= -1e-9) and ex['speed'][-1] < 1.5


def test_boxes_overlap():
    c2 = np.array([[3.0, 0.0], [10.0, 0.0], [0.0, 2.0]])
    hit = boxes_overlap((0.0, 0.0), 0.0, 5.0, 2.0, c2, np.zeros(3), np.full(3, 4.0), np.full(3, 2.0))
    assert list(hit) == [True, False, True]
    # a box at x = 3.5 turned 90 deg: its width lies along x, so 2.5 m wide reaches x = 2.25 (hit), 1 m wide does not
    rot = boxes_overlap((0.0, 0.0), 0.0, 5.0, 2.0, np.array([[3.5, 0.0], [3.5, 0.0]]), np.full(2, np.pi / 2), np.full(2, 4.0),
                        np.array([2.5, 1.0]))
    assert list(rot) == [True, False]
    assert len(boxes_overlap((0.0, 0.0), 0.0, 5.0, 2.0, np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0))) == 0


def _agents(xy):
    return AgentSet(positions=np.zeros((1, 100, 2)) + xy, headings=np.zeros((1, 100)), valid=np.ones((1, 100)),
                    lengths=np.array([4.0]), widths=np.array([2.0]))


def test_score_execution_on_synthetic_map():
    g = LaneGraph.from_map_features(synthetic_map_features())
    hist = np.stack([np.linspace(40, 50, 20), np.full(20, -3.5)], 1)
    change = np.stack([np.linspace(50.5, 80, 60), np.linspace(-3.5, 0.0, 60)], 1)
    ex = simulate(change, (50.0, -3.5), 0.0, 5.0, CFG)
    far = _agents([200.0, 200.0])
    s = score_execution(ex, change, far, 20, g, (50.0, -3.5), 0.0, hist, CFG)
    assert s['label'] == 'lane_left' and not s['collision'] and s['offlane_fraction'] == 0.0 and s['tracking_rms_m'] < 0.5
    assert completion_of(s['label'], s['final_speed_mps'], 'lane_left', CFG) == 1.0
    assert completion_of(s['label'], s['final_speed_mps'], 'lane_right', CFG) == 0.0
    assert np.isnan(completion_of(s['label'], s['final_speed_mps'], 'other', CFG)) and np.isnan(completion_of(None, 0.0, 'turn_left', CFG))
    assert score_execution(ex, change, _agents([60.0, -2.5]), 20, g, (50.0, -3.5), 0.0, hist, CFG)['collision']
    edge = np.stack([np.linspace(50.5, 80, 60), np.full(60, -5.4)], 1)             # just outside A2's polygon, 1.9 m off
    assert score_execution(simulate(edge, (50.0, -3.5), 0.0, 5.0, CFG), edge, far, 20, g, (50.0, -3.5), 0.0, hist, CFG)['offlane_fraction'] == 0.0
    off = np.stack([np.linspace(50.5, 80, 60), np.linspace(-3.5, -30, 60)], 1)
    s3 = score_execution(simulate(off, (50.0, -3.5), 0.0, 5.0, CFG), off, far, 20, g, (50.0, -3.5), 0.0, hist, CFG)
    assert s3['offlane_fraction'] > 0.5
    stop = np.stack([np.linspace(50.3, 56, 60), np.full(60, -3.5)], 1)              # decelerates to 0 within 6 s
    s4 = score_execution(simulate(stop, (50.0, -3.5), 0.0, 3.0, CFG), stop, far, 20, g, (50.0, -3.5), 0.0, hist, CFG)
    assert completion_of(s4['label'], s4['final_speed_mps'], 'stop', CFG) == 1.0


def test_run_scores_completion_per_pair(tmp_path):
    """Two pairs share one prediction; completion must follow each pair's intent, not the first pair's."""
    import pickle
    import torch
    sn = tmp_path / 'db'
    (sn / 'db_0').mkdir(parents=True)
    (sn / 'output_MTR').mkdir()
    T = 100
    ego = {'state': {'position': np.zeros((T, 3)), 'heading': np.zeros(T), 'velocity': np.zeros((T, 2)), 'valid': np.ones(T),
                     'length': np.full((T, 1), 5.0), 'width': np.full((T, 1), 2.0), 'height': np.full((T, 1), 1.5)}, 'type': 'VEHICLE', 'metadata': {}}
    far = {'state': {**ego['state'], 'position': np.zeros((T, 3)) + [200.0, 200.0, 0.0]}, 'type': 'VEHICLE', 'metadata': {}}
    with open(sn / 'db_0' / 'sd_nuplan_v1.1_S.pkl', 'wb') as f:
        pickle.dump({'tracks': {'ego': ego, 'v1': far}, 'metadata': {'sdc_id': 'ego'}, 'map_features': synthetic_map_features()}, f)
    xs = 30.5 + np.arange(1, 61) * 0.5
    modes = np.stack([np.stack([xs, np.linspace(-3.5, 1.0, 60)], 1), np.stack([xs, np.full(60, -3.5)], 1)]).astype(np.float32)
    gt = np.zeros((81, 10), dtype=np.float32)
    gt[:, 0] = np.concatenate([30.5 - np.arange(21, 0, -1) * 0.5, xs])
    gt[:, 1], gt[:, 9] = -3.5, 1.0
    torch.save({'scenario_id': 'S', 'pred_trajs': modes, 'pred_scores': np.array([0.4, 0.6], dtype=np.float32), 'gt_trajs': gt,
                'past_det_scores': np.float32(1.0), 'object_id': 'ego', 'object_type': 1, 'track_index_to_predict': 0,
                'current_time_index': 0}, sn / 'output_MTR' / 'S___ego___20___.pt')
    base = dict(pred_file='S___ego___20___.pt', scenario_id='S', log_name='logA', moment='logA:0', gated=True, time_index=20,
                old_top1=1, new_top1=0, fut_valid=60)
    reranked = pd.DataFrame([dict(base, annotation_id='a_other', rule_intent='other'),
                             dict(base, annotation_id='a_left', rule_intent='lane_left')])
    df = run(reranked, sn, sn / 'output_MTR', CFG)
    assert len(df) == 6 and set(df.execution) == {'mtr_top1', 'reranked_top1', 'recorded'}
    left = df[df.annotation_id == 'a_left'].set_index('execution')
    assert left.loc['reranked_top1', 'completion'] == 1.0 and left.loc['mtr_top1', 'completion'] == 0.0
    assert left.loc['recorded', 'completion'] == 0.0                         # the recorded future goes straight
    assert df[df.annotation_id == 'a_other'].completion.isna().all()
    assert not df.collision.any() and (df.offlane_fraction == 0).all()


def test_summary_pairs_executions():
    rows = []
    for i, (comp_a, comp_b, col_a, col_b) in enumerate([(0.0, 1.0, False, False), (1.0, 1.0, False, True), (np.nan, np.nan, False, False)]):
        for ex, comp, col in (('mtr_top1', comp_a, col_a), ('reranked_top1', comp_b, col_b), ('recorded', 1.0, False)):
            rows.append(dict(pred_file=f'p{i}', annotation_id='a', moment=f'm{i}', gated=True, execution=ex, completion=comp,
                             collision=col, offlane_fraction=0.0, tracking_rms_m=0.2, max_lat_accel=1.0, plan_mode=0))
    m = summarize_toysim(pd.DataFrame(rows), Config(bootstrap_n=20))
    assert m['n_pairs'] == 3 and m['gated']['mtr_top1']['completion'] == 0.5 and m['gated']['reranked_top1']['completion'] == 1.0
    assert m['gated']['recorded']['collision'] == 0.0 and m['gated']['reranked_top1']['collision'] == pytest.approx(1 / 3)
    d = m['gated']['diff_reranked_minus_mtr']
    assert d['completion']['mean'] == 0.5 and d['collision']['mean'] == pytest.approx(1 / 3)


SCENES = PROJECT_ROOT / 'scenarionet_data' / 'nuplan_scenes'


@pytest.mark.skipif(not (SCENES / 'output_MTR').is_dir(), reason='scene run data not present')
def test_recorded_future_of_scene7_is_clean():
    from doplan_lang.lanes import load_scenario
    from doplan_lang.rerank import load_prediction
    from doplan_lang.toysim import agents_from_tracks
    sc = load_scenario(SCENES, '3e1942159626505c')
    g = LaneGraph.from_map_features(sc['map_features'])
    agents = agents_from_tracks(sc['tracks'], sc['metadata']['sdc_id'])
    pred = load_prediction(SCENES / 'output_MTR' / '3e1942159626505c___ego___40___.pt')
    gt = np.asarray(pred['gt_trajs'])
    now, hist, fut = gt[20], gt[:21, :2], gt[21:, :2]
    v0 = float(np.linalg.norm(hist[-1] - hist[-2]) / CFG.sim_dt)
    ex = simulate(fut, now[:2], float(now[6]), v0, CFG)
    s = score_execution(ex, fut, agents, 40, g, now[:2], float(now[6]), hist, CFG, 5.176, 2.297)
    assert not s['collision'] and s['offlane_fraction'] == 0.0 and s['tracking_rms_m'] < 1.0
    assert completion_of(s['label'], s['final_speed_mps'], 'turn_right', CFG) == 1.0
