import numpy as np
import pytest
import torch

from doplan_lang.align import (AlignmentError, FRAMES_PER_STEP, LidarFrames, build_pairs, coverage_table,
                               list_ego_predictions, log_start_us, prediction_frame, prediction_time_us,
                               prediction_times, rear_axle_to_centre_m, sanity_check, scenario_start_frame,
                               window_us)
from doplan_lang.config import Config
from doplan_lang.text import make_instruction

T0 = 1_000_000_000


def frames(n=2000, jitter_us=-100):
    """20 Hz lidar frames; the frame nominally at 30.0 s is `jitter_us` early, like real logs."""
    ts = T0 + np.arange(n, dtype=np.int64) * 50_000
    ts[600] += jitter_us
    tokens = tuple(f'{i:08x}' for i in range(n))
    xy = np.stack([np.arange(n) * 0.25, np.zeros(n)], 1)          # 5 m/s along +x
    return LidarFrames(tokens=tokens, ts=ts, ego_xy=xy, ego_yaw=np.zeros(n))


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
    assert list(pt.frame) == [601, 611, 621, 631, 641]
    assert pt.moment.nunique() == 5 and pt.moment.iloc[0] == 'logA:60'


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
    assert cov.loc['x1', 'n_moments'] == 3 and cov.loc['x1', 'window_s'] == pytest.approx(1.2)
    assert cov.loc['x1', 'frac_covered'] == pytest.approx(1.0)
    assert cov.loc['x2', 'n_moments'] == 0 and cov.loc['x2', 'frac_covered'] == 0.0


def test_predictions_in_the_same_frame_bucket_share_a_moment():
    fr = frames()
    summary = {'bb': meta(), 'cc': {**meta(token_frame=663), 'scenario_id': 'cc'}}   # cc starts 2 frames after bb (bb's start is pushed to 601 by the jitter)
    preds = [('bb___ego___5___.pt', 'bb', 5), ('cc___ego___5___.pt', 'cc', 5), ('cc___ego___10___.pt', 'cc', 10)]
    pt = prediction_times(preds, summary, {'logA': fr}, Config())
    assert list(pt.frame) == [611, 613, 623]
    assert pt.moment.iloc[0] == pt.moment.iloc[1] != pt.moment.iloc[2]          # 611 and 613 share bucket 61
    ins = [make_instruction('x1', 'logA', 150, 156, 'Turn left.')]              # 30.0 s .. 31.2 s
    assert len(build_pairs(ins, pt, {'logA': fr}, Config())) == 3
    assert coverage_table(ins, pt, {'logA': fr}, Config()).iloc[0].n_moments == 2


def test_missing_log_raises_alignment_error():
    with pytest.raises(AlignmentError, match='no lidar frames'):
        prediction_times([('bb___ego___0___.pt', 'bb', 0)], {'bb': meta()}, {}, Config())


def _write_gt(path, xy, heading=0.0):
    gt = np.zeros((81, 10), dtype=np.float32)
    gt[:, 0], gt[:, 1], gt[:, 6], gt[:, 9] = xy[0], xy[1], heading, 1.0
    torch.save({'gt_trajs': gt, 'scenario_id': 'bb', 'object_id': 'ego'}, path)


def test_sanity_check_passes_on_consistent_track_and_fails_otherwise(tmp_path):
    fr = frames()
    _write_gt(tmp_path / 'bb___ego___0___.pt', (0.0, 0.0))
    _write_gt(tmp_path / 'bb___ego___20___.pt', (10.0, 0.0))        # frames 601 -> 641 = 40 * 0.25 m
    out = sanity_check(fr, meta(), tmp_path, Config())
    assert sanity_check(fr, meta(), tmp_path, Config(), predictions=list_ego_predictions(tmp_path)) == out
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


def test_sanity_check_compares_vehicle_centres_when_heading_changes(tmp_path):
    # nuPlan ego_pose is the rear axle; ScenarioNet writes the centre, 1.461 m ahead for the Pacifica.
    # Between frames 601 and 641 the recorded car turns 90 degrees left; the centre displacement differs
    # from the rear-axle displacement by about 1.461 * sqrt(2) = 2.07 m.
    params = {'front_length': 4.049, 'rear_length': 1.127}
    m = {**meta(), 'ego_vehicle_parameters': params}
    assert rear_axle_to_centre_m(m) == pytest.approx(1.461)
    assert rear_axle_to_centre_m(meta()) == 0.0
    n = 2000
    ts = T0 + np.arange(n, dtype=np.int64) * 50_000
    ts[600] -= 100
    yaw = np.zeros(n); yaw[641:] = np.pi / 2
    xy = np.stack([np.arange(n) * 0.25, np.zeros(n)], 1)
    fr = LidarFrames(tokens=tuple(f'{i:08x}' for i in range(n)), ts=ts, ego_xy=xy, ego_yaw=yaw)
    c = 1.461
    centre_disp = (xy[641] + c * np.array([0.0, 1.0])) - (xy[601] + c * np.array([1.0, 0.0]))
    _write_gt(tmp_path / 'bb___ego___0___.pt', (0.0, 0.0))
    _write_gt(tmp_path / 'bb___ego___20___.pt', tuple(centre_disp))   # MTR gt moves like the centre
    out = sanity_check(fr, m, tmp_path, Config())
    assert out['displacement_error_m'] < 1e-6
    with pytest.raises(AlignmentError, match='displacement'):            # rear-axle comparison would be 2 m off
        sanity_check(fr, meta(), tmp_path, Config())
