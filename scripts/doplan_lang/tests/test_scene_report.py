import numpy as np
import pandas as pd

from doplan_lang.align import TIME_COLUMNS, LidarFrames
from doplan_lang.config import Config
from doplan_lang.scene_report import (Scene, active_instructions, read_scenes, scene_table, select_moments,
                                      write_scene)
from doplan_lang.scene_paths import changed_moments, load_map_features, recorded_path, scene_frame
from doplan_lang.text import make_instruction

from .conftest import write_fake_prediction

T = 1_000_000_000


def _times():
    rows = []
    for k in range(0, 40):                           # predictions every 0.5 s on logA, 40 ms late
        t = T + k * 500_000 + 40_000
        rows.append(('A___ego___%d___.pt' % (20 + 5 * k), 'A', 'logA', 20 + 5 * k, 40 + 10 * k, t, f'logA:{k}'))
    rows.append(('B___ego___20___.pt', 'B', 'logB', 20, 40, T + 2_000_000, 'logB:0'))
    return pd.DataFrame(rows, columns=TIME_COLUMNS)


def test_select_moments_nearest_within_tolerance():
    sc = Scene('s1', 'logA', T + 3_000_000)
    m = select_moments(_times(), sc, horizon_s=2.0, stride_s=0.5, tol_s=0.3)
    assert list(m.t_rel_s) == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert list(m.pred_file) == ['A___ego___%d___.pt' % (20 + 5 * k) for k in range(6, 11)]
    assert np.allclose(m.dt_ms, 40.0)
    # a start between two predictions picks the nearer one; too far away yields nothing
    m = select_moments(_times(), Scene('s2', 'logA', T + 3_200_000), 0.0, 0.5, 0.3)
    assert list(m.pred_file) == ['A___ego___50___.pt'] and m.dt_ms.iloc[0] == -160.0
    assert select_moments(_times(), Scene('s3', 'logA', T + 90_000_000), 1.0, 0.5, 0.3).empty
    assert select_moments(_times(), Scene('s4', 'logC', T), 1.0, 0.5, 0.3).empty


def _frames(n=400):
    """A 20 Hz log driving east at 10 m/s from T; frame k is at (0.5 k, 0)."""
    k = np.arange(n)
    return LidarFrames(tokens=tuple(f'{i:016x}' for i in k), ts=T + 50_000 * k,
                       ego_xy=np.stack([0.5 * k, np.zeros(n)], 1), ego_yaw=np.zeros(n))


SUMMARY = {'A': {'scenario_id': 'A', 'log_name': 'logA', 'scenario_token': f'{0:016x}',
                 'scenario_extraction_info': {'extraction_offset': 0.0}}}


def test_scene_table_and_report(tmp_path):
    pt = tmp_path / 'output_MTR'
    pt.mkdir()
    write_fake_prediction(pt / 'A___ego___50___.pt', scores=(0.2, 0.3, 0.5), origin=(50.0, 0.0))
    write_fake_prediction(pt / 'A___ego___55___.pt', scores=(0.1, 0.1, 0.8), origin=(55.0, 0.0), future='left')
    sc = Scene('s1', 'logA', T + 3_000_000)
    moments = select_moments(_times(), sc, 0.5, 0.5, 0.3)
    reranked = pd.DataFrame([{
        'pred_file': 'A___ego___50___.pt', 'annotation_id': 'a1', 'instruction': 'turn left', 'intent': 'turn_left',
        'gated': True, 's_old_0': 0.2, 's_old_1': 0.3, 's_old_2': 0.5, 's_new_0': 0.9, 's_new_1': 0.05, 's_new_2': 0.05,
        'old_top1': 2, 'new_top1': 0, 'ade_old': 5.0, 'ade_new': 0.0, 'fde_old': 9.0, 'fde_new': 0.0, 'min_ade': 0.0,
        'old_top1_label': 'straight', 'new_top1_label': 'turn_left', 'fut_label': 'turn_left',
        'old_top1_stop': False, 'new_top1_stop': False, 'fut_stop': False}])
    cfg = Config()
    table = scene_table(moments, reranked, pt, cfg)
    assert list(table.t_rel_s) == [0.0, 0.5]
    assert table.iloc[0].annotation_id == 'a1' and table.iloc[0].new_top1 == 0
    r = table.iloc[1]                                 # no instruction covers the second moment: MTR-only
    assert r.annotation_id == '' and r.old_top1 == r.new_top1 == 2 and not r.gated
    assert np.isclose(r.s_new_2, 0.8) and np.isclose(r.ade_old, r.ade_new) and r.min_ade == 0.0
    ins = [make_instruction('a1', 'logA', 10, 20, 'turn left'), make_instruction('a2', 'logA', 0, 5, 'stop'),
           make_instruction('a3', 'logB', 0, 100, 'go straight')]
    active = active_instructions(ins, sc, T, cfg)      # start = 3 s = doPlan frame 15: only a1 covers it
    assert list(active.annotation_id) == ['a1'] and active.iloc[0].window_start_s == 2.0
    md = write_scene(sc, table, active, _frames(), SUMMARY, tmp_path / 'out', pt, tmp_path, cfg, plot_every_s=0.5)
    text = md.read_text()
    assert '| 0.0 | a1 | turn_left | open |' in text and '| 0.5 | - | - | closed |' in text
    assert '| straight / turn_left / turn_left |' in text and '| - / - / - |' in text    # MTR-only row has no graph
    assert (tmp_path / 'out' / 's1.csv').exists()
    assert sorted(p.name for p in (tmp_path / 'out').glob('*.png')) == [
        's1_changed.png', 's1_paths.png', 's1_t000.0s_a1.png', 's1_t000.5s_mtr_only.png']
    # the a1 row moved the top-1 from the straight mode (2) to the left arc (0), which is the recorded future
    assert 'changed the top-1 on 1 of 2 (moment, instruction) rows (1 of 2 moments)' in text
    assert '| 0.0 | a1 | turn_left | 2 -> 0 |' in text and '| straight -> turn_left (turn_left) | 5.00 -> 0.00 | 9.00 -> 0.00 | yes |' in text
    ch = pd.read_csv(tmp_path / 'out' / 's1_changed.csv')
    assert len(ch) == 1 and ch.iloc[0].closer and ch.iloc[0].new_dpsi > 80 and abs(ch.iloc[0].old_dpsi) < 1

    # nothing changed: the overview still draws the original top-1 at the plotted moments, no grid figure
    table.loc[:, 'new_top1'] = table.old_top1
    md = write_scene(sc, table, active, _frames(), SUMMARY, tmp_path / 'out2', pt, tmp_path, cfg, plot_every_s=0.5)
    assert 'left the top-1 unchanged on all 2 rows' in md.read_text()
    assert not (tmp_path / 'out2' / 's1_changed.png').exists() and (tmp_path / 'out2' / 's1_paths.png').exists()


def test_scene_frame_and_recorded_path():
    fr = _frames()
    table = pd.DataFrame([{'t_rel_s': 0.5, 'frame': 110, 'scenario_id': 'A'}, {'t_rel_s': 0.0, 'frame': 100, 'scenario_id': 'A'}])
    origin, shifts, c = scene_frame(table, SUMMARY, fr)
    assert np.allclose(origin, (50.0, 0.0)) and c == 0.0          # no vehicle parameters: rear axle == centre
    assert np.allclose(shifts['A'], (-50.0, 0.0))                 # scenario A starts at frame 0, 50 m behind
    path = recorded_path(fr, 0.0, origin, 90, 121)
    assert path.shape == (31, 2) and np.allclose(path[0], (-5.0, 0.0)) and np.allclose(path[-1], (10.0, 0.0))
    assert recorded_path(fr, 0.0, origin, -10, 10_000).shape == (400, 2)   # clipped to the log


def test_load_map_features_reads_scenarionet_pickle(tmp_path):
    import pickle
    import pytest
    # A pickle of metadrive's ScenarioDescription (a dict subclass) holding one lane, written opcode by
    # opcode because that module is not installed on the host: NEWOBJ on the class, then SETITEMS.
    inner = {'l1': {'type': 'LANE_SURFACE_STREET', 'polyline': [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]}}
    payload = (pickle.PROTO + b'\x02' + pickle.GLOBAL + b'metadrive.scenario.scenario_description\nScenarioDescription\n'
               + pickle.EMPTY_TUPLE + pickle.NEWOBJ + pickle.MARK
               + pickle.dumps('map_features', 2)[2:-1] + pickle.dumps(inner, 2)[2:-1] + pickle.SETITEMS + pickle.STOP)
    (tmp_path / 'db_0').mkdir()
    (tmp_path / 'db_0' / 'sd_nuplan_v1.1_abc.pkl').write_bytes(payload)
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(payload)
    feats = load_map_features(tmp_path, 'abc')
    assert list(feats) == ['l1'] and feats['l1']['type'] == 'LANE_SURFACE_STREET'
    assert load_map_features(tmp_path, 'missing') == {}


def test_read_scenes(tmp_path):
    p = tmp_path / 'scenes.csv'
    p.write_text('scene,clip_id,start_us\nscene1, logA ,1623262378149565\n')
    assert read_scenes(p) == [Scene('scene1', 'logA', 1623262378149565)]
