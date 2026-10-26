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
