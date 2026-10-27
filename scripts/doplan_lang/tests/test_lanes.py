import numpy as np

from doplan_lang.config import Config
from doplan_lang.lanes import LaneGraph, LanePose, final_speed, label, manoeuvre, wrap_deg

from .conftest import synthetic_map_features

CFG = Config()


def graph():
    return LaneGraph.from_map_features(synthetic_map_features())


def test_build_filters_and_links():
    g = graph()
    assert set(g.lanes) == {'A0', 'A1', 'A2', 'B0', 'B1', 'B2', 'C', 'D'}
    assert g.lanes['A1'].left == ('A0',) and g.lanes['A1'].right == ('A2',) and g.lanes['A2'].exits == ('B2', 'C')
    assert g.lanes['A1'].s[-1] == 60.0 and abs(g.lanes['A1'].heading[0]) < 1e-9
    assert len(LaneGraph.from_map_features({})) == 0 and LaneGraph.from_map_features({}).assign((0, 0)) is None


def test_assign_uses_polygon_then_tail():
    g = graph()
    p = g.assign((30.0, 0.4))
    assert p.lane_id == 'A1' and abs(p.s - 30.0) < 1e-6 and abs(p.dist - 0.4) < 1e-6
    assert g.assign((30.0, 40.0)) is None                                     # nothing within 3 m
    # x = 62, y = -3.6 lies inside both B2 and the connector C; the tail decides
    straight_tail = np.stack([np.linspace(58, 62, 20), np.full(20, -3.5)], 1)
    arc_th = np.radians(np.linspace(90, 78, 20))
    arc_tail = np.stack([60 + 10 * np.cos(arc_th), -13.5 + 10 * np.sin(arc_th)], 1)
    assert g.assign((62.0, -3.6), tail=straight_tail).lane_id == 'B2'
    assert g.assign(arc_tail[-1], tail=arc_tail).lane_id == 'C'
    assert g.assign((30.0, 0.4), heading=np.pi) is None                     # heading the wrong way
    west = np.stack([np.linspace(35, 30, 10), np.full(10, 0.4)], 1)       # the tail travels west: A1 heads east
    assert g.assign((30.0, 0.4), tail=west) is None
    assert g.assign((30.0, 0.4), tail=west[::-1]).lane_id == 'A1'
    assert g.contains((30.0, 1.0)) and not g.contains((30.0, 8.0))


def test_routes_and_labels():
    g = graph()
    a1, a0, a2 = g.assign((30, 0)), g.assign((30, 3.5)), g.assign((30, -3.5))
    b1, c, d = g.assign((90, 0)), g.assign((69.9, -12.5)), g.assign((69.9, 12.5))
    assert c.lane_id == 'C' and d.lane_id == 'D'
    assert label(g.route(a1, a1)) == 'straight' and g.route(a1, a1).moves == ''
    assert label(g.route(a1, b1)) == 'straight' and g.route(a1, b1).moves == 'F'
    assert label(g.route(a1, a0)) == 'lane_left' and label(g.route(a1, a2)) == 'lane_right'
    assert label(g.route(a2, c)) == 'turn_right' and label(g.route(a1, c)) == 'turn_right'   # heading beats the R move
    assert label(g.route(a0, d)) == 'turn_left'
    assert g.route(a1, LanePose('zzz', 0, 0, 0)) is None and label(None) is None and g.route(None, a1) is None
    assert g.route(b1, a1) is None                                   # no backwards edges
    assert g.route(a1, b1, Config(lane_route_max_cost=0.5)) is None


def test_final_speed_and_stop_on_creeping_tail():
    xy = np.stack([np.linspace(0, 20, 60), np.zeros(60)], 1)
    assert abs(final_speed(xy) - (20 / 59 * 5 / 0.5)) < 1e-6
    creep = xy.copy()
    creep[-10:] = [[20.0, 0], [20.4, 0], [20.1, 0], [20.2, 0], [20.1, 0], [20.2, 0], [20.1, 0], [20.2, 0], [20.1, 0], [20.0, 0]]
    assert final_speed(creep) < 1.0
    g = graph()
    m = manoeuvre(g, (10.0, 0.0), 0.0, np.stack([np.linspace(0, 10, 20), np.zeros(20)], 1), creep + [10.0, 0.0], CFG)
    assert m.label == 'straight' and m.stop and m.start.lane_id == 'A1' and m.end.lane_id == 'A1'


def test_manoeuvre_labels_turn_and_lane_change():
    g = graph()
    hist = np.stack([np.linspace(40, 50, 20), np.full(20, -3.5)], 1)
    th = np.radians(np.linspace(88, 2, 40))
    turn = np.vstack([np.stack([np.linspace(50.5, 60, 20), np.full(20, -3.5)], 1),
                      np.stack([60 + 10 * np.cos(th), -13.5 + 10 * np.sin(th)], 1)])
    assert manoeuvre(g, (50.0, -3.5), 0.0, hist, turn, CFG).label == 'turn_right'
    change = np.stack([np.linspace(50.5, 80, 60), np.linspace(-3.5, 0.0, 60)], 1)
    assert manoeuvre(g, (50.0, -3.5), 0.0, hist, change, CFG).label == 'lane_left'
    drift = np.stack([np.linspace(50.5, 80, 60), np.full(60, -2.2)], 1)      # 1.3 m left but still in A2/B2
    assert manoeuvre(g, (50.0, -3.5), 0.0, hist, drift, CFG).label == 'straight'
    off = np.stack([np.linspace(50.5, 80, 60), np.linspace(-3.5, -30, 60)], 1)
    assert manoeuvre(g, (50.0, -3.5), 0.0, hist, off, CFG).label is None
    assert manoeuvre(g, (50.0, -3.5), 0.0, hist, np.full((60, 2), np.nan), CFG).label is None
    assert wrap_deg(190.0) == -170.0
