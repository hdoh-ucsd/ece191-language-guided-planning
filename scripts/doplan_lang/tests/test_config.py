from doplan_lang.config import Config, EVAL_LOGS, INTENTS, LATERAL_INTENTS


def test_intent_order_is_fixed():
    assert INTENTS == ('turn_left', 'turn_right', 'lane_left', 'lane_right', 'stop', 'straight', 'other')
    assert set(LATERAL_INTENTS) < set(INTENTS)


def test_six_eval_logs():
    assert len(EVAL_LOGS) == 6 and len(set(EVAL_LOGS)) == 6


def test_config_is_frozen_with_defaults():
    cfg = Config()
    assert cfg.lam == 1.0 and cfg.past_len == 21
    try:
        cfg.lam = 2.0
    except Exception:
        return
    raise AssertionError('Config must be frozen')


def test_v2_knobs_present():
    cfg = Config()
    assert cfg.compat_source == 'map' and cfg.geom_lane_heading_deg == 20.0 and cfg.lane_heading_deg == 90.0
    assert cfg.stop_speed_mps == 1.0 and cfg.sim_dt == 0.1 and cfg.sim_max_curvature == 0.2
