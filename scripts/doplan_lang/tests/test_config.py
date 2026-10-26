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
