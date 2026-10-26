import csv
import json
import shutil

import pandas as pd
import pytest

from doplan_lang import run_all
from doplan_lang.align import load_lidar_frames

from .conftest import PROJECT_ROOT

TOY_LOG = '2021.08.24.13.12.55_veh-45_00386_00472'
SCENARIONET = PROJECT_ROOT / 'scenarionet_data' / 'nuplan_toy'
DB = PROJECT_ROOT / 'nuplan_data' / 'nuplan-v1.1' / 'splits' / 'mini' / f'{TOY_LOG}.db'
CSV = PROJECT_ROOT / 'nuplan_data' / 'doplan' / 'release_annotationsV1.csv'
have_data = (SCENARIONET / 'output_MTR').is_dir() and DB.exists() and CSV.exists()


@pytest.mark.skipif(not have_data, reason='toy nuPlan conversion, predictions or doPlan CSV not present')
def test_full_chain_on_toy_log(tmp_path):
    # The toy log has no doPlan annotation, so fabricate one covering the whole 86 s drive and
    # append it to the real CSV (the real rows are the training corpus).
    fake_csv = tmp_path / 'annotations.csv'
    with open(CSV, newline='', encoding='utf-8') as src, open(fake_csv, 'w', newline='', encoding='utf-8') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow(row)
        writer.writerow({'annotation_id': 'fake_0001', 'clip_id': TOY_LOG, 'start_frame': '0', 'end_frame': '430',
                         'window_duration_sec': '86.0', 'passenger_instruction': 'Turn left at the intersection.',
                         'referential_label': 'Static Referential', 'is_blank': 'false', 'annotator_id': 'test'})
        writer.writerow({'annotation_id': 'fake_0002', 'clip_id': TOY_LOG, 'start_frame': '150', 'end_frame': '200',
                         'window_duration_sec': '10.0', 'passenger_instruction': 'Stop at the light.',
                         'referential_label': 'Static Referential', 'is_blank': 'false', 'annotator_id': 'test'})
    out = tmp_path / 'out'
    metrics = run_all.main(['--scenarionet', str(SCENARIONET), '--nuplan', str(PROJECT_ROOT / 'nuplan_data'),
                            '--csv', str(fake_csv), '--out', str(out)])
    for name in ('word2vec.kv', 'word2vec_report.md', 'intent_clf.joblib', 'intent_report.md', 'pairs.csv',
                 'coverage.csv', 'reranked.csv', 'reranked_rules.csv', 'metrics.json', 'metrics.md'):
        assert (out / name).exists(), name
    pairs = pd.read_csv(out / 'pairs.csv')
    long_win, short_win = pairs[pairs.annotation_id == 'fake_0001'], pairs[pairs.annotation_id == 'fake_0002']
    assert len(long_win) == 58                                         # the 86 s window holds every prediction
    t0 = int(load_lidar_frames(DB).ts[0])
    lo, hi = t0 + 150 * 200_000, t0 + 200 * 200_000                   # frames 150..200 at 0.2 s = 30 s .. 40 s
    expected_short = int(((long_win.t_pred_us >= lo) & (long_win.t_pred_us <= hi)).sum())
    assert 0 < len(short_win) == expected_short < 58
    assert metrics['classifier']['n_pairs'] == 58 + expected_short
    assert metrics['classifier']['gated']['n_pairs'] > 0 and 0 < metrics['classifier']['gate_open_fraction'] < 1
    reranked = pd.read_csv(out / 'reranked.csv')
    assert (reranked.loc[reranked.annotation_id == 'fake_0001', 'intent'] == 'turn_left').all()
    assert (reranked.loc[reranked.annotation_id == 'fake_0002', 'intent'] == 'stop').all()
    assert json.loads((out / 'metrics.json').read_text())['rules_only']['n_pairs'] == 58 + expected_short
    assert 'NaN' not in (out / 'metrics.json').read_text()
    assert len(list((out / 'plots').glob('*.png'))) >= 1
    assert len(list((out / 'reranked_pt').glob('*.pt'))) == 58 + expected_short

    out2 = tmp_path / 'out2'
    out2.mkdir()
    for name in ('word2vec.kv', 'intent_clf.joblib'):
        shutil.copy(out / name, out2 / name)
    metrics2 = run_all.main(['--scenarionet', str(SCENARIONET), '--nuplan', str(PROJECT_ROOT / 'nuplan_data'),
                             '--csv', str(fake_csv), '--out', str(out2), '--no-pt'])
    assert not (out2 / 'reranked_pt').exists()
    assert metrics2['classifier']['n_pairs'] == metrics['classifier']['n_pairs']
