import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]      # .../scripts
PROJECT_ROOT = SCRIPTS_DIR.parent                      # .../valeo_mtr_docker
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


import numpy as np
import pytest
import torch


def _arc(radius, angle_deg, n=60, left=True):
    th = np.linspace(0, np.radians(angle_deg), n + 1)[1:]
    return np.stack([radius * np.sin(th), radius * (1 - np.cos(th)) * (1 if left else -1)], 1)


def write_fake_prediction(path, scores=(0.3, 0.3, 0.4), heading=0.0, origin=(0.0, 0.0), future='straight'):
    """Three modes (left arc, right arc, straight) around a fake ego; the recorded future is
    the mode named by `future`. Mirrors the fields predict.py writes."""
    modes = {'left': _arc(20, 90), 'right': _arc(20, 90, left=False),
             'straight': np.stack([np.arange(1, 61) * 0.5, np.zeros(60)], 1)}
    c, s = np.cos(heading), np.sin(heading)

    def to_world(xy):
        return np.stack([c * xy[:, 0] - s * xy[:, 1] + origin[0], s * xy[:, 0] + c * xy[:, 1] + origin[1]], 1)

    pred_trajs = np.stack([to_world(modes['left']), to_world(modes['right']), to_world(modes['straight'])]).astype(np.float32)
    gt = np.zeros((81, 10), dtype=np.float32)
    gt[:, 0], gt[:, 1], gt[:, 6], gt[:, 9] = origin[0], origin[1], heading, 1.0
    gt[21:, :2] = to_world(modes[future])
    torch.save({'scenario_id': 'fake', 'pred_trajs': pred_trajs, 'pred_scores': np.array(scores, dtype=np.float32),
                'past_det_scores': np.float32(1.0), 'object_id': 'ego', 'object_type': 1, 'gt_trajs': gt,
                'track_index_to_predict': 0, 'current_time_index': 0}, path)
    return path


@pytest.fixture
def fake_pred_dir(tmp_path):
    d = tmp_path / 'output_MTR'
    d.mkdir()
    write_fake_prediction(d / 'fake___ego___0___.pt')
    return d


def _lane(xy, exits=(), entries=(), left=(), right=(), kind='LANE_SURFACE_STREET', half_width=1.75):
    """A ScenarioNet-style lane feature: centreline, a polygon offset half_width to each side, links."""
    xy = np.asarray(xy, dtype=np.float64)
    seg = np.diff(xy, axis=0)
    n = np.stack([-seg[:, 1], seg[:, 0]], 1) / np.linalg.norm(seg, axis=1)[:, None]
    n = np.vstack([n, n[-1:]])
    polygon = np.vstack([xy + half_width * n, (xy - half_width * n)[::-1]])
    return {'type': kind, 'polyline': xy, 'polygon': polygon, 'exit_lanes': list(exits), 'entry_lanes': list(entries),
            'left_neighbor': list(left), 'right_neighbor': list(right)}


def _arc_xy(centre, radius, deg_from, deg_to, n=40):
    th = np.radians(np.linspace(deg_from, deg_to, n))
    return np.stack([centre[0] + radius * np.cos(th), centre[1] + radius * np.sin(th)], 1)


def synthetic_map_features():
    """Three straight lanes A0 (y=+3.5), A1 (y=0), A2 (y=-3.5) from x=0 to 60 continuing into B0..B2 to
    x=120; a right-turn connector C (A2 -> south, radius 10) and a left-turn connector D (A0 -> north)."""
    xs = np.linspace(0, 60, 61)
    f = {}
    for i, y in enumerate((3.5, 0.0, -3.5)):
        left, right = ([f'A{i - 1}'] if i > 0 else []), ([f'A{i + 1}'] if i < 2 else [])
        f[f'A{i}'] = _lane(np.stack([xs, np.full_like(xs, y)], 1),
                           exits=[f'B{i}'] + (['D'] if i == 0 else ['C'] if i == 2 else []), left=left, right=right)
        f[f'B{i}'] = _lane(np.stack([xs + 60, np.full_like(xs, y)], 1), entries=[f'A{i}'],
                           left=[f'B{i - 1}'] if i > 0 else [], right=[f'B{i + 1}'] if i < 2 else [])
    f['C'] = _lane(_arc_xy((60.0, -13.5), 10.0, 90, 0), entries=['A2'], kind='LANE_SURFACE_UNSTRUCTURE')
    f['D'] = _lane(_arc_xy((60.0, 13.5), 10.0, -90, 0), entries=['A0'], kind='LANE_SURFACE_UNSTRUCTURE')
    return f
