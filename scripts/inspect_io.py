#!/usr/bin/env python3
"""Describe what goes into and comes out of the Valeo4Cast MTR model.

Reads one preprocessed sample chunk from the UniTraj cache (the model input, before
collation) and one prediction file written by predict.py (the model output), prints every
field with its shape, and plots the first prediction against the recorded trajectory.

Both files are produced locally by run_inference.sh; they are pickles and are loaded as
trusted data.
"""
from __future__ import annotations
import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import torch

# Layout of the last axis of obj_trajs (NUM_INPUT_ATTR_AGENT = 39), see
# BaseDataset.get_agent_data. Coordinates are in the target agent's frame at the
# prediction time (origin at its position, x axis along its heading).
AGENT_FEATURES = [
    (0, 6, 'x, y, z, length, width, height (target-centred)'),
    (6, 11, 'one-hot: vehicle, pedestrian, cyclist, is-target, is-ego'),
    (11, 32, 'one-hot time index (21 past steps)'),
    (32, 33, 'timestamp in seconds'),
    (33, 35, 'sin(heading), cos(heading)'),
    (35, 37, 'vx, vy'),
    (37, 39, 'ax, ay (finite difference at 10 Hz)'),
]

# Layout of the last axis of map_polylines (NUM_INPUT_ATTR_MAP = 9), see get_map_data.
MAP_FEATURES = [
    (0, 3, 'x, y, z (target-centred)'),
    (3, 6, 'direction unit vector'),
    (6, 7, 'polyline type id'),
    (7, 9, 'previous point x, y'),
]

INPUT_NOTES = {
    'obj_trajs': 'agent history, [agents, past steps, 39]; zero where obj_trajs_mask is False',
    'obj_trajs_mask': 'which (agent, step) entries are observed',
    'track_index_to_predict': 'row of obj_trajs that is the target agent',
    'obj_trajs_pos': 'x,y,z of every agent per past step',
    'obj_trajs_last_pos': 'last observed x,y,z per agent',
    'obj_trajs_future_state': 'future x,y,vx,vy of every agent (used for loss only, not seen by the encoder)',
    'obj_trajs_future_mask': 'validity of the future states',
    'center_objects_world': 'target state at prediction time in world coordinates (x,y,z,l,w,h,heading,vx,vy,valid)',
    'center_objects_id': 'target object id',
    'center_objects_type': 'target type id (1 vehicle, 2 pedestrian, 3 cyclist)',
    'center_gt_trajs': 'target future x,y,vx,vy in target frame, [60, 4]',
    'center_gt_trajs_mask': 'validity of each future step',
    'center_gt_final_valid_idx': 'index of last valid future step',
    'center_gt_trajs_src': 'full world-frame source trajectory, [past+future, 10]',
    'map_polylines': 'vectorised map, [polylines, points per polyline, 9]',
    'map_polylines_mask': 'valid points per polyline',
    'map_polylines_center': 'x,y,z centre of each polyline',
    'map_center': 'global offset added back when converting predictions to world frame',
    'scenario_id': 'ScenarioNet scenario id',
    'current_time_indices': 'frame index of the prediction time inside the scenario',
    'kalman_difficulty': 'difficulty score of the ground-truth future (postprocess)',
    'trajectory_type': 'motion class of the ground-truth future (postprocess)',
    'dataset_name': 'source dataset',
}

OUTPUT_NOTES = {
    'pred_trajs': 'predicted x,y per mode in WORLD frame, [modes, future steps, 2]',
    'pred_scores': 'probability of each mode (softmax, after NMS)',
    'past_det_scores': 'detection confidence of the target (1.0 when inputs are ground truth)',
    'gt_trajs': 'same as input center_gt_trajs_src, world frame, [past+future, 10]',
    'object_id': 'target object id',
    'object_type': 'target type id',
    'scenario_id': 'ScenarioNet scenario id',
    'track_index_to_predict': 'row of obj_trajs that was the target',
    'current_time_index': 'frame index of the prediction time',
}


def describe(value):
    if isinstance(value, np.ndarray):
        return f'ndarray {tuple(value.shape)} {value.dtype}'
    if isinstance(value, torch.Tensor):
        return f'tensor {tuple(value.shape)} {value.dtype}'
    if isinstance(value, (str, bytes, int, float, bool, np.generic)):
        return f'{type(value).__name__} = {value!r}'
    return type(value).__name__


def print_fields(d, notes):
    width = max(len(k) for k in d)
    for key in d:
        note = notes.get(key, '')
        print(f'  {key:<{width}}  {describe(d[key]):<38}  {note}')


def print_layout(title, layout):
    print(title)
    for start, end, meaning in layout:
        print(f'  [{start:2d}:{end:2d}]  {meaning}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True, help='UniTraj cache dir (cache_test_MTR)')
    p.add_argument('--outputs', type=Path, required=True, help='predict.py output dir (output_MTR)')
    p.add_argument('--plot', type=Path, default=None, help='Write a PNG of the first prediction')
    p.add_argument('--past-len', type=int, default=21)
    args = p.parse_args()

    chunks = sorted(glob.glob(str(args.cache / '*___*.pkl')))
    chunks = [c for c in chunks if not c.endswith('scenario_summary.pkl')]
    if not chunks:
        raise SystemExit(f'No cached samples in {args.cache}')
    with open(chunks[0], 'rb') as f:
        samples = pickle.load(f)
    sample = samples[0]

    print('=' * 100)
    print('MODEL INPUT  (one preprocessed sample; collate_fn stacks these along a new batch axis)')
    print(f'cache file: {os.path.basename(chunks[0])}  ({len(samples)} samples in this chunk, '
          f'{len(chunks)} chunks in cache)')
    print('=' * 100)
    print_fields(sample, INPUT_NOTES)
    print()
    print_layout('obj_trajs last-axis layout (39 features):', AGENT_FEATURES)
    print_layout('map_polylines last-axis layout (9 features):', MAP_FEATURES)
    mask = np.asarray(sample['obj_trajs_mask'])
    print()
    print(f'agents observed at prediction time: {int(mask[:, -1].sum())} of {mask.shape[0]} slots, '
          f'past steps observed for target: {int(mask[int(sample["track_index_to_predict"]), :].sum())} '
          f'of {mask.shape[1]}')
    fut = np.asarray(sample['center_gt_trajs_mask'])
    print(f'future ground-truth steps available: {int(fut.sum())} of {fut.shape[0]}')

    outs = sorted(glob.glob(str(args.outputs / '*.pt')))
    if not outs:
        raise SystemExit(f'No predictions in {args.outputs}')
    pred = torch.load(outs[0], map_location='cpu', weights_only=False)

    print()
    print('=' * 100)
    print(f'MODEL OUTPUT  (one prediction file; {len(outs)} files = targets x prediction times)')
    print(f'file: {os.path.basename(outs[0])}')
    print('=' * 100)
    print_fields(pred, OUTPUT_NOTES)

    modes = np.asarray(pred['pred_trajs'])
    scores = np.asarray(pred['pred_scores'])
    gt = np.asarray(pred['gt_trajs'])
    future = gt[args.past_len:args.past_len + modes.shape[1]]
    valid = future[:, -1] > 0
    print()
    print(f'mode probabilities: {np.round(scores, 3).tolist()}  (best mode: {int(scores.argmax())})')
    if valid.any():
        err = np.linalg.norm(modes[:, valid, :] - future[valid, :2][None], axis=-1)  # [modes, T]
        ade = err.mean(axis=1)
        fde = err[:, -1]
        print(f'minADE over modes: {ade.min():.2f} m   minFDE: {fde.min():.2f} m   '
              f'(over {int(valid.sum())} valid future steps)')
    else:
        print('no valid future ground truth for this sample (end of scenario)')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        history = gt[:args.past_len]
        hv = history[:, -1] > 0
        fig, ax = plt.subplots(figsize=(8, 6))
        if hv.any():
            ax.plot(history[hv, 0], history[hv, 1], 'k-', linewidth=3, label='observed history')
        if valid.any():
            ax.plot(future[valid, 0], future[valid, 1], 'k--', linewidth=3, label='recorded future')
        for i, tr in enumerate(modes):
            ax.plot(tr[:, 0], tr[:, 1], alpha=.8, label=f'mode {i + 1} (p={scores[i]:.2f})')
        ax.set(xlabel='world x (m)', ylabel='world y (m)',
               title=f'{pred["scenario_id"]}\nobject {pred["object_id"]} @ frame {int(pred["current_time_index"])}')
        ax.axis('equal')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        print(f'plot saved: {args.plot}')


if __name__ == '__main__':
    main()
