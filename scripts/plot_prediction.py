#!/usr/bin/env python3
"""Plot one Valeo4Cast prediction .pt artifact vs its corresponding future trajectory."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction', type=Path, help='One output_MTR/*.pt prediction')
    parser.add_argument('--out', type=Path, default=Path('forecast_preview.png'))
    parser.add_argument('--past-len', type=int, default=21)
    parser.add_argument('--unsafe-trusted-file', action='store_true',
                        help='Allow pickle deserialization only for a prediction you generated or trust')
    args = parser.parse_args()
    try:
        pred = torch.load(args.prediction, map_location='cpu', weights_only=True)
    except Exception as e:
        if not args.unsafe_trusted_file:
            raise SystemExit(f'Safe load failed: {e}; retry with --unsafe-trusted-file only for a trusted file.')
        pred = torch.load(args.prediction, map_location='cpu', weights_only=False)
    modes = np.asarray(pred['pred_trajs'])
    scores = np.asarray(pred['pred_scores'])
    gt = np.asarray(pred['gt_trajs'])
    print('Scenario:', pred['scenario_id'], 'Object:', pred['object_id'])
    print('Prediction modes:', modes.shape, 'Probabilities:', scores)
    assert modes.ndim == 3 and modes.shape[-1] == 2
    assert gt.ndim == 2 and gt.shape[-1] >= 10
    history, future = gt[:args.past_len], gt[args.past_len:args.past_len+modes.shape[1]]
    fig, ax = plt.subplots(figsize=(8, 6))
    if (history[:, -1] > 0).any():
        ax.plot(history[history[:, -1] > 0, 0], history[history[:, -1] > 0, 1],
                label='Observed history', linewidth=3)
    if (future[:, -1] > 0).any():
        ax.plot(future[future[:, -1] > 0, 0], future[future[:, -1] > 0, 1],
                label='Recorded future', linewidth=3, linestyle='--')
    for i, tr in enumerate(modes):
        ax.plot(tr[:, 0], tr[:, 1], alpha=.75, label=f'MTR mode {i+1} ({scores[i]:.2f})')
    ax.set(xlabel='Local X (m)', ylabel='Local Y (m)', title='Pretrained MTR predictions vs recorded trajectory')
    ax.axis('equal')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print('Saved:', args.out)


if __name__ == '__main__':
    main()
