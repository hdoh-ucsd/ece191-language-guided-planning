#!/usr/bin/env python3
"""Check a Valeo4Cast/UniTraj MTR checkpoint; optionally compare model parameter shapes.

Run the --compare-model mode from UniTraj/unitraj after its dependencies are installed.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import torch

# Same path hack as UniTraj's predict.py: run from UniTraj/unitraj so that
# `models`, `datasets` and the compiled ops resolve as top-level modules.
sys.path.insert(0, os.path.join(os.getcwd(), 'models', 'mtr', 'ops'))
sys.path.insert(0, os.getcwd())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--compare-model', action='store_true', help='Compare exact names and tensor shapes to Valeo4Cast model')
    p.add_argument('--unsafe-trusted-file', action='store_true',
                   help='Allow pickle deserialization ONLY for a checkpoint from a trusted source')
    args = p.parse_args()
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    except Exception as e:
        if not args.unsafe_trusted_file:
            raise SystemExit(
                f'Safe load failed: {e}\nIf checkpoint source is trusted, retry with --unsafe-trusted-file.'
            )
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f'Expected dict, got {type(ckpt).__name__}')
    print('Checkpoint top-level keys:', list(ckpt)[:30])
    state = ckpt.get('state_dict', ckpt.get('model_state', ckpt))
    if not isinstance(state, dict):
        raise ValueError('No model state dictionary found')
    print('Model state entries:', len(state))
    for key in list(state)[:12]:
        value = state[key]
        shape = tuple(value.shape) if hasattr(value, 'shape') else type(value).__name__
        print('  ', key, shape)

    if args.compare_model:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from models import build_model
        with initialize_config_dir(config_dir=str((Path.cwd() / 'configs').resolve()), version_base=None):
            config = compose(config_name='challenge_config_noisy_val')
        OmegaConf.set_struct(config, False)
        config = OmegaConf.merge(config, config.method)
        model = build_model(config)
        target = model.state_dict()
        missing = sorted(set(target) - set(state))
        unexpected = sorted(set(state) - set(target))
        mismatch = [
            (key, tuple(target[key].shape), tuple(state[key].shape))
            for key in set(target) & set(state)
            if hasattr(state[key], 'shape') and tuple(target[key].shape) != tuple(state[key].shape)
        ]
        print('Missing keys:', len(missing), missing[:10])
        print('Unexpected keys:', len(unexpected), unexpected[:10])
        print('Shape mismatches:', len(mismatch), mismatch[:10])
        if missing or unexpected or mismatch:
            raise SystemExit('Checkpoint is incompatible with the selected model configuration')
        print('SUCCESS: checkpoint parameters match the model names and shapes')


if __name__ == '__main__':
    main()
