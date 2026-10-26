#!/usr/bin/env python3
"""Select a tiny ScenarioNet database, preserving the required summary/mapping files."""
import argparse
import pickle
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=Path, required=True, help='Existing ScenarioNet database root')
    p.add_argument('--dst', type=Path, required=True, help='New subset database root')
    p.add_argument('--count', type=int, default=2, help='Number of scenario files to copy')
    p.add_argument('--log-stem', default='', help='Optional substring of original nuPlan DB log stem')
    args = p.parse_args()
    if args.count < 1:
        p.error('--count must be >= 1')
    if args.src.resolve() == args.dst.resolve():
        p.error('Source and destination must differ')

    from metadrive.scenario.scenario_description import ScenarioDescription as SD
    summary_file = args.src / SD.DATASET.SUMMARY_FILE
    mapping_file = args.src / SD.DATASET.MAPPING_FILE
    with summary_file.open('rb') as f:
        summary = pickle.load(f)
    with mapping_file.open('rb') as f:
        mapping = pickle.load(f)
    candidates = [
        name for name, meta in summary.items()
        if not args.log_stem or args.log_stem in name
        or args.log_stem in str(meta.get('log_name', ''))
    ]
    if not candidates:
        raise RuntimeError(f'No ScenarioNet scenarios matched {args.log_stem!r}')
    selected = candidates[:args.count]
    args.dst.mkdir(parents=True, exist_ok=True)
    out_summary, out_mapping = {}, {}
    for name in selected:
        source = args.src / mapping.get(name, '') / name
        dest = args.dst / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if dest.exists():
            raise FileExistsError(f'Refusing to overwrite {dest}')
        shutil.copy2(source, dest)
        out_summary[name] = summary[name]
        out_mapping[name] = ''
    for fname, data in ((SD.DATASET.SUMMARY_FILE, out_summary),
                        (SD.DATASET.MAPPING_FILE, out_mapping)):
        with (args.dst / fname).open('wb') as f:
            pickle.dump(data, f)
    print(f'Copied {len(selected)} scenarios to {args.dst}')
    for name in selected:
        print(' -', name)


if __name__ == '__main__':
    main()
