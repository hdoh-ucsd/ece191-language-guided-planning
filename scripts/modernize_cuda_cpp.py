"""Adapt old CUDA extension pointer APIs to supported PyTorch spelling.

This is a targeted source-only patch, NOT a guarantee that the 2024 CUDA extensions
compile against PyTorch 2.7.1. It is safe to run repeatedly.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_directory", type=Path)
    args = ap.parse_args()
    directory = args.source_directory
    if not directory.is_dir():
        raise SystemExit(f"Missing source directory: {directory}")
    count=0
    for source in sorted(directory.rglob("*")):
        if source.suffix not in {".cpp", ".cu", ".h", ".hpp"} or not source.is_file():
            continue
        old = source.read_text()
        new = old.replace(".data<float>()", ".data_ptr<float>()")
        new = new.replace(".data<int>()", ".data_ptr<int>()")
        new = new.replace(".type().is_cuda()", ".is_cuda()")
        if new != old:
            source.write_text(new)
            count += 1
            print("Updated", source)
    print("Modernized", count, "source files")


if __name__ == "__main__":
    main()
