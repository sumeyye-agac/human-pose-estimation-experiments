#!/usr/bin/env python3
"""Ensure benchmark markdown is synchronized with benchmark.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify benchmark artifacts consistency.")
    parser.add_argument("--csv", default="results/benchmark.csv")
    parser.add_argument("--markdown", default="results/benchmark.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    md_path = Path(args.markdown)

    if not csv_path.exists() or not md_path.exists():
        raise SystemExit("Missing benchmark.csv or benchmark.md")

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    markdown_text = md_path.read_text(encoding="utf-8")

    missing = []
    for row in rows:
        tool = row.get("tool", "")
        status = row.get("status", "")
        signature = f"| {tool} | {status} |"
        if signature not in markdown_text:
            missing.append(f"{tool} ({status})")

    if missing:
        print("Benchmark markdown is missing rows:")
        for item in missing:
            print(f"- {item}")
        raise SystemExit(1)

    print(f"Consistency check passed for {len(rows)} benchmark rows.")


if __name__ == "__main__":
    main()
