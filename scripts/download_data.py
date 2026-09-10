#!/usr/bin/env python
"""Fetch the 18 Indic splits and print a verification table.

Thin wrapper around `cmb-indic download` for people who would rather run a
script than learn the CLI. Also re-checks row counts against the registry, which
is the check that catches an upstream dataset revision silently changing under a
published result.

    python scripts/download_data.py
    python scripts/download_data.py --revision <commit-sha>   # pin for a paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cmb_indic.data import ensure_downloaded, sha256_of, write_manifest  # noqa: E402
from cmb_indic.registry import DATASETS, TOTAL_ROWS, resolve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--revision", default=None, help="pin a dataset revision")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--datasets", default=None, help="comma-separated subset")
    args = ap.parse_args()

    specs = resolve(args.datasets.split(",") if args.datasets else None)
    data_dir = Path(args.data_dir)

    print(f"Fetching {len(specs)} split(s) -> {data_dir}/")
    print(f"{'split':<22}{'rows':>7}{'expected':>10}  {'bytes':>10}  sha256")
    print("-" * 88)

    entries: dict[str, dict] = {}
    mismatches: list[str] = []
    total_rows = 0

    for spec in specs:
        path = ensure_downloaded(spec, data_dir, revision=args.revision, force=args.force)
        n = len(pd.read_csv(path))
        total_rows += n
        digest = sha256_of(path)
        size = path.stat().st_size
        flag = "" if n == spec.n_rows else "  <-- MISMATCH"
        if n != spec.n_rows:
            mismatches.append(f"{spec.name}: got {n}, registry says {spec.n_rows}")
        print(f"{spec.name:<22}{n:>7}{spec.n_rows:>10}  {size:>10,}  {digest[:16]}{flag}")
        entries[spec.name] = {
            "hf_path": spec.hf_path,
            "local_path": str(path),
            "sha256": digest,
            "bytes": size,
            "rows": n,
            "expected_rows": spec.n_rows,
            "revision": args.revision or "main",
        }

    manifest = write_manifest(data_dir, entries)
    print("-" * 88)
    print(f"{'TOTAL':<22}{total_rows:>7}", end="")
    if len(specs) == len(DATASETS):
        print(f"{TOTAL_ROWS:>10}")
    else:
        print()
    print(f"\nManifest: {manifest}")

    if mismatches:
        print("\nRow counts differ from the registry:", file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(
            "\nThe Hub copy has changed since the registry was measured. Update "
            "src/cmb_indic/registry.py and re-verify before publishing any numbers.",
            file=sys.stderr,
        )
        return 1

    print("All row counts match the registry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
