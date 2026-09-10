#!/usr/bin/env python
"""Write predictions.slim.csv beside every predictions.csv.

`predictions.csv` carries `raw` and `raw_stripped` -- the model's full output --
which is 99.9% of its bytes and already stored in generations.jsonl. The slim
copy keeps the columns that make scoring auditable (index, gold, prediction,
correct) and drops the duplicated text, taking the study's audit trail from
476 MB to 12 MB so it can live in git.

It also carries `finish_reason`, `completion_tokens` and `max_tokens` across from
generations.jsonl. Those three are what truncation analysis needs, and without
them a fresh clone -- which has no generations.jsonl, that being gitignored --
could not rebuild the report.

    python scripts/slim_predictions.py [--run-dir runs]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DROP = ("raw", "raw_stripped")
#: carried over from generations.jsonl so truncation survives without it
CARRY = ("finish_reason", "completion_tokens", "max_tokens")


def budget_fields(gen: Path) -> dict[int, dict]:
    """Last write per index wins, exactly as the runner scores it."""
    if not gen.exists():
        return {}
    out: dict[int, dict] = {}
    with gen.open() as fh:
        for line in fh:
            r = json.loads(line)
            out[int(r["index"])] = {k: r.get(k) for k in CARRY}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs")
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)   # raw generations blow past the default
    n = before = after = 0
    for src in sorted(Path(args.run_dir).rglob("predictions.csv")):
        dst = src.with_name("predictions.slim.csv")
        gen = budget_fields(src.with_name("generations.jsonl"))
        with src.open(newline="") as fh:
            rd = csv.DictReader(fh)
            keep = [c for c in (rd.fieldnames or []) if c not in DROP]
            rows = []
            for r in rd:
                row = {k: r[k] for k in keep}
                row.update(gen.get(int(r["index"]), dict.fromkeys(CARRY)))
                rows.append(row)
        with dst.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keep + list(CARRY))
            wr.writeheader()
            wr.writerows(rows)
        before += src.stat().st_size
        after += dst.stat().st_size
        n += 1
    print(f"wrote {n} slim files: {before/1048576:.0f} MB -> {after/1048576:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
