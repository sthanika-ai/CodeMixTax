#!/usr/bin/env python
"""Truncation by model x condition x token budget.

Reconstructed from generations.jsonl, which is append-only: every regeneration
appends new rows tagged with the `max_tokens` in force at the time, superseding
earlier ones on read. So the full budget-sensitivity history is recoverable from
disk without re-running anything.

Rows written before per-row budget provenance existed carry max_tokens=None and
are attributed to the budget given by --initial (default 1024).

Read the tables as a staged process: at each budget only the rows that truncated
at the PREVIOUS budget are regenerated, so "attempted" shrinks down the column
while "still truncated" is what carried forward.

    python scripts/truncation_report.py
    python scripts/truncation_report.py --run-dir runs/tax-final --task gsm8k
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

COND = [("EN", "_en"), ("CM", ""), ("HI", "_hi"), ("CM-ROM", "_rom"), ("HI-ROM", "_hirom")]


def cond_of(split: str, task: str) -> str | None:
    for label, suffix in sorted(COND, key=lambda x: -len(x[1])):
        if split == f"{task}_hineng{suffix}":
            return label
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/tax-final")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--initial", type=int, default=1024)
    ap.add_argument("--log", default="logs/truncation_by_budget.log")
    ap.add_argument(
        "--restrict",
        help="file of row indices (one per line) to restrict every condition to, so all "
             "conditions share one denominator. For GSM8K use "
             "data/raw/gsm8k/gsm8k_hineng_paired_index.txt: the published CM split has "
             "1016 rows while the derived conditions have 955, and comparing rates across "
             "different denominators is misleading.",
    )
    args = ap.parse_args()

    lines: list[str] = []

    def out(*p):
        s = " ".join(str(x) for x in p)
        print(s)
        lines.append(s)

    root = Path(args.run_dir)
    models = sorted(d.name for d in root.iterdir() if d.is_dir())

    restrict: set[int] | None = None
    if args.restrict:
        restrict = {int(x) for x in Path(args.restrict).read_text().split()}

    out("=" * 92)
    out(f"TRUNCATION BY TOKEN BUDGET   run-dir={args.run_dir}  task={args.task}")
    out("=" * 92)

    grand: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for model in models:
        per: dict[str, dict[int, list[int]]] = {}
        for sd in sorted((root / model).iterdir()):
            c = cond_of(sd.name, args.task)
            if c is None:
                continue
            gp = sd / "generations.jsonl"
            if not gp.exists():
                continue
            stage: dict[int, list[int]] = defaultdict(lambda: [0, 0])  # budget -> [attempted, truncated]
            for line in gp.open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if restrict is not None and int(r["index"]) not in restrict:
                    continue
                b = r.get("max_tokens") or args.initial
                stage[b][0] += 1
                if r.get("finish_reason") == "length":
                    stage[b][1] += 1
            per[c] = stage
            for b, (a, t) in stage.items():
                grand[b][0] += a
                grand[b][1] += t

        if not per:
            continue
        budgets = sorted({b for st in per.values() for b in st})
        out(f"\n### {model}")
        # Denominator for the whole-split view: rows seen in the first pass.
        split_n = {c: st.get(min(st), [0, 0])[0] for c, st in per.items()}
        head = f"{'condition':<10}" + "".join(f"  |{f'{b} tok':^30}" for b in budgets)
        out(head)
        out(f"{'':<10}" + "".join(
            f"  |{'attempt':>8}{'trunc':>6}{'%attempt':>9}{'%split':>7} " for _ in budgets))
        out("-" * len(head))
        for label, _ in COND:
            st = per.get(label)
            if not st:
                continue
            row = f"{label:<10}"
            n_split = split_n.get(label) or 0
            for b in budgets:
                a, t = st.get(b, [0, 0])
                if a:
                    pa = f"{100*t/a:.1f}%"
                    ps = f"{100*t/n_split:.1f}%" if n_split else "-"
                    row += f"  |{a:>8}{t:>6}{pa:>9}{ps:>7} "
                else:
                    row += f"  |{'-':>8}{'-':>6}{'-':>9}{'-':>7} "
            out(row)

    out("\n" + "=" * 92)
    out("TOTALS across all models and conditions")
    out(f"{'budget':>10}{'rows attempted':>17}{'truncated':>12}{'rate':>8}   interpretation")
    out("-" * 92)
    notes = {
        args.initial: "initial pass over every row",
        4096: "only rows that truncated at the previous budget",
        8192: "only rows that truncated at 4096",
    }
    for b in sorted(grand):
        a, t = grand[b]
        out(f"{b:>10}{a:>17}{t:>12}{f'{100*t/a:.1f}%':>8}   {notes.get(b, '')}")

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nlog written to: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
