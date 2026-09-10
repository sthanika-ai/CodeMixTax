#!/usr/bin/env python
"""Paired bootstrap confidence intervals for the headline retention figures.

The uncertainty in these numbers comes from having sampled a finite set of
questions, so the resample is over ITEMS, not models -- and it is paired: one
resample of item indices is applied to every condition and every model at once,
which is what makes a difference between conditions interpretable.

Two things get intervals, because they answer different questions:

  conditions   the mean retention of each language form.
  differences  the gap BETWEEN two forms. This is the one to quote when
               comparing forms: two intervals that fail to overlap imply a
               real difference, but overlapping intervals do NOT imply the
               absence of one, and the paired difference is strictly more
               powerful than eyeballing two marginal intervals.

Also emits a sensitivity table: the headline means recomputed over a range of
eligibility cut-offs, so a reader can see the finding does not depend on where
that line was drawn.

Writes results/bootstrap_ci.json.

    python scripts/bootstrap_ci.py [--iters 2000]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

RUN = Path("runs/tax-final")
COND = [("EN", "_en"), ("CM", ""), ("HI", "_hi"), ("CM-ROM", "_rom"), ("HI-ROM", "_hirom")]
FLOOR = {"mmlu": 25.0, "gsm8k": 0.0}
CUT = {"mmlu": 35.0, "gsm8k": 15.0}

#: (label, a, b) -> reported as retention(a) - retention(b), in points.
#: Chosen to match the comparisons the report actually makes in prose, so that
#: every claim of the form "X costs more than Y" has an interval behind it.
DIFFS = [
    ("mixing (CM vs EN)",              "CM",     "EN"),
    ("Hindi script (HI vs EN)",        "HI",     "EN"),
    ("romanizing a mixed sentence",    "CM-ROM", "CM"),
    ("romanizing Hindi",               "HI-ROM", "HI"),
    ("mixing, once romanized",         "HI-ROM", "CM-ROM"),
    ("script vs lexicon (HI vs CM)",   "HI",     "CM"),
    ("best vs worst form (CM vs HI-ROM)", "CM",   "HI-ROM"),
]

#: The 2x2 interaction: is the penalty for romanizing LARGER when the sentence
#: is all-Hindi than when it is code-mixed? A positive interval means script
#: and lexicon compound rather than merely add.
INTERACTION = ("romanizing costs more in pure Hindi than in Hinglish",
               ("HI", "HI-ROM"), ("CM", "CM-ROM"))

#: Eligibility cut-offs to re-run the headline over, as a sensitivity check.
SWEEP = {"mmlu": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
         "gsm8k": [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    csv.field_size_limit(1 << 30)

    ds = json.loads(Path("results/tax_dataset.json").read_text())
    models = list(ds["models"])
    paired = set(json.loads(Path("results/gsm8k_cm_paired.json").read_text()))
    rng = np.random.default_rng(args.seed)
    out: dict = {"iters": args.iters, "seed": args.seed, "tasks": {}}

    for task in ("mmlu", "gsm8k"):
        # correct[model][cond] as a 0/1 vector, item order identical across all of them
        correct: dict[str, dict[str, np.ndarray]] = {}
        for slug in models:
            correct[slug] = {}
            for cond, sfx in COND:
                d = RUN / slug / f"{task}_hineng{sfx}"
                rows = list(csv.DictReader((d / "predictions.slim.csv").open(newline="")))
                if task == "gsm8k" and cond == "CM":
                    rows = [r for r in rows if int(r["index"]) in paired]
                rows.sort(key=lambda r: int(r["index"]))
                correct[slug][cond] = np.fromiter(
                    (r["correct"] == "True" for r in rows), dtype=np.int8, count=len(rows))
        n = len(correct[models[0]]["EN"])
        eligible = [s for s in models
                    if 100 * correct[s]["EN"].mean() >= CUT[task]]
        fl = FLOOR[task]

        # point estimates and the bootstrap distribution of the cross-model mean
        draws = {c: np.empty(args.iters) for c, _ in COND}
        for b in range(args.iters):
            pick = rng.integers(0, n, n)
            for cond, _ in COND:
                vals = []
                for s in eligible:
                    en = 100 * correct[s]["EN"][pick].mean()
                    if en - fl <= 0:
                        continue
                    a = 100 * correct[s][cond][pick].mean()
                    vals.append(100 * (a - fl) / (en - fl))
                draws[cond][b] = float(np.mean(vals))

        rec = {"n_items": int(n), "n_eligible": len(eligible), "eligible": eligible,
               "cut": CUT[task], "conditions": {}, "differences": {},
               "sensitivity": []}
        for cond, _ in COND:
            point = float(np.mean([
                100 * (100 * correct[s][cond].mean() - fl)
                / (100 * correct[s]["EN"].mean() - fl) for s in eligible]))
            lo, hi = np.percentile(draws[cond], [2.5, 97.5])
            rec["conditions"][cond] = {"mean_retention": round(point, 2),
                                       "ci95": [round(float(lo), 2), round(float(hi), 2)]}
        # Differences off the SAME resamples: draws[a][b] and draws[b][b] share
        # one item bootstrap, so subtracting them keeps the pairing and yields a
        # much tighter interval than comparing the two marginals would suggest.
        def _mean_ret(cond: str, models_: list[str], *,
                      _c: dict = correct, _fl: float = fl) -> float:
            return float(np.mean([
                100 * (100 * _c[s][cond].mean() - _fl)
                / (100 * _c[s]["EN"].mean() - _fl) for s in models_]))

        for label, a, b in DIFFS:
            d = draws[a] - draws[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rec["differences"][label] = {
                "a": a, "b": b,
                "delta": round(_mean_ret(a, eligible) - _mean_ret(b, eligible), 2),
                "ci95": [round(float(lo), 2), round(float(hi), 2)],
                "excludes_zero": bool(lo > 0 or hi < 0),
            }

        (ilabel, (p1, p2), (q1, q2)) = INTERACTION
        inter = (draws[p1] - draws[p2]) - (draws[q1] - draws[q2])
        lo, hi = np.percentile(inter, [2.5, 97.5])
        rec["interaction"] = {
            "label": ilabel,
            "delta": round((_mean_ret(p1, eligible) - _mean_ret(p2, eligible))
                           - (_mean_ret(q1, eligible) - _mean_ret(q2, eligible)), 2),
            "ci95": [round(float(lo), 2), round(float(hi), 2)],
            "excludes_zero": bool(lo > 0 or hi < 0),
        }

        # Sensitivity: does the headline move if the eligibility line moves?
        for cut in SWEEP[task]:
            elig = [s for s in models if 100 * correct[s]["EN"].mean() >= cut]
            if not elig:
                continue
            rec["sensitivity"].append({
                "cut": cut, "n_eligible": len(elig),
                "retention": {c: round(_mean_ret(c, elig), 2) for c, _ in COND},
            })

        out["tasks"][task] = rec
        print(f"  {task}: n={n} items, {len(eligible)} eligible models")
        for cond, _ in COND:
            c = rec["conditions"][cond]
            print(f"    {cond:8} {c['mean_retention']:6.2f}%  "
                  f"95% CI [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]")
        print("    -- differences (paired) --")
        for label, _, _ in DIFFS:
            d = rec["differences"][label]
            star = "" if d["excludes_zero"] else "   (includes 0)"
            print(f"    {label:34} {d['delta']:+7.2f} pp  "
                  f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]{star}")
        i = rec["interaction"]
        star = "" if i["excludes_zero"] else "   (includes 0)"
        print(f"    {'INTERACTION':34} {i['delta']:+7.2f} pp  "
              f"[{i['ci95'][0]:+.2f}, {i['ci95'][1]:+.2f}]{star}")
        print("    -- sensitivity to the eligibility cut --")
        for row in rec["sensitivity"]:
            mark = "  <- reported" if row["cut"] == CUT[task] else ""
            r = row["retention"]
            print(f"      cut {row['cut']:>5.1f}  n={row['n_eligible']:>2}  "
                  + "  ".join(f"{c}={r[c]:.1f}" for c, _ in COND) + mark)

    Path("results/bootstrap_ci.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/bootstrap_ci.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
