#!/usr/bin/env python
"""Two controls for the 'never finishing' finding: budget ladders and prompt length.

Both answer objections to reading the truncation numbers as a property of the
language form rather than of the measurement setup.

  budget ladder  generations.jsonl is append-only: when a split was re-run at a
                 larger token budget, every attempt is still on disk. So for
                 rows that hit the ceiling at budget B and were regenerated at
                 a larger B', we can ask how many finished the second time. If
                 most still do not, the failure is non-termination rather than a
                 budget set slightly too low, and 'a bigger budget does not fix
                 it' is a measurement rather than a guess.

  prompt length  Romanized text could simply cost more tokens, in which case
                 lower accuracy would partly be a longer-input artifact. Mean
                 prompt tokens per question, per form, each model normalised to
                 its own English, settles it: the two script variants of a form
                 cost nearly identically, so script is close to free in tokens
                 while being expensive in accuracy.

Needs runs/ with generations.jsonl. Slim predictions keep only the last attempt
per row, so they cannot substitute for the ladder half of this.

Note that results/ is gitignored: this repository ships code, not results. So a
fresh clone must re-run the sweep before this script -- and before the report
builders, which read its output -- has anything to read.

    python scripts/budget_fertility.py
"""
from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

RUN = Path("runs/tax-final")
OUT = Path("results/budget_fertility.json")
SUF = {"_en": "EN", "": "CM", "_hi": "HI", "_rom": "CM-ROM", "_hirom": "HI-ROM"}
FORMS = ["EN", "CM", "HI", "CM-ROM", "HI-ROM"]
#: Only models that appear in the report. The uncapped sarvam-30b run is on disk
#: but was excluded from the study, so it must not prop up a reported number.
REPORTED = set(json.loads(Path("results/tax_dataset.json").read_text())["models"])


def split_form(name: str) -> tuple[str, str] | tuple[None, None]:
    for task in ("mmlu", "gsm8k"):
        stem = f"{task}_hineng"
        if name.startswith(stem):
            form = SUF.get(name[len(stem):])
            return (task, form) if form else (None, None)
    return None, None


def main() -> int:
    ladder = collections.defaultdict(lambda: [0, 0])      # (task, form) -> [retried, still]
    steps: set[tuple[int, int]] = set()
    lad_models: set[str] = set()
    fert: dict[tuple[str, str], dict[str, float]] = collections.defaultdict(dict)

    for f in sorted(RUN.glob("*/*/generations.jsonl")):
        model, split = f.parts[-3], f.parts[-2]
        if model not in REPORTED:
            continue
        task, form = split_form(split)
        if not form:
            continue

        attempts: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
        prompt_tok: dict[int, int] = {}
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("max_tokens"):
                    attempts[int(r["index"])].append(
                        (int(r["max_tokens"]), r.get("finish_reason") or ""))
                if r.get("prompt_tokens"):
                    prompt_tok[int(r["index"])] = int(r["prompt_tokens"])

        if prompt_tok:
            fert[(model, task)][form] = st.mean(prompt_tok.values())

        budgets = sorted({b for v in attempts.values() for b, _ in v})
        if len(budgets) < 2:
            continue
        lo, hi = budgets[0], budgets[-1]
        # Rows that hit the ceiling at the smallest budget AND were retried at
        # the largest. Rows never retried say nothing about what a retry buys.
        retried = [v for v in attempts.values()
                   if any(b == lo and fr == "length" for b, fr in v)
                   and any(b == hi for b, _ in v)]
        if not retried:
            continue
        lad_models.add(model)
        steps.add((lo, hi))
        still = sum(1 for v in retried if any(b == hi and fr == "length" for b, fr in v))
        ladder[(task, form)][0] += len(retried)
        ladder[(task, form)][1] += still

    out: dict = {"budget_ladder": {}, "prompt_length": {}}

    for task in ("mmlu", "gsm8k"):
        per = {}
        for form in FORMS:
            n, s = ladder[(task, form)]
            if n:
                per[form] = {"retried": n, "still_truncated": s,
                             "still_pct": round(100 * s / n, 2)}
        if per:
            tn = sum(v["retried"] for v in per.values())
            ts = sum(v["still_truncated"] for v in per.values())
            out["budget_ladder"][task] = {
                "forms": per, "retried": tn, "still_truncated": ts,
                "still_pct": round(100 * ts / tn, 2)}
    out["budget_ladder"]["n_models"] = len(lad_models)
    out["budget_ladder"]["steps"] = sorted(f"{a}->{b}" for a, b in steps)
    tn = sum(out["budget_ladder"][t]["retried"] for t in ("mmlu", "gsm8k")
             if t in out["budget_ladder"])
    ts = sum(out["budget_ladder"][t]["still_truncated"] for t in ("mmlu", "gsm8k")
             if t in out["budget_ladder"])
    out["budget_ladder"]["overall"] = {"retried": tn, "still_truncated": ts,
                                       "still_pct": round(100 * ts / tn, 2)}

    for task in ("mmlu", "gsm8k"):
        # Each model normalised to its own English before averaging, so a model
        # with a coarse tokenizer cannot dominate the mean ratio.
        ratios = collections.defaultdict(list)
        absol = collections.defaultdict(list)
        for (_m, t), d in fert.items():
            if t != task or "EN" not in d:
                continue
            for form in FORMS:
                if form in d:
                    ratios[form].append(d[form] / d["EN"])
                    absol[form].append(d[form])
        out["prompt_length"][task] = {
            form: {"mean_tokens": round(st.mean(absol[form]), 1),
                   "ratio_to_en": round(st.mean(ratios[form]), 3),
                   "n_models": len(ratios[form])}
            for form in FORMS if ratios[form]}

    OUT.write_text(json.dumps(out, indent=2))

    bl = out["budget_ladder"]
    print(f"budget ladder ({bl['n_models']} models, steps {', '.join(bl['steps'])}):")
    for task in ("mmlu", "gsm8k"):
        if task not in bl:
            continue
        t = bl[task]
        print(f"  {task}: {t['retried']} rows retried at a larger budget, "
              f"{t['still_truncated']} ({t['still_pct']:.1f}%) still unfinished")
        for form, v in t["forms"].items():
            print(f"     {form:8} {v['retried']:>5} -> {v['still_pct']:5.1f}% still")
    o = bl["overall"]
    print(f"  overall: {o['retried']} retried, {o['still_pct']:.1f}% still unfinished")
    print("\nprompt length (mean tokens/question, each model vs its own English):")
    for task in ("mmlu", "gsm8k"):
        print(f"  {task}:")
        for form, v in out["prompt_length"][task].items():
            print(f"     {form:8} {v['mean_tokens']:7.1f} tok  {v['ratio_to_en']:.2f}x")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
