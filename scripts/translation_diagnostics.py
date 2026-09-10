#!/usr/bin/env python
"""Quantify how much of the measured Hindi gap is translation quality.

Three independent probes, written to results/translation_diagnostics.json:

  verified_vs_mt   Global-MMLU marks 21.6% of its Hindi items as human-verified.
                   Same models, same rows: does accuracy differ on the verified
                   subset? That is a direct read on translation quality.
  unanimous        Items every capable model answers correctly in English and
                   incorrectly in Hindi. Unanimous failure is what a broken
                   question looks like; the reverse direction is the control.
  script_only      Romanising is a deterministic transliteration of text we
                   already have, so any translation error is identical on both
                   sides and cancels exactly. These figures carry no exposure.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

RUN = Path("runs/tax-final")
CUT = {"mmlu": 35.0, "gsm8k": 15.0}


def correct_map(slug: str, split: str) -> dict[int, bool]:
    p = RUN / slug / split / "predictions.slim.csv"
    with p.open(newline="") as fh:
        return {int(r["index"]): r["correct"] == "True" for r in csv.DictReader(fh)}


def main() -> int:
    csv.field_size_limit(1 << 30)
    ds = json.loads(Path("results/tax_dataset.json").read_text())
    M = ds["models"]
    out: dict = {}

    elig = {t: [s for s in M if M[s]["tasks"][t]["EN"]["accuracy"] >= CUT[t]]
            for t in ("mmlu", "gsm8k")}

    # ---- 1. human-verified vs machine-translated (MMLU only; the flag ships there)
    from cmb_indic.data import load_split
    from cmb_indic.registry import ALL_SPECS
    frame = load_split(ALL_SPECS["mmlu_hineng_hi"], Path("data/raw"), download=False).frame
    ann = {i for i, a in enumerate(frame["annotated"]) if a}
    gaps: dict[str, list[float]] = {"HI": [], "HI-ROM": []}
    for slug in elig["mmlu"]:
        for cond, split in (("HI", "mmlu_hineng_hi"), ("HI-ROM", "mmlu_hineng_hirom")):
            c = correct_map(slug, split)
            a = [v for i, v in c.items() if i in ann]
            b = [v for i, v in c.items() if i not in ann]
            gaps[cond].append(100 * sum(a) / len(a) - 100 * sum(b) / len(b))
    out["verified_vs_mt"] = {
        "verified_items": len(ann), "total_items": len(frame),
        "verified_pct": round(100 * len(ann) / len(frame), 1),
        "n_models": len(elig["mmlu"]),
        "mean_gap_points": {k: round(st.mean(v), 2) for k, v in gaps.items()},
    }

    # ---- 2. unanimous English-pass / Hindi-fail, with the reverse as control
    out["unanimous"] = {}
    for task in ("mmlu", "gsm8k"):
        ms = elig[task]
        EN = {m: correct_map(m, f"{task}_hineng_en") for m in ms}
        HI = {m: correct_map(m, f"{task}_hineng_hi") for m in ms}
        idx = sorted(EN[ms[0]])
        sus = sum(1 for i in idx if all(EN[m][i] for m in ms) and not any(HI[m][i] for m in ms))
        ctl = sum(1 for i in idx if all(HI[m][i] for m in ms) and not any(EN[m][i] for m in ms))
        out["unanimous"][task] = {
            "n_models": len(ms), "n_items": len(idx),
            "en_pass_hi_fail": sus, "en_pass_hi_fail_pct": round(100 * sus / len(idx), 2),
            "control_hi_pass_en_fail": ctl,
        }

    # ---- 3. script-only effect: transliteration of identical text
    out["script_only"] = {}
    for task in ("mmlu", "gsm8k"):
        ms = elig[task]
        cm = [M[s]["tasks"][task]["CM"]["accuracy"] - M[s]["tasks"][task]["CM-ROM"]["accuracy"]
              for s in ms]
        hi = [M[s]["tasks"][task]["HI"]["accuracy"] - M[s]["tasks"][task]["HI-ROM"]["accuracy"]
              for s in ms]
        out["script_only"][task] = {
            "n_models": len(ms),
            "romanising_hinglish_costs": round(st.mean(cm), 2),
            "romanising_hindi_costs": round(st.mean(hi), 2),
            "ratio": round(st.mean(hi) / st.mean(cm), 2),
        }

    # ---- 4. exclusions already applied
    out["exclusions"] = {
        "gsm8k_items_dropped_for_number_mismatch": 1016 - ds["gsm8k_cm_paired_n"],
        "gsm8k_items_kept": ds["gsm8k_cm_paired_n"],
        "gsm8k_dropped_pct": round(100 * (1016 - ds["gsm8k_cm_paired_n"]) / 1016, 1),
        "eligibility_rule": {
            "mmlu": "English accuracy >= 35% (guess floor 25%)",
            "gsm8k": "English accuracy >= 15% (no guess floor)"},
        "excluded_models": {t: [M[s]["label"] for s in M if s not in elig[t]]
                            for t in ("mmlu", "gsm8k")},
    }

    Path("results/translation_diagnostics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:1400])
    print("\nwrote results/translation_diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
