#!/usr/bin/env python
"""Consolidated dataset for the Code-Mixing Tax report: MMLU + GSM8K only.

Emits results/tax_dataset.json with, per model x task x condition:
accuracy, floor-adjusted retention, truncation rate, unparsed rate.

Truncation is read from generations.jsonl, which is append-only: the last
entry for an index wins, matching how the runner scores. Rates therefore
reflect the FINAL budget each row reached.
"""
from __future__ import annotations

import json
from pathlib import Path

RUN = Path("runs/tax-final")
COND = [("EN", "_en"), ("CM", ""), ("HI", "_hi"), ("CM-ROM", "_rom"), ("HI-ROM", "_hirom")]
FLOOR = {"mmlu": 25.0, "gsm8k": 0.0}
# display name -> (label, indic post-training?)
MODELS = {
    "gemma3-4b-it":          ("Gemma 3 4B",          False),
    "gemma3-12b-it":         ("Gemma 3 12B",         False),
    "gemma3-27b-it":         ("Gemma 3 27B",         False),
    "llama3.1-8b-it":        ("Llama 3.1 8B",        False),
    "llama4-scout":          ("Llama 4 Scout",       False),
    "mistral-small-3.1-24b": ("Mistral Small 3.1 24B", False),
    "phi-4-14b":             ("Phi-4 14B",           False),
    "gpt-oss-20b":           ("gpt-oss-20b",         False),
    "sarvam-m-24b":          ("Sarvam-M 24B",        True),
    "krutrim-2-12b":         ("Krutrim-2 12B",       True),
    "sarvam-1-2b":           ("Sarvam-1 2B",         True),
    "param-1-2.9b":          ("Param-1 2.9B",        True),
    "openhathi-7b":          ("OpenHathi 7B",        True),
    "gemma3-12b-it-int4":    ("Gemma 3 12B (compressed)", False),
    # Added on the vLLM 0.27.1 stack (.venv-new); the 14 above ran 0.10.1.
    "nemotron-3.5-lightning-30b": ("Nemotron 3.5 Lightning", False),
    # Dagger, not "(cap 4k)": the label appears in every table and plot legend,
    # and the cap needs one footnote rather than four words of every row. The
    # footnote text lives with the results tables in the two report builders.
    "sarvam-30b-cap":        ("Sarvam-30B\u2020",   True),
}

#: Measurement caveats. These live here, in the code that produces the numbers,
#: rather than in the report -- a reader wants the finding, an engineer re-running
#: this wants to know the two models were not measured on identical footing:
#:
#:   nemotron  ran on vLLM 0.27.1 (the other 14 used 0.10.1, which has no
#:             implementation for its Mamba-2 hybrid MoE); budget 16000, with
#:             still-truncated rows regenerated at 32768.
#:   sarvam-30b-cap
#:             ran on vLLM 0.27.1 with reasoning capped at 4000 tokens
#:             (thinking_token_budget). Uncapped it left 25.5% of rows unfinished
#:             and was unreportable. Its numbers are therefore "accuracy under a
#:             reasoning cap", not accuracy with unconstrained reasoning.
#:
#: Also recorded per split in runs/<...>/run_meta.json, which is authoritative.
CAVEATS = {
    "nemotron-3.5-lightning-30b":
        "vLLM 0.27.1 (others 0.10.1); reasoning budget 16000, regenerated at 32768.",
    "sarvam-30b-cap":
        "vLLM 0.27.1; reasoning capped at 4000 tokens via thinking_token_budget, so "
        "this is accuracy under a reasoning cap, not with unconstrained reasoning. "
        "Uncapped the model left 25.5% of rows truncated and was unreportable.",
}

def paired_gsm8k_cm_indices() -> set[int]:
    """Row indices of gsm8k_hineng that pair with the four derived conditions.

    The published code-mixed split carries 1016 items; the derived conditions carry
    955, because items whose Hindi translation garbled a quantity were dropped (such
    an item is a DIFFERENT problem holding the English gold answer). Scoring CM over
    all 1016 while the other four use 955 breaks the within-item pairing the whole
    study rests on, so CM is restricted to the intersection here. The subset is
    recovered by fingerprinting on `cot`, which is verbatim English reasoning and
    unique per item -- the same join the registry uses to build the derived splits.
    """
    # Cached so that rebuilding the report does not require data/raw to still be
    # present: the join needs the source datasets, the report does not. Both the
    # cache and the report live under the gitignored results/, so a clone
    # regenerates the pair together after `make data`.
    cache = Path("results/gsm8k_cm_paired.json")
    if cache.exists():
        return set(json.loads(cache.read_text()))

    from cmb_indic.data import load_split
    from cmb_indic.registry import ALL_SPECS

    raw = Path("data/raw")
    cm = load_split(ALL_SPECS["gsm8k_hineng"], raw, download=False).frame
    en = load_split(ALL_SPECS["gsm8k_hineng_en"], raw, download=False).frame
    first: dict[str, int] = {}
    for i, c in enumerate(cm["cot"]):
        first.setdefault(c, i)
    idx = {first[c] for c in en["cot"] if c in first}
    if len(idx) != len(en):
        raise RuntimeError(
            f"CM/derived pairing incomplete: {len(idx)} of {len(en)} matched. "
            "Refusing to emit a table whose paired claim would be false."
        )
    cache.write_text(json.dumps(sorted(idx)))
    return idx


def live_rows(d: Path) -> list[dict]:
    """Per-row finish_reason / budget, last write wins as the runner scores it.

    Prefers predictions.slim.csv, the compact audit trail a run leaves beside the
    fat generations.jsonl, so that a pruned runs/ tree still rebuilds the report.
    Falls back to the raw file when a working tree has it.
    """
    slim = d / "predictions.slim.csv"
    if slim.exists():
        import csv as _csv
        with slim.open(newline="") as fh:
            return [{"index": int(r["index"]),
                     "finish_reason": r["finish_reason"] or None,
                     "completion_tokens": int(r["completion_tokens"]) if r["completion_tokens"] else 0,
                     "max_tokens": int(r["max_tokens"]) if r["max_tokens"] else None}
                    for r in _csv.DictReader(fh)]
    cache: dict[int, dict] = {}
    with (d / "generations.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            cache[int(row["index"])] = row      # last wins, as the runner does
    return list(cache.values())

def main() -> None:
    cm_paired = paired_gsm8k_cm_indices()
    out: dict = {"floor": FLOOR, "models": {}, "gsm8k_cm_paired_n": len(cm_paired)}
    for slug, (label, indic) in MODELS.items():
        rec = {"label": label, "indic": indic, "tasks": {},
               "caveat": CAVEATS.get(slug)}
        for task in ("mmlu", "gsm8k"):
            per = {}
            for cond, suffix in COND:
                d = RUN / slug / f"{task}_hineng{suffix}"
                m = json.loads((d / "metrics.json").read_text())
                rows = live_rows(d)
                acc = m["headline_value"]
                # GSM8K CM is the one split whose item set differs from its siblings;
                # rescore it on the paired subset so the column is comparable.
                if task == "gsm8k" and cond == "CM":
                    import csv as _csv
                    # Prefer slim, the compact audit trail; fall back to the fat
                    # original for a working tree that still has it.
                    pfile = d / "predictions.slim.csv"
                    if not pfile.exists():
                        pfile = d / "predictions.csv"
                        _csv.field_size_limit(1 << 30)
                    with pfile.open(newline="") as fh:
                        pr = [r for r in _csv.DictReader(fh)
                              if int(r["index"]) in cm_paired]
                    acc = 100 * sum(r["correct"] == "True" for r in pr) / len(pr)
                    rows = [r for r in rows if r["index"] in cm_paired]
                n = len(rows)
                trunc = sum(1 for r in rows if r.get("finish_reason") == "length")
                budgets = sorted({r.get("max_tokens") for r in rows if r.get("max_tokens")})
                per[cond] = {
                    # 4dp, not 2: consumers display 1dp, and rounding twice shifts
                    # values whose 3rd decimal straddles a boundary (64.746 -> 64.75
                    # -> "64.8", but correctly "64.7"). Same reason as trunc_pct.
                    "accuracy": round(acc, 4),
                    "_acc_exact": acc,
                    "n": m.get("n", n),
                    "unparsed": round(m["diagnostics"]["unparsed_rate"], 4),
                    # 4dp, not 2: consumers format to 1dp, and rounding twice
                    # shifts values whose 3rd decimal straddles a boundary
                    # (11/955 = 1.1518 -> 1.15 -> "1.1", but correctly "1.2").
                    "trunc_pct": round(100 * trunc / n, 4),
                    "trunc_n": trunc,
                    "rows": n,
                    "max_budget": budgets[-1] if budgets else None,
                }
            en = per["EN"]["_acc_exact"]
            fl = FLOOR[task]
            for cond, _ in COND:
                a = per[cond]["_acc_exact"]
                per[cond]["retention_raw"] = round(100 * a / en, 4) if en else None
                head = en - fl
                per[cond]["retention_adj"] = round(100 * (a - fl) / head, 4) if head > 0 else None
                per[cond]["delta_en"] = round(a - en, 4)
            for cond, _ in COND:          # drop the working field
                per[cond].pop("_acc_exact", None)
            rec["tasks"][task] = per
        out["models"][slug] = rec
    Path("results/tax_dataset.json").write_text(json.dumps(out, indent=2))
    print(f"wrote results/tax_dataset.json  ({len(out['models'])} models)")

if __name__ == "__main__":
    main()
