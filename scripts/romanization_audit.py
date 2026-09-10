#!/usr/bin/env python
"""Audit the romanization used to build the two Latin-script conditions.

The romanized conditions are produced by one deterministic transform
(scripts/build_conditions.py: indic_transliteration ITRANS, then final-schwa
deletion, anusvara -> n, danda -> full stop, lowercasing). Romanized Hindi has
no standard orthography, so any single scheme is one point in a wide space of
things real people type -- which makes the transform itself a threat to validity
and not merely an implementation detail.

Two things are measured, neither needing a human reference:

  known_divergence  build_conditions.to_hinglish documents that INTERNAL schwa
                    deletion is not modelled, so it emits "men" where a person
                    types "mein" and "jisake" for "jiske". This counts how often
                    those known-wrong forms actually appear. A high rate means
                    the conditions are systematically a little unlike natural
                    typing, in a direction that most likely OVERSTATES the
                    romanization penalty, since the text is further out of
                    distribution than a real user's would be.

  sample            a fixed, seeded sample of rows written out with the
                    Devanagari source beside the romanized output, so a Hindi
                    speaker can check the transform by eye. This script cannot
                    validate against human typing on its own; it prepares the
                    artifact for someone who can.

Needs data/raw, which `make data` populates. Both this script's outputs land in
results/, which is gitignored -- this repository ships code, not results -- so a
clone regenerates them rather than receiving them.

    python scripts/romanization_audit.py [--sample 50]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import pandas as pd

OUT = Path("results/romanization_audit.json")
SAMPLE_OUT = Path("results/romanization_sample.csv")

#: (emitted, what a native writer types). High-frequency function words only,
#: all instances of the one documented defect: unmodelled internal schwa.
KNOWN = [
    ("men", "mein"), ("jisake", "jiske"), ("usake", "uske"), ("kisake", "kiske"),
    ("isake", "iske"), ("unake", "unke"), ("inake", "inke"), ("jisaka", "jiska"),
    ("usaka", "uska"), ("kisaka", "kiska"), ("isaka", "iska"), ("jisamen", "jismein"),
    ("usamen", "usmein"), ("kitane", "kitne"), ("jitane", "jitne"), ("utane", "utne"),
    ("apane", "apne"), ("sabase", "sabse"), ("saphed", "safed"),
]

#: romanized condition -> the Devanagari condition it was derived from.
PAIRS = {
    "mmlu_hineng_rom":    ("mmlu/mmlu_hineng_rom.csv",       "mmlu/mmlu_hineng.csv"),
    "mmlu_hineng_hirom":  ("mmlu/mmlu_hineng_hirom.csv",     "mmlu/mmlu_hineng_hi.csv"),
    "gsm8k_hineng_rom":   ("gsm8k/gsm8k_hineng_rom.csv",     "gsm8k/gsm8k_hineng.csv"),
    "gsm8k_hineng_hirom": ("gsm8k/gsm8k_hineng_hirom.csv",   "gsm8k/gsm8k_hineng_hi.csv"),
}
RAW = Path("data/raw")
TEXT_COLS = ("question", "sentence", "text", "prompt")

#: The version that ACTUALLY produced the shipped conditions. Recorded as a
#: constant rather than read from the running interpreter, because this audit is
#: run from whichever env is to hand and the transliterator is only installed in
#: the one that built the data -- letting it report "unknown" would quietly
#: destroy the provenance it exists to record. Checked against the environment
#: below when the package happens to be importable.
BUILT_WITH = "2.3.82"


def text_of(df: pd.DataFrame) -> pd.Series:
    for c in TEXT_COLS:
        if c in df.columns:
            return df[c].astype(str)
    return df.iloc[:, 1].astype(str)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out: dict = {
        "tool": "indic_transliteration",
        "scheme": "DEVANAGARI -> ITRANS",
        "postprocess": ["final-schwa deletion", "anusvara/candrabindu -> n",
                        "visarga -> h", "danda -> full stop", "lowercasing",
                        "nukta stripped", "repeated-vowel collapse"],
        "implementation": "scripts/build_conditions.py:to_hinglish",
        "known_defect": "internal schwa deletion is not modelled",
        "human_reference": None,
        "conditions": {},
    }
    out["tool_version"] = BUILT_WITH
    try:
        import importlib.metadata as md
        live = md.version("indic_transliteration")
    except Exception:  # noqa: BLE001
        live = None
    out["tool_version_verified_here"] = live == BUILT_WITH
    if live and live != BUILT_WITH:
        print(f"  warning: this environment has {live}, the conditions were built "
              f"with {BUILT_WITH}; the recorded provenance is {BUILT_WITH}")

    sample_rows: list[dict] = []
    for name, (rom_p, dev_p) in PAIRS.items():
        rom = pd.read_csv(RAW / rom_p)
        rt = text_of(rom)
        toks = re.findall(r"[a-z]+", " ".join(rt).lower())
        tc = collections.Counter(toks)
        hits = {emit: tc[emit] for emit, _ in KNOWN if tc[emit]}
        pat = re.compile(r"\b(" + "|".join(e for e, _ in KNOWN) + r")\b")
        n_rows = int(sum(1 for t in rt if pat.search(t.lower())))
        out["conditions"][name] = {
            "rows": len(rom),
            "tokens": len(toks),
            "rows_with_divergent_form": n_rows,
            "rows_pct": round(100 * n_rows / len(rom), 2),
            "divergent_tokens": int(sum(hits.values())),
            "token_pct": round(100 * sum(hits.values()) / len(toks), 2),
            "top_forms": [{"emitted": k, "native": dict(KNOWN)[k], "count": int(v)}
                          for k, v in sorted(hits.items(), key=lambda x: -x[1])[:8]],
        }

        # Side-by-side sample for human review.
        dev_file = RAW / dev_p
        if not dev_file.exists():
            continue
        dev = pd.read_csv(dev_file)
        k = min(args.sample // len(PAIRS) + 1, len(rom), len(dev))
        idx = rom.sample(n=k, random_state=args.seed).index
        dt = text_of(dev)
        for i in idx:
            if i < len(dt):
                sample_rows.append({"condition": name, "index": int(i),
                                    "devanagari": dt.iloc[i], "romanized": rt.iloc[i]})

    out["sample_rows"] = len(sample_rows)
    out["sample_file"] = str(SAMPLE_OUT)
    OUT.write_text(json.dumps(out, indent=2))
    if sample_rows:
        pd.DataFrame(sample_rows).to_csv(SAMPLE_OUT, index=False)

    print(f"{out['tool']} {out['tool_version']}, {out['scheme']}")
    print(f"post-processing: {', '.join(out['postprocess'])}")
    print(f"known defect: {out['known_defect']}\n")
    print("How often the known-wrong forms appear:")
    for name, c in out["conditions"].items():
        print(f"  {name:20s} {c['rows_with_divergent_form']:>4}/{c['rows']} rows "
              f"({c['rows_pct']:5.1f}%)  {c['token_pct']:.2f}% of tokens")
        top = ", ".join("{}->{} x{}".format(f["emitted"], f["native"], f["count"])
                        for f in c["top_forms"][:3])
        print(f"     {top}")
    print(f"\nwrote {OUT}")
    if sample_rows:
        print(f"wrote {SAMPLE_OUT} ({len(sample_rows)} rows for human review)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
