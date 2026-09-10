#!/usr/bin/env python
"""Build the romanised TruthfulQA condition: truthfulqa_hineng -> _rom.

A script-only manipulation. The question and the four options keep their wording,
their order and their gold letter; only Devanagari is replaced by Latin, using the
same ITRANS + schwa-deletion pipeline as the MMLU and GSM8K romanised conditions.

Why this task gets only the romanised condition and not the full five: the English
and Hindi originals exist, but ``truthfulqa_hineng`` has no ``id`` column and the
benchmark authors sampled and shuffled four of TruthfulQA's variable-length mc1
choices per question. Recovering which distractors were used needs fuzzy matching,
and every route tried caps near 50% recall -- the code-mixed Hindi and the Hindi of
``alexandrainst/m_truthfulqa`` are independent translations, so real matches score
only 0.8-0.9 on string similarity. Romanisation needs no join, so the pairing here
is exact rather than inferred.

    python scripts/build_truthfulqa_rom.py
"""
from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import pandas as pd

#: Option lines look like "(A): text". The marker must survive untouched -- it is
#: what the extractor anchors the predicted letter to. [A-Z], not [A-D]: TruthfulQA
#: keeps every mc1 choice, so a question can carry up to 13 options.
_OPT = re.compile(r"^\(([A-Z])\):\s*")


def _load_transliterator():
    """Reuse the transliteration pipeline from the MMLU/GSM8K condition builder.

    Imported by path rather than copied: five separate bugs were found and fixed in
    that pipeline (schwa/lowercase ordering, candra vowels, nukta marks, ITRANS
    capital digraphs, danda), and a second copy would drift away from those fixes.
    """
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("build_conditions", root / "build_conditions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.to_hinglish


def romanise(sentence: str, to_rom) -> str:
    """Transliterate a question-plus-options block, preserving option markers."""
    out = []
    for line in str(sentence).split("\n"):
        m = _OPT.match(line)
        if m:
            out.append(f"({m.group(1)}): {to_rom(line[m.end():].strip())}")
        else:
            out.append(to_rom(line))
    return "\n".join(out)


_DEV = re.compile(r"[ऀ-ॿ]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/raw/truthfulqa/truthfulqa_hineng.csv")
    ap.add_argument("--out", default="data/raw/truthfulqa/truthfulqa_hineng_rom.csv")
    ap.add_argument("--log", default="logs/build_truthfulqa_rom.log")
    args = ap.parse_args()

    lines: list[str] = []

    def out(*p):
        s = " ".join(str(x) for x in p)
        print(s)
        lines.append(s)

    to_rom = _load_transliterator()
    df = pd.read_csv(args.src, dtype=str)
    out(f"source: {args.src}  rows={len(df)}")

    rom = df.copy()
    rom["sentence"] = [romanise(s, to_rom) for s in df["sentence"]]

    # Every invariant that makes this a paired condition, checked rather than assumed.
    bad_opts = bad_gold = still_dev = 0
    for (_, a), (_, b) in zip(df.iterrows(), rom.iterrows()):
        if _OPT.findall(str(a["sentence"])) != _OPT.findall(str(b["sentence"])):
            bad_opts += 1
        if str(a["answer"]).strip() != str(b["answer"]).strip():
            bad_gold += 1
        if _DEV.search(str(b["sentence"])):
            still_dev += 1
    out(f"option markers preserved : {len(df) - bad_opts}/{len(df)}")
    out(f"gold letters unchanged   : {len(df) - bad_gold}/{len(df)}")
    out(f"rows with Devanagari left: {still_dev}")
    if bad_opts or bad_gold:
        out("ABORT: the pairing invariants failed; not writing output.")
        Path(args.log).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 1

    # How much of each row actually changed -- a romanised condition that barely
    # differs from its source would make the comparison meaningless.
    changed = sum(1 for a, b in zip(df["sentence"], rom["sentence"]) if str(a) != str(b))
    out(f"rows whose text changed  : {changed}/{len(df)} ({100*changed/len(df):.1f}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rom.to_csv(args.out, index=False)
    out(f"wrote {args.out}")

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
