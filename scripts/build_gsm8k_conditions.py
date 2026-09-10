#!/usr/bin/env python
"""Build the paired language-form conditions for GSM8K.

GSM8K has no `id` column, so the pairing is a two-hop join, each hop validated:

  1. gsm8k_hineng.cot is verbatim English GSM8K reasoning (calculator annotations
     included), which fingerprints each item to (split, row) in openai/gsm8k.
     Verified: 1016/1016 matched, 0 ambiguous, 1015/1016 answer agreement.
  2. bingbangboom/gsm8k-hindi preserves openai/gsm8k row order (split sizes are
     identical: 7473 train / 1319 test). Verified by question number sets.

Items whose Hindi question does NOT carry the same number set as the English are
DROPPED: a garbled quantity makes it a different problem wearing the English gold
answer, which would penalise the Hindi condition for a data bug.

Gold answers always come from openai/gsm8k -- never from the Hindi dataset's
answer field, which is malformed (`###` instead of `####`, `80, 000`) in ~27% of rows.

    python scripts/build_gsm8k_conditions.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_conditions import DEV, to_hinglish  # noqa: E402  -- shared transliterator

from cmb_indic.data import load_split  # noqa: E402
from cmb_indic.registry import DATASETS, DERIVED  # noqa: E402

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
HI_SPLIT = {"train": "train_main", "test": "test_main"}


def norm_cot(t: str) -> str:
    t = re.sub(r"####.*$", "", str(t), flags=re.S)
    return re.sub(r"\s+", " ", t).strip().lower()


def numset(t: str) -> set[str]:
    return {x.replace(",", "") for x in NUM.findall(str(t))}


def final_answer(a: str) -> str:
    m = re.search(r"####\s*([-\d,\.]+)", str(a))
    return m.group(1).replace(",", "").strip(".") if m else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/gsm8k")
    ap.add_argument("--revision", default=None,
                     help="pin a Hub revision for openai/gsm8k and bingbangboom/gsm8k-hindi")
    args = ap.parse_args()

    from datasets import load_dataset

    cm = load_split(DATASETS["gsm8k_hineng"], "data/raw", download=False).frame
    print(f"source: gsm8k_hineng, {len(cm)} items")

    # hop 1: CoT fingerprint -> openai/gsm8k
    en_ds = {
        s: load_dataset("openai/gsm8k", "main", split=s, revision=args.revision)
        for s in ("train", "test")
    }
    fp: dict[str, tuple[str, int]] = {}
    for s, ds in en_ds.items():
        for i, r in enumerate(ds):
            fp.setdefault(norm_cot(r["answer"]), (s, i))

    # hop 2: positional -> bingbangboom/gsm8k-hindi
    hi_ds = load_dataset("bingbangboom/gsm8k-hindi", revision=args.revision)

    kept, dropped_nomatch, dropped_numset = [], 0, 0
    for _, r in cm.iterrows():
        hit = fp.get(norm_cot(r["cot"]))
        if hit is None:
            dropped_nomatch += 1
            continue
        split, row = hit
        e = en_ds[split][row]
        h = hi_ds[HI_SPLIT[split]][row]
        if numset(e["question"]) != numset(h["question"]):
            dropped_numset += 1
            continue
        kept.append({
            "index": int(r["index"]), "src_split": split, "src_row": row,
            "cm": str(r["sentence"]), "en": str(e["question"]), "hi": str(h["question"]),
            "cot": str(e["answer"]).split("####")[0].strip(),
            "answer": final_answer(e["answer"]),
        })

    k = pd.DataFrame(kept)
    print("\njoin results")
    print(f"  matched via CoT fingerprint : {len(cm) - dropped_nomatch} / {len(cm)}")
    print(f"  dropped, number-set mismatch: {dropped_numset}")
    print(f"  KEPT                        : {len(k)}")
    print(f"    from train: {(k.src_split == 'train').sum()}   from test: {(k.src_split == 'test').sum()}")

    # gold sanity: the English answer must equal CodeMixBench's own answer
    cm_ans = dict(zip([int(i) for i in cm["index"]],
                      [str(a).replace(",", "").strip() for a in cm["answer"]], strict=True))
    bad = [int(r["index"]) for _, r in k.iterrows() if cm_ans[int(r["index"])] != r["answer"]]
    print(f"  gold disagreements with gsm8k_hineng: {len(bad)}")
    if len(bad) > len(k) * 0.01:
        raise SystemExit(f"ABORT: too many gold disagreements ({len(bad)})")
    if bad:
        print(f"    dropping {len(bad)} disagreeing item(s): {bad[:5]}")
        k = k[~k["index"].isin(bad)].reset_index(drop=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = {
        "gsm8k_hineng_en": k["en"],
        "gsm8k_hineng_hi": k["hi"],
        "gsm8k_hineng_rom": [to_hinglish(s) for s in k["cm"]],
        "gsm8k_hineng_hirom": [to_hinglish(s) for s in k["hi"]],
    }
    print()
    for name, sentences in frames.items():
        df = pd.DataFrame({
            "index": k["index"], "sentence": list(sentences), "answer": k["answer"],
            "cot": k["cot"], "src_split": k["src_split"], "src_row": k["src_row"],
        })
        left = sum(1 for s in df["sentence"] if DEV.search(str(s)))
        spec = DERIVED[name]
        if len(df) != spec.n_rows:
            print(f"  NOTE {name}: {len(df)} rows, registry says {spec.n_rows} "
                  f"-- update _GSM8K_DERIVED_ROWS")
        p = out / f"{name}.csv"
        df.to_csv(p, index=False, encoding="utf-8")
        print(f"  wrote {p}  ({len(df)} rows, leftover-Devanagari rows: {left})")

    # the CM condition must be scored on the SAME subset; record the index list
    idxfile = out / "gsm8k_hineng_paired_index.txt"
    idxfile.write_text("\n".join(str(i) for i in k["index"]), encoding="utf-8")
    print(f"  wrote {idxfile}  ({len(k)} indices -- restrict CM to these when analysing)")

    print("\nsanity: same item, five conditions")
    r = k.iloc[0]
    for tag, txt in [("EN    ", r["en"]), ("HI    ", r["hi"]),
                     ("HI-ROM", to_hinglish(r["hi"])), ("CM    ", r["cm"]),
                     ("CM-ROM", to_hinglish(r["cm"]))]:
        print(f"  {tag}:", str(txt).replace("\n", " ")[:96])
    print(f"  gold: {r['answer']}   (src {r['src_split']}/{r['src_row']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
