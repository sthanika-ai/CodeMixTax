#!/usr/bin/env python
"""Is CM-GSM8K contaminated by GSM8K train-split items?

The CodeMixBench paper describes CM-GSM8K as derived from the GSM8K *test* set.
Fingerprinting each item's `cot` field against openai/gsm8k shows otherwise: a
third of the 1,016 items come from `train`, which is standard instruction-tuning
data for essentially every modern instruct model. If models score materially
higher on the train-derived portion, the reported GSM8K accuracy is partly
measuring memorisation.

This runs on cached generations only -- no GPU, no model.

Confounds are checked rather than assumed: train-derived items could simply be
easier, so problem difficulty proxies (reasoning steps, answer magnitude, question
length) are compared between the two subsets.

    python scripts/check_gsm8k_contamination.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmb_indic.data import load_split  # noqa: E402
from cmb_indic.registry import DATASETS  # noqa: E402

MODELS = [
    ("4B", "runs/gemma3-4b-full/gemma3-4b-it"),
    ("12B", "runs/gemma3-scale/gemma3-12b-it"),
    ("12B-INT4", "runs/gemma3-scale/gemma3-12b-it-int4"),
    ("27B", "runs/gemma3-scale/gemma3-27b-it"),
]
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")  # noqa: SIM115 -- closed in close()

    def __call__(self, *p):
        line = " ".join(str(x) for x in p)
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def norm_cot(t: str) -> str:
    t = re.sub(r"####.*$", "", str(t), flags=re.S)
    return re.sub(r"\s+", " ", t).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="logs/gsm8k_contamination.log")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--revision", default=None, help="pin a Hub revision for openai/gsm8k")
    args = ap.parse_args()
    out = Tee(Path(args.log))
    rng = np.random.default_rng(1234)

    import datetime

    out("=" * 78)
    out("GSM8K CONTAMINATION CHECK")
    out("generated_utc:", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    out("=" * 78)

    from datasets import load_dataset

    idx: dict[str, list[tuple[str, int, dict]]] = {}
    for split in ("train", "test"):
        g = load_dataset("openai/gsm8k", "main", split=split, revision=args.revision)
        for i, r in enumerate(g):
            idx.setdefault(norm_cot(r["answer"]), []).append((split, i, r))

    f = load_split(DATASETS["gsm8k_hineng"], "data/raw", download=False).frame
    rows = []
    for _, r in f.iterrows():
        hit = idx.get(norm_cot(r["cot"]))
        if not hit:
            continue
        split, i, orig = hit[0]
        steps = str(r["cot"]).count("<<")
        rows.append({
            "index": int(r["index"]), "src_split": split, "src_row": i,
            "steps": steps,
            "answer_mag": abs(float(str(r["answer"]).replace(",", "") or 0)),
            "q_chars": len(str(r["sentence"])),
        })
    prov = pd.DataFrame(rows)
    out(f"\nprovenance of the {len(f)} CM-GSM8K items")
    out(f"  matched to openai/gsm8k : {len(prov)}")
    for s, n in prov["src_split"].value_counts().items():
        out(f"    from {s:<6}: {n:>5}  ({100*n/len(prov):.1f}%)")

    # ---- difficulty confound ------------------------------------------------ #
    out("\n" + "-" * 78)
    out("CONFOUND CHECK: are train-derived items intrinsically easier?")
    out("-" * 78)
    out(f"{'metric':<22}{'train':>12}{'test':>12}{'difference':>13}")
    for col, label in [("steps", "reasoning steps"), ("answer_mag", "answer magnitude"),
                       ("q_chars", "question chars")]:
        a = prov[prov.src_split == "train"][col]
        b = prov[prov.src_split == "test"][col]
        out(f"{label:<22}{a.mean():>12.2f}{b.mean():>12.2f}{a.mean()-b.mean():>+13.2f}")

    # ---- accuracy by provenance -------------------------------------------- #
    out("\n" + "=" * 78)
    out("ACCURACY BY SOURCE SPLIT (cached generations, no GPU)")
    out("=" * 78)
    out(f"{'model':<10}{'train acc':>11}{'test acc':>10}{'gap':>8}{'95% CI (bootstrap)':>24}{'n tr/te':>12}")
    out("-" * 76)
    for lab, run in MODELS:
        p = Path(run) / "gsm8k_hineng" / "predictions.csv"
        if not p.exists():
            out(f"{lab:<10}(no predictions)")
            continue
        pr = pd.read_csv(p, dtype=str)
        pr["index"] = pr["index"].astype(int)
        pr["ok"] = pr["correct"].str.lower().eq("true").astype(float)
        d = prov.merge(pr[["index", "ok"]], on="index")
        tr = d[d.src_split == "train"]["ok"].to_numpy()
        te = d[d.src_split == "test"]["ok"].to_numpy()
        gap = (tr.mean() - te.mean()) * 100
        bs = np.empty(args.boot)
        for i in range(args.boot):
            bs[i] = (rng.choice(tr, len(tr), True).mean()
                     - rng.choice(te, len(te), True).mean()) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = " *" if not (lo < 0 < hi) else ""
        out(f"{lab:<10}{tr.mean()*100:>11.2f}{te.mean()*100:>10.2f}{gap:>+8.2f}"
            f"{f'[{lo:+.2f}, {hi:+.2f}]':>24}{f'{len(tr)}/{len(te)}':>12}{sig}")

    out("\n  * = 95% CI excludes zero.")
    out("  A large positive gap would indicate memorisation of GSM8K train items.")
    out(f"\nlog written to: {args.log}")
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
