#!/usr/bin/env python
"""Verify that every mmlu_hineng item exists in cais/mmlu AND Global-MMLU (hi).

The paired code-mixing-tax design only works if the SAME question is available in
every language form. This checks that precondition before any GPU time is spent,
across three sources:

  CM   CodeMixBench mmlu_hineng          (code-mixed Hindi-English)  -- the anchor
  EN   cais/mmlu                          (English original)
  HI   CohereLabs/Global-MMLU config hi   (professionally translated Hindi)

For each source it reports: id coverage, gold-answer agreement with CM, and -- for
HI -- the fraction of items Global-MMLU marks as human-annotated (`is_annotated`),
since the remainder is machine translation with lighter review.

Exit code is non-zero if any source fails to cover all CM items, so this can gate
a build step.

    python scripts/verify_item_coverage.py
    python scripts/verify_item_coverage.py --log logs/item_coverage.log
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmb_indic.data import load_split  # noqa: E402
from cmb_indic.registry import DATASETS  # noqa: E402

LETTERS = "ABCD"
DEV = re.compile(r"[ऀ-ॿ]")


class Tee:
    """Write to stdout and a log file at once."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")  # noqa: SIM115 -- closed in close()

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="logs/item_coverage.log")
    ap.add_argument("--samples", type=int, default=5,
                    help="how many items to print in full, side by side, for manual verification")
    ap.add_argument("--revision", default=None,
                     help="pin a Hub revision for cais/mmlu and Global-MMLU")
    args = ap.parse_args()
    out = Tee(Path(args.log))

    import datetime

    out("=" * 78)
    out("ITEM COVERAGE VERIFICATION")
    out("generated_utc :", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    out("=" * 78)

    # ---- anchor: CodeMixBench --------------------------------------------- #
    cm = load_split(DATASETS["mmlu_hineng"], "data/raw", download=False).frame
    cm_ids = [str(x) for x in cm["id"]]
    cm_gold = {i: str(a).strip() for i, a in zip(cm_ids, cm["answer"], strict=True)}
    subjects = sorted({i.split("/")[0] for i in cm_ids})
    out(f"\n[CM] CodeMixBench mmlu_hineng: {len(cm)} rows, {len(set(cm_ids))} unique ids, "
        f"{len(subjects)} subjects")
    out(f"     id format example: {cm_ids[0]}")
    out(f"     splits referenced: {sorted({i.split('/')[1] for i in cm_ids})}")

    from datasets import load_dataset

    # ---- EN: cais/mmlu ----------------------------------------------------- #
    out("\n" + "-" * 78)
    out("[EN] cais/mmlu  (English original)")
    out("-" * 78)
    en_found, en_missing, en_gold_ok, en_gold_bad = set(), [], 0, []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test", revision=args.revision)
        n = len(ds)
        for i in cm_ids:
            s, _sp, row = i.split("/")
            if s != subj:
                continue
            r = int(row)
            if r >= n:
                en_missing.append((i, f"row {r} >= test size {n}"))
                continue
            en_found.add(i)
            g = LETTERS[int(ds[r]["answer"])]
            if g == cm_gold[i]:
                en_gold_ok += 1
            else:
                en_gold_bad.append((i, g, cm_gold[i]))
    out(f"  ids covered            : {len(en_found)} / {len(cm_ids)}")
    out(f"  ids missing            : {len(en_missing)}")
    for i, why in en_missing[:5]:
        out(f"      {i}: {why}")
    out(f"  gold answers agree     : {en_gold_ok} / {len(en_found)}")
    for i, g, c in en_gold_bad[:5]:
        out(f"      MISMATCH {i}: cais={g} codemixbench={c}")

    # ---- HI: Global-MMLU --------------------------------------------------- #
    out("\n" + "-" * 78)
    out("[HI] CohereLabs/Global-MMLU config 'hi'  (professional Hindi translation)")
    out("-" * 78)
    gm = load_dataset("CohereLabs/Global-MMLU", "hi", split="test", revision=args.revision)
    gm_by_id = {}
    for r in gm:
        gm_by_id[str(r["sample_id"])] = r
    out(f"  Global-MMLU hi rows    : {len(gm)}   unique sample_ids: {len(gm_by_id)}")

    hi_found = [i for i in cm_ids if i in gm_by_id]
    hi_missing = [i for i in cm_ids if i not in gm_by_id]
    out(f"  ids covered            : {len(hi_found)} / {len(cm_ids)}")
    out(f"  ids missing            : {len(hi_missing)}")
    for i in hi_missing[:5]:
        out(f"      {i}")
    hi_gold_ok = sum(1 for i in hi_found if str(gm_by_id[i]["answer"]).strip() == cm_gold[i])
    out(f"  gold answers agree     : {hi_gold_ok} / {len(hi_found)}")
    bad = [(i, gm_by_id[i]["answer"], cm_gold[i]) for i in hi_found
           if str(gm_by_id[i]["answer"]).strip() != cm_gold[i]]
    for i, g, c in bad[:5]:
        out(f"      MISMATCH {i}: global-mmlu={g} codemixbench={c}")

    ann = Counter(bool(gm_by_id[i].get("is_annotated")) for i in hi_found)
    out(f"  human-annotated        : {ann.get(True, 0)} / {len(hi_found)} "
        f"({100 * ann.get(True, 0) / max(1, len(hi_found)):.1f}%)  "
        f"[remainder is MT with lighter review]")

    # script composition: is the HI text actually Devanagari, and are English
    # technical terms calqued or kept as loanwords?
    dev_chars = lat_chars = 0
    for i in hi_found[:400]:
        q = str(gm_by_id[i]["question"])
        dev_chars += sum(1 for c in q if DEV.match(c))
        lat_chars += sum(1 for c in q if c.isascii() and c.isalpha())
    tot = dev_chars + lat_chars
    out(f"  script mix (400 items) : {100 * dev_chars / max(1, tot):.1f}% Devanagari, "
        f"{100 * lat_chars / max(1, tot):.1f}% Latin")
    out("     -> residual Latin is loanwords/symbols kept untranslated, which is")
    out("        normal for Hindi technical prose. HI is therefore 'native script',")
    out("        not strictly 'zero English'.")

    # ---- three-way intersection -------------------------------------------- #
    out("\n" + "=" * 78)
    tri = [i for i in cm_ids if i in en_found and i in gm_by_id]
    out(f"THREE-WAY USABLE SET (CM & EN & HI): {len(tri)} / {len(cm_ids)} items")
    out("=" * 78)
    if tri:
        by_subj = Counter(i.split("/")[0] for i in tri)
        out(f"  subjects: {len(by_subj)}   top: {by_subj.most_common(5)}")

        # ---- full side-by-side samples, for manual verification ------------ #
        # Spread the samples across different subjects rather than taking the
        # first N, which would all come from one subject and prove less.
        seen_subj: set[str] = set()
        picks: list[str] = []
        for i in tri:
            sj = i.split("/")[0]
            if sj not in seen_subj:
                seen_subj.add(sj)
                picks.append(i)
            if len(picks) >= args.samples:
                break

        # cais/mmlu rows, fetched once per needed subject
        en_rows = {}
        for sj in {i.split("/")[0] for i in picks}:
            en_rows[sj] = load_dataset("cais/mmlu", sj, split="test", revision=args.revision)

        out("\n" + "=" * 78)
        out(f"SIDE-BY-SIDE SAMPLES ({len(picks)} items, one per subject)")
        out("=" * 78)
        for n, i in enumerate(picks, 1):
            sj, _sp, row = i.split("/")
            e = en_rows[sj][int(row)]
            h = gm_by_id[i]
            c = cm[cm["id"] == i].iloc[0]
            out(f"\n--- [{n}] {i} ---")
            out("  EN  Q: " + str(e["question"]).replace("\n", " "))
            for k, ch in enumerate(e["choices"]):
                out(f"        ({LETTERS[k]}) {str(ch).replace(chr(10), ' ')}")
            out(f"      gold: {LETTERS[int(e['answer'])]}")
            out("  HI  Q: " + str(h["question"]).replace("\n", " "))
            for k in range(4):
                out(f"        ({LETTERS[k]}) {str(h['option_' + LETTERS[k].lower()]).replace(chr(10), ' ')}")
            out(f"      gold: {h['answer']}   annotated: {bool(h.get('is_annotated'))}")
            cm_lines = str(c["sentence"]).split("\n")
            out("  CM  Q: " + cm_lines[0])
            for ln in cm_lines[1:]:
                out(f"        {ln}")
            out(f"      gold: {cm_gold[i]}")
            out(f"  >>> gold agreement: EN={LETTERS[int(e['answer'])]} "
                f"HI={h['answer']} CM={cm_gold[i]}  -> "
                f"{'ALL MATCH' if LETTERS[int(e['answer'])] == str(h['answer']).strip() == cm_gold[i] else 'MISMATCH'}")

    ok = (len(en_found) == len(cm_ids) and not en_gold_bad
          and len(hi_found) == len(cm_ids) and not bad)
    out("\nVERDICT:", "PASS -- all sources cover all items with matching gold"
        if ok else "PARTIAL -- see counts above; paired design must use the intersection")
    out(f"log written to: {args.log}")
    out.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
