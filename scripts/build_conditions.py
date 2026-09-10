#!/usr/bin/env python
"""Build the paired language-form conditions for the code-mixing tax study.

From `mmlu_hineng` (code-mixed Hindi-English, 1,024 items) construct two paired
conditions over the SAME items, so question difficulty is held constant and only
language form varies:

  EN   English originals, recovered from cais/mmlu via the `id` column
       (subject/test/row). Exact source text -- no synthesis. Gold answers are
       verified to agree with mmlu_hineng; any disagreement aborts the build.

  ROM  mmlu_hineng with Devanagari transliterated to informal romanized Hindi.
       Word-for-word faithful: a script transform, not a translation, so a
       ROM-vs-CM difference isolates script from vocabulary/mixing.

Deliberately NOT built: a pure-Hindi (native script, no English) condition. That
requires machine-translating the English tokens, and any degradation would then
confound "native script is harder" with "our MT was mediocre".

    python scripts/build_conditions.py
    python scripts/build_conditions.py --out data/raw/mmlu
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmb_indic.data import load_split  # noqa: E402
from cmb_indic.registry import DATASETS, DERIVED  # noqa: E402

LETTERS = "ABCD"
DEV = re.compile(r"[ऀ-ॿ]")

#: "Candra" vowels are used in Hindi to spell English loanword sounds -- मॉडल
#: (model), न्यूरॉन (neuron), एपिग्लॉटिस (epiglottis). ITRANS has no mapping for
#: them, so without this they survive transliteration and leave Devanagari inside
#: a supposedly romanized sentence (121 of 1,024 rows before this fix). Normalise
#: each to its nearest standard vowel so the transliterator can handle it.
#: Nukta letters (base + U+093C) that ITRANS leaves untransliterated, e.g. ढ़ (RHA).
#: Decomposing and dropping the nukta maps them to their base consonant --
#: ज़ -> ज (z -> j), ड़ -> ड, फ़ -> फ (f -> ph) -- which is how these are commonly
#: typed informally. Slight phonetic loss, affects a handful of tokens.
NUKTA = "\u093c"

CANDRA = {
    "\u0911": "\u0913",  # ऑ -> ओ   candra O      -> O
    "\u0949": "\u094b",  # ॉ -> ो   candra O sign -> o sign
    "\u090d": "\u090f",  # ऍ -> ए   candra E      -> E
    "\u0945": "\u0947",  # ॅ -> े   candra E sign -> e sign
}


# --------------------------------------------------------------------------- #
# ROM: Devanagari -> informal romanized Hindi
# --------------------------------------------------------------------------- #
def to_hinglish(text: str) -> str:
    """Transliterate only the Devanagari tokens; leave English/punctuation alone.

    ITRANS is used as the base scheme because it is lossless and ASCII, then
    normalised toward how Hindi is actually typed in Latin script:

      * schwa deletion -- word-final short 'a' is dropped (hotA stays "hota",
        but "eka" -> "ek"). Done BEFORE lowercasing, because ITRANS distinguishes
        short 'a' from long 'A' by case and lowercasing destroys that.
      * anusvara 'M' -> 'n'  (darshakoM -> darshakon)
      * retroflex/long-vowel capitals -> lowercase

    Known limitation: internal schwa deletion is not modelled, so `jisake`
    appears where a native writer would type `jiske`, and `men` for `mein`.
    Word-initial and final forms are correct; this affects readability slightly,
    not identity of content.
    """
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    out: list[str] = []
    for tok in re.split(r"(\s+)", text):
        if not DEV.search(tok):
            out.append(tok)
            continue
        for src, dst in CANDRA.items():
            tok = tok.replace(src, dst)
        # Strip nukta via NFD so precomposed forms (ढ़, ज़, ड़) decompose first.
        tok = unicodedata.normalize("NFC",
                                    unicodedata.normalize("NFD", tok).replace(NUKTA, ""))
        r = transliterate(tok, sanscript.DEVANAGARI, sanscript.ITRANS)
        core = r.rstrip(".,;:!?()\"'।")
        tail = r[len(core):]
        if len(core) >= 3 and core.endswith("a") and core[-2] not in "aAeEiIoOuU":
            core = core[:-1]
        r = core + tail
        # ITRANS dotted digraphs: ".n"/".m" = candrabindu/anusvara, ".h" = visarga.
        # Without this, पहुँचाने -> "pahu.nchane" instead of "pahunchane" (163 rows
        # in the Hindi condition, 78 in the code-mixed one).
        # Capitals matter: ITRANS emits ".N" for candrabindu, and this runs BEFORE
        # .lower() (which must stay after schwa deletion), so both cases are handled.
        for dot, repl in ((".N", "n"), (".n", "n"), (".M", "n"), (".m", "n"),
                         (".H", "h"), (".h", "h")):
            r = r.replace(dot, repl)
        r = (r.replace("~N", "n").replace("~n", "n").replace("N^", "n")
              .replace("M", "n").replace("Sh", "sh").replace("S", "sh")
              .replace("R^i", "ri"))
        r = r.lower()
        r = re.sub(r"([aeiou])\1+", r"\1", r)
        # ITRANS renders the Devanagari danda (।) and double danda (॥) as "|" / "||".
        # Left as-is that puts a stray pipe mid-sentence in 51-87% of rows, which is
        # not how anyone writes romanized Hindi. Map them to the sentence-final period
        # a native writer would actually type.
        r = r.replace("||", ".").replace("|", ".").replace("।", ".").replace("॥", ".")
        out.append(r)
    return "".join(out)


# --------------------------------------------------------------------------- #
def build_en(frame: pd.DataFrame, revision: str | None = None) -> pd.DataFrame:
    """Recover the English originals by id, formatted exactly like mmlu_hineng."""
    from datasets import load_dataset

    subjects = sorted({str(i).split("/")[0] for i in frame["id"]})
    print(f"  loading {len(subjects)} cais/mmlu subjects...")
    cache = {}
    for s in subjects:
        cache[s] = load_dataset("cais/mmlu", s, split="test", revision=revision)

    rows, mismatch, oor = [], [], []
    for _, r in frame.iterrows():
        subject, _split, idx = str(r["id"]).split("/")
        ds = cache[subject]
        idx = int(idx)
        if idx >= len(ds):
            oor.append(r["id"])
            continue
        item = ds[idx]
        gold = LETTERS[int(item["answer"])]
        if gold != str(r["answer"]).strip():
            mismatch.append((r["id"], gold, r["answer"]))
            continue
        opts = "\n".join(f"({LETTERS[i]}): {c}" for i, c in enumerate(item["choices"]))
        rows.append({
            "index": int(r["index"]), "id": r["id"],
            "sentence": f"{item['question']}\n{opts}",
            "answer": gold, "src": "cais/mmlu test",
        })

    if oor:
        raise SystemExit(f"ABORT: {len(oor)} ids out of range, e.g. {oor[:3]}")
    if mismatch:
        raise SystemExit(
            f"ABORT: {len(mismatch)} gold-answer disagreements between mmlu_hineng and "
            f"cais/mmlu, e.g. {mismatch[:3]}. The id join is unsafe; do not use."
        )
    return pd.DataFrame(rows)


def build_hi(frame: pd.DataFrame, revision: str | None = None) -> pd.DataFrame:
    """Retrieve monolingual Hindi from Global-MMLU, joined on sample_id."""
    from datasets import load_dataset

    gm = load_dataset("CohereLabs/Global-MMLU", "hi", split="test", revision=revision)
    by_id = {str(r["sample_id"]): r for r in gm}

    rows, missing, mismatch = [], [], []
    for _, r in frame.iterrows():
        key = str(r["id"])
        h = by_id.get(key)
        if h is None:
            missing.append(key)
            continue
        gold = str(h["answer"]).strip()
        if gold != str(r["answer"]).strip():
            mismatch.append((key, gold, r["answer"]))
            continue
        opts = "\n".join(f"({L}): {h['option_' + L.lower()]}" for L in LETTERS)
        rows.append({
            "index": int(r["index"]), "id": key,
            "sentence": f"{h['question']}\n{opts}",
            "answer": gold, "src": "CohereLabs/Global-MMLU hi",
            "annotated": bool(h.get("is_annotated")),
        })
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} ids absent from Global-MMLU, e.g. {missing[:3]}")
    if mismatch:
        raise SystemExit(
            f"ABORT: {len(mismatch)} gold disagreements with Global-MMLU, e.g. {mismatch[:3]}")
    return pd.DataFrame(rows)


def romanize_frame(df: pd.DataFrame, src_label: str) -> pd.DataFrame:
    """Apply the ROM transform to an already-built condition frame."""
    out = df.copy()
    out["sentence"] = [to_hinglish(str(s)) for s in df["sentence"]]
    out["src"] = src_label
    return out


def build_rom(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in frame.iterrows():
        rows.append({
            "index": int(r["index"]), "id": r["id"],
            "sentence": to_hinglish(str(r["sentence"])),
            "answer": str(r["answer"]).strip(), "src": "mmlu_hineng transliterated",
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/mmlu")
    ap.add_argument("--revision", default=None,
                     help="pin a Hub revision for cais/mmlu and Global-MMLU")
    args = ap.parse_args()

    cm = load_split(DATASETS["mmlu_hineng"], "data/raw", download=False).frame
    print(f"source: mmlu_hineng, {len(cm)} items")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("\n[EN] recovering English originals by id")
    en = build_en(cm, revision=args.revision)
    print(f"  built {len(en)} rows, gold agreement 100% (build aborts otherwise)")

    print("\n[ROM] transliterating Devanagari -> romanized Hindi")
    rom = build_rom(cm)
    still_dev = sum(1 for s in rom["sentence"] if DEV.search(str(s)))
    print(f"  built {len(rom)} rows; rows with leftover Devanagari: {still_dev}")

    print("\n[HI] retrieving monolingual Hindi from Global-MMLU (sample_id join)")
    hi = build_hi(cm, revision=args.revision)
    n_ann = int(hi["annotated"].sum())
    print(f"  built {len(hi)} rows, gold agreement 100%; human-annotated {n_ann} "
          f"({100*n_ann/len(hi):.1f}%)")

    print("\n[HI-ROM] transliterating the Hindi condition")
    hirom = romanize_frame(hi, "Global-MMLU hi transliterated")
    left = sum(1 for s in hirom["sentence"] if DEV.search(str(s)))
    print(f"  built {len(hirom)} rows; rows with leftover Devanagari: {left}")

    for name, df in [("mmlu_hineng_en", en), ("mmlu_hineng_rom", rom),
                     ("mmlu_hineng_hi", hi), ("mmlu_hineng_hirom", hirom)]:
        spec = DERIVED[name]
        if len(df) != spec.n_rows:
            raise SystemExit(f"ABORT: {name} has {len(df)} rows, registry expects {spec.n_rows}")
        p = out / f"{name}.csv"
        df.to_csv(p, index=False, encoding="utf-8")
        print(f"  wrote {p}  ({len(df)} rows)")

    print("\nsanity: same item, five conditions")
    i = 0
    for tag, df in [("EN    ", en), ("HI    ", hi), ("HI-ROM", hirom),
                    ("CM    ", cm), ("CM-ROM", rom)]:
        print(f"  {tag}:", str(df['sentence'].iloc[i]).split("\n")[0][:96])
    golds = {t: df['answer'].iloc[i] for t, df in
             [("EN", en), ("HI", hi), ("HI-ROM", hirom), ("CM", cm), ("CM-ROM", rom)]}
    print("  gold:", golds, "-> all equal:", len(set(map(str, golds.values()))) == 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
