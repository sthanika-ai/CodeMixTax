"""Turn raw model generations into scoreable predictions.

Two extraction modes are available per run, selected with ``--extraction``:

``robust`` (default)
    Anchored, order-aware parsing that handles the ways instruction-tuned local
    models actually reply ("Answer: B", "**B**", "The answer is (C).", a chain of
    thought followed by a final line, ...). Use this for new numbers.

``paper``
    Bug-for-bug reproduction of the upstream `analyse_Sentence_Label_Result` /
    `result2pred` logic. Use this only when you need numbers directly comparable
    to the published CodeMixBench table.

The two modes differ materially, and the differences favour ``robust``:

1.  MCQ. Upstream runs ``re.findall(r"[ABCD]", answer)`` over the whole raw reply
    and takes the *first* hit. Any capital A-D anywhere wins -- so the reply
    "Answer: B" is scored as **A**, because "Answer" begins with a capital A.
    This silently penalises every model that prefixes its choice, which is most
    of them. ``robust`` anchors on explicit answer patterns instead.

2.  GSM8K. When upstream finds several numbers in the reply it checks whether
    the *gold* answer is among them and, if so, predicts the gold answer
    (`utils.py` ~line 810). That is label leakage: it inflates accuracy for
    verbose models. ``robust`` never looks at the gold value and takes the final
    number, the standard GSM8K convention.

3.  Token tasks. Upstream aligns predicted tags to gold tokens through
    ``dict(zip(tokens, pred))``, so a sentence containing the same token twice
    collapses to one entry and later occurrences inherit the earlier tag.
    ``robust`` aligns positionally whenever the parsed length matches the gold
    length, and only falls back to by-name lookup otherwise.

Reasoning-model handling applies in both modes, because it is a transport
concern rather than a scoring choice: hidden chain-of-thought segments
(``<think>``, gpt-oss harmony ``analysis`` channels, ...) are stripped before
extraction so that stray letters and numbers inside the scratchpad cannot be
mistaken for the answer.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from .metrics import UNK

ExtractionMode = str  # "robust" | "paper"


# --------------------------------------------------------------------------- #
# Reasoning / scratchpad stripping
# --------------------------------------------------------------------------- #
_THINK_BLOCKS = [
    # Qwen3.x, DeepSeek-R1/V3.x/V4, and most open reasoning models.
    re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning\b[^>]*>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<scratchpad\b[^>]*>.*?</scratchpad>", re.DOTALL | re.IGNORECASE),
    # gpt-oss harmony format: keep only the final channel. Note the pipes on BOTH
    # sides -- distinct from Gemma 4's `<|channel>` below, which is a different scheme.
    re.compile(r"<\|channel\|>analysis<\|message\|>.*?(?=<\|channel\|>|<\|end\|>|\Z)", re.DOTALL),
    # Gemma 4: `<|channel>thought ... <channel|>`. Mirrors the `strip_thinking`
    # macro in Google's canonical chat template, which splits on `<channel|>` and
    # discards whatever followed a `<|channel>` open marker. Present whenever the
    # model is run with enable_thinking=true; the template's default pre-closes an
    # empty thought block instead, which this pattern also removes harmlessly.
    re.compile(r"<\|channel>\s*\w*\s*.*?<channel\|>", re.DOTALL),
]

#: An unterminated reasoning block means the model ran out of tokens mid-thought.
#: Everything from the opening marker onward is discarded, so the row is scored as
#: unparsed rather than contributing a letter or number lifted from the scratchpad.
#: Covers both the `<think>` family and Gemma 4's `<|channel>thought`.
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>|<\|channel>", re.IGNORECASE)

#: Chat-template scaffolding that can leak into a raw completion. Gemma 4 uses
#: single-sided pipes (`<|turn>` ... `<turn|>`), unlike the `<|...|>` convention,
#: so both shapes are stripped.
_SCAFFOLDING = re.compile(
    r"<\|[a-z_]+\|>"          # <|think|>, <|message|>, <|end|>
    r"|<\|(?:turn|channel|tool|tool_call|tool_response|image|audio|video)>"
    r"|<(?:turn|channel|tool|tool_call|tool_response)\|>",
    re.IGNORECASE,
)

_HARMONY_FINAL = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|\Z)", re.DOTALL
)

#: vLLM 0.10.1 renders gpt-oss harmony output as PLAIN TEXT, with the bare words
#: `analysis` and `assistantfinal` as delimiters and no pipe markers at all:
#:
#:     analysisWe have a multiple choice question... So answer: (A).assistantfinal(A)
#:
#: Matching only the pipe form left the whole reasoning trace in place, and because
#: the trace enumerates the options ("(A): dijitala. (B): printa.") the extractor
#: happily returned a letter read out of the model's thinking rather than its answer.
_HARMONY_FINAL_BARE = re.compile(r"assistantfinal(.*)\Z", re.DOTALL)

#: A reply that opened a reasoning trace but never produced a final channel did not
#: answer -- it ran out of budget mid-thought. Such rows must score as unparsed, not
#: as whatever option letter happens to appear in the reasoning.
_HARMONY_ANALYSIS_OPEN = re.compile(r"\A\s*analysis|<\|channel\|>analysis")

#: Sarvam-M emits its reasoning as bare prose and closes it with an UNMATCHED
#: `</think>`, with no opening tag anywhere:
#:
#:     Okay, let's tackle this question. The options are (A) digital, (B) print...
#:     </think>
#:
#:     (B)
#:
#: The paired `<think>...</think>` pattern therefore never matches, the reasoning
#: survives into the extractor, and because the trace enumerates the options the
#: extractor returns a letter the model was merely considering. Keeping only what
#: follows the LAST closing tag fixes it; `rfind` rather than a greedy group so a
#: trace that mentions `</think>` in passing cannot split it early.
_THINK_CLOSE = "</think>"


def strip_reasoning(text: str) -> str:
    """Remove hidden chain-of-thought so only the user-visible answer remains."""
    if not text:
        return ""
    out = str(text)

    # gpt-oss: if an explicit final channel exists, that *is* the answer. Check the
    # pipe-delimited form first, then the bare-word form vLLM 0.10.1 emits.
    final = _HARMONY_FINAL.search(out) or _HARMONY_FINAL_BARE.search(out)
    if final:
        out = final.group(1)
    elif _HARMONY_ANALYSIS_OPEN.search(out):
        # Reasoning started and never finished: no answer was produced at all.
        return ""
    elif _THINK_CLOSE in out and "<think" not in out.lower():
        # Unmatched closing tag: everything before it is reasoning.
        out = out[out.rfind(_THINK_CLOSE) + len(_THINK_CLOSE):]
    else:
        for pat in _THINK_BLOCKS:
            out = pat.sub(" ", out)
        # Unterminated reasoning: drop everything from the opening tag onward.
        m = _UNCLOSED_THINK.search(out)
        if m:
            out = out[: m.start()]

    # Leftover chat-template scaffolding.
    out = _SCAFFOLDING.sub(" ", out)
    return out.strip()


def clean_answer(text: str) -> str:
    """Upstream `cleanAnswer`, used before re-parsing a malformed token list."""
    return str(text).replace('""', '"').replace("\n", "").strip(" ,.")


def _strip_markup(text: str) -> str:
    """Drop markdown emphasis/code fences that wrap short answers."""
    out = re.sub(r"```[a-zA-Z]*\n?", " ", text).replace("```", " ")
    return out.replace("**", "").replace("__", "").replace("`", "")


# --------------------------------------------------------------------------- #
# Multiple choice: MMLU (A-D), TruthfulQA (A-N)
# --------------------------------------------------------------------------- #
def _mcq_letters(labels: tuple[str, ...] | None, default: str) -> str:
    if not labels:
        return default
    letters = "".join(sorted({str(x).strip().upper() for x in labels if len(str(x).strip()) == 1}))
    return letters or default


def extract_mcq(
    text: str,
    labels: tuple[str, ...] | None,
    *,
    mode: ExtractionMode = "robust",
    default_letters: str = "ABCD",
) -> str:
    """Recover the chosen option letter, or UNK."""
    letters = _mcq_letters(labels, default_letters)

    if mode == "paper":
        # Faithful upstream behaviour: first bare letter anywhere in the reply.
        hits = re.findall(f"[{letters}]", str(text), re.MULTILINE)
        return hits[0].strip() if hits else UNK

    body = _strip_markup(strip_reasoning(text))
    if not body.strip():
        return UNK

    # Two reply shapes need opposite treatment, and the shape -- not the model -- is
    # what decides:
    #
    #   answer-FIRST   " C\n\nD\n\nD\n\n..."          (Sarvam-1 base, Llama 3.1)
    #                  the answer opens the reply and everything after is noise or
    #                  restated option text, so the FIRST token is the answer.
    #   answer-LAST    "## Step 1: option (A)... The best answer is B."
    #                  (Llama 4 Scout, Sarvam-M, gpt-oss) the reply reasons through the
    #                  options before committing, so a first-match returns a letter the
    #                  model merely considered.
    #
    # Taking the first match everywhere mis-scored 185 of 513 verbose Llama 4 English
    # replies (worth 14.36 points); taking the last match everywhere cost Sarvam-1 8.69
    # points by reading its repetition tail. So: a reply that OPENS with a standalone
    # answer resolves on that, and only otherwise do we look for a closing verdict.
    lead = re.match(rf"\s*[\(\[\{{]?\s*([{letters}])\s*[\)\]\}}]?\s*(?:$|[\n\.\):,])",
                    body, re.IGNORECASE)
    if lead:
        return lead.group(1).upper()

    # No opening answer: find a closing verdict, taking the LAST occurrence.
    verdict = [
        rf"(?:best\s+answer|answer|ans|option|choice|correct)\s*(?:is|:|=|-)?\s*"
        rf"[\(\[\{{]?\s*([{letters}])\b",
        rf"[\(\[]\s*([{letters}])\s*[\)\]]",
        rf"^\s*([{letters}])\s*[\)\.:,\-]",
        rf"^\s*([{letters}])\s*$",
    ]
    for pat in verdict:
        hits = list(re.finditer(pat, body, re.IGNORECASE | re.MULTILINE))
        if hits:
            return hits[-1].group(1).upper()

    # Last resort: a standalone letter token. Prefer the last one, since a
    # verbose reply usually lands on its conclusion at the end.
    solo = re.findall(rf"(?<![A-Za-z0-9])([{letters}])(?![A-Za-z0-9])", body)
    if solo:
        return solo[-1].upper()
    return UNK


# --------------------------------------------------------------------------- #
# GSM8K
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_number(raw: str) -> str:
    """Canonicalise a numeric string so exact match is not defeated by format.

    "1,234.0" -> "1234", "-0.50" -> "-0.5", "12." -> "12".
    """
    s = str(raw).strip().strip(".").replace(",", "").replace(":", "").replace("$", "")
    s = s.replace("%", "").strip()
    if not s:
        return UNK
    try:
        val = float(s)
    except ValueError:
        return s
    if val.is_integer():
        return str(int(val))
    return repr(round(val, 6)).rstrip("0").rstrip(".")


def extract_gsm8k(
    text: str,
    *,
    mode: ExtractionMode = "robust",
    gold: str | None = None,
) -> str:
    """Recover the final numeric answer, or UNK.

    In ``paper`` mode `gold` reproduces upstream's gold-peeking tie-break. In
    ``robust`` mode `gold` is ignored entirely -- passing it has no effect, by
    design, so the scoring path cannot leak the label.
    """
    raw = str(text)

    if mode == "paper":
        if "Final Answer:" in raw:
            tail = raw.split("Final Answer:", 1)[1]
        elif "Final answer:" in raw:
            tail = raw.split("Final answer:", 1)[1]
        else:
            tail = raw
        if "[stop]" in raw:
            tail = tail.split("[stop]", 1)[0]
        hits = re.findall(r"\-?[0-9\.\,]+", tail.strip("."), re.MULTILINE)
        hits = [h.strip().strip(".").replace(",", "").replace(":", "") for h in hits if h not in (".", ",")]
        hits = [h for h in hits if h]
        if not hits:
            return UNK
        if len(hits) == 1:
            return hits[0]
        # Upstream label leakage, reproduced only in this mode.
        if gold is not None and str(gold) in hits:
            return str(gold)
        return hits[0]

    body = _strip_markup(strip_reasoning(raw))

    # Prefer an explicit final-answer marker.
    for marker in ("final answer", "final_answer", "answer is", "answer:", "####"):
        idx = body.lower().rfind(marker)
        if idx != -1:
            tail = body[idx + len(marker):]
            tail = tail.split("[stop]", 1)[0]
            nums = _NUM.findall(tail)
            if nums:
                return normalize_number(nums[0])

    # Otherwise the last number in the reply, the standard GSM8K convention.
    nums = _NUM.findall(body)
    return normalize_number(nums[-1]) if nums else UNK


# --------------------------------------------------------------------------- #
# Sentence classification: SA (label spaces differ per split)
# --------------------------------------------------------------------------- #
def extract_label(
    text: str,
    labels: tuple[str, ...],
    *,
    mode: ExtractionMode = "robust",
) -> str:
    """Match the reply against a split's closed label set, preserving gold casing.

    Gold casing varies across splits (`positive` for Hindi/Marathi vs `Positive`
    for Tamil/Malayalam, and `O`/`N` for the Bengali offensive-language split),
    so matching is case-insensitive but the value returned is always the exact
    gold spelling from the registry -- otherwise accuracy would be destroyed by
    capitalisation alone.
    """
    if mode == "paper":
        return str(text).strip()

    body = _strip_markup(strip_reasoning(text)).strip()
    if not body:
        return UNK
    low = body.lower()

    # Longest label first so "Mixed_feelings" is not shadowed by a substring.
    ordered = sorted(labels, key=lambda s: -len(str(s)))

    # 1. Whole reply is exactly a label.
    for lab in ordered:
        if low == str(lab).lower():
            return lab

    # 2. Explicit "answer: <label>".
    for lab in ordered:
        variants = {str(lab).lower(), str(lab).lower().replace("_", " ")}
        for v in variants:
            if re.search(
                rf"(?:answer|label|sentiment|class)\s*(?:is|:|=|-)?\s*['\"\(\[]?{re.escape(v)}\b",
                low,
            ):
                return lab

    # 3. Single-character label sets (the O/N offensive split) need word
    #    boundaries, or any stray "o" in prose would match.
    if all(len(str(x)) == 1 for x in labels):
        hits = [
            lab for lab in ordered
            if re.search(rf"(?<![A-Za-z]){re.escape(str(lab))}(?![A-Za-z])", body)
        ]
        if len(hits) == 1:
            return hits[0]
        if hits:
            return hits[-1]
        return UNK

    # 4. The reply OPENS with a label. This has to be checked before any
    #    trailing-mention rule: on SA these models answer first and explain after --
    #    "Neutral.\n\n**Reasoning:** The sentence consists solely of..." -- and the
    #    explanation names other sentiments. Taking the last mention there returned
    #    "negative" for a reply whose verdict was plainly "Neutral". Only the opening
    #    label is the answer; everything after it is discussion.
    head = low.lstrip()[:64]
    for lab in ordered:
        for v in {str(lab).lower(), str(lab).lower().replace("_", " ")}:
            if re.match(rf"['\"\(\[]?{re.escape(v)}\b", head):
                return lab

    # 5. Label mentioned anywhere. No opening label and no explicit "answer:", so a
    #    trailing mention is the best remaining guess -- a reply that reasons its way
    #    to a verdict states it at the end.
    found: list[tuple[int, str]] = []
    for lab in ordered:
        for v in {str(lab).lower(), str(lab).lower().replace("_", " ")}:
            pos = low.rfind(v)
            if pos != -1:
                found.append((pos, lab))
    if found:
        found.sort()
        return found[-1][1]
    return UNK


# --------------------------------------------------------------------------- #
# MT
# --------------------------------------------------------------------------- #
_MT_PREAMBLE = re.compile(
    r"^\s*(?:sure[,!.]?\s*)?(?:here(?:'s| is)[^:\n]*:|"
    r"(?:the\s+)?(?:english\s+)?translation(?:\s+is)?\s*:|your answer\s*:|answer\s*:)\s*",
    re.IGNORECASE,
)


def extract_translation(text: str, *, mode: ExtractionMode = "robust") -> str:
    """Recover the translated sentence.

    BLEU is unforgiving about boilerplate, so conversational preambles ("Sure!
    Here is the translation:") and surrounding quotes are removed. Anything left
    empty scores as an empty hypothesis rather than UNK, since BLEU has no
    notion of an unparsed row.
    """
    if mode == "paper":
        return str(text).strip()

    body = _strip_markup(strip_reasoning(text)).strip()
    if not body:
        return ""

    # A single-line reply is the translation; otherwise drop empty lines and
    # take the first substantive one after stripping any preamble.
    body = _MT_PREAMBLE.sub("", body).strip()
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if lines:
        body = lines[0] if len(lines) > 1 else lines[0]

    body = body.strip()
    if len(body) >= 2 and body[0] in "\"'“‘" and body[-1] in "\"'”’":
        body = body[1:-1].strip()
    return body


# --------------------------------------------------------------------------- #
# Token-level tasks: LID, POS, NER
# --------------------------------------------------------------------------- #
@dataclass
class TokenExtraction:
    labels: list[str]  # always len(gold_tokens)
    ok: bool  # False if nothing parseable was found
    aligned_by: str  # "position" | "name" | "none"
    n_missing: int  # tokens that fell back to UNK


def parse_token_list(text: str) -> list[tuple[str, str]] | None:
    """Parse ``[{"tok": "LAB"}, ...]`` into ordered (token, label) pairs.

    Tolerates the shapes local models actually emit: JSON vs Python literals,
    a single flat dict, a fenced code block, trailing prose, and ``tok: LAB``
    line lists. Returns None if nothing usable is found.
    """
    body = strip_reasoning(text)
    if not body:
        return None

    # Prefer the outermost [...] span; models often wrap it in prose.
    span = re.search(r"\[.*\]", body, re.DOTALL)
    candidates = [span.group(0)] if span else []
    candidates.append(body)
    candidates.append(clean_answer(body))

    for cand in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(cand)
            except Exception:  # nosec B112 -- fall through to the next parse candidate
                continue
            pairs = _pairs_from_obj(obj)
            if pairs:
                return pairs

    # Fall back to "token: LABEL" or '"token": "LABEL"' lines.
    line_pairs = re.findall(
        r'["\']?([^"\'\n:{}\[\]]{1,120}?)["\']?\s*:\s*["\']?([A-Za-z][A-Za-z0-9_\-]*)["\']?',
        body,
    )
    cleaned = [(tok.strip(), lab.strip()) for tok, lab in line_pairs if tok.strip()]
    return cleaned or None


def _pairs_from_obj(obj) -> list[tuple[str, str]] | None:
    """Normalise the parsed JSON/literal into ordered (token, label) pairs."""
    if isinstance(obj, dict):
        return [(str(k), str(v)) for k, v in obj.items()]
    if isinstance(obj, list):
        pairs: list[tuple[str, str]] = []
        for item in obj:
            if isinstance(item, dict):
                pairs.extend((str(k), str(v)) for k, v in item.items())
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((str(item[0]), str(item[1])))
            elif isinstance(item, str):
                # A bare label list, positionally aligned.
                pairs.append(("", item))
        return pairs or None
    return None


def extract_token_labels(
    text: str,
    gold_tokens: list[str],
    *,
    mode: ExtractionMode = "robust",
    valid_labels: tuple[str, ...] | None = None,
) -> TokenExtraction:
    """Align predicted per-token labels to `gold_tokens`.

    Always returns exactly ``len(gold_tokens)`` labels, padding with UNK, so the
    result can be scored without a length check. That mirrors upstream, which
    also emits an all-UNK sequence on a parse failure.
    """
    n = len(gold_tokens)
    pairs = parse_token_list(text)
    if not pairs:
        return TokenExtraction([UNK] * n, ok=False, aligned_by="none", n_missing=n)

    tokens = [t for t, _ in pairs]
    tags = [lab for _, lab in pairs]

    if mode == "paper":
        # Upstream: by-name lookup, so duplicate tokens collapse.
        lookup = dict(zip(tokens, tags, strict=True))
        out = [lookup.get(tok, UNK) for tok in gold_tokens]
        return TokenExtraction(
            out, ok=True, aligned_by="name", n_missing=sum(1 for x in out if x == UNK)
        )

    # Positional alignment when the model returned one tag per gold token: this
    # is both the common case and the only correct handling of repeated tokens.
    if len(tags) == n:
        out = [_norm_tag(t, valid_labels) for t in tags]
        return TokenExtraction(
            out, ok=True, aligned_by="position", n_missing=sum(1 for x in out if x == UNK)
        )

    # Otherwise fall back to by-name, consuming duplicates left to right.
    from collections import defaultdict

    buckets: dict[str, list[str]] = defaultdict(list)
    for tok, tag in zip(tokens, tags, strict=True):
        buckets[tok].append(tag)
    cursor: dict[str, int] = defaultdict(int)

    out = []
    for tok in gold_tokens:
        seq = buckets.get(tok)
        if seq and cursor[tok] < len(seq):
            out.append(_norm_tag(seq[cursor[tok]], valid_labels))
            cursor[tok] += 1
        else:
            out.append(UNK)
    return TokenExtraction(
        out, ok=True, aligned_by="name", n_missing=sum(1 for x in out if x == UNK)
    )


def _norm_tag(tag: str, valid: tuple[str, ...] | None) -> str:
    """Snap a tag onto the split's label set, case-insensitively."""
    t = str(tag).strip().strip("'\"")
    if not t:
        return UNK
    if not valid:
        return t
    if t in valid:
        return t
    low = {str(v).lower(): v for v in valid}
    return low.get(t.lower(), t)


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
def extract_prediction(
    task: str,
    text: str,
    *,
    labels: tuple[str, ...] | None = None,
    gold_tokens: list[str] | None = None,
    gold: str | None = None,
    mode: ExtractionMode = "robust",
) -> str | TokenExtraction:
    """Dispatch to the right extractor for `task`. Used by the runner."""
    if task in ("lid", "pos", "ner"):
        if gold_tokens is None:
            raise ValueError(f"task {task} requires gold_tokens")
        return extract_token_labels(text, gold_tokens, mode=mode, valid_labels=labels)
    if task == "mmlu":
        return extract_mcq(text, labels, mode=mode, default_letters="ABCD")
    if task == "truthfulqa":
        return extract_mcq(text, labels, mode=mode, default_letters="ABCDEFGHIJKLMN")
    if task == "gsm8k":
        return extract_gsm8k(text, mode=mode, gold=gold)
    if task == "sa":
        if not labels:
            raise ValueError("task sa requires a label set")
        return extract_label(text, labels, mode=mode)
    if task == "mt":
        return extract_translation(text, mode=mode)
    raise ValueError(f"unknown task: {task!r}")
