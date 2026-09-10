"""Prompt construction -- same templates and wording as upstream, made deterministic.

The system prompts come verbatim from the upstream ``prompt.json`` (a copy of
which lives at the repo root and is not modified), so a model sees exactly the
instructions the published benchmark used. The user turn is assembled with the
same f-strings as upstream's five `generate_*_Prompts` functions.

One deliberate change: **few-shot exemplar selection is seeded.** Upstream calls
``df.sample(n=1)`` with no ``random_state``, so every run draws different shots
and a re-run of the same command can move scores by a point or more. Here shots
are drawn from a per-(dataset, row) seeded RNG, which makes a run bit-for-bit
reproducible while keeping the shots row-dependent as upstream intended.

Two output shapes are supported:

``chat``       list of ``{"role", "content"}`` turns -- for instruction-tuned models.
``completion`` a single plain string -- for base models with no chat template
               (``sarvamai/sarvam-1`` is the one in this suite), mirroring the
               branch upstream reserved for ``gpt-3.5-turbo-instruct``.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .registry import DatasetSpec

Message = dict[str, str]
Prompt = list[Message] | str

#: Upstream caps exemplar length so a long shot cannot crowd out the question.
_MAX_SHOT_CHARS = {"mmlu": 600, "gsm8k": 600, "truthfulqa": 600}
#: How many draws before giving up on the length filter (upstream loops forever).
_MAX_DRAWS = 200


@lru_cache(maxsize=1)
def load_templates(prompt_file: str) -> dict[str, list[Message]]:
    """Load and cache the upstream prompt template file."""
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template file not found: {path}. This is upstream's "
            "prompt.json and must sit at the repo root (or be pointed at with "
            "--prompt-file)."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def get_template(dataset: str, prompt_file: str) -> list[Message]:
    templates = load_templates(prompt_file)
    if dataset not in templates:
        raise KeyError(
            f"No prompt template for {dataset!r} in {prompt_file}. "
            f"Available: {sorted(templates)[:5]}... ({len(templates)} total)"
        )
    return templates[dataset]


def system_text(dataset: str, prompt_file: str) -> str:
    return get_template(dataset, prompt_file)[0]["content"]


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_prompts(
    spec: DatasetSpec,
    frame: pd.DataFrame,
    *,
    prompt_file: str = "prompt.json",
    shots: int = 0,
    seed: int = 1234,
    style: str = "chat",
) -> list[Prompt]:
    """Build one prompt per row of `frame`, in row order.

    Args:
        shots: number of in-context exemplars. Only MMLU / GSM8K / TruthfulQA
            consume this, matching upstream, where the token-level and
            SA/MT prompts carry their single example inside the system text.
        style: ``"chat"`` or ``"completion"``.
    """
    if style not in ("chat", "completion"):
        raise ValueError(f"style must be 'chat' or 'completion', got {style!r}")

    # template_key, not name: derived condition splits deliberately reuse their
    # parent's prompt so only the question text differs between conditions.
    sys_text = system_text(spec.template_key, prompt_file)

    if spec.task in ("lid", "pos", "ner"):
        return [_token_prompt(sys_text, row, style) for _, row in frame.iterrows()]
    if spec.task in ("sa", "mt"):
        return [_sentence_prompt(sys_text, row, style) for _, row in frame.iterrows()]
    if spec.task == "mmlu":
        return _fewshot_prompts(spec, frame, sys_text, shots, seed, style, kind="mmlu")
    if spec.task == "truthfulqa":
        return _fewshot_prompts(spec, frame, sys_text, shots, seed, style, kind="truthfulqa")
    if spec.task == "gsm8k":
        return _fewshot_prompts(spec, frame, sys_text, shots, seed, style, kind="gsm8k")
    raise ValueError(f"unknown task: {spec.task!r}")


def _token_prompt(sys_text: str, row: pd.Series, style: str) -> Prompt:
    user = f"Tokenized sentence: {row['tokens']}; Your final answer:"
    if style == "completion":
        return f"{sys_text} {user}"
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]


def _sentence_prompt(sys_text: str, row: pd.Series, style: str) -> Prompt:
    user = f"Sentence: {row['sentence']}; Your final answer:"
    if style == "completion":
        return f"{sys_text}\n{user}"
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]


#: Extra instruction upstream appends to the GSM8K system turn so the answer is
#: locatable. Kept verbatim -- the extractor's "Final answer:" anchor depends on it.
_GSM8K_FORMAT = (
    "\nOutput the solution and final answer for the next problem. The solution "
    "should include the entire process of calculating the final answer. The final "
    "answer to the problem is just one definite numerical value. Don't output the "
    "problem. Output in this format:\nSolution:\nFinal answer: (one definite "
    "numerical value)"
)


def _fewshot_prompts(
    spec: DatasetSpec,
    frame: pd.DataFrame,
    sys_text: str,
    shots: int,
    seed: int,
    style: str,
    *,
    kind: str,
) -> list[Prompt]:
    """Build MMLU / TruthfulQA / GSM8K prompts with seeded in-context exemplars."""
    if kind == "gsm8k":
        sys_text = f"{sys_text}{_GSM8K_FORMAT}"

    # Exemplars are drawn from the split itself (upstream behaviour). The pool
    # is the *full* frame passed in; the current row is always excluded so a
    # question is never its own shot.
    pool = frame.reset_index(drop=True)
    max_chars = _MAX_SHOT_CHARS.get(kind, 600)

    prompts: list[Prompt] = []
    for pos, (_, row) in enumerate(pool.iterrows()):
        exemplars = _draw_shots(pool, row, shots, seed, pos, max_chars, needs_cot=(kind == "gsm8k"))
        prompts.append(_render(kind, sys_text, row, exemplars, style))
    return prompts


def _draw_shots(
    pool: pd.DataFrame,
    row: pd.Series,
    shots: int,
    seed: int,
    pos: int,
    max_chars: int,
    *,
    needs_cot: bool,
) -> list[pd.Series]:
    """Pick `shots` exemplars deterministically for this row.

    Seeded on (seed, row position) so the choice is stable across runs but still
    varies per question, as upstream's unseeded sampling did.
    """
    if shots <= 0 or len(pool) <= 1:
        return []

    rng = random.Random((seed * 1_000_003) ^ pos)  # nosec B311 -- deterministic exemplar sampling, not a security decision
    chosen: list[pd.Series] = []
    seen_text: set[str] = set()

    for _ in range(_MAX_DRAWS):
        if len(chosen) >= shots:
            break
        cand = pool.iloc[rng.randrange(len(pool))]
        if cand["index"] == row["index"]:
            continue
        text = str(cand["sentence"])
        if len(text) > max_chars:
            continue
        if text in seen_text:
            continue
        if needs_cot and (pd.isna(cand.get("cot")) or not str(cand.get("cot")).strip()):
            continue
        seen_text.add(text)
        chosen.append(cand)

    # If the filters starved us -- every candidate too long, or a split with many
    # duplicate sentences -- relax them rather than loop forever. Upstream's
    # unbounded `while` can hang outright on such splits. The row itself stays
    # excluded; everything else is now allowed, duplicates included, because
    # returning fewer shots than requested would silently change the prompt.
    if len(chosen) < shots:
        eligible = [i for i in range(len(pool)) if pool.iloc[i]["index"] != row["index"]]
        while eligible and len(chosen) < shots:
            chosen.append(pool.iloc[eligible[rng.randrange(len(eligible))]])
    return chosen


def _render(
    kind: str,
    sys_text: str,
    row: pd.Series,
    exemplars: list[pd.Series],
    style: str,
) -> Prompt:
    """Lay out system text + exemplars + question in the requested shape."""
    if kind == "gsm8k":
        def shot_pair(ex: pd.Series) -> tuple[str, str]:
            return (
                f"Problem:\n{ex['sentence']}",
                f"Solution:\n{ex['cot']}\n\n\nFinal Answer: {ex['answer']}",
            )

        question = f"Problem:\n{row['sentence']}"
    else:  # mmlu / truthfulqa
        def shot_pair(ex: pd.Series) -> tuple[str, str]:
            return f"{ex['sentence']}\n\nAnswer:", f"{ex['answer']}"

        question = f"{row['sentence']}\n\nAnswer:"

    if style == "completion":
        parts = [sys_text]
        for ex in exemplars:
            u, a = shot_pair(ex)
            parts.append(f"{u} {a}")
        parts.append(question)
        return "\n\n".join(parts)

    msgs: list[Message] = [{"role": "system", "content": sys_text}]
    for ex in exemplars:
        u, a = shot_pair(ex)
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": question})
    return msgs


# --------------------------------------------------------------------------- #
def prompt_fingerprint(prompts: list[Prompt]) -> str:
    """Stable hash of a prompt set, recorded in run metadata.

    Lets you prove that two runs of a split saw byte-identical inputs -- the
    thing that most often silently differs between "the same" evaluations.
    """
    import hashlib

    h = hashlib.sha256()
    for p in prompts:
        payload: Any = p if isinstance(p, str) else json.dumps(p, sort_keys=True, ensure_ascii=False)
        h.update(str(payload).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
