"""A dependency-free fake backend, for tests, CI, and dry runs.

`EchoBackend` produces syntactically plausible answers without a model, so the
whole pipeline -- data loading, prompting, extraction, scoring, reporting -- can
be exercised in CI on a machine with no GPU and no network.

`scripted` lets a test pin exact replies per prompt index.
"""

from __future__ import annotations

import ast
import re

from .base import Backend, Generation, Prompt, as_text_prompt


class EchoBackend(Backend):
    """Returns canned, task-shaped answers. Never contacts a model."""

    name = "echo"

    def __init__(self, cfg, *, scripted: list[str] | None = None) -> None:
        super().__init__(cfg)
        self.scripted = scripted

    def generate(self, prompts: list[Prompt], *, desc: str = "") -> list[Generation]:
        out: list[Generation] = []
        for i, prompt in enumerate(prompts):
            if self.scripted is not None:
                text = self.scripted[i % len(self.scripted)]
            else:
                text = self._synthesise(as_text_prompt(prompt))
            out.append(
                Generation(text=text, finish_reason="stop", completion_tokens=len(text.split()))
            )
        return out

    @staticmethod
    def _synthesise(text: str) -> str:
        """Emit something the extractors can parse, inferred from the prompt."""
        low = text.lower()

        # Token-level tasks: echo a label per token so alignment can be tested.
        m = re.search(r"tokenized sentence:\s*(\[.*?\])\s*;", text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                tokens = ast.literal_eval(m.group(1))
            except Exception:
                tokens = []
            if tokens:
                label = _default_token_label(low)
                body = ", ".join(f'{{"{t}": "{label}"}}' for t in tokens)
                return f"[{body}]"

        # Order matters: dispatch on distinctive system-prompt markers only. Do NOT
        # test for "final answer" -- every task's user turn ends with "Your final
        # answer:", so it matches everything and would route SA to the GSM8K reply.
        if "sentiment analysis" in low:
            # Echo back a label the split actually uses, read off its own prompt.
            for lab in ("Mixed_feelings", "Positive", "Negative", "Neutral",
                        "positive", "negative", "neutral"):
                if lab.lower() in low:
                    return lab
            if "'o'" in low or "offensive" in low:
                return "N"
            return "neutral"
        if "translate" in low:
            return "this is a placeholder translation"
        if "mathematical problems" in low or "solving mathematical" in low:
            return "Solution:\nStep 1: assume 42.\nFinal answer: 42"
        if "multiple choice" in low or re.search(r"^\(a\):", text, re.MULTILINE | re.IGNORECASE):
            return "Answer: A"
        if "part of speech" in low:
            return "[]"
        return "A"


def _default_token_label(low: str) -> str:
    if "language identification" in low:
        return "MAR" if "marathi" in low else "lang1"
    if "part of speech" in low:
        return "NOUN"
    if "named entity" in low:
        return "O"
    return "O"
