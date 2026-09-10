"""Backend interface.

A backend turns prompts into text. Nothing else in the pipeline knows how the
model is served, so adding a new runtime means implementing one method.

Contract:
  * ``generate`` returns exactly one `Generation` per input prompt, in order.
  * A per-item failure is returned as a `Generation` with ``error`` set and
    ``text=""`` -- it must not raise, so one bad row cannot lose a 3000-row sweep.
  * Prompts are either a list of chat messages or a plain string; a backend that
    only supports one shape converts or raises in ``__init__``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config import ModelConfig

Prompt = list[dict[str, str]] | str


@dataclass
class Generation:
    """One model reply plus the bookkeeping the runner records."""

    text: str
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def truncated(self) -> bool:
        """Hit the token ceiling -- the usual cause of unparseable reasoning output."""
        return self.finish_reason == "length"


class Backend(ABC):
    """Base class for all runtimes."""

    #: Human-readable name recorded in run metadata.
    name: str = "base"

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def generate(self, prompts: list[Prompt], *, desc: str = "") -> list[Generation]:
        """Generate one reply per prompt, in input order."""

    def describe(self) -> dict[str, Any]:
        """Runtime details recorded in ``run_meta.json`` for reproducibility."""
        return {
            "backend": self.name,
            "model": self.cfg.hf_id,
            "served_model_name": self.cfg.served_model_name,
            "prompt_style": self.cfg.prompt_style,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
            "seed": self.cfg.seed,
            "quantization": self.cfg.quantization,
            "dtype": self.cfg.dtype,
        }

    def close(self) -> None:  # noqa: B027 -- optional hook; most backends hold nothing
        """Release resources. Safe to call more than once."""

    #: Fields fixed when the runtime is constructed. Changing any of them needs a
    #: new backend instance; everything else is a per-request sampling knob.
    LOAD_TIME_FIELDS: tuple[str, ...] = (
        "backend", "hf_id", "served_model_name", "base_url",
        "dtype", "quantization", "tensor_parallel_size", "gpu_memory_utilization",
        "max_model_len", "hf_revision", "max_num_seqs", "trust_remote_code", "load_in_4bit",
    )

    def retarget(self, cfg: ModelConfig) -> None:
        """Swap in a per-task config without reloading weights.

        Lets one loaded model serve every split even when tasks need different
        `max_tokens` -- reloading a 27B checkpoint 18 times would dominate the
        wall clock. Raises if a load-time field differs, since that genuinely
        does require a fresh instance.
        """
        for name in self.LOAD_TIME_FIELDS:
            if getattr(cfg, name) != getattr(self.cfg, name):
                raise ValueError(
                    f"cannot retarget {self.name}: load-time field {name!r} differs "
                    f"({getattr(self.cfg, name)!r} -> {getattr(cfg, name)!r}). "
                    "Remove it from task_overrides, or run this task separately."
                )
        self.cfg = cfg

    def __enter__(self) -> Backend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def as_text_prompt(prompt: Prompt) -> str:
    """Flatten a chat prompt into a plain string, for completion-only runtimes."""
    if isinstance(prompt, str):
        return prompt
    # Roles are concatenated without markers: a completion-only runtime has no
    # chat template, so inventing "System:"/"User:" labels would put tokens in
    # front of the model that it never saw in training.
    parts = [msg.get("content", "") for msg in prompt]
    return "\n\n".join(p for p in parts if p)
