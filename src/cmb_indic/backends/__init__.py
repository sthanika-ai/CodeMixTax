"""Backend registry and factory.

Heavy runtimes (vLLM, torch) are imported lazily inside `build_backend`, so
`import cmb_indic` stays fast and works on a machine with neither installed.
"""

from __future__ import annotations

from ..config import ModelConfig
from .base import Backend, Generation, Prompt, as_text_prompt
from .echo import EchoBackend

__all__ = [
    "Backend",
    "Generation",
    "Prompt",
    "as_text_prompt",
    "EchoBackend",
    "build_backend",
    "BACKENDS",
]

BACKENDS = ("openai", "vllm", "hf", "echo")


def build_backend(cfg: ModelConfig, **kwargs) -> Backend:
    """Instantiate the backend named by ``cfg.backend``."""
    if cfg.backend == "openai":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(cfg)
    if cfg.backend == "vllm":
        from .vllm_offline import VLLMOfflineBackend

        return VLLMOfflineBackend(cfg)
    if cfg.backend == "hf":
        from .hf_local import HFLocalBackend

        return HFLocalBackend(cfg, **kwargs)
    if cfg.backend == "echo":
        return EchoBackend(cfg, **kwargs)
    raise ValueError(f"unknown backend {cfg.backend!r}; expected one of {BACKENDS}")
