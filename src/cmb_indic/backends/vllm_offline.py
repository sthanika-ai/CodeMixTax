"""In-process vLLM backend -- fastest option for a full sweep.

Loads the model once inside this process and batches every prompt of a split
through vLLM's continuous scheduler. Compared to the HTTP backend this removes
per-request overhead and lets vLLM see the whole batch at once, which on a
21k-row sweep is typically a large wall-clock win.

Trade-off: one model per process. The runner therefore instantiates one backend
per model and evaluates all of that model's splits before moving on.
"""

from __future__ import annotations

import os
import time

from ..config import ModelConfig
from .base import Backend, Generation, Prompt, as_text_prompt


class VLLMOfflineBackend(Backend):
    """Run vLLM in this process via ``vllm.LLM``."""

    name = "vllm-offline"

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The vllm backend needs vLLM: pip install 'cmb-indic[vllm]'. "
                "On an unsupported platform, serve the model separately and use "
                "backend: openai instead."
            ) from exc

        self._SamplingParams = SamplingParams
        kwargs: dict = {
            "model": cfg.hf_id,
            "dtype": cfg.dtype,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "trust_remote_code": cfg.trust_remote_code,
            "seed": cfg.seed,
        }
        if cfg.quantization:
            kwargs["quantization"] = cfg.quantization
        if cfg.max_model_len:
            kwargs["max_model_len"] = cfg.max_model_len
        if cfg.hf_revision:
            kwargs["revision"] = cfg.hf_revision
        if cfg.max_num_seqs:
            kwargs["max_num_seqs"] = cfg.max_num_seqs

        # A ReasoningConfig is the enable switch for thinking_token_budget: without
        # it the sampler has no end-token to inject and the per-request cap is a
        # no-op. Built here rather than in _sampling() because it is engine-level.
        if cfg.thinking_token_budget is not None:
            if len(cfg.reasoning_delimiters) != 2:
                raise ValueError(
                    "thinking_token_budget needs reasoning_delimiters: "
                    '[start, end], e.g. ["<think>", "</think>"]'
                )
            from vllm.config.reasoning import ReasoningConfig

            start, end = cfg.reasoning_delimiters
            kwargs["reasoning_config"] = ReasoningConfig(
                reasoning_start_str=start, reasoning_end_str=end
            )

        self._llm = LLM(**kwargs)

    # ------------------------------------------------------------------ #
    def _sampling(self):
        extra: dict = {}
        if self.cfg.thinking_token_budget is not None:
            extra["thinking_token_budget"] = self.cfg.thinking_token_budget
        return self._SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
            stop=self.cfg.stop or None,
            seed=self.cfg.seed,
            **extra,
        )

    def generate(self, prompts: list[Prompt], *, desc: str = "") -> list[Generation]:
        started = time.perf_counter()
        params = self._sampling()

        if self.cfg.prompt_style == "completion":
            texts = [as_text_prompt(p) for p in prompts]
            outputs = self._llm.generate(texts, params)
        else:
            chats = [p if isinstance(p, list) else [{"role": "user", "content": p}] for p in prompts]
            kwargs = {}
            if self.cfg.chat_template_kwargs:
                kwargs["chat_template_kwargs"] = self.cfg.chat_template_kwargs
            outputs = self._llm.chat(chats, params, **kwargs)

        elapsed = time.perf_counter() - started
        per_item = elapsed / max(1, len(prompts))

        results: list[Generation] = []
        for out in outputs:
            comp = out.outputs[0]
            results.append(
                Generation(
                    text=comp.text or "",
                    finish_reason=getattr(comp, "finish_reason", "") or "",
                    prompt_tokens=len(out.prompt_token_ids or []),
                    completion_tokens=len(comp.token_ids or []),
                    latency_s=per_item,
                )
            )
        return results

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            {
                "tensor_parallel_size": self.cfg.tensor_parallel_size,
                "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
                "max_model_len": self.cfg.max_model_len,
                "hf_revision": self.cfg.hf_revision,
                "max_num_seqs": self.cfg.max_num_seqs,
                "thinking_token_budget": self.cfg.thinking_token_budget,
                "reasoning_delimiters": self.cfg.reasoning_delimiters or None,
            }
        )
        try:
            import vllm

            info["vllm_version"] = vllm.__version__
        except Exception:  # noqa: BLE001 # nosec B110 -- version probe, not security-relevant
            pass
        # Which model runner served the request decides whether the thinking cap
        # was honoured at all: V2 ignores it with only a warning.
        info["v2_model_runner"] = bool(os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "1")
        return info

    def close(self) -> None:
        # Free GPU memory so the runner can load the next model in the same process.
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        try:
            del self._llm
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 # nosec B110 -- best-effort GPU cleanup, not security-relevant
            pass
