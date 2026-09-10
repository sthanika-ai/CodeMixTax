"""Backend for any OpenAI-compatible HTTP endpoint.

This is the default and the one to reach for first: it covers essentially every
local serving stack without code changes --

    vLLM        `vllm serve <model> --port 8000`
    SGLang      `python -m sglang.launch_server --model-path <model>`
    Ollama      built in at http://localhost:11434/v1
    llama.cpp   `llama-server --port 8000`
    LM Studio   local server tab
    TGI         `--port 8080` with the /v1 shim

Requests are issued concurrently with a bounded semaphore, retried with
exponential backoff on transient errors, and reported with a progress bar.
Failures degrade to a per-row error rather than aborting the sweep.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time

from tqdm.auto import tqdm

from ..config import ModelConfig
from .base import Backend, Generation, Prompt, as_text_prompt

#: Errors worth retrying: server still loading, rate limited, transient network.
_RETRY_MARKERS = (
    "timeout", "timed out", "connection", "temporarily", "overloaded",
    "429", "500", "502", "503", "504", "rate limit", "econnreset", "incomplete chunked",
)


class OpenAICompatBackend(Backend):
    """Drive a local OpenAI-compatible server."""

    name = "openai-compat"

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install 'openai>=1.40' to use the openai backend") from exc

        self._client = AsyncOpenAI(
            base_url=cfg.resolved_base_url(),
            api_key=cfg.resolved_api_key(),
            timeout=cfg.request_timeout,
            max_retries=0,  # retries handled here, so backoff is observable
        )

    # ------------------------------------------------------------------ #
    def generate(self, prompts: list[Prompt], *, desc: str = "") -> list[Generation]:
        return asyncio.run(self._generate_all(prompts, desc=desc))

    async def _generate_all(self, prompts: list[Prompt], *, desc: str) -> list[Generation]:
        sem = asyncio.Semaphore(max(1, self.cfg.concurrency))
        results: list[Generation | None] = [None] * len(prompts)
        bar = tqdm(total=len(prompts), desc=desc or self.cfg.key, unit="req", leave=False)

        async def worker(i: int, prompt: Prompt) -> None:
            async with sem:
                results[i] = await self._one(prompt)
                bar.update(1)

        try:
            await asyncio.gather(*(worker(i, p) for i, p in enumerate(prompts)))
        finally:
            bar.close()

        return [r or Generation(text="", error="no result") for r in results]

    async def _one(self, prompt: Prompt) -> Generation:
        cfg = self.cfg
        last_err = ""
        for attempt in range(cfg.max_retries + 1):
            started = time.perf_counter()
            try:
                if cfg.prompt_style == "completion" or isinstance(prompt, str):
                    return await self._completion(as_text_prompt(prompt), started)
                return await self._chat(prompt, started)
            except Exception as exc:  # noqa: BLE001 -- classify below
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt >= cfg.max_retries or not _retryable(last_err):
                    break
                # Full jitter, capped, so a cold server gets time to load.
                delay = min(2.0 * (2**attempt), 60.0) * (0.5 + random.random() / 2)  # nosec B311 # nosemgrep -- retry jitter, not a security decision
                await asyncio.sleep(delay)
        return Generation(text="", error=last_err)

    async def _chat(self, prompt: Prompt, started: float) -> Generation:
        cfg = self.cfg
        kwargs: dict = {
            "model": cfg.served_model_name,
            "messages": prompt,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.stop:
            kwargs["stop"] = cfg.stop
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed
        # vLLM/SGLang accept template kwargs (e.g. Qwen3's enable_thinking) via
        # extra_body; servers that don't understand it ignore the field.
        if cfg.chat_template_kwargs:
            kwargs["extra_body"] = {"chat_template_kwargs": cfg.chat_template_kwargs}

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        # Some servers expose reasoning separately; keep it out of the answer but
        # record that it happened.
        reasoning = getattr(choice.message, "reasoning_content", None)
        usage = resp.usage
        return Generation(
            text=text,
            finish_reason=choice.finish_reason or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=time.perf_counter() - started,
            extra={"had_reasoning_channel": bool(reasoning)} if reasoning else {},
        )

    async def _completion(self, text_prompt: str, started: float) -> Generation:
        cfg = self.cfg
        kwargs: dict = {
            "model": cfg.served_model_name,
            "prompt": text_prompt,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.stop:
            kwargs["stop"] = cfg.stop
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed

        resp = await self._client.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = resp.usage
        return Generation(
            text=choice.text or "",
            finish_reason=choice.finish_reason or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=time.perf_counter() - started,
        )

    # ------------------------------------------------------------------ #
    def describe(self) -> dict:
        info = super().describe()
        info["base_url"] = self.cfg.resolved_base_url()
        info["concurrency"] = self.cfg.concurrency
        info.update(self.probe())
        return info

    def probe(self) -> dict:
        """Ask the server what it is serving. Recorded for provenance."""
        try:
            import httpx

            url = self.cfg.resolved_base_url().rstrip("/") + "/models"
            resp = httpx.get(
                url,
                headers={"Authorization": f"Bearer {self.cfg.resolved_api_key()}"},
                timeout=15.0,
            )
            resp.raise_for_status()
            served = [m.get("id") for m in resp.json().get("data", [])]
            return {"server_models": served, "server_reachable": True}
        except Exception as exc:  # noqa: BLE001
            return {"server_reachable": False, "server_probe_error": str(exc)[:200]}

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            # Best effort: the event loop may already be closed at interpreter exit.
            with contextlib.suppress(Exception):
                asyncio.run(client.close())


def _retryable(err: str) -> bool:
    low = err.lower()
    return any(marker in low for marker in _RETRY_MARKERS)
