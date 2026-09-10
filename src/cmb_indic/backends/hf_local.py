"""Hugging Face transformers backend.

Slower than vLLM, but it is the pragmatic choice in three situations:

  * bitsandbytes INT4/NF4 quantisation of a model with no pre-quantised AWQ/GPTQ
    checkpoint on the Hub,
  * architectures a pinned vLLM release does not support yet,
  * small models on a single GPU where load time dominates anyway.

Prompts are rendered with the model's own chat template, so instruction
formatting matches what the model was trained on.
"""

from __future__ import annotations

import time

from tqdm.auto import tqdm

from ..config import ModelConfig
from .base import Backend, Generation, Prompt, as_text_prompt


class HFLocalBackend(Backend):
    """Batched greedy/sampled generation with transformers."""

    name = "hf-local"

    def __init__(self, cfg: ModelConfig, *, batch_size: int = 8) -> None:
        super().__init__(cfg)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The hf backend needs torch + transformers: pip install 'cmb-indic[hf]'"
            ) from exc

        self._torch = torch
        self.batch_size = batch_size

        self._tok = AutoTokenizer.from_pretrained(
            cfg.hf_id,
            revision=cfg.hf_revision,
            trust_remote_code=cfg.trust_remote_code,
            padding_side="left",
        )
        if self._tok.pad_token is None:
            # Common for base models (e.g. sarvam-1); reuse EOS for padding.
            self._tok.pad_token = self._tok.eos_token

        model_kwargs: dict = {
            "trust_remote_code": cfg.trust_remote_code,
            "device_map": "auto",
        }
        if cfg.dtype not in ("auto", ""):
            model_kwargs["torch_dtype"] = getattr(torch, cfg.dtype, "auto")
        else:
            model_kwargs["torch_dtype"] = "auto"

        if cfg.load_in_4bit or (cfg.quantization or "").lower() == "bitsandbytes":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_id, revision=cfg.hf_revision, **model_kwargs
        )
        self._model.eval()
        torch.manual_seed(cfg.seed)

    # ------------------------------------------------------------------ #
    def _render(self, prompt: Prompt) -> str:
        if self.cfg.prompt_style == "completion" or isinstance(prompt, str):
            return as_text_prompt(prompt)
        if self._tok.chat_template is None:
            # No template (base model configured as chat by mistake): flatten.
            return as_text_prompt(prompt)
        return self._tok.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            **(self.cfg.chat_template_kwargs or {}),
        )

    def generate(self, prompts: list[Prompt], *, desc: str = "") -> list[Generation]:
        torch = self._torch
        rendered = [self._render(p) for p in prompts]
        results: list[Generation] = []

        bar = tqdm(total=len(rendered), desc=desc or self.cfg.key, unit="row", leave=False)
        try:
            for start in range(0, len(rendered), self.batch_size):
                chunk = rendered[start : start + self.batch_size]
                t0 = time.perf_counter()
                enc = self._tok(
                    chunk, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.cfg.max_model_len or 4096,
                )
                # Several trust_remote_code models declare a forward() that does not
                # accept token_type_ids, and transformers raises rather than ignoring
                # them: "The following `model_kwargs` are not used by the model".
                enc.pop("token_type_ids", None)
                enc = enc.to(self._model.device)

                with torch.inference_mode():
                    out = self._model.generate(
                        **enc,
                        max_new_tokens=self.cfg.max_tokens,
                        do_sample=self.cfg.temperature > 0,
                        temperature=self.cfg.temperature if self.cfg.temperature > 0 else None,
                        top_p=self.cfg.top_p if self.cfg.temperature > 0 else None,
                        pad_token_id=self._tok.pad_token_id,
                    )

                elapsed = (time.perf_counter() - t0) / max(1, len(chunk))
                prompt_len = enc["input_ids"].shape[1]
                for row in range(len(chunk)):
                    new_ids = out[row][prompt_len:]
                    text = self._tok.decode(new_ids, skip_special_tokens=True)
                    results.append(
                        Generation(
                            text=text,
                            finish_reason="length" if len(new_ids) >= self.cfg.max_tokens else "stop",
                            prompt_tokens=int(prompt_len),
                            completion_tokens=int(len(new_ids)),
                            latency_s=elapsed,
                        )
                    )
                bar.update(len(chunk))
        finally:
            bar.close()
        return results

    def describe(self) -> dict:
        info = super().describe()
        info["batch_size"] = self.batch_size
        info["load_in_4bit"] = self.cfg.load_in_4bit
        try:
            import torch
            import transformers

            info["transformers_version"] = transformers.__version__
            info["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 # nosec B110 -- version/GPU probe, not security-relevant
            pass
        return info

    def close(self) -> None:
        if getattr(self, "_model", None) is None:
            return
        try:
            del self._model
            import gc

            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 # nosec B110 -- best-effort GPU cleanup, not security-relevant
            pass
