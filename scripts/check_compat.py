#!/usr/bin/env python
"""CPU-only compatibility check for model configs against vLLM and transformers.

Answers "will this model actually load and prompt correctly?" without allocating a
single byte of GPU memory. Three independent checks per model:

  1. ARCHITECTURE  Read `architectures` from the Hub's config.json (a few KB, no
                   weights) and look it up in vLLM's model registry. This is the
                   definitive answer to "does vLLM support this model".
  2. TOKENIZER     Download the tokenizer and render one of our real benchmark
                   prompts through the model's own chat template. Catches missing
                   templates, base-vs-instruct mixups, and template kwargs that the
                   model rejects -- all of which otherwise surface only mid-sweep.
  3. GATING        Confirm the repo is readable with the current HF credentials.

Usage:
    python scripts/check_compat.py --models gemma3-4b-it,gemma3-12b-it
    python scripts/check_compat.py --pattern gemma      # all Gemma configs
    python scripts/check_compat.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmb_indic.config import available_models, load_defaults, load_model_config  # noqa: E402


def hf_token() -> str:
    import os

    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    cached = Path.home() / ".cache/huggingface/token"
    return cached.read_text().strip() if cached.exists() else ""


def fetch_config(repo: str, token: str, revision: str | None = None) -> tuple[dict | None, str]:
    """Fetch config.json only -- kilobytes, never the weights.

    Goes through httpx, which speaks only http/https, so neither `repo` nor
    `revision` can steer this into file:/ or another scheme. `revision` is the
    ref the model config pins, so the architecture checked here is the
    architecture the run will actually load.
    """
    ref = revision or "main"
    url = f"https://huggingface.co/{repo}/resolve/{ref}/config.json"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = httpx.get(url, headers=headers, timeout=45, follow_redirects=True)
        if resp.status_code in (401, 403):
            return None, f"gated (HTTP {resp.status_code}) -- accept the licence on the model page"
        resp.raise_for_status()
        return resp.json(), ""
    except httpx.HTTPStatusError as exc:
        return None, f"HTTP {exc.response.status_code}"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)[:80]


def vllm_registry() -> tuple[set[str], str]:
    """Architecture names vLLM can serve, plus the vLLM version string."""
    try:
        import vllm
    except ImportError:
        return set(), ""
    version = getattr(vllm, "__version__", "unknown")
    names: set[str] = set()
    # The registry module has moved between releases; try the known locations.
    try:
        from vllm.model_executor.models import ModelRegistry

        names = set(ModelRegistry.get_supported_archs())
    except Exception:  # noqa: BLE001
        try:
            from vllm.model_executor.models.registry import ModelRegistry

            names = set(ModelRegistry.get_supported_archs())
        except Exception:  # noqa: BLE001 # nosec B110 -- registry moved across vLLM releases, not security-relevant
            pass
    return names, version


def check_tokenizer(repo: str, cfg, token: str) -> tuple[str, str]:
    """Render a real benchmark prompt through the model's own chat template."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return "skip", "transformers not installed"

    try:
        tok = AutoTokenizer.from_pretrained(
            repo,
            revision=cfg.hf_revision,
            token=token or None,
            trust_remote_code=cfg.trust_remote_code,
        )
    except Exception as exc:  # noqa: BLE001
        return "FAIL", f"tokenizer load: {type(exc).__name__}: {str(exc)[:70]}"

    messages = [
        {"role": "system", "content": "You are a smart sentiment analysis (SA) system."},
        {"role": "user", "content": "Sentence: Mammookka fans like adi; Your final answer:"},
    ]
    if cfg.prompt_style == "completion" or tok.chat_template is None:
        if cfg.prompt_style != "completion":
            return "WARN", "no chat template (base model?) -- set prompt_style: completion"
        return "ok", "completion style, no template needed"

    try:
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **(cfg.chat_template_kwargs or {}),
        )
    except Exception as exc:  # noqa: BLE001
        return "FAIL", f"chat template: {type(exc).__name__}: {str(exc)[:70]}"

    n = len(tok(rendered)["input_ids"])
    return "ok", f"{n} tokens; starts {rendered[:34]!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", help="comma-separated config names")
    ap.add_argument("--pattern", help="substring filter over config names")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    names = available_models()
    if args.models:
        names = [n.strip() for n in args.models.split(",")]
    elif args.pattern:
        names = [n for n in names if args.pattern.lower() in n.lower()]
    elif not args.all:
        ap.error("pass --models, --pattern, or --all")

    token = hf_token()
    archs, vllm_version = vllm_registry()
    print(f"vLLM: {vllm_version or 'NOT INSTALLED'}"
          + (f" ({len(archs)} architectures registered)" if archs else ""))
    print(f"HF token: {'present' if token else 'MISSING (gated repos will fail)'}")
    print()

    defaults = load_defaults()
    rows = []
    for name in names:
        try:
            cfg = load_model_config(name, defaults=defaults)
        except Exception as exc:  # noqa: BLE001
            rows.append((name, "?", "CONFIG-FAIL", str(exc)[:60], "", ""))
            continue

        hub_cfg, err = fetch_config(cfg.hf_id, token, cfg.hf_revision)
        if hub_cfg is None:
            rows.append((name, cfg.hf_id, "NO-ACCESS", err, "", ""))
            continue

        arch_list = hub_cfg.get("architectures") or []
        arch = arch_list[0] if arch_list else "?"

        if not archs:
            verdict, detail = "UNKNOWN", "vLLM not installed; cannot check registry"
        elif arch in archs:
            verdict, detail = "SUPPORTED", ""
        else:
            verdict, detail = "UNSUPPORTED", f"{arch} not in this vLLM's registry"

        tok_status, tok_detail = check_tokenizer(cfg.hf_id, cfg, token)
        rows.append((name, cfg.hf_id, verdict, detail, arch, f"{tok_status}: {tok_detail}"))

    print(f"{'config':<24}{'vLLM':<13}{'architecture':<40}tokenizer / chat template")
    print("-" * 124)
    bad = 0
    for name, _hf, verdict, detail, arch, tok in rows:
        if verdict not in ("SUPPORTED",) or tok.startswith("FAIL"):
            bad += 1
        print(f"{name:<24}{verdict:<13}{arch:<40}{tok}")
        if detail:
            print(f"{'':<24}{'':<13}-> {detail}")

    print()
    print(f"{len(rows) - bad}/{len(rows)} fully compatible.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
