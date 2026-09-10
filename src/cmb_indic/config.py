"""Typed configuration: model cards, run settings, and YAML loading.

Layering, lowest precedence first:

    configs/default.yaml  ->  configs/models/<model>.yaml  ->  configs/suites/<suite>.yaml
                          ->  CLI flags

Every resolved value ends up in the run's ``run_meta.json``, so a published
number can always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
MODELS_DIR = CONFIG_DIR / "models"
SUITES_DIR = CONFIG_DIR / "suites"
DEFAULT_FILE = CONFIG_DIR / "default.yaml"


# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Everything needed to generate from one model."""

    key: str                      # short slug, used in output paths
    hf_id: str                    # canonical Hugging Face repo id
    display_name: str = ""
    backend: str = "openai"       # openai | vllm | hf | echo

    # --- OpenAI-compatible server (vLLM, SGLang, Ollama, llama.cpp, LM Studio) ---
    #: Name the server advertises. Defaults to hf_id, which is what vLLM uses.
    served_model_name: str = ""
    base_url: str = ""            # falls back to $CMB_BASE_URL
    api_key_env: str = "CMB_API_KEY"  # nosec B105 # nosemgrep -- names an env var to read, holds no secret itself
    concurrency: int = 16         # in-flight requests
    request_timeout: float = 600.0
    max_retries: int = 5

    # --- local execution (vllm / hf backends) ---
    dtype: str = "auto"
    quantization: str | None = None       # awq | gptq | compressed-tensors | bitsandbytes | None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int | None = None
    #: Pin the checkpoint to an exact commit. Worth recording for any community
    #: requantisation, where the same repo name can be re-uploaded with different
    #: weights -- a silent change to what a published number refers to.
    hf_revision: str | None = None
    #: Cap on concurrent sequences in the vLLM scheduler. Lower values trade
    #: throughput for a smaller KV-cache footprint, which is what makes a large
    #: quantised MoE fit alongside its experts.
    max_num_seqs: int | None = None
    trust_remote_code: bool = False
    load_in_4bit: bool = False            # hf backend only

    # --- prompting / decoding ---
    prompt_style: str = "chat"            # chat | completion (base models)
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int = 1234
    stop: list[str] = field(default_factory=list)
    #: Forwarded to the chat template, e.g. {"enable_thinking": false} for Qwen3.x.
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    #: True for models that emit hidden chain-of-thought; raises the token budget
    #: and enables reasoning stripping diagnostics.
    reasoning: bool = False
    #: Hard cap on tokens spent inside the reasoning block. vLLM injects the
    #: reasoning end token once the cap is hit, so the model is forced out of
    #: thinking and must commit to an answer instead of running to the ceiling
    #: and scoring UNK. Needs ``reasoning_delimiters`` and, on vLLM, the V1 model
    #: runner -- under V2 the parameter is silently IGNORED, so any run using
    #: this must assert the runner from the log rather than trust the default.
    thinking_token_budget: int | None = None
    #: (start, end) strings bounding the reasoning block, e.g. ("<think>",
    #: "</think>"). Required by ``thinking_token_budget``; token ids are derived.
    reasoning_delimiters: list[str] = field(default_factory=list)

    # --- per-task overrides, e.g. {"gsm8k": {"max_tokens": 1024}} ---
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- documentation / provenance (surfaced in the report) ---
    params_b: float | None = None
    architecture: str = ""                # "dense" | "moe" | "moe (2.4B active)"
    vision: bool = False
    license: str = ""
    gated: bool = False
    #: "verified"    -- hf_id confirmed to exist on the Hub
    #: "substituted" -- requested model does not exist; nearest match used
    #: "unverified"  -- not checked
    status: str = "unverified"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.key
        if not self.served_model_name:
            self.served_model_name = self.hf_id
        if self.backend not in ("openai", "vllm", "hf", "echo"):
            raise ValueError(
                f"{self.key}: unknown backend {self.backend!r} "
                "(expected openai | vllm | hf | echo)"
            )
        if self.prompt_style not in ("chat", "completion"):
            raise ValueError(f"{self.key}: prompt_style must be chat|completion")

    def resolved_base_url(self) -> str:
        return self.base_url or os.environ.get("CMB_BASE_URL", "http://localhost:8000/v1")

    def resolved_api_key(self) -> str:
        # api_key_env names the environment variable to read (default "CMB_API_KEY");
        # it never holds a secret value itself. "EMPTY" is the vLLM/SGLang/etc.
        # convention for "no auth needed" -- their clients reject an empty string.
        return os.environ.get(self.api_key_env) or "EMPTY"

    def for_task(self, task: str) -> ModelConfig:
        """Apply this model's per-task overrides."""
        over = self.task_overrides.get(task)
        if not over:
            return self
        unknown = set(over) - {f.name for f in self.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"{self.key}: unknown task_overrides keys for {task}: {sorted(unknown)}")
        return replace(self, **over)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def est_vram_gb(self) -> float | None:
        """Rough weight-only VRAM need. Ignores KV cache and activations."""
        if self.params_b is None:
            return None
        if self.load_in_4bit or (self.quantization or "").lower() in (
            "awq", "gptq", "bitsandbytes", "compressed-tensors", "w4a16", "int4", "nvfp4"
        ):
            per_b = 0.65
        elif self.dtype in ("float32", "fp32"):
            per_b = 4.2
        elif (self.quantization or "").lower() in ("fp8", "w8a8", "int8"):
            per_b = 1.15
        else:
            per_b = 2.1
        return round(self.params_b * per_b, 1)


# --------------------------------------------------------------------------- #
# Run settings
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    """Settings that apply to a whole sweep, independent of the model."""

    datasets: list[str] = field(default_factory=lambda: ["all"])
    shots: int = 0
    limit: int | None = None
    sample: str = "head"            # head | random
    seed: int = 1234
    extraction: str = "robust"      # robust | paper
    prompt_file: str = "prompt.json"
    data_dir: str = "data/raw"
    out_dir: str = "runs"
    run_id: str = ""
    resume: bool = True
    dataset_revision: str | None = None
    save_generations: bool = True
    fail_fast: bool = False
    #: Selectively invalidate cached generations so only those rows regenerate.
    #: Any of: "truncated" (finish_reason == length), "errors", "unparsed".
    #: Everything else is reused, so raising a token budget costs only the rows
    #: that actually hit the ceiling rather than the whole split.
    regenerate: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.extraction not in ("robust", "paper"):
            raise ValueError(f"extraction must be robust|paper, got {self.extraction!r}")
        if self.sample not in ("head", "random"):
            raise ValueError(f"sample must be head|random, got {self.sample!r}")
        if self.shots < 0:
            raise ValueError("shots must be >= 0")
        allowed = {"truncated", "errors", "unparsed"}
        bad = set(self.regenerate) - allowed
        if bad:
            raise ValueError(f"regenerate must be a subset of {sorted(allowed)}, got {sorted(bad)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Hash of the settings that affect scores (not paths or resume flags)."""
        material = {
            "shots": self.shots,
            "limit": self.limit,
            "sample": self.sample,
            "seed": self.seed,
            "extraction": self.extraction,
            "dataset_revision": self.dataset_revision,
        }
        return hashlib.sha256(
            json.dumps(material, sort_keys=True).encode()
        ).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_defaults(path: Path | None = None) -> dict:
    p = path or DEFAULT_FILE
    return _read_yaml(p) if p.exists() else {}


def available_models(models_dir: Path | None = None) -> list[str]:
    d = models_dir or MODELS_DIR
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.yaml") if not p.stem.startswith("_"))


#: Suffixes that mark an argument as a filesystem path rather than a config name.
_YAML_SUFFIXES = (".yaml", ".yml")


def _resolve_config_path(name_or_path: str, directory: Path) -> Path:
    """Interpret an argument as either a config name or a path to a YAML file.

    Tested against ``Path.suffix`` membership rather than truthiness: several model
    names legitimately contain dots (``llama3.1-8b-it``, ``qwen3.6-27b``), and
    ``Path("llama3.1-8b-it").suffix`` is ``".1-8b-it"``, which would otherwise be
    mistaken for a file extension and looked up in the wrong place.
    """
    path = Path(name_or_path)
    if path.suffix.lower() in _YAML_SUFFIXES:
        return path
    return directory / f"{name_or_path}.yaml"


def load_model_config(
    name_or_path: str,
    *,
    models_dir: Path | None = None,
    defaults: dict | None = None,
    overrides: dict | None = None,
) -> ModelConfig:
    """Load one model card by config name (``gemma3-12b-it``) or explicit path."""
    d = models_dir or MODELS_DIR
    path = _resolve_config_path(name_or_path, d)
    if not path.exists():
        raise FileNotFoundError(
            f"No model config at {path}. Available: {available_models(d)}"
        )

    base = dict((defaults or {}).get("model", {}))
    raw = _deep_merge(base, _read_yaml(path))
    raw = _deep_merge(raw, overrides or {})
    raw.setdefault("key", path.stem)

    known = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{path}: unknown model config key(s) {sorted(unknown)}. Known keys: {sorted(known)}"
        )
    return ModelConfig(**raw)


def load_suite(name_or_path: str, *, suites_dir: Path | None = None) -> dict:
    """Load a suite file: a named set of models + run settings."""
    d = suites_dir or SUITES_DIR
    path = _resolve_config_path(name_or_path, d)
    if not path.exists():
        avail = sorted(p.stem for p in d.glob("*.yaml")) if d.exists() else []
        raise FileNotFoundError(f"No suite at {path}. Available: {avail}")
    return _read_yaml(path)


def load_run_config(
    *,
    defaults: dict | None = None,
    suite: dict | None = None,
    overrides: dict | None = None,
) -> RunConfig:
    raw = dict((defaults or {}).get("run", {}))
    raw = _deep_merge(raw, (suite or {}).get("run", {}))
    raw = _deep_merge(raw, {k: v for k, v in (overrides or {}).items() if v is not None})

    known = {f.name for f in RunConfig.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown run config key(s): {sorted(unknown)}. Known: {sorted(known)}")
    return RunConfig(**raw)
