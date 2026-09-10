# Security posture

What has been hardened, what is deliberately left as-is, and what risk remains.
Written to be read alongside a static-analysis report, because several scanner
findings on this repo are structural rather than exploitable, and a couple of
real risks are *not* things a scanner flags.

## 1. Hardened

### Hub downloads are revision-pinnable (CWE-494)

Every Hugging Face fetch takes a `revision` and forwards it explicitly:

| Call site | Pin source |
|---|---|
| `src/cmb_indic/data.py` (`hf_hub_download`) | `--dataset-revision` |
| `src/cmb_indic/backends/hf_local.py` (`AutoTokenizer`, `AutoModelForCausalLM`) | `hf_revision` in the model config |
| `src/cmb_indic/backends/vllm_offline.py` | `hf_revision` |
| `scripts/check_compat.py` (config.json + tokenizer probe) | `hf_revision` |
| `scripts/build_conditions.py`, `build_gsm8k_conditions.py`, `verify_item_coverage.py`, `check_gsm8k_contamination.py` | `--revision` |
| `utils.py` (vendored, unused by this pipeline) | `revision` argument |

`revision` is passed as a literal keyword argument, never inside a `**kwargs`
dict. That is not cosmetic: a value hidden in `**kwargs` is invisible to static
analysis, so it reads as an unpinned download in every audit.

The defaults are `None`, which resolves the repo's default branch. Pinning is
therefore *available* rather than *mandatory* -- pass a commit sha to get a
byte-reproducible fetch. `run_meta.json` records the sha256 of every dataset CSV
actually used, so a run remains auditable even when it was not pinned.

### No dynamic evaluation (CWE-95)

`eval()` is not called anywhere in the repository. The three vendored call sites
that parsed stringified Python literals now use `ast.literal_eval()`; see
`docs/DEVIATIONS.md` item 11. `exec`, `pickle.load`, `yaml.load` without a safe
loader, `os.system`, and `subprocess(..., shell=True)` are likewise absent.

### Network fetches cannot change scheme (CWE-22 / bandit B310)

`scripts/check_compat.py` and `cmb-indic validate --check-hub` fetch over
`httpx`, which speaks only http/https. A repo id or revision read out of a
config file therefore cannot redirect a fetch to `file:/` or a custom scheme.
This replaced `urllib.request.urlopen`, where the same guarantee needed a
hand-written scheme check that a scanner cannot see.

### Subprocess execution uses an absolute path (CWE-78 / bandit B607)

`src/cmb_indic/runner.py` resolves `git` through `shutil.which()` before
invoking it, so provenance capture cannot pick up a `git` planted earlier on
`PATH`. The call passes an argument list, never a shell string.

### Secrets are never committed

API keys and tokens are read from the environment (`CMB_API_KEY` by default,
`HF_TOKEN` for gated repos); `.env` is gitignored and `.env.example` carries
names only. `ModelConfig.api_key_env` holds the *name* of the variable to read,
not a value -- scanners regularly misreport that field as a hardcoded
credential. The fallback string `"EMPTY"` is the vLLM/SGLang convention for "no
auth required" (their clients reject an empty string); it is not a credential.

## 2. Dependency floors

Fixed by raising a minimum version:

| Package | Floor | Advisory |
|---|---|---|
| scikit-learn | 1.5.0 | CVE-2024-5206 -- `TfidfVectorizer` leaked training tokens via `stop_words_` |
| datasets | 5.0.1 | CVE-2026-66007 -- path traversal via `file_name` in folder-based builders |
| tqdm | 4.66.3 | CVE-2024-34062 -- argument injection in the tqdm CLI |
| transformers | 4.53.0 | ReDoS: CVE-2025-3933, -3777, -5197, -6051, -6638, -6921 |
| sentencepiece | 0.2.1 | CVE-2026-1260 -- heap overflow |
| pytest | 9.0.3 | CVE-2025-71176 -- tmpdir handling |
| black | 26.3.1 | CVE-2024-21503, CVE-2026-31900, CVE-2026-32274 |

`vllm` moved from `0.10.1` to `0.10.1.1`, the patch release for CVE-2025-9141
(RCE in the Qwen3-Coder tool-call parser) and CVE-2025-48956 (DoS).

The vendored `requirements.txt` -- upstream's install manifest, kept for
provenance -- received the same treatment for `scikit_learn`, `datasets` and
`tqdm`.

## 3. Residual risk

**The `vllm` extra is pinned to a version with open advisories.** vLLM 0.10.1.1
is the newest release that still pins `torch==2.7.1+cu126`, i.e. the newest one
that runs on a CUDA 12.x driver. Advisories fixed only in 0.11 and later stay
open on this pin. Most of them are in the OpenAI-compatible *server* and its
media handling; this pipeline drives vLLM in-process (`vllm_offline.py`) and
starts no server, which removes the network-facing ones from the picture. The
model-loading RCEs (CVE-2026-22807 `auto_map`, CVE-2025-66448 `get_config`,
CVE-2026-27893) are reachable and are the reason `trust_remote_code` defaults to
false. On a driver >= 580, relax the `vllm`/`transformers` pins as the comment
in `pyproject.toml` describes and this whole class goes away.

**The `torch` floor stays at 2.3.** vLLM 0.10.1.x pins `torch==2.7.1`, so a
higher floor makes `pip install -e ".[all]"` unresolvable. `pip install -e
".[hf]"` on its own resolves torch to the current release, which is patched. If
you install from a lockfile, pin torch >= 2.10.0 yourself (CVE-2025-32434 and
CVE-2026-24747, both `weights_only` unpickler RCEs).

**transformers RCEs fixed only in 5.x** (CVE-2026-1839, -4372, -5241, -9856)
remain open on the `vllm` extra, which pins `transformers==4.55.2` because 5.x
breaks vLLM 0.10.1 with `rope_scaling should have a 'rope_type' key`. The `hf`
extra carries no such pin and resolves to current transformers, but its floor of
4.53.0 still *permits* a 4.x install for the same `[all]` reason as torch --
raising it to 5.x would make `[all]` unresolvable. If you install from a
lockfile and do not need the in-process vLLM backend, pin transformers >= 5.10.0
and torch >= 2.10.0 yourself.

**`trust_remote_code`.** Defaults to false everywhere and is opt-in per model
config. Turning it on executes code from the model repository by design; only
enable it for a repo you pinned to a reviewed revision.

**Model weights are untrusted input.** Nothing here verifies a checkpoint
beyond what `transformers`/`vllm` do. Pin a revision and prefer safetensors.

**Local endpoints are unauthenticated by default.** `cmb-indic run --backend
openai` talks to whatever `base_url` says. Those servers are meant to be
loopback or a trusted LAN; do not expose one to an untrusted network.

## 4. Scanner findings that are not defects

A static scan of this repo reports ~267 LOW findings. All of them fall into
these classes:

* **`assert` in tests (bandit B101, ~252).** pytest's assertion rewriting is
  how the suite works. `assert` is not used for control flow or validation in
  `src/`.
* **`try/except/pass` and `try/except/continue` (B110/B112, 8).** Optional
  imports, best-effort provenance capture, and per-row error isolation, where a
  failure is either recorded in the run metadata or intentionally skipped. Each
  carries a comment saying why.
* **Non-cryptographic `random` (B311, 2).** `backends/openai_compat.py:91`
  scales exponential backoff with full jitter; `prompts.py:181` seeds
  `random.Random` to draw few-shot exemplars reproducibly (`docs/DEVIATIONS.md`
  item 6). Both need determinism or fairness, not unpredictability, and neither
  feeds a security decision -- a CSPRNG would make the second one worse.
* **"Hardcoded password" (B105, 3).** `registry.py:445` compares against the
  string `"derived"`, a keyword that selects a split group. `utils.py:484` and
  `:487` assign `max_token_key = 'max_tokens'` / `'max_new_tokens'` -- flagged
  only because the variable name ends in `_key`. No credential in any of them.
  Separately, an InfoSec scan reported `config.py`'s `api_key_env` field as a
  hardcoded credential (CWE-798); as described above it holds a variable *name*,
  and `"EMPTY"` is the local-server "no auth" convention.
* **`subprocess` import and call (B404/B603, 2).** The `git` provenance call in
  `runner.py`, with an absolute path and an argument list.

## 5. Reporting

This is a research pipeline, not a service. Open an issue for anything found
here.
