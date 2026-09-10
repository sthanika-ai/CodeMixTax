# Runtime compatibility: driver, vLLM, and the Gemma architectures

Findings from a CPU-only compatibility audit on the development host
(2× A100 80GB PCIe, driver **550.54.15** = CUDA 12.4).

Reproduce any of this yourself:

```bash
python scripts/check_compat.py --pattern gemma      # architecture + chat template
python scripts/check_compat.py --all
```

---

## 1. Verdict: Gemma 3 works, Gemma 4 is blocked by the GPU driver

| Model config | Declared architecture | vLLM support | Runs on driver 550? |
|---|---|---|---|
| `gemma3-4b-it` | `Gemma3ForConditionalGeneration` | ≥ 0.18 | **yes — verified running** |
| `gemma3-12b-it` | `Gemma3ForConditionalGeneration` | ≥ 0.18 | **yes** |
| `gemma3-12b-it-int4` | `Gemma3ForConditionalGeneration` | ≥ 0.18 | **yes** |
| `gemma3-27b-it` | `Gemma3ForConditionalGeneration` | ≥ 0.18 | **yes** |
| `gemma4-12b-it` | `Gemma4UnifiedForConditionalGeneration` | **≥ 0.23 only** | **no — needs driver ≥ 580** |

All five load, tokenize, and render their chat templates correctly, and all five are
registered in vLLM 0.27.1 (367 architectures). The blocker is purely the CUDA
major version, explained below.

## 2. The driver / CUDA-major conflict

```
installed:  vllm 0.27.1  ->  torch 2.13.0+cu130  ->  requires CUDA 13  ->  driver >= 580
host:       driver 550.54.15  =  CUDA 12.4
result:     torch.cuda.is_available() == False   (device_count() still reports 2)
```

`device_count()` returning 2 while `is_available()` is False is the trap here: code
that only checks the count will look fine and then fail at the first kernel launch
with *"The NVIDIA driver on your system is too old (found version 12040)"*.

CUDA **minor** version compatibility means any cu12.x build runs on a 12.x driver.
Crossing to CUDA **13** does not work — it needs a new driver.

Mapping the torch pins (from PyPI metadata):

| vLLM | torch | CUDA major | Works on driver 550 | `Gemma4Unified` |
|---|---|---|---|---|
| 0.27.x, 0.27.0 | 2.13.0 | 13 | no | yes |
| 0.20–0.26 | 2.11.0 | 13 | no | yes (≥ 0.23) |
| **0.19.1** | **2.10.0** | **12.8** | **yes** | **no** |
| 0.18.0 | 2.10.0 | 12.8 | yes | no (no Gemma 4 at all) |

**The conflict is exact and unavoidable:** `Gemma4UnifiedForConditionalGeneration`
first appears in vLLM 0.23.0, and every vLLM from 0.20 onward requires CUDA 13.
There is no vLLM release that supports Gemma 4 *and* runs on a CUDA 12 driver.

## 3. RESOLVED: the working stack on this host

**vLLM 0.10.1 + torch 2.7.1+cu126 + transformers 4.55.2, with `flashinfer` uninstalled.**
Verified end to end: a real GPU matmul succeeds, and `gemma3-4b-it` completed all 18
Indic splits (21,256 rows) in 317 s on an A100 80GB.

```bash
pip install "vllm==0.10.1"
pip install "transformers==4.55.2"    # see below -- 5.x breaks 0.10.1
pip uninstall -y flashinfer-python    # see below -- stale CUDA-13 build
```

Three things had to line up, and each failed loudly before it was fixed:

1. **CUDA major version.** `pip install vllm` (0.27.1) pulls `torch 2.13.0+cu130`,
   which needs CUDA 13 → driver ≥ 580. vLLM 0.10.1 pins `torch 2.7.1+cu126`; CUDA
   12.6 runs fine on a 12.4 driver via minor-version compatibility.

2. **transformers upper bound.** vLLM 0.10.1 declares `transformers>=4.55.0` with
   *no upper bound*, so pip happily installed 5.15.0 — which renamed the rope
   config keys. Symptom:

   ```
   ValueError: rope_scaling should have a 'rope_type' key
   ```

   Pin `transformers==4.55.2`. Consequence: the Gemma 4 tokenizer needs 5.x, so
   Gemma 4 becomes unloadable — irrelevant here, since 0.10.1 lacks its
   architecture anyway.

3. **Stale flashinfer.** Upgrading *down* from 0.27.1 leaves
   `flashinfer-python 0.6.16.post3` behind, built against CUDA 13. It tries to
   JIT-compile sampling kernels and dies:

   ```
   flashinfer/jit/cpp_ext.py -> ninja ... returned non-zero exit status 127
   RuntimeError: Engine core initialization failed.
   ```

   Uninstall it. vLLM 0.10.1 falls back to its native PyTorch sampler and logs
   `FlashInfer is not available. Falling back to the PyTorch-native implementation`.

Also left behind by the downgrade: orphaned `nvidia-*-cu13` packages
(`nvidia-cublas 13.1.1.3`, `nvidia-cuda-runtime 13.0.96`, `nvidia-cudnn-cu13`).
They are inert — torch loads the `-cu12` variants — but they make `pip list`
confusing.

### `torch.cuda.is_available()` is not a reliable check here

It returned **False** in some runs and **True** in others on the *same* broken
stack, depending on when CUDA init was first attempted; `device_count()` reported
2 throughout. The only trustworthy test is launching a real kernel:

```python
import torch; x = torch.randn(64, 64, device="cuda"); torch.cuda.synchronize()
```

### If the driver is upgraded to ≥ 580

Revert to current vLLM (`pip install -U vllm`) and drop the transformers pin. That
restores Gemma 4 support. Nothing else in this repo needs to change.

## 4. Chat template findings (independent of the driver)

**Gemma 3 has no system role.** Its template merges a system message into the first
user turn:

```
<bos><start_of_turn>user
You are a smart and intelligent sentiment analysis (SA) system. ...

Sentence: Mammookka fans like adi; Your final answer:<end_of_turn>
<start_of_turn>model
```

The instruction text survives intact — verified for the system prompt, the label
set, and the user content — so CodeMixBench prompts work unmodified. Worth stating
in the report, since Gemma 3 sees the task instruction as user text while other
models see it as a system message.

**Gemma 4 has a real system role and a thinking channel.** Its canonical template
(dated 2026-07-09) defaults `enable_thinking` to `false` and emits an *empty,
pre-closed* thought block to suppress reasoning:

```
<bos><|turn>system
You are a smart ... <turn|>
<|turn>user
Sentence: ...<turn|>
<|turn>model
<|channel>thought
<channel|>
```

Note the single-sided pipes — `<|channel>` … `<channel|>` — which is a **different
scheme from gpt-oss harmony** (`<|channel|>…<|message|>`). `extract.strip_reasoning`
handles both, plus Gemma 4's `<|turn>` / `<turn|>` scaffolding. If you run Gemma 4
with `chat_template_kwargs: {enable_thinking: true}`, raise the task token budgets
substantially and check `n_truncated`.

## 5. CPU verification that was possible

vLLM's PyPI wheel is CUDA-only: forcing CPU fails in `gpu_worker.py` with
*"DP adjusted local rank 0 is out of bounds for 0 devices"*. CPU inference needs a
source build (`VLLM_TARGET_DEVICE=cpu`), so **no vLLM generation test can run on
CPU here**.

What was verified on CPU instead, end to end through the real pipeline
(`backend: hf`, transformers 5.15.0, tiny random-weight models of each
architecture, on real benchmark splits):

| Architecture | Test model | Loads | Generates | Scored |
|---|---|---|---|---|
| `Gemma3ForConditionalGeneration` | `trl-internal-testing/tiny-Gemma3ForConditionalGeneration` | yes | yes | yes, 0 errors |
| `Gemma4UnifiedForConditionalGeneration` | `optimum-intel-internal-testing/tiny-random-gemma4-unified` | yes | yes | yes, 0 errors |

Exercised on both a sentence-level split (`sa_maleng`) and a token-level split
(`lid_hineng`), so both scoring paths are covered. `unparsed_rate` was 100% in every
case, which is the **correct** result: these are randomly-initialised weights that
emit gibberish, and the pipeline properly recorded that as unparseable rather than
inventing a score. This confirms plumbing, not model quality.

## 6. Environment as installed

```
python        3.12.9
vllm          0.10.1     <- driver-550-compatible; 0.20+ all require CUDA 13
torch         2.7.1+cu126
transformers  4.55.2     <- pinned; 5.x breaks vLLM 0.10.1 config parsing
accelerate    1.14.0     <- required by the hf backend; not pulled by [vllm]
flashinfer    (removed)  <- stale CUDA-13 build blocked engine startup
driver        550.54.15 (CUDA 12.4)
```

`pip install vllm` downgraded numpy 2.5.2 → 2.3.5 and installed transformers 5.15.0.
All 167 tests still pass afterwards.
