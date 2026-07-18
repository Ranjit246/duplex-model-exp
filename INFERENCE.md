# Inference Setup — Hinglish Kame (oracle-enabled)

This document covers running **live oracle-enabled inference** on a finetuned Kame
checkpoint (e.g. `Ranjit/moshiko-kame-hinglish-ft-exp`), including the changes made
to the upstream `kame.server_oracle` server for this project.

> TL;DR: prepare a cleaned checkpoint, apply the three `kame.server_oracle` patches,
> then `bash run_inference.sh`. Connect over `localhost` (secure context) — mic won't
> work over a plain-HTTP public IP.

---

## 1. Prepare the checkpoint for inference

The HF checkpoints are raw DeepSpeed training states. For **inference only** you do
**not** need the optimizer states (~94 GB) — just the 15.6 GB model weights.

```bash
# 1) download only the model weights of the desired step (skip the optimizer parts)
uv run huggingface-cli download Ranjit/moshiko-kame-hinglish-ft-exp \
  step_12000/pytorch_model/mp_rank_00_model_states.pt \
  --local-dir output/moshiko-finetuned

# 2) architecture config (no 15 GB base-model download needed — read it from the kame pkg)
uv run python - <<'PY'
import json, os
from kame.models import loaders
os.makedirs("init_models/moshiko-one_streams-bfloat16", exist_ok=True)
json.dump(dict(loaders._lm_kwargs),
          open("init_models/moshiko-one_streams-bfloat16/moshi_lm_kwargs.json","w"), indent=4)
PY

# 3) extract the bf16 module weights -> model.safetensors (+ copy the kwargs)
uv run python - <<'PY'
import torch, os, shutil
from safetensors.torch import save_file
ck = torch.load("output/moshiko-finetuned/step_12000/pytorch_model/mp_rank_00_model_states.pt",
                map_location="cpu", weights_only=False)
sd = {k: v.contiguous() for k, v in ck["module"].items()}
out = "output/moshiko-finetuned/step_12000_ft"; os.makedirs(out, exist_ok=True)
save_file(sd, f"{out}/model.safetensors", metadata={"format": "pt"})
shutil.copy2("init_models/moshiko-one_streams-bfloat16/moshi_lm_kwargs.json", f"{out}/moshi_lm_kwargs.json")
PY

# 4) clean into an inference-ready model (strips finetuning-only modules)
uv run -m tools.clean_moshi \
  --moshi_ft_dir output/moshiko-finetuned/step_12000_ft \
  --save_dir     output/moshiko-finetuned/step_12000_cleaned \
  --model_dtype bfloat16

# 5) tokenizer (small)
uv run huggingface-cli download kyutai/moshiko-pytorch-bf16 tokenizer_spm_32k_3.model \
  --local-dir init_models
```

Result: `output/moshiko-finetuned/step_12000_cleaned/{model.safetensors, moshi_lm_kwargs.json}`.

---

## 2. Server patches (`kame.server_oracle`)

These modify the **installed** package at
`.venv/lib/python3.12/site-packages/kame/server_oracle.py`.

> ⚠️ **These live in `.venv` and are reverted by `uv sync`.** Re-apply them after any
> dependency re-sync. (Consider vendoring a patch file if this becomes frequent.)

1. **Oracle model is env-driven** — the hardcoded `model="gpt-4.1"` now reads
   `ORACLE_MODEL` (so it can point at a self-hosted vLLM/OpenAI-compatible model):
   ```python
   model=os.getenv("ORACLE_MODEL", "gpt-4.1"),
   ```

2. **Devanagari-aligned oracle `SYSTEM_PROMPT`** — the generic English Moshi prompt was
   replaced with one that keeps English in Roman and writes Hindi/Urdu words in
   **Devanagari**, matching how the training oracle predictions were generated.

3. **Pluggable ASR provider (Deepgram added)** — `AsyncASRProcessor` gained a
   `deepgram` provider alongside Google. Selected via env:
   - `ASR_PROVIDER=auto` (default) → uses Deepgram when `DEEPGRAM_API_KEY` is set, else Google.
   - `DEEPGRAM_MODEL` (default `nova-3`), `DEEPGRAM_LANGUAGE` (default `multi`, for
     Hindi+English code-switching).
   - Streams the existing 16 kHz PCM over Deepgram's realtime WebSocket via `aiohttp`
     (no new dependency); interim results → oracle trigger, `is_final` → committed transcript.
   - Google Cloud Speech remains the fallback (needs `GOOGLE_APPLICATION_CREDENTIALS`;
     ASR just stays disabled if neither is configured).

---

## 3. Launch — `run_inference.sh`

Secrets are **env-only** (never commit them):

```bash
export OPENAI_BASE_URL="https://<your-openai-compatible-endpoint>/v1"
export OPENAI_API_KEY="<oracle-llm-key>"
export ORACLE_MODEL="<served-model-name>"     # e.g. a Qwen3 model
export DEEPGRAM_API_KEY="<deepgram-key>"       # optional; enables Deepgram ASR

bash run_inference.sh
```

Overridable env: `CLEAN_DIR`, `TOKENIZER_PATH`, `HOST`, `PORT` (default **8001**),
`ASR_PROVIDER`, `DEEPGRAM_MODEL`, `DEEPGRAM_LANGUAGE`, `LOG_DIR`.

**File logging** (both stream live):
- `logs/inference_server.log` — server output incl. `[LLM]` oracle-hint tokens
- `logs/inference_sessions/` — per-session conversation + token transcripts (`--log-dir`)

---

## 4. Connecting (mic requires a secure context)

Browsers only expose the microphone over **HTTPS** or **`localhost`**. A plain-HTTP
public IP (`http://<ip>:8001`) will load the page but **Connect fails** (no mic).

Use an SSH tunnel from your machine:

```bash
ssh -L 8001:localhost:8001 <user>@<host>
# then open http://localhost:8001
```

(Alternatively run the server with `--ssl <dir-with-cert.pem-key.pem>` for HTTPS.)

---

## 5. Oracle training-data cleaning

The oracle predictions in `Ranjit/hinglish-moshi-kame-ip-data` were found to be
**~87% corrupted** (failed LLM generation → token-soup like `Only Text {}`), which
taught the model to ignore the oracle. A cleaned parquet was produced by keeping the
good Devanagari predictions and replacing the rest with each event's **hint** (the ASR
of the real next utterance, verified clean), zeroing hint-less events:

- Output: `Ranjit/hinglish-moshi-kame-ip-data` →
  `train_text_oracle_a0b1_events_cleaned-001-of-001.parquet`
- 0 garbage remaining, schema identical to the original (drop-in for training).

Point `TRAIN_DATA_GLOB` at the `_cleaned` file when retraining.

---

## 6. Known limitations of the current checkpoint

- **step_12000 ≈ 37%** of the planned 3-epoch (32,502-step) run — early.
- The model's **text stream is Roman/Latin** by data design (0% Devanagari), so spoken
  *text* output is Roman Hinglish / English, never Devanagari.
- With the previously-corrupt oracle, the model largely **ignores oracle hints**;
  retraining on the cleaned data + more steps is the expected fix.
