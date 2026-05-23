<h1 align="center">Kame Finetuning Workflow — Indic / Hinglish Adaptation</h1>

<p align="center">
  <strong>KAME: TANDEM ARCHITECTURE FOR ENHANCING KNOWLEDGE IN REAL-TIME SPEECH-TO-SPEECH CONVERSATIONAL AI</strong>
</p>

<p align="center">
  <a href=".github/workflows/code-quality.yml"><img alt="Checks" src="https://img.shields.io/badge/checks-passing-brightgreen"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.12-blue">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <a href="https://docs.astral.sh/ruff/"><img alt="ruff" src="https://img.shields.io/badge/code%20style-ruff-informational"></a>
  <a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/dep_manager-uv-success"></a>
</p>

<p align="center">
  <a href="https://github.com/SakanaAI/kame">KAME</a> ·
  <a href="https://arxiv.org/abs/2510.02327">Paper</a> ·
  <a href="https://pub.sakana.ai/kame/">Blog post</a>
</p>

Kame is an oracle-enabled extension of Moshi for full-duplex spoken dialogue. This repository provides the preprocessing, finetuning, checkpoint conversion, and inference workflow for Kame, adapted here for **Hinglish** (Hindi + English) Indian customer service calls.

The public preprocessing entry point is a canonical dataset layout built from stereo audio and word-level transcripts:

- `audio/<dialogue_id>.wav`
- `text/<dialogue_id>.json`
- optional: `oracle_raw/<dialogue_id>.json`

This repository also includes a small sample under `data/spokenwoz_sample/{audio,text,oracle_raw}` so you can run the preprocessing steps on a bundled example before using your own data.

## Indic / Hinglish Adaptation Notes

This fork adapts KAME for Hindi–English (Hinglish) customer service data with the following changes:

**Speaker convention:** In this dataset **A = customer, B = agent/assistant** (the reverse of the upstream default where A is Moshi/bot). Pass `--moshi_speakers B` at finetune time so the model trains on the correct channel.

**Audio format:** Source audio files are MP3 content stored with a `.wav` extension (32 kbps, 8 kHz stereo). `tokenize_audio` handles this transparently via a format-fallback in `torchaudio.load`.

**Hinglish oracle prompts:** Oracle predictions are generated in proper Hinglish — English words in Roman script, Hindi/Urdu words in Devanagari script (e.g. `हाँ sir, आपका account check करते हैं`). The conversation context captured by ASR may contain Roman-transliterated Hindi; the prompt instructs the LLM to convert these to Devanagari in its output.

**Local LLM inference:** Oracle generation uses a self-hosted vLLM server instead of OpenAI. Set `--base_url` and `--model` accordingly. For Qwen3-family models with built-in chain-of-thought, omit `--disable_thinking` to keep reasoning enabled (recommended for quality). Ensure `max_model_len` on the vLLM server is at least 32000 when running 32+ parallel workers.

---

## Installation

Python 3.12+ is required.

Install the project dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env && uv sync --python 3.12
```

This repository is intended to be used as a `uv`-managed finetuning workflow
repo rather than as a standalone installable Python package.

If you use Weights & Biases for training logs:

```bash
wandb login
```

## Canonical Dataset Layout

Each stereo wav must contain speaker A in the left channel and speaker B in the right channel. Each transcript JSON file must contain a word-level transcript for both speakers with timestamps aligned to the corresponding wav file.

The canonical `text/*.json` format is:

```json
[
  {"speaker": "A", "word": "hello", "start": 0.46, "end": 1.52},
  {"speaker": "B", "word": "hi", "start": 1.82, "end": 2.04},
  {"speaker": "B", "word": "customer", "start": 2.04, "end": 2.703},
  {"speaker": "B", "word": "service", "start": 2.703, "end": 3.145},
  {"speaker": "B", "word": "how", "start": 3.145, "end": 3.366}
]
```

Each entry contains:

- `speaker`: `"A"` or `"B"`
- `word`: a word or tokenizer unit
- `start`: start time in seconds
- `end`: end time in seconds

If your dataset does not include word-level timestamps, create them first with your preferred alignment pipeline before continuing.

If oracle predictions are already present, `oracle_raw/*.json` should follow this format:

```json
[
  {
    "timestamp_ms": 1500,
    "conversation_context": "A: hello there",
    "prediction": "हाँ sir, आपकी कैसे help करूँ",
    "total_word_count": 2,
    "trigger_word": "there",
    "recent_words": "hello there",
    "current_spoken_ratio": 0.75,
    "channel": 1,
    "hint": "हाँ sir, आपकी कैसे help करूँ"
  }
]
```

The `hint` field is intentionally empty when `current_spoken_ratio <= 0.5`.

## Standard Preprocessing

### 1. Optional: Generate `oracle_raw` From Canonical Text Transcripts

If your dataset does not already include oracle predictions, generate them directly from `text/*.json`.

**Using OpenAI or OpenRouter:**

```bash
export OPENAI_API_KEY=...   # or OPENROUTER_API_KEY=...

uv run --extra oracle -m tools.generate_oracle_from_text \
  --text_dir data/my_dataset/text \
  --output_dir data/my_dataset/oracle_raw \
  --target_channel 1 \
  --B_channel 1 \
  --time_interval 1.0 \
  --workers 8 \
  --resume \
  --fallback_to_hint_on_error
```

**Using a self-hosted vLLM server (e.g. Qwen3):**

```bash
uv run --extra oracle -m tools.generate_oracle_from_text \
  --text_dir data/my_dataset/text \
  --output_dir data/my_dataset/oracle_raw \
  --base_url http://<vllm-host>:8001/v1 \
  --model <model-name> \
  --target_channel 1 \
  --B_channel 1 \
  --time_interval 1.0 \
  --workers 32 \
  --resume \
  --fallback_to_hint_on_error
```

> **Note:** For Qwen3 models, omit `--disable_thinking` to keep chain-of-thought reasoning enabled. Pass `--disable_thinking` only if your vLLM deployment does not support it. Ensure the server is configured with `max_model_len >= 32000` when using 32+ workers to avoid KV cache exhaustion.

The `--target_channel 1` and `--B_channel 1` flags tell the generator that channel 1 (right) is the agent whose responses are predicted. By default `A_channel=0` and `B_channel=1`.

**Watchdog auto-restart** (`run_oracle.sh`) — if the vLLM server stalls, use the included watchdog script which monitors output file count and restarts the process automatically:

```bash
bash run_oracle.sh
```

Edit `ORACLE_DIR`, `TEXT_DIR`, `WORKERS`, and the `CMD` array inside the script to match your setup.

**Quality check** — after generation, verify that no files are pure hint echoes (a sign the LLM server was unreachable and fallback mode triggered):

```python
import json
from pathlib import Path

bad = []
for p in Path("data/my_dataset/oracle_raw").glob("*.json"):
    data = json.loads(p.read_text())
    with_hint = [r for r in data if r.get("hint")]
    if not with_hint:
        continue
    matches = sum(1 for r in with_hint if r.get("prediction", "").strip() == r.get("hint", "").strip())
    if matches / len(with_hint) > 0.8:
        bad.append(p)

print(f"{len(bad)} bad files")
for p in bad:
    p.unlink()
```

### 2. Audio Tokenization

Convert all wav files into discrete Mimi tokens. Files that are MP3 content with a `.wav` extension are handled automatically.

```bash
uv run -m tools.tokenize_audio \
  --audio_dir data/my_dataset/audio \
  --output_dir data/my_dataset/tokenized_audio
```

This creates `data/my_dataset/tokenized_audio/*.npz`. Each npz file contains audio tokens for A and B.

**On a GPU server** with limited VRAM (e.g. 6 GB), use a smaller chunk size to avoid OOM:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m tools.tokenize_audio \
  --audio_dir data/my_dataset/audio \
  --output_dir data/my_dataset/tokenized_audio \
  --audio_chunk_size 10 \
  --num_workers 1 \
  --resume
```

### 3. Text Tokenization

Convert the canonical transcripts into frame-level text streams. Steps 2 and 3 are independent and can run in parallel.

```bash
uv run -m tools.tokenize_text \
  --word_transcript_dir data/my_dataset/text \
  --output_dir data/my_dataset/tokenized_text \
  --allow_missing_speakers \
  --allow_alignment_warnings
```

This creates `data/my_dataset/tokenized_text/*.npz`. Each npz file contains text tokens for A and B.

- `--allow_missing_speakers`: skip dialogues where one speaker is absent rather than crashing.
- `--allow_alignment_warnings`: warn and continue when a few tokens extend slightly beyond the audio duration (common with ASR timestamps), rather than crashing.

If you use a different tokenizer, also pass `--text_tokenizer_repo`, `--text_tokenizer_name`, and the matching padding IDs.

### 4. Optional: Oracle Tokenization

If `oracle_raw/*.json` is present, convert it into event-level oracle records:

```bash
uv run -m tools.tokenize_oracle \
  --oracle_dir data/my_dataset/oracle_raw \
  --oracle_suffix ".json" \
  --tokenized_audio_dir data/my_dataset/tokenized_audio \
  --output_dir data/my_dataset/tokenized_oracle_a0b1_events \
  --A_channel 0 \
  --B_channel 1
```

This creates `data/my_dataset/tokenized_oracle_a0b1_events/*.npz`.

### 5. Build Parquet Files

Concatenate audio, text, and optional oracle streams into a parquet dataset ready for finetuning:

```bash
uv run -m tools.prepare_dataset \
  --tokenized_text_dir data/my_dataset/tokenized_text \
  --tokenized_audio_dir data/my_dataset/tokenized_audio \
  --tokenized_oracle_dir data/my_dataset/tokenized_oracle_a0b1_events \
  --output_prefix processed_data/my_dataset/train_text_oracle_a0b1_events
```

If you do not use oracle predictions, omit `--tokenized_oracle_dir`.

## Model Initialization

The example scripts in this README assume a KAME finetuning model initialized
from the original Kyutai weights. The current English example scripts use:

- `init_models/moshiko-one_streams-bfloat16`

To initialize from the original Kyutai weights:

```bash
uv run -m tools.init_moshi_for_ft \
  --moshi_lm_repo kyutai/moshiko-pytorch-bf16 \
  --save_dir init_models/moshiko-one_streams-bfloat16 \
  --model_dtype bfloat16
```

If you change the text tokenizer, also use `--init_text_embeddings` and keep the vocabulary size compatible.

## Training

For a low-memory smoke test, run:

```bash
MAX_TRAIN_STEPS=3 bash examples/finetune_accelerate_cpu_offload.sh
```

This smoke example keeps the default finetuning target but uses a more conservative DeepSpeed configuration with CPU offload. It is intentionally slower, but is a better fit for validating that the public workflow runs end to end on a single GPU.

For a fuller training run, use:

```bash
bash examples/finetune_accelerate.sh
```

This reference script uses the default KAME finetuning target and a faster DeepSpeed configuration, but it may require substantial GPU memory. In practice, full finetuning may need multi-GPU execution depending on your hardware.

> **Indic / Hinglish note:** Because B is the agent in this dataset (not A), pass `--moshi_speakers B` to the finetune script so the model trains on the correct channel.

The current training implementation requires DeepSpeed, so both examples use Accelerate with a DeepSpeed config. On managed clusters you may wrap these commands in your own scheduler submission flow such as `sbatch`, but scheduler-specific scripts are intentionally omitted from this public repository.

## Convert and Clean Checkpoints for Inference

After training, convert checkpoints in two stages:

### 1. Convert DeepSpeed checkpoints to fp32 safetensors

```bash
uv run -m tools.zero_to_fp32 \
  output/moshiko-finetuned/step_10000 \
  output/moshiko-finetuned/step_10000_fp32 \
  --moshi_lm_kwargs_path init_models/moshiko-one_streams-bfloat16/moshi_lm_kwargs.json
```

### 2. Clean the finetuning model for inference

```bash
uv run -m tools.clean_moshi \
  --moshi_ft_dir output/moshiko-finetuned/step_10000_fp32 \
  --save_dir output/moshiko-finetuned/step_10000_fp32_cleaned \
  --model_dtype float32
```

## Oracle-Enabled Inference

Start the oracle-enabled server:

```bash
CLEAN_DIR=output/moshiko-finetuned/step_10000_fp32_cleaned
TOKENIZER_PATH=/path/to/tokenizer_spm_32k_3.model

uv run -m kame.server_oracle \
  --moshi-weight "$CLEAN_DIR/model.safetensors" \
  --config-path "$CLEAN_DIR/moshi_lm_kwargs.json" \
  --tokenizer "$TOKENIZER_PATH" \
  --host 0.0.0.0 \
  --port 8998
```

If the server runs on a remote node, use SSH port forwarding from your local machine:

```bash
ssh -L 8998:localhost:8998 user@remote-host
```

## License

This repository is provided under the [Apache 2.0 License](LICENSE), following the license of the upstream [moshi-finetune](https://github.com/nu-dialogue/moshi-finetune) repository. The SpokenWOZ sample data included in `data/spokenwoz_sample` is provided under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Attribution

`kame_finetune` is derived from the `moshi-finetune` codebase and adapted for
the KAME training workflow.

## Citation

If you use KAME or this finetuning workflow in your research, please cite:

```bibtex
@article{kuroki2025kame,
  title={KAME: Tandem Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI},
  author={Kuroki, So and Kubo, Yotaro and Akiba, Takuya and Tang, Yujin},
  journal={arXiv preprint arXiv:2510.02327},
  year={2025}
}
```
