# Moshi Speech-to-Speech — Instrumentation & Experiments

## Overview

This repo contains instrumentation, logging, and tooling built around [Kyutai's Moshi](https://github.com/kyutai-labs/moshi) — a full-duplex speech-to-speech model. The goal was to understand the internal pipeline (STT encode → LLM inference → TTS decode), add observability, and run offline experiments.

## What was done

### 1. Environment Setup (`deploy.sh`)

Automated deployment script that handles the full setup:

- Checks for Python 3.12 (recommended), installs from source if missing
- Creates a virtualenv at `/workspace/venv`
- Installs moshi from **local source** (`pip install -e`) so code changes take effect without reinstalling
- Installs `gradio-webrtc` for tunnel support
- Creates timestamped log files in `/workspace/logs/`
- Starts the moshi server with gradio tunnel

```bash
bash deploy.sh
```

### 2. Per-Stage Logging (`moshi/moshi/moshi/server.py`)

Added detailed timing and data logging to the moshi server's processing loop. Every audio frame now logs:

| Tag | What it logs |
|-----|-------------|
| `[RECV]` | Timestamp, audio chunk duration, sample count |
| `[STT ]` | Encode time, codes shape, **actual codebook values** |
| `[LLM ]` | Inference time, whether tokens were produced |
| `[TTS ]` | Decode time, generated audio duration |
| `[SEND]` | Timestamp + opus bytes (audio) or text piece (text) |
| `[TOTAL]` | End-to-end frame processing time |

Example output:
```
[Info] [RECV] audio chunk at 07:11:05 | 80ms of audio (1920 samples)
[Info] [STT ] encode done in 5.3ms | codes shape [1, 8, 1] | codes [412, 1023, 87, 556, 201, 334, 89, 712]
[Info] [LLM ] step in 1.9ms | tokens shape [1, 9, 1]
[Info] [TTS ] decode done in 1.8ms | 80ms of audio (1920 samples)
[Info] [SEND] audio at 07:11:05 | 302 opus bytes
[Info] [SEND] text at 07:11:05 | ' Hello'
[Info] [TOTAL] frame handled in 42.4ms
```

### 3. Offline File Inference (`run_audio.py`)

Standalone script that takes a single audio file, runs it through the full Moshi pipeline, and saves the audio response — with per-frame logging.

```bash
python3 run_audio.py input.wav -o output.wav
```

**Key feature: silence padding.** Moshi is a duplex model — it speaks and listens simultaneously. Without padding, it talks over your input. The script prepends 3s of silence (so Moshi finishes its greeting) and appends 5s (so Moshi finishes responding).

```
[3s silence] → Moshi greets into silence
[your audio] → Moshi hears and responds to your question
[5s silence] → Moshi finishes its answer
```

Options:
```bash
python3 run_audio.py input.wav -o output.wav --pre-silence 4 --post-silence 10
python3 run_audio.py input.wav -o output.wav --pre-silence 0  # skip greeting
```

## Experiments & Findings

### Timing Analysis (from server logs)

| Stage | Avg | Min | Max |
|-------|-----|-----|-----|
| STT (Mimi encode) | 5.5ms | 5.4ms | 6.9ms |
| LLM (inference step) | 1.9ms | 1.9ms | 3.5ms |
| TTS (Mimi decode) | 1.9ms | 1.9ms | 2.2ms |
| **Total per frame** | **42.8ms** | **19.2ms** | **77.4ms** |

Actual processing (STT+LLM+TTS) takes ~9ms per frame. The remaining ~34ms is streaming loop overhead (frame buffering, WebSocket I/O, opus encoding).

### Experiment: Offline Q&A

**Input:** "Tell me about India" (voice recording)

**Without silence padding (log2):**
> "Hello, how can I help you? Sure, Nidaa is a town in the Łódź Voivodeship of Poland..."

Moshi talked over the question and hallucinated an unrelated answer.

**With 3s pre-silence + 5s post-silence (log3):**
> "Good day, how is it going? India is a country in South Asia that is known for its diverse culture, rich history, and unique architecture."

Correct response — the silence gave Moshi proper conversational turn structure.

### Key Insight: Moshi is Duplex

Moshi is **not** a listen-then-respond model. It processes every 80ms frame and generates audio+text output simultaneously while consuming input. This means:

- In **live server mode**: works naturally as a conversation — both parties talk and listen
- In **file mode**: you must simulate turn-taking with silence padding
- There is **no STT transcript of user input** — Moshi encodes audio to discrete codes (not text). To transcribe user speech, you'd need a separate STT model (e.g., Whisper)

## File Structure

```
/workspace/
├── deploy.sh              # Automated setup & server launch
├── run_audio.py           # Offline single-file inference with logging
├── output.wav             # Generated audio response
├── logs/
│   ├── log1               # Server log — basic frame timing (before per-stage logging)
│   ├── log2               # Server log — with per-stage logging (STT/LLM/TTS/SEND)
│   └── log3               # Offline inference — with silence padding fix
└── moshi/                 # Moshi source (local, editable install)
    └── moshi/moshi/
        └── server.py      # Modified — added per-stage logging
```

## Model

Using `kyutai/moshika-pytorch-bf16` (Moshi 7B) on CUDA with bfloat16.
