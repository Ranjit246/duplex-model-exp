# What Are We Actually Training?

Short answer: we finetune the **Moshi/Kame real-time speech-to-speech model** (the agent
voice). **Deepgram (ASR)** and the **Qwen oracle LLM** are *not* trained — they are frozen
external services that supply inputs at inference time.

## Trained vs. frozen

| Component | Trained? | Role |
|---|---|---|
| **Moshi LM** — temporal transformer ("tempformer") + depth transformer ("depformer") | ✅ **yes** (`parameters_to_finetune: all`, lr 3e-5) | Consumes the user's audio + its own audio/text streams and generates the **agent's** audio + text in real time, full-duplex |
| **Oracle embedding** (`oracle_emb`) | ✅ **yes** | Learns to ingest the oracle's prediction tokens and let them steer generation |
| **Mimi** (neural audio codec) | ❌ frozen | Audio ↔ discrete tokens |
| **Deepgram** (ASR) | ❌ external service | Speech → text (inference only) |
| **Qwen** (oracle LLM, private API) | ❌ external service | Knowledge / next-utterance prediction |

Finetuned from `kyutai/moshiko` on Hinglish (Hindi + English) call-centre data, modelling
the **agent** channel (`--moshi_speakers B`).

## Why Deepgram and Qwen are not "in" the training

They behave differently in the two phases:

**Training (offline, from the parquet)**
- **ASR:** not used at all — the transcripts were already in the dataset.
- **Oracle:** Qwen's predictions were **pre-generated offline** (`tools.generate_oracle_from_text`)
  and baked into the parquet as token streams (`B_oracle_pred_values`). The model only learns
  to *consume* those tokens via `oracle_emb`.

**Inference (live)**
- **Deepgram** transcribes the user's speech → builds the conversation context →
- **Qwen** generates oracle hints live → fed to the trained Moshi model, which speaks.

So Deepgram/Qwen are **runtime plumbing** that supply inputs; the model is trained to *use*
those inputs, not to reproduce them.

## The point — the KAME "tandem" idea

A small, fast speech model cannot hold all knowledge or reason in real time. KAME is a
**tandem architecture**: a large LLM (the *oracle*) supplies knowledge/predictions
asynchronously, and the speech model is trained to **weave that knowledge into fluent,
low-latency Hinglish speech** as the agent. That is the whole design — *"Tandem Architecture
for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI."*

**In one line:** we train a Hinglish full-duplex *talker* (the agent voice) that defers
*knowledge* to an external LLM oracle. Recent data work (cleaning the ~87%-corrupt oracle
predictions — see [`INFERENCE.md`](./INFERENCE.md)) exists to make that oracle signal
trustworthy so the talker actually learns to use it.
