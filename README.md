# Workspace Notes

Working notes on the PersonaPlex deployment and how it relates to vanilla Moshi.

## Changes made

### `personaplex/deploy.sh`
Rewrote to mirror the structure of `duplex-model-exp/deploy.sh`:
- Installs `libopus-dev` (required by `sphn`)
- Uses the container's existing `python3` (PPA install was stripped out — the container's Python works, and `add-apt-repository` / `dirmngr` weren't available in the minimal image)
- Creates a venv at `/workspace/venv`
- Installs `moshi-personaplex` (editable) from `/workspace/personaplex/moshi`
- Exports `HF_TOKEN` for model download
- Starts `python -m moshi.server --ssl <tmpdir>` and tees output to `/workspace/logs/moshi_<timestamp>.log`

**Note**: server runs HTTPS on port 8998 because of `--ssl`. Open `https://localhost:8998/` (not `http://`) and accept the self-signed cert.

### `personaplex/moshi/moshi/server.py`
Added duplex-style per-frame logs so we can see timing for every stage of a streaming turn:

| Tag | Where | Meaning |
|-----|-------|---------|
| `[RECV] opus` | `recv_loop` | incoming opus payload size |
| `[RECV] audio chunk` | `opus_loop` | assembled PCM chunk (ms + samples) |
| `[STT ] encode done` | `opus_loop` | Mimi encode time, codes shape, codes |
| `[LLM ] step` | `opus_loop` | `lm_gen.step()` time, token shape (or "no tokens yet") |
| `[TTS ] decode done` | `opus_loop` | Mimi decode time, audio duration |
| `[SEND] text` | `opus_loop` | text token emitted to client |
| `[TOTAL] frame` | `opus_loop` | wall time per audio frame |
| `[SEND] audio` | `send_loop` | opus bytes flushed to client |

All logs use `clog` (`ColorizedLog.randomize()`) so each websocket session gets its own color.

---

## PersonaPlex vs vanilla Moshi

README says PersonaPlex is "a finetune of Moshi". That describes the weights — the **code** is a real architectural extension, not just a checkpoint swap.

### Added

**Voice prompt conditioning** — [personaplex/moshi/moshi/models/lm.py:960-1127](personaplex/moshi/moshi/models/lm.py#L960)
- `load_voice_prompt()` — loads a WAV, normalizes to −24 LUFS
- `load_voice_prompt_embeddings()` — loads pre-computed `.pt` embeddings
- `_step_voice_prompt_core()` — encodes audio through Mimi and streams it through `LMGen` before the conversation starts
- `step_system_prompts()` — orchestrates voice → silence → text → silence warmup

**Text persona injection** — [personaplex/moshi/moshi/server.py:79-86](personaplex/moshi/moshi/server.py#L79)
- `wrap_with_system_tags()` wraps the text prompt in `<system>…<system>`
- `text_prompt_tokens` are fed through the LM during the same warmup pass

**Multi-stream LM step** — [personaplex/moshi/moshi/models/lm.py:727-812](personaplex/moshi/moshi/models/lm.py#L727)
- `step()` takes three parallel streams: user audio, agent audio feedback, text token
- Depformer accepts an `audio_provided` mask to force specific codebook values

**Dual Mimi encoder** — [personaplex/moshi/moshi/server.py:92](personaplex/moshi/moshi/server.py#L92)
- `self.mimi` and `self.other_mimi` both encode every input chunk and both decode every output frame

**Server API surface**
- New query params on `/api/chat`: `voice_prompt`, `text_prompt`, `seed`
- New CLI flags: `--voice-prompt-dir`, `--cpu-offload`

**CUDA-graphed Mimi** — [personaplex/moshi/moshi/models/compression.py](personaplex/moshi/moshi/models/compression.py)
- `_MimiState` adds `graphed_encoder` / `graphed_decoder` for streaming

### Removed

**Classifier-free guidance (CFG)**
Vanilla Moshi uses `ConditionProvider` / `ConditionFuser` with a `cfg_coef` for guidance. PersonaPlex drops the whole `conditioners/` module and `cfg_coef` from `LMGen`. Persona control is delivered through voice + text prompts instead of learned guidance.

### Bottom line

| Aspect | Vanilla Moshi | PersonaPlex |
|---|---|---|
| Persona control | CFG + condition tensors | voice prompt audio + text `<system>` tags |
| Input streams to `step()` | 1 (input audio) | 3 (user audio, agent audio, text) |
| Mimi instances | 1 | 2 |
| Warmup | zero frames only | zero frames + voice prompt + text prompt |
| CLI | `--hf-repo`, `--cfg-coef` | `--voice-prompt-dir`, `--cpu-offload` |
| Module removed | — | `conditioners/` |

---

## Observations from a sample run (`logs/log1`)

### Startup (cold)
| Stage | Time |
|---|---|
| retrieve voice prompts | ~0.2s |
| load Mimi | ~1.3s |
| load Moshi LM | ~9.7s |
| warmup (4 zero-frame iters × 2 Mimis) | ~34s |
| mkcert auto-install + cert generation | ~0.4s |

### Auto-cert
Passing `--ssl <dir>` with an empty dir triggers [moshi/utils/connection.py](personaplex/moshi/moshi/utils/connection.py) to **download `mkcert` from GitHub on demand**, install a local CA, and generate a cert for `localhost` + the detected LAN IP. No manual OpenSSL needed.

### Steady-state per 80ms audio frame
| Stage | Typical |
|---|---|
| `[STT ]` Mimi encode | 8–11ms |
| `[LLM ]` `lm_gen.step()` | ~1.1ms |
| `[TTS ]` Mimi decode | ~5.3ms |
| `[TOTAL]` frame handled | ~54ms |

54ms of compute per 80ms of audio → ~0.68× real-time, comfortable headroom. Token shape is `[1, 17, 1]` (1 text + 16 audio codebooks). Mimi emits 8 codebooks per input frame.

### Persona signal confirmed
On handshake the server logs the active `text prompt` and `voice prompt` path, e.g.:
```
[2P8H] text prompt: You work for First Neuron Bank ... your name is Alexis Kim. ...
[2P8H] voice prompt: .../voices/NATF0.pt
```
and the conversation tokens emit matching content (`' Thank' ' you' ' for' ' calling' ' First' ' Neuro' 'n' ...`), so voice + text conditioning is live in generation.

### Known warning
```
lm.py:979: FutureWarning: torch.load with weights_only=False ...
```
Triggered by `load_voice_prompt_embeddings()` when loading cached `.pt` embeddings. Benign for now (trusted files from HF), but will break when torch flips the default to `weights_only=True`. Fix: pass `weights_only=True` + allowlist the embedding dataclass.

### Session log prefix
Each websocket session gets a 4-char random ID (e.g. `[2P8H]`) from `ColorizedLog.randomize()`, so parallel connections can be disambiguated in logs.