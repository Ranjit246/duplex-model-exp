#!/usr/bin/env python3
"""
Single audio file inference through Moshi pipeline with detailed logging.

Usage:
    python3 run_audio.py input.wav
    python3 run_audio.py /workspace/PTT-20260412-WA0014.wav -o output.wav
    python3 run_audio.py input.wav -o output.wav --hf-repo kyutai/moshika-pytorch-bf16
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import sentencepiece
import sphn
import torch

from moshi.models import loaders, LMGen, LMModel, MimiModel
from moshi.run_inference import get_condition_tensors


def log(tag, msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{tag:5s}] {msg}")


def main():
    parser = argparse.ArgumentParser(description="Run single audio file through Moshi pipeline with logging")
    parser.add_argument("input", type=str, help="Input audio file (wav, mp3, etc.)")
    parser.add_argument("-o", "--output", type=str, default="output.wav", help="Output audio file (default: output.wav)")
    parser.add_argument("--hf-repo", type=str, default="kyutai/moshika-pytorch-bf16", help="HF model repo")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--cfg-coef", type=float, default=1.0, help="CFG coefficient")
    parser.add_argument("--pre-silence", type=float, default=3.0,
                        help="Seconds of silence to prepend so Moshi finishes its greeting before your audio (default: 3.0)")
    parser.add_argument("--post-silence", type=float, default=5.0,
                        help="Seconds of silence to append so Moshi has time to finish responding (default: 5.0)")
    args = parser.parse_args()

    random.seed(4242)
    np.random.seed(4242)
    torch.manual_seed(4242)

    # --- Load models ---
    log("LOAD", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    log("LOAD", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("LOAD", f"mimi loaded | sample_rate={mimi.sample_rate} frame_rate={mimi.frame_rate}")
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    log("LOAD", "loading moshi LM")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=torch.bfloat16)
    log("LOAD", "moshi loaded")

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    condition_tensors = get_condition_tensors(checkpoint_info.model_type, lm, batch_size=1, cfg_coef=args.cfg_coef)
    lm_gen = LMGen(lm, cfg_coef=args.cfg_coef, condition_tensors=condition_tensors, **checkpoint_info.lm_gen_config)

    # --- Load input audio ---
    log("INPUT", f"reading {args.input}")
    in_pcm, _ = sphn.read(args.input, sample_rate=mimi.sample_rate)
    duration_in = in_pcm.shape[-1] / mimi.sample_rate
    log("INPUT", f"loaded {duration_in:.2f}s of audio ({in_pcm.shape[-1]} samples, {mimi.sample_rate}Hz)")

    # Prepend silence so Moshi gets its greeting out before hearing your question
    if args.pre_silence > 0:
        silence_samples = int(args.pre_silence * mimi.sample_rate)
        silence = np.zeros((in_pcm.shape[0], silence_samples), dtype=in_pcm.dtype)
        in_pcm = np.concatenate([silence, in_pcm], axis=-1)
        log("INPUT", f"prepended {args.pre_silence:.1f}s silence | total now {in_pcm.shape[-1] / mimi.sample_rate:.2f}s")

    if args.post_silence > 0:
        silence_samples = int(args.post_silence * mimi.sample_rate)
        silence = np.zeros((in_pcm.shape[0], silence_samples), dtype=in_pcm.dtype)
        in_pcm = np.concatenate([in_pcm, silence], axis=-1)
        log("INPUT", f"appended {args.post_silence:.1f}s silence | total now {in_pcm.shape[-1] / mimi.sample_rate:.2f}s")

    in_pcm = torch.from_numpy(in_pcm).to(device=args.device)
    in_pcm = in_pcm[None, 0:1]  # [1, 1, T]

    # Split into frames
    chunks = deque([
        chunk for chunk in in_pcm.split(frame_size, dim=2)
        if chunk.shape[-1] == frame_size
    ])
    log("INPUT", f"split into {len(chunks)} frames of {frame_size} samples ({1000 * frame_size / mimi.sample_rate:.0f}ms each)")

    # --- Run pipeline ---
    out_pcms = []
    out_text = []
    total_stt_ms = 0
    total_llm_ms = 0
    total_tts_ms = 0

    log("START", "beginning inference")
    pipeline_start = time.time()

    with torch.no_grad():
        mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)

        first_frame = True
        for i, chunk in enumerate(chunks):
            frame_start = time.time()
            chunk_dur_ms = chunk.shape[-1] / mimi.sample_rate * 1000

            # --- STT: encode audio to codes ---
            t0 = time.time()
            codes = mimi.encode(chunk)
            stt_ms = 1000 * (time.time() - t0)
            total_stt_ms += stt_ms
            codes_vals = [int(c) for c in codes[0, :, 0].tolist()]
            log("STT", f"frame {i:3d} | encode {stt_ms:5.1f}ms | codes {codes_vals}")

            if first_frame:
                # First frame is used to prime the model
                tokens = lm_gen.step(codes)
                first_frame = False

            # --- LLM: generate tokens ---
            for c in range(codes.shape[-1]):
                t1 = time.time()
                tokens = lm_gen.step(codes[:, :, c: c + 1])
                llm_ms = 1000 * (time.time() - t1)
                total_llm_ms += llm_ms

                if tokens is None:
                    log("LLM", f"frame {i:3d} | step {llm_ms:5.1f}ms | no tokens yet")
                    continue

                token_vals = [int(t) for t in tokens[0, :, 0].tolist()]
                log("LLM", f"frame {i:3d} | step {llm_ms:5.1f}ms | tokens {token_vals[:3]}...")

                # --- Text token ---
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    text = text_tokenizer.id_to_piece(text_token)
                    text = text.replace("\u2581", " ")
                    out_text.append(text)
                    log("TEXT", f"frame {i:3d} | '{text}'")

                # --- TTS: decode tokens to audio ---
                t2 = time.time()
                out_pcm = mimi.decode(tokens[:, 1:])
                tts_ms = 1000 * (time.time() - t2)
                total_tts_ms += tts_ms
                pcm_samples = out_pcm.shape[-1]
                pcm_dur_ms = pcm_samples / mimi.sample_rate * 1000
                log("TTS", f"frame {i:3d} | decode {tts_ms:5.1f}ms | {pcm_dur_ms:.0f}ms audio ({pcm_samples} samples)")

                out_pcms.append(out_pcm[0].cpu())

            frame_ms = 1000 * (time.time() - frame_start)
            log("TOTAL", f"frame {i:3d} | {frame_ms:5.1f}ms")

    pipeline_ms = 1000 * (time.time() - pipeline_start)

    # --- Save output ---
    if out_pcms:
        out_audio = torch.cat(out_pcms, dim=1)
        duration_out = out_audio.shape[-1] / mimi.sample_rate
        sphn.write_wav(args.output, out_audio[0].numpy(), sample_rate=mimi.sample_rate)
        log("OUT", f"saved {args.output} | {duration_out:.2f}s of audio")
    else:
        log("OUT", "no audio generated")

    # --- Summary ---
    n_frames = len(chunks)
    full_text = "".join(out_text).strip()
    print()
    print("=" * 60)
    print(f"  Input:      {args.input} ({duration_in:.2f}s)")
    print(f"  Output:     {args.output} ({duration_out:.2f}s)" if out_pcms else "  Output:     (none)")
    print(f"  Frames:     {n_frames}")
    print(f"  Pipeline:   {pipeline_ms:.0f}ms total")
    print(f"  STT  avg:   {total_stt_ms / n_frames:.1f}ms/frame")
    print(f"  LLM  avg:   {total_llm_ms / n_frames:.1f}ms/frame")
    print(f"  TTS  avg:   {total_tts_ms / max(len(out_pcms), 1):.1f}ms/frame")
    print(f"  Text:       {full_text}")
    print("=" * 60)


if __name__ == "__main__":
    main()
