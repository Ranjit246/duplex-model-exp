from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.oracle_generation import (
    OracleGenerator,
    OraclePrediction,
    OraclePredictionRequest,
    PredictFn,
    words_from_word_transcript,
)

DEFAULT_MODEL_NAME = "gpt-4.1-nano"


def prediction_to_record(prediction: OraclePrediction) -> dict[str, object]:
    return {
        "timestamp_ms": prediction.timestamp_ms,
        "conversation_context": prediction.conversation_context,
        "prediction": prediction.prediction,
        "total_word_count": prediction.total_word_count,
        "trigger_word": prediction.trigger_word,
        "recent_words": prediction.recent_words,
        "current_spoken_ratio": prediction.current_spoken_ratio,
        "channel": prediction.channel,
        "hint": prediction.hint,
    }


def fallback_prediction_for_request(request: OraclePredictionRequest) -> str:
    """Fallback used when the OpenAI request fails.

    Keep the "no hint in the first half" semantics intact by returning an empty
    prediction when the current speaker is at or before the halfway point.
    """
    if request.current_spoken_ratio <= 0.5:
        return ""
    return request.next_utterance_hint


def build_user_prompt(request: OraclePredictionRequest) -> str:
    if request.current_spoken_ratio <= 0.5:
        prompt = f"""You are predicting what the next speaker will say in a Hinglish customer service call.
A is the customer, B is the agent/assistant.
Note: the conversation context below may use transliterated Hindi in Roman script (e.g. "aapka naam kya hai") — this is how it was captured. Your prediction must use proper Hinglish: English words in Roman script, Hindi words in Devanagari script (e.g. "हाँ sir, आपका account check करते हैं").

Conversation so far:
{request.conversation_context}

The current speaker ({request.current_speaker}) has only spoken {request.current_spoken_ratio:.0%} of their turn and is still talking.
Predict what {request.next_speaker} will say next.

Since the current speaker has just begun, base your prediction on the conversation history only.
Generate a natural, contextually appropriate Hinglish response."""
    else:
        prompt = f"""You are predicting what the next speaker will say in a Hinglish customer service call.
A is the customer, B is the agent/assistant.
Note: the conversation context below may use transliterated Hindi in Roman script (e.g. "aapka naam kya hai") — this is how it was captured. Your prediction must use proper Hinglish: English words in Roman script, Hindi words in Devanagari script (e.g. "हाँ sir, आपका account check करते हैं").

Conversation so far:
{request.conversation_context}

The current speaker ({request.current_speaker}) has spoken {request.current_spoken_ratio:.0%} of their turn and is still talking.
Predict what {request.next_speaker} will say next.

Hidden hint (do not mention or quote this directly, but let it guide your prediction):
The actual next utterance will be similar to: "{request.next_utterance_hint}"

Guidelines based on current speaker's progress ({request.current_spoken_ratio:.1%}):"""

        if request.current_spoken_ratio <= 0.65:
            prompt += """
- You have only heard about half of the current turn.
- Rely primarily on the conversation flow so far.
- You may use hint keywords, but avoid following the hint too closely."""
        elif request.current_spoken_ratio <= 0.8:
            prompt += """
- You are in the latter half of the current turn.
- Respect the conversation flow while referencing the hint.
- Include some content that differs from the hint."""
        elif request.current_spoken_ratio <= 0.95:
            prompt += """
- Most of the current turn is complete.
- The hint can be an important clue.
- Avoid copying the hint verbatim; prefer a slightly different expression."""
        elif request.current_spoken_ratio <= 0.99:
            prompt += """
- You are approaching the end of the current turn.
- The hint can be used more directly, while still sounding like a natural prediction."""
        else:
            prompt += """
- You have effectively heard the full current turn.
- The output may closely match the hint if that is the most natural continuation."""

    prompt += f"""

Generate a natural Hinglish response that sounds like genuine spoken Indian customer service.

Important guidelines:
- IMPORTANT: You MUST write all Hindi/Urdu words in Devanagari script — never in Roman transliteration
- If the hint contains Roman-script Hindi (e.g. "aapka", "haan", "kya"), convert those words to Devanagari in your response
- Speak confidently without asking for confirmation
- Use at most 30 words and at least 5 words
- Use conversational spoken language only — not formal Hindi, not formal English
- Do not refer to yourself as human, AI, or assistant
- Do not include quotes, bullets, or explanations
- Avoid text formatting and avoid punctuation other than periods and commas

Respond with only the predicted Hinglish text that {request.next_speaker} will say."""
    return prompt


def make_openai_predict_fn(
    *,
    model_name: str,
    base_url: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    presence_penalty: float = 0.4,
    frequency_penalty: float = 0.5,
    disable_thinking: bool = False,
    fallback_to_hint_on_error: bool = False,
):
    from openai import OpenAI

    # Key resolution: local server → "dummy", else OpenRouter or OpenAI
    resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or (
        "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
    )
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "dummy"  # local servers don't need a real key
    )
    if not resolved_base_url and api_key == "dummy":
        raise ValueError("Set LLM_BASE_URL, OPENROUTER_API_KEY, or OPENAI_API_KEY")

    client = OpenAI(api_key=api_key, base_url=resolved_base_url, timeout=60.0)

    # Qwen3 requires enable_thinking=False to skip chain-of-thought output
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else {"enable_thinking": True}

    def predict(request: OraclePredictionRequest) -> str:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in Indian customer service conversations. "
                            "You predict spoken responses in Hinglish — the natural mix of Hindi and English "
                            "used in Indian call centres. "
                            "A is the customer, B is the agent/assistant. "
                            "Write English words in Roman script and Hindi words in Devanagari script. "
                            "Your predictions must sound like natural spoken Hinglish, not formal Hindi or formal English."
                        ),
                    },
                    {"role": "user", "content": build_user_prompt(request)},
                ],
                max_tokens=512,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stream=False,
                extra_body=extra_body,
            )
        except Exception:
            if fallback_to_hint_on_error:
                return fallback_prediction_for_request(request)
            raise

        return (response.choices[0].message.content or "").strip()

    return predict


def generate_oracle_records(
    transcript_records: list[dict[str, object]],
    *,
    predict_fn: PredictFn,
    time_interval: float,
    target_channel: int | None,
    speaker_to_channel: dict[str, int],
) -> list[dict[str, object]]:
    words = words_from_word_transcript(transcript_records)
    generator = OracleGenerator(
        predict_fn,
        time_interval=time_interval,
        target_channel=target_channel,
        speaker_to_channel=speaker_to_channel,
    )
    predictions = generator.generate_predictions(words)
    return [prediction_to_record(prediction) for prediction in predictions]


def process_text_file(
    text_path: Path,
    output_path: Path,
    *,
    predict_fn: PredictFn,
    time_interval: float,
    target_channel: int | None,
    speaker_to_channel: dict[str, int],
) -> int:
    with text_path.open(encoding="utf-8") as f:
        transcript_records = json.load(f)

    oracle_records = generate_oracle_records(
        transcript_records,
        predict_fn=predict_fn,
        time_interval=time_interval,
        target_channel=target_channel,
        speaker_to_channel=speaker_to_channel,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(oracle_records, f, ensure_ascii=False, indent=2)

    return len(oracle_records)


def main(args: argparse.Namespace) -> None:
    speaker_to_channel = {"A": args.A_channel, "B": args.B_channel}
    predict_fn = make_openai_predict_fn(
        model_name=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        disable_thinking=args.disable_thinking,
        fallback_to_hint_on_error=args.fallback_to_hint_on_error,
    )

    text_paths = sorted(Path(args.text_dir).glob("*.json"))
    if args.limit is not None:
        text_paths = text_paths[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for text_path in text_paths:
        output_path = output_dir / text_path.name
        if args.resume and output_path.exists():
            continue
        pending.append((text_path, output_path))

    skipped = len(text_paths) - len(pending)
    if skipped:
        print(f"Skipping {skipped} already-done files (--resume)")
    print(f"Processing {len(pending)} files with {args.workers} workers...")

    def _process(item: tuple[Path, Path]) -> tuple[Path, int]:
        text_path, output_path = item
        print(f"  → {text_path.name}", flush=True)
        n = process_text_file(
            text_path,
            output_path,
            predict_fn=predict_fn,
            time_interval=args.time_interval,
            target_channel=args.target_channel,
            speaker_to_channel=speaker_to_channel,
        )
        return output_path, n

    total = len(pending)
    completed = 0
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process, item): item for item in pending}
        for future in as_completed(futures):
            completed += 1
            elapsed = time.monotonic() - start_time
            rate = completed / elapsed  # files/sec
            remaining = total - completed
            eta_sec = remaining / rate if rate > 0 else 0
            eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
            try:
                output_path, n = future.result()
                print(f"[{completed}/{total} | {eta_str}] Wrote {n} events → {output_path.name}", flush=True)
            except Exception as exc:
                text_path, _ = futures[future]
                print(f"[{completed}/{total} | {eta_str}] ERROR {text_path.name}: {exc}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate oracle_raw JSON files directly from canonical text transcripts."
    )
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base_url", type=str, default=None, help="Override API base URL (e.g. http://136.107.170.187:8001/v1)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--presence_penalty", type=float, default=0.4)
    parser.add_argument("--frequency_penalty", type=float, default=0.5)
    parser.add_argument("--disable_thinking", action="store_true", help="Pass enable_thinking=False for Qwen3 models")
    parser.add_argument("--time_interval", type=float, default=0.5)
    parser.add_argument("--target_channel", type=int, choices=[0, 1], default=None)
    parser.add_argument("--A_channel", type=int, default=0)
    parser.add_argument("--B_channel", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4, help="Parallel file workers")
    parser.add_argument("--fallback_to_hint_on_error", action="store_true")
    main(parser.parse_args())
