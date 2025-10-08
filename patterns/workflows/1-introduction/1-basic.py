import os
import time
import json
from contextlib import contextmanager

from openai import OpenAI
from dotenv import load_dotenv
from client_util import load_env, get_client, get_model, model_to_json


"""
Bedrock ValidationException fix:
Some Bedrock (Anthropic) models do NOT allow specifying both `temperature` and `top_p`.
Your gateway is likely forwarding BOTH with default values when neither is supplied.
Explicitly send ONLY ONE (we choose temperature) so the gateway can omit the other.

If your gateway still injects both, update the gateway logic to drop one when both present.
"""


perf_log: dict[str, float] = {}
MODEL = None  # will be set in main


@contextmanager
def timed(phase: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        perf_log[phase] = duration
        # MODEL may not yet be initialized when first used
        model_name = MODEL if MODEL else "(unset)"
        print(f"[perf] model={model_name} phase={phase} latency_s={duration:.3f}")


def get_client():
    # Prefer an env var instead of hard‑coding secrets.
    api_key = os.getenv("OPENAI_API_KEY", "???")  # fallback for local dev
    # Allow overriding the gateway base URL via env (e.g. OPENAI_BASE_URL) with a sensible default.
    # Ensure it points at the OpenAI-compatible root (often ends with /v1)
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def main():
    # Load variables from a local .env file (if present)
    load_dotenv()

    client = get_client()
    global MODEL  # set global for perf logging
    MODEL = get_model()

    with timed("basic_completion"):
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You're a helpful assistant."},
                {
                    "role": "user",
                    "content": "Write a limerick about the Python programming language.",
                },
            ],
            # IMPORTANT: Specify only one of temperature OR top_p.
            temperature=0.7,
        )

    # OpenAI Python library (>=1.0) structure
    response_text = completion.choices[0].message.content
    print(response_text)

    total_latency = sum(perf_log.values())
    print(f"[perf] model={MODEL} phase=total_workflow latency_s={total_latency:.3f}")
    summary = {"model": MODEL, **{k: round(v, 4) for k, v in perf_log.items()}, "total": round(total_latency, 4)}
    print("[perf-summary]", json.dumps(summary))
    print("DONE")


if __name__ == "__main__":
    main()
