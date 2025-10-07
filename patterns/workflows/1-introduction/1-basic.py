import os

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


    MODEL = get_model()
    

    

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
        # top_p=None  # (Leave commented / None so it is NOT sent.)
    )

    # OpenAI Python library (>=1.0) structure
    response_text = completion.choices[0].message.content
    print(response_text)


if __name__ == "__main__":
    main()
