import json
import os
import time
from contextlib import contextmanager
from typing import Any, List

from pydantic import BaseModel, Field

from client_util import load_env, get_client, get_model

load_env()
client = get_client()
MODEL = get_model()

perf_log: dict[str, float] = {}


@contextmanager
def timed(phase: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        perf_log[phase] = duration
        print(f"[perf] model={MODEL} phase={phase} latency_s={duration:.3f}")

"""
docs: https://platform.openai.com/docs/guides/function-calling
"""

# --------------------------------------------------------------
# Define the knowledge base retrieval tool
# --------------------------------------------------------------


def search_kb(question: str):
    """Load the whole knowledge base from the JSON file specified by KB_JSON_PATH.

    (This is a mock function for demonstration purposes, we don't search.)
    Falls back to a local `kb.json` in the current working directory if the env var
    is not set. Raises FileNotFoundError with a clean message if the file can't be found.
    """
    kb_path = os.getenv("KB_JSON_PATH", "kb.json")
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Knowledge base file not found at '{kb_path}'. Set KB_JSON_PATH in your .env if needed."
        ) from e


# --------------------------------------------------------------
# Step 1: Call model with search_kb tool defined
# --------------------------------------------------------------

tools: List[Any] = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."

messages: List[Any] = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

with timed("initial_create"):
    completion = client.chat.completions.create(  # type: ignore[arg-type]
        model=MODEL,
        messages=messages,  # type: ignore[arg-type]
        tools=tools,  # type: ignore[arg-type]
    )

# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------

completion.model_dump()

# --------------------------------------------------------------
# Step 3: Execute search_kb function
# --------------------------------------------------------------


def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)


tool_calls = completion.choices[0].message.tool_calls
if tool_calls:
    for tool_call in tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        messages.append(completion.choices[0].message)  # type: ignore[list-item]

        result = call_function(name, args)
        messages.append(  # type: ignore[list-item]
            {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
        )

# --------------------------------------------------------------
# Step 4: Supply result and call model again
# --------------------------------------------------------------


class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")


with timed("second_parse"):
    completion_2 = client.beta.chat.completions.parse(  # type: ignore[arg-type]
        model=MODEL,
        messages=messages,  # type: ignore[arg-type]
        tools=tools,  # type: ignore[arg-type]
        response_format=KBResponse,
    )

# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------

final_response = completion_2.choices[0].message.parsed
if final_response is None:
    raise ValueError("Model did not return a KBResponse")
print("Answer:", final_response.answer)
print("Source ID:", final_response.source)

# --------------------------------------------------------------
# Question that doesn't trigger the tool
# --------------------------------------------------------------

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

with timed("third_parse_non_kb"):
    completion_3 = client.beta.chat.completions.parse(  # type: ignore[arg-type]
        model=MODEL,
        messages=messages,  # type: ignore[arg-type]
        tools=tools,  # type: ignore[arg-type]
    )

print("Non-KB question response:", completion_3.choices[0].message.content)

total_latency = sum(perf_log.values())
print(f"[perf] model={MODEL} phase=total_workflow latency_s={total_latency:.3f}")
summary = {"model": MODEL, **{k: round(v, 4) for k, v in perf_log.items()}, "total": round(total_latency, 4)}
print("[perf-summary]", json.dumps(summary))
print("DONE")



