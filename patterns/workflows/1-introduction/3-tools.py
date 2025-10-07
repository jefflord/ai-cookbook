import json
import time
from typing import Any, List, Dict
from contextlib import contextmanager

import requests
from pydantic import BaseModel, Field

from client_util import load_env, get_client, get_model, model_to_json

load_env()
client = get_client()
MODEL = get_model()

perf_log: Dict[str, float] = {}


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
# Define the tool (function) that we want to call
# --------------------------------------------------------------


def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]


# --------------------------------------------------------------
# Step 1: Call model with get_weather tool defined
# --------------------------------------------------------------

tools: List[Any] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


system_prompt = "You are a helpful weather assistant."

messages: List[Any] = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Paris today?"},
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

print("\n--- Model response ---")
print(completion.choices[0].message)  # type: ignore[attr-defined]
completion.model_dump()

# --------------------------------------------------------------
# Step 3: Execute get_weather function
# --------------------------------------------------------------


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)


tool_calls = completion.choices[0].message.tool_calls
if tool_calls:
    for tool_call in tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        messages.append(completion.choices[0].message)  # type: ignore[list-item]

        result = call_function(name, args)
        messages.append(  # type: ignore[list-item]
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )

# --------------------------------------------------------------
# Step 4: Supply result and call model again
# --------------------------------------------------------------


class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )


if False:
    # get model_to_json(WeatherResponse)
    WEATHER_RESPONSE_EXAMPLE_SCHEMA = model_to_json(WeatherResponse)

    system_prompt = (
        "You are a helpful weather assistant. "
        "Use the provided tool to get the current temperature. "
        "Respond in JSON format matching the schema exactly. "
        "Schema fields: temperature (float), response (str). "
        "Example response JSON (do not include comments):\n"
        + WEATHER_RESPONSE_EXAMPLE_SCHEMA
    )

    completion_2 = client.beta.chat.completions.parse(  # type: ignore[arg-type]
        model=MODEL,
        messages=messages,  # type: ignore[arg-type]
        tools=tools,  # type: ignore[arg-type]
        response_format=WeatherResponse,
    )
else:

    with timed("second_parse"):
        completion_2 = client.beta.chat.completions.parse(
            model=MODEL,
            messages=messages,
            tools=tools,
            response_format=WeatherResponse,
        )

# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------

final_response = completion_2.choices[0].message.parsed
if final_response is None:
    raise ValueError("Model did not return a WeatherResponse")
print("Temperature:", final_response.temperature)
print("Response:", final_response.response)
total_latency = sum(perf_log.values())
print(f"[perf] model={MODEL} phase=total_workflow latency_s={total_latency:.3f}")
summary = {"model": MODEL, **{k: round(v, 4) for k, v in perf_log.items()}, "total": round(total_latency, 4)}
print("[perf-summary]", json.dumps(summary))
print("DONE")
