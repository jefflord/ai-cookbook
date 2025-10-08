from pydantic import BaseModel
import json
import time
from contextlib import contextmanager
from typing import Any, Type

from client_util import load_env, get_client, get_model, model_to_json

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


# --------------------------------------------------------------
# Step 1: Define the response format in a Pydantic model
# --------------------------------------------------------------


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


"""Example structured output workflow script using a shared JSON example generator.

The helper `model_to_json` now lives in `client_util` so other scripts can
reuse consistent illustrative JSON examples without duplicating heuristics.
"""


# Generic example JSON for the target model (auto-generated)
CALENDAR_EVENT_EXAMPLE_SCHEMA = model_to_json(CalendarEvent)


#print("\n--- Example CalendarEvent JSON ---")
#print(EXAMPLE_JSON)

# --------------------------------------------------------------
# Step 2: Call the model
# --------------------------------------------------------------

default_content = "Extract the event information."
default_content = "Extract the event information and return ONLY valid JSON matching the schema.\n"
default_content += "Schema fields: name (str), date (str), participants (list[str]).\n"
default_content += "Example response JSON (do not include comments):\n" + CALENDAR_EVENT_EXAMPLE_SCHEMA


with timed("structured_parse"):
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": default_content,
            },
            {
                "role": "user",
                "content": "Jeff and Gary will be at the moon landing on July 20th.",
            },
        ],
        response_format=CalendarEvent,
    )

# --------------------------------------------------------------
# Step 3: Parse the response
# --------------------------------------------------------------

# Always log the raw body returned from the completion before parsing
if False:  # Set to True to enable raw output logging
    print("\n--- Raw Completion Object (may include additional metadata) ---")
    try:
        # Attempt to serialize if the object supports pydantic-style export
        if hasattr(completion, "model_dump_json"):
            print(completion.model_dump_json(indent=2))
        elif hasattr(completion, "model_dump"):
            print(json.dumps(completion.model_dump(), indent=2))
        else:
            # Fallback to generic repr
            print(repr(completion))
    except Exception as e:  # pragma: no cover - defensive
        print(f"[WARN] Failed to serialize raw completion: {e}")

# Parse the structured response
event = completion.choices[0].message.parsed
if event is None:
    raise ValueError("Model did not return a structured CalendarEvent response")

print("\n--- Parsed CalendarEvent ---")
print("Event Name:", event.name)
print("Event Date:", event.date)
print("Participants:", event.participants)

total_latency = sum(perf_log.values())
print(f"[perf] model={MODEL} phase=total_workflow latency_s={total_latency:.3f}")
summary = {"model": MODEL, **{k: round(v, 4) for k, v in perf_log.items()}, "total": round(total_latency, 4)}
print("[perf-summary]", json.dumps(summary))
print("DONE")
