import os
import json
import datetime
import random
from functools import lru_cache
from typing import Optional, Any, Type, get_origin, get_args

try:  # Optional import; only needed when using model_to_json
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover - defensive if pydantic not installed
    BaseModel = object  # type: ignore

from dotenv import load_dotenv
from openai import OpenAI

# Centralized environment/bootstrap helpers for introduction examples.
# Usage pattern in scripts:
#   from client_util import load_env, get_client, get_model
#   load_env()
#   client = get_client()
#   model = get_model()
#
# Environment variables:
#   OPENAI_API_KEY   - API key (required unless using a local proxy)
#   OPENAI_BASE_URL  - Override base URL (default: http://localhost:1234/v1)
#   OPENAI_MODEL     - Default model name (default: gpt-4o)
#
# NOTE: Do not print secrets in real applications.


@lru_cache(maxsize=1)
def load_env() -> None:
    """Load .env exactly once (cached)."""
    load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Return a cached OpenAI-compatible client using env configuration."""
    api_key = os.getenv("OPENAI_API_KEY", "???")
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model(default: str = "gpt-4o") -> str:
    """Return the model name from env (OPENAI_MODEL) with a fallback."""
    return os.getenv("OPENAI_MODEL", default)


def mask_key(key: Optional[str]) -> str:
    if not key:
        return "<missing>" 
    if len(key) <= 8:
        return f"***{key[-2:]}"
    return f"{key[:4]}...{key[-4:]}"


# --------------------------------------------------------------
# Structured output example JSON generation helpers
# --------------------------------------------------------------

def _example_scalar(py_type: Any, field_name: str, model_name: str):
    """Return an example scalar value informed by field & model names.

    Heuristics:
    - *date* in name -> future ISO date within ~6 months
    - *time* in name -> HH:MM string
    - *name* in name -> "<ModelName> <Field Title>"
    - *id* in name -> slug with random numeric suffix
    Otherwise fall back by type.
    """
    lname = field_name.lower()
    seed_basis = f"{model_name}:{field_name}"
    rng = random.Random(seed_basis)

    if "date" in lname:
        days_ahead = rng.randint(1, 180)
        return (datetime.date.today() + datetime.timedelta(days=days_ahead)).isoformat()
    if "time" in lname:
        minutes = rng.randint(0, 23 * 60 + 59)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"
    if "name" in lname:
        return f"{model_name} {field_name.replace('_', ' ').title()}"
    if lname.endswith("id") or lname == "id" or lname.endswith("_id"):
        return f"{model_name.lower()}_{field_name.lower()}_{rng.randint(1000,9999)}"

    if py_type in (str, Any):
        return f"{model_name} {field_name.replace('_', ' ').title()}"
    if py_type is int:
        return 0
    if py_type is float:
        return 0.0
    if py_type is bool:
        return True
    return f"{field_name}_value"


def _example_for_type(tp: Any, field_name: str, model_name: str):
    origin = get_origin(tp)
    if origin in (list, tuple, set):
        args = get_args(tp)
        inner = args[0] if args else str
        singular = field_name.rstrip("s") or field_name
        first = _example_for_type(inner, singular, model_name)
        second = _example_for_type(inner, singular, model_name)
        if isinstance(first, str) and isinstance(second, str) and first == second:
            second = first + " 2"
        seq = [first, second]
        if origin is tuple:
            return tuple(seq)
        if origin is set:
            return set(seq)
        return seq
    if origin is dict:
        k_t, v_t = get_args(tp) if get_args(tp) else (str, str)
        return {
            _example_for_type(k_t, "key", model_name): _example_for_type(
                v_t, field_name + "_value", model_name
            )
        }
    if isinstance(tp, type) and isinstance(BaseModel, type) and issubclass(tp, BaseModel):  # type: ignore[arg-type]
        return json.loads(model_to_json(tp))
    return _example_scalar(tp, field_name, model_name)


def model_to_json(model_cls: Type[BaseModel]) -> str:  # type: ignore[valid-type]
    """Generate an example JSON string for a Pydantic model class.

    Safe to import in contexts where pydantic may not actually be used—will
    raise a helpful error if BaseModel was not imported correctly.
    """
    if not isinstance(BaseModel, type):  # pydantic missing
        raise RuntimeError("pydantic BaseModel not available; install pydantic to use model_to_json")
    data = {}
    for name, field in model_cls.model_fields.items():  # type: ignore[attr-defined]
        data[name] = _example_for_type(field.annotation, name, model_cls.__name__)
    return json.dumps(data, indent=2)


__all__ = [
    "load_env",
    "get_client",
    "get_model",
    "mask_key",
    "model_to_json",
]


if __name__ == "__main__":  # Small self-test utility
    load_env()
    client = get_client()
    print("Loaded client with base_url=", client.base_url)
    print("Model=", get_model())
    print("API Key (masked)=", mask_key(os.getenv("OPENAI_API_KEY")))
