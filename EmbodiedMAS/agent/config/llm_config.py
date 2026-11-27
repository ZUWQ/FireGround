"""Explicit reviewer-supplied OpenAI-compatible endpoint for the Oracle runner."""
import os
from typing import Any

LLM_PROFILES = {}

def select_llm_profile(name):
    raise ValueError("Set OPENAI_MODEL explicitly for the review runner.")

def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} before starting a live episode.")
    return value

class LLMConfig:
    def get_model(self):
        return required("OPENAI_MODEL")

    def get_client_kwargs(self):
        return {"base_url": required("OPENAI_BASE_URL"), "api_key": required("OPENAI_API_KEY"),
                "timeout": 60.0, "max_retries": 0}

    def get_chat_completion_extra_kwargs(self):
        return {}

def configured_model_name():
    return LLMConfig().get_model()

def get_llm_config():
    return LLMConfig()

def _merge_chat_completion_kwargs(
    original_kwargs: dict[str, Any],
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    合并 chat completion 参数：
    - 调用方显式传入的参数优先，不被覆盖
    - extra_body 支持浅层 dict merge
    """
    merged = dict(original_kwargs)
    for key, val in extra_kwargs.items():
        if key == "extra_body":
            if "extra_body" not in merged:
                merged["extra_body"] = val
            elif isinstance(merged.get("extra_body"), dict) and isinstance(val, dict):
                merged["extra_body"] = {**val, **merged["extra_body"]}
            continue
        merged.setdefault(key, val)
    return merged


