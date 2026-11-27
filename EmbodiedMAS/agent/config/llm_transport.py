"""Read a completion and the gateway's reported cost without extra requests."""
from __future__ import annotations

import math
from typing import Any


_COST_HEADER = "x-litellm-response-cost"


def create_chat_completion(client: Any, request_kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Preserve create-only clients while retaining LiteLLM's response cost.

    This is the cost reported by the gateway, not an independently verified
    provider bill. Missing or invalid values remain unknown. Errors from an
    available raw-response interface propagate; retrying through ``create``
    could issue another billable request.
    """
    completions = client.chat.completions
    raw_interface = getattr(completions, "with_raw_response", None)
    raw_create = getattr(raw_interface, "create", None)
    cost = {
        "gateway_response_cost_usd": None,
        "gateway_response_cost_source": None,
        "gateway_response_cost_status": "raw_response_unavailable",
    }
    if not callable(raw_create):
        return completions.create(**request_kwargs), cost

    raw_response = raw_create(**request_kwargs)
    completion = raw_response.parse()
    # Deliberately retain only this allowlisted cost field, never all headers.
    headers = getattr(raw_response, "headers", None)
    value = headers.get(_COST_HEADER) if headers is not None else None
    cost["gateway_response_cost_source"] = _COST_HEADER
    if value is None:
        cost["gateway_response_cost_status"] = "missing_header"
        return completion, cost
    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        amount = math.nan
    if math.isfinite(amount) and amount >= 0:
        cost.update(gateway_response_cost_usd=amount, gateway_response_cost_status="reported")
    else:
        cost["gateway_response_cost_status"] = "invalid_header"
    return completion, cost
