from __future__ import annotations

from typing import Any, Dict


SUPPORTED_ALPHA_POLICY_TYPES = {
    "constant",
    "piecewise_by_t_env",
    "piecewise_by_episode",
    "linear_decay",
}


def validate_alpha_policy(alpha_policy: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(alpha_policy, dict):
        raise ValueError("alpha_policy must be a dict")

    policy_type = alpha_policy.get("type")
    if policy_type not in SUPPORTED_ALPHA_POLICY_TYPES:
        raise ValueError(
            "alpha_policy.type must be one of: "
            + ", ".join(sorted(SUPPORTED_ALPHA_POLICY_TYPES))
        )

    if policy_type == "constant":
        if "value" not in alpha_policy:
            raise ValueError('constant alpha_policy must include key "value"')
        _ensure_numeric(alpha_policy["value"], "alpha_policy.value")
        return alpha_policy

    if policy_type in {"piecewise_by_t_env", "piecewise_by_episode"}:
        segments = alpha_policy.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"{policy_type} alpha_policy must include a non-empty segments list")
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise ValueError(f"alpha_policy.segments[{index}] must be a dict")
            if "start" not in segment:
                raise ValueError(f'alpha_policy.segments[{index}] must include key "start"')
            _ensure_numeric(segment["start"], f"alpha_policy.segments[{index}].start")
            if "end" in segment and segment["end"] is not None:
                _ensure_numeric(segment["end"], f"alpha_policy.segments[{index}].end")
            if "alpha" not in segment:
                raise ValueError(f'alpha_policy.segments[{index}] must include key "alpha"')
            _ensure_numeric(segment["alpha"], f"alpha_policy.segments[{index}].alpha")
        if "default" in alpha_policy:
            _ensure_numeric(alpha_policy["default"], "alpha_policy.default")
        return alpha_policy

    if policy_type == "linear_decay":
        required_keys = ["start", "end", "start_t", "end_t"]
        for key in required_keys:
            if key not in alpha_policy:
                raise ValueError(f'linear_decay alpha_policy must include key "{key}"')
            _ensure_numeric(alpha_policy[key], f"alpha_policy.{key}")
        return alpha_policy

    return alpha_policy


def _ensure_numeric(value: Any, field_name: str) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
