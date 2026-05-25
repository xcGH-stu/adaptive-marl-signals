from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


POLICY_GUIDANCE_SPEC_VERSION = 1
SUPPORTED_EXPLORATION_BIAS_MODES = {"none", "encourage_exploration", "reduce_exploration"}
SUPPORTED_SCHEDULE_RECOMMENDATION_MODES = {"none", "constant", "phase_shift", "anneal"}


DEFAULT_POLICY_GUIDANCE_SPEC: Dict[str, Any] = {
    "version": POLICY_GUIDANCE_SPEC_VERSION,
    "exploration_bias": {
        "enabled": False,
        "mode": "none",
        "strength": 0.0,
        "target_agents": "all",
    },
    "curriculum_hint": {
        "enabled": False,
        "phase": "default",
        "focus": "",
    },
    "phase_advice": {
        "enabled": False,
        "stage": "",
        "advice": "",
    },
    "schedule_recommendation": {
        "enabled": False,
        "mode": "none",
        "value": None,
    },
    "metadata": {},
}


def get_default_policy_guidance_spec() -> Dict[str, Any]:
    return deepcopy(DEFAULT_POLICY_GUIDANCE_SPEC)


def validate_policy_guidance_spec(spec: Dict[str, Any] | None) -> Dict[str, Any]:
    if spec is None:
        return get_default_policy_guidance_spec()
    if not isinstance(spec, dict):
        raise ValueError("policy_guidance_spec must be a dict")

    version = spec.get("version", POLICY_GUIDANCE_SPEC_VERSION)
    if version != POLICY_GUIDANCE_SPEC_VERSION:
        raise ValueError(
            "policy_guidance_spec.version must equal "
            f"{POLICY_GUIDANCE_SPEC_VERSION}"
        )

    exploration_bias = _validate_exploration_bias(spec.get("exploration_bias"))
    curriculum_hint = _validate_curriculum_hint(spec.get("curriculum_hint"))
    phase_advice = _validate_phase_advice(spec.get("phase_advice"))
    schedule_recommendation = _validate_schedule_recommendation(
        spec.get("schedule_recommendation")
    )
    metadata = spec.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("policy_guidance_spec.metadata must be a dict")

    return {
        "version": version,
        "exploration_bias": exploration_bias,
        "curriculum_hint": curriculum_hint,
        "phase_advice": phase_advice,
        "schedule_recommendation": schedule_recommendation,
        "metadata": deepcopy(metadata),
    }


def summarize_policy_guidance_spec(spec: Dict[str, Any] | None) -> Dict[str, Any]:
    validated = validate_policy_guidance_spec(spec)
    return {
        "exploration_bias_enabled": bool(
            validated["exploration_bias"].get("enabled", False)
        ),
        "exploration_bias_mode": validated["exploration_bias"].get("mode"),
        "curriculum_hint_enabled": bool(
            validated["curriculum_hint"].get("enabled", False)
        ),
        "curriculum_phase": validated["curriculum_hint"].get("phase"),
        "phase_advice_enabled": bool(validated["phase_advice"].get("enabled", False)),
        "phase_advice_stage": validated["phase_advice"].get("stage"),
        "schedule_recommendation_enabled": bool(
            validated["schedule_recommendation"].get("enabled", False)
        ),
        "schedule_recommendation_mode": validated["schedule_recommendation"].get("mode"),
    }


def _validate_exploration_bias(value: Any) -> Dict[str, Any]:
    default = deepcopy(DEFAULT_POLICY_GUIDANCE_SPEC["exploration_bias"])
    if value is None:
        return default
    if not isinstance(value, dict):
        raise ValueError("policy_guidance_spec.exploration_bias must be a dict")
    merged = dict(default)
    merged.update(value)
    if not isinstance(merged.get("enabled"), bool):
        raise ValueError("policy_guidance_spec.exploration_bias.enabled must be a bool")
    if merged.get("mode") not in SUPPORTED_EXPLORATION_BIAS_MODES:
        raise ValueError(
            "policy_guidance_spec.exploration_bias.mode must be one of: "
            + ", ".join(sorted(SUPPORTED_EXPLORATION_BIAS_MODES))
        )
    if not isinstance(merged.get("strength"), (int, float)):
        raise ValueError(
            "policy_guidance_spec.exploration_bias.strength must be numeric"
        )
    merged["strength"] = float(merged["strength"])
    target_agents = merged.get("target_agents", "all")
    if not isinstance(target_agents, (str, list)):
        raise ValueError(
            "policy_guidance_spec.exploration_bias.target_agents must be a string or list"
        )
    return merged


def _validate_curriculum_hint(value: Any) -> Dict[str, Any]:
    default = deepcopy(DEFAULT_POLICY_GUIDANCE_SPEC["curriculum_hint"])
    if value is None:
        return default
    if not isinstance(value, dict):
        raise ValueError("policy_guidance_spec.curriculum_hint must be a dict")
    merged = dict(default)
    merged.update(value)
    if not isinstance(merged.get("enabled"), bool):
        raise ValueError("policy_guidance_spec.curriculum_hint.enabled must be a bool")
    for key in ("phase", "focus"):
        if not isinstance(merged.get(key), str):
            raise ValueError(f"policy_guidance_spec.curriculum_hint.{key} must be a string")
    return merged


def _validate_phase_advice(value: Any) -> Dict[str, Any]:
    default = deepcopy(DEFAULT_POLICY_GUIDANCE_SPEC["phase_advice"])
    if value is None:
        return default
    if not isinstance(value, dict):
        raise ValueError("policy_guidance_spec.phase_advice must be a dict")
    merged = dict(default)
    merged.update(value)
    if not isinstance(merged.get("enabled"), bool):
        raise ValueError("policy_guidance_spec.phase_advice.enabled must be a bool")
    for key in ("stage", "advice"):
        if not isinstance(merged.get(key), str):
            raise ValueError(f"policy_guidance_spec.phase_advice.{key} must be a string")
    return merged


def _validate_schedule_recommendation(value: Any) -> Dict[str, Any]:
    default = deepcopy(DEFAULT_POLICY_GUIDANCE_SPEC["schedule_recommendation"])
    if value is None:
        return default
    if not isinstance(value, dict):
        raise ValueError("policy_guidance_spec.schedule_recommendation must be a dict")
    merged = dict(default)
    merged.update(value)
    if not isinstance(merged.get("enabled"), bool):
        raise ValueError(
            "policy_guidance_spec.schedule_recommendation.enabled must be a bool"
        )
    if merged.get("mode") not in SUPPORTED_SCHEDULE_RECOMMENDATION_MODES:
        raise ValueError(
            "policy_guidance_spec.schedule_recommendation.mode must be one of: "
            + ", ".join(sorted(SUPPORTED_SCHEDULE_RECOMMENDATION_MODES))
        )
    return merged
