from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from rewarding.spec_schema import get_term_definition, validate_reward_spec
from workflows.reward_update_schema import sanitize_reward_update_plan


DEFAULT_WEIGHT_STEP = 0.01
DEFAULT_PBRS_STEP = {
    "beta": 0.1,
    "wc": 0.05,
    "wp": 0.05,
    "gamma": 0.01,
    "gate_radius": 1.0,
}
PBRS_LIMITS = {
    "beta": (0.0, 2.0),
    "wc": (0.0, 1.0),
    "wp": (0.0, 1.0),
    "gamma": (0.0, 1.0),
    "gate_radius": (1.0, 10.0),
}
PBRS_NESTED_FIELDS = {
    "variant": ("variant",),
    "gate_mode": ("gate", "mode"),
    "gate_radius": ("gate", "radius"),
    "closeness_mode": ("closeness", "mode"),
}
PBRS_CATEGORICAL_FIELDS = {"variant", "gate_mode", "closeness_mode"}


def apply_reward_update_plan(
    reward_spec: Dict[str, Any],
    reward_update_plan: Dict[str, Any],
    *,
    allowed_pbrs_fields: set[str] | None = None,
    allow_term_updates: bool = True,
) -> Dict[str, Any]:
    validated_spec = validate_reward_spec(reward_spec)
    validated_plan, _warnings = sanitize_reward_update_plan(reward_update_plan)

    updated_spec = deepcopy(validated_spec)
    terms_by_name = {term["name"]: term for term in updated_spec["terms"]}

    if allow_term_updates:
        for term_name in validated_plan["disable_terms"]:
            if term_name in terms_by_name:
                terms_by_name[term_name]["enabled"] = False

        for update in validated_plan["trigger_variant_updates"]:
            term_name = update["name"]
            if term_name in terms_by_name:
                terms_by_name[term_name]["trigger_variant"] = update["trigger_variant"]

        for update in validated_plan["weight_updates"]:
            term_name = update["name"]
            if term_name not in terms_by_name:
                continue
            term = terms_by_name[term_name]
            direction = update["direction"]
            if direction == "keep":
                continue
            if direction == "set":
                term["weight"] = float(update["value"])
                continue
            term_definition = get_term_definition(term_name)
            weight_min = term_definition["weight_range"]["min"]
            weight_max = term_definition["weight_range"]["max"]
            current_weight = float(term["weight"])
            if direction == "increase":
                term["weight"] = min(weight_max, current_weight + DEFAULT_WEIGHT_STEP)
            elif direction == "decrease":
                term["weight"] = max(weight_min, current_weight - DEFAULT_WEIGHT_STEP)

        preserve_terms = set(validated_plan["preserve_terms"])
        for term_name in preserve_terms:
            if term_name in terms_by_name:
                terms_by_name[term_name]["enabled"] = True

    pbrs_config = dict(updated_spec.get("pbrs", {}))
    pbrs_config.setdefault("gate", dict(pbrs_config.get("gate", {}) or {}))
    pbrs_config.setdefault("closeness", dict(pbrs_config.get("closeness", {}) or {}))
    for update in validated_plan["pbrs_updates"]:
        field_name = update["name"]
        if allowed_pbrs_fields is not None and field_name not in allowed_pbrs_fields:
            continue
        direction = update["direction"]
        if direction == "keep":
            continue
        if field_name in PBRS_CATEGORICAL_FIELDS:
            if direction == "set":
                if field_name == "variant":
                    pbrs_config["variant"] = update["value"]
                elif field_name == "gate_mode":
                    pbrs_config.setdefault("gate", {})["mode"] = update["value"]
                elif field_name == "closeness_mode":
                    pbrs_config.setdefault("closeness", {})["mode"] = update["value"]
            continue
        if field_name == "gate_radius":
            current_value = float(
                (pbrs_config.get("gate", {}) or {}).get("radius", pbrs_config.get("semi_strict_gate_radius", 2))
            )
        else:
            current_value = float(pbrs_config.get(field_name, 0.0))
        lower, upper = PBRS_LIMITS[field_name]
        if direction == "set":
            target_value = float(update["value"])
            if field_name == "gate_radius":
                pbrs_config.setdefault("gate", {})["radius"] = int(round(target_value))
                pbrs_config["semi_strict_gate_radius"] = int(round(target_value))
            else:
                pbrs_config[field_name] = target_value
            continue
        step = DEFAULT_PBRS_STEP[field_name]
        if direction == "increase":
            new_value = min(upper, current_value + step)
        elif direction == "decrease":
            new_value = max(lower, current_value - step)
        else:
            continue
        if field_name == "gate_radius":
            pbrs_config.setdefault("gate", {})["radius"] = int(round(new_value))
            pbrs_config["semi_strict_gate_radius"] = int(round(new_value))
        else:
            pbrs_config[field_name] = new_value

    updated_spec["terms"] = list(terms_by_name.values())
    if pbrs_config:
        updated_spec["pbrs"] = pbrs_config
    return validate_reward_spec(updated_spec)


def derive_suggested_reward_spec(
    previous_round: Dict[str, Any],
    *,
    allowed_pbrs_fields: set[str] | None = None,
    allow_term_updates: bool = True,
) -> Dict[str, Any] | None:
    if not isinstance(previous_round, dict):
        return None

    reward_spec = previous_round.get("reward_spec")
    if not isinstance(reward_spec, dict):
        return None

    critic_response = previous_round.get("critic_response")
    if not isinstance(critic_response, dict):
        return None

    parsed = critic_response.get("parsed")
    if not isinstance(parsed, dict):
        return None

    reward_update_plan = parsed.get("reward_update_plan")
    if not isinstance(reward_update_plan, dict):
        return None

    return apply_reward_update_plan(
        reward_spec,
        reward_update_plan,
        allowed_pbrs_fields=allowed_pbrs_fields,
        allow_term_updates=allow_term_updates,
    )
