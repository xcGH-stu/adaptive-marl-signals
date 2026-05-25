from __future__ import annotations

from typing import Any, Dict

from rewarding.spec_schema import validate_reward_spec


DEFAULT_REWARD_SPEC_CHANGE_BUDGET = {
    "max_weight_updates": 2,
    "max_trigger_variant_updates": 1,
    "max_enable_changes": 1,
    "max_pbrs_updates": 1,
}


def validate_reward_spec_change_budget(budget: Dict[str, Any] | None) -> Dict[str, int]:
    merged = dict(DEFAULT_REWARD_SPEC_CHANGE_BUDGET)
    if budget is None:
        return merged
    if not isinstance(budget, dict):
        raise ValueError("reward_spec_change_budget must be a dict")
    for key in DEFAULT_REWARD_SPEC_CHANGE_BUDGET:
        if key in budget:
            value = budget[key]
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"reward_spec_change_budget.{key} must be a non-negative int")
            merged[key] = value
    return merged


def summarize_reward_spec_changes(
    base_spec: Dict[str, Any],
    candidate_spec: Dict[str, Any],
) -> Dict[str, int]:
    validated_base = validate_reward_spec(base_spec)
    validated_candidate = validate_reward_spec(candidate_spec)
    base_terms = {term["name"]: term for term in validated_base["terms"]}
    candidate_terms = {term["name"]: term for term in validated_candidate["terms"]}

    summary = {
        "weight_updates": 0,
        "trigger_variant_updates": 0,
        "enable_changes": 0,
        "pbrs_updates": 0,
    }

    for term_name, base_term in base_terms.items():
        candidate_term = candidate_terms.get(term_name)
        if candidate_term is None:
            summary["enable_changes"] += 1
            continue
        if bool(base_term["enabled"]) != bool(candidate_term["enabled"]):
            summary["enable_changes"] += 1
        if float(base_term["weight"]) != float(candidate_term["weight"]):
            summary["weight_updates"] += 1
        if base_term["trigger_variant"] != candidate_term["trigger_variant"]:
            summary["trigger_variant_updates"] += 1

    for term_name in candidate_terms:
        if term_name not in base_terms:
            summary["enable_changes"] += 1

    base_pbrs = dict(validated_base.get("pbrs", {}))
    candidate_pbrs = dict(validated_candidate.get("pbrs", {}))
    for key in {
        "enabled",
        "variant",
        "beta",
        "gamma",
        "wc",
        "wp",
        "phi_clip",
        "semi_strict_gate_radius",
        "gate",
        "closeness",
    }:
        if base_pbrs.get(key) != candidate_pbrs.get(key):
            summary["pbrs_updates"] += 1

    return summary


def enforce_reward_spec_change_budget(
    *,
    base_spec: Dict[str, Any],
    candidate_spec: Dict[str, Any],
    budget: Dict[str, Any] | None = None,
) -> Dict[str, int]:
    validated_budget = validate_reward_spec_change_budget(budget)
    summary = summarize_reward_spec_changes(base_spec, candidate_spec)

    violations = []
    if summary["weight_updates"] > validated_budget["max_weight_updates"]:
        violations.append(
            f'weight_updates={summary["weight_updates"]} exceeds max_weight_updates='
            f'{validated_budget["max_weight_updates"]}'
        )
    if summary["trigger_variant_updates"] > validated_budget["max_trigger_variant_updates"]:
        violations.append(
            f'trigger_variant_updates={summary["trigger_variant_updates"]} exceeds '
            f'max_trigger_variant_updates={validated_budget["max_trigger_variant_updates"]}'
        )
    if summary["enable_changes"] > validated_budget["max_enable_changes"]:
        violations.append(
            f'enable_changes={summary["enable_changes"]} exceeds max_enable_changes='
            f'{validated_budget["max_enable_changes"]}'
        )
    if summary["pbrs_updates"] > validated_budget["max_pbrs_updates"]:
        violations.append(
            f'pbrs_updates={summary["pbrs_updates"]} exceeds max_pbrs_updates='
            f'{validated_budget["max_pbrs_updates"]}'
        )

    if violations:
        raise ValueError(
            "reward_spec change budget exceeded: " + "; ".join(violations)
        )

    return summary
