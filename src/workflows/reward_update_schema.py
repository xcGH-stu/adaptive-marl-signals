from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

from rewarding.spec_schema import (
    SUPPORTED_PBRS_CLOSENESS_MODES,
    SUPPORTED_PBRS_GATE_MODES,
    SUPPORTED_PBRS_VARIANTS,
    get_term_definition,
    get_term_names,
)
from workflows.reward_spec_budget import DEFAULT_REWARD_SPEC_CHANGE_BUDGET


SUPPORTED_WEIGHT_DIRECTIONS = {"increase", "decrease", "set", "keep"}
SUPPORTED_PBRS_FIELDS = {
    "beta",
    "wc",
    "wp",
    "gamma",
    "variant",
    "gate_mode",
    "gate_radius",
    "closeness_mode",
}
PBRS_FIELD_RANGES = {
    "beta": (0.0, 2.0),
    "wc": (0.0, 1.0),
    "wp": (0.0, 1.0),
    "gamma": (0.0, 1.0),
    "gate_radius": (1.0, 10.0),
}
PBRS_CATEGORICAL_FIELDS = {"variant", "gate_mode", "closeness_mode"}
PBRS_CATEGORICAL_VALUES = {
    "variant": SUPPORTED_PBRS_VARIANTS,
    "gate_mode": SUPPORTED_PBRS_GATE_MODES,
    "closeness_mode": SUPPORTED_PBRS_CLOSENESS_MODES,
}


def _resolve_pbrs_update_name(item: Dict[str, Any]) -> Any:
    if not isinstance(item, dict):
        return None
    if item.get("name") is not None:
        return item.get("name")
    return item.get("field")


def validate_reward_update_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("reward_update_plan must be a dict")

    validated = {
        "preserve_terms": _validate_term_list(plan.get("preserve_terms", []), "preserve_terms"),
        "disable_terms": _validate_term_list(plan.get("disable_terms", []), "disable_terms"),
        "weight_updates": _validate_weight_updates(plan.get("weight_updates", [])),
        "trigger_variant_updates": _validate_trigger_variant_updates(
            plan.get("trigger_variant_updates", [])
        ),
        "pbrs_updates": _validate_pbrs_updates(plan.get("pbrs_updates", [])),
    }
    return validated


def normalize_reward_update_plan(
    plan: Dict[str, Any],
    budget: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    validated = validate_reward_update_plan(plan)
    budget_config = dict(DEFAULT_REWARD_SPEC_CHANGE_BUDGET)
    if budget is not None:
        budget_config.update(budget)

    normalized = deepcopy(validated)
    normalized["disable_terms"] = normalized["disable_terms"][
        : budget_config["max_enable_changes"]
    ]
    normalized["weight_updates"] = normalized["weight_updates"][
        : budget_config["max_weight_updates"]
    ]
    normalized["trigger_variant_updates"] = normalized["trigger_variant_updates"][
        : budget_config["max_trigger_variant_updates"]
    ]
    normalized["pbrs_updates"] = normalized["pbrs_updates"][
        : budget_config.get("max_pbrs_updates", 0)
    ]
    return normalized


def sanitize_reward_update_plan(
    plan: Dict[str, Any],
    budget: Dict[str, int] | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    if not isinstance(plan, dict):
        raise ValueError("reward_update_plan must be a dict")

    warnings: List[str] = []
    valid_terms = set(get_term_names())

    preserve_terms = _sanitize_term_list(
        plan.get("preserve_terms", []),
        "preserve_terms",
        valid_terms,
        warnings,
    )
    disable_terms = _sanitize_term_list(
        plan.get("disable_terms", []),
        "disable_terms",
        valid_terms,
        warnings,
    )
    weight_updates = _sanitize_weight_updates(
        plan.get("weight_updates", []),
        valid_terms,
        warnings,
    )
    trigger_variant_updates = _sanitize_trigger_variant_updates(
        plan.get("trigger_variant_updates", []),
        valid_terms,
        warnings,
    )
    pbrs_updates = _sanitize_pbrs_updates(
        plan.get("pbrs_updates", []),
        warnings,
    )

    sanitized = {
        "preserve_terms": preserve_terms,
        "disable_terms": disable_terms,
        "weight_updates": weight_updates,
        "trigger_variant_updates": trigger_variant_updates,
        "pbrs_updates": pbrs_updates,
    }
    return normalize_reward_update_plan(sanitized, budget=budget), warnings


def _validate_term_list(value: Any, field_name: str) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"reward_update_plan.{field_name} must be a list")
    valid_terms = set(get_term_names())
    validated: List[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"reward_update_plan.{field_name}[{index}] must be a string")
        if item not in valid_terms:
            raise ValueError(
                f"reward_update_plan.{field_name}[{index}] must be one of: "
                + ", ".join(sorted(valid_terms))
            )
        validated.append(item)
    return validated


def _sanitize_term_list(
    value: Any,
    field_name: str,
    valid_terms: set[str],
    warnings: List[str],
) -> List[str]:
    if not isinstance(value, list):
        warnings.append(f"reward_update_plan.{field_name} must be a list; dropped field")
        return []
    validated: List[str] = []
    seen = set()
    for index, item in enumerate(value):
        if not isinstance(item, str):
            warnings.append(
                f"reward_update_plan.{field_name}[{index}] must be a string; dropped item"
            )
            continue
        if item not in valid_terms:
            warnings.append(
                f"reward_update_plan.{field_name}[{index}] has unknown term {item!r}; dropped item"
            )
            continue
        if item in seen:
            continue
        validated.append(item)
        seen.add(item)
    return validated


def _validate_weight_updates(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("reward_update_plan.weight_updates must be a list")
    valid_terms = set(get_term_names())
    validated: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"reward_update_plan.weight_updates[{index}] must be a dict")
        name = item.get("name")
        if name not in valid_terms:
            raise ValueError(
                f"reward_update_plan.weight_updates[{index}].name must be one of: "
                + ", ".join(sorted(valid_terms))
            )
        direction = item.get("direction")
        if direction not in SUPPORTED_WEIGHT_DIRECTIONS:
            raise ValueError(
                f"reward_update_plan.weight_updates[{index}].direction must be one of: "
                + ", ".join(sorted(SUPPORTED_WEIGHT_DIRECTIONS))
            )
        validated_item = {"name": name, "direction": direction}
        if direction == "set":
            if "value" not in item or not isinstance(item["value"], (int, float)):
                raise ValueError(
                    f'reward_update_plan.weight_updates[{index}] with direction "set" must include numeric "value"'
                )
            term_definition = get_term_definition(name)
            weight_min = term_definition["weight_range"]["min"]
            weight_max = term_definition["weight_range"]["max"]
            value_float = float(item["value"])
            if value_float < weight_min or value_float > weight_max:
                raise ValueError(
                    f"reward_update_plan.weight_updates[{index}].value for {name} must be between "
                    f"{weight_min} and {weight_max}"
                )
            validated_item["value"] = value_float
        validated.append(validated_item)
    return validated


def _sanitize_weight_updates(
    value: Any,
    valid_terms: set[str],
    warnings: List[str],
) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        warnings.append("reward_update_plan.weight_updates must be a list; dropped field")
        return []
    validated: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            warnings.append(
                f"reward_update_plan.weight_updates[{index}] must be a dict; dropped item"
            )
            continue
        name = item.get("name")
        if name not in valid_terms:
            warnings.append(
                f"reward_update_plan.weight_updates[{index}] has unknown term {name!r}; dropped item"
            )
            continue
        direction = item.get("direction")
        if direction not in SUPPORTED_WEIGHT_DIRECTIONS:
            warnings.append(
                f"reward_update_plan.weight_updates[{index}] has invalid direction {direction!r}; dropped item"
            )
            continue
        validated_item = {"name": name, "direction": direction}
        if direction == "set":
            if "value" not in item or not isinstance(item["value"], (int, float)):
                warnings.append(
                    f"reward_update_plan.weight_updates[{index}] set operation is missing numeric value; dropped item"
                )
                continue
            term_definition = get_term_definition(name)
            weight_min = term_definition["weight_range"]["min"]
            weight_max = term_definition["weight_range"]["max"]
            value_float = float(item["value"])
            if value_float < weight_min or value_float > weight_max:
                warnings.append(
                    f"reward_update_plan.weight_updates[{index}] set value {value_float} for {name} is outside [{weight_min}, {weight_max}]; dropped item"
                )
                continue
            validated_item["value"] = value_float
        validated.append(validated_item)
    return validated


def _validate_trigger_variant_updates(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("reward_update_plan.trigger_variant_updates must be a list")
    validated: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"reward_update_plan.trigger_variant_updates[{index}] must be a dict"
            )
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError(
                f"reward_update_plan.trigger_variant_updates[{index}].name must be a string"
            )
        term_definition = get_term_definition(name)
        trigger_variant = item.get("trigger_variant")
        supported_variants = term_definition["trigger_variants"]
        if trigger_variant not in supported_variants:
            raise ValueError(
                f"reward_update_plan.trigger_variant_updates[{index}].trigger_variant for {name} "
                "must be one of: " + ", ".join(sorted(supported_variants))
            )
        validated.append({"name": name, "trigger_variant": trigger_variant})
    return validated


def _sanitize_trigger_variant_updates(
    value: Any,
    valid_terms: set[str],
    warnings: List[str],
) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        warnings.append(
            "reward_update_plan.trigger_variant_updates must be a list; dropped field"
        )
        return []
    validated: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            warnings.append(
                f"reward_update_plan.trigger_variant_updates[{index}] must be a dict; dropped item"
            )
            continue
        name = item.get("name")
        if not isinstance(name, str) or name not in valid_terms:
            warnings.append(
                f"reward_update_plan.trigger_variant_updates[{index}] has unknown term {name!r}; dropped item"
            )
            continue
        trigger_variant = item.get("trigger_variant")
        supported_variants = get_term_definition(name)["trigger_variants"]
        if trigger_variant not in supported_variants:
            warnings.append(
                f"reward_update_plan.trigger_variant_updates[{index}] has invalid trigger_variant {trigger_variant!r} for {name}; dropped item"
            )
            continue
        validated.append({"name": name, "trigger_variant": trigger_variant})
    return validated


def _validate_pbrs_updates(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("reward_update_plan.pbrs_updates must be a list")
    validated: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"reward_update_plan.pbrs_updates[{index}] must be a dict")
        name = _resolve_pbrs_update_name(item)
        if name not in SUPPORTED_PBRS_FIELDS:
            raise ValueError(
                f"reward_update_plan.pbrs_updates[{index}].name must be one of: "
                + ", ".join(sorted(SUPPORTED_PBRS_FIELDS))
            )
        direction = item.get("direction")
        if direction not in SUPPORTED_WEIGHT_DIRECTIONS:
            raise ValueError(
                f"reward_update_plan.pbrs_updates[{index}].direction must be one of: "
                + ", ".join(sorted(SUPPORTED_WEIGHT_DIRECTIONS))
            )
        validated_item = {"name": name, "direction": direction}
        if name in PBRS_CATEGORICAL_FIELDS and direction not in {"keep", "set"}:
            raise ValueError(
                f"reward_update_plan.pbrs_updates[{index}].direction for {name} "
                'must be "keep" or "set"'
            )
        if direction == "set":
            if "value" not in item:
                raise ValueError(
                    f'reward_update_plan.pbrs_updates[{index}] with direction "set" must include "value"'
                )
            if name in PBRS_CATEGORICAL_FIELDS:
                value = item["value"]
                if value not in PBRS_CATEGORICAL_VALUES[name]:
                    raise ValueError(
                        f"reward_update_plan.pbrs_updates[{index}].value for {name} must be one of: "
                        + ", ".join(sorted(PBRS_CATEGORICAL_VALUES[name]))
                    )
                validated_item["value"] = value
            else:
                if not isinstance(item["value"], (int, float)):
                    raise ValueError(
                        f'reward_update_plan.pbrs_updates[{index}] with direction "set" must include numeric "value"'
                    )
                value_float = float(item["value"])
                lower, upper = PBRS_FIELD_RANGES[name]
                if value_float < lower or value_float > upper:
                    raise ValueError(
                        f"reward_update_plan.pbrs_updates[{index}].value for {name} must be between "
                        f"{lower} and {upper}"
                    )
                validated_item["value"] = value_float
        validated.append(validated_item)
    return validated


def _sanitize_pbrs_updates(
    value: Any,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        warnings.append("reward_update_plan.pbrs_updates must be a list; dropped field")
        return []
    validated: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            warnings.append(
                f"reward_update_plan.pbrs_updates[{index}] must be a dict; dropped item"
            )
            continue
        name = _resolve_pbrs_update_name(item)
        if name not in SUPPORTED_PBRS_FIELDS:
            warnings.append(
                f"reward_update_plan.pbrs_updates[{index}] has unknown PBRS field {name!r}; dropped item"
            )
            continue
        direction = item.get("direction")
        if direction not in SUPPORTED_WEIGHT_DIRECTIONS:
            warnings.append(
                f"reward_update_plan.pbrs_updates[{index}] has invalid direction {direction!r}; dropped item"
            )
            continue
        validated_item = {"name": name, "direction": direction}
        if name in PBRS_CATEGORICAL_FIELDS and direction not in {"keep", "set"}:
            warnings.append(
                f"reward_update_plan.pbrs_updates[{index}] has invalid direction {direction!r} for categorical field {name!r}; dropped item"
            )
            continue
        if direction == "set":
            if "value" not in item:
                warnings.append(
                    f"reward_update_plan.pbrs_updates[{index}] set operation is missing value; dropped item"
                )
                continue
            if name in PBRS_CATEGORICAL_FIELDS:
                value = item["value"]
                if value not in PBRS_CATEGORICAL_VALUES[name]:
                    warnings.append(
                        f"reward_update_plan.pbrs_updates[{index}] set value {value!r} for {name} is invalid; dropped item"
                    )
                    continue
                validated_item["value"] = value
            else:
                if not isinstance(item["value"], (int, float)):
                    warnings.append(
                        f"reward_update_plan.pbrs_updates[{index}] set operation is missing numeric value; dropped item"
                    )
                    continue
                value_float = float(item["value"])
                lower, upper = PBRS_FIELD_RANGES[name]
                if value_float < lower or value_float > upper:
                    warnings.append(
                        f"reward_update_plan.pbrs_updates[{index}] set value {value_float} for {name} is outside [{lower}, {upper}]; dropped item"
                    )
                    continue
                validated_item["value"] = value_float
        validated.append(validated_item)
    return validated
