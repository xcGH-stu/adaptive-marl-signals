from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from workflows.alpha_policy import validate_alpha_policy


REWARD_SPEC_VERSION = 1
SUPPORTED_PBRS_VARIANTS = {
    "original",
    "semi_strict_gate",
}
SUPPORTED_PBRS_GATE_MODES = {
    "topk_level_sum",
    "feasible_assignment",
}
SUPPORTED_PBRS_CLOSENESS_MODES = {
    "topk_mean",
    "topk_nearest",
    "assignment_mean",
    "assignment_nearest",
}

DEFAULT_PBRS_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "variant": "original",
    "beta": 0.3,
    "gamma": 0.99,
    "wc": 0.6,
    "wp": 0.4,
    "phi_clip": [0.0, 1.0],
    "semi_strict_gate_radius": 2,
    "gate": {
        "mode": "topk_level_sum",
        "radius": 2,
    },
    "closeness": {
        "mode": "topk_mean",
    },
}

SUPPORTED_REWARD_TERMS: Dict[str, Dict[str, Any]] = {
    "approach_target_food": {
        "description": (
            "Dense shaping for making progress toward a target food. "
            "This term should remain easier to trigger than true task success."
        ),
        "trigger_variants": {
            "distance_decrease": {
                "description": "Reward decreases in distance to a target food.",
            },
            "distance_decrease_not_adjacent": {
                "description": (
                    "Reward distance decrease only before the agent becomes adjacent "
                    "to the target food."
                ),
            },
            "distance_decrease_same_target": {
                "description": (
                    "Reward distance decrease when multiple agents are implicitly "
                    "converging on the same target food."
                ),
            },
        },
        "weight_range": {"min": -1.0, "max": 1.0},
    },
    "coordination_ready": {
        "description": (
            "Reward intermediate states that are closer to a successful coordinated "
            "load than simple movement alone."
        ),
        "trigger_variants": {
            "shared_food_adjacent": {
                "description": "Agents become adjacent to the same food.",
            },
            "coalition_level_sufficient": {
                "description": (
                    "Reward states where nearby agents' combined level is sufficient "
                    "for the target food."
                ),
            },
            "coalition_level_improves": {
                "description": (
                    "Reward steps that improve coalition readiness for a target food."
                ),
            },
        },
        "weight_range": {"min": -1.0, "max": 1.0},
    },
    "load_attempt": {
        "description": (
            "Reward a load attempt that is more task-aligned than generic movement, "
            "without collapsing into sparse success duplication."
        ),
        "trigger_variants": {
            "any_adjacent_load": {
                "description": "Reward any LOAD action when the agent is adjacent to food.",
            },
            "coalition_valid_load": {
                "description": (
                    "Reward LOAD attempts only when coalition conditions indicate a "
                    "plausible coordinated attempt."
                ),
            },
        },
        "weight_range": {"min": -1.0, "max": 1.0},
    },
    "successful_load": {
        "description": (
            "Reward events tightly coupled to successful collection, but still expose "
            "the design explicitly as a shaped term."
        ),
        "trigger_variants": {
            "positive_sparse_reward": {
                "description": "Trigger when the underlying sparse reward is positive.",
            },
            "food_removed": {
                "description": "Trigger when a food item is actually removed from the field.",
            },
        },
        "weight_range": {"min": -1.0, "max": 1.0},
    },
    "failed_load_penalty": {
        "description": (
            "Penalize unsuccessful load attempts to discourage repeated invalid or "
            "poorly prepared coordination attempts."
        ),
        "trigger_variants": {
            "failed_load": {
                "description": "Penalize failed LOAD attempts.",
            },
            "failed_load_when_adjacent": {
                "description": (
                    "Penalize failed LOAD attempts only when the agent is adjacent to food."
                ),
            },
        },
        "weight_range": {"min": -1.0, "max": 0.0},
    },
}


DEFAULT_REWARD_SPEC: Dict[str, Any] = {
    "version": REWARD_SPEC_VERSION,
    "terms": [
        {
            "name": "approach_target_food",
            "enabled": True,
            "weight": 0.03,
            "trigger_variant": "distance_decrease_same_target",
            "constraints": {
                "only_when_not_adjacent": True,
            },
        },
        {
            "name": "coordination_ready",
            "enabled": True,
            "weight": 0.02,
            "trigger_variant": "shared_food_adjacent",
            "constraints": {},
        },
        {
            "name": "load_attempt",
            "enabled": False,
            "weight": 0.02,
            "trigger_variant": "coalition_valid_load",
            "constraints": {},
        },
        {
            "name": "successful_load",
            "enabled": False,
            "weight": 0.05,
            "trigger_variant": "food_removed",
            "constraints": {},
        },
        {
            "name": "failed_load_penalty",
            "enabled": True,
            "weight": -0.05,
            "trigger_variant": "failed_load",
            "constraints": {},
        },
    ],
    "alpha_policy": {
        "type": "constant",
        "value": 1.0,
    },
    "pbrs": deepcopy(DEFAULT_PBRS_CONFIG),
}


def get_default_reward_spec() -> Dict[str, Any]:
    return deepcopy(DEFAULT_REWARD_SPEC)


def get_supported_reward_terms() -> Dict[str, Dict[str, Any]]:
    return deepcopy(SUPPORTED_REWARD_TERMS)


def validate_reward_spec(reward_spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(reward_spec, dict):
        raise ValueError("reward_spec must be a dict")

    version = reward_spec.get("version", REWARD_SPEC_VERSION)
    if version != REWARD_SPEC_VERSION:
        raise ValueError(
            f"reward_spec.version must equal {REWARD_SPEC_VERSION} for the first scaffold version"
        )

    terms = reward_spec.get("terms")
    if not isinstance(terms, list) or not terms:
        raise ValueError('reward_spec must include a non-empty "terms" list')

    validated_terms = []
    seen_names = set()
    for index, term in enumerate(terms):
        validated_term = _validate_term(term, index=index)
        term_name = validated_term["name"]
        if term_name in seen_names:
            raise ValueError(f'reward_spec.terms contains duplicate term "{term_name}"')
        seen_names.add(term_name)
        validated_terms.append(validated_term)

    alpha_policy = reward_spec.get("alpha_policy")
    if alpha_policy is not None:
        validate_alpha_policy(alpha_policy)

    pbrs = validate_pbrs_config(reward_spec.get("pbrs"))

    return {
        "version": version,
        "terms": validated_terms,
        "alpha_policy": alpha_policy,
        "pbrs": pbrs,
    }


def _validate_term(term: Dict[str, Any], *, index: int) -> Dict[str, Any]:
    if not isinstance(term, dict):
        raise ValueError(f"reward_spec.terms[{index}] must be a dict")

    term_name = term.get("name")
    if term_name not in SUPPORTED_REWARD_TERMS:
        raise ValueError(
            "reward_spec.terms[{index}].name must be one of: {names}".format(
                index=index,
                names=", ".join(sorted(SUPPORTED_REWARD_TERMS)),
            )
        )

    enabled = term.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError(f"reward_spec.terms[{index}].enabled must be a bool")

    if "weight" not in term:
        raise ValueError(f'reward_spec.terms[{index}] must include key "weight"')
    weight = term["weight"]
    if not isinstance(weight, (int, float)):
        raise ValueError(f"reward_spec.terms[{index}].weight must be numeric")

    weight_min = SUPPORTED_REWARD_TERMS[term_name]["weight_range"]["min"]
    weight_max = SUPPORTED_REWARD_TERMS[term_name]["weight_range"]["max"]
    if weight < weight_min or weight > weight_max:
        raise ValueError(
            f"reward_spec.terms[{index}].weight for {term_name} must be between "
            f"{weight_min} and {weight_max}"
        )

    trigger_variant = term.get("trigger_variant")
    supported_variants = SUPPORTED_REWARD_TERMS[term_name]["trigger_variants"]
    if trigger_variant not in supported_variants:
        raise ValueError(
            f"reward_spec.terms[{index}].trigger_variant for {term_name} must be one of: "
            + ", ".join(sorted(supported_variants))
        )

    constraints = term.get("constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError(f"reward_spec.terms[{index}].constraints must be a dict")

    return {
        "name": term_name,
        "enabled": enabled,
        "weight": float(weight),
        "trigger_variant": trigger_variant,
        "constraints": deepcopy(constraints),
    }


def get_term_definition(term_name: str) -> Dict[str, Any]:
    if term_name not in SUPPORTED_REWARD_TERMS:
        raise KeyError(f"Unsupported reward term: {term_name}")
    return deepcopy(SUPPORTED_REWARD_TERMS[term_name])


def get_term_names() -> List[str]:
    return sorted(SUPPORTED_REWARD_TERMS)


def get_default_pbrs_config() -> Dict[str, Any]:
    return deepcopy(DEFAULT_PBRS_CONFIG)


def get_supported_pbrs_design_space() -> Dict[str, Any]:
    return {
        "variants": sorted(SUPPORTED_PBRS_VARIANTS),
        "gate_modes": sorted(SUPPORTED_PBRS_GATE_MODES),
        "closeness_modes": sorted(SUPPORTED_PBRS_CLOSENESS_MODES),
    }


def _variant_default_gate_mode(variant: str) -> str:
    return "feasible_assignment" if variant == "semi_strict_gate" else "topk_level_sum"


def _variant_default_closeness_mode(variant: str) -> str:
    return "assignment_mean" if variant == "semi_strict_gate" else "topk_mean"


def validate_pbrs_config(pbrs_config: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_PBRS_CONFIG)
    if pbrs_config is None:
        return merged
    if not isinstance(pbrs_config, dict):
        raise ValueError("reward_spec.pbrs must be a dict when provided")

    merged.update(deepcopy(pbrs_config))

    enabled = merged.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("reward_spec.pbrs.enabled must be a bool")

    variant = merged.get("variant")
    if variant not in SUPPORTED_PBRS_VARIANTS:
        raise ValueError(
            "reward_spec.pbrs.variant must be one of: "
            + ", ".join(sorted(SUPPORTED_PBRS_VARIANTS))
        )

    beta = merged.get("beta")
    gamma = merged.get("gamma")
    wc = merged.get("wc")
    wp = merged.get("wp")
    for field_name, value in {
        "beta": beta,
        "gamma": gamma,
        "wc": wc,
        "wp": wp,
    }.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"reward_spec.pbrs.{field_name} must be numeric")

    beta = float(beta)
    gamma = float(gamma)
    wc = float(wc)
    wp = float(wp)
    if beta < 0.0 or beta > 2.0:
        raise ValueError("reward_spec.pbrs.beta must be between 0.0 and 2.0")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("reward_spec.pbrs.gamma must be between 0.0 and 1.0")
    if wc < 0.0 or wc > 1.0:
        raise ValueError("reward_spec.pbrs.wc must be between 0.0 and 1.0")
    if wp < 0.0 or wp > 1.0:
        raise ValueError("reward_spec.pbrs.wp must be between 0.0 and 1.0")
    if wc == 0.0 and wp == 0.0:
        raise ValueError("reward_spec.pbrs.wc and reward_spec.pbrs.wp cannot both be 0.0")

    phi_clip = merged.get("phi_clip")
    if (
        not isinstance(phi_clip, list)
        or len(phi_clip) != 2
        or not all(isinstance(value, (int, float)) for value in phi_clip)
    ):
        raise ValueError("reward_spec.pbrs.phi_clip must be a 2-item numeric list")
    phi_min = float(phi_clip[0])
    phi_max = float(phi_clip[1])
    if phi_min > phi_max:
        raise ValueError("reward_spec.pbrs.phi_clip[0] must be <= phi_clip[1]")

    semi_strict_gate_radius = merged.get("semi_strict_gate_radius")
    if not isinstance(semi_strict_gate_radius, int) or semi_strict_gate_radius <= 0:
        raise ValueError(
            "reward_spec.pbrs.semi_strict_gate_radius must be a positive int"
        )

    gate_config = merged.get("gate", {})
    if gate_config is None:
        gate_config = {}
    if not isinstance(gate_config, dict):
        raise ValueError("reward_spec.pbrs.gate must be a dict")
    gate_mode = gate_config.get("mode", _variant_default_gate_mode(variant))
    if gate_mode not in SUPPORTED_PBRS_GATE_MODES:
        raise ValueError(
            "reward_spec.pbrs.gate.mode must be one of: "
            + ", ".join(sorted(SUPPORTED_PBRS_GATE_MODES))
        )
    gate_radius = gate_config.get("radius", semi_strict_gate_radius)
    if not isinstance(gate_radius, int) or gate_radius <= 0:
        raise ValueError("reward_spec.pbrs.gate.radius must be a positive int")

    closeness_config = merged.get("closeness", {})
    if closeness_config is None:
        closeness_config = {}
    if not isinstance(closeness_config, dict):
        raise ValueError("reward_spec.pbrs.closeness must be a dict")
    closeness_mode = closeness_config.get("mode", _variant_default_closeness_mode(variant))
    if closeness_mode not in SUPPORTED_PBRS_CLOSENESS_MODES:
        raise ValueError(
            "reward_spec.pbrs.closeness.mode must be one of: "
            + ", ".join(sorted(SUPPORTED_PBRS_CLOSENESS_MODES))
        )

    merged["beta"] = beta
    merged["gamma"] = gamma
    merged["wc"] = wc
    merged["wp"] = wp
    merged["phi_clip"] = [phi_min, phi_max]
    merged["semi_strict_gate_radius"] = int(gate_radius)
    merged["gate"] = {
        "mode": gate_mode,
        "radius": int(gate_radius),
    }
    merged["closeness"] = {
        "mode": closeness_mode,
    }
    return merged
