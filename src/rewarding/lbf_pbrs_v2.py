from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, List, Sequence

from rewarding.lbf_pbrs import AgentState, FoodState, LBFStateSnapshot


TERM_NAMES = ("col", "app", "cov", "ready", "alloc", "stab")
DEFAULT_MODE = "balanced_collection_ready"
DEFAULT_GAMMA = 0.99
DEFAULT_BETA = 0.50

DEFAULT_MODE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "balanced_collection_ready": {
        "beta": 0.50,
        "weights": {
            "col": 0.25,
            "app": 0.25,
            "cov": 0.25,
            "ready": 0.25,
            "alloc": 0.00,
            "stab": 0.00,
        },
    },
    "coverage_ready_balance": {
        "beta": 0.50,
        "weights": {
            "col": 0.20,
            "app": 0.15,
            "cov": 0.30,
            "ready": 0.25,
            "alloc": 0.10,
            "stab": 0.00,
        },
    },
    "approach_collection_push": {
        "beta": 0.55,
        "weights": {
            "col": 0.25,
            "app": 0.35,
            "cov": 0.20,
            "ready": 0.15,
            "alloc": 0.05,
            "stab": 0.00,
        },
    },
    "allocation_stability_support": {
        "beta": 0.40,
        "weights": {
            "col": 0.15,
            "app": 0.10,
            "cov": 0.20,
            "ready": 0.20,
            "alloc": 0.15,
            "stab": 0.20,
        },
    },
}

LBF_PBRS_V2_DEFAULT_CONFIG: Dict[str, Any] = {
    "pbrs_version": "lbf_pbrs_v2",
    "mode": DEFAULT_MODE,
    "beta": DEFAULT_BETA,
    "gamma": DEFAULT_GAMMA,
    "weights": deepcopy(DEFAULT_MODE_CONFIGS[DEFAULT_MODE]["weights"]),
    "active_terms": ["col", "app", "cov", "ready"],
}


def _clamp_unit_interval(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    return min(1.0, max(0.0, numeric))


def _normalize_term_subset(active_terms: Sequence[Any] | None) -> List[str]:
    normalized: List[str] = []
    for term in list(active_terms or []):
        term_text = str(term or "").strip()
        if term_text in TERM_NAMES and term_text not in normalized:
            normalized.append(term_text)
    return normalized


def legacy_lbf_candidate_to_v2(
    *,
    beta: Any,
    wc: Any,
    wp: Any = None,
    candidate_id: Any = None,
    candidate_type: Any = None,
    evidence_keys_used: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    wc_value = _clamp_unit_interval(wc, default=0.5)
    if wp is None:
        wp_value = max(0.0, 1.0 - wc_value)
    else:
        wp_value = _clamp_unit_interval(wp, default=max(0.0, 1.0 - wc_value))
    total = wc_value + wp_value
    if total <= 0.0:
        wc_value = 0.5
        wp_value = 0.5
        total = 1.0
    wc_value /= total
    wp_value /= total
    return normalize_lbf_pbrs_v2_config(
        {
            "pbrs_version": "lbf_pbrs_v2",
            "mode": DEFAULT_MODE,
            "beta": beta,
            "active_terms": ["col", "app", "cov", "ready"],
            "weights": {
                "col": wc_value * 0.5,
                "ready": wc_value * 0.5,
                "app": wp_value * 0.5,
                "cov": wp_value * 0.5,
                "alloc": 0.0,
                "stab": 0.0,
            },
            "candidate_id": candidate_id,
            "candidate_type": candidate_type,
            "evidence_keys_used": list(evidence_keys_used or []),
            "legacy_projection": True,
        }
    )


def sanitize_lbf_pbrs_v2_config(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = deepcopy(config or {})
    if not isinstance(payload, dict):
        raise ValueError("lbf_pbrs_v2 config must be a dict")
    mode = str(payload.get("mode") or DEFAULT_MODE)
    if mode not in DEFAULT_MODE_CONFIGS:
        raise ValueError(f"Unsupported LBF PBRS-v2 mode: {mode}")
    defaults = deepcopy(DEFAULT_MODE_CONFIGS[mode])
    weights = deepcopy(defaults.get("weights") or {})
    user_weights = payload.get("weights") or {}
    if user_weights and not isinstance(user_weights, dict):
        raise ValueError("lbf_pbrs_v2 weights must be a dict")
    for term in TERM_NAMES:
        if term in user_weights:
            weights[term] = float(user_weights[term] or 0.0)
        else:
            weights.setdefault(term, 0.0)
    active_terms = _normalize_term_subset(payload.get("active_terms"))
    return {
        "pbrs_version": str(payload.get("pbrs_version") or "lbf_pbrs_v2"),
        "mode": mode,
        "beta": float(payload.get("beta", defaults.get("beta", DEFAULT_BETA)) or 0.0),
        "gamma": float(payload.get("gamma", DEFAULT_GAMMA) or DEFAULT_GAMMA),
        "weights": weights,
        "active_terms": active_terms,
        "candidate_id": str(payload.get("candidate_id") or ""),
        "candidate_type": str(payload.get("candidate_type") or ""),
        "evidence_keys_used": [str(item) for item in list(payload.get("evidence_keys_used") or []) if str(item)],
        "legacy_projection": bool(payload.get("legacy_projection", False)),
    }


def normalize_lbf_pbrs_v2_config(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = sanitize_lbf_pbrs_v2_config(config)
    normalized = deepcopy(payload)
    normalized["beta"] = _clamp_unit_interval(normalized["beta"], default=DEFAULT_BETA)
    normalized["gamma"] = _clamp_unit_interval(normalized["gamma"], default=DEFAULT_GAMMA)
    raw_weights = {
        term: max(0.0, float(payload["weights"].get(term, 0.0)))
        for term in TERM_NAMES
    }
    requested_active_terms = list(payload.get("active_terms") or [])
    if requested_active_terms:
        for term in TERM_NAMES:
            if term not in requested_active_terms:
                raw_weights[term] = 0.0
    weight_sum = sum(raw_weights.values())
    if weight_sum <= 0.0:
        defaults = deepcopy(DEFAULT_MODE_CONFIGS[normalized["mode"]]["weights"])
        raw_weights = {term: float(defaults.get(term, 0.0)) for term in TERM_NAMES}
        if requested_active_terms:
            for term in TERM_NAMES:
                if term not in requested_active_terms:
                    raw_weights[term] = 0.0
        weight_sum = sum(raw_weights.values())
    if weight_sum <= 0.0:
        raise ValueError("LBF PBRS-v2 candidate weights must sum to a positive value")
    normalized["weights"] = {
        term: float(raw_weights[term]) / float(weight_sum) for term in TERM_NAMES
    }
    normalized["active_terms"] = (
        requested_active_terms
        if requested_active_terms
        else [term for term in TERM_NAMES if normalized["weights"][term] > 0.0]
    )
    normalized["wc"] = round(
        float(normalized["weights"]["col"]) + float(normalized["weights"]["ready"]),
        10,
    )
    normalized["wp"] = round(
        float(normalized["weights"]["app"])
        + float(normalized["weights"]["cov"])
        + float(normalized["weights"]["alloc"])
        + float(normalized["weights"]["stab"]),
        10,
    )
    return normalized


class LBFPBRSV2Runtime:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = normalize_lbf_pbrs_v2_config(config)
        self.beta = float(self.config["beta"])
        self.gamma = float(self.config["gamma"])
        self.mode = str(self.config["mode"])
        self.weights = deepcopy(self.config["weights"])
        self._initial_total_food_value = 0.0
        self._agent_levels: List[int] = []

    def reset(self, state: LBFStateSnapshot) -> None:
        self._initial_total_food_value = float(sum(food.level for food in state.foods))
        self._agent_levels = [int(agent.level) for agent in state.agents]

    def apply_config(self, config: Dict[str, Any]) -> None:
        normalized = normalize_lbf_pbrs_v2_config(config)
        self.config = normalized
        self.beta = float(normalized["beta"])
        self.gamma = float(normalized["gamma"])
        self.mode = str(normalized["mode"])
        self.weights = deepcopy(normalized["weights"])

    def compute_terms(self, state: LBFStateSnapshot) -> Dict[str, float]:
        if self._initial_total_food_value <= 0.0:
            self.reset(state)
        if self._initial_total_food_value <= 0.0:
            return {term: 0.0 for term in TERM_NAMES}
        rows, cols = state.field_size
        d_max = max(float((rows - 1) + (cols - 1)), 1.0)
        total_food_value = float(sum(food.level for food in state.foods))
        collected_food_value = max(self._initial_total_food_value - total_food_value, 0.0)
        col = min(1.0, max(0.0, collected_food_value / self._initial_total_food_value))

        if not state.foods or not state.agents:
            return {
                "col": col,
                "app": 0.0,
                "cov": 0.0,
                "ready": 0.0,
                "alloc": 0.0,
                "stab": 0.0,
            }

        agent_closeness_values: List[float] = []
        covered_foods: set[int] = set()
        ready_score = 0.0
        alloc_slots = 0.0
        nearest_distances: List[float] = []

        for agent in state.agents:
            best_food_index = None
            best_distance = None
            for food_index, food in enumerate(state.foods):
                distance = _manhattan_distance(agent.position, food.position)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_food_index = food_index
            if best_distance is None:
                continue
            nearest_distances.append(float(best_distance))
            agent_closeness_values.append(max(0.0, 1.0 - (float(best_distance) / d_max)))
            if best_food_index is not None:
                covered_foods.add(int(best_food_index))

        for food in state.foods:
            nearby_agents = [
                agent
                for agent in state.agents
                if _manhattan_distance(agent.position, food.position) <= 2
            ]
            total_level = sum(int(agent.level) for agent in nearby_agents)
            readiness = min(1.0, float(total_level) / max(float(food.level), 1.0))
            ready_score += float(food.level) * readiness
            alloc_slots += min(1.0, len(nearby_agents) / max(1.0, float(len(state.agents))))

        app = sum(agent_closeness_values) / len(agent_closeness_values) if agent_closeness_values else 0.0
        cov = float(len(covered_foods)) / max(1.0, float(len(state.foods)))
        ready = ready_score / max(self._initial_total_food_value, 1.0)
        alloc = alloc_slots / max(1.0, float(len(state.foods)))

        if nearest_distances:
            mean_distance = sum(nearest_distances) / len(nearest_distances)
            variance = sum((distance - mean_distance) ** 2 for distance in nearest_distances) / len(nearest_distances)
            std_distance = math.sqrt(max(0.0, variance))
            stab = max(0.0, 1.0 - (std_distance / d_max))
        else:
            stab = 0.0

        return {
            "col": min(1.0, max(0.0, col)),
            "app": min(1.0, max(0.0, app)),
            "cov": min(1.0, max(0.0, cov)),
            "ready": min(1.0, max(0.0, ready)),
            "alloc": min(1.0, max(0.0, alloc)),
            "stab": min(1.0, max(0.0, stab)),
        }

    def compute_lbf_phi(self, state: LBFStateSnapshot) -> float:
        terms = self.compute_terms(state)
        return float(sum(float(self.weights[term]) * float(terms[term]) for term in TERM_NAMES))

    def build_info(self, *, phi_t: float, phi_tp1: float, reward_shape: float, reward_env: float, reward_total: float, terms_tp1: Dict[str, float]) -> Dict[str, Any]:
        info = {
            "pbrs_version": "lbf_pbrs_v2",
            "pbrs_mode": self.mode,
            "pbrs_v2_runtime_used": 1.0,
            "reward_env": float(reward_env),
            "reward_shape": float(reward_shape),
            "reward_total": float(reward_total),
            "phi_t": float(phi_t),
            "phi_tp1": float(phi_tp1),
            "pbrs_beta": float(self.beta),
            "pbrs_wc": float(self.config["wc"]),
            "pbrs_wp": float(self.config["wp"]),
        }
        for term in TERM_NAMES:
            info[f"pbrs_weights__{term}"] = float(self.weights[term])
            info[f"pbrs_term__{term}"] = float(terms_tp1.get(term, 0.0))
        return info


def _manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))
