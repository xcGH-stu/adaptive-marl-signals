from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
from typing import Any, Dict, Iterable, List, Tuple

from workflows.policy_guidance.rware_state_extractor import extract_rware_state


TERM_NAMES = ("shelf", "pickup", "goal", "deliv", "traffic", "stab")
DEFAULT_GAMMA = 0.99
DEFAULT_BETA = 0.30
DEFAULT_MODE = "balanced_delivery_progress"
RECENT_WINDOW = 6

DEFAULT_MODE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "requested_shelf_acquisition": {
        "beta": 0.30,
        "weights": {
            "shelf": 0.45,
            "pickup": 0.20,
            "goal": 0.15,
            "deliv": 0.05,
            "traffic": 0.15,
            "stab": 0.00,
        },
    },
    "balanced_delivery_progress": {
        "beta": 0.30,
        "weights": {
            "shelf": 0.25,
            "pickup": 0.15,
            "goal": 0.35,
            "deliv": 0.15,
            "traffic": 0.10,
            "stab": 0.00,
        },
    },
    "carrying_to_goal": {
        "beta": 0.30,
        "weights": {
            "shelf": 0.10,
            "pickup": 0.15,
            "goal": 0.50,
            "deliv": 0.15,
            "traffic": 0.10,
            "stab": 0.00,
        },
    },
    "traffic_conservative": {
        "beta": 0.20,
        "weights": {
            "shelf": 0.15,
            "pickup": 0.10,
            "goal": 0.25,
            "deliv": 0.10,
            "traffic": 0.35,
            "stab": 0.05,
        },
    },
    "late_stability": {
        "beta": 0.20,
        "weights": {
            "shelf": 0.05,
            "pickup": 0.10,
            "goal": 0.30,
            "deliv": 0.25,
            "traffic": 0.15,
            "stab": 0.15,
        },
    },
}

RWARE_PBRS_V2_DEFAULT_CONFIG: Dict[str, Any] = {
    "pbrs_version": "rware_pbrs_v2",
    "mode": DEFAULT_MODE,
    "beta": DEFAULT_BETA,
    "gamma": DEFAULT_GAMMA,
    "weights": deepcopy(DEFAULT_MODE_CONFIGS[DEFAULT_MODE]["weights"]),
}


def sanitize_rware_pbrs_v2_config(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = deepcopy(config or {})
    if not isinstance(payload, dict):
        raise ValueError("rware_pbrs_v2 config must be a dict")
    mode = str(payload.get("mode") or DEFAULT_MODE)
    if mode not in DEFAULT_MODE_CONFIGS:
        raise ValueError(f"Unsupported RWARE PBRS-v2 mode: {mode}")
    defaults = deepcopy(DEFAULT_MODE_CONFIGS[mode])
    weights = deepcopy(defaults.get("weights") or {})
    user_weights = payload.get("weights") or {}
    if user_weights and not isinstance(user_weights, dict):
        raise ValueError("rware_pbrs_v2 weights must be a dict")
    for term in TERM_NAMES:
        if term in user_weights:
            weights[term] = float(user_weights[term] or 0.0)
        else:
            weights.setdefault(term, 0.0)
    return {
        "pbrs_version": str(payload.get("pbrs_version") or "rware_pbrs_v2"),
        "mode": mode,
        "beta": float(payload.get("beta", defaults.get("beta", DEFAULT_BETA)) or 0.0),
        "gamma": float(payload.get("gamma", DEFAULT_GAMMA) or DEFAULT_GAMMA),
        "weights": weights,
    }


def normalize_rware_pbrs_v2_config(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = sanitize_rware_pbrs_v2_config(config)
    normalized = deepcopy(payload)
    normalized["beta"] = max(0.0, float(normalized["beta"]))
    normalized["gamma"] = min(1.0, max(0.0, float(normalized["gamma"])))
    raw_weights = {term: max(0.0, float(payload["weights"].get(term, 0.0))) for term in TERM_NAMES}
    weight_sum = sum(raw_weights.values())
    if weight_sum <= 0.0:
        raw_weights = deepcopy(DEFAULT_MODE_CONFIGS[normalized["mode"]]["weights"])
        weight_sum = sum(float(value) for value in raw_weights.values())
    normalized["weights"] = {
        term: float(raw_weights.get(term, 0.0)) / float(weight_sum) for term in TERM_NAMES
    }
    normalized["active_terms"] = [term for term in TERM_NAMES if normalized["weights"][term] > 0.0]
    return normalized


class RWAREPBRSV2Runtime:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = normalize_rware_pbrs_v2_config(config)
        self._episode_idx: int | None = None
        self._last_episode_step: int | None = None
        self._prev_state: Dict[str, Any] | None = None
        self._initial_request_queue_size: int = 1
        self._delivered_shelf_ids: set[int] = set()
        self._recent_requested_ids: set[int] = set()
        self._position_histories: Dict[int, deque[Tuple[int, int]]] = {}
        self._recent_repeated_ratios: deque[float] = deque(maxlen=RECENT_WINDOW)
        self._recent_oscillation_scores: deque[float] = deque(maxlen=RECENT_WINDOW)
        self._recent_idle_no_progress_rates: deque[float] = deque(maxlen=RECENT_WINDOW)

    def reset_episode(self) -> None:
        self._last_episode_step = None
        self._prev_state = None
        self._initial_request_queue_size = 1
        self._delivered_shelf_ids = set()
        self._recent_requested_ids = set()
        self._position_histories = {}
        self._recent_repeated_ratios.clear()
        self._recent_oscillation_scores.clear()
        self._recent_idle_no_progress_rates.clear()

    def compute_dense_reward(self, context: Any) -> tuple[float, Dict[str, Any]]:
        self._maybe_reset_for_context(context)
        current_state = extract_rware_state(
            context.raw_env,
            prev_state=self._prev_state,
            actions=context.actions,
            reward=context.sparse_reward,
        )
        current_request_ids = {
            int(item.get("id", -1))
            for item in list(current_state.get("requested_shelves") or [])
            if item.get("id") is not None
        }
        self._recent_requested_ids.update(current_request_ids)
        if self._prev_state is None:
            self._initial_request_queue_size = max(1, len(current_request_ids))
        delivery_events = list((current_state.get("events") or {}).get("delivery_events") or [])
        for event in delivery_events:
            shelf_id = event.get("shelf_id")
            if shelf_id is not None:
                self._delivered_shelf_ids.add(int(shelf_id))

        warnings: List[str] = [str(item) for item in list(current_state.get("warnings") or [])]
        previous_state = self._prev_state
        previous_terms, previous_metadata = self._compute_terms(previous_state) if previous_state else ({term: 0.0 for term in TERM_NAMES}, {"warnings": ["No previous RWARE state available; first-step shaping forced to zero."], "flags": {"low_confidence_pickup_identity": False, "low_confidence_delivery_inference": False, "stab_neutral_used": True}})
        current_terms, current_metadata = self._compute_terms(current_state)
        warnings.extend([str(item) for item in list(previous_metadata.get("warnings") or [])])
        warnings.extend([str(item) for item in list(current_metadata.get("warnings") or [])])

        potential = self._weighted_sum(previous_terms)
        potential_next = self._weighted_sum(current_terms)
        beta = float(self.config["beta"])
        gamma = float(self.config["gamma"])
        shaping = beta * (gamma * potential_next - potential)
        if previous_state is None:
            potential = potential_next
            shaping = 0.0

        if not math.isfinite(shaping):
            warnings.append("Non-finite shaping encountered; forced to 0.0.")
            shaping = 0.0

        breakdown = self._build_breakdown(
            context=context,
            current_state=current_state,
            potential=potential,
            potential_next=potential_next,
            shaping=shaping,
            terms=current_terms,
            metadata=current_metadata,
            warnings=warnings,
        )
        self._prev_state = deepcopy(current_state)
        self._last_episode_step = int(getattr(context, "episode_step", 0) or 0)
        return float(shaping), breakdown

    def _maybe_reset_for_context(self, context: Any) -> None:
        episode_idx = int(getattr(context, "episode_idx", 0) or 0)
        episode_step = int(getattr(context, "episode_step", 0) or 0)
        if self._episode_idx is None or episode_idx != self._episode_idx:
            self._episode_idx = episode_idx
            self.reset_episode()
            return
        if episode_step == 0 and self._last_episode_step not in (None, 0):
            self.reset_episode()
            return
        if self._last_episode_step is not None and episode_step < self._last_episode_step:
            self.reset_episode()

    def _compute_terms(self, state: Dict[str, Any] | None) -> tuple[Dict[str, float], Dict[str, Any]]:
        terms = {term: 0.0 for term in TERM_NAMES}
        metadata = {
            "warnings": [],
            "flags": {
                "low_confidence_pickup_identity": False,
                "low_confidence_delivery_inference": False,
                "stab_neutral_used": False,
            },
        }
        if not isinstance(state, dict):
            metadata["warnings"].append("RWARE state unavailable; all PBRS-v2 terms defaulted to 0.")
            metadata["flags"]["stab_neutral_used"] = True
            return terms, metadata

        agents = list(state.get("agents") or [])
        requested_shelves = list(state.get("requested_shelves") or [])
        goals = list(state.get("goals") or [])
        grid_size = list(state.get("grid_size") or [])
        normalizer = max(1.0, float(sum(int(value) for value in grid_size[:2]) - 2)) if len(grid_size) == 2 else 1.0
        current_requested_ids = {int(item.get("id", -1)) for item in requested_shelves if item.get("id") is not None}
        relevant_requested_ids = set(self._recent_requested_ids) | current_requested_ids

        carrying_agents = [agent for agent in agents if agent.get("carrying_shelf_id") is not None]
        carrying_relevant = [
            agent
            for agent in carrying_agents
            if int(agent.get("carrying_shelf_id")) in relevant_requested_ids
        ]

        if carrying_relevant:
            terms["shelf"] = 1.0
        elif requested_shelves:
            distances = []
            for agent in agents:
                agent_pos = tuple(agent.get("position") or ())
                if len(agent_pos) != 2:
                    continue
                for shelf in requested_shelves:
                    shelf_pos = tuple(shelf.get("position") or ())
                    if len(shelf_pos) != 2:
                        continue
                    distance = _manhattan(agent_pos, shelf_pos)
                    distances.append(1.0 - (float(distance) / normalizer))
            terms["shelf"] = _clamp01(max(distances) if distances else 0.0)
        elif carrying_agents:
            terms["shelf"] = 1.0
            metadata["warnings"].append(
                "No requested shelves visible; shelf term promoted from carrying state."
            )

        if carrying_relevant:
            terms["pickup"] = 1.0
        elif carrying_agents:
            terms["pickup"] = 1.0
            metadata["flags"]["low_confidence_pickup_identity"] = True
            metadata["warnings"].append(
                "Pickup identity fell back to any carrying shelf because requested identity was uncertain."
            )

        if carrying_agents and goals:
            goal_values = []
            for agent in carrying_agents:
                position = tuple(agent.get("position") or ())
                if len(position) != 2:
                    continue
                nearest = min(_manhattan(position, tuple(goal)) for goal in goals if len(tuple(goal)) == 2)
                goal_values.append(1.0 - (float(nearest) / normalizer))
            terms["goal"] = _clamp01(max(goal_values) if goal_values else 0.0)

        terms["deliv"] = _clamp01(
            float(len(self._delivered_shelf_ids)) / float(max(1, self._initial_request_queue_size))
        )
        if any(bool(event.get("low_confidence_delivery_inference")) for event in list((state.get("events") or {}).get("delivery_events") or [])):
            metadata["flags"]["low_confidence_delivery_inference"] = True
            metadata["warnings"].append(
                "Delivery inference used low-confidence request-queue evidence."
            )

        close_pair_ratio = _close_agent_pair_ratio(agents)
        blocked_rate = _safe_divide(
            len(list((state.get("events") or {}).get("blocked_forward_events") or [])),
            max(1, len(agents)),
        )
        congestion_score = _clamp01(0.5 * close_pair_ratio + 0.5 * blocked_rate)
        terms["traffic"] = _clamp01(1.0 - congestion_score)

        repeated_ratio, oscillation_ratio, idle_no_progress_ratio = self._update_stability_buffers(state)
        if not self._recent_repeated_ratios:
            terms["stab"] = 0.5
            metadata["flags"]["stab_neutral_used"] = True
            metadata["warnings"].append(
                "Stability history unavailable; stab term defaulted to neutral 0.5."
            )
        else:
            instability = _clamp01(
                0.5 * _mean(self._recent_repeated_ratios)
                + 0.3 * _mean(self._recent_oscillation_scores)
                + 0.2 * _mean(self._recent_idle_no_progress_rates)
            )
            terms["stab"] = _clamp01(1.0 - instability)
        metadata["close_pair_ratio"] = close_pair_ratio
        metadata["blocked_rate"] = blocked_rate
        metadata["repeated_ratio_recent"] = repeated_ratio
        metadata["route_oscillation_score_recent"] = oscillation_ratio
        metadata["idle_no_progress_rate_recent"] = idle_no_progress_ratio
        return terms, metadata

    def _update_stability_buffers(self, state: Dict[str, Any]) -> tuple[float, float, float]:
        agents = list(state.get("agents") or [])
        repeated_hits = 0
        oscillation_hits = 0
        idle_or_no_progress_hits = 0
        for agent in agents:
            agent_id = int(agent.get("id", -1))
            position = tuple(agent.get("position") or ())
            if len(position) != 2:
                continue
            history = self._position_histories.setdefault(agent_id, deque(maxlen=3))
            history.append(position)
            prev_position = tuple(agent.get("prev_position") or ())
            unchanged = len(prev_position) == 2 and prev_position == position
            if unchanged:
                repeated_hits += 1
            if len(history) == 3 and history[0] == history[2] and history[0] != history[1]:
                oscillation_hits += 1
            requested_action = str(agent.get("requested_action") or "")
            if requested_action == "NOOP" or unchanged:
                idle_or_no_progress_hits += 1
        denominator = max(1, len(agents))
        repeated_ratio = _safe_divide(repeated_hits, denominator)
        oscillation_ratio = _safe_divide(oscillation_hits, denominator)
        idle_no_progress_ratio = _safe_divide(idle_or_no_progress_hits, denominator)
        self._recent_repeated_ratios.append(repeated_ratio)
        self._recent_oscillation_scores.append(oscillation_ratio)
        self._recent_idle_no_progress_rates.append(idle_no_progress_ratio)
        return repeated_ratio, oscillation_ratio, idle_no_progress_ratio

    def _weighted_sum(self, terms: Dict[str, float]) -> float:
        return _clamp01(
            sum(float(self.config["weights"][term]) * float(terms.get(term, 0.0)) for term in TERM_NAMES)
        )

    def _build_breakdown(
        self,
        *,
        context: Any,
        current_state: Dict[str, Any],
        potential: float,
        potential_next: float,
        shaping: float,
        terms: Dict[str, float],
        metadata: Dict[str, Any],
        warnings: List[str],
    ) -> Dict[str, Any]:
        events = dict(current_state.get("events") or {})
        breakdown: Dict[str, Any] = {
            "pbrs_version": "rware_pbrs_v2",
            "mode": str(self.config["mode"]),
            "rware_pbrs_v2_runtime_used": True,
            "rware_pbrs_v2_potential": float(potential),
            "rware_pbrs_v2_potential_next": float(potential_next),
            "rware_pbrs_v2_shaping": float(shaping),
            "rware_pbrs_v2_shaping_abs": abs(float(shaping)),
            "rware_pbrs_v2_term_shelf": float(terms["shelf"]),
            "rware_pbrs_v2_term_pickup": float(terms["pickup"]),
            "rware_pbrs_v2_term_goal": float(terms["goal"]),
            "rware_pbrs_v2_term_deliv": float(terms["deliv"]),
            "rware_pbrs_v2_term_traffic": float(terms["traffic"]),
            "rware_pbrs_v2_term_stab": float(terms["stab"]),
            "rware_pbrs_weights__shelf": float(self.config["weights"]["shelf"]),
            "rware_pbrs_weights__pickup": float(self.config["weights"]["pickup"]),
            "rware_pbrs_weights__goal": float(self.config["weights"]["goal"]),
            "rware_pbrs_weights__deliv": float(self.config["weights"]["deliv"]),
            "rware_pbrs_weights__traffic": float(self.config["weights"]["traffic"]),
            "rware_pbrs_weights__stab": float(self.config["weights"]["stab"]),
            "rware_delivery_events_inferred": int(len(list(events.get("delivery_events") or []))),
            "rware_pickup_events_inferred": int(len(list(events.get("pickup_events") or []))),
            "rware_blocked_forward_events_inferred": int(len(list(events.get("blocked_forward_events") or []))),
            "rware_low_confidence_pickup_identity": bool(metadata["flags"]["low_confidence_pickup_identity"]),
            "rware_low_confidence_delivery_inference": bool(metadata["flags"]["low_confidence_delivery_inference"]),
            "rware_stab_neutral_used": bool(metadata["flags"]["stab_neutral_used"]),
            "rware_close_agent_pair_ratio": float(metadata.get("close_pair_ratio", 0.0) or 0.0),
            "rware_recent_blocked_forward_rate": float(metadata.get("blocked_rate", 0.0) or 0.0),
            "rware_repeated_position_ratio_recent": float(metadata.get("repeated_ratio_recent", 0.0) or 0.0),
            "rware_route_oscillation_score_recent": float(metadata.get("route_oscillation_score_recent", 0.0) or 0.0),
            "rware_idle_no_progress_rate_recent": float(metadata.get("idle_no_progress_rate_recent", 0.0) or 0.0),
            "eval_use_pbrs": False,
            "test_sparse_reward_only": bool(getattr(context, "test_mode", False)),
            "warnings": list(dict.fromkeys(str(item) for item in warnings if item)),
        }
        return breakdown


def compute_dense_reward_with_config(
    context: Any,
    config: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, Any]]:
    runtime = RWAREPBRSV2Runtime(config)
    return runtime.compute_dense_reward(context)


_DEFAULT_RWARE_RUNTIME = RWAREPBRSV2Runtime(RWARE_PBRS_V2_DEFAULT_CONFIG)


def compute_dense_reward(context: Any) -> tuple[float, Dict[str, Any]]:
    return _DEFAULT_RWARE_RUNTIME.compute_dense_reward(context)


def _manhattan(left: Tuple[int, int], right: Tuple[int, int]) -> int:
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))


def _close_agent_pair_ratio(agents: Iterable[Dict[str, Any]]) -> float:
    positions = [
        tuple(agent.get("position") or ())
        for agent in agents
        if len(tuple(agent.get("position") or ())) == 2
    ]
    if len(positions) < 2:
        return 0.0
    total_pairs = 0
    close_pairs = 0
    for index, left in enumerate(positions):
        for right in positions[index + 1 :]:
            total_pairs += 1
            if _manhattan(left, right) <= 1:
                close_pairs += 1
    return _safe_divide(close_pairs, total_pairs)


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
