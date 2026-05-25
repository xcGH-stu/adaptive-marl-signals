from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

from .policy_guidance_schema import (
    build_insufficient_behavior_summary,
    validate_policy_behavior_summary,
)


def build_rware_behavior_summary(
    *,
    milestone: Dict[str, Any],
    near_shelf_distance_threshold: int = 1,
) -> Dict[str, Any]:
    env_key = str(milestone.get("env_key") or "")
    stage_label = str(milestone.get("stage_label") or "")
    episodes = list(milestone.get("episodes") or [])
    num_eval_episodes = int(milestone.get("num_eval_episodes") or len(episodes))
    if "rware" not in env_key:
        return build_insufficient_behavior_summary(
            intervention_point=stage_label,
            env_key=env_key,
            eval_episodes=num_eval_episodes,
            reason="RWARE summary requested for a non-RWARE env_key",
        )
    if not episodes:
        return build_insufficient_behavior_summary(
            intervention_point=stage_label,
            env_key=env_key,
            eval_episodes=num_eval_episodes,
            reason="no RWARE episode traces were provided",
        )

    episode_summaries = [
        summarize_rware_episode(
            episode=episode,
            near_shelf_distance_threshold=near_shelf_distance_threshold,
        )
        for episode in episodes
    ]
    valid_areas = [item.get("_grid_area") for item in episode_summaries if item.get("_grid_area")]
    grid_area = max(valid_areas) if valid_areas else None
    joint_visited = set()
    for item in episode_summaries:
        joint_visited.update(set(item.get("_joint_visited_cells") or []))
    if not grid_area:
        return build_insufficient_behavior_summary(
            intervention_point=stage_label,
            env_key=env_key,
            eval_episodes=num_eval_episodes,
            reason="could not recover RWARE grid size from traces",
        )

    shelf_metrics = {
        "episodes_without_requested_shelf_contact": int(
            sum(1 for item in episode_summaries if int(item.get("near_requested_shelf_steps") or 0) == 0)
        ),
        "first_near_requested_shelf_step_mean": _mean_optional(
            item.get("_first_near_requested_shelf_step") for item in episode_summaries
        ),
        "near_requested_shelf_steps_mean": _mean(
            item.get("near_requested_shelf_steps") for item in episode_summaries
        ),
        "joint_visited_cell_ratio": _safe_ratio(len(joint_visited), grid_area),
        "per_agent_visited_cell_ratio_mean": _mean(
            item.get("visited_cell_ratio") for item in episode_summaries
        ),
    }
    pickup_metrics = {
        "pickup_success_events_mean": _mean(
            item.get("pickup_success_events") for item in episode_summaries
        ),
        "pickup_success_events_total": int(
            sum(int(item.get("pickup_success_events") or 0) for item in episode_summaries)
        ),
        "near_requested_without_pickup_steps_total": int(
            sum(int(item.get("_near_requested_without_pickup_steps") or 0) for item in episode_summaries)
        ),
    }
    carrying_metrics = {
        "carrying_steps_mean": _mean(
            item.get("carrying_steps") for item in episode_summaries
        ),
        "near_goal_while_carrying_steps_mean": _mean(
            item.get("near_goal_while_carrying_steps") for item in episode_summaries
        ),
    }
    delivery_metrics = {
        "delivery_success_events_total": int(
            sum(int(item.get("delivery_success_events") or 0) for item in episode_summaries)
        ),
        "delivery_success_events_mean": _mean(
            item.get("delivery_success_events") for item in episode_summaries
        ),
    }
    traffic_metrics = {
        "blocked_move_steps_total": int(
            sum(int(item.get("blocked_move_steps") or 0) for item in episode_summaries)
        ),
        "blocked_move_steps_mean": _mean(
            item.get("blocked_move_steps") for item in episode_summaries
        ),
        "congestion_proxy_mean": _mean(
            item.get("congestion_proxy") for item in episode_summaries
        ),
        "requested_overlap_ratio_mean": _mean(
            item.get("_requested_overlap_ratio") for item in episode_summaries
        ),
        "target_entropy_mean": _mean(
            item.get("_requested_target_entropy_mean") for item in episode_summaries
        ),
    }
    stability_metrics = {
        "route_oscillation_score_mean": _mean(
            item.get("route_oscillation_score") for item in episode_summaries
        ),
        "repeated_position_ratio_mean": _mean(
            item.get("_repeated_position_ratio") for item in episode_summaries
        ),
        "action_switch_count_mean": _mean(
            item.get("_action_switch_count_mean") for item in episode_summaries
        ),
        "target_switch_count_mean": _mean(
            item.get("_target_switch_count_mean") for item in episode_summaries
        ),
        "episode_length_mean": _mean(item.get("episode_length") for item in episode_summaries),
        "return_mean": _mean(item.get("return") for item in episode_summaries),
    }

    evidence_flags = {
        "requested_shelf_access_low": shelf_metrics["episodes_without_requested_shelf_contact"]
        >= max(1, len(episode_summaries) // 2),
        "pickup_progress_weak": (
            shelf_metrics["near_requested_shelf_steps_mean"] > 0
            and pickup_metrics["pickup_success_events_total"] == 0
        ),
        "carrying_progress_weak": (
            pickup_metrics["pickup_success_events_total"] > 0
            and carrying_metrics["near_goal_while_carrying_steps_mean"] <= 0.0
        ),
        "delivery_progress_weak": (
            carrying_metrics["carrying_steps_mean"] > 0.0
            and delivery_metrics["delivery_success_events_total"] == 0
        ),
        "traffic_blocking_high": traffic_metrics["congestion_proxy_mean"] > 0.35
        or traffic_metrics["blocked_move_steps_total"] > 0,
        "route_stability_low": stability_metrics["route_oscillation_score_mean"] > 0.35
        or stability_metrics["repeated_position_ratio_mean"] > 0.40,
        "insufficient_behavior_evidence": False,
    }

    behavior_signals = _build_behavior_signals(
        shelf_metrics=shelf_metrics,
        pickup_metrics=pickup_metrics,
        carrying_metrics=carrying_metrics,
        delivery_metrics=delivery_metrics,
        traffic_metrics=traffic_metrics,
        stability_metrics=stability_metrics,
        evidence_flags=evidence_flags,
    )
    summary = {
        "intervention_point": stage_label,
        "env_key": env_key,
        "env_family": "rware",
        "eval_episodes": num_eval_episodes,
        "insufficient_evidence": False,
        "insufficient_evidence_reason": "",
        "collector_status": "rware_state_collected",
        "milestone": {
            "stage_label": stage_label,
            "target_step": int(milestone.get("target_step") or 0),
            "actual_step": int(milestone.get("actual_step") or 0),
            "num_eval_episodes": num_eval_episodes,
        },
        "shelf_acquisition": {
            "score": _invert_distance_proxy(
                shelf_metrics["episodes_without_requested_shelf_contact"],
                len(episode_summaries),
            ),
            "evidence": behavior_signals["shelf_acquisition"],
            "metrics": shelf_metrics,
        },
        "pickup_progress": {
            "score": min(1.0, float(pickup_metrics["pickup_success_events_mean"] or 0.0)),
            "evidence": behavior_signals["pickup_progress"],
            "metrics": pickup_metrics,
        },
        "carrying_to_goal": {
            "score": min(1.0, float(carrying_metrics["near_goal_while_carrying_steps_mean"] or 0.0)),
            "evidence": behavior_signals["carrying_to_goal"],
            "metrics": carrying_metrics,
        },
        "delivery_success": {
            "score": min(1.0, float(delivery_metrics["delivery_success_events_mean"] or 0.0)),
            "evidence": behavior_signals["delivery_success"],
            "metrics": delivery_metrics,
        },
        "traffic_blocking": {
            "score": min(1.0, float(traffic_metrics["congestion_proxy_mean"] or 0.0)),
            "evidence": behavior_signals["traffic_blocking"],
            "metrics": traffic_metrics,
        },
        "route_stability": {
            "score": 1.0 - min(1.0, float(stability_metrics["route_oscillation_score_mean"] or 0.0)),
            "evidence": behavior_signals["route_stability"],
            "metrics": stability_metrics,
        },
        "evidence_flags": evidence_flags,
        "behavior_signals": (
            behavior_signals["shelf_acquisition"]
            + behavior_signals["pickup_progress"]
            + behavior_signals["carrying_to_goal"]
            + behavior_signals["delivery_success"]
            + behavior_signals["traffic_blocking"]
            + behavior_signals["route_stability"]
        ),
        "episode_summaries": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in episode_summaries
        ],
        "metadata": {
            "env_type": "rware",
            "grid_area": grid_area,
            "feature_source": "env_state_then_inferred_events",
            "warning_count": int(
                sum(len(item.get("warnings") or []) for item in episode_summaries)
            ),
        },
    }
    return validate_policy_behavior_summary(summary)


def summarize_rware_episode(
    *,
    episode: Dict[str, Any],
    near_shelf_distance_threshold: int = 1,
) -> Dict[str, Any]:
    steps = list(episode.get("steps") or [])
    if not steps:
        return {
            "env_type": "rware",
            "episode_id": int(episode.get("episode_id") or 0),
            "return": 0.0,
            "episode_length": 0,
            "visited_cell_ratio": 0.0,
            "near_shelf_steps": 0,
            "near_requested_shelf_steps": 0,
            "pickup_success_events": 0,
            "carrying_steps": 0,
            "near_goal_while_carrying_steps": 0,
            "delivery_success_events": 0,
            "blocked_move_steps": 0,
            "idle_or_wait_fraction": 0.0,
            "congestion_proxy": 0.0,
            "route_oscillation_score": 0.0,
            "summary_text": "No RWARE steps recorded.",
            "warnings": ["No RWARE steps recorded."],
            "_grid_area": None,
            "_joint_visited_cells": [],
            "_repeated_position_ratio": 0.0,
            "_route_oscillation_events": 0,
            "_first_near_requested_shelf_step": None,
            "_near_requested_without_pickup_steps": 0,
            "_requested_overlap_ratio": 0.0,
            "_requested_target_entropy_mean": 0.0,
            "_target_switch_count_mean": 0.0,
            "_overlap_requested_steps": 0,
            "_action_switch_count_mean": 0.0,
        }

    first_state = steps[0].get("env_state") or {}
    grid_size = list(first_state.get("grid_size") or [])
    grid_area = int(grid_size[0] * grid_size[1]) if len(grid_size) == 2 else None
    joint_visited: set[Tuple[int, int]] = set()
    per_agent_visited: Dict[int, set[Tuple[int, int]]] = {}
    previous_positions: Dict[int, List[Tuple[int, int]]] = {}
    previous_actions: Dict[int, str] = {}
    requested_target_history: Dict[int, List[int]] = {}
    warnings: List[str] = []

    total_reward = 0.0
    total_actions = 0
    idle_actions = 0
    carrying_steps = 0
    near_shelf_steps = 0
    near_requested_shelf_steps = 0
    near_goal_while_carrying_steps = 0
    pickup_success_events = 0
    delivery_success_events = 0
    blocked_move_steps = 0
    repeated_position_steps = 0
    route_oscillation_events = 0
    first_near_requested_shelf_step = None
    near_requested_without_pickup_steps = 0
    overlap_requested_steps = 0
    target_overlap_ratios: List[float] = []
    target_entropy_values: List[float] = []
    action_switch_counts: Counter[int] = Counter()

    for step_index, step in enumerate(steps):
        env_state = step.get("env_state") or {}
        agents = list(env_state.get("agents") or [])
        shelves = list(env_state.get("shelves") or [])
        requested = list(env_state.get("requested_shelves") or [])
        goals = list(env_state.get("goals") or [])
        events = dict(env_state.get("events") or {})
        step_actions = list(step.get("actions") or env_state.get("actions") or [])
        total_reward += float(step.get("reward", env_state.get("reward", 0.0)) or 0.0)
        step_pickups = len(list(events.get("pickup_events") or []))
        step_deliveries = len(list(events.get("delivery_events") or []))
        step_blocked = len(list(events.get("blocked_forward_events") or []))
        pickup_success_events += step_pickups
        delivery_success_events += step_deliveries
        blocked_move_steps += step_blocked
        if env_state.get("warnings"):
            warnings.extend([str(item) for item in list(env_state.get("warnings") or [])])

        step_near_any_shelf = False
        step_near_requested = False
        step_near_goal_carry = False
        requested_targets_this_step: List[int] = []

        for index, agent in enumerate(agents):
            agent_id = int(agent.get("id", index + 1))
            position = tuple(agent.get("position") or ())
            if len(position) != 2:
                continue
            joint_visited.add(position)
            per_agent_visited.setdefault(agent_id, set()).add(position)
            action = str(step_actions[index]) if index < len(step_actions) else str(agent.get("requested_action") or "UNKNOWN")
            total_actions += 1
            if action == "NOOP":
                idle_actions += 1
            if previous_actions.get(agent_id) not in (None, action):
                action_switch_counts[agent_id] += 1
            previous_actions[agent_id] = action

            history = previous_positions.setdefault(agent_id, [])
            history.append(position)
            if len(history) >= 2 and history[-1] == history[-2]:
                repeated_position_steps += 1
            if len(history) >= 3 and history[-1] == history[-3] and history[-1] != history[-2]:
                route_oscillation_events += 1

            carrying = agent.get("carrying_shelf_id") is not None
            if carrying:
                carrying_steps += 1

            min_shelf_distance = _nearest_distance(position, shelves)
            min_requested_distance, requested_target_id = _nearest_distance_with_id(position, requested)
            min_goal_distance = _nearest_distance(position, [{"position": item} for item in goals])

            if min_shelf_distance is not None and min_shelf_distance <= near_shelf_distance_threshold:
                step_near_any_shelf = True
            if min_requested_distance is not None and min_requested_distance <= near_shelf_distance_threshold:
                step_near_requested = True
                if first_near_requested_shelf_step is None:
                    first_near_requested_shelf_step = step_index
                if requested_target_id is not None:
                    requested_targets_this_step.append(requested_target_id)
                    requested_target_history.setdefault(agent_id, []).append(requested_target_id)
            if carrying and min_goal_distance is not None and min_goal_distance <= near_shelf_distance_threshold:
                step_near_goal_carry = True

        if step_near_any_shelf:
            near_shelf_steps += 1
        if step_near_requested:
            near_requested_shelf_steps += 1
            if step_pickups == 0:
                near_requested_without_pickup_steps += 1
        if step_near_goal_carry:
            near_goal_while_carrying_steps += 1

        if len(requested_targets_this_step) > 1:
            counts = Counter(requested_targets_this_step)
            max_count = counts.most_common(1)[0][1]
            target_overlap_ratios.append(_safe_ratio(max_count, len(requested_targets_this_step)))
            if max_count > 1:
                overlap_requested_steps += 1
            target_entropy_values.append(_entropy_from_counter(counts))
        elif requested_targets_this_step:
            target_overlap_ratios.append(1.0)
            target_entropy_values.append(0.0)

    per_agent_ratios = [
        _safe_ratio(len(cells), grid_area) for cells in per_agent_visited.values()
    ]
    target_switch_counts = []
    for targets in requested_target_history.values():
        switches = 0
        for left, right in zip(targets, targets[1:]):
            if left != right:
                switches += 1
        target_switch_counts.append(switches)

    episode_length = len(steps)
    summary_text = (
        f"Visited {len(joint_visited)} cells, near requested shelves on {near_requested_shelf_steps} steps, "
        f"pickups={pickup_success_events}, deliveries={delivery_success_events}, blocked={blocked_move_steps}."
    )
    return {
        "env_type": "rware",
        "episode_id": int(episode.get("episode_id") or 0),
        "return": total_reward,
        "episode_length": episode_length,
        "visited_cell_ratio": _safe_ratio(len(joint_visited), grid_area),
        "near_shelf_steps": near_shelf_steps,
        "near_requested_shelf_steps": near_requested_shelf_steps,
        "pickup_success_events": pickup_success_events,
        "carrying_steps": carrying_steps,
        "near_goal_while_carrying_steps": near_goal_while_carrying_steps,
        "delivery_success_events": delivery_success_events,
        "blocked_move_steps": blocked_move_steps,
        "idle_or_wait_fraction": _safe_ratio(idle_actions, total_actions),
        "congestion_proxy": _mean(target_overlap_ratios)
        + _safe_ratio(blocked_move_steps, max(episode_length, 1)),
        "route_oscillation_score": _safe_ratio(route_oscillation_events, max(total_actions, 1)),
        "summary_text": summary_text,
        "warnings": warnings,
        "_grid_area": grid_area,
        "_joint_visited_cells": list(joint_visited),
        "_repeated_position_ratio": _safe_ratio(repeated_position_steps, max(total_actions, 1)),
        "_route_oscillation_events": route_oscillation_events,
        "_first_near_requested_shelf_step": first_near_requested_shelf_step,
        "_near_requested_without_pickup_steps": near_requested_without_pickup_steps,
        "_requested_overlap_ratio": _mean(target_overlap_ratios),
        "_requested_target_entropy_mean": _mean(target_entropy_values),
        "_target_switch_count_mean": _mean(target_switch_counts),
        "_overlap_requested_steps": overlap_requested_steps,
        "_action_switch_count_mean": _mean(action_switch_counts.values()),
    }


def build_rware_behavior_summary_from_traces(
    *,
    env_key: str,
    intervention_point: str,
    episode_traces: List[List[Dict[str, Any]]],
    num_eval_episodes: int,
    stage_label: str,
    target_step: int,
    actual_step: int,
    source_run_ref: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    milestone = {
        "stage_label": stage_label,
        "target_step": target_step,
        "actual_step": actual_step,
        "env_key": env_key,
        "num_eval_episodes": num_eval_episodes,
        "episodes": [
            _convert_trace_to_episode(trace, episode_id)
            for episode_id, trace in enumerate(episode_traces)
        ],
    }
    summary = build_rware_behavior_summary(milestone=milestone)
    summary["intervention_point"] = str(intervention_point)
    metadata = dict(summary.get("metadata") or {})
    metadata["source_run_ref"] = deepcopy(source_run_ref or {})
    metadata["collector_mode"] = "rware_trace_collection"
    summary["metadata"] = metadata
    return validate_policy_behavior_summary(summary)


def _convert_trace_to_episode(trace: List[Dict[str, Any]], episode_id: int) -> Dict[str, Any]:
    total_reward = sum(float(item.get("reward") or 0.0) for item in trace)
    return {
        "episode_id": episode_id,
        "steps": [
            {
                "env_state": deepcopy(item.get("env_state") or {}),
                "actions": list(item.get("actions") or []),
                "reward": float(item.get("reward") or 0.0),
                "terminated": bool(item.get("terminated", False)),
                "info": deepcopy(item.get("info") or {}),
            }
            for item in trace
        ],
        "total_reward": total_reward,
    }


def _nearest_distance(position: Tuple[int, int], items: List[Dict[str, Any]]) -> int | None:
    minimum = None
    for item in items:
        target = tuple(item.get("position") or ())
        if len(target) != 2:
            continue
        distance = abs(position[0] - target[0]) + abs(position[1] - target[1])
        if minimum is None or distance < minimum:
            minimum = distance
    return minimum


def _nearest_distance_with_id(
    position: Tuple[int, int],
    items: List[Dict[str, Any]],
) -> Tuple[int | None, int | None]:
    minimum = None
    minimum_id = None
    for item in items:
        target = tuple(item.get("position") or ())
        if len(target) != 2:
            continue
        distance = abs(position[0] - target[0]) + abs(position[1] - target[1])
        if minimum is None or distance < minimum:
            minimum = distance
            minimum_id = int(item.get("id", -1))
    return minimum, minimum_id


def _build_behavior_signals(
    *,
    shelf_metrics: Dict[str, Any],
    pickup_metrics: Dict[str, Any],
    carrying_metrics: Dict[str, Any],
    delivery_metrics: Dict[str, Any],
    traffic_metrics: Dict[str, Any],
    stability_metrics: Dict[str, Any],
    evidence_flags: Dict[str, Any],
) -> Dict[str, List[str]]:
    shelf = [
        f"shelf_acquisition.metrics.near_requested_shelf_steps_mean={shelf_metrics['near_requested_shelf_steps_mean']:.3f}",
        f"shelf_acquisition.metrics.first_near_requested_shelf_step_mean={shelf_metrics['first_near_requested_shelf_step_mean']}",
    ]
    pickup = [
        f"pickup_progress.metrics.pickup_success_events_total={pickup_metrics['pickup_success_events_total']}",
        f"pickup_progress.metrics.near_requested_without_pickup_steps_total={pickup_metrics['near_requested_without_pickup_steps_total']}",
    ]
    carrying = [
        f"carrying_to_goal.metrics.carrying_steps_mean={carrying_metrics['carrying_steps_mean']:.3f}",
        f"carrying_to_goal.metrics.near_goal_while_carrying_steps_mean={carrying_metrics['near_goal_while_carrying_steps_mean']:.3f}",
    ]
    delivery = [
        f"delivery_success.metrics.delivery_success_events_total={delivery_metrics['delivery_success_events_total']}",
        f"delivery_success.metrics.delivery_success_events_mean={delivery_metrics['delivery_success_events_mean']:.3f}",
    ]
    traffic = [
        f"traffic_blocking.metrics.blocked_move_steps_total={traffic_metrics['blocked_move_steps_total']}",
        f"traffic_blocking.metrics.congestion_proxy_mean={traffic_metrics['congestion_proxy_mean']:.3f}",
    ]
    stability = [
        f"route_stability.metrics.route_oscillation_score_mean={stability_metrics['route_oscillation_score_mean']:.3f}",
        f"route_stability.metrics.repeated_position_ratio_mean={stability_metrics['repeated_position_ratio_mean']:.3f}",
    ]
    if evidence_flags["requested_shelf_access_low"]:
        shelf.append("evidence_flags.requested_shelf_access_low=true")
    if evidence_flags["pickup_progress_weak"]:
        pickup.append("evidence_flags.pickup_progress_weak=true")
    if evidence_flags["carrying_progress_weak"]:
        carrying.append("evidence_flags.carrying_progress_weak=true")
    if evidence_flags["delivery_progress_weak"]:
        delivery.append("evidence_flags.delivery_progress_weak=true")
    if evidence_flags["traffic_blocking_high"]:
        traffic.append("evidence_flags.traffic_blocking_high=true")
    if evidence_flags["route_stability_low"]:
        stability.append("evidence_flags.route_stability_low=true")
    return {
        "shelf_acquisition": shelf,
        "pickup_progress": pickup,
        "carrying_to_goal": carrying,
        "delivery_success": delivery,
        "traffic_blocking": traffic,
        "route_stability": stability,
    }


def _entropy_from_counter(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log(probability, 2)
    return float(entropy)


def _invert_distance_proxy(missed_episodes: int, total_episodes: int) -> float:
    if total_episodes <= 0:
        return 0.0
    return 1.0 - _safe_ratio(missed_episodes, total_episodes)


def _failure_score(numerator: int, denominator: int) -> float:
    return min(1.0, float(numerator) / max(float(denominator), 1.0))


def _mean(values: Iterable[Any]) -> float:
    items = [float(value) for value in values if value is not None]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _mean_optional(values: Iterable[Any]) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return sum(items) / len(items)


def _safe_ratio(numerator: int | float, denominator: int | float | None) -> float:
    if denominator in (None, 0):
        return 0.0
    return float(numerator) / float(denominator)
