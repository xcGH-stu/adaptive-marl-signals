from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from typing import Any, Dict, Iterable, List, Tuple

from .policy_guidance_schema import (
    build_insufficient_behavior_summary,
    validate_policy_behavior_summary,
)


def build_compact_behavior_summary(
    *,
    milestone: Dict[str, Any],
    near_food_distance_threshold: int = 1,
    idle_action_values: Iterable[Any] | None = None,
    collect_action_values: Iterable[Any] | None = None,
) -> Dict[str, Any]:
    env_key = str(milestone.get("env_key") or "")
    stage_label = str(milestone.get("stage_label") or "")
    episodes = list(milestone.get("episodes") or [])
    num_eval_episodes = int(milestone.get("num_eval_episodes") or len(episodes))
    if not env_key.startswith("lbforaging"):
        return build_insufficient_behavior_summary(
            intervention_point=stage_label,
            env_key=env_key,
            eval_episodes=num_eval_episodes,
            reason="first-pass collector only supports LBF/lbforaging envs",
        )
    if not episodes:
        return build_insufficient_behavior_summary(
            intervention_point=stage_label,
            env_key=env_key,
            eval_episodes=num_eval_episodes,
            reason="no episode traces were provided",
        )

    idle_values = set(idle_action_values or [None, 0, 4, "stay", "STAY"])
    collect_values = set(collect_action_values or [5, "collect", "COLLECT", "load", "LOAD"])
    episode_summaries: List[Dict[str, Any]] = []
    aggregate_joint_visited: set[Tuple[int, int]] = set()
    field_area = None
    collection_event_observable_any = False

    for episode in episodes:
        summary = _summarize_episode(
            episode=episode,
            near_food_distance_threshold=near_food_distance_threshold,
            idle_action_values=idle_values,
            collect_action_values=collect_values,
        )
        episode_summaries.append(summary)
        aggregate_joint_visited.update(set(summary.get("_joint_visited_cells") or []))
        if field_area is None:
            field_area = summary.get("_field_area")
        collection_event_observable_any = (
            collection_event_observable_any
            or bool(summary.get("collection_event_observable", False))
        )

    if field_area in (None, 0):
        return build_insufficient_behavior_summary(
            intervention_point=stage_label,
            env_key=env_key,
            eval_episodes=num_eval_episodes,
            reason="could not recover field size or agent state from traces",
        )

    coverage_metrics = {
        "joint_visited_cell_count": len(aggregate_joint_visited),
        "joint_visited_cell_ratio": _safe_ratio(len(aggregate_joint_visited), field_area),
        "per_agent_visited_cell_ratio_mean": _mean(
            item["per_agent_visited_cell_ratio_mean"] for item in episode_summaries
        ),
        "idle_action_ratio": _mean(item["idle_action_ratio"] for item in episode_summaries),
        "backtracking_ratio": _mean(item["backtracking_ratio"] for item in episode_summaries),
        "oscillation_count": int(sum(item["oscillation_count"] for item in episode_summaries)),
    }
    food_metrics = {
        "mean_nearest_food_distance": _mean(
            item["mean_nearest_food_distance"] for item in episode_summaries
        ),
        "min_nearest_food_distance_mean": _mean(
            item["min_nearest_food_distance"] for item in episode_summaries
        ),
        "episodes_never_close_to_food": int(
            sum(1 for item in episode_summaries if item["near_food_steps"] == 0)
        ),
        "near_food_steps_mean": _mean(item["near_food_steps"] for item in episode_summaries),
        "first_near_food_step_mean": _mean_optional(
            item["first_near_food_step"] for item in episode_summaries
        ),
        "food_contact_or_near_count": int(
            sum(item["food_contact_or_near_count"] for item in episode_summaries)
        ),
    }
    collection_metrics = {
        "successful_collection_events": int(
            sum(item["successful_collection_events"] for item in episode_summaries)
        ),
        "failed_collect_actions": int(
            sum(item["failed_collect_actions"] for item in episode_summaries)
        ),
        "near_food_but_no_collection_episodes": int(
            sum(
                1
                for item in episode_summaries
                if item["near_food_steps"] > 0 and item["successful_collection_events"] == 0
            )
        ),
        "joint_near_food_no_collection_steps": int(
            sum(item["joint_near_food_no_collection_steps"] for item in episode_summaries)
        ),
        "collection_success_rate_if_available": _collection_success_rate(
            episode_summaries,
            observable=collection_event_observable_any,
        ),
        "collection_event_observable": collection_event_observable_any,
    }
    target_metrics = {
        "same_target_ratio": _mean(item["same_target_ratio"] for item in episode_summaries),
        "target_entropy_mean": _mean(item["target_entropy_mean"] for item in episode_summaries),
        "target_switch_count_mean": _mean(item["target_switch_count_mean"] for item in episode_summaries),
        "agents_chasing_same_food_steps": int(
            sum(item["agents_chasing_same_food_steps"] for item in episode_summaries)
        ),
    }
    stability_metrics = {
        "action_switch_count_mean": _mean(
            item["action_switch_count_mean"] for item in episode_summaries
        ),
        "backtracking_ratio_mean": _mean(
            item["backtracking_ratio"] for item in episode_summaries
        ),
        "repeated_position_ratio": _mean(
            item["repeated_position_ratio"] for item in episode_summaries
        ),
        "episode_length_mean": _mean(item["episode_length"] for item in episode_summaries),
        "return_mean": _mean(item["return"] for item in episode_summaries),
    }

    evidence_flags = {
        "low_coverage": coverage_metrics["joint_visited_cell_ratio"] < 0.25,
        "food_discovery_failure": _safe_ratio(
            food_metrics["episodes_never_close_to_food"], max(len(episode_summaries), 1)
        )
        >= 0.5,
        "near_food_collection_failure": (
            food_metrics["near_food_steps_mean"] > 0
            and collection_metrics["successful_collection_events"] == 0
        ),
        "over_concentration_possible": (
            target_metrics["same_target_ratio"] > 0.7
            and collection_metrics["successful_collection_events"] <= 1
        ),
        "unstable_behavior": (
            stability_metrics["backtracking_ratio_mean"] > 0.35
            or stability_metrics["repeated_position_ratio"] > 0.45
        ),
        "insufficient_behavior_evidence": False,
    }

    behavior_signals = _build_behavior_signals(
        coverage_metrics=coverage_metrics,
        food_metrics=food_metrics,
        collection_metrics=collection_metrics,
        target_metrics=target_metrics,
        stability_metrics=stability_metrics,
        evidence_flags=evidence_flags,
    )
    summary = {
        "intervention_point": stage_label,
        "env_key": env_key,
        "eval_episodes": num_eval_episodes,
        "insufficient_evidence": False,
        "insufficient_evidence_reason": "",
        "collector_status": "lbf_mock_state_collected",
        "milestone": {
            "stage_label": stage_label,
            "target_step": int(milestone.get("target_step") or 0),
            "actual_step": int(milestone.get("actual_step") or 0),
            "num_eval_episodes": num_eval_episodes,
        },
        "coverage": {
            "score": coverage_metrics["joint_visited_cell_ratio"],
            "evidence": behavior_signals["coverage"],
            "metrics": coverage_metrics,
        },
        "food_discovery": {
            "score": _invert_distance_score(food_metrics["mean_nearest_food_distance"]),
            "evidence": behavior_signals["food_discovery"],
            "metrics": food_metrics,
        },
        "failed_collect": {
            "score": _failure_score(
                collection_metrics["failed_collect_actions"],
                max(collection_metrics["successful_collection_events"], 1),
            ),
            "evidence": behavior_signals["failed_collect"],
            "metrics": collection_metrics,
        },
        "target_concentration": {
            "score": target_metrics["same_target_ratio"],
            "evidence": behavior_signals["target_concentration"],
            "metrics": target_metrics,
        },
        "stability": {
            "score": 1.0 - min(
                1.0,
                (stability_metrics["backtracking_ratio_mean"] + stability_metrics["repeated_position_ratio"]) / 2.0,
            ),
            "evidence": behavior_signals["stability"],
            "metrics": stability_metrics,
        },
        "evidence_flags": evidence_flags,
        "behavior_signals": (
            behavior_signals["coverage"]
            + behavior_signals["food_discovery"]
            + behavior_signals["failed_collect"]
            + behavior_signals["target_concentration"]
            + behavior_signals["stability"]
        ),
        "episode_summaries": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in episode_summaries
        ],
        "metadata": {
            "collection_event_observable": collection_event_observable_any,
            "field_area": field_area,
            "raw_trajectory_saved": False,
        },
    }
    return validate_policy_behavior_summary(summary)


def build_mock_behavior_summary(
    *,
    intervention_point: str,
    env_key: str = "lbforaging:Foraging-10x10-3p-3f-v3",
    eval_episodes: int = 10,
    profile: str = "stage1_sparse",
) -> Dict[str, Any]:
    from .policy_behavior_collector import build_synthetic_profile_traces

    if profile == "insufficient":
        return build_insufficient_behavior_summary(
            intervention_point=intervention_point,
            env_key=env_key,
            eval_episodes=eval_episodes,
            reason="mock profile requested insufficient evidence fallback",
        )
    traces = build_synthetic_profile_traces(
        profile=profile,
        env_key=env_key,
        episode_count=eval_episodes,
    )
    milestone = {
        "stage_label": intervention_point,
        "target_step": 0,
        "actual_step": 0,
        "env_key": env_key,
        "num_eval_episodes": eval_episodes,
        "episodes": [
            _convert_trace_to_episode(trace, index)
            for index, trace in enumerate(traces)
        ],
    }
    return build_compact_behavior_summary(milestone=milestone)


def build_branch_behavior_summaries(
    branch_ids: Iterable[str],
    *,
    intervention_point: str,
    env_key: str,
    eval_episodes_per_branch: int,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for index, branch_id in enumerate(branch_ids):
        summary = build_mock_behavior_summary(
            intervention_point=intervention_point,
            env_key=env_key,
            eval_episodes=eval_episodes_per_branch,
            profile="c1_round1_branch",
        )
        summary["metadata"]["branch_id"] = str(branch_id)
        summary["behavior_signals"].append(
            f"Branch {branch_id} synthetic summary index {index} retained for payload testing."
        )
        summaries.append(summary)
    return summaries


def _summarize_episode(
    *,
    episode: Dict[str, Any],
    near_food_distance_threshold: int,
    idle_action_values: set[Any],
    collect_action_values: set[Any],
) -> Dict[str, Any]:
    steps = list(episode.get("steps") or [])
    if not steps:
        return {
            "episode_id": int(episode.get("episode_id") or 0),
            "return": 0.0,
            "episode_length": 0,
            "visited_cell_ratio": 0.0,
            "per_agent_visited_cell_ratio_mean": 0.0,
            "idle_action_ratio": 0.0,
            "backtracking_ratio": 0.0,
            "oscillation_count": 0,
            "mean_nearest_food_distance": None,
            "min_nearest_food_distance": None,
            "near_food_steps": 0,
            "first_near_food_step": None,
            "food_contact_or_near_count": 0,
            "successful_collection_events": 0,
            "failed_collect_actions": 0,
            "joint_near_food_no_collection_steps": 0,
            "same_target_ratio": 0.0,
            "target_entropy_mean": 0.0,
            "target_switch_count_mean": 0.0,
            "agents_chasing_same_food_steps": 0,
            "action_switch_count_mean": 0.0,
            "repeated_position_ratio": 0.0,
            "collection_event_observable": False,
            "summary_text": "No steps recorded.",
            "_joint_visited_cells": [],
            "_field_area": None,
        }

    first_step = steps[0]
    field_size = first_step.get("field_size")
    field_area = int(field_size[0] * field_size[1]) if field_size else None
    joint_visited: set[Tuple[int, int]] = set()
    per_agent_visited: Dict[int, set[Tuple[int, int]]] = {}
    previous_positions: Dict[int, List[Tuple[int, int]]] = {}
    previous_actions: Dict[int, Any] = {}
    target_history: Dict[int, List[str]] = {}

    total_agent_steps = 0
    idle_actions = 0
    backtracking_events = 0
    repeated_position_steps = 0
    oscillation_count = 0
    distance_samples: List[float] = []
    min_distance_per_step: List[float] = []
    near_food_steps = 0
    first_near_food_step = None
    food_contact_or_near_count = 0
    successful_collection_events = 0
    failed_collect_actions = 0
    joint_near_food_no_collection_steps = 0
    same_target_steps = 0
    target_entropy_values: List[float] = []
    action_switch_counts: Counter[int] = Counter()

    collection_event_observable = bool(episode.get("collection_event_observable", False))
    total_reward = float(episode.get("total_reward") or 0.0)

    for step_index, step in enumerate(steps):
        agents = list(step.get("agents") or [])
        foods = list(step.get("foods") or [])
        actions = list(step.get("actions") or [])
        collection_events = list(step.get("collection_events") or [])
        if step.get("collection_event_observable", False):
            collection_event_observable = True
        if collection_events:
            successful_collection_events += len(collection_events)
        step_targets: List[str] = []
        step_near_any = False
        step_collection = bool(collection_events)
        for agent_index, agent in enumerate(agents):
            agent_id = int(agent.get("id", agent_index))
            position = tuple(agent.get("position") or ())
            if len(position) != 2:
                continue
            joint_visited.add(position)
            per_agent_visited.setdefault(agent_id, set()).add(position)
            total_agent_steps += 1
            action = actions[agent_index] if agent_index < len(actions) else None
            if action in idle_action_values:
                idle_actions += 1
            if agent_id in previous_actions and previous_actions[agent_id] != action:
                action_switch_counts[agent_id] += 1
            previous_actions[agent_id] = action

            history = previous_positions.setdefault(agent_id, [])
            history.append(position)
            if len(history) >= 2 and history[-1] == history[-2]:
                repeated_position_steps += 1
            if len(history) >= 3 and history[-1] == history[-3] and history[-1] != history[-2]:
                backtracking_events += 1
                oscillation_count += 1

            nearest_food_id = None
            nearest_distance = None
            for food in foods:
                food_position = tuple(food.get("position") or ())
                if len(food_position) != 2:
                    continue
                distance = abs(position[0] - food_position[0]) + abs(position[1] - food_position[1])
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_food_id = str(food.get("id"))
            if nearest_distance is not None:
                distance_samples.append(float(nearest_distance))
                min_distance_per_step.append(float(nearest_distance))
                if nearest_distance <= near_food_distance_threshold:
                    step_near_any = True
                    food_contact_or_near_count += 1
            if nearest_food_id:
                step_targets.append(nearest_food_id)
                target_history.setdefault(agent_id, []).append(nearest_food_id)
            if nearest_distance is not None and nearest_distance <= near_food_distance_threshold:
                if action in collect_action_values and not step_collection:
                    failed_collect_actions += 1

        if step_near_any:
            near_food_steps += 1
            if first_near_food_step is None:
                first_near_food_step = step_index
            if not step_collection:
                joint_near_food_no_collection_steps += 1
        if step_targets:
            counts = Counter(step_targets)
            if counts and counts.most_common(1)[0][1] > 1:
                same_target_steps += 1
            target_entropy_values.append(_entropy_from_counter(counts))

    per_agent_ratios = [
        _safe_ratio(len(cells), field_area) for cells in per_agent_visited.values()
    ]
    target_switch_counts = []
    for targets in target_history.values():
        switches = 0
        for left, right in zip(targets, targets[1:]):
            if left != right:
                switches += 1
        target_switch_counts.append(switches)

    summary_text = (
        f"Visited {len(joint_visited)} cells, near-food steps={near_food_steps}, "
        f"collections={successful_collection_events}, failed_collect_actions={failed_collect_actions}."
    )
    return {
        "episode_id": int(episode.get("episode_id") or 0),
        "return": total_reward,
        "episode_length": len(steps),
        "visited_cell_ratio": _safe_ratio(len(joint_visited), field_area),
        "per_agent_visited_cell_ratio_mean": _mean(per_agent_ratios),
        "idle_action_ratio": _safe_ratio(idle_actions, total_agent_steps),
        "backtracking_ratio": _safe_ratio(backtracking_events, total_agent_steps),
        "oscillation_count": oscillation_count,
        "mean_nearest_food_distance": _mean(distance_samples),
        "min_nearest_food_distance": min(min_distance_per_step) if min_distance_per_step else None,
        "near_food_steps": near_food_steps,
        "first_near_food_step": first_near_food_step,
        "food_contact_or_near_count": food_contact_or_near_count,
        "successful_collection_events": successful_collection_events,
        "failed_collect_actions": failed_collect_actions,
        "joint_near_food_no_collection_steps": joint_near_food_no_collection_steps,
        "same_target_ratio": _safe_ratio(same_target_steps, len(steps)),
        "target_entropy_mean": _mean(target_entropy_values),
        "target_switch_count_mean": _mean(target_switch_counts),
        "agents_chasing_same_food_steps": same_target_steps,
        "action_switch_count_mean": _mean(action_switch_counts.values()),
        "repeated_position_ratio": _safe_ratio(repeated_position_steps, total_agent_steps),
        "collection_event_observable": collection_event_observable,
        "summary_text": summary_text,
        "_joint_visited_cells": list(joint_visited),
        "_field_area": field_area,
    }


def _build_behavior_signals(
    *,
    coverage_metrics: Dict[str, Any],
    food_metrics: Dict[str, Any],
    collection_metrics: Dict[str, Any],
    target_metrics: Dict[str, Any],
    stability_metrics: Dict[str, Any],
    evidence_flags: Dict[str, Any],
) -> Dict[str, List[str]]:
    coverage = [
        f"coverage.metrics.joint_visited_cell_ratio={coverage_metrics['joint_visited_cell_ratio']:.3f}",
        f"coverage.metrics.idle_action_ratio={coverage_metrics['idle_action_ratio']:.3f}",
    ]
    food = [
        f"food_discovery.metrics.episodes_never_close_to_food={food_metrics['episodes_never_close_to_food']}",
        f"food_discovery.metrics.first_near_food_step_mean={food_metrics['first_near_food_step_mean']}",
    ]
    failed_collect = [
        f"failed_collect.metrics.successful_collection_events={collection_metrics['successful_collection_events']}",
        f"failed_collect.metrics.failed_collect_actions={collection_metrics['failed_collect_actions']}",
    ]
    target = [
        f"target_concentration.metrics.same_target_ratio={target_metrics['same_target_ratio']:.3f}",
        f"target_concentration.metrics.target_entropy_mean={target_metrics['target_entropy_mean']:.3f}",
    ]
    stability = [
        f"stability.metrics.backtracking_ratio_mean={stability_metrics['backtracking_ratio_mean']:.3f}",
        f"stability.metrics.repeated_position_ratio={stability_metrics['repeated_position_ratio']:.3f}",
    ]
    if evidence_flags["low_coverage"]:
        coverage.append("evidence_flags.low_coverage=true")
    if evidence_flags["food_discovery_failure"]:
        food.append("evidence_flags.food_discovery_failure=true")
    if evidence_flags["near_food_collection_failure"]:
        failed_collect.append("evidence_flags.near_food_collection_failure=true")
    if evidence_flags["over_concentration_possible"]:
        target.append("evidence_flags.over_concentration_possible=true")
    if evidence_flags["unstable_behavior"]:
        stability.append("evidence_flags.unstable_behavior=true")
    return {
        "coverage": coverage,
        "food_discovery": food,
        "failed_collect": failed_collect,
        "target_concentration": target,
        "stability": stability,
    }


def _convert_trace_to_episode(trace: List[Dict[str, Any]], episode_id: int) -> Dict[str, Any]:
    total_reward = sum(float(item.get("reward") or 0.0) for item in trace)
    collection_event_observable = any(
        bool((item.get("info") or {}).get("collection_events"))
        for item in trace
        if isinstance(item, dict)
    )
    return {
        "episode_id": episode_id,
        "steps": [
            {
                "field_size": ((item.get("env_state") or {}).get("field_size")),
                "agents": _normalize_agents((item.get("env_state") or {}).get("agents")),
                "foods": _normalize_foods((item.get("env_state") or {}).get("foods")),
                "actions": list(item.get("actions") or []),
                "reward": float(item.get("reward") or 0.0),
                "collection_events": list(((item.get("info") or {}).get("collection_events")) or []),
                "collection_event_observable": collection_event_observable,
            }
            for item in trace
        ],
        "total_reward": total_reward,
        "collection_event_observable": collection_event_observable,
    }


def _normalize_agents(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        position = item.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            continue
        normalized.append(
            {
                "id": int(item.get("id", index)),
                "position": (int(position[0]), int(position[1])),
            }
        )
    return normalized


def _normalize_foods(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        position = item.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            continue
        normalized.append(
            {
                "id": str(item.get("id", f"food_{index}")),
                "position": (int(position[0]), int(position[1])),
            }
        )
    return normalized


def _entropy_from_counter(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counter.values():
        probability = value / total
        entropy -= probability * math.log(probability + 1e-12, 2)
    return entropy


def _collection_success_rate(
    episode_summaries: List[Dict[str, Any]],
    *,
    observable: bool,
) -> float | None:
    if not observable:
        return None
    success = sum(int(item["successful_collection_events"]) for item in episode_summaries)
    failed = sum(int(item["failed_collect_actions"]) for item in episode_summaries)
    return _safe_ratio(success, success + failed)


def _invert_distance_score(value: float | None) -> float | None:
    if value is None:
        return None
    return 1.0 / (1.0 + float(value))


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
