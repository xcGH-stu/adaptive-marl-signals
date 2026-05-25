from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from typing import Any, Dict, Iterable, List, Optional

from .policy_behavior_summary import (
    build_branch_behavior_summaries,
    build_compact_behavior_summary,
)
from .policy_guidance_config import validate_policy_guidance_config
from .policy_guidance_schema import (
    build_insufficient_behavior_summary,
    validate_policy_behavior_summary,
)
from .rware_behavior_summary import build_rware_behavior_summary_from_traces


class PolicyBehaviorCollector:
    def __init__(
        self,
        *,
        near_food_distance_threshold: int = 1,
        idle_action_values: Iterable[Any] | None = None,
        collect_action_values: Iterable[Any] | None = None,
    ) -> None:
        self.near_food_distance_threshold = int(near_food_distance_threshold)
        self.idle_action_values = set(idle_action_values or [None, 0, 4, "stay", "STAY"])
        self.collect_action_values = set(
            collect_action_values or [5, "collect", "COLLECT", "load", "LOAD"]
        )
        self.current_milestone: Dict[str, Any] | None = None
        self.current_episode: Dict[str, Any] | None = None
        self._summary: Dict[str, Any] | None = None

    def start_milestone(
        self,
        stage_label: str,
        target_step: int,
        actual_step: int,
        env_key: str,
        num_eval_episodes: int,
    ) -> None:
        self.current_milestone = {
            "stage_label": str(stage_label),
            "target_step": int(target_step),
            "actual_step": int(actual_step),
            "env_key": str(env_key),
            "num_eval_episodes": int(num_eval_episodes),
            "episodes": [],
            "collector_status": "collecting",
            "insufficient_evidence_reason": "",
            "supports_state": False,
        }
        self.current_episode = None
        self._summary = None

    def start_episode(self, episode_id: int) -> None:
        if self.current_milestone is None:
            raise RuntimeError("start_milestone must be called before start_episode")
        self.current_episode = {
            "episode_id": int(episode_id),
            "steps": [],
            "has_any_state": False,
            "has_any_obs": False,
            "field_size": None,
            "total_reward": 0.0,
            "collection_event_observable": False,
        }

    def record_step(
        self,
        step_index: int,
        env_state: Any = None,
        obs: Any = None,
        actions: Any = None,
        reward: Any = None,
        terminated: bool = False,
        info: Any = None,
    ) -> None:
        if self.current_episode is None:
            raise RuntimeError("start_episode must be called before record_step")
        parsed = _parse_lbf_like_step(env_state=env_state, obs=obs, actions=actions, reward=reward, terminated=terminated, info=info)
        if parsed["has_state"]:
            self.current_episode["has_any_state"] = True
            self.current_episode["field_size"] = parsed["field_size"]
            if self.current_milestone is not None:
                self.current_milestone["supports_state"] = True
        if parsed["has_obs"]:
            self.current_episode["has_any_obs"] = True
        self.current_episode["total_reward"] += float(parsed["reward"] or 0.0)
        if parsed["collection_event_observable"]:
            self.current_episode["collection_event_observable"] = True
        parsed["step_index"] = int(step_index)
        self.current_episode["steps"].append(parsed)

    def end_episode(self) -> None:
        if self.current_episode is None or self.current_milestone is None:
            raise RuntimeError("start_episode must be called before end_episode")
        self.current_milestone["episodes"].append(self.current_episode)
        self.current_episode = None

    def summarize_milestone(self) -> Dict[str, Any]:
        if self.current_milestone is None:
            raise RuntimeError("start_milestone must be called before summarize_milestone")
        summary = build_compact_behavior_summary(
            milestone=self.current_milestone,
            near_food_distance_threshold=self.near_food_distance_threshold,
            idle_action_values=self.idle_action_values,
            collect_action_values=self.collect_action_values,
        )
        self._summary = validate_policy_behavior_summary(summary)
        return deepcopy(self._summary)

    def to_json(self) -> Dict[str, Any]:
        return {
            "milestone": deepcopy(self.current_milestone),
            "summary": deepcopy(self._summary),
        }


def collect_policy_behavior_summary(
    *,
    env_key: str,
    intervention_point: str,
    eval_episodes: int,
    policy_guidance_config: Dict[str, Any] | None = None,
    source_run_ref: Dict[str, Any] | None = None,
    mock_profile: str | None = None,
    milestone_stage_label: str | None = None,
    target_step: int | None = None,
    actual_step: int | None = None,
    episode_traces: Optional[List[List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    config = validate_policy_guidance_config(policy_guidance_config)
    supported_envs = list(config.get("policy_guidance_first_supported_envs") or [])
    env_family = _resolve_env_family(env_key)
    if env_family not in supported_envs:
        return build_insufficient_behavior_summary(
            intervention_point=intervention_point,
            env_key=env_key,
            eval_episodes=eval_episodes,
            reason=(
                "env not supported by the first-pass policy guidance collector; "
                "fallback to reward-only"
            ),
        )
    if episode_traces is None:
        episode_traces = build_synthetic_profile_traces(
            profile=mock_profile or "stage1_sparse",
            env_key=env_key,
            episode_count=eval_episodes,
        )
    return collect_policy_behavior_summary_from_traces(
        env_key=env_key,
        intervention_point=intervention_point,
        episode_traces=episode_traces,
        num_eval_episodes=eval_episodes,
        stage_label=milestone_stage_label or intervention_point,
        target_step=int(target_step or 0),
        actual_step=int(actual_step or 0),
        source_run_ref=source_run_ref,
        extractor_preference=config["policy_guidance_extractor_preference"],
    )


def collect_policy_behavior_summary_from_traces(
    *,
    env_key: str,
    intervention_point: str,
    episode_traces: List[List[Dict[str, Any]]],
    num_eval_episodes: int,
    stage_label: str,
    target_step: int,
    actual_step: int,
    source_run_ref: Dict[str, Any] | None = None,
    extractor_preference: str = "env_state_then_obs_action",
) -> Dict[str, Any]:
    env_family = _resolve_env_family(env_key)
    if env_family == "rware":
        return build_rware_behavior_summary_from_traces(
            env_key=env_key,
            intervention_point=intervention_point,
            episode_traces=episode_traces,
            num_eval_episodes=num_eval_episodes,
            stage_label=stage_label,
            target_step=target_step,
            actual_step=actual_step,
            source_run_ref=source_run_ref,
        )
    collector = PolicyBehaviorCollector()
    collector.start_milestone(
        stage_label=stage_label,
        target_step=target_step,
        actual_step=actual_step,
        env_key=env_key,
        num_eval_episodes=num_eval_episodes,
    )
    for episode_id, steps in enumerate(episode_traces):
        collector.start_episode(episode_id)
        for step_index, step in enumerate(steps):
            collector.record_step(
                step_index=step_index,
                env_state=step.get("env_state"),
                obs=step.get("obs"),
                actions=step.get("actions"),
                reward=step.get("reward"),
                terminated=bool(step.get("terminated", False)),
                info=step.get("info"),
            )
        collector.end_episode()
    summary = collector.summarize_milestone()
    summary["intervention_point"] = str(intervention_point)
    summary["metadata"]["source_run_ref"] = deepcopy(source_run_ref or {})
    summary["metadata"]["collector_mode"] = "synthetic_trace_collection"
    summary["metadata"]["extractor_preference"] = str(extractor_preference)
    return validate_policy_behavior_summary(summary)


def collect_policy_behavior_summaries_for_branches(
    *,
    env_key: str,
    intervention_point: str,
    branch_ids: Iterable[str],
    eval_episodes_per_branch: int,
    policy_guidance_config: Dict[str, Any] | None = None,
    branch_episode_traces: Optional[Dict[str, List[List[Dict[str, Any]]]]] = None,
) -> List[Dict[str, Any]]:
    config = validate_policy_guidance_config(policy_guidance_config)
    env_family = _resolve_env_family(env_key)
    if env_family not in list(config.get("policy_guidance_first_supported_envs") or []):
        return [
            build_insufficient_behavior_summary(
                intervention_point=intervention_point,
                env_key=env_key,
                eval_episodes=eval_episodes_per_branch,
                reason="env not supported for branch behavior collection",
            )
        ]
    summaries: List[Dict[str, Any]] = []
    traces_by_branch = branch_episode_traces or {}
    for branch_id in branch_ids:
        traces = traces_by_branch.get(str(branch_id))
        if traces is None:
            traces = build_synthetic_profile_traces(
                profile="c1_round1_branch",
                env_key=env_key,
                episode_count=eval_episodes_per_branch,
            )
        summary = collect_policy_behavior_summary_from_traces(
            env_key=env_key,
            intervention_point=intervention_point,
            episode_traces=traces,
            num_eval_episodes=eval_episodes_per_branch,
            stage_label=intervention_point,
            target_step=0,
            actual_step=0,
            source_run_ref={"branch_id": str(branch_id)},
            extractor_preference=config["policy_guidance_extractor_preference"],
        )
        summary["metadata"]["branch_id"] = str(branch_id)
        summaries.append(summary)
    return summaries


def _resolve_env_family(env_key: str) -> str:
    env_key_str = str(env_key or "")
    if env_key_str.startswith("rware"):
        return "rware"
    if env_key_str.startswith("lbforaging"):
        return "lbforaging"
    return env_key_str.split(":", 1)[0]


def build_synthetic_profile_traces(
    *,
    profile: str,
    env_key: str,
    episode_count: int,
) -> List[List[Dict[str, Any]]]:
    builders = {
        "stage1_sparse": _build_low_coverage_episode,
        "c1_pre_checkpoint": _build_over_concentrated_episode,
        "c1_round1_branch": _build_successful_collection_episode,
        "near_food_but_no_collection": _build_near_food_but_no_collection_episode,
        "over_concentrated_target": _build_over_concentrated_episode,
        "successful_collection": _build_successful_collection_episode,
        "low_coverage_no_food_discovery": _build_low_coverage_episode,
    }
    if profile == "insufficient":
        return [[{"env_state": None, "obs": None, "actions": None, "reward": 0.0, "terminated": True, "info": {}}]]
    builder = builders.get(profile)
    if builder is None:
        raise ValueError(f"Unknown synthetic behavior profile: {profile}")
    return [builder(index=index, env_key=env_key) for index in range(int(episode_count))]


def _build_low_coverage_episode(*, index: int, env_key: str) -> List[Dict[str, Any]]:
    foods = [{"id": "f0", "position": [8, 8], "level": 2}]
    positions = [
        [[1, 1], [1, 2], [2, 1]],
        [[1, 2], [1, 1], [2, 1]],
        [[1, 1], [1, 2], [2, 1]],
        [[1, 2], [1, 1], [2, 1]],
        [[1, 1], [1, 2], [2, 1]],
    ]
    actions = [[4, 4, 0], [1, 0, 4], [0, 1, 4], [1, 0, 4], [0, 1, 4]]
    return _episode_from_positions(positions=positions, actions=actions, foods=foods, rewards=[0, 0, 0, 0, 0], env_key=env_key)


def _build_near_food_but_no_collection_episode(*, index: int, env_key: str) -> List[Dict[str, Any]]:
    foods = [{"id": "f0", "position": [5, 5], "level": 2}]
    positions = [
        [[4, 5], [5, 4]],
        [[5, 5], [5, 4]],
        [[4, 5], [5, 5]],
        [[5, 4], [4, 5]],
        [[4, 5], [5, 4]],
    ]
    actions = [[2, 3], [5, 4], [4, 5], [5, 5], [4, 4]]
    return _episode_from_positions(positions=positions, actions=actions, foods=foods, rewards=[0, 0, 0, 0, 0], env_key=env_key)


def _build_over_concentrated_episode(*, index: int, env_key: str) -> List[Dict[str, Any]]:
    foods = [
        {"id": "f0", "position": [6, 6], "level": 2},
        {"id": "f1", "position": [2, 8], "level": 1},
    ]
    positions = [
        [[3, 3], [3, 4], [4, 3]],
        [[4, 4], [4, 5], [5, 4]],
        [[5, 5], [5, 5], [5, 5]],
        [[5, 6], [5, 6], [5, 6]],
        [[5, 5], [5, 5], [5, 5]],
    ]
    actions = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [5, 5, 5], [4, 4, 4]]
    return _episode_from_positions(positions=positions, actions=actions, foods=foods, rewards=[0, 0, 0, 0, 0], env_key=env_key)


def _build_successful_collection_episode(*, index: int, env_key: str) -> List[Dict[str, Any]]:
    foods = [{"id": "f0", "position": [5, 5], "level": 2}]
    positions = [
        [[3, 5], [4, 5]],
        [[4, 5], [5, 5]],
        [[5, 5], [5, 5]],
        [[5, 5], [5, 5]],
    ]
    actions = [[1, 1], [1, 1], [5, 5], [4, 4]]
    infos = [{}, {}, {"collection_events": [{"food_id": "f0", "agents": [0, 1]}]}, {}]
    rewards = [0, 0, 1, 0]
    return _episode_from_positions(
        positions=positions,
        actions=actions,
        foods=foods,
        rewards=rewards,
        env_key=env_key,
        infos=infos,
    )


def _episode_from_positions(
    *,
    positions: List[List[List[int]]],
    actions: List[List[Any]],
    foods: List[Dict[str, Any]],
    rewards: List[float],
    env_key: str,
    infos: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    field_size = [10, 10]
    steps: List[Dict[str, Any]] = []
    info_list = infos or [{} for _ in positions]
    for index, agent_positions in enumerate(positions):
        env_state = {
            "field_size": field_size,
            "agents": [
                {"id": agent_id, "position": list(position), "level": 1}
                for agent_id, position in enumerate(agent_positions)
            ],
            "foods": deepcopy(foods),
            "current_step": index,
        }
        steps.append(
            {
                "env_state": env_state,
                "actions": list(actions[index]),
                "reward": float(rewards[index]),
                "terminated": index == len(positions) - 1,
                "info": deepcopy(info_list[index]),
                "env_key": env_key,
            }
        )
    return steps


def _parse_lbf_like_step(
    *,
    env_state: Any,
    obs: Any,
    actions: Any,
    reward: Any,
    terminated: bool,
    info: Any,
) -> Dict[str, Any]:
    payload = {
        "has_state": False,
        "has_obs": obs is not None,
        "field_size": None,
        "agents": [],
        "foods": [],
        "actions": list(actions) if isinstance(actions, (list, tuple)) else [],
        "reward": _scalarize_reward(reward),
        "terminated": bool(terminated),
        "info": deepcopy(info or {}),
        "collection_events": [],
        "collection_event_observable": False,
    }
    source = env_state if isinstance(env_state, dict) else None
    if source is None and isinstance(obs, dict):
        source = obs
    if isinstance(source, dict):
        field_size = source.get("field_size") or source.get("grid_size")
        if isinstance(field_size, (list, tuple)) and len(field_size) == 2:
            payload["field_size"] = [int(field_size[0]), int(field_size[1])]
        payload["agents"] = _normalize_agents(source.get("agents"))
        payload["foods"] = _normalize_foods(source.get("foods"))
        payload["has_state"] = bool(payload["field_size"] and payload["agents"] is not None)
    info_dict = payload["info"] if isinstance(payload["info"], dict) else {}
    collection_events = info_dict.get("collection_events")
    if isinstance(collection_events, list):
        payload["collection_events"] = deepcopy(collection_events)
        payload["collection_event_observable"] = True
    elif isinstance(info_dict.get("successful_collection_events"), int):
        payload["collection_events"] = [
            {"source": "successful_collection_events"}
            for _ in range(int(info_dict.get("successful_collection_events") or 0))
        ]
        payload["collection_event_observable"] = True
    elif info_dict.get("collected_food_ids"):
        payload["collection_events"] = [
            {"food_id": item} for item in list(info_dict.get("collected_food_ids") or [])
        ]
        payload["collection_event_observable"] = True
    return payload


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
                "level": int(item.get("level", 1)),
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
                "level": int(item.get("level", 1)),
            }
        )
    return normalized


def _scalarize_reward(reward: Any) -> float:
    if reward is None:
        return 0.0
    if isinstance(reward, (list, tuple)):
        return float(sum(float(item) for item in reward))
    return float(reward)
