from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


DEFAULT_POLICY_GUIDANCE_CONFIG: Dict[str, Any] = {
    "enable_policy_guidance": False,
    "policy_guidance_mode": "context_only",
    "policy_guidance_direct_action_override": False,
    "policy_guidance_modify_learner": False,
    "policy_guidance_fallback_to_reward_only": True,
    "policy_guidance_collect_stage1_milestones": True,
    "policy_guidance_points": [
        "stage1_after_sparse_baseline",
        "stage3_c1_pre_checkpoint",
        "stage3_c1_after_round1",
        "stage3_c2_pre_checkpoint",
        "stage3_c2_after_round1",
    ],
    "policy_guidance_stage1_milestones": [200000, 500000, 800000, 2050000],
    "policy_guidance_eval_episodes_stage1_per_milestone": 10,
    "policy_guidance_stage1_milestone_match_tolerance": 25000,
    "policy_guidance_c1_pre_eval_episodes": 10,
    "policy_guidance_c1_after_round1_eval_episodes_per_branch": 5,
    "policy_guidance_c2_pre_eval_episodes": 10,
    "policy_guidance_c2_after_round1_eval_episodes_per_branch": 10,
    "policy_guidance_collect_stage1_intervals": False,
    "policy_guidance_collect_stage1_final": True,
    "policy_guidance_save_raw_trajectories": False,
    "policy_guidance_save_compact_episode_summaries": True,
    "policy_guidance_stage1_behavior_summary_required": False,
    "policy_guidance_stage1_behavior_summary_path": None,
    "policy_guidance_stage1_behavior_summary_reuse_allowed": False,
    "policy_guidance_stage1_behavior_summary_must_be_fresh": True,
    "allow_reuse_stage1_behavior_summary": False,
    "policy_guidance_stage1b_use_integrated_guidance": True,
    "policy_guidance_stage1b_fallback_to_reward_only": True,
    "policy_guidance_stage3_use_integrated_guidance": True,
    "policy_guidance_stage3_fallback_to_reward_only": True,
    "policy_guidance_stage3_behavior_summary_required": False,
    "policy_guidance_stage3_record_manifest": True,
    "policy_guidance_stage3_record_memory": True,
    "policy_guidance_first_supported_envs": ["lbforaging", "rware"],
    "policy_guidance_extractor_preference": "env_state_then_obs_action",
}


def get_default_policy_guidance_config() -> Dict[str, Any]:
    return deepcopy(DEFAULT_POLICY_GUIDANCE_CONFIG)


def validate_policy_guidance_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(DEFAULT_POLICY_GUIDANCE_CONFIG)
    if config is None:
        return payload
    if not isinstance(config, dict):
        raise ValueError("policy_guidance_config must be a dict")
    payload.update(deepcopy(config))
    payload["enable_policy_guidance"] = bool(payload.get("enable_policy_guidance", False))
    payload["policy_guidance_mode"] = str(payload.get("policy_guidance_mode") or "context_only")
    if payload["policy_guidance_mode"] != "context_only":
        raise ValueError("policy_guidance_mode must be context_only in this scaffold")
    for key in (
        "policy_guidance_direct_action_override",
        "policy_guidance_modify_learner",
        "policy_guidance_fallback_to_reward_only",
        "policy_guidance_collect_stage1_milestones",
        "policy_guidance_collect_stage1_intervals",
        "policy_guidance_collect_stage1_final",
        "policy_guidance_save_raw_trajectories",
        "policy_guidance_save_compact_episode_summaries",
        "policy_guidance_stage1_behavior_summary_required",
        "policy_guidance_stage1_behavior_summary_reuse_allowed",
        "policy_guidance_stage1_behavior_summary_must_be_fresh",
        "allow_reuse_stage1_behavior_summary",
        "policy_guidance_stage1b_use_integrated_guidance",
        "policy_guidance_stage1b_fallback_to_reward_only",
        "policy_guidance_stage3_use_integrated_guidance",
        "policy_guidance_stage3_fallback_to_reward_only",
        "policy_guidance_stage3_behavior_summary_required",
        "policy_guidance_stage3_record_manifest",
        "policy_guidance_stage3_record_memory",
    ):
        payload[key] = bool(payload.get(key, False))
    for key in (
        "policy_guidance_eval_episodes_stage1_per_milestone",
        "policy_guidance_stage1_milestone_match_tolerance",
        "policy_guidance_c1_pre_eval_episodes",
        "policy_guidance_c1_after_round1_eval_episodes_per_branch",
        "policy_guidance_c2_pre_eval_episodes",
        "policy_guidance_c2_after_round1_eval_episodes_per_branch",
    ):
        payload[key] = int(payload.get(key) or 0)
    payload["policy_guidance_stage1_milestones"] = [
        int(value) for value in (payload.get("policy_guidance_stage1_milestones") or [])
    ]
    payload["policy_guidance_points"] = [
        str(value) for value in (payload.get("policy_guidance_points") or [])
    ]
    payload["policy_guidance_first_supported_envs"] = [
        str(value) for value in (payload.get("policy_guidance_first_supported_envs") or [])
    ]
    summary_path = payload.get("policy_guidance_stage1_behavior_summary_path")
    payload["policy_guidance_stage1_behavior_summary_path"] = (
        None if summary_path in (None, "") else str(summary_path)
    )
    payload["policy_guidance_stage1_behavior_summary_reuse_allowed"] = bool(
        payload.get("policy_guidance_stage1_behavior_summary_reuse_allowed", False)
    )
    payload["allow_reuse_stage1_behavior_summary"] = bool(
        payload.get("allow_reuse_stage1_behavior_summary", False)
    )
    if payload["allow_reuse_stage1_behavior_summary"]:
        payload["policy_guidance_stage1_behavior_summary_reuse_allowed"] = True
    payload["policy_guidance_stage1_behavior_summary_must_be_fresh"] = bool(
        payload.get("policy_guidance_stage1_behavior_summary_must_be_fresh", True)
    )
    payload["policy_guidance_extractor_preference"] = str(
        payload.get("policy_guidance_extractor_preference")
        or "env_state_then_obs_action"
    )
    if payload["policy_guidance_direct_action_override"]:
        raise ValueError("policy_guidance_direct_action_override must remain false")
    if payload["policy_guidance_modify_learner"]:
        raise ValueError("policy_guidance_modify_learner must remain false")
    return payload


def summarize_policy_guidance_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = validate_policy_guidance_config(config)
    return {
        "enable_policy_guidance": payload["enable_policy_guidance"],
        "policy_guidance_mode": payload["policy_guidance_mode"],
        "fallback_to_reward_only": payload["policy_guidance_fallback_to_reward_only"],
        "collect_stage1_milestones": payload["policy_guidance_collect_stage1_milestones"],
        "policy_guidance_points": payload["policy_guidance_points"],
        "stage1_milestones": payload["policy_guidance_stage1_milestones"],
        "stage1_behavior_summary_required": payload["policy_guidance_stage1_behavior_summary_required"],
        "stage1_behavior_summary_path": payload["policy_guidance_stage1_behavior_summary_path"],
        "stage1_behavior_summary_reuse_allowed": payload[
            "policy_guidance_stage1_behavior_summary_reuse_allowed"
        ],
        "stage1_behavior_summary_must_be_fresh": payload[
            "policy_guidance_stage1_behavior_summary_must_be_fresh"
        ],
        "stage1b_use_integrated_guidance": payload["policy_guidance_stage1b_use_integrated_guidance"],
        "stage1b_fallback_to_reward_only": payload["policy_guidance_stage1b_fallback_to_reward_only"],
        "stage3_use_integrated_guidance": payload["policy_guidance_stage3_use_integrated_guidance"],
        "stage3_fallback_to_reward_only": payload["policy_guidance_stage3_fallback_to_reward_only"],
        "stage3_behavior_summary_required": payload["policy_guidance_stage3_behavior_summary_required"],
        "stage3_record_manifest": payload["policy_guidance_stage3_record_manifest"],
        "stage3_record_memory": payload["policy_guidance_stage3_record_memory"],
        "stage1_milestone_match_tolerance": payload["policy_guidance_stage1_milestone_match_tolerance"],
        "supported_envs": payload["policy_guidance_first_supported_envs"],
        "extractor_preference": payload["policy_guidance_extractor_preference"],
    }
