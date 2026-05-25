from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


COMPACTION_VERSION = "v1"
DEFAULT_LLM_INPUT_COMPACTION = {
    "enabled": True,
    "max_payload_chars_validation": 12000,
    "max_payload_chars_formal": 20000,
    "max_behavior_items": 8,
    "max_metric_points": 8,
    "max_candidates_in_context": 4,
    "include_raw_trajectory": False,
    "include_full_metrics_history": False,
    "include_full_behavior_summary": False,
    "include_local_artifact_paths": True,
    "include_evidence_keys": True,
}


def detect_env_family(
    env_key: str | None = None,
    pbrs_version: str | None = None,
) -> str:
    env_text = str(env_key or "").strip().lower()
    pbrs_text = str(pbrs_version or "").strip().lower()
    if "rware" in pbrs_text or "rware" in env_text:
        return "rware"
    if "lbf" in pbrs_text or "lbforaging" in env_text or "foraging" in env_text:
        return "lbf"
    return "generic"


def estimate_payload_size_chars(payload: Any) -> int:
    try:
        return len(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
    except Exception:
        return len(str(payload))


def attach_compaction_metadata(payload: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    attached = deepcopy(payload or {})
    attached["llm_input_compaction"] = deepcopy(metadata or {})
    return attached


def compact_stage1b_payload(
    payload: Dict[str, Any],
    *,
    env_key: str | None = None,
    pbrs_version: str | None = None,
    tier: str = "validation",
) -> Dict[str, Any]:
    compacted = deepcopy(payload or {})
    env_family = detect_env_family(
        env_key=env_key or _extract_env_key(compacted),
        pbrs_version=pbrs_version or _extract_pbrs_version(compacted),
    )
    cfg = _tier_config(tier=tier)
    pre_chars = estimate_payload_size_chars(compacted)
    included_sections: List[str] = []
    omitted_sections: List[str] = []

    _compact_stage1_like_payload(
        compacted,
        env_family=env_family,
        cfg=cfg,
        included_sections=included_sections,
        omitted_sections=omitted_sections,
    )
    local_refs = _collect_local_artifact_references(compacted)
    compacted = enforce_payload_budget(compacted, max_chars=int(cfg["max_payload_chars"]))
    metadata = _build_metadata(
        compacted=compacted,
        env_family=env_family,
        pre_chars=pre_chars,
        cfg=cfg,
        included_sections=included_sections,
        omitted_sections=omitted_sections,
        local_refs=local_refs,
    )
    return attach_compaction_metadata(compacted, metadata)


def compact_stage3_payload(
    payload: Dict[str, Any],
    *,
    env_key: str | None = None,
    pbrs_version: str | None = None,
    tier: str = "validation",
) -> Dict[str, Any]:
    compacted = deepcopy(payload or {})
    env_family = detect_env_family(
        env_key=env_key or _extract_env_key(compacted),
        pbrs_version=pbrs_version or _extract_pbrs_version(compacted),
    )
    cfg = _tier_config(tier=tier)
    pre_chars = estimate_payload_size_chars(compacted)
    included_sections: List[str] = []
    omitted_sections: List[str] = []

    if "candidate_results" in compacted and isinstance(compacted.get("candidate_results"), list):
        compacted["candidate_results"] = [
            _compact_stage3_candidate_result(item, env_family=env_family)
            for item in list(compacted.get("candidate_results") or [])
        ]
        included_sections.append("candidate_results")
        omitted_sections.extend(
            [
                "candidate_results.run_dir",
                "candidate_results.full_metrics_curves",
                "candidate_results.verbose_metadata",
            ]
        )
    if "dense_reference_run_summary" in compacted:
        compacted["dense_reference_run_summary"] = compact_metrics_history_for_llm(
            compacted.get("dense_reference_run_summary"),
            max_points=int(cfg["max_metric_points"]),
        )
        included_sections.append("dense_reference_run_summary")
        omitted_sections.append("dense_reference_run_summary.run_metadata.env_args")
    if "sparse_reference_run_summary" in compacted:
        compacted["sparse_reference_run_summary"] = compact_metrics_history_for_llm(
            compacted.get("sparse_reference_run_summary"),
            max_points=int(cfg["max_metric_points"]),
        )
        included_sections.append("sparse_reference_run_summary")
        omitted_sections.append("sparse_reference_run_summary.run_metadata.env_args")
    if "reference_context" in compacted:
        compacted["reference_context"] = _compact_generic_context(compacted.get("reference_context"))
        included_sections.append("reference_context")
    if "reference_comparison" in compacted:
        compacted["reference_comparison"] = _compact_reference_comparison(
            compacted.get("reference_comparison"),
            env_family=env_family,
        )
        included_sections.append("reference_comparison")
    if "winner_history" in compacted and isinstance(compacted.get("winner_history"), list):
        compacted["winner_history"] = [
            _compact_stage3_candidate_result(item, env_family=env_family)
            for item in list(compacted.get("winner_history") or [])[-int(cfg["max_candidates_in_context"]) :]
        ]
        included_sections.append("winner_history")
        omitted_sections.append("winner_history.older_entries")
    if "round1_result_summary" in compacted:
        compacted["round1_result_summary"] = _compact_round_result_summary(
            compacted.get("round1_result_summary")
        )
        included_sections.append("round1_result_summary")
    if "policy_guidance" in compacted and isinstance(compacted.get("policy_guidance"), dict):
        compacted["policy_guidance"] = _compact_policy_guidance_payload(
            compacted.get("policy_guidance"),
            env_family=env_family,
            cfg=cfg,
        )
        included_sections.append("policy_guidance")
    if "integrated_guidance_card" in compacted and isinstance(compacted.get("integrated_guidance_card"), dict):
        compacted["integrated_guidance_card"] = _compact_integrated_guidance_card_for_llm(
            compacted.get("integrated_guidance_card"),
            include_evidence_keys=bool(cfg["include_evidence_keys"]),
        )
        included_sections.append("integrated_guidance_card")
    if "behavior_summary" in compacted and isinstance(compacted.get("behavior_summary"), dict):
        compacted["behavior_summary"] = compact_behavior_summary_for_llm(
            compacted.get("behavior_summary"),
            env_family=env_family,
        )
        included_sections.append("behavior_summary")
        omitted_sections.append("behavior_summary.full_episode_summaries")
    if "branch_behavior_summaries" in compacted and isinstance(compacted.get("branch_behavior_summaries"), list):
        compacted["branch_behavior_summaries"] = [
            compact_behavior_summary_for_llm(item, env_family=env_family)
            for item in list(compacted.get("branch_behavior_summaries") or [])[: int(cfg["max_candidates_in_context"])]
        ]
        included_sections.append("branch_behavior_summaries")
        omitted_sections.append("branch_behavior_summaries.truncated")
    if "branch_results" in compacted and isinstance(compacted.get("branch_results"), list):
        compacted["branch_results"] = [
            _compact_stage3_candidate_result(item, env_family=env_family)
            for item in list(compacted.get("branch_results") or [])
        ]
        included_sections.append("branch_results")
        omitted_sections.extend(
            ["branch_results.run_metadata", "branch_results.full_history", "branch_results.checkpoint_paths"]
        )
    local_refs = _collect_local_artifact_references(compacted)
    compacted = enforce_payload_budget(compacted, max_chars=int(cfg["max_payload_chars"]))
    metadata = _build_metadata(
        compacted=compacted,
        env_family=env_family,
        pre_chars=pre_chars,
        cfg=cfg,
        included_sections=included_sections,
        omitted_sections=omitted_sections,
        local_refs=local_refs,
    )
    return attach_compaction_metadata(compacted, metadata)


def compact_behavior_summary_for_llm(
    behavior_summary: Dict[str, Any] | None,
    *,
    env_family: str,
) -> Dict[str, Any]:
    summary = deepcopy(behavior_summary or {})
    if not isinstance(summary, dict):
        return {}
    if env_family == "rware":
        return compact_rware_behavior_summary_for_llm(summary)
    if env_family == "lbf":
        return compact_lbf_behavior_summary_for_llm(summary)
    return _compact_generic_behavior_summary(summary)


def compact_metrics_history_for_llm(
    metrics_summary: Dict[str, Any] | None,
    *,
    max_points: int,
) -> Dict[str, Any]:
    payload = deepcopy(metrics_summary or {})
    if not isinstance(payload, dict):
        return {}
    compacted = {}
    if "run_metadata" in payload:
        compacted["run_metadata"] = _compact_run_metadata(payload.get("run_metadata"))
    if "metric_summary" in payload and isinstance(payload.get("metric_summary"), dict):
        metric_summary = {}
        preferred_keys = _preferred_metric_keys(payload.get("metric_summary"))
        for key in preferred_keys:
            metric_summary[key] = _compact_metric_stat_block((payload.get("metric_summary") or {}).get(key))
        compacted["metric_summary"] = metric_summary
    if "reward_breakdown_summary" in payload and isinstance(payload.get("reward_breakdown_summary"), dict):
        reward_breakdown = {}
        for key in list((payload.get("reward_breakdown_summary") or {}).keys())[: max(4, min(max_points, 8))]:
            reward_breakdown[key] = _compact_metric_stat_block((payload.get("reward_breakdown_summary") or {}).get(key))
        compacted["reward_breakdown_summary"] = reward_breakdown
    if "metric_curves" in payload and isinstance(payload.get("metric_curves"), dict):
        metric_curves = {}
        for key in _preferred_metric_keys(payload.get("metric_curves")):
            metric_curves[key] = _compact_metric_curve((payload.get("metric_curves") or {}).get(key), max_points=max_points)
        compacted["metric_curves"] = metric_curves
    for passthrough_key in ("run_dir", "collector_status", "status"):
        if passthrough_key in payload:
            compacted[passthrough_key] = payload.get(passthrough_key)
    return compacted


def compact_lbf_behavior_summary_for_llm(
    behavior_summary: Dict[str, Any] | None,
) -> Dict[str, Any]:
    summary = deepcopy(behavior_summary or {})
    if not isinstance(summary, dict):
        return {}
    episode_summaries = list(summary.get("episode_summaries") or [])
    return {
        "intervention_point": str(summary.get("intervention_point") or ""),
        "env_key": str(summary.get("env_key") or ""),
        "eval_episodes": int(summary.get("eval_episodes") or 0),
        "insufficient_evidence": bool(summary.get("insufficient_evidence", False)),
        "insufficient_evidence_reason": str(summary.get("insufficient_evidence_reason") or ""),
        "collector_status": str(summary.get("collector_status") or ""),
        "milestone": _compact_milestone(summary.get("milestone")),
        "coverage": _compact_metric_block(
            summary.get("coverage"),
            metric_keys=("joint_visited_cell_ratio", "per_agent_visited_cell_ratio_mean", "idle_action_ratio"),
        ),
        "food_discovery": _compact_metric_block(
            summary.get("food_discovery"),
            metric_keys=("episodes_never_close_to_food", "near_food_steps_mean", "first_near_food_step_mean"),
        ),
        "failed_collect": _compact_metric_block(
            summary.get("failed_collect"),
            metric_keys=("successful_collection_events", "failed_collect_actions", "joint_near_food_no_collection_steps"),
        ),
        "target_concentration": _compact_metric_block(
            summary.get("target_concentration"),
            metric_keys=("same_target_ratio", "target_entropy_mean", "target_switch_count_mean"),
        ),
        "stability": _compact_metric_block(
            summary.get("stability"),
            metric_keys=("backtracking_ratio_mean", "repeated_position_ratio", "episode_length_mean", "return_mean"),
        ),
        "evidence_flags": _compact_evidence_flags(summary.get("evidence_flags")),
        "behavior_signals": _trim_string_list(summary.get("behavior_signals"), limit=8),
        "episode_summaries": [
            _compact_lbf_episode_summary(item) for item in episode_summaries[:3]
        ],
        "metadata": _compact_behavior_metadata(summary.get("metadata")),
        "compact_evidence_keys": ["col", "app", "cov", "ready", "alloc", "stab"],
    }


def compact_rware_behavior_summary_for_llm(
    behavior_summary: Dict[str, Any] | None,
) -> Dict[str, Any]:
    summary = deepcopy(behavior_summary or {})
    if not isinstance(summary, dict):
        return {}
    episode_summaries = list(summary.get("episode_summaries") or [])
    return {
        "intervention_point": str(summary.get("intervention_point") or ""),
        "env_key": str(summary.get("env_key") or ""),
        "eval_episodes": int(summary.get("eval_episodes") or 0),
        "insufficient_evidence": bool(summary.get("insufficient_evidence", False)),
        "insufficient_evidence_reason": str(summary.get("insufficient_evidence_reason") or ""),
        "collector_status": str(summary.get("collector_status") or ""),
        "milestone": _compact_milestone(summary.get("milestone")),
        "coverage": _compact_metric_block(
            summary.get("coverage"),
            metric_keys=("joint_visited_cell_ratio", "per_agent_visited_cell_ratio_mean", "idle_action_ratio"),
        ),
        "food_discovery": _compact_metric_block(
            summary.get("food_discovery"),
            metric_keys=("episodes_never_close_to_food", "first_near_food_step_mean", "near_requested_shelf_steps_mean", "pickup_success_events_mean"),
        ),
        "failed_collect": _compact_metric_block(
            summary.get("failed_collect"),
            metric_keys=("successful_collection_events", "failed_collect_actions", "blocked_move_steps", "delivery_success_events"),
        ),
        "target_concentration": _compact_metric_block(
            summary.get("target_concentration"),
            metric_keys=("same_target_ratio", "target_entropy_mean", "target_switch_count_mean", "congestion_proxy_mean"),
        ),
        "stability": _compact_metric_block(
            summary.get("stability"),
            metric_keys=("backtracking_ratio_mean", "repeated_position_ratio", "route_oscillation_score_mean", "return_mean"),
        ),
        "evidence_flags": _compact_evidence_flags(summary.get("evidence_flags")),
        "behavior_signals": _trim_string_list(summary.get("behavior_signals"), limit=8),
        "episode_summaries": [
            _compact_rware_episode_summary(item) for item in episode_summaries[:3]
        ],
        "metadata": _compact_behavior_metadata(summary.get("metadata")),
        "compact_evidence_keys": ["shelf", "pickup", "goal", "deliv", "traffic", "stab"],
    }


def enforce_payload_budget(payload: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    compacted = deepcopy(payload or {})
    if estimate_payload_size_chars(compacted) <= int(max_chars):
        return compacted
    pruning_steps = [
        _drop_full_episode_summaries,
        _drop_branch_evidence,
        _drop_behavior_signals,
        _drop_metric_curve_points,
        _drop_env_args,
        _drop_path_like_fields,
        _trim_winner_history,
    ]
    for step in pruning_steps:
        compacted = step(compacted)
        if estimate_payload_size_chars(compacted) <= int(max_chars):
            break
    return compacted


def _tier_config(*, tier: str) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_LLM_INPUT_COMPACTION)
    normalized_tier = str(tier or "validation").strip().lower()
    cfg["max_payload_chars"] = int(
        cfg["max_payload_chars_formal"]
        if normalized_tier == "formal"
        else cfg["max_payload_chars_validation"]
    )
    cfg["tier"] = normalized_tier
    return cfg


def _extract_env_key(payload: Dict[str, Any]) -> str:
    candidates = [
        payload.get("env_key"),
        (payload.get("task_metadata") or {}).get("env_key"),
        ((payload.get("run_metadata") or {}).get("env_args") or {}).get("key"),
        ((payload.get("sparse_reference_run_summary") or {}).get("run_metadata") or {}).get("env_args", {}).get("key")
        if isinstance((payload.get("sparse_reference_run_summary") or {}).get("run_metadata"), dict)
        else None,
        ((payload.get("dense_reference_run_summary") or {}).get("run_metadata") or {}).get("env_args", {}).get("key")
        if isinstance((payload.get("dense_reference_run_summary") or {}).get("run_metadata"), dict)
        else None,
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_pbrs_version(payload: Dict[str, Any]) -> str:
    candidates = [
        payload.get("pbrs_version"),
        (payload.get("current_config") or {}).get("pbrs_version"),
        (payload.get("current_mainline_config") or {}).get("pbrs_version"),
        (payload.get("initial_dense_config") or {}).get("pbrs_version"),
    ]
    for key in ("candidate_results", "all_candidate_results", "round1_results", "round2_results", "branch_results"):
        for item in list(payload.get(key) or []):
            if isinstance(item, dict) and str(item.get("pbrs_version") or "").strip():
                candidates.append(item.get("pbrs_version"))
                break
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _compact_stage1_like_payload(
    payload: Dict[str, Any],
    *,
    env_family: str,
    cfg: Dict[str, Any],
    included_sections: List[str],
    omitted_sections: List[str],
) -> None:
    if "sparse_run_summary" in payload:
        payload["sparse_run_summary"] = compact_metrics_history_for_llm(
            payload.get("sparse_run_summary"),
            max_points=int(cfg["max_metric_points"]),
        )
        included_sections.append("sparse_run_summary")
        omitted_sections.append("sparse_run_summary.run_metadata.env_args")
    if "sparse_baseline_metrics" in payload:
        payload["sparse_baseline_metrics"] = _compact_sparse_baseline_metrics(
            payload.get("sparse_baseline_metrics")
        )
        included_sections.append("sparse_baseline_metrics")
    if "dense_reference_run_summary" in payload:
        payload["dense_reference_run_summary"] = compact_metrics_history_for_llm(
            payload.get("dense_reference_run_summary"),
            max_points=int(cfg["max_metric_points"]),
        )
        included_sections.append("dense_reference_run_summary")
        omitted_sections.append("dense_reference_run_summary.run_metadata.env_args")
    if "sparse_reference_run_summary" in payload:
        payload["sparse_reference_run_summary"] = compact_metrics_history_for_llm(
            payload.get("sparse_reference_run_summary"),
            max_points=int(cfg["max_metric_points"]),
        )
        included_sections.append("sparse_reference_run_summary")
        omitted_sections.append("sparse_reference_run_summary.run_metadata.env_args")
    if "stage1_behavior_summary" in payload and isinstance(payload.get("stage1_behavior_summary"), dict):
        payload["stage1_behavior_summary"] = compact_behavior_summary_for_llm(
            payload.get("stage1_behavior_summary"),
            env_family=env_family,
        )
        included_sections.append("stage1_behavior_summary")
        omitted_sections.append("stage1_behavior_summary.full_episode_summaries")
    if "behavior_summary" in payload and isinstance(payload.get("behavior_summary"), dict):
        payload["behavior_summary"] = compact_behavior_summary_for_llm(
            payload.get("behavior_summary"),
            env_family=env_family,
        )
        included_sections.append("behavior_summary")
    if "integrated_guidance_card" in payload and isinstance(payload.get("integrated_guidance_card"), dict):
        payload["integrated_guidance_card"] = _compact_integrated_guidance_card_for_llm(
            payload.get("integrated_guidance_card"),
            include_evidence_keys=bool(cfg["include_evidence_keys"]),
        )
        included_sections.append("integrated_guidance_card")
        omitted_sections.append("integrated_guidance_card.long_prose")
    for key in ("round1_results", "round2_results", "all_candidate_results"):
        if key in payload and isinstance(payload.get(key), list):
            payload[key] = [
                _compact_stage1_candidate_result(item, env_family=env_family)
                for item in list(payload.get(key) or [])
            ]
            included_sections.append(key)
            omitted_sections.extend(
                [
                    f"{key}.run_dir",
                    f"{key}.endpoint_checkpoint_path",
                    f"{key}.reward_breakdown_summary",
                ]
            )
    if "round1_critic_analysis" in payload:
        payload["round1_critic_analysis"] = _compact_round_analysis(payload.get("round1_critic_analysis"))
        included_sections.append("round1_critic_analysis")
    if "pbrs_definition" in payload and isinstance(payload.get("pbrs_definition"), dict):
        payload["pbrs_definition"] = _compact_pbrs_definition(payload.get("pbrs_definition"))
        included_sections.append("pbrs_definition")
    if payload.get("policy_guidance_used", False):
        payload["policy_guidance_evidence_keys"] = _trim_string_list(
            payload.get("policy_guidance_evidence_keys"), limit=int(cfg["max_behavior_items"])
        )
    else:
        payload["policy_guidance_evidence_keys"] = []


def _build_metadata(
    *,
    compacted: Dict[str, Any],
    env_family: str,
    pre_chars: int,
    cfg: Dict[str, Any],
    included_sections: Iterable[str],
    omitted_sections: Iterable[str],
    local_refs: Iterable[str],
) -> Dict[str, Any]:
    post_chars = estimate_payload_size_chars(compacted)
    reduction_ratio = 0.0
    if pre_chars > 0:
        reduction_ratio = max(0.0, min(1.0, float(pre_chars - post_chars) / float(pre_chars)))
    return {
        "llm_input_compaction_enabled": True,
        "compaction_version": COMPACTION_VERSION,
        "env_family": str(env_family),
        "pre_compaction_chars": int(pre_chars),
        "post_compaction_chars": int(post_chars),
        "reduction_ratio": round(reduction_ratio, 6),
        "max_payload_chars": int(cfg["max_payload_chars"]),
        "included_sections": list(dict.fromkeys(str(item) for item in included_sections if str(item))),
        "omitted_sections": list(dict.fromkeys(str(item) for item in omitted_sections if str(item))),
        "evidence_keys_included": _collect_evidence_keys(compacted),
        "local_artifact_references": list(dict.fromkeys(str(item) for item in local_refs if str(item))),
        "raw_trajectory_included": False,
        "full_metrics_history_included": False,
        "full_behavior_summary_included": False,
    }


def _preferred_metric_keys(payload: Dict[str, Any] | None) -> List[str]:
    if not isinstance(payload, dict):
        return []
    preferred = [
        "test_sparse_return_mean",
        "sparse_return_mean",
        "test_return_mean",
        "return_mean",
        "dense_return_mean",
        "test_dense_return_mean",
        "loss",
        "grad_norm",
        "td_error_abs",
    ]
    keys = [key for key in preferred if key in payload]
    for key in list(payload.keys()):
        if key not in keys:
            keys.append(key)
    return keys[:8]


def _compact_metric_stat_block(block: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(block or {})
    if not isinstance(payload, dict):
        return {}
    kept = {}
    for key in (
        "best",
        "best_step",
        "final",
        "final_step",
        "mean",
        "median",
        "min",
        "max",
        "num_points",
        "last_k_mean",
        "last_k_std",
        "auc",
    ):
        if key in payload:
            kept[key] = payload.get(key)
    return kept


def _compact_metric_curve(curve: Dict[str, Any] | None, *, max_points: int) -> Dict[str, Any]:
    payload = deepcopy(curve or {})
    if not isinstance(payload, dict):
        return {}
    sampled_points = list(payload.get("sampled_points") or [])
    if len(sampled_points) > int(max_points):
        if int(max_points) <= 2:
            sampled_points = sampled_points[: int(max_points)]
        else:
            head = sampled_points[: int(max_points) // 2]
            tail = sampled_points[-(int(max_points) - len(head)) :]
            sampled_points = head + tail
    return {
        "num_points": int(payload.get("num_points") or len(sampled_points)),
        "sampled_points": sampled_points,
    }


def _compact_run_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(metadata or {})
    if not isinstance(payload, dict):
        return {}
    env_args = payload.get("env_args")
    compact_env_args = {}
    if isinstance(env_args, dict):
        for key in (
            "key",
            "time_limit",
            "n_agents",
            "pbrs_beta",
            "pbrs_wc",
            "pbrs_wp",
            "pbrs_variant",
            "pbrs_version",
        ):
            if key in env_args:
                compact_env_args[key] = env_args.get(key)
    compacted = {}
    for key in (
        "workflow_id",
        "workflow_round",
        "reward_paradigm",
        "seed",
        "active_pbrs_field",
        "active_pbrs_field_source",
        "candidate_value",
        "algorithm",
        "env_config",
        "checkpoint_root_dir",
        "available_checkpoint_steps",
        "latest_checkpoint_step",
        "run_id",
        "run_dir",
        "load_step",
    ):
        if key in payload:
            compacted[key] = payload.get(key)
    if compact_env_args:
        compacted["env_args"] = compact_env_args
    return compacted


def _compact_metric_block(block: Dict[str, Any] | None, *, metric_keys: Tuple[str, ...]) -> Dict[str, Any]:
    payload = deepcopy(block or {})
    if not isinstance(payload, dict):
        return {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    compact_metrics = {key: metrics.get(key) for key in metric_keys if key in metrics}
    return {
        "score": payload.get("score"),
        "evidence": _trim_string_list(payload.get("evidence"), limit=4),
        "metrics": compact_metrics,
    }


def _compact_evidence_flags(flags: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(flags or {})
    if not isinstance(payload, dict):
        return {}
    return {str(key): bool(value) for key, value in payload.items()}


def _compact_behavior_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(metadata or {})
    if not isinstance(payload, dict):
        return {}
    kept = {}
    for key in (
        "env_type",
        "grid_area",
        "field_area",
        "feature_source",
        "warning_count",
        "collection_event_observable",
        "collector_mode",
        "extractor_preference",
        "source_run_ref",
        "branch_id",
    ):
        if key in payload:
            kept[key] = payload.get(key)
    return kept


def _compact_milestone(milestone: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(milestone or {})
    if not isinstance(payload, dict):
        return {}
    return {
        "stage_label": str(payload.get("stage_label") or ""),
        "target_step": int(payload.get("target_step") or 0),
        "actual_step": int(payload.get("actual_step") or 0),
        "num_eval_episodes": int(payload.get("num_eval_episodes") or 0),
    }


def _compact_lbf_episode_summary(item: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(item or {})
    if not isinstance(payload, dict):
        return {}
    return {
        "episode_id": int(payload.get("episode_id") or 0),
        "return": payload.get("return"),
        "episode_length": payload.get("episode_length"),
        "near_food_steps": payload.get("near_food_steps"),
        "successful_collection_events": payload.get("successful_collection_events"),
        "failed_collect_actions": payload.get("failed_collect_actions"),
        "same_target_ratio": payload.get("same_target_ratio"),
        "repeated_position_ratio": payload.get("repeated_position_ratio"),
        "summary_text": _compact_text(payload.get("summary_text"), max_chars=160),
    }


def _compact_rware_episode_summary(item: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(item or {})
    if not isinstance(payload, dict):
        return {}
    return {
        "episode_id": int(payload.get("episode_id") or 0),
        "return": payload.get("return"),
        "episode_length": payload.get("episode_length"),
        "near_requested_shelf_steps": payload.get("near_requested_shelf_steps"),
        "pickup_success_events": payload.get("pickup_success_events"),
        "carrying_steps": payload.get("carrying_steps"),
        "near_goal_while_carrying_steps": payload.get("near_goal_while_carrying_steps"),
        "delivery_success_events": payload.get("delivery_success_events"),
        "blocked_move_steps": payload.get("blocked_move_steps"),
        "congestion_proxy": payload.get("congestion_proxy"),
        "route_oscillation_score": payload.get("route_oscillation_score"),
        "summary_text": _compact_text(payload.get("summary_text"), max_chars=160),
    }


def _compact_generic_behavior_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "intervention_point": str(summary.get("intervention_point") or ""),
        "env_key": str(summary.get("env_key") or ""),
        "eval_episodes": int(summary.get("eval_episodes") or 0),
        "insufficient_evidence": bool(summary.get("insufficient_evidence", False)),
        "insufficient_evidence_reason": str(summary.get("insufficient_evidence_reason") or ""),
        "collector_status": str(summary.get("collector_status") or ""),
        "milestone": _compact_milestone(summary.get("milestone")),
        "evidence_flags": _compact_evidence_flags(summary.get("evidence_flags")),
        "behavior_signals": _trim_string_list(summary.get("behavior_signals"), limit=6),
        "metadata": _compact_behavior_metadata(summary.get("metadata")),
    }


def _compact_integrated_guidance_card_for_llm(
    card: Dict[str, Any] | None,
    *,
    include_evidence_keys: bool,
) -> Dict[str, Any]:
    payload = deepcopy(card or {})
    if not isinstance(payload, dict):
        return {}
    policy_diag = deepcopy(payload.get("policy_diagnosis") or {})
    candidate_guidance = deepcopy(payload.get("candidate_generation_guidance") or {})
    branch_evidence = list(payload.get("branch_evidence") or [])
    round2 = deepcopy(payload.get("round2_search_guidance") or {})
    compacted = {
        "intervention_point": str(payload.get("intervention_point") or ""),
        "checkpoint_id": str(payload.get("checkpoint_id") or ""),
        "checkpoint_step": int(payload.get("checkpoint_step") or 0),
        "round_id": int(payload.get("round_id") or 0),
        "fallback_to_reward_only": bool(payload.get("fallback_to_reward_only", False)),
        "use_in_reward_generation": bool(payload.get("use_in_reward_generation", False)),
        "policy_diagnosis": {
            "policy_failure_mode": str(policy_diag.get("policy_failure_mode") or ""),
            "confidence": str(policy_diag.get("confidence") or ""),
            "insufficient_evidence": bool(policy_diag.get("insufficient_evidence", False)),
            "failure_reason": str(policy_diag.get("failure_reason") or ""),
            "behavior_evidence": _trim_string_list(
                policy_diag.get("behavior_evidence"),
                limit=6 if include_evidence_keys else 0,
            ),
        },
        "candidate_generation_guidance": {
            "candidate_types": _trim_string_list(candidate_guidance.get("candidate_types"), limit=4),
            "beta_direction": str(candidate_guidance.get("beta_direction") or ""),
            "wc_direction": str(candidate_guidance.get("wc_direction") or ""),
            "wp_direction": str(candidate_guidance.get("wp_direction") or ""),
            "constraints": _trim_string_list(candidate_guidance.get("constraints"), limit=6),
            "reward_search_implications": _trim_string_list(
                candidate_guidance.get("reward_search_implications"), limit=4
            ),
        },
        "round2_search_guidance": {
            "around_candidate": str(round2.get("around_candidate") or ""),
            "avoid_candidate_types": _trim_string_list(round2.get("avoid_candidate_types"), limit=4),
            "stability_warning": _compact_text(round2.get("stability_warning"), max_chars=160),
            "prefer_stable_last_window": bool(round2.get("prefer_stable_last_window", False)),
        },
        "metadata": _compact_behavior_metadata(payload.get("metadata")),
    }
    evidence_keys = _trim_string_list(policy_diag.get("behavior_evidence"), limit=6) if include_evidence_keys else []
    compacted["policy_diagnosis_summary"] = _compact_text(
        f"{policy_diag.get('policy_failure_mode') or 'unknown'}; confidence={policy_diag.get('confidence') or 'low'}; "
        f"constraints={','.join(_trim_string_list(candidate_guidance.get('constraints'), limit=3))}",
        max_chars=180,
    )
    compacted["top_failure_modes"] = _trim_string_list(
        [
            policy_diag.get("policy_failure_mode"),
            policy_diag.get("failure_reason"),
        ],
        limit=3,
    )
    compacted["evidence_keys"] = evidence_keys
    if branch_evidence:
        compacted["branch_evidence"] = [
            {
                "candidate_id": str(item.get("candidate_id") or ""),
                "metric_evidence": _trim_string_list(item.get("metric_evidence"), limit=3),
                "behavior_evidence": _trim_string_list(item.get("behavior_evidence"), limit=2 if include_evidence_keys else 0),
            }
            for item in branch_evidence[:4]
        ]
    else:
        compacted["branch_evidence"] = []
    return compacted


def _compact_policy_guidance_payload(
    payload: Dict[str, Any] | None,
    *,
    env_family: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    compacted = deepcopy(payload or {})
    if not isinstance(compacted, dict):
        return {}
    if compacted.get("policy_guidance_used", False):
        if "integrated_guidance_card" in compacted and isinstance(compacted.get("integrated_guidance_card"), dict):
            compacted["integrated_guidance_card"] = _compact_integrated_guidance_card_for_llm(
                compacted.get("integrated_guidance_card"),
                include_evidence_keys=bool(cfg["include_evidence_keys"]),
            )
        if "behavior_summary" in compacted and isinstance(compacted.get("behavior_summary"), dict):
            compacted["behavior_summary"] = compact_behavior_summary_for_llm(
                compacted.get("behavior_summary"),
                env_family=env_family,
            )
        compacted["policy_guidance_evidence_keys"] = _trim_string_list(
            compacted.get("policy_guidance_evidence_keys"), limit=int(cfg["max_behavior_items"])
        )
    else:
        compacted["policy_guidance_evidence_keys"] = []
        compacted.pop("behavior_summary", None)
    return compacted


def _compact_stage1_candidate_result(item: Dict[str, Any] | None, *, env_family: str) -> Dict[str, Any]:
    payload = deepcopy(item or {})
    if not isinstance(payload, dict):
        return {}
    compacted = {
        "candidate_id": str(payload.get("candidate_id") or ""),
        "candidate_type": str(payload.get("candidate_type") or ""),
        "pbrs_version": str(payload.get("pbrs_version") or _default_pbrs_version_for_family(env_family)),
        "mode": str(payload.get("mode") or ""),
        "beta": payload.get("beta"),
        "active_terms": list(payload.get("active_terms") or []),
        "weights": deepcopy(payload.get("weights") or {}),
        "evidence_keys_used": _trim_string_list(payload.get("evidence_keys_used"), limit=6),
        "endpoint_step": payload.get("endpoint_checkpoint_step") or payload.get("endpoint_step"),
        "endpoint_distance": payload.get("endpoint_distance_from_target") or payload.get("endpoint_distance"),
        "checkpoint_ok": bool(payload.get("endpoint_checkpoint_path") or payload.get("checkpoint_ok", False)),
        "final_test_sparse_return_mean": payload.get("final_test_sparse_return_mean"),
        "best_test_sparse_return_mean": payload.get("best_test_sparse_return_mean"),
        "last_k_mean": payload.get("last_k_mean"),
        "auc": payload.get("auc"),
        "trend": _derive_trend(payload),
        "run_status": payload.get("run_status"),
    }
    issues = _trim_string_list(
        (payload.get("rejection_reasons") or [])
        + ([payload.get("invalid_reason")] if payload.get("invalid_reason") else []),
        limit=4,
    )
    if issues:
        compacted["rejection_reasons"] = issues
    if env_family == "rware":
        compacted["active_terms"] = [term for term in compacted["active_terms"] if term in {"shelf", "pickup", "goal", "deliv", "traffic", "stab"}]
    return compacted


def _compact_stage3_candidate_result(item: Dict[str, Any] | None, *, env_family: str) -> Dict[str, Any]:
    payload = deepcopy(item or {})
    if not isinstance(payload, dict):
        return {}
    compacted = {
        "candidate_id": str(payload.get("candidate_id") or ""),
        "candidate_source": str(payload.get("candidate_source") or payload.get("source_config") or ""),
        "candidate_type": str(payload.get("candidate_type") or ""),
        "config_diff": _compact_candidate_config_diff(payload),
        "pbrs_version": str(payload.get("pbrs_version") or _default_pbrs_version_for_family(env_family)),
        "mode": str(payload.get("mode") or ""),
        "beta": payload.get("beta"),
        "active_terms": list(payload.get("active_terms") or []),
        "weights": deepcopy(payload.get("weights") or {}),
        "final": payload.get("final"),
        "max": payload.get("best_test_sparse_return_mean") or payload.get("max"),
        "last_k_mean": payload.get("last_k_mean"),
        "auc": payload.get("auc"),
        "stability_flags": _compact_stability_flags(payload),
        "no_change_gap": payload.get("no_change_gap"),
        "dense_reference_gap": payload.get("dense_reference_gap"),
        "fallback_flags": _compact_fallback_flags(payload),
        "evidence_keys_used": _trim_string_list(payload.get("evidence_keys_used"), limit=6),
        "selected": bool(payload.get("selected", False)),
        "winner": bool(payload.get("winner", False)),
        "run_status": payload.get("run_status") or payload.get("status"),
        "metrics_valid": payload.get("metrics_valid"),
    }
    issues = _trim_string_list(
        (payload.get("rejection_reasons") or [])
        + ([payload.get("invalid_reason")] if payload.get("invalid_reason") else []),
        limit=4,
    )
    if issues:
        compacted["rejection_reasons"] = issues
    if env_family == "rware":
        compacted["active_terms"] = [term for term in compacted["active_terms"] if term in {"shelf", "pickup", "goal", "deliv", "traffic", "stab"}]
    return compacted


def _compact_candidate_config_diff(payload: Dict[str, Any]) -> Dict[str, Any]:
    diff = {}
    for key in ("beta_delta", "wc_delta", "wp_delta", "weights_l1_delta", "max_core_weight_delta"):
        if key in payload:
            diff[key] = payload.get(key)
    if "config_diff" in payload and isinstance(payload.get("config_diff"), dict):
        for key in ("beta_delta", "wc_delta", "wp_delta", "weights_l1_delta", "active_terms_changed", "mode_changed"):
            if key in payload["config_diff"]:
                diff[key] = payload["config_diff"].get(key)
    return diff


def _compact_stability_flags(payload: Dict[str, Any]) -> Dict[str, Any]:
    flags = {}
    for key in ("stability_concern", "metrics_valid", "passes_stability_gate", "spike_gap", "late_window_slope"):
        if key in payload:
            flags[key] = payload.get(key)
    return flags


def _compact_fallback_flags(payload: Dict[str, Any]) -> Dict[str, Any]:
    flags = {}
    for key in ("deterministic_fallback", "fallback_style_candidate", "effective_update_candidate"):
        if key in payload:
            flags[key] = payload.get(key)
    return flags


def _compact_round_result_summary(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    data = deepcopy(payload or {})
    if not isinstance(data, dict):
        return {}
    return {
        "round_id": data.get("round_id"),
        "best_candidate_id": data.get("best_candidate_id"),
        "best_non_no_change_candidate_id": data.get("best_non_no_change_candidate_id"),
        "no_change_candidate_id": data.get("no_change_candidate_id"),
        "ranking": [
            {
                "candidate_id": item.get("candidate_id"),
                "score": item.get("score"),
                "run_id": item.get("run_id"),
            }
            for item in list(data.get("ranking") or [])[:4]
        ],
    }


def _compact_round_analysis(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    data = deepcopy(payload or {})
    if not isinstance(data, dict):
        return {}
    return {
        "goal": str(data.get("goal") or ""),
        "supported_hypotheses": _trim_string_list(data.get("supported_hypotheses"), limit=4),
        "rejected_or_uncertain_hypotheses": _trim_string_list(data.get("rejected_or_uncertain_hypotheses"), limit=4),
        "risky_patterns": _trim_string_list(data.get("risky_patterns"), limit=4),
        "generator_revision_advice": deepcopy(data.get("generator_revision_advice") or {}),
        "candidate_reviews": [
            {
                "candidate_id": item.get("candidate_id"),
                "run_status": item.get("run_status"),
                "supports_hypothesis": item.get("supports_hypothesis"),
                "risk_note": _compact_text(item.get("risk_note"), max_chars=120),
            }
            for item in list(data.get("candidate_reviews") or [])[:4]
        ],
    }


def _compact_pbrs_definition(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    data = deepcopy(payload or {})
    if not isinstance(data, dict):
        return {}
    compacted = {}
    for key in ("reward_paradigm", "pbrs_version", "supported_terms", "supported_modes", "active_terms"):
        if key in data:
            compacted[key] = deepcopy(data.get(key))
    return compacted


def _compact_reference_comparison(payload: Dict[str, Any] | None, *, env_family: str) -> Dict[str, Any]:
    data = deepcopy(payload or {})
    if not isinstance(data, dict):
        return {}
    compacted = {
        "underperforming_vs_fixed_reference": bool(data.get("underperforming_vs_fixed_reference", False)),
        "fixed_reference_config": _compact_stage3_candidate_result(
            data.get("fixed_reference_config"),
            env_family=env_family,
        )
        if isinstance(data.get("fixed_reference_config"), dict)
        else deepcopy(data.get("fixed_reference_config") or {}),
        "no_change_gap": data.get("no_change_gap"),
        "dense_reference_gap": data.get("dense_reference_gap"),
    }
    return compacted


def _compact_generic_context(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    data = deepcopy(payload or {})
    if not isinstance(data, dict):
        return {}
    compacted = {}
    for key, value in data.items():
        if any(token in key.lower() for token in ("path", "run_dir", "config_json", "metrics_json", "info_json")):
            continue
        if isinstance(value, dict):
            compacted[key] = _compact_generic_context(value)
        elif isinstance(value, list):
            compacted[key] = deepcopy(value[:4])
        else:
            compacted[key] = value
    return compacted


def _compact_sparse_baseline_metrics(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    data = deepcopy(payload or {})
    if not isinstance(data, dict):
        return {}
    keep = {}
    for key in (
        "best_test_sparse_return_mean",
        "final_test_sparse_return_mean",
        "train_return_mean",
        "sample_efficiency_status",
        "performance_issue",
        "run_dir",
    ):
        if key in data:
            keep[key] = data.get(key)
    return keep


def _derive_trend(payload: Dict[str, Any]) -> str:
    slope = payload.get("late_window_slope")
    if slope is None:
        return ""
    try:
        slope_value = float(slope)
    except (TypeError, ValueError):
        return ""
    if slope_value > 0.0:
        return "up"
    if slope_value < 0.0:
        return "down"
    return "flat"


def _collect_evidence_keys(payload: Any) -> List[str]:
    keys: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "policy_guidance_evidence_keys":
                    keys.extend(_trim_string_list(value, limit=12))
                if key == "evidence_keys":
                    keys.extend(_trim_string_list(value, limit=12))
                if key == "behavior_evidence":
                    keys.extend(_trim_string_list(value, limit=12))
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return list(dict.fromkeys(item for item in keys if item))


def _collect_local_artifact_references(payload: Any) -> List[str]:
    refs: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_lower = str(key).lower()
                if isinstance(value, str) and any(
                    token in key_lower
                    for token in ("path", "run_dir", "artifact", "cache_path", "manifest", "summary_path")
                ):
                    refs.append(_shorten_path(value))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return refs[:16]


def _shorten_path(path_value: str) -> str:
    text = str(path_value or "")
    if not text:
        return ""
    try:
        path = Path(text)
        parts = path.parts[-3:]
        return str(Path(*parts))
    except Exception:
        return _compact_text(text, max_chars=96)


def _trim_string_list(values: Any, *, limit: int) -> List[str]:
    if limit <= 0:
        return []
    result: List[str] = []
    if isinstance(values, str):
        values = [values]
    for item in list(values or []):
        text = str(item or "").strip()
        if text:
            result.append(_compact_text(text, max_chars=180))
        if len(result) >= int(limit):
            break
    return result


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= int(max_chars):
        return text
    return text[: max(0, int(max_chars) - 3)] + "..."


def _drop_full_episode_summaries(payload: Dict[str, Any]) -> Dict[str, Any]:
    compacted = deepcopy(payload)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "episode_summaries" in node and isinstance(node.get("episode_summaries"), list):
                node["episode_summaries"] = list(node.get("episode_summaries") or [])[:2]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(compacted)
    return compacted


def _drop_branch_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
    compacted = deepcopy(payload)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "branch_evidence" in node:
                node["branch_evidence"] = list(node.get("branch_evidence") or [])[:2]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(compacted)
    return compacted


def _drop_behavior_signals(payload: Dict[str, Any]) -> Dict[str, Any]:
    compacted = deepcopy(payload)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "behavior_signals" in node:
                node["behavior_signals"] = list(node.get("behavior_signals") or [])[:4]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(compacted)
    return compacted


def _drop_metric_curve_points(payload: Dict[str, Any]) -> Dict[str, Any]:
    compacted = deepcopy(payload)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "metric_curves" in node and isinstance(node.get("metric_curves"), dict):
                for curve in node["metric_curves"].values():
                    if isinstance(curve, dict) and "sampled_points" in curve:
                        curve["sampled_points"] = list(curve.get("sampled_points") or [])[:4]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(compacted)
    return compacted


def _drop_env_args(payload: Dict[str, Any]) -> Dict[str, Any]:
    compacted = deepcopy(payload)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "run_metadata" in node and isinstance(node.get("run_metadata"), dict):
                run_metadata = node["run_metadata"]
                env_args = run_metadata.get("env_args")
                if isinstance(env_args, dict):
                    run_metadata["env_args"] = {
                        key: env_args.get(key)
                        for key in ("key", "time_limit", "pbrs_beta", "pbrs_wc", "pbrs_wp", "pbrs_variant", "pbrs_version")
                        if key in env_args
                    }
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(compacted)
    return compacted


def _drop_path_like_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    compacted = deepcopy(payload)
    removable_tokens = ("path", "run_dir", "artifact", "cache_path")

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            remove_keys = [
                key
                for key in list(node.keys())
                if any(token in str(key).lower() for token in removable_tokens)
                and str(key) != "input_behavior_summary_path"
                and str(key) != "policy_guidance_behavior_summary_path"
            ]
            for key in remove_keys:
                node.pop(key, None)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(compacted)
    return compacted


def _trim_winner_history(payload: Dict[str, Any]) -> Dict[str, Any]:
    compacted = deepcopy(payload)
    if isinstance(compacted.get("winner_history"), list):
        compacted["winner_history"] = list(compacted.get("winner_history") or [])[-2:]
    return compacted


def _default_pbrs_version_for_family(env_family: str) -> str:
    if env_family == "rware":
        return "rware_pbrs_v2"
    if env_family == "lbf":
        return "lbf_pbrs_v2"
    return ""
