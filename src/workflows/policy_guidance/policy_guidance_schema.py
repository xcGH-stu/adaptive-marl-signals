from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List

from rewarding.lbf_pbrs_v2 import TERM_NAMES as LBF_PBRS_V2_TERMS
from rewarding.lbf_pbrs_v2 import normalize_lbf_pbrs_v2_config


CONFIDENCE_LEVELS = {"low", "medium", "high"}
DIRECTION_VALUES = {"up", "down", "same", "unknown"}
LBF_BEHAVIOR_BLOCKS = (
    "coverage",
    "food_discovery",
    "failed_collect",
    "target_concentration",
    "stability",
)
RWARE_BEHAVIOR_BLOCKS = (
    "shelf_acquisition",
    "pickup_progress",
    "carrying_to_goal",
    "delivery_success",
    "traffic_blocking",
    "route_stability",
)


def _policy_guidance_env_family(env_key: Any) -> str:
    env_text = str(env_key or "").strip().lower()
    if "rware" in env_text:
        return "rware"
    if "lbforaging" in env_text or env_text.startswith("lbf"):
        return "lbf"
    return "unknown"


def build_insufficient_behavior_summary(
    *,
    intervention_point: str,
    env_key: str,
    eval_episodes: int,
    reason: str,
) -> Dict[str, Any]:
    env_family = _policy_guidance_env_family(env_key)
    summary = {
        "intervention_point": str(intervention_point),
        "env_key": str(env_key),
        "env_family": env_family,
        "eval_episodes": int(eval_episodes),
        "insufficient_evidence": True,
        "insufficient_evidence_reason": str(reason),
        "collector_status": "insufficient_behavior_evidence",
        "episode_summaries": [],
        "milestone": {
            "stage_label": str(intervention_point),
            "target_step": 0,
            "actual_step": 0,
            "num_eval_episodes": int(eval_episodes),
        },
        "behavior_signals": [],
        "metadata": {},
    }
    if env_family == "rware":
        for field in RWARE_BEHAVIOR_BLOCKS:
            summary[field] = {"score": None, "evidence": []}
        summary["evidence_flags"] = {
            "requested_shelf_access_low": False,
            "pickup_progress_weak": False,
            "carrying_progress_weak": False,
            "delivery_progress_weak": False,
            "traffic_blocking_high": False,
            "route_stability_low": False,
            "insufficient_behavior_evidence": True,
        }
        return summary
    for field in LBF_BEHAVIOR_BLOCKS:
        summary[field] = {"score": None, "evidence": []}
    summary["evidence_flags"] = {
        "low_coverage": False,
        "food_discovery_failure": False,
        "near_food_collection_failure": False,
        "over_concentration_possible": False,
        "unstable_behavior": False,
        "insufficient_behavior_evidence": True,
    }
    return summary


def build_fallback_integrated_guidance_card(
    *,
    intervention_point: str,
    failure_reason: str,
    reward_diagnosis: Dict[str, Any] | None = None,
    checkpoint_id: str | None = None,
    checkpoint_step: int | None = None,
    round_id: int | None = None,
) -> Dict[str, Any]:
    return {
        "intervention_point": str(intervention_point),
        "checkpoint_id": str(checkpoint_id or ""),
        "checkpoint_step": int(checkpoint_step or 0),
        "round_id": int(round_id or 0),
        "reward_diagnosis": deepcopy(reward_diagnosis or {}),
        "policy_diagnosis": {
            "policy_failure_mode": "insufficient_evidence",
            "behavior_evidence": [],
            "confidence": "low",
            "insufficient_evidence": True,
            "failure_reason": str(failure_reason),
        },
        "candidate_generation_guidance": {
            "candidate_types": [],
            "beta_direction": "unknown",
            "wc_direction": "unknown",
            "wp_direction": "unknown",
            "constraints": ["fallback_to_reward_only"],
            "reward_search_implications": [
                "Policy guidance is unavailable for this intervention point.",
            ],
        },
        "branch_evidence": [],
        "round2_search_guidance": {
            "around_candidate": "",
            "avoid_candidate_types": [],
            "stability_warning": "",
            "prefer_stable_last_window": True,
        },
        "use_in_reward_generation": False,
        "fallback_to_reward_only": True,
        "metadata": {"failure_reason": str(failure_reason)},
    }


def sanitize_policy_behavior_summary(summary: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(summary or {})
    if not isinstance(payload, dict):
        raise ValueError("PolicyBehaviorSummary must be a dict")
    payload.setdefault("intervention_point", "")
    payload.setdefault("env_key", "")
    payload["env_family"] = str(
        payload.get("env_family") or _policy_guidance_env_family(payload.get("env_key"))
    )
    payload["eval_episodes"] = int(payload.get("eval_episodes") or 0)
    payload["insufficient_evidence"] = bool(payload.get("insufficient_evidence", False))
    payload["insufficient_evidence_reason"] = str(
        payload.get("insufficient_evidence_reason") or ""
    )
    payload["collector_status"] = str(payload.get("collector_status") or "mock")
    metric_blocks = (
        RWARE_BEHAVIOR_BLOCKS
        if payload["env_family"] == "rware"
        else LBF_BEHAVIOR_BLOCKS
    )
    for field in metric_blocks:
        payload[field] = _sanitize_metric_block(payload.get(field))
    payload["evidence_flags"] = _sanitize_evidence_flags(payload.get("evidence_flags"))
    payload["episode_summaries"] = _sanitize_episode_summaries(
        payload.get("episode_summaries")
    )
    payload["milestone"] = _sanitize_milestone(payload.get("milestone"))
    payload["behavior_signals"] = _coerce_string_list(payload.get("behavior_signals"))
    payload["metadata"] = _sanitize_metadata(payload.get("metadata"))
    return payload


def validate_policy_behavior_summary(summary: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = sanitize_policy_behavior_summary(summary)
    if not payload["intervention_point"]:
        raise ValueError("PolicyBehaviorSummary.intervention_point is required")
    if not payload["env_key"]:
        raise ValueError("PolicyBehaviorSummary.env_key is required")
    return payload


def sanitize_policy_guidance_card(card: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(card or {})
    if not isinstance(payload, dict):
        raise ValueError("PolicyGuidanceCard must be a dict")
    payload["intervention_point"] = str(payload.get("intervention_point") or "")
    payload["policy_failure_mode"] = str(
        payload.get("policy_failure_mode") or "insufficient_evidence"
    )
    payload["behavior_evidence"] = _coerce_string_list(payload.get("behavior_evidence"))
    payload["confidence"] = _sanitize_confidence(payload.get("confidence"))
    payload["insufficient_evidence"] = bool(payload.get("insufficient_evidence", False))
    payload["reward_search_implications"] = _coerce_string_list(
        payload.get("reward_search_implications")
    )
    payload["recommended_candidate_types"] = _coerce_string_list(
        payload.get("recommended_candidate_types")
    )
    payload["constraints"] = _coerce_string_list(payload.get("constraints"))
    payload["metadata"] = _sanitize_metadata(payload.get("metadata"))
    return payload


def validate_policy_guidance_card(card: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = sanitize_policy_guidance_card(card)
    if not payload["intervention_point"]:
        raise ValueError("PolicyGuidanceCard.intervention_point is required")
    return payload


def sanitize_integrated_guidance_card(card: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(card or {})
    if not isinstance(payload, dict):
        raise ValueError("IntegratedGuidanceCard must be a dict")
    payload["intervention_point"] = str(payload.get("intervention_point") or "")
    payload["checkpoint_id"] = str(payload.get("checkpoint_id") or "")
    payload["checkpoint_step"] = int(payload.get("checkpoint_step") or 0)
    payload["round_id"] = int(payload.get("round_id") or 0)
    reward_diagnosis = deepcopy(payload.get("reward_diagnosis") or {})
    reward_diagnosis["learning_stage"] = str(reward_diagnosis.get("learning_stage") or "")
    reward_diagnosis["metric_evidence"] = _coerce_string_list(
        reward_diagnosis.get("metric_evidence")
    )
    reward_diagnosis["performance_issue"] = str(
        reward_diagnosis.get("performance_issue") or ""
    )
    payload["reward_diagnosis"] = reward_diagnosis
    payload["policy_diagnosis"] = {
        "policy_failure_mode": str(
            ((payload.get("policy_diagnosis") or {}).get("policy_failure_mode"))
            or "insufficient_evidence"
        ),
        "behavior_evidence": _coerce_string_list(
            (payload.get("policy_diagnosis") or {}).get("behavior_evidence")
        ),
        "confidence": _sanitize_confidence(
            (payload.get("policy_diagnosis") or {}).get("confidence")
        ),
        "insufficient_evidence": bool(
            (payload.get("policy_diagnosis") or {}).get(
                "insufficient_evidence", False
            )
        ),
        "failure_reason": str(
            (payload.get("policy_diagnosis") or {}).get("failure_reason") or ""
        ),
    }
    guidance = deepcopy(payload.get("candidate_generation_guidance") or {})
    guidance["candidate_types"] = _coerce_string_list(guidance.get("candidate_types"))
    guidance["beta_direction"] = _sanitize_direction(guidance.get("beta_direction"))
    guidance["wc_direction"] = _sanitize_direction(guidance.get("wc_direction"))
    guidance["wp_direction"] = _sanitize_direction(guidance.get("wp_direction"))
    guidance["constraints"] = _coerce_string_list(guidance.get("constraints"))
    guidance["reward_search_implications"] = _coerce_string_list(
        guidance.get("reward_search_implications")
    )
    payload["candidate_generation_guidance"] = guidance
    branch_evidence = list(payload.get("branch_evidence") or [])
    normalized_branch_evidence: List[Dict[str, Any]] = []
    for item in branch_evidence:
        item_payload = deepcopy(item or {})
        if not isinstance(item_payload, dict):
            raise ValueError("branch_evidence items must be dicts")
        normalized_branch_evidence.append(
            {
                "candidate_id": str(item_payload.get("candidate_id") or ""),
                "metric_evidence": _coerce_string_list(item_payload.get("metric_evidence")),
                "behavior_evidence": _coerce_string_list(item_payload.get("behavior_evidence")),
            }
        )
    payload["branch_evidence"] = normalized_branch_evidence
    round2 = deepcopy(payload.get("round2_search_guidance") or {})
    payload["round2_search_guidance"] = {
        "around_candidate": str(round2.get("around_candidate") or ""),
        "avoid_candidate_types": _coerce_string_list(round2.get("avoid_candidate_types")),
        "stability_warning": str(round2.get("stability_warning") or ""),
        "prefer_stable_last_window": bool(round2.get("prefer_stable_last_window", False)),
    }
    payload["use_in_reward_generation"] = bool(
        payload.get("use_in_reward_generation", False)
    )
    payload["fallback_to_reward_only"] = bool(
        payload.get("fallback_to_reward_only", False)
    )
    payload["metadata"] = _sanitize_metadata(payload.get("metadata"))
    return payload


def validate_integrated_guidance_card(card: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = sanitize_integrated_guidance_card(card)
    if not payload["intervention_point"]:
        raise ValueError("IntegratedGuidanceCard.intervention_point is required")
    return payload


def sanitize_policy_guidance_call_record(record: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(record or {})
    if not isinstance(payload, dict):
        raise ValueError("PolicyGuidanceCallRecord must be a dict")
    payload["guidance_id"] = str(payload.get("guidance_id") or "")
    payload["intervention_point"] = str(payload.get("intervention_point") or "")
    payload["input_behavior_summary_path"] = str(
        payload.get("input_behavior_summary_path") or ""
    )
    payload["integrated_guidance_card"] = sanitize_integrated_guidance_card(
        payload.get("integrated_guidance_card")
    )
    payload["used_by_critic"] = bool(payload.get("used_by_critic", False))
    payload["used_by_generator"] = bool(payload.get("used_by_generator", False))
    payload["fallback_used"] = bool(payload.get("fallback_used", False))
    payload["failure_reason"] = str(payload.get("failure_reason") or "")
    payload["metadata"] = _sanitize_metadata(payload.get("metadata"))
    return payload


def validate_policy_guidance_call_record(record: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = sanitize_policy_guidance_call_record(record)
    if not payload["guidance_id"]:
        raise ValueError("PolicyGuidanceCallRecord.guidance_id is required")
    if not payload["intervention_point"]:
        raise ValueError("PolicyGuidanceCallRecord.intervention_point is required")
    return payload


def sanitize_policy_guided_candidate(candidate: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(candidate or {})
    if not isinstance(payload, dict):
        raise ValueError("policy-guided candidate must be a dict")
    payload["candidate_id"] = str(payload.get("candidate_id") or "")
    payload["candidate_type"] = str(payload.get("candidate_type") or "conservative")
    payload["pbrs_version"] = str(payload.get("pbrs_version") or "")
    payload["beta"] = _sanitize_unit_interval(payload.get("beta"))
    if payload["pbrs_version"] == "rware_pbrs_v2":
        payload["mode"] = str(payload.get("mode") or "balanced_delivery_progress")
        weights = deepcopy(payload.get("weights") or {})
        if not isinstance(weights, dict):
            raise ValueError("RWARE PBRS-v2 candidate weights must be a dict")
        normalized_weights = {}
        for term in ("shelf", "pickup", "goal", "deliv", "traffic", "stab"):
            normalized_weights[term] = _sanitize_unit_interval(weights.get(term, 0.0))
        weight_sum = sum(normalized_weights.values())
        if weight_sum <= 0.0:
            raise ValueError("RWARE PBRS-v2 candidate weights must sum to a positive value")
        payload["weights"] = {
            term: float(value) / float(weight_sum) for term, value in normalized_weights.items()
        }
        active_terms = payload.get("active_terms")
        if active_terms is None:
            payload["active_terms"] = [
                term for term, value in payload["weights"].items() if float(value) > 0.0
            ]
        else:
            payload["active_terms"] = _coerce_string_list(active_terms)
        payload["wc"] = None
        payload["wp"] = None
    elif payload["pbrs_version"] == "lbf_pbrs_v2":
        normalized = normalize_lbf_pbrs_v2_config(payload)
        payload["mode"] = str(normalized.get("mode") or "")
        payload["weights"] = deepcopy(normalized.get("weights") or {})
        payload["active_terms"] = _coerce_string_list(normalized.get("active_terms"))
        payload["wc"] = float(normalized.get("wc", 0.0))
        payload["wp"] = float(normalized.get("wp", 0.0))
    else:
        payload["wc"] = _sanitize_unit_interval(payload.get("wc"))
        wp = payload.get("wp")
        if wp is None:
            wp = 1.0 - payload["wc"]
        payload["wp"] = _sanitize_unit_interval(wp)
        payload["mode"] = str(payload.get("mode") or "")
        payload["weights"] = deepcopy(payload.get("weights") or {})
        payload["active_terms"] = _coerce_string_list(payload.get("active_terms"))
    payload["rationale"] = str(payload.get("rationale") or "")
    payload["evidence_key_used"] = str(payload.get("evidence_key_used") or "")
    return payload


def validate_policy_guided_candidate(candidate: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = sanitize_policy_guided_candidate(candidate)
    if not payload["candidate_id"]:
        raise ValueError("candidate_id is required")
    if payload.get("pbrs_version") == "rware_pbrs_v2":
        if not payload.get("mode"):
            raise ValueError("RWARE PBRS-v2 candidate mode is required")
        if not isinstance(payload.get("weights"), dict):
            raise ValueError("RWARE PBRS-v2 candidate weights are required")
        weight_sum = sum(float(value) for value in dict(payload["weights"]).values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError("RWARE PBRS-v2 candidate weights must sum to 1")
        return payload
    if payload.get("pbrs_version") == "lbf_pbrs_v2":
        if not payload.get("mode"):
            raise ValueError("LBF PBRS-v2 candidate mode is required")
        if not isinstance(payload.get("weights"), dict):
            raise ValueError("LBF PBRS-v2 candidate weights are required")
        unexpected_terms = [
            key for key in dict(payload["weights"]).keys() if key not in LBF_PBRS_V2_TERMS
        ]
        if unexpected_terms:
            raise ValueError(
                "LBF PBRS-v2 candidate has unsupported weight terms: "
                + ", ".join(sorted(unexpected_terms))
            )
        weight_sum = sum(float(value) for value in dict(payload["weights"]).values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError("LBF PBRS-v2 candidate weights must sum to 1")
        active_terms = _coerce_string_list(payload.get("active_terms"))
        if not active_terms:
            raise ValueError("LBF PBRS-v2 candidate active_terms are required")
        invalid_active_terms = [term for term in active_terms if term not in LBF_PBRS_V2_TERMS]
        if invalid_active_terms:
            raise ValueError(
                "LBF PBRS-v2 candidate has unsupported active_terms: "
                + ", ".join(sorted(invalid_active_terms))
            )
        return payload
    if abs(float(payload["wp"]) - (1.0 - float(payload["wc"]))) > 1e-6:
        raise ValueError("wp must equal 1 - wc")
    return payload


def sanitize_policy_guided_candidate_batch(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    batch = deepcopy(payload or {})
    if not isinstance(batch, dict):
        raise ValueError("policy-guided candidate batch must be a dict")
    batch["policy_guidance_used"] = bool(batch.get("policy_guidance_used", False))
    basis = deepcopy(batch.get("candidate_generation_basis") or {})
    batch["candidate_generation_basis"] = {
        "reward_diagnosis_used": bool(basis.get("reward_diagnosis_used", False)),
        "policy_diagnosis_used": bool(basis.get("policy_diagnosis_used", False)),
        "evidence_keys": _coerce_string_list(basis.get("evidence_keys")),
    }
    batch["policy_guidance_candidate_collapsed"] = bool(
        batch.get("policy_guidance_candidate_collapsed", False)
    )
    batch["collapsed_reason"] = str(batch.get("collapsed_reason") or "")
    candidates = list(batch.get("candidates") or [])
    batch["candidates"] = [validate_policy_guided_candidate(item) for item in candidates]
    return batch


def validate_policy_guided_candidate_batch(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    batch = sanitize_policy_guided_candidate_batch(payload)
    seen = set()
    for item in batch["candidates"]:
        candidate_id = item["candidate_id"]
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
    return batch


def _sanitize_metric_block(value: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(value or {})
    if not isinstance(payload, dict):
        payload = {}
    score = payload.get("score")
    if score is not None:
        score = float(score)
    metrics = payload.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        raise ValueError("metric block metrics must be a dict when present")
    return {
        "score": score,
        "evidence": _coerce_string_list(payload.get("evidence")),
        "metrics": deepcopy(metrics or {}),
    }


def _sanitize_evidence_flags(value: Dict[str, Any] | None) -> Dict[str, bool]:
    payload = deepcopy(value or {})
    if not isinstance(payload, dict):
        raise ValueError("evidence_flags must be a dict")
    if any(
        key in payload
        for key in (
            "requested_shelf_access_low",
            "pickup_progress_weak",
            "carrying_progress_weak",
            "delivery_progress_weak",
            "traffic_blocking_high",
            "route_stability_low",
        )
    ):
        return {
            "requested_shelf_access_low": bool(payload.get("requested_shelf_access_low", False)),
            "pickup_progress_weak": bool(payload.get("pickup_progress_weak", False)),
            "carrying_progress_weak": bool(payload.get("carrying_progress_weak", False)),
            "delivery_progress_weak": bool(payload.get("delivery_progress_weak", False)),
            "traffic_blocking_high": bool(payload.get("traffic_blocking_high", False)),
            "route_stability_low": bool(payload.get("route_stability_low", False)),
            "insufficient_behavior_evidence": bool(
                payload.get("insufficient_behavior_evidence", False)
            ),
        }
    return {
        "low_coverage": bool(payload.get("low_coverage", False)),
        "food_discovery_failure": bool(payload.get("food_discovery_failure", False)),
        "near_food_collection_failure": bool(
            payload.get("near_food_collection_failure", False)
        ),
        "over_concentration_possible": bool(
            payload.get("over_concentration_possible", False)
        ),
        "unstable_behavior": bool(payload.get("unstable_behavior", False)),
        "insufficient_behavior_evidence": bool(
            payload.get("insufficient_behavior_evidence", False)
        ),
    }


def _sanitize_episode_summaries(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("episode_summaries must be a list")
    normalized: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each episode summary must be a dict")
        normalized.append(deepcopy(item))
    return normalized


def _sanitize_milestone(value: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(value or {})
    if not isinstance(payload, dict):
        raise ValueError("milestone must be a dict")
    return {
        "stage_label": str(payload.get("stage_label") or ""),
        "target_step": int(payload.get("target_step") or 0),
        "actual_step": int(payload.get("actual_step") or 0),
        "num_eval_episodes": int(payload.get("num_eval_episodes") or 0),
    }


def _coerce_string_list(value: Iterable[Any] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    return [str(item) for item in value]


def _sanitize_metadata(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metadata must be a dict")
    return deepcopy(value)


def _sanitize_confidence(value: Any) -> str:
    candidate = str(value or "low")
    if candidate not in CONFIDENCE_LEVELS:
        return "low"
    return candidate


def _sanitize_direction(value: Any) -> str:
    candidate = str(value or "unknown")
    if candidate not in DIRECTION_VALUES:
        return "unknown"
    return candidate


def _sanitize_unit_interval(value: Any) -> float:
    if value is None:
        return 0.0
    coerced = float(value)
    return max(0.0, min(1.0, coerced))
