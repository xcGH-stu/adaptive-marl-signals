from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from workflows.baseline_run_support import inspect_baseline_checkpoint_readiness
from workflows.clients.llm_routing import llm_routing_artifact_fields
from workflows.clients.openai_backend import OpenAIChatBackend, probe_openai_backend
from workflows.diagnostics import improvement_slope, load_metric_series, series_summary
from workflows.final_stage_conditioned_spec import synthesize_final_stage_conditioned_spec
from workflows.llm_api_ledger import extract_request_id
from workflows.llm_api_ledger import stable_payload_hash
from workflows.llm_api_ledger import utc_now_iso
from workflows.llm_api_ledger import write_ledger_event
from workflows.llm_experiment_memory import (
    load_or_init_memory,
    save_memory,
    update_after_fixed_checkpoint_plan,
    update_after_stage3_decision,
)
from workflows.policy_guidance import (
    append_policy_guidance_call_record,
    append_stage3_policy_guided_candidate_generation,
    build_policy_guided_stage3_candidate_batch,
    build_insufficient_behavior_summary,
    build_stage3_after_round1_policy_guidance,
    build_stage3_pre_checkpoint_policy_guidance,
)
from workflows.stage_conditioned_branching import (
    _build_result_record,
    _classify_existing_candidate_state,
    _launch_candidate_background_and_bind_run,
    _recover_or_load_existing_branch_result,
)
from workflows.storage import WorkflowStorage
from workflows.train_checkpointing import normalize_base_train_config, normalize_checkpoint_steps
from workflows.train_launcher import EPyMARLTrainLauncher
from rewarding.lbf_pbrs_v2 import DEFAULT_MODE_CONFIGS as LBF_PBRS_V2_MODE_CONFIGS
from rewarding.lbf_pbrs_v2 import legacy_lbf_candidate_to_v2
from rewarding.lbf_pbrs_v2 import normalize_lbf_pbrs_v2_config
from rewarding.rware_pbrs_v2 import normalize_rware_pbrs_v2_config

RWARE_STAGE3_TERMS = ("shelf", "pickup", "goal", "deliv", "traffic", "stab")
RWARE_STAGE3_MODES = (
    "requested_shelf_acquisition",
    "balanced_delivery_progress",
    "carrying_to_goal",
    "traffic_conservative",
    "late_stability",
)
LBF_STAGE3_TERMS = ("col", "app", "cov", "ready", "alloc", "stab")
LBF_STAGE3_RUNTIME_MODES = tuple(LBF_PBRS_V2_MODE_CONFIGS.keys())
LBF_STAGE3_MODE_ALIAS_TO_RUNTIME_MODE = {
    "balanced": "balanced_collection_ready",
    "balanced_progress": "balanced_collection_ready",
    "collection_readiness": "balanced_collection_ready",
    "early_discovery": "coverage_ready_balance",
    "exploration_boost": "coverage_ready_balance",
    "coverage_recovery": "coverage_ready_balance",
    "progress_shift": "approach_collection_push",
    "collection_geometry": "approach_collection_push",
    "allocation_rebalance": "allocation_stability_support",
    "stability_recovery": "allocation_stability_support",
    "coordination_recovery": "allocation_stability_support",
    "traffic_conservative": "allocation_stability_support",
    "late_stability": "allocation_stability_support",
    "conservative": "allocation_stability_support",
}
RWARE_WEIGHT_TOLERANCE = 1e-9
LBF_CONFIG_FLOAT_TOLERANCE = 1e-8


DEFAULT_RESULTS_ROOT = (
    Path(__file__).resolve().parents[2] / "results" / "adaptive_checkpoint_replacement_workflows"
)
DEFAULT_REQUIRED_STAGE_LABELS = ("early_exploration", "mid_progress", "late_plateau")
DEFAULT_STAGE3_CANDIDATE_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "adaptive_stage3_candidate_generation_prompt.md"
)
DEFAULT_POLICY_GUIDED_STAGE3_ROUND1_CANDIDATE_PROMPT_PATH = (
    Path(__file__).resolve().parent
    / "prompts"
    / "policy_guidance"
    / "policy_guided_stage3_round1_generator_prompt.md"
)
DEFAULT_POLICY_GUIDED_STAGE3_ROUND2_CANDIDATE_PROMPT_PATH = (
    Path(__file__).resolve().parent
    / "prompts"
    / "policy_guidance"
    / "policy_guided_stage3_round2_generator_prompt.md"
)
DEFAULT_STAGE3_DIAGNOSIS_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "adaptive_stage3_result_diagnosis_prompt.md"
)
DEFAULT_FIXED_INTERVENTION_STAGE_LABELS = (
    "post_initialization_transition",
    "late_stability",
)


def _stage_key_from_label(stage_label: str) -> Optional[str]:
    return {
        "early_exploration": "early",
        "mid_progress": "mid",
        "late_plateau": "late",
        "post_initialization_transition": "mid",
        "late_stability": "late",
    }.get(stage_label)


def _fmt_value(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _candidate_id(beta: float, wc: float) -> str:
    return f"beta_{_fmt_value(beta)}_wc_{_fmt_value(wc)}"


def _decision_id(workflow_id: str, checkpoint_name: str) -> str:
    return f"{workflow_id}_{checkpoint_name}"


def _adaptive_manifest_alias_path(workflow_dir: Path) -> Path:
    return workflow_dir / "adaptive_checkpoint_replacement_manifest.json"


def _adaptive_storage_manifest_path(workflow_dir: Path) -> Path:
    return workflow_dir / "manifest.json"


def _load_existing_adaptive_manifest(workflow_dir: Path) -> Dict[str, Any]:
    for path in (
        _adaptive_manifest_alias_path(workflow_dir),
        _adaptive_storage_manifest_path(workflow_dir),
    ):
        if path.exists():
            return _load_json(path)
    return {}


def _save_adaptive_manifest(workflow_dir: Path, manifest: Dict[str, Any]) -> None:
    alias_path = _adaptive_manifest_alias_path(workflow_dir)
    storage_path = _adaptive_storage_manifest_path(workflow_dir)
    workflow_summary = manifest.setdefault("workflow_summary", {})
    if isinstance(workflow_summary, dict):
        workflow_summary["real_llm_called"] = _adaptive_real_llm_called(workflow_dir)
    _save_json(alias_path, manifest)
    if storage_path != alias_path:
        _save_json(storage_path, manifest)
    _rewrite_adaptive_timeline(workflow_dir, manifest)


def _checkpoint_schedule_from_adaptive_method(
    adaptive_method: Dict[str, Any],
) -> List[Dict[str, Any]]:
    fixed_steps = normalize_checkpoint_steps(
        adaptive_method.get("fixed_intervention_checkpoints") or []
    )
    planned: List[Dict[str, Any]] = []
    for index, step in enumerate(fixed_steps, start=1):
        stage_label = DEFAULT_FIXED_INTERVENTION_STAGE_LABELS[
            min(index - 1, len(DEFAULT_FIXED_INTERVENTION_STAGE_LABELS) - 1)
        ]
        planned.append(
            {
                "name": f"C{index}",
                "step": int(step),
                "stage_label": stage_label,
                "reason": "Recovered from fixed_intervention_checkpoints profile fallback.",
                "recommended_candidate_focus": [],
                "branch_budget_feasible": True,
                "adaptation_window_quality": "fixed",
            }
        )
    return planned


def _resolve_selected_checkpoints(
    *,
    stage_selection_result: Dict[str, Any],
    adaptive_method: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    selected = list(
        ((stage_selection_result.get("selection") or {}).get("selected_checkpoints")) or []
    )
    if selected:
        return deepcopy(selected), "stage_2_selection"
    fallback = _checkpoint_schedule_from_adaptive_method(adaptive_method)
    if fallback:
        return fallback, "profile_fixed_checkpoints"
    return [], "missing"


def _stage_selection_with_selected_checkpoints(
    stage_selection_result: Dict[str, Any],
    selected_checkpoints: List[Dict[str, Any]],
    *,
    schedule_source: str,
) -> Dict[str, Any]:
    patched = deepcopy(stage_selection_result or {})
    patched.setdefault("selection", {})
    patched["selection"]["selected_checkpoints"] = deepcopy(selected_checkpoints)
    patched["selection_mode"] = str(
        patched.get("selection_mode")
        or ("fixed_intervention_schedule" if schedule_source == "profile_fixed_checkpoints" else "")
    )
    patched.setdefault("selection_constraints", {})
    patched["selection_constraints"]["checkpoint_count"] = len(selected_checkpoints)
    patched["selection_constraints"]["fixed_intervention_checkpoints"] = [
        int(checkpoint.get("step"))
        for checkpoint in selected_checkpoints
        if checkpoint.get("step") is not None
    ]
    return patched


def _manifest_requires_decision_initialization(
    manifest: Dict[str, Any],
    *,
    selected_checkpoints: List[Dict[str, Any]],
) -> bool:
    if not manifest:
        return False
    decisions = list(manifest.get("decisions") or [])
    if decisions:
        return False
    if not selected_checkpoints:
        return False
    if manifest.get("workflow_kind") != "adaptive_checkpoint_replacement":
        return False
    return True


def _round_cache_suffix(decision: Dict[str, Any]) -> str:
    round_id = decision.get("active_round_id")
    return f"_round{int(round_id)}" if round_id is not None else ""


def _stage1b_required_source_fields() -> tuple[str, ...]:
    return (
        "selected_candidate_id",
        "selected_initial_dense_config",
        "selected_endpoint_checkpoint_path",
        "selected_endpoint_checkpoint_step",
        "dense_reference_source_checkpoint_path",
        "adaptive_mainline_source_checkpoint_path",
        "adaptive_mainline_initial_pbrs_config",
    )


def _validate_stage1b_source_metadata(stage1b_source_metadata: Dict[str, Any]) -> Dict[str, Any]:
    missing = [
        field
        for field in _stage1b_required_source_fields()
        if not stage1b_source_metadata.get(field)
    ]
    if missing:
        raise ValueError(
            "Stage 3 production wiring requires Stage 1b source metadata fields: "
            + ", ".join(missing)
        )
    return deepcopy(stage1b_source_metadata)


def _load_stage3_fork_spec_json(stage3_fork_spec_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(stage3_fork_spec_json, str) or not stage3_fork_spec_json.strip():
        return None
    path = Path(stage3_fork_spec_json)
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("stage3 fork spec json must decode to an object")
    payload = deepcopy(payload)
    payload["fork_spec_json"] = str(path)
    return payload


def _infer_checkpoint_step_from_path(checkpoint_path: Any) -> Optional[int]:
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        return None
    try:
        return int(Path(checkpoint_path).name)
    except (TypeError, ValueError):
        return None


def _normalize_stage3_fork_winner_config(
    winner_config: Dict[str, Any],
    *,
    env_key: Any,
    pbrs_version: Any,
) -> Dict[str, Any]:
    if is_rware_workflow(env_key=env_key, pbrs_version=pbrs_version):
        normalized = deepcopy(winner_config)
        normalized.setdefault("pbrs_version", "rware_pbrs_v2")
        normalized.setdefault("candidate_type", "fork_bootstrap_winner")
        return _normalize_rware_stage_config(normalized, require_candidate_type=False)
    normalized = deepcopy(winner_config)
    normalized.setdefault("pbrs_version", "lbf_pbrs_v2")
    return _normalize_lbf_stage_config(normalized)


def _apply_stage3_fork_bootstrap(
    *,
    manifest: Dict[str, Any],
    stage3_fork_spec: Dict[str, Any],
) -> Dict[str, Any]:
    decisions = list(manifest.get("decisions") or [])
    if len(decisions) < 2:
        raise ValueError("stage3 fork bootstrap requires at least C1 and C2 decisions")
    first_decision = decisions[0]
    endpoint_checkpoint_path = str(stage3_fork_spec.get("endpoint_checkpoint_path") or "").strip()
    if not endpoint_checkpoint_path:
        raise ValueError("stage3 fork spec must include endpoint_checkpoint_path")
    endpoint_checkpoint_step = int(
        stage3_fork_spec.get("endpoint_checkpoint_step")
        or _infer_checkpoint_step_from_path(endpoint_checkpoint_path)
        or 0
    )
    if endpoint_checkpoint_step <= 0:
        raise ValueError("stage3 fork spec must include a valid endpoint checkpoint step")
    next_checkpoint_step = int(stage3_fork_spec.get("next_checkpoint_step") or 0)
    if next_checkpoint_step <= endpoint_checkpoint_step:
        raise ValueError("stage3 fork spec next_checkpoint_step must exceed endpoint checkpoint step")
    final_target_t_max = int(stage3_fork_spec.get("final_target_t_max") or manifest.get("t_max") or 0)
    if final_target_t_max < next_checkpoint_step:
        raise ValueError("stage3 fork spec final_target_t_max must be >= next_checkpoint_step")
    winner_config_raw = deepcopy(stage3_fork_spec.get("winner_config") or {})
    if not winner_config_raw:
        raise ValueError("stage3 fork spec must include winner_config")
    winner_config = _normalize_stage3_fork_winner_config(
        winner_config_raw,
        env_key=first_decision.get("env_key"),
        pbrs_version=first_decision.get("pbrs_version"),
    )
    winner_candidate_id = str(
        stage3_fork_spec.get("winner_candidate_id")
        or _candidate_id(float(winner_config["beta"]), float(winner_config["wc"]))
    )
    candidate_payload = deepcopy(winner_config)
    candidate_payload.update(
        {
            "candidate_id": winner_candidate_id,
            "candidate_type": str(stage3_fork_spec.get("winner_candidate_type") or "fork_bootstrap_winner"),
            "status": "completed",
            "run_id": stage3_fork_spec.get("endpoint_run_id"),
            "run_dir": stage3_fork_spec.get("endpoint_run_dir") or stage3_fork_spec.get("source_run_dir"),
            "checkpoint_root_dir": str(Path(endpoint_checkpoint_path).parent),
            "available_checkpoint_steps": [endpoint_checkpoint_step],
            "latest_checkpoint_step": endpoint_checkpoint_step,
            "latest_checkpoint_path": endpoint_checkpoint_path,
            "endpoint_checkpoint_path": endpoint_checkpoint_path,
            "endpoint_checkpoint_step": endpoint_checkpoint_step,
            "result_record": deepcopy(stage3_fork_spec.get("winner_result_record") or {}),
            "fork_bootstrap_selected": True,
        }
    )
    first_decision["source_run_dir"] = str(
        stage3_fork_spec.get("source_run_dir")
        or stage3_fork_spec.get("endpoint_run_dir")
        or first_decision.get("source_run_dir")
        or ""
    )
    first_decision["source_checkpoint_root_dir"] = str(Path(endpoint_checkpoint_path).parent)
    first_decision["source_available_steps"] = [endpoint_checkpoint_step]
    first_decision["source_latest_checkpoint_step"] = endpoint_checkpoint_step
    first_decision["source_latest_checkpoint_path"] = endpoint_checkpoint_path
    first_decision["candidate_generation"] = {"source": "stage3_fork_bootstrap"}
    first_decision["candidates"] = [candidate_payload]
    first_decision["winner_candidate_id"] = winner_candidate_id
    first_decision["selected_winner_candidate_id"] = winner_candidate_id
    first_decision["selected_winner_branch_run_id"] = candidate_payload.get("run_id")
    first_decision["selected_winner_endpoint_checkpoint_path"] = endpoint_checkpoint_path
    first_decision["selected_winner_endpoint_checkpoint_step"] = endpoint_checkpoint_step
    first_decision["winner_config"] = deepcopy(winner_config)
    first_decision["replacement_accepted"] = True
    first_decision["promoted_to_mainline"] = False
    first_decision["promotion_reason"] = "stage3_fork_bootstrap"
    first_decision["decision_source"] = "stage3_fork_bootstrap"
    first_decision["final_decision_source"] = "stage3_fork_bootstrap"
    first_decision["deterministic_gate_completed"] = True
    first_decision["fallback_used"] = False
    first_decision["critic_pre_diagnosis_status"] = "completed"
    first_decision["critic_pre_diagnosis"] = {
        "fork_bootstrap": True,
        "endpoint_checkpoint_path": endpoint_checkpoint_path,
        "endpoint_checkpoint_step": endpoint_checkpoint_step,
    }
    first_decision["llm_candidate_generation_status"] = "skipped_fork_bootstrap"
    first_decision["llm_result_diagnosis_status"] = "skipped_fork_bootstrap"
    first_decision["fork_bootstrap"] = {
        "enabled": True,
        "fork_spec_json": stage3_fork_spec.get("fork_spec_json"),
        "skip_branch_search": True,
        "winner_candidate_id": winner_candidate_id,
        "endpoint_checkpoint_path": endpoint_checkpoint_path,
        "endpoint_checkpoint_step": endpoint_checkpoint_step,
        "next_checkpoint_step": next_checkpoint_step,
        "final_target_t_max": final_target_t_max,
    }
    for round_payload in list(first_decision.get("rounds") or []):
        round_payload["status"] = "skipped_fork_bootstrap"
        round_payload["candidate_generation_status"] = "skipped_fork_bootstrap"
        round_payload["candidate_review_status"] = "skipped_fork_bootstrap"
        round_payload["critic_analysis_status"] = "skipped_fork_bootstrap"
        round_payload["analysis"] = {"fork_bootstrap": True}
        round_payload["generator_candidates"] = []
        round_payload["candidates"] = []
    first_decision["continuation"]["target_step"] = next_checkpoint_step
    first_decision["continuation"]["status"] = "pending"
    first_decision["completed"] = False
    manifest["t_max"] = final_target_t_max
    decisions[-1]["continuation"]["target_step"] = final_target_t_max
    manifest["fork_bootstrap"] = {
        "enabled": True,
        "fork_spec_json": stage3_fork_spec.get("fork_spec_json"),
        "winner_candidate_id": winner_candidate_id,
        "winner_config": deepcopy(winner_config),
        "endpoint_checkpoint_path": endpoint_checkpoint_path,
        "endpoint_checkpoint_step": endpoint_checkpoint_step,
        "next_checkpoint_step": next_checkpoint_step,
        "final_target_t_max": final_target_t_max,
    }
    manifest["decisions"] = decisions
    return manifest


def _decision_round_config(decision: Dict[str, Any]) -> Dict[str, Any]:
    round_budget = max(
        1,
        int(
            decision.get("stage3_update_candidates_per_round")
            or decision.get("stage3_candidates_per_round")
            or 3
        ),
    )
    include_no_change = bool(decision.get("stage3_include_no_change_control", True))
    no_change_counts = bool(
        decision.get("stage3_no_change_counts_toward_round_budget", False)
    )
    round1_generated = round_budget - 1 if include_no_change and no_change_counts else round_budget
    return {
        "round_budget": round_budget,
        "include_no_change": include_no_change,
        "no_change_counts": no_change_counts,
        "round1_generated": max(0, round1_generated),
        "round2_generated": max(0, round_budget),
    }


def _stage3_candidate_evidence_keys(candidate: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    single = candidate.get("evidence_key_used")
    if isinstance(single, str) and single.strip():
        keys.append(single.strip())
    for item in list(candidate.get("evidence_keys_used") or []):
        if isinstance(item, str) and item.strip():
            keys.append(item.strip())
    return list(dict.fromkeys(keys))


_STAGE3_GENERIC_PROVENANCE_EVIDENCE_KEYS = {
    "stage_label",
    "checkpoint_name",
    "checkpoint_step",
    "round_id",
    "current_mainline_config",
    "initial_dense_config",
    "stage_selection_summary",
    "dense_reference_run_summary",
    "sparse_reference_run_summary",
    "round1_result_summary",
    "reference_comparison",
}


def _stage3_policy_evidence_keys(candidate: Dict[str, Any]) -> List[str]:
    keys = _stage3_candidate_evidence_keys(candidate)
    policy_keys: List[str] = []
    for item in keys:
        normalized = str(item or "").strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in _STAGE3_GENERIC_PROVENANCE_EVIDENCE_KEYS:
            continue
        policy_keys.append(normalized)
    return policy_keys


def _stage3_candidate_delta_metrics(
    *,
    current_config: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    if is_rware_workflow(
        env_key=current_config.get("env_key") or candidate.get("env_key"),
        pbrs_version=current_config.get("pbrs_version") or candidate.get("pbrs_version"),
    ):
        current_norm = _normalize_rware_stage_config(current_config, require_candidate_type=False)
        candidate_norm = _normalize_rware_stage_config(candidate)
        weight_deltas = [
            abs(
                float((candidate_norm.get("weights") or {}).get(term, 0.0))
                - float((current_norm.get("weights") or {}).get(term, 0.0))
            )
            for term in RWARE_STAGE3_TERMS
        ]
        return {
            "mode_changed": str(candidate_norm.get("mode") or "") != str(current_norm.get("mode") or ""),
            "beta_delta": abs(float(candidate_norm["beta"]) - float(current_norm["beta"])),
            "max_core_weight_delta": max(weight_deltas) if weight_deltas else 0.0,
            "weights_l1_delta": sum(weight_deltas),
            "active_terms_changed": list(candidate_norm.get("active_terms") or [])
            != list(current_norm.get("active_terms") or []),
        }
    current_beta = float(current_config["beta"])
    current_wc = float(current_config["wc"])
    current_wp = round(1.0 - current_wc, 10)
    candidate_beta = float(candidate["beta"])
    candidate_wc = float(candidate["wc"])
    candidate_wp = float(candidate.get("wp", round(1.0 - candidate_wc, 10)))
    beta_delta = abs(candidate_beta - current_beta)
    wc_delta = abs(candidate_wc - current_wc)
    wp_delta = abs(candidate_wp - current_wp)
    candidate_terms = {str(item) for item in list(candidate.get("active_terms") or []) if str(item)}
    current_terms = {str(item) for item in list(current_config.get("active_terms") or []) if str(item)}
    return {
        "mode_changed": bool(candidate.get("mode")) and candidate.get("mode") != current_config.get("mode"),
        "beta_delta": beta_delta,
        "max_core_weight_delta": max(wc_delta, wp_delta),
        "weights_l1_delta": wc_delta + wp_delta,
        "active_terms_changed": candidate_terms != current_terms,
    }


def _is_nontrivial_stage3_update(
    *,
    current_config: Dict[str, float],
    candidate: Dict[str, Any],
) -> bool:
    delta = _stage3_candidate_delta_metrics(current_config=current_config, candidate=candidate)
    return bool(
        delta["mode_changed"]
        or delta["beta_delta"] >= 0.1
        or delta["max_core_weight_delta"] >= 0.10
        or delta["weights_l1_delta"] >= 0.20
        or delta["active_terms_changed"]
    )


def _nontrivial_stage3_update_failure_detail(
    *,
    candidate_id: str,
    current_config: Dict[str, Any],
    candidate: Dict[str, Any],
) -> str:
    delta = _stage3_candidate_delta_metrics(current_config=current_config, candidate=candidate)
    return (
        "near_duplicate_or_non_effective_update:"
        f"{candidate_id}:"
        "compared_to=current_config:"
        f"mode_changed={delta['mode_changed']}:"
        f"beta_delta={delta['beta_delta']:.6f}:"
        "beta_threshold=0.100000:"
        f"max_core_weight_delta={delta['max_core_weight_delta']:.6f}:"
        "max_core_weight_threshold=0.100000:"
        f"weights_l1_delta={delta['weights_l1_delta']:.6f}:"
        "weights_l1_threshold=0.200000:"
        f"active_terms_changed={delta['active_terms_changed']}:"
        f"comparison_basis={'rware_mode_active_terms_weights' if is_rware_workflow(env_key=current_config.get('env_key'), pbrs_version=current_config.get('pbrs_version') or candidate.get('pbrs_version')) else 'legacy_beta_wc_wp'}"
    )


def _stage3_candidate_category(candidate_type: str) -> str:
    normalized = str(candidate_type or "").strip().lower()
    if normalized in {"reference_like", "conservative", "conservative_neighbor"}:
        return "conservative_or_reference_like"
    if normalized in {"stability_recovery", "stability_aware", "late_stability"}:
        return "stability_aware"
    if normalized in {
        "coverage_recovery",
        "collection_geometry",
        "allocation_rebalance",
        "progress_shift",
        "behavior_targeted",
    }:
        return "behavior_targeted"
    if "recovery" in normalized or normalized in {"beta_down", "beta_up", "wc_down_wp_up", "wc_up_wp_down"}:
        return "exploratory_or_recovery"
    return "generic_update"


def _is_fallback_style_stage3_candidate(candidate: Dict[str, Any]) -> bool:
    candidate_type = str(candidate.get("candidate_type") or "").strip().lower()
    source_config = str(candidate.get("source_config") or "").strip().lower()
    if bool(candidate.get("deterministic_fallback")):
        return True
    if source_config in {"deterministic_fallback_topup", "reference_preserve"}:
        return True
    return candidate_type in {"conservative", "conservative_neighbor"}


def _candidate_mentions_forbidden_policy_evidence(candidate: Dict[str, Any]) -> bool:
    searchable = " ".join(
        str(candidate.get(field) or "")
        for field in ("rationale", "expected_effect", "candidate_type", "evidence_key_used")
    ).lower()
    return any(
        token in searchable
        for token in ("policy_guidance", "behavior summary", "behavior_evidence", "guidance card")
    ) or bool(_stage3_policy_evidence_keys(candidate))


def _default_stage3_candidate_type(
    *,
    decision: Dict[str, Any],
    round_id: int,
    index: int,
    policy_guidance_used: bool,
) -> str:
    if policy_guidance_used:
        if round_id <= 1:
            ordered = ["reference_like", "progress_shift", "coverage_recovery"]
        elif str(decision.get("checkpoint_name") or "").upper() == "C2":
            ordered = ["stability_recovery", "allocation_rebalance", "progress_shift"]
        else:
            ordered = ["stability_recovery", "progress_shift", "allocation_rebalance"]
    else:
        if round_id <= 1:
            ordered = ["conservative_neighbor", "recovery", "beta_down"]
        elif str(decision.get("checkpoint_name") or "").upper() == "C2":
            ordered = ["stability_recovery", "recovery", "beta_up"]
        else:
            ordered = ["stability_recovery", "recovery", "beta_up"]
    return ordered[index % len(ordered)]


def _normalize_lbf_stage3_mode_alias(
    candidate_like: Dict[str, Any],
) -> Dict[str, Any]:
    def resolve_alias(alias_like: str) -> Tuple[Optional[str], Optional[str]]:
        text = str(alias_like or "").strip().lower()
        if not text:
            return None, None
        if text in LBF_STAGE3_MODE_ALIAS_TO_RUNTIME_MODE:
            return LBF_STAGE3_MODE_ALIAS_TO_RUNTIME_MODE[text], text
        trimmed = text.replace("-", "_")
        if trimmed in LBF_STAGE3_MODE_ALIAS_TO_RUNTIME_MODE:
            return LBF_STAGE3_MODE_ALIAS_TO_RUNTIME_MODE[trimmed], trimmed
        prefix = trimmed.split("__", 1)[0].split("_", 3)
        while prefix:
            candidate_prefix = "_".join(prefix)
            if candidate_prefix in LBF_STAGE3_MODE_ALIAS_TO_RUNTIME_MODE:
                return LBF_STAGE3_MODE_ALIAS_TO_RUNTIME_MODE[candidate_prefix], candidate_prefix
            prefix = prefix[:-1]
        return None, None

    normalized = deepcopy(candidate_like)
    raw_mode = str(candidate_like.get("mode") or "").strip()
    if raw_mode in LBF_STAGE3_RUNTIME_MODES:
        return normalized
    candidate_type = str(candidate_like.get("candidate_type") or "").strip().lower()
    normalized_mode, normalized_from_alias = resolve_alias(raw_mode)
    if normalized_mode is None:
        normalized_mode, normalized_from_alias = resolve_alias(candidate_type)
    if normalized_mode is None:
        raise ValueError(
            "Unsupported LBF PBRS-v2 mode: "
            f"{raw_mode or '<missing>'}"
            f" (candidate_type={candidate_type or '<missing>'})"
        )
    normalized["mode"] = normalized_mode
    normalized["original_mode"] = raw_mode
    normalized["normalized_from_alias"] = normalized_from_alias
    return normalized


def _build_stage3_deterministic_update_topups(
    *,
    decision: Dict[str, Any],
    current_config: Dict[str, float],
    round_id: int,
    existing_identities: set[tuple[float, float]],
    existing_candidates: List[Dict[str, Any]],
    target_count: int,
    policy_guidance: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if is_rware_workflow(
        env_key=decision.get("env_key"),
        pbrs_version=decision.get("pbrs_version") or current_config.get("pbrs_version"),
    ):
        return []
    candidates: List[Dict[str, Any]] = []
    policy_guidance_used = bool(policy_guidance.get("policy_guidance_used", False))
    evidence_keys = list(policy_guidance.get("policy_guidance_evidence_keys") or [])
    checkpoint_name = str(decision.get("checkpoint_name") or "ckpt").lower()
    base_specs = [
        (0.10, 0.10),
        (-0.10, 0.10),
        (0.15, -0.10),
        (-0.15, -0.10),
        (0.20, 0.00),
        (0.00, 0.15),
    ]
    spec_index = 0
    while len(existing_candidates) + len(candidates) < target_count and spec_index < len(base_specs):
        beta_shift, wc_shift = base_specs[spec_index]
        spec_index += 1
        beta = min(1.0, max(0.0, round(float(current_config["beta"]) + beta_shift, 3)))
        wc = min(1.0, max(0.0, round(float(current_config["wc"]) + wc_shift, 3)))
        normalized = _normalize_candidate_config(beta=beta, wc=wc)
        if normalized is None:
            continue
        identity = (normalized["beta"], normalized["wc"])
        if identity in existing_identities:
            continue
        candidate_type = _default_stage3_candidate_type(
            decision=decision,
            round_id=round_id,
            index=len(existing_candidates) + len(candidates),
            policy_guidance_used=policy_guidance_used,
        )
        normalized.update(
            {
                "candidate_id": f"{checkpoint_name}_r{int(round_id)}_topup_{candidate_type}_{len(candidates)+1}",
                "candidate_type": candidate_type,
                "expected_effect": "local schema-preserving top-up after candidate repair/validation shortfall",
                "rationale": "Local clean-schema top-up inserted only to satisfy the strict Stage3 update count after replay/repair shortfall.",
                "source_config": "schema_contract_topup",
                "is_no_change": False,
                "fallback_style_candidate": _is_fallback_style_stage3_candidate({"candidate_type": candidate_type}),
                "evidence_keys_used": evidence_keys[:1] if policy_guidance_used and evidence_keys else [],
            }
        )
        _attach_env_specific_pbrs_metadata(
            normalized,
            env_key=decision.get("env_key"),
            pbrs_version=decision.get("pbrs_version"),
        )
        if not _is_nontrivial_stage3_update(current_config=current_config, candidate=normalized):
            continue
        existing_identities.add(identity)
        candidates.append(normalized)
    return candidates


def _repair_stage3_candidate_shortfall(
    *,
    decision: Dict[str, Any],
    current_config: Dict[str, float],
    candidate_generation: Dict[str, Any],
    round_policy_guidance: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    required_effective_candidates: int,
    validation_errors: List[str],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    original_valid_count = len(list(candidates or []))
    audit = {
        "original_valid_count": int(original_valid_count),
        "topup_count": 0,
        "repair_status": "not_needed",
        "repair_type": "schema_contract_topup",
        "deterministic_fallback_used": False,
    }
    if original_valid_count >= required_effective_candidates:
        return list(candidates), audit, list(validation_errors)

    round_id = int(decision.get("active_round_id") or 1)
    max_candidates = max(1, int(candidate_generation.get("max_candidates_per_checkpoint") or 9))
    existing_identities = {_candidate_identity(item) for item in candidates}
    topups = _build_stage3_deterministic_update_topups(
        decision=decision,
        current_config=current_config,
        round_id=round_id,
        existing_identities=existing_identities,
        existing_candidates=list(candidates),
        target_count=required_effective_candidates,
        policy_guidance=deepcopy(round_policy_guidance or {}),
    )
    repaired = list(candidates)
    for candidate in topups:
        if len(repaired) >= min(max_candidates, required_effective_candidates):
            break
        repaired.append(candidate)

    audit["topup_count"] = max(0, len(repaired) - original_valid_count)
    audit["repair_status"] = (
        "shortfall_repaired"
        if len(repaired) >= required_effective_candidates
        else "shortfall_unresolved"
    )
    updated_errors = list(validation_errors)
    updated_errors.append(
        "schema_contract_topup_attempted:"
        f"required={required_effective_candidates}:"
        f"original_valid={original_valid_count}:"
        f"topup_added={audit['topup_count']}:"
        f"final_valid={len(repaired)}"
    )
    return repaired, audit, updated_errors


def _all_round_candidates(decision: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for round_payload in list(decision.get("rounds") or []):
        rows.extend(list(round_payload.get("candidates") or []))
    return rows


def _stage3_memory_static_context(
    *,
    workflow_id: str,
    dense_reference_run_summary: Dict[str, Any],
    branch_budget_steps: int,
    llm_routing: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    config = deepcopy(dense_reference_run_summary.get("config") or {})
    return {
        "workflow_id": workflow_id,
        "env_key": str(config.get("env_args", {}).get("key") or config.get("env_key") or ""),
        "algorithm": str(config.get("name") or config.get("train_config") or ""),
        "seed": int(config.get("seed") or 1),
        "objective_metric": "test_sparse_return_mean",
        "early_budget_steps": int(branch_budget_steps),
        "llm_routing": deepcopy(llm_routing_artifact_fields(llm_routing or {})),
    }


def _candidate_score(result_record: Dict[str, Any], decision_rule: Dict[str, Any]) -> Tuple[float, float]:
    primary_metric = str(decision_rule.get("primary_metric") or "best_test_sparse_return_mean")
    secondary_metric = str(decision_rule.get("secondary_metric") or "last_test_sparse_return_mean")
    primary = result_record.get(primary_metric)
    secondary = result_record.get(secondary_metric)
    try:
        primary_value = float(primary)
    except (TypeError, ValueError):
        primary_value = float("-inf")
    try:
        secondary_value = float(secondary)
    except (TypeError, ValueError):
        secondary_value = float("-inf")
    return (primary_value, secondary_value)


def _extract_json_object(text: str) -> Dict[str, Any]:
    candidate_texts = [text.strip()]
    stripped = text.strip()
    if "```" in stripped:
        fence_start = stripped.find("```")
        while fence_start != -1:
            fence_end = stripped.find("```", fence_start + 3)
            if fence_end == -1:
                break
            block = stripped[fence_start + 3 : fence_end].strip()
            if "\n" in block:
                first_line, remainder = block.split("\n", 1)
                if first_line.strip().lower() == "json":
                    block = remainder.strip()
            candidate_texts.append(block)
            fence_start = stripped.find("```", fence_end + 3)
    left = stripped.find("{")
    right = stripped.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidate_texts.append(stripped[left : right + 1])

    last_error: Optional[Exception] = None
    for candidate in candidate_texts:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ValueError(f"LLM output is not valid JSON: {last_error}")


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_std(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean_value = float(sum(values) / len(values))
    variance = sum((float(value) - mean_value) ** 2 for value in values) / len(values)
    return float(math.sqrt(max(0.0, variance)))


def _normalize_candidate_config(*, beta: Any, wc: Any) -> Optional[Dict[str, Any]]:
    beta_value = _safe_float(beta)
    wc_value = _safe_float(wc)
    if beta_value is None or wc_value is None:
        return None
    if not (0.0 <= beta_value <= 1.0):
        return None
    if not (0.0 <= wc_value <= 1.0):
        return None
    wp_value = round(1.0 - wc_value, 10)
    return {
        "candidate_id": _candidate_id(beta_value, wc_value),
        "beta": beta_value,
        "wc": wc_value,
        "wp": wp_value,
    }


def _normalize_rware_stage_config(
    candidate: Dict[str, Any],
    *,
    require_candidate_type: bool = True,
) -> Dict[str, Any]:
    if str(candidate.get("pbrs_version") or "") != "rware_pbrs_v2":
        raise ValueError("RWARE Stage3 candidates must declare pbrs_version=rware_pbrs_v2")
    candidate_type = str(candidate.get("candidate_type") or "").strip()
    if require_candidate_type and not candidate_type:
        raise ValueError("RWARE Stage3 candidates must include candidate_type")
    mode = str(candidate.get("mode") or "").strip()
    if mode not in RWARE_STAGE3_MODES:
        raise ValueError(f"Unsupported RWARE Stage3 mode: {mode}")
    beta_value = _safe_float(candidate.get("beta"))
    if beta_value is None or not (0.0 <= beta_value <= 1.0):
        raise ValueError("RWARE Stage3 beta must be within [0.0, 1.0]")
    active_terms_raw = candidate.get("active_terms")
    if not isinstance(active_terms_raw, list):
        raise ValueError("RWARE Stage3 active_terms must be a list")
    active_terms = [str(term).strip() for term in active_terms_raw if str(term).strip()]
    active_terms = list(dict.fromkeys(active_terms))
    if not active_terms:
        raise ValueError("RWARE Stage3 active_terms must be a non-empty subset")
    invalid_terms = [term for term in active_terms if term not in RWARE_STAGE3_TERMS]
    if invalid_terms:
        raise ValueError(
            "RWARE Stage3 active_terms contain unsupported terms: " + ",".join(invalid_terms)
        )
    raw_weights = candidate.get("weights")
    if not isinstance(raw_weights, dict):
        raise ValueError("RWARE Stage3 weights must be a dict")
    missing_terms = [term for term in RWARE_STAGE3_TERMS if term not in raw_weights]
    if missing_terms:
        raise ValueError("RWARE Stage3 weights missing terms: " + ",".join(missing_terms))
    parsed_weights: Dict[str, float] = {}
    for term in RWARE_STAGE3_TERMS:
        value = _safe_float(raw_weights.get(term))
        if value is None or value < 0.0:
            raise ValueError(f"RWARE Stage3 weight for {term} must be a non-negative float")
        parsed_weights[term] = float(value)
    weight_sum = sum(parsed_weights.values())
    if weight_sum <= 0.0:
        raise ValueError("RWARE Stage3 weights must sum to a positive value")
    normalized_weights = {
        term: float(parsed_weights[term] / weight_sum) for term in RWARE_STAGE3_TERMS
    }
    positive_terms = [
        term for term in RWARE_STAGE3_TERMS if normalized_weights[term] > RWARE_WEIGHT_TOLERANCE
    ]
    if set(active_terms) != set(positive_terms):
        raise ValueError(
            "RWARE Stage3 active_terms must exactly match positive-weight terms:"
            f" active_terms={active_terms}:positive_terms={positive_terms}"
        )
    evidence_keys = candidate.get("evidence_keys_used")
    if evidence_keys is None:
        evidence_keys = []
    if not isinstance(evidence_keys, list):
        raise ValueError("RWARE Stage3 evidence_keys_used must be a list")
    normalized = normalize_rware_pbrs_v2_config(
        {
            "pbrs_version": "rware_pbrs_v2",
            "mode": mode,
            "beta": beta_value,
            "gamma": candidate.get("gamma", 0.99),
            "weights": normalized_weights,
        }
    )
    normalized_active_terms = list(normalized.get("active_terms") or [])
    if normalized_active_terms != [term for term in RWARE_STAGE3_TERMS if term in active_terms]:
        raise ValueError(
            "RWARE Stage3 normalized active_terms diverged from provided schema:"
            f" provided={active_terms}:normalized={normalized_active_terms}"
        )
    payload = {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_type": candidate_type,
        "pbrs_version": "rware_pbrs_v2",
        "mode": str(normalized.get("mode") or mode),
        "beta": float(normalized["beta"]),
        "active_terms": normalized_active_terms,
        "weights": {term: float((normalized.get("weights") or {}).get(term, 0.0)) for term in RWARE_STAGE3_TERMS},
        "gamma": float(normalized.get("gamma", 0.99)),
        "evidence_keys_used": [str(item).strip() for item in evidence_keys if str(item).strip()],
    }
    if payload["beta"] < 0.1 or payload["beta"] > 0.7:
        payload["sanitizer_warnings"] = [
            "rware_beta_outside_formal_recommendation_range:[0.1,0.7]"
        ]
    return payload


def _build_rware_no_change_candidate(
    *,
    current_config: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = _normalize_rware_stage_config(
        {
            "candidate_id": "no_change",
            "candidate_type": "no_change_control",
            "pbrs_version": "rware_pbrs_v2",
            "mode": current_config.get("mode"),
            "beta": current_config.get("beta"),
            "active_terms": list(current_config.get("active_terms") or []),
            "weights": deepcopy(current_config.get("weights") or {}),
            "gamma": current_config.get("gamma", 0.99),
            "evidence_keys_used": [],
        }
    )
    normalized["candidate_id"] = "no_change"
    normalized["candidate_type"] = "no_change_control"
    normalized["source_config"] = "current_mainline"
    normalized["is_no_change"] = True
    return normalized


def _build_lbf_no_change_candidate(
    *,
    current_config: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = _normalize_lbf_stage_config(deepcopy(current_config))
    normalized["candidate_id"] = "no_change"
    normalized["candidate_type"] = "no_change_control"
    normalized["source_config"] = "current_mainline"
    normalized["is_no_change"] = True
    return normalized


def _lbf_pbrs_v2_metadata(*, wc: Any, wp: Any) -> Dict[str, Any]:
    return normalize_lbf_pbrs_v2_config(
        legacy_lbf_candidate_to_v2(beta=0.5, wc=wc, wp=wp)
    )


def detect_env_family(env_key: Any = None, pbrs_version: Any = None) -> str:
    env_text = str(env_key or "").strip().lower()
    pbrs_text = str(pbrs_version or "").strip().lower()
    if "rware" in env_text or pbrs_text == "rware_pbrs_v2":
        return "rware"
    if "lbforaging" in env_text or env_text.startswith("lbf") or pbrs_text == "lbf_pbrs_v2":
        return "lbf"
    return "generic"


def is_lbf_workflow(env_key: Any = None, pbrs_version: Any = None) -> bool:
    return detect_env_family(env_key=env_key, pbrs_version=pbrs_version) == "lbf"


def is_rware_workflow(env_key: Any = None, pbrs_version: Any = None) -> bool:
    return detect_env_family(env_key=env_key, pbrs_version=pbrs_version) == "rware"


def filter_pbrs_terms_for_env_family(
    *,
    env_family: str,
    active_terms: List[Any] | None,
    weights: Dict[str, Any] | None,
) -> tuple[List[str], Dict[str, Any]]:
    filtered_terms = [str(term) for term in list(active_terms or []) if str(term)]
    filtered_weights = deepcopy(weights or {}) if isinstance(weights, dict) else {}
    if env_family == "rware":
        allowed = ("shelf", "pickup", "goal", "deliv", "traffic", "stab")
        return (
            [term for term in filtered_terms if term in allowed],
            {term: filtered_weights[term] for term in allowed if term in filtered_weights},
        )
    if env_family == "lbf":
        allowed = ("col", "app", "cov", "ready", "alloc", "stab")
        return (
            [term for term in filtered_terms if term in allowed],
            {term: filtered_weights[term] for term in allowed if term in filtered_weights},
        )
    return filtered_terms, filtered_weights


def _rware_pbrs_v2_metadata(candidate: Dict[str, Any]) -> Dict[str, Any]:
    config = normalize_rware_pbrs_v2_config(
        {
            "pbrs_version": "rware_pbrs_v2",
            "mode": candidate.get("mode") or "balanced_delivery_progress",
            "beta": candidate.get("beta", 0.30),
            "weights": deepcopy(candidate.get("weights") or {}),
        }
    )
    active_terms, weights = filter_pbrs_terms_for_env_family(
        env_family="rware",
        active_terms=config.get("active_terms") or [],
        weights=config.get("weights") or {},
    )
    return {
        "pbrs_version": "rware_pbrs_v2",
        "mode": str(config.get("mode") or "balanced_delivery_progress"),
        "active_terms": active_terms,
        "weights": weights,
    }


def build_env_specific_review_metadata(
    *,
    candidate: Dict[str, Any],
    env_key: Any = None,
    pbrs_version: Any = None,
) -> Dict[str, Any]:
    detected_pbrs_version = str(
        pbrs_version or candidate.get("pbrs_version") or ""
    )
    env_family = detect_env_family(env_key=env_key, pbrs_version=detected_pbrs_version)
    if env_family == "rware":
        return _rware_pbrs_v2_metadata(candidate)
    if env_family == "lbf":
        if str(candidate.get("pbrs_version") or "") == "lbf_pbrs_v2":
            config = normalize_lbf_pbrs_v2_config(candidate)
        elif any(
            key in candidate
            for key in ("mode", "weights", "active_terms", "gamma")
        ):
            config = normalize_lbf_pbrs_v2_config(
                {
                    "pbrs_version": "lbf_pbrs_v2",
                    "mode": candidate.get("mode"),
                    "beta": candidate.get("beta"),
                    "wc": candidate.get("wc"),
                    "wp": candidate.get("wp"),
                    "active_terms": list(candidate.get("active_terms") or []),
                    "weights": deepcopy(candidate.get("weights") or {}),
                    "gamma": candidate.get("gamma", 0.99),
                }
            )
        else:
            config = _lbf_pbrs_v2_metadata(
                wc=candidate.get("wc", 0.5),
                wp=candidate.get("wp", 1.0 - float(candidate.get("wc", 0.5) or 0.5)),
            )
        active_terms, weights = filter_pbrs_terms_for_env_family(
            env_family="lbf",
            active_terms=config.get("active_terms") or [],
            weights=config.get("weights") or {},
        )
        return {
            "pbrs_version": "lbf_pbrs_v2",
            "mode": str(config.get("mode") or ""),
            "active_terms": active_terms,
            "weights": weights,
        }
    active_terms, weights = filter_pbrs_terms_for_env_family(
        env_family=env_family,
        active_terms=list(candidate.get("active_terms") or []),
        weights=deepcopy(candidate.get("weights") or {}),
    )
    return {
        "pbrs_version": detected_pbrs_version,
        "mode": str(candidate.get("mode") or ""),
        "active_terms": active_terms,
        "weights": weights,
    }


def _attach_env_specific_pbrs_metadata(
    candidate: Dict[str, Any],
    *,
    env_key: Any = None,
    pbrs_version: Any = None,
) -> Dict[str, Any]:
    candidate.update(
        build_env_specific_review_metadata(
            candidate=candidate,
            env_key=env_key,
            pbrs_version=pbrs_version,
        )
    )
    return candidate


def _attach_lbf_pbrs_v2_metadata(candidate: Dict[str, Any]) -> Dict[str, Any]:
    if "wc" not in candidate or "wp" not in candidate:
        return candidate
    candidate.update(
        _lbf_pbrs_v2_metadata(
            wc=candidate.get("wc"),
            wp=candidate.get("wp"),
        )
    )
    return candidate


def _normalize_lbf_stage_config(candidate: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_lbf_pbrs_v2_config(candidate)
    return {
        "pbrs_version": "lbf_pbrs_v2",
        "mode": str(normalized.get("mode") or ""),
        "beta": float(normalized["beta"]),
        "wc": float(normalized["wc"]),
        "wp": float(normalized["wp"]),
        "active_terms": list(normalized.get("active_terms") or []),
        "weights": deepcopy(normalized.get("weights") or {}),
        "gamma": float(normalized.get("gamma", 0.99)),
    }


def _canonicalize_stage3_active_terms(
    active_terms: List[Any] | Tuple[Any, ...] | None,
    *,
    allowed_terms: Tuple[str, ...],
) -> List[str]:
    seen = {
        str(term).strip()
        for term in list(active_terms or [])
        if str(term).strip() in allowed_terms
    }
    return [term for term in allowed_terms if term in seen]


def _floats_match_with_tolerance(left: Any, right: Any, *, tolerance: float) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return abs(left_value - right_value) <= float(tolerance)


def _canonicalize_lbf_stage_config_for_compare(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_lbf_stage_config(config)
    normalized["active_terms"] = _canonicalize_stage3_active_terms(
        normalized.get("active_terms"),
        allowed_terms=LBF_STAGE3_TERMS,
    )
    normalized["weights"] = {
        term: float((normalized.get("weights") or {}).get(term, 0.0))
        for term in LBF_STAGE3_TERMS
    }
    return normalized


def _lbf_stage_configs_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    try:
        left_norm = _canonicalize_lbf_stage_config_for_compare(left)
        right_norm = _canonicalize_lbf_stage_config_for_compare(right)
    except Exception:
        return False
    for key in ("pbrs_version", "mode"):
        if str(left_norm.get(key) or "") != str(right_norm.get(key) or ""):
            return False
    if list(left_norm.get("active_terms") or []) != list(right_norm.get("active_terms") or []):
        return False
    for key in ("beta", "gamma", "wc", "wp"):
        if not _floats_match_with_tolerance(
            left_norm.get(key),
            right_norm.get(key),
            tolerance=LBF_CONFIG_FLOAT_TOLERANCE,
        ):
            return False
    for term in LBF_STAGE3_TERMS:
        if not _floats_match_with_tolerance(
            (left_norm.get("weights") or {}).get(term, 0.0),
            (right_norm.get("weights") or {}).get(term, 0.0),
            tolerance=LBF_CONFIG_FLOAT_TOLERANCE,
        ):
            return False
    return True


def _build_native_pbrs_env_args(config: Dict[str, Any]) -> Dict[str, Any]:
    if str(config.get("pbrs_version") or "") == "rware_pbrs_v2":
        normalized = _normalize_rware_stage_config(
            {
                "candidate_id": str(config.get("candidate_id") or "env_args"),
                "candidate_type": str(config.get("candidate_type") or "adaptive_branch"),
                "pbrs_version": "rware_pbrs_v2",
                "mode": config.get("mode"),
                "beta": config.get("beta"),
                "active_terms": list(config.get("active_terms") or []),
                "weights": deepcopy(config.get("weights") or {}),
                "gamma": config.get("gamma", 0.99),
                "evidence_keys_used": list(config.get("evidence_keys_used") or []),
            }
        )
        return {
            "use_pbrs": True,
            "eval_use_pbrs": False,
            "pbrs_variant": "original",
            "pbrs_gamma": float(normalized.get("gamma", 0.99)),
            "pbrs_version": "rware_pbrs_v2",
            "pbrs_mode": str(normalized.get("mode") or ""),
            "pbrs_active_terms": list(normalized.get("active_terms") or []),
            "pbrs_weights": deepcopy(normalized.get("weights") or {}),
            "pbrs_beta": float(normalized["beta"]),
        }
    env_args = {
        "use_pbrs": True,
        "eval_use_pbrs": False,
        "pbrs_variant": "original",
        "pbrs_gamma": 0.99,
        "pbrs_beta": float(config["beta"]),
        "pbrs_wc": float(config["wc"]),
        "pbrs_wp": float(config["wp"]),
    }
    if str(config.get("pbrs_version") or "") == "lbf_pbrs_v2":
        normalized = normalize_lbf_pbrs_v2_config(config)
        env_args.update(
            {
                "pbrs_version": "lbf_pbrs_v2",
                "pbrs_mode": str(normalized.get("mode") or ""),
                "pbrs_active_terms": list(normalized.get("active_terms") or []),
                "pbrs_weights": deepcopy(normalized.get("weights") or {}),
                "pbrs_beta": float(normalized["beta"]),
                "pbrs_wc": float(normalized["wc"]),
                "pbrs_wp": float(normalized["wp"]),
                "pbrs_gamma": float(normalized.get("gamma", 0.99)),
            }
        )
    return env_args


def _recommended_short_resume_interval(budget_steps: int) -> int:
    budget_steps = max(1, int(budget_steps))
    if budget_steps <= 5000:
        return max(200, budget_steps // 5)
    return min(50000, max(1000, budget_steps // 5))


def _inject_candidate_metadata_into_plan(
    *,
    plan: Dict[str, Any],
    candidate: Dict[str, Any],
) -> None:
    train_config = plan.get("resume_train_config") or {}
    raw_metadata = train_config.get("experiment_metadata")
    metadata: Dict[str, Any] = {}
    if isinstance(raw_metadata, str) and raw_metadata.strip():
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                metadata = parsed
        except Exception:
            metadata = {}
    elif isinstance(raw_metadata, dict):
        metadata = deepcopy(raw_metadata)
    review_metadata = build_env_specific_review_metadata(
        candidate=candidate,
        env_key=(train_config.get("env_args") or {}).get("key") or train_config.get("env_key"),
        pbrs_version=candidate.get("pbrs_version"),
    )
    metadata["stage3_candidate_config"] = {
        "pbrs_version": str(review_metadata.get("pbrs_version") or ""),
        "mode": str(review_metadata.get("mode") or ""),
        "active_terms": list(review_metadata.get("active_terms") or []),
        "weights": deepcopy(review_metadata.get("weights") or {}),
        "beta": float(candidate.get("beta") or 0.0),
    }
    if "wc" in candidate:
        metadata["stage3_candidate_config"]["wc"] = float(candidate.get("wc") or 0.0)
    if "wp" in candidate:
        metadata["stage3_candidate_config"]["wp"] = float(candidate.get("wp") or 0.0)
    train_config["experiment_metadata"] = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    plan["resume_train_config"] = train_config


def _configs_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_version = str(left.get("pbrs_version") or "")
    right_version = str(right.get("pbrs_version") or "")
    if left_version == "rware_pbrs_v2" or right_version == "rware_pbrs_v2":
        try:
            left_norm = _normalize_rware_stage_config(
                {
                    "candidate_id": str(left.get("candidate_id") or "left"),
                    "candidate_type": str(left.get("candidate_type") or "config_compare"),
                    "pbrs_version": left_version or "rware_pbrs_v2",
                    "mode": left.get("mode"),
                    "beta": left.get("beta"),
                    "active_terms": list(left.get("active_terms") or []),
                    "weights": deepcopy(left.get("weights") or {}),
                    "gamma": left.get("gamma", 0.99),
                    "evidence_keys_used": list(left.get("evidence_keys_used") or []),
                },
                require_candidate_type=False,
            )
            right_norm = _normalize_rware_stage_config(
                {
                    "candidate_id": str(right.get("candidate_id") or "right"),
                    "candidate_type": str(right.get("candidate_type") or "config_compare"),
                    "pbrs_version": right_version or "rware_pbrs_v2",
                    "mode": right.get("mode"),
                    "beta": right.get("beta"),
                    "active_terms": list(right.get("active_terms") or []),
                    "weights": deepcopy(right.get("weights") or {}),
                    "gamma": right.get("gamma", 0.99),
                    "evidence_keys_used": list(right.get("evidence_keys_used") or []),
                },
                require_candidate_type=False,
            )
        except Exception:
            return False
        if abs(float(left_norm["beta"]) - float(right_norm["beta"])) > 1e-9:
            return False
        if str(left_norm.get("mode") or "") != str(right_norm.get("mode") or ""):
            return False
        if list(left_norm.get("active_terms") or []) != list(right_norm.get("active_terms") or []):
            return False
        for term in RWARE_STAGE3_TERMS:
            if abs(
                float((left_norm.get("weights") or {}).get(term, 0.0))
                - float((right_norm.get("weights") or {}).get(term, 0.0))
            ) > 1e-9:
                return False
        return True
    if left_version == "lbf_pbrs_v2" or right_version == "lbf_pbrs_v2":
        return _lbf_stage_configs_match(left, right)
    for key in ("beta", "wc", "wp"):
        left_value = _safe_float(left.get(key))
        right_value = _safe_float(right.get(key))
        if left_value is None or right_value is None:
            return False
        if abs(left_value - right_value) > 1e-9:
            return False
    return True


def _candidate_matches_config(candidate: Dict[str, Any], config: Dict[str, Any]) -> bool:
    if is_rware_workflow(
        env_key=config.get("env_key") or candidate.get("env_key"),
        pbrs_version=config.get("pbrs_version") or candidate.get("pbrs_version"),
    ):
        return _configs_match(candidate, config)
    return _configs_match(
        {
            "beta": candidate.get("beta"),
            "wc": candidate.get("wc"),
            "wp": candidate.get("wp"),
        },
        config,
    )


def _load_candidate_metric_series(
    run_dir: Optional[str],
    *,
    preferred_metric: str = "test_sparse_return_mean",
    fallback_metric: str = "test_return_mean",
) -> Dict[str, Any]:
    if not run_dir:
        return {
            "run_dir": run_dir,
            "metric_name": None,
            "steps": [],
            "values": [],
            "load_status": "missing_run_dir",
        }
    attempts: List[Tuple[str, str]] = [
        (preferred_metric, "preferred"),
        (fallback_metric, "fallback"),
    ]
    for metric_name, source in attempts:
        try:
            series = load_metric_series(run_dir, metric_name=metric_name)
        except Exception:
            continue
        steps = [int(step) for step in (series.get("steps") or [])]
        values = [float(value) for value in (series.get("values") or [])]
        if values:
            return {
                "run_dir": run_dir,
                "metric_name": metric_name,
                "steps": steps,
                "values": values,
                "load_status": source,
            }
    return {
        "run_dir": run_dir,
        "metric_name": None,
        "steps": [],
        "values": [],
        "load_status": "metrics_missing",
    }


def _candidate_sync_root(workflow_dir: Optional[Path]) -> Optional[Path]:
    if workflow_dir is None:
        return None
    parent = workflow_dir.parent
    if parent.name in {"adaptive_checkpoint_replacement_workflows", "end_to_end_workflows", "sacred", "results"}:
        return parent.parent
    return parent


def _resolve_synced_result_path(path_value: Optional[str], *, workflow_dir: Optional[Path]) -> Optional[Path]:
    if not isinstance(path_value, str) or not path_value:
        return None
    candidate_path = Path(path_value)
    if candidate_path.exists():
        return candidate_path
    sync_root = _candidate_sync_root(workflow_dir)
    if sync_root is None:
        return None
    try:
        result_index = candidate_path.parts.index("results")
    except ValueError:
        result_index = -1
    if result_index >= 0 and result_index + 1 < len(candidate_path.parts):
        remapped = sync_root.joinpath(*candidate_path.parts[result_index + 1 :])
        if remapped.exists():
            return remapped
    return None


def _candidate_branch_result_payload(
    *,
    candidate: Dict[str, Any],
    workflow_dir: Optional[Path],
) -> tuple[Dict[str, Any], Optional[Path], List[str]]:
    searched_paths: List[str] = []
    embedded = candidate.get("branch_result")
    if isinstance(embedded, dict) and embedded:
        return deepcopy(embedded), None, searched_paths
    branch_result_path = candidate.get("branch_result_path")
    resolved = _resolve_synced_result_path(branch_result_path, workflow_dir=workflow_dir)
    if resolved is not None:
        searched_paths.append(str(resolved))
        return _load_json_if_exists(resolved), resolved, searched_paths
    if isinstance(branch_result_path, str) and branch_result_path:
        searched_paths.append(branch_result_path)
    return {}, None, searched_paths


def _hydrate_stage3_candidate_from_branch_result(
    candidate: Dict[str, Any],
    *,
    workflow_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    hydrated = False
    branch_result, resolved_branch_result_path, searched_paths = _candidate_branch_result_payload(
        candidate=candidate,
        workflow_dir=workflow_dir,
    )
    run_reference = {}
    if isinstance(candidate.get("run_reference"), dict):
        run_reference.update(deepcopy(candidate.get("run_reference") or {}))
    if isinstance(branch_result.get("run_reference"), dict):
        run_reference.update(deepcopy(branch_result.get("run_reference") or {}))
    if run_reference and candidate.get("run_reference") != run_reference:
        candidate["run_reference"] = deepcopy(run_reference)
        hydrated = True

    field_sources = {
        "run_id": run_reference.get("run_id") or branch_result.get("run_id"),
        "run_dir": run_reference.get("run_dir") or branch_result.get("run_dir"),
        "candidate_valid": branch_result.get("candidate_valid"),
        "metrics_missing": branch_result.get("metrics_missing"),
        "branch_result_path": (
            str(resolved_branch_result_path)
            if resolved_branch_result_path is not None
            else branch_result.get("branch_result_path")
        ),
        "endpoint_checkpoint_path": branch_result.get("endpoint_checkpoint_path"),
        "endpoint_checkpoint_step": branch_result.get("actual_checkpoint_step"),
        "status": branch_result.get("status"),
    }
    optional_fields = {
        "checkpoint_root_dir": run_reference.get("checkpoint_root_dir"),
        "available_checkpoint_steps": list(run_reference.get("available_checkpoint_steps") or []),
        "latest_checkpoint_step": run_reference.get("latest_checkpoint_step") or branch_result.get("actual_checkpoint_step"),
        "metrics_source": branch_result.get("metrics_source"),
        "available_metric_keys": list(branch_result.get("available_metric_keys") or []),
        "checkpoint_exists": branch_result.get("checkpoint_exists"),
        "invalid_reason": branch_result.get("invalid_reason"),
    }
    for field_name, field_value in {**field_sources, **optional_fields}.items():
        existing = candidate.get(field_name)
        should_set = existing in (None, "", [], {})
        if field_name in {"candidate_valid", "metrics_missing", "checkpoint_exists"} and existing is None:
            should_set = True
        if should_set and field_value not in (None, "", [], {}):
            candidate[field_name] = deepcopy(field_value)
            hydrated = True
    if branch_result and not isinstance(candidate.get("branch_result"), dict):
        candidate["branch_result"] = deepcopy(branch_result)
        hydrated = True
    if resolved_branch_result_path is not None and candidate.get("branch_result_path") != str(resolved_branch_result_path):
        candidate["branch_result_path"] = str(resolved_branch_result_path)
        hydrated = True
    if searched_paths:
        candidate["branch_result_search_paths"] = searched_paths
    candidate["hydrated_from_branch_result"] = bool(hydrated)
    return candidate


def _resolve_candidate_run_dir(
    *,
    candidate: Dict[str, Any],
    workflow_dir: Optional[Path],
) -> Dict[str, Any]:
    _hydrate_stage3_candidate_from_branch_result(candidate, workflow_dir=workflow_dir)
    searched_paths: List[str] = []
    branch_result = candidate.get("branch_result") if isinstance(candidate.get("branch_result"), dict) else {}
    candidate_run_reference = candidate.get("run_reference") if isinstance(candidate.get("run_reference"), dict) else {}
    branch_run_reference = branch_result.get("run_reference") if isinstance(branch_result.get("run_reference"), dict) else {}
    path_candidates = [
        ("candidate.run_dir", candidate.get("run_dir")),
        ("candidate.run_reference.run_dir", candidate_run_reference.get("run_dir")),
        ("candidate.branch_result.run_reference.run_dir", branch_run_reference.get("run_dir")),
    ]
    for source_name, path_value in path_candidates:
        if not isinstance(path_value, str) or not path_value:
            continue
        searched_paths.append(path_value)
        resolved = _resolve_synced_result_path(path_value, workflow_dir=workflow_dir)
        if resolved is not None:
            candidate["run_dir"] = str(resolved)
            return {
                "run_dir": str(resolved),
                "metrics_source": source_name,
                "searched_paths": searched_paths,
                "hydrated_from_branch_result": bool(candidate.get("hydrated_from_branch_result")),
            }
    result_path = candidate.get("result_path")
    resolved_result_path = _resolve_synced_result_path(result_path, workflow_dir=workflow_dir)
    if resolved_result_path is not None:
        searched_paths.append(str(resolved_result_path))
        result_payload = _load_json_if_exists(resolved_result_path)
        fallback_run_dir = (
            ((result_payload.get("run_reference") or {}).get("run_dir"))
            or result_payload.get("run_dir")
            or ((result_payload.get("metrics_summary") or {}).get("run_metadata") or {}).get("run_dir")
        )
        resolved_fallback_run_dir = _resolve_synced_result_path(
            fallback_run_dir,
            workflow_dir=workflow_dir,
        )
        if resolved_fallback_run_dir is not None:
            candidate["run_dir"] = str(resolved_fallback_run_dir)
            return {
                "run_dir": str(resolved_fallback_run_dir),
                "metrics_source": "candidate.result_path.run_reference.run_dir",
                "searched_paths": searched_paths,
                "hydrated_from_branch_result": bool(candidate.get("hydrated_from_branch_result")),
            }
    branch_result_path = candidate.get("branch_result_path")
    if isinstance(branch_result_path, str) and branch_result_path:
        searched_paths.append(branch_result_path)
    return {
        "run_dir": None,
        "metrics_source": "missing_run_dir_after_hydration",
        "searched_paths": searched_paths,
        "hydrated_from_branch_result": bool(candidate.get("hydrated_from_branch_result")),
    }


def _compute_auc_mean(steps: List[int], values: List[float]) -> Tuple[Optional[float], str]:
    if not values:
        return None, "unavailable"
    if len(values) == 1 or len(steps) <= 1:
        return float(values[-1]), "single_eval_point"
    span = float(steps[-1] - steps[0])
    if span <= 0:
        return _safe_mean([float(value) for value in values]), "mean_over_eval_points"
    area = 0.0
    for index in range(1, len(values)):
        delta_step = float(steps[index] - steps[index - 1])
        if delta_step <= 0:
            continue
        area += 0.5 * (float(values[index - 1]) + float(values[index])) * delta_step
    return float(area / span), "trapezoid_step_weighted_mean"


def _series_slice_for_window(series: Dict[str, Any], *, start_step: int, end_step: int) -> Dict[str, Any]:
    steps = [int(step) for step in (series.get("steps") or [])]
    values = [float(value) for value in (series.get("values") or [])]
    exact_pairs = [(step, value) for step, value in zip(steps, values) if int(start_step) <= step <= int(end_step)]
    if exact_pairs:
        return {
            "steps": [step for step, _ in exact_pairs],
            "values": [value for _, value in exact_pairs],
            "alignment": "exact",
        }
    if not steps or not values:
        return {"steps": [], "values": [], "alignment": "unavailable"}
    start_idx = min(range(len(steps)), key=lambda idx: abs(int(steps[idx]) - int(start_step)))
    end_idx = min(range(len(steps)), key=lambda idx: abs(int(steps[idx]) - int(end_step)))
    left_idx = min(start_idx, end_idx)
    right_idx = max(start_idx, end_idx)
    return {
        "steps": steps[left_idx : right_idx + 1],
        "values": values[left_idx : right_idx + 1],
        "alignment": "nearest_window",
    }


def _summarize_series_for_stability_gate(
    *,
    series: Dict[str, Any],
    last_k_points: int,
    candidate_valid: bool,
    result_record: Dict[str, Any],
) -> Dict[str, Any]:
    steps = [int(step) for step in (series.get("steps") or [])]
    values = [float(value) for value in (series.get("values") or [])]
    eval_count = len(values)
    resolved_k = max(1, min(int(last_k_points), eval_count)) if eval_count else 0
    tail = values[-resolved_k:] if resolved_k else []
    best = float(max(values)) if values else _safe_float(result_record.get("best_test_sparse_return_mean"))
    final = float(values[-1]) if values else _safe_float(result_record.get("last_test_sparse_return_mean"))
    last_k_mean = _safe_mean([float(value) for value in tail]) if tail else None
    last_k_std = _safe_std([float(value) for value in tail]) if tail else None
    auc_mean, auc_method = _compute_auc_mean(steps, values)
    metrics_valid = bool(candidate_valid) and eval_count > 0 and None not in {
        best,
        final,
        last_k_mean,
        auc_mean,
    }
    if last_k_std is None:
        last_k_std = 0.0 if eval_count else None
    spike_gap = None
    if best is not None and last_k_mean is not None:
        spike_gap = float(best - last_k_mean)
    return {
        "metric_name_used": series.get("metric_name"),
        "metric_load_status": series.get("load_status"),
        "best": best,
        "final": final,
        "last_k_mean": last_k_mean,
        "last_k_std": last_k_std,
        "auc_mean": auc_mean,
        "auc_mean_method": auc_method,
        "spike_gap": spike_gap,
        "eval_count": eval_count,
        "low_eval_count": bool(eval_count < 3),
        "metrics_valid": metrics_valid,
        "metrics_missing": not metrics_valid,
    }


def _stage3_gate_config(adaptive_method: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "gate_mode": str(adaptive_method.get("stage3_gate_mode") or "hard_conservative"),
        "selection_metric_mode": str(adaptive_method.get("stage3_selection_metric_mode") or "stability_aware"),
        "last_k_eval_points": int(adaptive_method.get("stage3_last_k_eval_points") or 5),
        "best_metric_weight": float(adaptive_method.get("stage3_best_metric_weight") or 0.10),
        "last_k_mean_weight": float(adaptive_method.get("stage3_last_k_mean_weight") or 0.35),
        "auc_mean_weight": float(adaptive_method.get("stage3_auc_mean_weight") or 0.35),
        "final_weight": float(adaptive_method.get("stage3_final_weight") or 0.20),
        "use_stability_penalty": bool(adaptive_method.get("stage3_use_stability_penalty", True)),
        "stability_penalty_weight": float(adaptive_method.get("stage3_stability_penalty_weight") or 0.10),
        "update_margin_last_k": float(adaptive_method.get("stage3_update_margin_last_k") or 0.03),
        "update_margin_auc": float(adaptive_method.get("stage3_update_margin_auc") or 0.02),
        "final_tolerance": float(adaptive_method.get("stage3_final_tolerance") or 0.02),
        "spike_gap_threshold": float(adaptive_method.get("stage3_spike_gap_threshold") or 0.10),
        "std_tolerance": float(adaptive_method.get("stage3_std_tolerance") or 0.05),
        "tie_tolerance": float(adaptive_method.get("stage3_tie_tolerance") or 0.01),
        "low_return_adaptive_margin": bool(adaptive_method.get("stage3_low_return_adaptive_margin", True)),
        "min_margin_last_k": float(adaptive_method.get("stage3_min_margin_last_k") or 0.01),
        "min_margin_auc": float(adaptive_method.get("stage3_min_margin_auc") or 0.005),
        "relative_margin_ratio": float(adaptive_method.get("stage3_relative_margin_ratio") or 0.10),
        "reference_aware_gate": bool(adaptive_method.get("stage3_reference_aware_gate", True)),
        "reference_lead_threshold": float(adaptive_method.get("stage3_reference_lead_threshold") or 0.05),
        "reference_tolerance": float(adaptive_method.get("stage3_reference_tolerance") or 0.03),
        "reference_extra_margin": float(adaptive_method.get("stage3_reference_extra_margin") or 0.02),
        "first_replacement_extra_margin": float(
            adaptive_method.get("stage3_first_replacement_extra_margin") or 0.02
        ),
        "soft_promotion_margin": float(adaptive_method.get("stage3_soft_promotion_margin") or 0.03),
        "reference_soft_penalty": float(adaptive_method.get("stage3_reference_soft_penalty") or 0.005),
        "first_replacement_soft_penalty": float(
            adaptive_method.get("stage3_first_replacement_soft_penalty") or 0.0
        ),
        "detect_revert_to_stage1b_config": bool(
            adaptive_method.get("stage3_detect_revert_to_stage1b_config", True)
        ),
        "oscillation_detection": bool(adaptive_method.get("stage3_oscillation_detection", True)),
    }


def _is_soft_risk_adjusted_gate_mode(gate_config: Dict[str, Any]) -> bool:
    return str(gate_config.get("gate_mode") or "hard_conservative") == "soft_risk_adjusted"


def _append_unique_reason(reasons: List[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def _stage3_margin_value(
    *,
    base: float,
    minimum: float,
    ratio: float,
    reference_value: Optional[float],
    use_low_return_mode: bool,
) -> float:
    if not use_low_return_mode or reference_value is None:
        return float(base)
    return float(max(float(minimum), float(ratio) * abs(float(reference_value))))


def _stability_gate_reference_lead_threshold(
    *,
    gate_config: Dict[str, Any],
    no_change_last_k_mean: Optional[float],
) -> float:
    if no_change_last_k_mean is None or not bool(gate_config.get("low_return_adaptive_margin")):
        return float(gate_config["reference_lead_threshold"])
    adaptive_threshold = max(
        0.02,
        float(gate_config["relative_margin_ratio"]) * abs(float(no_change_last_k_mean)),
    )
    return float(min(float(gate_config["reference_lead_threshold"]), adaptive_threshold))


def _candidate_stage3_metrics(
    *,
    candidate: Dict[str, Any],
    adaptive_method: Dict[str, Any],
    workflow_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if isinstance(candidate.get("stability_metrics_override"), dict):
        override = deepcopy(candidate.get("stability_metrics_override") or {})
        override.setdefault("metrics_valid", True)
        override.setdefault("metrics_missing", not bool(override.get("metrics_valid")))
        override.setdefault("auc_mean_method", "synthetic_override")
        override.setdefault("metrics_source", "synthetic_override")
        override.setdefault("metrics_invalid_reason", None)
        override.setdefault("hydrated_from_branch_result", False)
        override.setdefault("run_dir_used", candidate.get("run_dir"))
        weighted_score = (
            float(_stage3_gate_config(adaptive_method)["last_k_mean_weight"]) * float(override["last_k_mean"])
            + float(_stage3_gate_config(adaptive_method)["auc_mean_weight"]) * float(override["auc_mean"])
            + float(_stage3_gate_config(adaptive_method)["final_weight"]) * float(override["final"])
            + float(_stage3_gate_config(adaptive_method)["best_metric_weight"]) * float(override["best"])
        )
        instability_penalty = float(override.get("spike_gap") or 0.0) + float(
            override.get("last_k_std") or 0.0
        )
        score = weighted_score
        if bool(_stage3_gate_config(adaptive_method).get("use_stability_penalty")):
            score = float(
                weighted_score
                - float(_stage3_gate_config(adaptive_method)["stability_penalty_weight"])
                * instability_penalty
            )
        return {
            **override,
            "weighted_score": float(weighted_score),
            "instability_penalty": float(instability_penalty),
            "score": float(score),
            "risk_adjusted_score": None,
            "soft_warning_reasons": [],
            "hard_rejection_reasons": [],
            "soft_acceptance_margin": None,
            "accepted_by_soft_gate": False,
            "original_hard_gate_would_reject": False,
        }
    gate_config = _stage3_gate_config(adaptive_method)
    resolved_run = _resolve_candidate_run_dir(candidate=candidate, workflow_dir=workflow_dir)
    result_record = deepcopy(candidate.get("result_record") or {})
    run_dir = resolved_run.get("run_dir")
    series = _load_candidate_metric_series(run_dir)
    metrics = _summarize_series_for_stability_gate(
        series=series,
        last_k_points=int(gate_config["last_k_eval_points"]),
        candidate_valid=bool(candidate.get("candidate_valid")),
        result_record=result_record,
    )
    metrics["metrics_source"] = resolved_run.get("metrics_source") or series.get("load_status")
    metrics["run_dir_used"] = run_dir
    metrics["searched_paths"] = list(resolved_run.get("searched_paths") or [])
    metrics["hydrated_from_branch_result"] = bool(resolved_run.get("hydrated_from_branch_result"))
    if run_dir is None:
        metrics["metrics_valid"] = False
        metrics["metrics_missing"] = True
        metrics["metrics_invalid_reason"] = "missing_run_dir_after_hydration"
    elif not bool(metrics.get("metrics_valid")):
        metrics["metrics_invalid_reason"] = "metrics_missing_or_unreadable"
    else:
        metrics["metrics_invalid_reason"] = None
    if not bool(metrics.get("metrics_valid")):
        return {
            **metrics,
            "weighted_score": None,
            "instability_penalty": None,
            "score": None,
            "risk_adjusted_score": None,
            "soft_warning_reasons": [],
            "hard_rejection_reasons": [],
            "soft_acceptance_margin": None,
            "accepted_by_soft_gate": False,
            "original_hard_gate_would_reject": False,
        }
    weighted_score = (
        float(gate_config["last_k_mean_weight"]) * float(metrics["last_k_mean"])
        + float(gate_config["auc_mean_weight"]) * float(metrics["auc_mean"])
        + float(gate_config["final_weight"]) * float(metrics["final"])
        + float(gate_config["best_metric_weight"]) * float(metrics["best"])
    )
    instability_penalty = float(metrics.get("spike_gap") or 0.0) + float(
        metrics.get("last_k_std") or 0.0
    )
    score = float(weighted_score)
    if bool(gate_config.get("use_stability_penalty")):
        score = float(
            weighted_score
            - float(gate_config["stability_penalty_weight"]) * instability_penalty
        )
    return {
        **metrics,
        "metrics_source": series.get("load_status"),
        "run_dir_used": series.get("run_dir"),
        "weighted_score": float(weighted_score),
        "instability_penalty": float(instability_penalty),
        "score": float(score),
        "risk_adjusted_score": None,
        "soft_warning_reasons": [],
        "hard_rejection_reasons": [],
        "soft_acceptance_margin": None,
        "accepted_by_soft_gate": False,
        "original_hard_gate_would_reject": False,
    }


def _fixed_reference_window_metrics(
    *,
    run_dir: Optional[str],
    checkpoint_step: int,
    branch_budget_steps: int,
    adaptive_method: Dict[str, Any],
) -> Dict[str, Any]:
    if not run_dir:
        return {
            "fixed_dense_reference_available": False,
            "fixed_reference_alignment": "unavailable",
            "fixed_dense_reference_last_k_mean": None,
            "fixed_dense_reference_auc_mean": None,
            "fixed_dense_reference_final": None,
            "fixed_reference_leading": False,
        }
    gate_config = _stage3_gate_config(adaptive_method)
    end_step = int(checkpoint_step) + int(branch_budget_steps)
    series = _load_candidate_metric_series(run_dir)
    windowed = _series_slice_for_window(
        series,
        start_step=int(checkpoint_step),
        end_step=int(end_step),
    )
    metrics = _summarize_series_for_stability_gate(
        series={
            "metric_name": series.get("metric_name"),
            "load_status": series.get("load_status"),
            "steps": windowed.get("steps") or [],
            "values": windowed.get("values") or [],
        },
        last_k_points=int(gate_config["last_k_eval_points"]),
        candidate_valid=True,
        result_record={},
    )
    return {
        "fixed_dense_reference_available": bool(metrics.get("metrics_valid")),
        "fixed_reference_alignment": str(windowed.get("alignment") or "unavailable"),
        "fixed_dense_reference_last_k_mean": metrics.get("last_k_mean"),
        "fixed_dense_reference_auc_mean": metrics.get("auc_mean"),
        "fixed_dense_reference_final": metrics.get("final"),
        "fixed_reference_leading": False,
    }


def _previous_non_stage1b_update_exists(
    *,
    previous_decisions: List[Dict[str, Any]],
    stage1b_config: Dict[str, Any],
) -> bool:
    for previous in previous_decisions:
        winner_config = deepcopy(previous.get("winner_config") or {})
        if winner_config and not _configs_match(winner_config, stage1b_config):
            return True
    return False


def _evaluate_stability_gate(
    *,
    workflow_dir: Optional[Path],
    decision: Dict[str, Any],
    adaptive_method: Dict[str, Any],
    initial_config: Dict[str, Any],
    previous_decisions: List[Dict[str, Any]],
    fixed_reference_run_dir: Optional[str],
    branch_budget_steps: int,
) -> Dict[str, Any]:
    gate_config = _stage3_gate_config(adaptive_method)
    candidates = list(decision.get("candidates") or [])
    candidate_metrics: Dict[str, Dict[str, Any]] = {
        str(candidate.get("candidate_id")): {}
        for candidate in candidates
    }
    metrics_repair_attempted = bool(candidates)
    candidate_metrics_hydrated = False
    for candidate in candidates:
        _hydrate_stage3_candidate_from_branch_result(candidate, workflow_dir=workflow_dir)
        if bool(candidate.get("hydrated_from_branch_result")):
            candidate_metrics_hydrated = True
        metrics = _candidate_stage3_metrics(
            candidate=candidate,
            adaptive_method=adaptive_method,
            workflow_dir=workflow_dir,
        )
        metrics.setdefault("passes_no_change_gate", False)
        metrics.setdefault("passes_stability_gate", False)
        metrics.setdefault("passes_reference_gate", False)
        metrics.setdefault("passes_first_replacement_gate", False)
        metrics.setdefault("rejection_reasons", [])
        metrics.setdefault("soft_warning_reasons", [])
        metrics.setdefault("hard_rejection_reasons", [])
        metrics.setdefault("risk_adjusted_score", None)
        metrics.setdefault("soft_acceptance_margin", None)
        metrics.setdefault("accepted_by_soft_gate", False)
        metrics.setdefault("original_hard_gate_would_reject", False)
        candidate_metrics[str(candidate.get("candidate_id"))] = metrics

    no_change = next((candidate for candidate in candidates if bool(candidate.get("is_no_change"))), None)
    no_change_id = str(no_change.get("candidate_id")) if no_change is not None else None
    no_change_metrics = deepcopy(candidate_metrics.get(no_change_id or "") or {})
    reference_context = _fixed_reference_window_metrics(
        run_dir=fixed_reference_run_dir,
        checkpoint_step=int(decision.get("checkpoint_step") or 0),
        branch_budget_steps=int(branch_budget_steps),
        adaptive_method=adaptive_method,
    )
    lead_threshold = _stability_gate_reference_lead_threshold(
        gate_config=gate_config,
        no_change_last_k_mean=no_change_metrics.get("last_k_mean"),
    )
    valid_updates = [
        candidate
        for candidate in candidates
        if not bool(candidate.get("is_no_change"))
        and bool((candidate_metrics.get(str(candidate.get("candidate_id"))) or {}).get("metrics_valid"))
        and (candidate_metrics.get(str(candidate.get("candidate_id"))) or {}).get("score") is not None
    ]
    valid_updates.sort(
        key=lambda item: float(
            (candidate_metrics.get(str(item.get("candidate_id"))) or {}).get("score") or float("-inf")
        ),
        reverse=True,
    )
    tentative = valid_updates[0] if valid_updates else None
    tentative_id = str(tentative.get("candidate_id")) if tentative is not None else None
    tentative_metrics = deepcopy(candidate_metrics.get(tentative_id or "") or {})

    margin_last_k = _stage3_margin_value(
        base=float(gate_config["update_margin_last_k"]),
        minimum=float(gate_config["min_margin_last_k"]),
        ratio=float(gate_config["relative_margin_ratio"]),
        reference_value=no_change_metrics.get("last_k_mean"),
        use_low_return_mode=bool(gate_config.get("low_return_adaptive_margin")),
    )
    margin_auc = _stage3_margin_value(
        base=float(gate_config["update_margin_auc"]),
        minimum=float(gate_config["min_margin_auc"]),
        ratio=float(gate_config["relative_margin_ratio"]),
        reference_value=no_change_metrics.get("auc_mean"),
        use_low_return_mode=bool(gate_config.get("low_return_adaptive_margin")),
    )
    base_margin_last_k = float(margin_last_k)
    base_margin_auc = float(margin_auc)
    first_replacement_margin_applied = False
    first_replacement_active = bool(
        str(decision.get("checkpoint_name") or "").upper() == "C1"
        or str(decision.get("stage_label") or "") == "post_initialization_transition"
    )
    if tentative is not None and first_replacement_active and _configs_match(
        deepcopy(decision.get("current_config") or {}),
        deepcopy(initial_config or {}),
    ) and not _candidate_matches_config(tentative, deepcopy(initial_config or {})):
        margin_last_k += float(gate_config["first_replacement_extra_margin"])
        margin_auc += float(gate_config["first_replacement_extra_margin"])
        first_replacement_margin_applied = True

    reference_leading = bool(
        bool(gate_config.get("reference_aware_gate"))
        and reference_context.get("fixed_dense_reference_available")
        and reference_context.get("fixed_dense_reference_last_k_mean") is not None
        and no_change_metrics.get("last_k_mean") is not None
        and float(reference_context["fixed_dense_reference_last_k_mean"])
        >= float(no_change_metrics["last_k_mean"]) + float(lead_threshold)
    )
    reference_context["fixed_reference_leading"] = reference_leading
    reference_context["reference_lead_threshold_used"] = float(lead_threshold)
    reference_extra_margin_applied = False
    if reference_leading:
        margin_last_k += float(gate_config["reference_extra_margin"])
        margin_auc += float(gate_config["reference_extra_margin"])
        reference_extra_margin_applied = True

    gate_mode = str(gate_config.get("gate_mode") or "hard_conservative")
    soft_gate_enabled = _is_soft_risk_adjusted_gate_mode(gate_config)

    safety_gate = {
        "gate_mode": gate_mode,
        "final_decision_source": (
            "deterministic_soft_risk_adjusted_gate" if soft_gate_enabled else "deterministic_stability_gate"
        ),
        "selected_candidate_id": no_change_id,
        "selected_action": "no_change",
        "no_change_candidate_id": no_change_id,
        "no_change_defaulted": True,
        "tie_to_no_change": False,
        "spike_only_rejected": False,
        "reference_gate_rejected": False,
        "first_replacement_gate_rejected": False,
        "llm_recommendation_used_for_final_decision": False,
        "override_reason": None,
        "valid_branch_metric_comparison": False,
        "decision_meaningful": False,
        "no_change_defaulted_due_to_invalid_metrics": False,
        "candidate_metrics_hydrated": bool(candidate_metrics_hydrated),
        "metrics_repair_attempted": bool(metrics_repair_attempted),
        "metrics_repair_succeeded": False,
        "no_change_metrics_valid": bool(no_change_metrics.get("metrics_valid")),
        "margin_last_k_used": float(margin_last_k),
        "margin_auc_used": float(margin_auc),
        "base_margin_last_k": float(base_margin_last_k),
        "base_margin_auc": float(base_margin_auc),
        "reference_extra_margin_applied": bool(reference_extra_margin_applied),
        "first_replacement_margin_applied": bool(first_replacement_margin_applied),
        "soft_promotion_margin": float(gate_config["soft_promotion_margin"]),
        "soft_warning_reasons": [],
        "hard_rejection_reasons": [],
        "accepted_by_soft_gate": False,
        "original_hard_gate_would_reject": False,
        "soft_acceptance_margin": None,
    }

    if no_change is None or not bool(no_change_metrics.get("metrics_valid")):
        safety_gate["override_reason"] = "invalid_no_change_metrics"
        safety_gate["no_change_defaulted_due_to_invalid_metrics"] = True
        return {
            "no_change": no_change,
            "no_change_metrics": no_change_metrics,
            "tentative_winner": tentative,
            "tentative_winner_metrics": tentative_metrics,
            "candidate_metrics": candidate_metrics,
            "reference_context": reference_context,
            "safety_gate": safety_gate,
            "deterministic_winner": no_change,
        }

    candidate_metrics[no_change_id]["passes_no_change_gate"] = True
    candidate_metrics[no_change_id]["passes_stability_gate"] = True
    candidate_metrics[no_change_id]["passes_reference_gate"] = True
    candidate_metrics[no_change_id]["passes_first_replacement_gate"] = True
    candidate_metrics[no_change_id]["gate_mode"] = gate_mode
    candidate_metrics[no_change_id]["risk_adjusted_score"] = no_change_metrics.get("score")
    candidate_metrics[no_change_id]["no_change_score"] = no_change_metrics.get("score")
    candidate_metrics[no_change_id]["soft_promotion_margin"] = float(gate_config["soft_promotion_margin"])
    candidate_metrics[no_change_id]["soft_acceptance_margin"] = None
    candidate_metrics[no_change_id]["accepted_by_soft_gate"] = False
    candidate_metrics[no_change_id]["original_hard_gate_would_reject"] = False

    if tentative is None:
        safety_gate["override_reason"] = "no_valid_update_candidate"
        safety_gate["metrics_repair_succeeded"] = True
        safety_gate["no_change_metrics_valid"] = True
        safety_gate["decision_meaningful"] = True
        return {
            "no_change": no_change,
            "no_change_metrics": no_change_metrics,
            "tentative_winner": tentative,
            "tentative_winner_metrics": tentative_metrics,
            "candidate_metrics": candidate_metrics,
            "reference_context": reference_context,
            "safety_gate": safety_gate,
            "deterministic_winner": no_change,
        }

    rejection_reasons: List[str] = []
    soft_warning_reasons: List[str] = []
    hard_rejection_reasons: List[str] = []
    candidate_status = str(tentative.get("status") or "")
    if not bool(tentative.get("candidate_valid")):
        _append_unique_reason(hard_rejection_reasons, "candidate_invalid")
    if candidate_status not in {"completed", "completed_invalid", "failed_invalid"}:
        _append_unique_reason(hard_rejection_reasons, "branch_not_completed")
    if tentative_metrics.get("metrics_missing") or not bool(tentative_metrics.get("metrics_valid")):
        _append_unique_reason(hard_rejection_reasons, "metrics_missing")
    if tentative.get("latest_checkpoint_step") in {None, 0}:
        _append_unique_reason(hard_rejection_reasons, "endpoint_checkpoint_missing")
    if _candidate_matches_config(tentative, deepcopy(decision.get("current_config") or {})):
        _append_unique_reason(hard_rejection_reasons, "duplicate_current_config_non_control")
    passes_no_change_gate = bool(
        tentative_metrics.get("last_k_mean") is not None
        and tentative_metrics.get("auc_mean") is not None
        and tentative_metrics.get("final") is not None
        and tentative_metrics.get("last_k_std") is not None
        and tentative_metrics.get("last_k_mean") >= float(no_change_metrics["last_k_mean"]) + float(margin_last_k)
        and tentative_metrics.get("auc_mean") >= float(no_change_metrics["auc_mean"]) + float(margin_auc)
        and tentative_metrics.get("final") >= float(no_change_metrics["final"]) - float(gate_config["final_tolerance"])
    )
    if not passes_no_change_gate:
        _append_unique_reason(rejection_reasons, "no_change_gate_rejected")
        _append_unique_reason(soft_warning_reasons, "no_change_gate_rejected")
    passes_stability_gate = bool(
        tentative_metrics.get("spike_gap") is not None
        and tentative_metrics.get("last_k_std") is not None
        and tentative_metrics.get("spike_gap") <= float(gate_config["spike_gap_threshold"])
        and tentative_metrics.get("last_k_std")
        <= float(no_change_metrics.get("last_k_std") or 0.0) + float(gate_config["std_tolerance"])
    )
    if not passes_stability_gate:
        _append_unique_reason(rejection_reasons, "stability_gate_rejected")
        _append_unique_reason(soft_warning_reasons, "stability_gate_rejected")
    passes_reference_gate = True
    if bool(gate_config.get("reference_aware_gate")) and reference_leading:
        passes_reference_gate = bool(
            reference_context.get("fixed_dense_reference_last_k_mean") is not None
            and tentative_metrics.get("last_k_mean") is not None
            and tentative_metrics.get("last_k_mean")
            >= float(reference_context["fixed_dense_reference_last_k_mean"])
            - float(gate_config["reference_tolerance"])
        )
        if not passes_reference_gate:
            _append_unique_reason(rejection_reasons, "reference_gate_rejected")
            _append_unique_reason(soft_warning_reasons, "reference_gate_rejected")
            safety_gate["reference_gate_rejected"] = True
    passes_first_replacement_gate = True
    if first_replacement_margin_applied and not passes_no_change_gate:
        passes_first_replacement_gate = False
        _append_unique_reason(rejection_reasons, "first_replacement_gate_rejected")
        _append_unique_reason(soft_warning_reasons, "first_replacement_gate_rejected")
        safety_gate["first_replacement_gate_rejected"] = True
    if (
        tentative_metrics.get("final") is not None
        and no_change_metrics.get("final") is not None
        and tentative_metrics.get("final") < float(no_change_metrics["final"]) - float(gate_config["final_tolerance"])
    ):
        _append_unique_reason(hard_rejection_reasons, "final_severe_under_no_change")
    if (
        tentative_metrics.get("last_k_mean") is not None
        and no_change_metrics.get("last_k_mean") is not None
        and tentative_metrics.get("last_k_mean") < float(no_change_metrics["last_k_mean"]) - float(base_margin_last_k)
    ):
        _append_unique_reason(hard_rejection_reasons, "last_k_mean_severe_under_no_change")
    if (
        tentative_metrics.get("auc_mean") is not None
        and no_change_metrics.get("auc_mean") is not None
        and tentative_metrics.get("auc_mean") < float(no_change_metrics["auc_mean"]) - float(base_margin_auc)
    ):
        _append_unique_reason(hard_rejection_reasons, "auc_mean_severe_under_no_change")
    reference_soft_penalty = (
        float(gate_config["reference_soft_penalty"]) if not passes_reference_gate else 0.0
    )
    first_replacement_soft_penalty = (
        float(gate_config["first_replacement_soft_penalty"]) if not passes_first_replacement_gate else 0.0
    )
    severe_warning_penalty = 0.0
    risk_adjusted_score = (
        None
        if tentative_metrics.get("score") is None
        else float(tentative_metrics.get("score"))
        - float(reference_soft_penalty)
        - float(first_replacement_soft_penalty)
        - float(severe_warning_penalty)
    )
    soft_acceptance_margin = (
        None
        if risk_adjusted_score is None or no_change_metrics.get("score") is None
        else float(risk_adjusted_score)
        - (float(no_change_metrics["score"]) + float(gate_config["soft_promotion_margin"]))
    )
    hard_gate_would_accept = bool(
        not hard_rejection_reasons
        and tentative_metrics.get("score") is not None
        and passes_no_change_gate
        and passes_stability_gate
        and passes_reference_gate
        and passes_first_replacement_gate
    )
    tie_to_no_change = bool(
        risk_adjusted_score is None
        or no_change_metrics.get("score") is None
        or (
            float(risk_adjusted_score)
            <= float(no_change_metrics["score"])
            + (
                float(gate_config["soft_promotion_margin"])
                if soft_gate_enabled
                else float(gate_config["tie_tolerance"])
            )
        )
    )
    if tie_to_no_change:
        _append_unique_reason(rejection_reasons, "tie_to_no_change")
        safety_gate["tie_to_no_change"] = True
    if (
        tentative_metrics.get("best") is not None
        and no_change_metrics.get("best") is not None
        and tentative_metrics.get("best") > float(no_change_metrics["best"])
        and not passes_no_change_gate
    ):
        safety_gate["spike_only_rejected"] = True

    candidate_metrics[tentative_id]["passes_no_change_gate"] = passes_no_change_gate
    candidate_metrics[tentative_id]["passes_stability_gate"] = passes_stability_gate
    candidate_metrics[tentative_id]["passes_reference_gate"] = passes_reference_gate
    candidate_metrics[tentative_id]["passes_first_replacement_gate"] = passes_first_replacement_gate
    candidate_metrics[tentative_id]["rejection_reasons"] = rejection_reasons
    candidate_metrics[tentative_id]["soft_warning_reasons"] = soft_warning_reasons
    candidate_metrics[tentative_id]["hard_rejection_reasons"] = hard_rejection_reasons
    candidate_metrics[tentative_id]["gate_mode"] = gate_mode
    candidate_metrics[tentative_id]["risk_adjusted_score"] = risk_adjusted_score
    candidate_metrics[tentative_id]["no_change_score"] = no_change_metrics.get("score")
    candidate_metrics[tentative_id]["soft_promotion_margin"] = float(gate_config["soft_promotion_margin"])
    candidate_metrics[tentative_id]["soft_acceptance_margin"] = soft_acceptance_margin
    candidate_metrics[tentative_id]["accepted_by_soft_gate"] = False
    candidate_metrics[tentative_id]["original_hard_gate_would_reject"] = not hard_gate_would_accept

    selected_candidate = no_change
    accepted_by_soft_gate = False
    if soft_gate_enabled:
        accepted_by_soft_gate = bool(
            not hard_rejection_reasons
            and not tie_to_no_change
            and soft_acceptance_margin is not None
            and float(soft_acceptance_margin) > 0.0
        )
    hard_accept = bool(hard_gate_would_accept and not tie_to_no_change)
    if hard_accept or accepted_by_soft_gate:
        selected_candidate = tentative
        safety_gate["selected_candidate_id"] = tentative_id
        safety_gate["selected_action"] = "update"
        safety_gate["no_change_defaulted"] = False
        safety_gate["override_reason"] = None
        safety_gate["accepted_by_soft_gate"] = bool(accepted_by_soft_gate)
    else:
        final_reasons = hard_rejection_reasons if hard_rejection_reasons else rejection_reasons
        safety_gate["override_reason"] = ",".join(final_reasons) if final_reasons else "insufficient_evidence"
    candidate_metrics[tentative_id]["accepted_by_soft_gate"] = bool(accepted_by_soft_gate)
    safety_gate["soft_warning_reasons"] = soft_warning_reasons
    safety_gate["hard_rejection_reasons"] = hard_rejection_reasons
    safety_gate["original_hard_gate_would_reject"] = not hard_gate_would_accept
    safety_gate["soft_acceptance_margin"] = soft_acceptance_margin

    safety_gate["metrics_repair_succeeded"] = True
    safety_gate["no_change_metrics_valid"] = True
    safety_gate["valid_branch_metric_comparison"] = True
    safety_gate["decision_meaningful"] = True

    return {
        "no_change": no_change,
        "no_change_metrics": no_change_metrics,
        "candidate_metrics": candidate_metrics,
        "reference_context": reference_context,
        "safety_gate": safety_gate,
        "deterministic_winner": selected_candidate,
        "tentative_winner": tentative,
        "tentative_winner_metrics": tentative_metrics,
    }


def _candidate_generation_cache_path(workflow_dir: Path, decision_index: int, stage_label: str) -> Path:
    return _decision_dir(workflow_dir, decision_index, stage_label) / "llm_candidate_generation.json"


def _stage3_artifact_path(workflow_dir: Path, decision: Dict[str, Any], round_id: int) -> Path:
    return _decision_dir(
        workflow_dir,
        int(decision["decision_index"]),
        str(decision["stage_label"]),
    ) / f"llm_candidate_generation_round{int(round_id)}.json"


def _stage3_ledger_context(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    adaptive_method: Dict[str, Any],
    model: str,
    base_url: str,
) -> Dict[str, Any]:
    return {
        "workflow_dir": workflow_dir,
        "workflow_id": str(decision.get("workflow_id") or workflow_dir.name),
        "profile": str(adaptive_method.get("workflow_profile") or adaptive_method.get("profile") or ""),
        "env_key": str(decision.get("env_key") or ""),
        "algorithm": str(decision.get("train_config") or ""),
        "stage": f"stage3_{str(decision.get('checkpoint_name') or '').lower()}",
        "checkpoint": str(decision.get("checkpoint_name") or ""),
        "round_id": int(decision.get("active_round_id") or 1),
        "model": str(model or ""),
        "provider": str(((adaptive_method.get("llm_routing") or {}).get("provider")) or ""),
        "base_url": str(((adaptive_method.get("llm_routing") or {}).get("base_url")) or base_url or ""),
    }


def _persist_stage3_candidate_generation_artifact(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    payload_hash: str | None,
    backend_metadata: Dict[str, Any] | None,
    raw_text: str,
    parsed_candidates_count: int,
    valid_candidates_count: int,
    validation_errors: List[str],
    repair_retry_used: bool,
    repair_retry_count: int,
    fallback_used: bool,
    fallback_reason: str | None,
    candidate_generation_source: str,
    final_source: str,
    ledger_event_refs: List[str],
    original_valid_count: Optional[int] = None,
    topup_count: int = 0,
    repair_status: Optional[str] = None,
    repair_type: Optional[str] = None,
    deterministic_fallback_used: bool = False,
) -> str:
    round_id = int(decision.get("active_round_id") or 1)
    artifact_path = _stage3_artifact_path(workflow_dir, decision, round_id)
    raw_excerpt = str(raw_text or "")[:1000]
    artifact_payload = {
        "workflow_id": str(decision.get("workflow_id") or workflow_dir.name),
        "decision_index": int(decision.get("decision_index") or 0),
        "checkpoint_name": str(decision.get("checkpoint_name") or ""),
        "round_id": round_id,
        "candidate_generation_source": str(candidate_generation_source or ""),
        "final_source": str(final_source or ""),
        "payload_hash": payload_hash,
        "backend_metadata": deepcopy(backend_metadata or {}),
        "raw_text_excerpt": raw_excerpt,
        "parsed_candidates_count": int(parsed_candidates_count),
        "valid_candidates_count": int(valid_candidates_count),
        "validation_errors": list(validation_errors or []),
        "repair_retry_used": bool(repair_retry_used),
        "repair_retry_count": int(repair_retry_count or 0),
        "fallback_used": bool(fallback_used),
        "fallback_reason": None if fallback_reason in (None, "") else str(fallback_reason),
        "original_valid_count": (
            None if original_valid_count is None else int(original_valid_count)
        ),
        "topup_count": int(topup_count or 0),
        "repair_status": None if repair_status in (None, "") else str(repair_status),
        "repair_type": None if repair_type in (None, "") else str(repair_type),
        "deterministic_fallback_used": bool(deterministic_fallback_used),
        "timestamp": utc_now_iso(),
        "ledger_record_ids": list(ledger_event_refs or []),
        "ledger_event_refs": list(ledger_event_refs or []),
    }
    _save_json(artifact_path, artifact_payload)
    decision["llm_candidate_generation_artifact_path"] = str(artifact_path)
    return str(artifact_path)


def _result_diagnosis_cache_path(workflow_dir: Path, decision_index: int, stage_label: str) -> Path:
    return _decision_dir(workflow_dir, decision_index, stage_label) / "llm_result_diagnosis.json"


def _stable_payload_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _build_adaptive_final_spec(
    *,
    selected_checkpoints: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if any(str((decision.get("winner_config") or {}).get("pbrs_version") or "") == "rware_pbrs_v2" for decision in decisions):
        stage_pbrs_configs: Dict[str, Dict[str, Any]] = {}
        for decision in decisions:
            if not bool(decision.get("completed")):
                continue
            stage_label = str(decision.get("stage_label") or "")
            winner_config = deepcopy(decision.get("winner_config") or {})
            if not stage_label or str(winner_config.get("pbrs_version") or "") != "rware_pbrs_v2":
                continue
            stage_pbrs_configs[stage_label] = _normalize_rware_stage_config(
                {
                    "candidate_id": str(winner_config.get("candidate_id") or f"{stage_label}_winner"),
                    "candidate_type": str(winner_config.get("candidate_type") or "winner"),
                    "pbrs_version": "rware_pbrs_v2",
                    "mode": winner_config.get("mode"),
                    "beta": winner_config.get("beta"),
                    "active_terms": list(winner_config.get("active_terms") or []),
                    "weights": deepcopy(winner_config.get("weights") or {}),
                    "gamma": winner_config.get("gamma", 0.99),
                    "evidence_keys_used": list(winner_config.get("evidence_keys_used") or []),
                }
            )
        return {
            "workflow_kind": "adaptive_checkpoint_replacement_final_spec",
            "source_manifest_workflow_id": "adaptive_checkpoint_replacement",
            "pbrs_version": "rware_pbrs_v2",
            "selected_checkpoints": deepcopy(selected_checkpoints),
            "stage_pbrs_configs": stage_pbrs_configs,
            "field_round_summaries": [],
            "stage_reward_specs": {},
            "native_original_pbrs_stage_schedule": None,
        }
    if any(str((decision.get("winner_config") or {}).get("pbrs_version") or "") == "lbf_pbrs_v2" for decision in decisions):
        stage_pbrs_configs: Dict[str, Dict[str, Any]] = {}
        for decision in decisions:
            if not bool(decision.get("completed")):
                continue
            stage_label = str(decision.get("stage_label") or "")
            winner_config = deepcopy(decision.get("winner_config") or {})
            if not stage_label or str(winner_config.get("pbrs_version") or "") != "lbf_pbrs_v2":
                continue
            stage_pbrs_configs[stage_label] = _normalize_lbf_stage_config(winner_config)
        return {
            "workflow_kind": "adaptive_checkpoint_replacement_final_spec",
            "source_manifest_workflow_id": "adaptive_checkpoint_replacement",
            "pbrs_version": "lbf_pbrs_v2",
            "selected_checkpoints": deepcopy(selected_checkpoints),
            "stage_pbrs_configs": stage_pbrs_configs,
            "field_round_summaries": [],
            "stage_reward_specs": {},
            "native_original_pbrs_stage_schedule": None,
        }

    stage_selection_manifest = {
        "workflow_id": "adaptive_checkpoint_replacement",
        "readiness": {"ready_for_final_synthesis": True},
        "field_rounds": ["beta", "wc"],
        "stage_selection_result": {
            "selection": {
                "selected_checkpoints": deepcopy(selected_checkpoints),
            }
        },
        "workflow_spec": {
            "pbrs_tuning": {
                "base_pbrs": {
                    "enabled": True,
                    "variant": "original",
                    "gamma": 0.99,
                    "beta": 0.5,
                    "wc": 0.5,
                    "wp": 0.5,
                }
            }
        },
        "rounds": [],
        "field_round_summaries": [],
    }

    beta_by_stage: Dict[str, Any] = {}
    wc_by_stage: Dict[str, Any] = {}
    for decision in decisions:
        if not bool(decision.get("completed")):
            continue
        stage_label = str(decision.get("stage_label"))
        winner_config = decision.get("winner_config") or {}
        beta_by_stage[stage_label] = winner_config.get("beta")
        wc_by_stage[stage_label] = winner_config.get("wc")

    for field_name, mapping in (("beta", beta_by_stage), ("wc", wc_by_stage)):
        stage_selection_manifest["rounds"].append(
            {
                "field_name": field_name,
                "field_recommendation": {
                    "recommended_values_by_stage": deepcopy(mapping),
                },
                "field_round_summary": {
                    "field_name": field_name,
                    "field_recommendation": {
                        "recommended_values_by_stage": deepcopy(mapping),
                    },
                    "readiness": {"ready_for_final_synthesis": True},
                },
            }
        )

    stage_selection_manifest["field_round_summaries"] = [
        round_payload["field_round_summary"] for round_payload in stage_selection_manifest["rounds"]
    ]
    return synthesize_final_stage_conditioned_spec(stage_selection_manifest)


def _build_candidate_configs(
    *,
    current_config: Dict[str, float],
    candidate_generation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    beta_values = list(candidate_generation.get("beta_values") or [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    wc_values = list(candidate_generation.get("wc_values") or [0.1, 0.3, 0.5, 0.7, 0.9])
    beta_radius = max(0, int(candidate_generation.get("beta_neighbor_radius") or 1))
    wc_radius = max(0, int(candidate_generation.get("wc_neighbor_radius") or 1))
    include_diagonal = bool(candidate_generation.get("include_diagonal_pairs", True))
    include_no_change = bool(candidate_generation.get("include_no_change", True))
    max_candidates = max(1, int(candidate_generation.get("max_candidates_per_checkpoint") or 9))

    current_beta = float(current_config["beta"])
    current_wc = float(current_config["wc"])
    beta_index = min(range(len(beta_values)), key=lambda idx: abs(float(beta_values[idx]) - current_beta))
    wc_index = min(range(len(wc_values)), key=lambda idx: abs(float(wc_values[idx]) - current_wc))

    candidate_pairs: List[Tuple[float, float]] = []
    if include_no_change:
        candidate_pairs.append((float(beta_values[beta_index]), float(wc_values[wc_index])))

    for delta in range(-beta_radius, beta_radius + 1):
        idx = beta_index + delta
        if 0 <= idx < len(beta_values):
            pair = (float(beta_values[idx]), float(wc_values[wc_index]))
            if pair not in candidate_pairs:
                candidate_pairs.append(pair)

    for delta in range(-wc_radius, wc_radius + 1):
        idx = wc_index + delta
        if 0 <= idx < len(wc_values):
            pair = (float(beta_values[beta_index]), float(wc_values[idx]))
            if pair not in candidate_pairs:
                candidate_pairs.append(pair)

    if include_diagonal:
        for beta_delta in range(-beta_radius, beta_radius + 1):
            beta_idx = beta_index + beta_delta
            if not (0 <= beta_idx < len(beta_values)):
                continue
            for wc_delta in range(-wc_radius, wc_radius + 1):
                wc_idx = wc_index + wc_delta
                if not (0 <= wc_idx < len(wc_values)):
                    continue
                pair = (float(beta_values[beta_idx]), float(wc_values[wc_idx]))
                if pair not in candidate_pairs:
                    candidate_pairs.append(pair)

    candidate_payloads: List[Dict[str, Any]] = []
    for beta, wc in candidate_pairs[:max_candidates]:
        candidate_payloads.append(
            {
                "candidate_id": _candidate_id(beta, wc),
                "beta": beta,
                "wc": wc,
                "wp": round(1.0 - float(wc), 10),
                "is_no_change": abs(beta - current_beta) < 1e-12 and abs(wc - current_wc) < 1e-12,
            }
        )
    return candidate_payloads


def _deterministic_mock_candidate_generation(
    *,
    current_config: Dict[str, float],
    candidate_generation: Dict[str, Any],
) -> Dict[str, Any]:
    candidates = _build_candidate_configs(
        current_config=current_config,
        candidate_generation=candidate_generation,
    )
    return {
        "stage_diagnosis": {
            "stage_type": "uncertain",
            "main_failure_mode": "mock_llm_mode",
            "evidence_summary": "deterministic fallback-style mock candidate generation",
        },
        "search_policy": {
            "radius": "conservative",
            "reason": "mock_llm_mode deterministic local neighborhood",
        },
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "beta": candidate["beta"],
                "wc": candidate["wc"],
                "wp": candidate["wp"],
                "candidate_type": (
                    "conservative_neighbor" if not candidate["is_no_change"] else "reference_like"
                ),
                "expected_effect": "mock local exploration",
                "rationale": "mock_llm_mode deterministic local candidate",
            }
            for candidate in candidates
        ]
    }


def _build_candidate_generation_payload(
    *,
    decision: Dict[str, Any],
    current_config: Dict[str, float],
    candidate_generation: Dict[str, Any],
    stage_selection_result: Dict[str, Any],
    dense_reference_selection: Dict[str, Any],
    dense_reference_run_summary: Dict[str, Any],
    branch_budget_steps: int,
    fixed_reference_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    active_round_id = decision.get("active_round_id")
    policy_guidance_payload = {}
    if active_round_id is not None and int(active_round_id) >= 1:
        round_guidance = deepcopy(
            (_round_payload(decision, int(active_round_id)).get("policy_guidance") or {})
        )
        if bool(round_guidance.get("policy_guidance_used")) and round_guidance.get(
            "integrated_guidance_card"
        ):
            policy_guidance_payload = round_guidance
    reference_comparison = _build_reference_comparison_context(
        decision=decision,
        stage_selection_result=stage_selection_result,
        fixed_reference_context=deepcopy(fixed_reference_context or {}),
    )
    payload = {
        "decision_index": decision.get("decision_index"),
        "stage_label": decision.get("stage_label"),
        "checkpoint_name": decision.get("checkpoint_name"),
        "checkpoint_step": decision.get("checkpoint_step"),
        "round_id": active_round_id,
        "current_mainline_config": deepcopy(current_config),
        "initial_dense_config": {
            "beta": dense_reference_selection.get("selected_beta"),
            "wc": dense_reference_selection.get("selected_wc"),
            "wp": dense_reference_selection.get("selected_wp"),
        },
        "stage_selection_summary": {
            "selection_source": stage_selection_result.get("selection_source"),
            "selected_checkpoints": deepcopy(((stage_selection_result.get("selection") or {}).get("selected_checkpoints")) or []),
        },
        "dense_reference_run_summary": deepcopy(dense_reference_run_summary),
        "sparse_reference_run_summary": deepcopy(stage_selection_result.get("sparse_summary") or {}),
        "round1_result_summary": deepcopy(decision.get("round1_result_summary") or {}),
        "reference_comparison": reference_comparison,
        "candidate_generation_constraints": {
            "beta_range": [0.0, 1.0],
            "wc_range": [0.0, 1.0],
            "wp_rule": "wp = 1 - wc",
            "must_include_no_change": False,
            "controller_inserts_no_change_control": True,
            "update_candidates_requested": int(_decision_round_config(decision)["round1_generated"] if active_round_id <= 1 else _decision_round_config(decision)["round2_generated"]),
            "local_neighbor_hint": {
                "beta_neighbor_radius": int(candidate_generation.get("beta_neighbor_radius") or 1),
                "wc_neighbor_radius": int(candidate_generation.get("wc_neighbor_radius") or 1),
                "beta_values": list(candidate_generation.get("beta_values") or []),
                "wc_values": list(candidate_generation.get("wc_values") or []),
            },
            "branch_budget_steps": int(branch_budget_steps),
            "nontrivial_delta_rule": {
                "each_candidate_must_clear_one_of": [
                    "mode_changed",
                    "beta_delta >= 0.10",
                    "max_core_weight_delta >= 0.10",
                    "weights_l1_delta >= 0.20",
                    "active_terms_changed"
                ]
            },
        },
    }
    if policy_guidance_payload:
        payload["policy_guidance"] = _clean_stage3_policy_guidance_payload_for_generator(
            policy_guidance_payload=policy_guidance_payload,
            current_config=current_config,
            decision=decision,
        )
    if str(decision.get("pbrs_version") or "") == "lbf_pbrs_v2":
        payload["generator_output_requirements"] = {
            "pbrs_version": "lbf_pbrs_v2",
            "required_candidate_fields": [
                "pbrs_version",
                "mode",
                "beta",
                "active_terms",
                "weights",
                "candidate_type",
                "evidence_keys_used",
            ],
            "active_terms_allowed": ["col", "app", "cov", "ready", "alloc", "stab"],
            "weights_must_include": ["col", "app", "cov", "ready", "alloc", "stab"],
            "legacy_beta_wc_wp_only_candidates_forbidden": True,
        }
    return payload


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _map_stage3_lbf_v2_candidate_types(candidate_types: List[Any]) -> List[str]:
    mapped: List[str] = []
    for raw_item in list(candidate_types or []):
        text = str(raw_item or "").strip().lower()
        if not text:
            continue
        if "coverage" in text:
            mapped.append("coverage_recovery")
        elif any(token in text for token in ("collect", "geometry", "approach")):
            mapped.append("collection_geometry")
        elif any(token in text for token in ("alloc", "concentration", "rebalance")):
            mapped.append("allocation_rebalance")
        elif any(token in text for token in ("stable", "stability", "oscillation")):
            mapped.append("stability_recovery")
        elif any(token in text for token in ("progress", "transition", "ready")):
            mapped.append("progress_shift")
        elif "reference" in text or "conservative" in text:
            mapped.append("reference_like")
    return _dedupe_preserve_order(mapped)


def _clean_stage3_policy_guidance_payload_for_generator(
    *,
    policy_guidance_payload: Dict[str, Any],
    current_config: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    cleaned = deepcopy(policy_guidance_payload or {})
    if str(decision.get("pbrs_version") or "") != "lbf_pbrs_v2":
        return cleaned
    card = deepcopy(cleaned.get("integrated_guidance_card") or {})
    if not isinstance(card, dict):
        return cleaned
    guidance = deepcopy(card.get("candidate_generation_guidance") or {})
    evidence_keys = _dedupe_preserve_order(
        list(((card.get("policy_diagnosis") or {}).get("behavior_evidence")) or [])
    )[:4]
    current_terms = list(current_config.get("active_terms") or [])
    if not current_terms:
        current_terms = ["col", "app", "cov", "ready", "alloc", "stab"]
    mapped_candidate_types = _map_stage3_lbf_v2_candidate_types(
        list(guidance.get("candidate_types") or [])
    )
    default_candidate_types = [
        "coverage_recovery",
        "progress_shift",
        "allocation_rebalance",
        "stability_recovery",
        "reference_like",
    ]
    mapped_candidate_types = _dedupe_preserve_order(
        mapped_candidate_types + default_candidate_types
    )
    existing_constraints = [
        item
        for item in list(guidance.get("constraints") or [])
        if not any(token in str(item or "").lower() for token in ("wc", "wp", "beta/wc/wp"))
    ]
    mode = str(current_config.get("mode") or "")
    beta = current_config.get("beta")
    guidance.pop("wc_direction", None)
    guidance.pop("wp_direction", None)
    guidance["candidate_types"] = mapped_candidate_types[:4]
    guidance["constraints"] = _dedupe_preserve_order(
        existing_constraints
        + [
            "Output only clean lbf_pbrs_v2 update candidates.",
            "Use only col/app/cov/ready/alloc/stab as active_terms.",
            "Return pbrs_version, mode, beta, active_terms, weights, candidate_type, evidence_keys_used for every candidate.",
            "Do not return legacy beta/wc/wp-only candidates or explicit wc/wp search plans.",
            "Keep updates nontrivial relative to the current mainline configuration.",
        ]
    )[:8]
    guidance["reward_search_implications"] = _dedupe_preserve_order(
        [
            (
                f"Treat current lbf_pbrs_v2 mode={mode or 'unknown'} beta={beta} as the local anchor; "
                f"vary only valid v2 outputs over active_terms/weights."
            ),
            (
                "Express search in v2 terms only: col/app/cov/ready/alloc/stab, not raw wc/wp discovery."
            ),
            (
                "At least one candidate should respond mechanistically to observed evidence via candidate_type "
                "and evidence_keys_used."
            ),
            (
                "Weights must be a full six-term map even if some terms remain zero."
            ),
        ]
    )[:6]
    guidance["v2_term_focus"] = [term for term in current_terms if term in {"col", "app", "cov", "ready", "alloc", "stab"}]
    guidance["v2_output_contract"] = {
        "required_candidate_fields": [
            "pbrs_version",
            "mode",
            "beta",
            "active_terms",
            "weights",
            "candidate_type",
            "evidence_keys_used",
        ],
        "allowed_active_terms": ["col", "app", "cov", "ready", "alloc", "stab"],
        "weights_must_include": ["col", "app", "cov", "ready", "alloc", "stab"],
        "legacy_wc_wp_search_removed": True,
    }
    card["candidate_generation_guidance"] = guidance
    if evidence_keys:
        cleaned["policy_guidance_evidence_keys"] = evidence_keys
    cleaned["integrated_guidance_card"] = card
    return cleaned


def _stage3_policy_guidance_enabled(adaptive_method: Dict[str, Any]) -> bool:
    return bool(adaptive_method.get("enable_policy_guidance", False)) and bool(
        adaptive_method.get("policy_guidance_stage3_use_integrated_guidance", True)
    )


def _decision_policy_guidance_dir(
    workflow_dir: Path,
    decision: Dict[str, Any],
) -> Path:
    return _decision_dir(
        workflow_dir,
        int(decision["decision_index"]),
        str(decision["stage_label"]),
    ) / "policy_guidance"


def _policy_guidance_mock_context(
    adaptive_method: Dict[str, Any],
    *,
    checkpoint_name: str,
    key: str,
) -> Dict[str, Any]:
    return deepcopy(
        (((adaptive_method.get("policy_guidance_mock_context") or {}).get(str(checkpoint_name))) or {}).get(key)
        or {}
    )


def _load_policy_guidance_summary_from_path(path_like: str) -> Dict[str, Any]:
    if not path_like:
        return {}
    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        return {}
    try:
        payload = _load_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_policy_guidance_behavior_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    direct = deepcopy(payload.get("behavior_summary") or {})
    if direct:
        return direct
    milestones = list(payload.get("milestones") or [])
    for item in reversed(milestones):
        summary = deepcopy((item or {}).get("behavior_summary") or {})
        if summary:
            return summary
    if payload.get("evidence_flags") or payload.get("behavior_signals"):
        return deepcopy(payload)
    return {}


def _discover_stage3_behavior_summary_path(
    *,
    workflow_dir: Path,
    adaptive_method: Dict[str, Any],
) -> str:
    configured = str(adaptive_method.get("policy_guidance_stage1_behavior_summary_path") or "")
    if configured:
        return configured
    workflow_name = str(workflow_dir.name)
    if workflow_name.endswith("_stage234"):
        workflow_name = workflow_name[: -len("_stage234")]
    candidate = (
        workflow_dir.resolve().parents[1]
        / "end_to_end_workflows"
        / workflow_name
        / "stage1_sparse_policy_behavior_summary.json"
    )
    return str(candidate) if candidate.exists() else ""


def _policy_guidance_reward_metrics(
    *,
    decision: Dict[str, Any],
    dense_reference_run_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "current_test_sparse_return_mean": (
            (decision.get("round1_result_summary") or {}).get("ranking", [{}])[0].get("score", [None])[0]
            if decision.get("round1_result_summary")
            else None
        ),
        "dense_reference_context_available": bool(dense_reference_run_summary),
        "performance_issue": f"stage3 {decision.get('checkpoint_name')} policy-guidance hook",
    }


def _stage3_hook_call_record(
    *,
    guidance_id: str,
    intervention_point: str,
    checkpoint_id: str,
    checkpoint_step: int,
    round_id: int,
    summary_path: str,
    hook_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "guidance_id": guidance_id,
        "intervention_point": intervention_point,
        "input_behavior_summary_path": str(summary_path or ""),
        "integrated_guidance_card": deepcopy(hook_result.get("integrated_guidance_card") or {}),
        "used_by_critic": True,
        "used_by_generator": True,
        "fallback_used": not bool(hook_result.get("policy_guidance_used", False)),
        "failure_reason": str(hook_result.get("policy_guidance_failure_reason") or ""),
        "metadata": {
            "checkpoint_id": str(checkpoint_id),
            "checkpoint_step": int(checkpoint_step),
            "round_id": int(round_id),
            "evidence_keys": list(hook_result.get("policy_guidance_evidence_keys") or []),
            "production_hook": True,
        },
    }


def _maybe_attach_stage3_policy_guidance(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    round_payload: Dict[str, Any],
    round_id: int,
    current_config: Dict[str, float],
    adaptive_method: Dict[str, Any],
    dense_reference_run_summary: Dict[str, Any],
    memory: Dict[str, Any],
) -> Dict[str, Any]:
    default_policy_guidance = {
        "policy_guidance_used": False,
        "policy_guidance_source": "reward_only_fallback",
        "policy_guidance_fallback_to_reward_only": True,
        "policy_guidance_evidence_keys": [],
        "policy_guidance_failure_reason": "policy guidance unavailable",
        "integrated_guidance_card_path": None,
        "integrated_guidance_card": {},
        "behavior_summary_available": False,
        "behavior_summary_source": "unavailable",
        "intervention_point": (
            f"stage3_{str(decision.get('checkpoint_name') or '').lower()}_pre_checkpoint"
            if int(round_id) == 1
            else f"stage3_{str(decision.get('checkpoint_name') or '').lower()}_after_round1"
        ),
    }
    round_payload["policy_guidance"] = deepcopy(default_policy_guidance)
    if not _stage3_policy_guidance_enabled(adaptive_method):
        round_payload["policy_guidance"]["policy_guidance_failure_reason"] = "policy guidance disabled"
        return memory

    checkpoint_id = str(decision.get("checkpoint_name") or "")
    checkpoint_step = int(decision.get("checkpoint_step") or 0)
    guidance_dir = _decision_policy_guidance_dir(workflow_dir, decision)
    guidance_dir.mkdir(parents=True, exist_ok=True)
    use_real_llm = bool(adaptive_method.get("use_real_llm", False))
    guidance_mock_mode = bool(adaptive_method.get("mock_llm_mode", False) or not use_real_llm)
    existing_card_path = None
    if int(round_id) == 1:
        existing_card_path = decision.get("pre_checkpoint_integrated_guidance_card_path")
    else:
        existing_card_path = decision.get("after_round1_integrated_guidance_card_path")
    if existing_card_path and Path(str(existing_card_path)).exists():
        existing_card = _load_json(Path(str(existing_card_path)))
        hook_used = bool(
            existing_card.get("use_in_reward_generation", False)
            or existing_card.get("fallback_to_reward_only", True) is False
        )
        round_payload["policy_guidance"] = {
            **deepcopy(default_policy_guidance),
            **{
                "policy_guidance_used": hook_used,
                "policy_guidance_source": "cached_resume",
                "policy_guidance_fallback_to_reward_only": not hook_used,
                "policy_guidance_evidence_keys": [],
                "policy_guidance_failure_reason": None,
                "integrated_guidance_card_path": str(existing_card_path),
                "integrated_guidance_card": deepcopy(existing_card),
            },
        }
        if int(round_id) == 1:
            decision["policy_guidance_used_pre_checkpoint"] = hook_used
            decision["pre_checkpoint_integrated_guidance_card_path"] = str(existing_card_path)
            decision["policy_guidance_pre_checkpoint_failure_reason"] = None
        else:
            decision["policy_guidance_used_after_round1"] = hook_used
            decision["after_round1_integrated_guidance_card_path"] = str(existing_card_path)
            decision["policy_guidance_after_round1_failure_reason"] = None
        decision["policy_guidance_fallback_to_reward_only"] = not hook_used
        decision["policy_guidance_evidence_keys"] = []
        return memory

    if int(round_id) == 1:
        mock_context = _policy_guidance_mock_context(
            adaptive_method,
            checkpoint_name=checkpoint_id,
            key="pre_checkpoint",
        )
        discovered_behavior_summary_path = _discover_stage3_behavior_summary_path(
            workflow_dir=workflow_dir,
            adaptive_method=adaptive_method,
        )
        behavior_summary = deepcopy(mock_context.get("behavior_summary"))
        behavior_summary_path = str(
            mock_context.get("behavior_summary_path")
            or discovered_behavior_summary_path
            or ""
        )
        behavior_summary_source = str(mock_context.get("behavior_summary_source") or "")
        if not behavior_summary and behavior_summary_path:
            behavior_summary = _extract_policy_guidance_behavior_summary(
                _load_policy_guidance_summary_from_path(behavior_summary_path)
            )
            if behavior_summary:
                behavior_summary_source = "file"
        if not behavior_summary and mock_context.get("synthetic_mock", False):
            behavior_summary_source = "synthetic_mock"
        elif behavior_summary and not behavior_summary_source:
            behavior_summary_source = "object"
        behavior_summary_available = bool(behavior_summary)
        if not behavior_summary:
            behavior_summary = build_insufficient_behavior_summary(
                intervention_point=f"stage3_{checkpoint_id.lower()}_pre_checkpoint",
                env_key=str((dense_reference_run_summary.get("env_key") or "lbforaging:unknown")),
                eval_episodes=0,
                reason="stage3 pre-checkpoint behavior summary unavailable",
            )
            behavior_summary_source = "unavailable"
        hook_result = build_stage3_pre_checkpoint_policy_guidance(
            task_metadata={
                "workflow_id": str(workflow_dir.name),
                "env_key": str(dense_reference_run_summary.get("env_key") or ""),
                "algorithm": str(dense_reference_run_summary.get("config_name") or ""),
                "seed": int((dense_reference_run_summary.get("args") or {}).get("seed") or 1),
            },
            workflow_dir=workflow_dir,
            checkpoint_id=checkpoint_id,
            checkpoint_step=checkpoint_step,
            current_pbrs_config=deepcopy(current_config),
            reward_metrics=_policy_guidance_reward_metrics(
                decision=decision,
                dense_reference_run_summary=dense_reference_run_summary,
            ),
            behavior_summary=behavior_summary,
            behavior_summary_path=behavior_summary_path or None,
            sparse_dense_context=mock_context.get("sparse_dense_context") or {},
            policy_guidance_enabled=True,
            fallback_to_reward_only=bool(
                adaptive_method.get("policy_guidance_stage3_fallback_to_reward_only", True)
            ),
            mock_llm_mode=guidance_mock_mode,
            use_real_llm=use_real_llm,
            api_key_env=str(adaptive_method.get("api_key_env") or "IUSEAPI_API_KEY"),
            base_url=str(adaptive_method.get("base_url") or "https://www.iuseapi.com/v1"),
            model=str(adaptive_method.get("model") or "gpt-5.2"),
            temperature=float(adaptive_method.get("temperature") or 0.2),
            llm_timeout=float(adaptive_method.get("llm_timeout") or 60.0),
            llm_max_retries=int(adaptive_method.get("llm_max_retries") or 3),
            llm_retry_backoff=float(adaptive_method.get("llm_retry_backoff") or 5.0),
            reuse_policy_guidance_cache=bool(
                adaptive_method.get("reuse_policy_guidance_cache", False)
            ),
            policy_guidance_cache_root=adaptive_method.get(
                "policy_guidance_cache_root"
            ),
            policy_guidance_cache_lookup_mode=str(
                adaptive_method.get(
                    "policy_guidance_cache_lookup_mode",
                    "exact_or_latest_by_intervention",
                )
            ),
        )
        card_path = guidance_dir / f"{hook_result['intervention_point']}_integrated_guidance_card.json"
        if hook_result.get("integrated_guidance_card"):
            _save_json(card_path, hook_result["integrated_guidance_card"])
            hook_result["integrated_guidance_card_path"] = str(card_path)
        hook_used = bool(hook_result.get("policy_guidance_used", False))
        round_payload["policy_guidance"] = {
            **deepcopy(default_policy_guidance),
            **{
                "policy_guidance_used": hook_used,
                "policy_guidance_source": str(hook_result.get("policy_guidance_source") or "reward_only_fallback"),
                "policy_guidance_fallback_to_reward_only": not hook_used,
                "policy_guidance_evidence_keys": list(hook_result.get("policy_guidance_evidence_keys") or []),
                "policy_guidance_failure_reason": hook_result.get("policy_guidance_failure_reason"),
                "integrated_guidance_card_path": hook_result.get("integrated_guidance_card_path"),
                "integrated_guidance_card": deepcopy(hook_result.get("integrated_guidance_card") or {}),
                "behavior_summary_available": bool(behavior_summary_available),
                "behavior_summary_source": behavior_summary_source,
                "intervention_point": str(hook_result.get("intervention_point") or default_policy_guidance["intervention_point"]),
            },
        }
        decision["policy_guidance_enabled"] = True
        decision["policy_guidance_used_pre_checkpoint"] = hook_used
        decision["pre_checkpoint_integrated_guidance_card_path"] = hook_result.get("integrated_guidance_card_path")
        decision["policy_guidance_pre_checkpoint_failure_reason"] = hook_result.get("policy_guidance_failure_reason")
        decision["policy_guidance_fallback_to_reward_only"] = not hook_used
        decision["policy_guidance_evidence_keys"] = list(hook_result.get("policy_guidance_evidence_keys") or [])
        append_policy_guidance_call_record(
            memory,
            _stage3_hook_call_record(
                guidance_id=f"pg_{checkpoint_id.lower()}_pre_checkpoint",
                intervention_point=str(hook_result.get("intervention_point") or default_policy_guidance["intervention_point"]),
                checkpoint_id=checkpoint_id,
                checkpoint_step=checkpoint_step,
                round_id=0,
                summary_path=behavior_summary_path,
                hook_result=hook_result,
            ),
        )
        return memory

    mock_context = _policy_guidance_mock_context(
        adaptive_method,
        checkpoint_name=checkpoint_id,
        key="after_round1",
    )
    discovered_behavior_summary_path = _discover_stage3_behavior_summary_path(
        workflow_dir=workflow_dir,
        adaptive_method=adaptive_method,
    )
    branch_behavior_summaries = list(mock_context.get("branch_behavior_summaries") or [])
    summary_for_policy_diagnosis = deepcopy(mock_context.get("summary_for_policy_diagnosis"))
    behavior_summary_path = str(
        mock_context.get("behavior_summary_path")
        or discovered_behavior_summary_path
        or ""
    )
    behavior_summary_source = str(mock_context.get("behavior_summary_source") or "")
    if behavior_summary_path and (not branch_behavior_summaries or not summary_for_policy_diagnosis):
        summary_payload = _load_policy_guidance_summary_from_path(behavior_summary_path)
        if summary_payload:
            branch_behavior_summaries = list(
                summary_payload.get("branch_summaries")
                or summary_payload.get("branch_behavior_summaries")
                or branch_behavior_summaries
            )
            summary_for_policy_diagnosis = deepcopy(
                summary_payload.get("summary_for_policy_diagnosis")
                or _extract_policy_guidance_behavior_summary(summary_payload)
                or summary_for_policy_diagnosis
            )
            behavior_summary_source = "file"
    if mock_context.get("synthetic_mock", False) and not behavior_summary_source:
        behavior_summary_source = "synthetic_mock"
    elif (branch_behavior_summaries or summary_for_policy_diagnosis) and not behavior_summary_source:
        behavior_summary_source = "object"
    if not summary_for_policy_diagnosis:
        summary_for_policy_diagnosis = build_insufficient_behavior_summary(
            intervention_point=f"stage3_{checkpoint_id.lower()}_after_round1",
            env_key=str((dense_reference_run_summary.get("env_key") or "lbforaging:unknown")),
            eval_episodes=0,
            reason="stage3 after-round1 behavior summary unavailable",
        )
        behavior_summary_source = "unavailable"
    branch_results = []
    round1 = _round_payload(decision, 1)
    for candidate in list(round1.get("candidates") or []):
        result_record = deepcopy(candidate.get("result_record") or {})
        branch_results.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "best_test_sparse_return_mean": result_record.get("best_test_sparse_return_mean"),
                "last_test_sparse_return_mean": result_record.get("last_test_sparse_return_mean"),
            }
        )
    hook_result = build_stage3_after_round1_policy_guidance(
        task_metadata={
            "workflow_id": str(workflow_dir.name),
            "env_key": str(dense_reference_run_summary.get("env_key") or ""),
            "algorithm": str(dense_reference_run_summary.get("config_name") or ""),
            "seed": int((dense_reference_run_summary.get("args") or {}).get("seed") or 1),
        },
        workflow_dir=workflow_dir,
        checkpoint_id=checkpoint_id,
        checkpoint_step=checkpoint_step,
        round_id=1,
        reward_metrics=_policy_guidance_reward_metrics(
            decision=decision,
            dense_reference_run_summary=dense_reference_run_summary,
        ),
        branch_results=branch_results,
        branch_behavior_summaries=branch_behavior_summaries,
        summary_for_policy_diagnosis=summary_for_policy_diagnosis,
        behavior_summary_path=behavior_summary_path or None,
        previous_integrated_guidance_card=deepcopy((_round_payload(decision, 1).get("policy_guidance") or {}).get("integrated_guidance_card") or {}),
        policy_guidance_enabled=True,
        fallback_to_reward_only=bool(
            adaptive_method.get("policy_guidance_stage3_fallback_to_reward_only", True)
        ),
        mock_llm_mode=guidance_mock_mode,
        use_real_llm=use_real_llm,
        api_key_env=str(adaptive_method.get("api_key_env") or "IUSEAPI_API_KEY"),
        base_url=str(adaptive_method.get("base_url") or "https://www.iuseapi.com/v1"),
        model=str(adaptive_method.get("model") or "gpt-5.2"),
        temperature=float(adaptive_method.get("temperature") or 0.2),
        llm_timeout=float(adaptive_method.get("llm_timeout") or 60.0),
        llm_max_retries=int(adaptive_method.get("llm_max_retries") or 3),
        llm_retry_backoff=float(adaptive_method.get("llm_retry_backoff") or 5.0),
        reuse_policy_guidance_cache=bool(
            adaptive_method.get("reuse_policy_guidance_cache", False)
        ),
        policy_guidance_cache_root=adaptive_method.get("policy_guidance_cache_root"),
        policy_guidance_cache_lookup_mode=str(
            adaptive_method.get(
                "policy_guidance_cache_lookup_mode",
                "exact_or_latest_by_intervention",
            )
        ),
    )
    card_path = guidance_dir / f"{hook_result['intervention_point']}_integrated_guidance_card.json"
    if hook_result.get("integrated_guidance_card"):
        _save_json(card_path, hook_result["integrated_guidance_card"])
        hook_result["integrated_guidance_card_path"] = str(card_path)
    hook_used = bool(hook_result.get("policy_guidance_used", False))
    round_payload["policy_guidance"] = {
        **deepcopy(default_policy_guidance),
        **{
            "policy_guidance_used": hook_used,
            "policy_guidance_source": str(hook_result.get("policy_guidance_source") or "reward_only_fallback"),
            "policy_guidance_fallback_to_reward_only": not hook_used,
            "policy_guidance_evidence_keys": list(hook_result.get("policy_guidance_evidence_keys") or []),
            "policy_guidance_failure_reason": hook_result.get("policy_guidance_failure_reason"),
            "integrated_guidance_card_path": hook_result.get("integrated_guidance_card_path"),
            "integrated_guidance_card": deepcopy(hook_result.get("integrated_guidance_card") or {}),
            "behavior_summary_available": bool(branch_behavior_summaries),
            "behavior_summary_source": behavior_summary_source,
            "intervention_point": str(hook_result.get("intervention_point") or default_policy_guidance["intervention_point"]),
        },
    }
    decision["policy_guidance_enabled"] = True
    decision["policy_guidance_used_after_round1"] = hook_used
    decision["after_round1_integrated_guidance_card_path"] = hook_result.get("integrated_guidance_card_path")
    decision["policy_guidance_after_round1_failure_reason"] = hook_result.get("policy_guidance_failure_reason")
    decision["policy_guidance_fallback_to_reward_only"] = not hook_used
    decision["policy_guidance_evidence_keys"] = list(hook_result.get("policy_guidance_evidence_keys") or [])
    append_policy_guidance_call_record(
        memory,
        _stage3_hook_call_record(
            guidance_id=f"pg_{checkpoint_id.lower()}_after_round1",
            intervention_point=str(hook_result.get("intervention_point") or default_policy_guidance["intervention_point"]),
            checkpoint_id=checkpoint_id,
            checkpoint_step=checkpoint_step,
            round_id=1,
            summary_path=behavior_summary_path,
            hook_result=hook_result,
        ),
    )
    return memory


def _sanitize_llm_candidate_payload(
    *,
    llm_payload: Dict[str, Any],
    current_config: Dict[str, float],
    candidate_generation: Dict[str, Any],
    round_policy_guidance: Optional[Dict[str, Any]] = None,
    decision: Optional[Dict[str, Any]] = None,
) -> tuple[List[Dict[str, Any]], List[str]]:
    max_candidates = max(1, int(candidate_generation.get("max_candidates_per_checkpoint") or 9))
    raw_candidates = list(
        llm_payload.get("candidates")
        or llm_payload.get("update_candidates")
        or []
    )
    sanitized: List[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    validation_errors: List[str] = []
    policy_guidance = deepcopy(round_policy_guidance or {})
    policy_guidance_used = bool(policy_guidance.get("policy_guidance_used", False))
    require_behavior_evidence = bool(
        (decision or {}).get("policy_guided_stage3_require_behavior_evidence", True)
    )
    decision_stub = decision or {}

    def add_candidate(candidate_like: Dict[str, Any]) -> None:
        requires_lbf_v2 = str(decision_stub.get("pbrs_version") or "") == "lbf_pbrs_v2"
        requires_rware_v2 = str(decision_stub.get("pbrs_version") or "") == "rware_pbrs_v2"
        if requires_lbf_v2:
            if str(candidate_like.get("pbrs_version") or "") != "lbf_pbrs_v2":
                validation_errors.append(
                    f"legacy_candidate_rejected_under_formal_lbf_v2:{candidate_like.get('candidate_id') or '<unknown>'}"
                )
                return
            try:
                alias_normalized_candidate = _normalize_lbf_stage3_mode_alias(candidate_like)
                lbf_v2_config = normalize_lbf_pbrs_v2_config(alias_normalized_candidate)
            except Exception as exc:
                validation_errors.append(
                    "invalid_lbf_pbrs_v2_candidate:"
                    f"{candidate_like.get('candidate_id') or '<unknown>'}:"
                    f"{exc}"
                )
                return
            normalized = {
                "candidate_id": str(
                    candidate_like.get("candidate_id")
                    or lbf_v2_config.get("candidate_id")
                    or _candidate_id(lbf_v2_config["beta"], lbf_v2_config["wc"])
                ),
                "beta": float(lbf_v2_config["beta"]),
                "wc": float(lbf_v2_config["wc"]),
                "wp": float(lbf_v2_config["wp"]),
                "pbrs_version": "lbf_pbrs_v2",
                "mode": str(lbf_v2_config.get("mode") or ""),
                "active_terms": list(lbf_v2_config.get("active_terms") or []),
                "weights": deepcopy(lbf_v2_config.get("weights") or {}),
            }
            original_mode = alias_normalized_candidate.get("original_mode")
            normalized_from_alias = alias_normalized_candidate.get("normalized_from_alias")
            if isinstance(original_mode, str) and original_mode:
                normalized["original_mode"] = original_mode
            if isinstance(normalized_from_alias, str) and normalized_from_alias:
                normalized["normalized_from_alias"] = normalized_from_alias
        elif requires_rware_v2:
            if str(candidate_like.get("pbrs_version") or "") != "rware_pbrs_v2":
                validation_errors.append(
                    f"non_rware_candidate_rejected_under_rware:{candidate_like.get('candidate_id') or '<unknown>'}"
                )
                return
            try:
                normalized = _normalize_rware_stage_config(candidate_like)
            except Exception as exc:
                validation_errors.append(
                    "invalid_rware_pbrs_v2_candidate:"
                    f"{candidate_like.get('candidate_id') or '<unknown>'}:"
                    f"{exc}"
                )
                return
            normalized["candidate_id"] = str(
                candidate_like.get("candidate_id")
                or normalized.get("candidate_id")
                or f"rware_candidate_{len(sanitized)+1}"
            )
        else:
            normalized = _normalize_candidate_config(
                beta=candidate_like.get("beta"),
                wc=candidate_like.get("wc"),
            )
            if normalized is None:
                validation_errors.append(
                    f"invalid_candidate:{json.dumps(candidate_like, ensure_ascii=False, sort_keys=True)}"
                )
                return
        key = _candidate_identity(normalized)
        if key in seen:
            validation_errors.append(f"duplicate_candidate:{normalized['candidate_id']}")
            return
        normalized["is_no_change"] = _candidate_matches_config(normalized, current_config)
        if normalized["is_no_change"]:
            validation_errors.append(f"llm_returned_no_change_candidate:{normalized['candidate_id']}")
            return
        if requires_rware_v2:
            normalized["candidate_type"] = str(normalized.get("candidate_type") or "")
        else:
            normalized["candidate_type"] = candidate_like.get("candidate_type") or _default_stage3_candidate_type(
                decision=decision_stub,
                round_id=int((decision_stub.get("active_round_id") or 1)),
                index=len(sanitized),
                policy_guidance_used=policy_guidance_used,
            )
        normalized["expected_effect"] = candidate_like.get("expected_effect")
        normalized["rationale"] = candidate_like.get("rationale")
        if requires_rware_v2:
            normalized["evidence_keys_used"] = list(normalized.get("evidence_keys_used") or [])
        else:
            normalized["evidence_keys_used"] = _stage3_candidate_evidence_keys(candidate_like)
        _attach_env_specific_pbrs_metadata(
            normalized,
            env_key=decision_stub.get("env_key"),
            pbrs_version=decision_stub.get("pbrs_version"),
        )
        if policy_guidance_used:
            if require_behavior_evidence and not normalized["evidence_keys_used"]:
                validation_errors.append(
                    f"policy_guided_candidate_missing_behavior_evidence:{normalized['candidate_id']}"
                )
                return
        elif _candidate_mentions_forbidden_policy_evidence(candidate_like):
            validation_errors.append(
                f"reward_only_candidate_uses_policy_evidence:{normalized['candidate_id']}"
            )
            return
        if bool((decision or {}).get("stage3_require_nontrivial_config_delta", True)) and not _is_nontrivial_stage3_update(
            current_config=current_config,
            candidate=normalized,
        ):
            validation_errors.append(
                _nontrivial_stage3_update_failure_detail(
                    candidate_id=normalized["candidate_id"],
                    current_config=current_config,
                    candidate=normalized,
                )
            )
            return
        normalized["effective_update_candidate"] = True
        normalized["candidate_category"] = _stage3_candidate_category(
            str(normalized.get("candidate_type") or "")
        )
        normalized["fallback_style_candidate"] = _is_fallback_style_stage3_candidate(normalized)
        seen.add(key)
        sanitized.append(normalized)

    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            validation_errors.append(f"non_dict_candidate:{candidate!r}")
            continue
        add_candidate(candidate)
        if len(sanitized) >= max_candidates:
            break
    return sanitized[:max_candidates], validation_errors


def _run_dir_from_metrics_json(metrics_json: str | None) -> Optional[str]:
    if not metrics_json:
        return None
    try:
        return str(Path(metrics_json).resolve().parent)
    except Exception:
        return None


def _nearest_metric_value(
    run_dir: str | None,
    step: int,
    metric_name: str = "test_sparse_return_mean",
) -> Optional[float]:
    if not run_dir:
        return None
    try:
        series = load_metric_series(run_dir, metric_name=metric_name)
    except Exception:
        return None
    steps = list(series.get("steps") or [])
    values = list(series.get("values") or [])
    if not steps or not values:
        return None
    try:
        index = min(range(len(steps)), key=lambda idx: abs(int(steps[idx]) - int(step)))
        return float(values[index])
    except Exception:
        return None


def _build_reference_comparison_context(
    *,
    decision: Dict[str, Any],
    stage_selection_result: Dict[str, Any],
    fixed_reference_context: Dict[str, Any],
) -> Dict[str, Any]:
    checkpoint_step = int(decision.get("checkpoint_step") or 0)
    current_run_dir = str(decision.get("source_run_dir") or "")
    sparse_run_dir = _run_dir_from_metrics_json(stage_selection_result.get("sparse_metrics_json"))
    fixed_run_dir = str(fixed_reference_context.get("run_dir") or "")
    current_value = _nearest_metric_value(current_run_dir, checkpoint_step)
    sparse_value = _nearest_metric_value(sparse_run_dir, checkpoint_step)
    fixed_value = _nearest_metric_value(fixed_run_dir, checkpoint_step)
    fixed_margin = None
    sparse_margin = None
    if fixed_value is not None and current_value is not None:
        fixed_margin = float(fixed_value - current_value)
    if sparse_value is not None and current_value is not None:
        sparse_margin = float(sparse_value - current_value)
    underperforming_vs_fixed = bool(fixed_margin is not None and fixed_margin >= 0.05)
    underperforming_vs_sparse = bool(sparse_margin is not None and sparse_margin >= 0.05)
    strongly_underperforming = bool(
        (fixed_margin is not None and fixed_margin >= 0.12)
        or (sparse_margin is not None and sparse_margin >= 0.12)
    )
    return {
        "current_mainline_run_dir": current_run_dir or None,
        "current_mainline_test_sparse_return_mean": current_value,
        "sparse_reference_run_dir": sparse_run_dir,
        "sparse_reference_test_sparse_return_mean": sparse_value,
        "fixed_reference_run_dir": fixed_run_dir or None,
        "fixed_reference_test_sparse_return_mean": fixed_value,
        "fixed_reference_config": deepcopy(fixed_reference_context.get("config") or {}),
        "underperforming_vs_fixed_reference": underperforming_vs_fixed,
        "underperforming_vs_sparse_reference": underperforming_vs_sparse,
        "strongly_underperforming": strongly_underperforming,
        "fixed_reference_margin": fixed_margin,
        "sparse_reference_margin": sparse_margin,
    }


def _requires_reference_or_recovery(payload: Dict[str, Any]) -> bool:
    comparison = payload.get("reference_comparison") or {}
    return bool(
        comparison.get("underperforming_vs_fixed_reference")
        or comparison.get("underperforming_vs_sparse_reference")
        or comparison.get("strongly_underperforming")
    )


def _has_reference_or_recovery_candidate(candidates: List[Dict[str, Any]]) -> bool:
    return any(
        str(candidate.get("candidate_type") or "") in {"reference_like", "recovery"}
        for candidate in candidates
    )


def _deterministic_reference_or_recovery_candidate(
    *,
    current_config: Dict[str, float],
    fixed_reference_config: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_fixed = _normalize_candidate_config(
        beta=(fixed_reference_config or {}).get("beta"),
        wc=(fixed_reference_config or {}).get("wc"),
    )
    if normalized_fixed is not None:
        normalized_fixed["candidate_type"] = "reference_like"
        normalized_fixed["expected_effect"] = "move toward the stronger fixed PBRS reference"
        normalized_fixed["rationale"] = "deterministic fallback added because underperforming case lacked reference_like or recovery"
        normalized_fixed["is_no_change"] = False
        normalized_fixed["added_by_fallback"] = True
        _attach_env_specific_pbrs_metadata(
            normalized_fixed,
            env_key=current_config.get("env_key"),
            pbrs_version=current_config.get("pbrs_version"),
        )
        return normalized_fixed

    beta = min(1.0, max(0.0, float(current_config["beta"]) + (0.2 if float(current_config["beta"]) <= 0.8 else -0.2)))
    wc = min(1.0, max(0.0, float(current_config["wc"]) - 0.2 if float(current_config["wc"]) >= 0.5 else float(current_config["wc"]) + 0.2))
    fallback = _normalize_candidate_config(beta=beta, wc=wc)
    assert fallback is not None
    fallback["candidate_type"] = "recovery"
    fallback["expected_effect"] = "controlled recovery move away from the current underperforming configuration"
    fallback["rationale"] = "deterministic fallback added because underperforming case lacked reference_like or recovery"
    fallback["is_no_change"] = False
    fallback["added_by_fallback"] = True
    _attach_env_specific_pbrs_metadata(
        fallback,
        env_key=current_config.get("env_key"),
        pbrs_version=current_config.get("pbrs_version"),
    )
    return fallback


def _append_candidate_with_limit(
    *,
    candidates: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    max_candidates: int,
    validation_errors: List[str],
) -> List[Dict[str, Any]]:
    existing_ids = {str(item.get("candidate_id")) for item in candidates}
    if str(candidate.get("candidate_id")) in existing_ids:
        validation_errors.append(f"duplicate_deterministic_recovery_candidate:{candidate.get('candidate_id')}")
        return candidates
    if len(candidates) < max_candidates:
        return candidates + [candidate]
    for idx in range(len(candidates) - 1, -1, -1):
        if bool(candidates[idx].get("is_no_change")):
            continue
        replaced = list(candidates)
        replaced[idx] = candidate
        validation_errors.append(f"replaced_candidate_with_deterministic_recovery:{candidate.get('candidate_id')}")
        return replaced
    validation_errors.append(f"unable_to_insert_deterministic_recovery:{candidate.get('candidate_id')}")
    return candidates


def _prompt_version(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return f"{path.name}:missing"
    return f"{path.name}:{hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]}"


def _diagnostic_series_stats(run_dir: Optional[str]) -> Dict[str, Any]:
    if not run_dir:
        return {}
    try:
        series = load_metric_series(run_dir)
    except Exception:
        return {}
    summary = series_summary(series, last_k=5)
    late_tail = list(series.get("values") or [])[-5:]
    last_k_mean = None if not late_tail else float(sum(float(v) for v in late_tail) / len(late_tail))
    late_window_slope = improvement_slope(series, start_step=(series.get("steps") or [None])[-5] if len(series.get("steps") or []) >= 5 else None, points=5)
    return {
        "last_k_mean": last_k_mean if last_k_mean is not None else summary.get("last_5_mean"),
        "auc": summary.get("auc"),
        "late_window_slope": late_window_slope,
    }


def _build_stage3_llm_backend(
    *,
    use_real_llm: bool,
    api_key_env: str,
    base_url: str,
    model: str,
    temperature: float,
    llm_timeout: float,
    llm_retry_count: int,
    llm_retry_backoff: float,
) -> Optional[OpenAIChatBackend]:
    if not use_real_llm:
        return None
    return OpenAIChatBackend(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=llm_timeout,
        max_retries=max(1, int(llm_retry_count)),
        retry_backoff_seconds=llm_retry_backoff,
    )


def _stage3_backend_probe(
    *,
    use_real_llm: bool,
    api_key_env: str,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    probe = probe_openai_backend(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
    )
    if not use_real_llm:
        probe["backend_reason"] = "use_real_llm_disabled"
    elif not bool(probe.get("api_key_detected")):
        probe["backend_reason"] = "api_key_missing"
    elif not bool(probe.get("openai_import_available")):
        probe["backend_reason"] = "openai_import_failed"
    else:
        probe["backend_reason"] = "backend_ready"
    return probe


def _stage3_candidate_prompt_path(decision: Dict[str, Any]) -> Path:
    round_id = int(decision.get("active_round_id") or 1)
    policy_guidance = deepcopy((_round_payload(decision, round_id).get("policy_guidance") or {}))
    if bool(policy_guidance.get("policy_guidance_used")) and policy_guidance.get("integrated_guidance_card"):
        return (
            DEFAULT_POLICY_GUIDED_STAGE3_ROUND1_CANDIDATE_PROMPT_PATH
            if round_id <= 1
            else DEFAULT_POLICY_GUIDED_STAGE3_ROUND2_CANDIDATE_PROMPT_PATH
        )
    return DEFAULT_STAGE3_CANDIDATE_PROMPT_PATH


def _generate_candidates_with_llm_or_fallback(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    current_config: Dict[str, float],
    adaptive_method: Dict[str, Any],
    stage_selection_result: Dict[str, Any],
    dense_reference_selection: Dict[str, Any],
    dense_reference_run_summary: Dict[str, Any],
    branch_budget_steps: int,
    use_real_llm: bool,
    api_key_env: str,
    base_url: str,
    model: str,
    temperature: float,
    llm_timeout: float,
    llm_retry_count: int,
    llm_retry_backoff: float,
) -> List[Dict[str, Any]]:
    candidate_generation = deepcopy(adaptive_method.get("candidate_generation") or {})
    llm_enabled = bool(adaptive_method.get("use_llm_stage3_candidate_generation", True))
    use_cache = bool(adaptive_method.get("llm_stage3_use_cache", True))
    fallback_on_error = bool(adaptive_method.get("fallback_to_deterministic_on_llm_error", True))
    enforce_recovery_candidate = bool(
        adaptive_method.get("enforce_recovery_candidate_when_underperforming", True)
    )
    repair_retry_count = max(0, int(adaptive_method.get("llm_candidate_repair_retry_count", 1) or 0))
    add_deterministic_recovery_fallback = bool(
        adaptive_method.get("add_deterministic_recovery_fallback", True)
    )
    mock_mode = bool(adaptive_method.get("mock_llm_mode", False))
    max_candidates = max(1, int(candidate_generation.get("max_candidates_per_checkpoint") or 9))
    decision["llm_candidate_generation_status"] = "pending"
    decision["llm_candidate_generation_original_valid_count"] = 0
    decision["llm_candidate_generation_topup_count"] = 0
    decision["llm_candidate_generation_repair_status"] = "not_needed"
    decision["llm_candidate_generation_repair_type"] = "schema_contract_topup"
    decision["llm_candidate_generation_deterministic_fallback_used"] = False
    cache_path = _decision_dir(
        workflow_dir,
        int(decision["decision_index"]),
        str(decision["stage_label"]),
    ) / f"llm_candidate_generation{_round_cache_suffix(decision)}.json"
    decision["llm_candidate_generation_cache_path"] = str(cache_path)
    payload = _build_candidate_generation_payload(
        decision=decision,
        current_config=current_config,
        candidate_generation=candidate_generation,
        stage_selection_result=stage_selection_result,
        dense_reference_selection=dense_reference_selection,
        dense_reference_run_summary=dense_reference_run_summary,
        branch_budget_steps=branch_budget_steps,
        fixed_reference_context=deepcopy(adaptive_method.get("fixed_reference_context") or {}),
    )
    payload["llm_model"] = str(model)
    payload["llm_routing"] = deepcopy(
        llm_routing_artifact_fields(adaptive_method.get("llm_routing") or {})
    )
    payload["prompt_version"] = _prompt_version(_stage3_candidate_prompt_path(decision))
    payload_hash = _stable_payload_hash(payload)
    ledger_context = _stage3_ledger_context(
        workflow_dir=workflow_dir,
        decision=decision,
        adaptive_method=adaptive_method,
        model=model,
        base_url=base_url,
    )
    ledger_event_refs: List[str] = []
    raw_text = ""
    backend_metadata: Dict[str, Any] = {}
    parsed_candidates_count = 0
    requires_reference_or_recovery = _requires_reference_or_recovery(payload)
    round_guidance = deepcopy(
        (_round_payload(decision, int(decision.get("active_round_id") or 1)).get("policy_guidance") or {})
    )
    effective_mock_mode = bool(
        mock_mode
        or (
            not use_real_llm
            and bool(round_guidance.get("policy_guidance_used"))
            and bool(round_guidance.get("integrated_guidance_card"))
        )
    )
    decision["llm_candidate_generation_underperforming_vs_fixed_reference"] = bool(
        ((payload.get("reference_comparison") or {}).get("underperforming_vs_fixed_reference"))
    )
    decision["llm_candidate_generation_fixed_reference_config"] = deepcopy(
        ((payload.get("reference_comparison") or {}).get("fixed_reference_config")) or {}
    )
    decision["llm_candidate_generation_requires_reference_or_recovery"] = bool(
        enforce_recovery_candidate and requires_reference_or_recovery
    )
    decision["llm_candidate_generation_repair_used"] = False
    decision["llm_candidate_generation_deterministic_recovery_added"] = False
    decision["llm_candidate_generation_provided_reference_or_recovery"] = False
    decision["candidate_generation_source"] = None
    decision["llm_candidate_generation_backend_probe"] = _stage3_backend_probe(
        use_real_llm=use_real_llm,
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
    )
    decision["llm_candidate_generation_backend_error_type"] = None
    decision["llm_candidate_generation_backend_error"] = None
    decision["llm_candidate_generation_prompt_path"] = str(_stage3_candidate_prompt_path(decision))

    def deterministic_candidates() -> List[Dict[str, Any]]:
        round_id = int(decision.get("active_round_id") or 1)
        round_guidance = deepcopy((_round_payload(decision, round_id).get("policy_guidance") or {}))
        if bool(round_guidance.get("policy_guidance_used")) and round_guidance.get("integrated_guidance_card"):
            guided_batch = build_policy_guided_stage3_candidate_batch(
                integrated_guidance_card=deepcopy(round_guidance.get("integrated_guidance_card") or {}),
                checkpoint_id=str(decision.get("checkpoint_name") or ""),
                round_id=round_id,
                max_candidates=max_candidates,
                env_family="rware" if is_rware_workflow(decision.get("env_key"), decision.get("pbrs_version")) else "lbf",
            )
            candidates = list(guided_batch.get("candidates") or [])
        else:
            candidates = _build_candidate_configs(
                current_config=current_config,
                candidate_generation=candidate_generation,
            )
        candidates = [item for item in candidates if not bool(item.get("is_no_change"))]
        decision["llm_candidate_generation_status"] = "deterministic_fallback"
        if decision.get("candidate_generation_source") is None:
            decision["candidate_generation_source"] = "deterministic_fallback"
        return candidates

    if not llm_enabled:
        decision["llm_candidate_generation_validation_errors"] = [
            "stage3_llm_candidate_generation_disabled"
        ]
        candidates = deterministic_candidates()
        artifact_path = _persist_stage3_candidate_generation_artifact(
            workflow_dir=workflow_dir,
            decision=decision,
            payload_hash=payload_hash,
            backend_metadata={"llm_enabled": False},
            raw_text=raw_text,
            parsed_candidates_count=0,
            valid_candidates_count=len(candidates),
            validation_errors=decision["llm_candidate_generation_validation_errors"],
            repair_retry_used=False,
            repair_retry_count=0,
            fallback_used=True,
            fallback_reason="stage3_llm_candidate_generation_disabled",
            candidate_generation_source=str(decision.get("candidate_generation_source") or "deterministic_fallback"),
            final_source=str(decision.get("candidate_generation_source") or "deterministic_fallback"),
            ledger_event_refs=[],
        )
        write_ledger_event(
            **ledger_context,
            call_type="fallback",
            reason="stage3 candidate generation disabled",
            cache_key=str(cache_path),
            cache_hit=False,
            payload=payload,
            payload_hash=payload_hash,
            success=None,
            repair_used=False,
            repair_retry_count=0,
            fallback_used=True,
            fallback_reason="stage3_llm_candidate_generation_disabled",
            artifact_path=artifact_path,
            caller_file=__file__,
            caller_function="_generate_candidates_with_llm_or_fallback",
            event_state="fallback",
            metadata={"decision_index": decision.get("decision_index"), "checkpoint_name": decision.get("checkpoint_name")},
        )
        return candidates

    if use_cache and cache_path.exists():
        try:
            cache_payload = _load_json(cache_path)
            if str(cache_payload.get("payload_hash") or "") == payload_hash:
                parsed = cache_payload.get("parsed") or {}
                sanitized, validation_errors = _sanitize_llm_candidate_payload(
                    llm_payload=parsed,
                    current_config=current_config,
                    candidate_generation=candidate_generation,
                    round_policy_guidance=round_guidance,
                    decision=decision,
                )
                if enforce_recovery_candidate and requires_reference_or_recovery:
                    has_required_type = _has_reference_or_recovery_candidate(sanitized)
                    decision["llm_candidate_generation_provided_reference_or_recovery"] = bool(has_required_type)
                    if not has_required_type:
                        validation_errors.append("missing_reference_like_or_recovery_when_underperforming")
                minimum_candidates = max_candidates
                if len(sanitized) >= minimum_candidates:
                    decision["llm_candidate_generation_status"] = str(cache_payload.get("status") or "cache_hit")
                    decision["candidate_generation_source"] = "cache"
                    decision["llm_candidate_generation_validation_errors"] = list(
                        cache_payload.get("validation_errors") or validation_errors
                    )
                    decision["llm_candidate_generation_repair_used"] = bool(
                        cache_payload.get("repair_retry_used", False)
                    )
                    decision["llm_candidate_generation_deterministic_recovery_added"] = bool(
                        cache_payload.get("deterministic_recovery_candidate_added", False)
                    )
                    cache_event = write_ledger_event(
                        **ledger_context,
                        call_type="cache_hit",
                        reason="stage3 candidate generation cache hit",
                        cache_key=str(cache_path),
                        cache_hit=True,
                        payload=payload,
                        payload_hash=payload_hash,
                        usage=None,
                        request_id=extract_request_id(cache_payload),
                        success=True,
                        repair_used=bool(cache_payload.get("repair_retry_used", False)),
                        repair_retry_count=1 if bool(cache_payload.get("repair_retry_used", False)) else 0,
                        fallback_used=False,
                        artifact_path=None,
                        caller_file=__file__,
                        caller_function="_generate_candidates_with_llm_or_fallback",
                        event_state="cache_hit",
                        metadata={"decision_index": decision.get("decision_index"), "checkpoint_name": decision.get("checkpoint_name")},
                    )
                    ledger_event_refs.append(str(cache_event.get("record_id") or ""))
                    artifact_path = _persist_stage3_candidate_generation_artifact(
                        workflow_dir=workflow_dir,
                        decision=decision,
                        payload_hash=payload_hash,
                        backend_metadata=deepcopy(cache_payload.get("backend_metadata") or {}),
                        raw_text=str(cache_payload.get("raw_text") or ""),
                        parsed_candidates_count=len(list((cache_payload.get("parsed") or {}).get("candidates") or [])),
                        valid_candidates_count=len(sanitized),
                        validation_errors=decision["llm_candidate_generation_validation_errors"],
                        repair_retry_used=bool(cache_payload.get("repair_retry_used", False)),
                        repair_retry_count=1 if bool(cache_payload.get("repair_retry_used", False)) else 0,
                        fallback_used=False,
                        fallback_reason=None,
                        candidate_generation_source="cache",
                        final_source="cache",
                        ledger_event_refs=ledger_event_refs,
                    )
                    decision["llm_candidate_generation_artifact_path"] = artifact_path
                    return sanitized
        except Exception:
            pass

    try:
        if effective_mock_mode:
            if bool(round_guidance.get("policy_guidance_used")) and round_guidance.get("integrated_guidance_card"):
                parsed = build_policy_guided_stage3_candidate_batch(
                    integrated_guidance_card=deepcopy(round_guidance.get("integrated_guidance_card") or {}),
                    checkpoint_id=str(decision.get("checkpoint_name") or ""),
                    round_id=int(decision.get("active_round_id") or 1),
                    max_candidates=max_candidates,
                    env_family="rware" if is_rware_workflow(decision.get("env_key"), decision.get("pbrs_version")) else "lbf",
                )
            else:
                parsed = _deterministic_mock_candidate_generation(
                    current_config=current_config,
                    candidate_generation=candidate_generation,
                )
            raw_text = json.dumps(parsed, ensure_ascii=False, indent=2)
            backend_metadata = {
                "mock_llm_mode": True,
                "policy_guided_mock_generation": bool(round_guidance.get("policy_guidance_used")),
            }
            parsed_candidates_count = len(list(parsed.get("candidates") or parsed.get("update_candidates") or []))
            status = "mock_success"
            backend = None
            prompt_path = _stage3_candidate_prompt_path(decision)
        else:
            prompt_path = _stage3_candidate_prompt_path(decision)
            backend = _build_stage3_llm_backend(
                use_real_llm=use_real_llm,
                api_key_env=api_key_env,
                base_url=base_url,
                model=model,
                temperature=temperature,
                llm_timeout=llm_timeout,
                llm_retry_count=llm_retry_count,
                llm_retry_backoff=llm_retry_backoff,
            )
            if backend is None:
                backend_reason = str(
                    (decision.get("llm_candidate_generation_backend_probe") or {}).get("backend_reason")
                    or "backend_unavailable"
                )
                raise ValueError(f"stage3_llm_backend_unavailable:{backend_reason}")
            prompt_spec = prompt_path.read_text(encoding="utf-8")
            prompt = "\n".join(
                [
                    "Prompt specification:",
                    prompt_spec,
                    "",
                    "Adaptive Stage 3 candidate-generation payload:",
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                    "",
                    "Return strict JSON only.",
                ]
            )
            attempt_event = write_ledger_event(
                **ledger_context,
                call_type="generator",
                reason="stage3 candidate generation live llm attempt",
                cache_key=str(cache_path),
                cache_hit=False,
                payload=payload,
                payload_hash=payload_hash,
                success=None,
                artifact_path=None,
                caller_file=__file__,
                caller_function="_generate_candidates_with_llm_or_fallback",
                event_state="attempt",
                metadata={"decision_index": decision.get("decision_index"), "checkpoint_name": decision.get("checkpoint_name")},
            )
            ledger_event_refs.append(str(attempt_event.get("record_id") or ""))
            llm_result = backend.generate_text(
                system_prompt=(
                    "You propose local PBRS candidate configurations for adaptive MARL checkpoint validation. "
                    "Return strictly valid JSON."
                ),
                user_prompt=prompt,
                metadata={
                    "role": "adaptive_stage3_candidate_generation",
                    "workflow_id": decision.get("workflow_id"),
                    "decision_index": decision.get("decision_index"),
                    "stage_label": decision.get("stage_label"),
                },
            )
            parsed = _extract_json_object(llm_result["text"])
            raw_text = llm_result["text"]
            backend_metadata = llm_result.get("backend_metadata") or {}
            parsed_candidates_count = len(list(parsed.get("candidates") or []))
            status = "success"
        sanitized, validation_errors = _sanitize_llm_candidate_payload(
            llm_payload=parsed,
            current_config=current_config,
            candidate_generation=candidate_generation,
            round_policy_guidance=round_guidance,
            decision=decision,
        )
        round_cfg = _decision_round_config(decision)
        active_round_id = int(decision.get("active_round_id") or 1)
        required_effective_candidates = (
            round_cfg["round1_generated"] if active_round_id <= 1 else round_cfg["round2_generated"]
        )
        needs_repair = False
        repair_reasons: List[str] = []
        if len(sanitized) < required_effective_candidates:
            needs_repair = True
            repair_reasons.append(
                f"returned {len(sanitized)} effective update candidates but {required_effective_candidates} are required for this round"
            )
        if enforce_recovery_candidate and requires_reference_or_recovery and not _has_reference_or_recovery_candidate(sanitized):
            validation_errors.append("missing_reference_like_or_recovery_when_underperforming")
            needs_repair = True
            repair_reasons.append(
                "include at least one reference_like or recovery candidate"
            )
        if needs_repair and repair_retry_count > 0 and not effective_mock_mode and backend is not None:
            decision["llm_candidate_generation_repair_used"] = True
            existing_valid_candidates = deepcopy(sanitized)
            missing_candidates = max(0, required_effective_candidates - len(existing_valid_candidates))
            repair_instruction = (
                "The previous response did not satisfy the Stage 3 round-local effective update budget. "
                + " ; ".join(repair_reasons)
                + f". Return exactly {missing_candidates} replacement update candidates under the top-level JSON key `candidates`."
                + " Do not regenerate or repeat the already valid candidates supplied below."
                + " Keep `mode` strictly as a runtime PBRS mode enum for the requested env family."
                + " Put search-intent labels such as coverage_recovery, allocation_rebalance, stability_recovery, or progress_shift in `candidate_type`, not in `mode`."
                + " Every replacement candidate must individually satisfy the nontrivial-delta rule against current_mainline_config."
                + " Use validation_errors to avoid the rejected pattern."
            )
            repair_prompt = "\n".join(
                [
                    "Prompt specification:",
                    prompt_path.read_text(encoding="utf-8"),
                    "",
                    "Adaptive Stage 3 candidate-generation payload:",
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                    "",
                    "Already valid candidates to keep unchanged:",
                    json.dumps(existing_valid_candidates, ensure_ascii=False, indent=2, sort_keys=True),
                    "",
                    "Validation errors from the previous attempt:",
                    json.dumps(validation_errors, ensure_ascii=False, indent=2),
                    "",
                    "Repair instruction:",
                    repair_instruction,
                    "",
                    "Return strict JSON only.",
                ]
            )
            repair_result = backend.generate_text(
                system_prompt=(
                    "You propose local PBRS candidate configurations for adaptive MARL checkpoint validation. "
                    "Return strictly valid JSON."
                ),
                user_prompt=repair_prompt,
                metadata={
                    "role": "adaptive_stage3_candidate_generation_repair",
                    "workflow_id": decision.get("workflow_id"),
                    "decision_index": decision.get("decision_index"),
                    "stage_label": decision.get("stage_label"),
                },
            )
            repair_event = write_ledger_event(
                **ledger_context,
                call_type="repair",
                reason="stage3 candidate generation repair",
                cache_key=str(cache_path),
                cache_hit=False,
                payload=payload,
                payload_hash=payload_hash,
                usage=repair_result.get("usage"),
                request_id=extract_request_id(repair_result),
                success=True,
                repair_used=True,
                repair_retry_count=1,
                fallback_used=False,
                artifact_path=None,
                caller_file=__file__,
                caller_function="_generate_candidates_with_llm_or_fallback",
                event_state="repair",
                metadata={"decision_index": decision.get("decision_index"), "checkpoint_name": decision.get("checkpoint_name")},
            )
            ledger_event_refs.append(str(repair_event.get("record_id") or ""))
            repair_parsed = _extract_json_object(repair_result["text"])
            repair_sanitized, repair_validation_errors = _sanitize_llm_candidate_payload(
                llm_payload=repair_parsed,
                current_config=current_config,
                candidate_generation=candidate_generation,
                round_policy_guidance=round_guidance,
                decision=decision,
            )
            validation_errors = list(validation_errors) + list(repair_validation_errors)
            sanitized = _merge_stage3_unique_candidates(
                existing_candidates=existing_valid_candidates,
                new_candidates=repair_sanitized,
                validation_errors=validation_errors,
                max_candidates=required_effective_candidates,
            )
            raw_text = repair_result["text"]
            parsed = repair_parsed
            backend_metadata = repair_result.get("backend_metadata") or {}
            parsed_candidates_count = len(list(parsed.get("candidates") or parsed.get("update_candidates") or []))
            status = "success_repaired"
        if enforce_recovery_candidate and requires_reference_or_recovery and not _has_reference_or_recovery_candidate(sanitized):
            validation_errors.append("missing_reference_like_or_recovery_when_underperforming")
            if (
                add_deterministic_recovery_fallback
                and enforce_recovery_candidate
                and requires_reference_or_recovery
                and not _has_reference_or_recovery_candidate(sanitized)
            ):
                decision["llm_candidate_generation_repair_used"] = True
                fallback_candidate = _deterministic_reference_or_recovery_candidate(
                    current_config=current_config,
                    fixed_reference_config=deepcopy(((payload.get("reference_comparison") or {}).get("fixed_reference_config")) or {}),
                )
                sanitized = _append_candidate_with_limit(
                    candidates=sanitized,
                    candidate=fallback_candidate,
                    max_candidates=max_candidates,
                    validation_errors=validation_errors,
                )
                decision["llm_candidate_generation_deterministic_recovery_added"] = True
                status = f"{status}_with_deterministic_recovery"
        decision["llm_candidate_generation_provided_reference_or_recovery"] = bool(
            _has_reference_or_recovery_candidate(sanitized)
        )
        sanitized, shortfall_audit, validation_errors = _repair_stage3_candidate_shortfall(
            decision=decision,
            current_config=current_config,
            candidate_generation=candidate_generation,
            round_policy_guidance=round_guidance,
            candidates=sanitized,
            required_effective_candidates=required_effective_candidates,
            validation_errors=validation_errors,
        )
        decision["llm_candidate_generation_original_valid_count"] = int(
            shortfall_audit.get("original_valid_count") or 0
        )
        decision["llm_candidate_generation_topup_count"] = int(
            shortfall_audit.get("topup_count") or 0
        )
        decision["llm_candidate_generation_repair_status"] = str(
            shortfall_audit.get("repair_status") or "not_needed"
        )
        decision["llm_candidate_generation_repair_type"] = str(
            shortfall_audit.get("repair_type") or "schema_contract_topup"
        )
        decision["llm_candidate_generation_deterministic_fallback_used"] = bool(
            shortfall_audit.get("deterministic_fallback_used", False)
        )
        minimum_candidates = required_effective_candidates
        if len(sanitized) < minimum_candidates:
            shortage_message = (
                "LLM candidate generation returned too few valid candidates after sanitization: "
                f"{len(sanitized)}"
            )
            validation_errors = list(validation_errors) + [shortage_message]
            artifact_path = _persist_stage3_candidate_generation_artifact(
                workflow_dir=workflow_dir,
                decision=decision,
                payload_hash=payload_hash,
                backend_metadata=backend_metadata,
                raw_text=raw_text,
                parsed_candidates_count=parsed_candidates_count,
                valid_candidates_count=len(sanitized),
                validation_errors=validation_errors,
                repair_retry_used=bool(decision.get("llm_candidate_generation_repair_used")),
                repair_retry_count=1 if bool(decision.get("llm_candidate_generation_repair_used")) else 0,
                fallback_used=False,
                fallback_reason=shortage_message,
                candidate_generation_source="live_llm" if not effective_mock_mode else "mock_llm_mode",
                final_source=f"{status}_insufficient_valid_candidates",
                ledger_event_refs=ledger_event_refs,
                original_valid_count=decision.get("llm_candidate_generation_original_valid_count"),
                topup_count=int(decision.get("llm_candidate_generation_topup_count") or 0),
                repair_status=str(decision.get("llm_candidate_generation_repair_status") or ""),
                repair_type=str(decision.get("llm_candidate_generation_repair_type") or ""),
                deterministic_fallback_used=bool(
                    decision.get("llm_candidate_generation_deterministic_fallback_used", False)
                ),
            )
            decision["llm_candidate_generation_validation_errors"] = list(validation_errors)
            decision["llm_candidate_generation_artifact_path"] = artifact_path
            if use_cache:
                _save_json(
                    cache_path,
                    {
                        "payload_hash": payload_hash,
                        "status": f"{status}_insufficient_valid_candidates",
                        "prompt_version": _prompt_version(prompt_path),
                        "prompt_path": str(prompt_path),
                        "input_payload": payload,
                        "parsed": parsed,
                        "raw_text": raw_text,
                        "backend_metadata": backend_metadata,
                        "underperforming_vs_fixed_reference": bool(
                            ((payload.get("reference_comparison") or {}).get("underperforming_vs_fixed_reference"))
                        ),
                        "fixed_reference_config": deepcopy(
                            ((payload.get("reference_comparison") or {}).get("fixed_reference_config")) or {}
                        ),
                        "reference_like_or_recovery_required": bool(
                            enforce_recovery_candidate and requires_reference_or_recovery
                        ),
                        "reference_like_or_recovery_provided": bool(
                            _has_reference_or_recovery_candidate(sanitized)
                        ),
                        "repair_retry_used": bool(decision.get("llm_candidate_generation_repair_used")),
                        "deterministic_recovery_candidate_added": False,
                        "validation_errors": validation_errors,
                        "final_candidates": deepcopy(sanitized),
                    },
                )
            raise ValueError(
                shortage_message
            )
        decision["llm_candidate_generation_status"] = status
        decision["candidate_generation_source"] = "live_llm" if not effective_mock_mode else "mock_llm_mode"
        decision["llm_candidate_generation_validation_errors"] = list(validation_errors)
        artifact_path = _persist_stage3_candidate_generation_artifact(
            workflow_dir=workflow_dir,
            decision=decision,
            payload_hash=payload_hash,
            backend_metadata=backend_metadata,
            raw_text=raw_text,
            parsed_candidates_count=parsed_candidates_count,
            valid_candidates_count=len(sanitized),
            validation_errors=validation_errors,
            repair_retry_used=bool(decision.get("llm_candidate_generation_repair_used")),
            repair_retry_count=1 if bool(decision.get("llm_candidate_generation_repair_used")) else 0,
            fallback_used=bool(decision.get("llm_candidate_generation_deterministic_recovery_added")),
            fallback_reason=(
                "deterministic_reference_or_recovery_topup"
                if bool(decision.get("llm_candidate_generation_deterministic_recovery_added"))
                else None
            ),
            candidate_generation_source=str(decision.get("candidate_generation_source") or ""),
            final_source=str(status),
            ledger_event_refs=ledger_event_refs,
            original_valid_count=decision.get("llm_candidate_generation_original_valid_count"),
            topup_count=int(decision.get("llm_candidate_generation_topup_count") or 0),
            repair_status=str(decision.get("llm_candidate_generation_repair_status") or ""),
            repair_type=str(decision.get("llm_candidate_generation_repair_type") or ""),
            deterministic_fallback_used=bool(
                decision.get("llm_candidate_generation_deterministic_fallback_used", False)
            ),
        )
        final_event = write_ledger_event(
            **ledger_context,
            call_type="generator",
            reason="stage3 candidate generation completed",
            cache_key=str(cache_path),
            cache_hit=False,
            payload=payload,
            payload_hash=payload_hash,
            usage=None if effective_mock_mode else llm_result.get("usage"),
            request_id=None if effective_mock_mode else extract_request_id(llm_result),
            success=True,
            repair_used=bool(decision.get("llm_candidate_generation_repair_used")),
            repair_retry_count=1 if bool(decision.get("llm_candidate_generation_repair_used")) else 0,
            fallback_used=bool(decision.get("llm_candidate_generation_deterministic_recovery_added")),
            fallback_reason=(
                "deterministic_reference_or_recovery_topup"
                if bool(decision.get("llm_candidate_generation_deterministic_recovery_added"))
                else None
            ),
            artifact_path=artifact_path,
            caller_file=__file__,
            caller_function="_generate_candidates_with_llm_or_fallback",
            event_state="success",
            metadata={"decision_index": decision.get("decision_index"), "checkpoint_name": decision.get("checkpoint_name"), "backend_metadata": backend_metadata},
            record_id=ledger_event_refs[0] if ledger_event_refs else None,
        )
        ledger_event_refs.append(str(final_event.get("record_id") or ""))
        if use_cache:
            _save_json(
                cache_path,
                {
                    "payload_hash": payload_hash,
                    "status": status,
                    "prompt_version": _prompt_version(prompt_path),
                    "prompt_path": str(prompt_path),
                    "input_payload": payload,
                    "parsed": parsed,
                    "raw_text": raw_text,
                    "backend_metadata": backend_metadata,
                    "underperforming_vs_fixed_reference": bool(
                        ((payload.get("reference_comparison") or {}).get("underperforming_vs_fixed_reference"))
                    ),
                    "fixed_reference_config": deepcopy(
                        ((payload.get("reference_comparison") or {}).get("fixed_reference_config")) or {}
                    ),
                    "reference_like_or_recovery_required": bool(
                        enforce_recovery_candidate and requires_reference_or_recovery
                    ),
                    "reference_like_or_recovery_provided": bool(
                        _has_reference_or_recovery_candidate(sanitized)
                    ),
                    "repair_retry_used": bool(decision.get("llm_candidate_generation_repair_used")),
                    "deterministic_recovery_candidate_added": bool(
                        decision.get("llm_candidate_generation_deterministic_recovery_added")
                    ),
                    "validation_errors": validation_errors,
                    "final_candidates": deepcopy(sanitized),
                },
            )
        return sanitized
    except Exception as exc:
        decision["llm_candidate_generation_status"] = f"error:{exc.__class__.__name__}"
        decision["llm_candidate_generation_validation_errors"] = [str(exc)]
        decision["llm_candidate_generation_backend_error_type"] = exc.__class__.__name__
        decision["llm_candidate_generation_backend_error"] = str(exc)
        if str(exc).startswith("stage3_llm_backend_unavailable:") or "Missing API key" in str(exc) or "required for the real LLM backend" in str(exc):
            decision["candidate_generation_source"] = "api_failure_fallback"
        else:
            decision["candidate_generation_source"] = "deterministic_fallback"
        if not fallback_on_error:
            raise
        fallback_candidates = deterministic_candidates()
        fallback_event = write_ledger_event(
            **ledger_context,
            call_type="fallback",
            reason="stage3 candidate generation deterministic fallback",
            cache_key=str(cache_path),
            cache_hit=False,
            payload=payload,
            payload_hash=payload_hash,
            success=False,
            repair_used=bool(decision.get("llm_candidate_generation_repair_used")),
            repair_retry_count=1 if bool(decision.get("llm_candidate_generation_repair_used")) else 0,
            fallback_used=True,
            fallback_reason=str(exc),
            artifact_path=None,
            caller_file=__file__,
            caller_function="_generate_candidates_with_llm_or_fallback",
            event_state="fallback",
            metadata={"decision_index": decision.get("decision_index"), "checkpoint_name": decision.get("checkpoint_name"), "backend_metadata": backend_metadata},
            error_type=exc.__class__.__name__,
        )
        ledger_event_refs.append(str(fallback_event.get("record_id") or ""))
        artifact_path = _persist_stage3_candidate_generation_artifact(
            workflow_dir=workflow_dir,
            decision=decision,
            payload_hash=payload_hash,
            backend_metadata=backend_metadata,
            raw_text=raw_text,
            parsed_candidates_count=parsed_candidates_count,
            valid_candidates_count=len(fallback_candidates),
            validation_errors=[str(exc)],
            repair_retry_used=bool(decision.get("llm_candidate_generation_repair_used")),
            repair_retry_count=1 if bool(decision.get("llm_candidate_generation_repair_used")) else 0,
            fallback_used=True,
            fallback_reason=str(exc),
            candidate_generation_source=str(decision.get("candidate_generation_source") or "deterministic_fallback"),
            final_source="deterministic_fallback",
            ledger_event_refs=ledger_event_refs,
        )
        decision["llm_candidate_generation_artifact_path"] = artifact_path
        return fallback_candidates


def _build_initial_manifest(
    *,
    workflow_id: str,
    dense_reference_run_dir: str,
    stage_selection_result: Dict[str, Any],
    initial_config: Dict[str, float],
    branch_budget_steps: int,
    t_max: int,
    adaptive_method: Dict[str, Any],
) -> Dict[str, Any]:
    sparse_summary = deepcopy(stage_selection_result.get("sparse_summary") or {})
    env_key = str(sparse_summary.get("env_key") or adaptive_method.get("env_key") or "")
    pbrs_version = str(
        adaptive_method.get("pbrs_version")
        or sparse_summary.get("pbrs_version")
        or ""
    )
    checkpoints = list(((stage_selection_result.get("selection") or {}).get("selected_checkpoints")) or [])
    rounds_per_decision = max(1, int(adaptive_method.get("stage3_branch_rounds") or 2))
    candidates_per_round = max(
        1,
        int(
            adaptive_method.get("stage3_update_candidates_per_round")
            or adaptive_method.get("stage3_candidates_per_round")
            or 3
        ),
    )
    include_no_change = bool(adaptive_method.get("stage3_include_no_change_control", True))
    no_change_counts = bool(
        adaptive_method.get("stage3_no_change_counts_toward_round_budget", False)
    )
    review_enabled = bool(adaptive_method.get("stage3_candidate_review_enabled", True))
    review_repair_rounds = max(0, int(adaptive_method.get("stage3_candidate_repair_max_rounds") or 1))
    use_winner_branch_promotion = bool(adaptive_method.get("use_winner_branch_promotion", True))
    decisions: List[Dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_name = str(checkpoint.get("name") or f"C{index + 1}")
        decisions.append(
            {
                "decision_id": _decision_id(workflow_id, checkpoint_name),
                "decision_index": index + 1,
                "stage_label": str(checkpoint.get("stage_label")),
                "checkpoint_name": checkpoint_name,
                "checkpoint_step": int(checkpoint.get("step")),
                "env_key": env_key,
                "pbrs_version": pbrs_version,
                "llm_routing": deepcopy(adaptive_method.get("llm_routing") or {}),
                "source_run_dir": None,
                "source_checkpoint_root_dir": None,
                "source_available_steps": [],
                "current_config": None,
                "candidate_generation": None,
                "llm_candidate_generation_status": None,
                "llm_candidate_generation_cache_path": None,
                "llm_candidate_generation_validation_errors": [],
                "llm_candidate_generation_underperforming_vs_fixed_reference": None,
                "llm_candidate_generation_fixed_reference_config": {},
                "llm_candidate_generation_requires_reference_or_recovery": False,
                "llm_candidate_generation_provided_reference_or_recovery": False,
                "llm_candidate_generation_repair_used": False,
                "llm_candidate_generation_deterministic_recovery_added": False,
                "policy_guidance_enabled": bool(adaptive_method.get("enable_policy_guidance", False)),
                "policy_guidance_used_pre_checkpoint": False,
                "policy_guidance_used_after_round1": False,
                "pre_checkpoint_integrated_guidance_card_path": None,
                "after_round1_integrated_guidance_card_path": None,
                "policy_guidance_pre_checkpoint_failure_reason": None,
                "policy_guidance_after_round1_failure_reason": None,
                "policy_guidance_fallback_to_reward_only": True,
                "policy_guidance_evidence_keys": [],
                "candidates": [],
                "stage3_branch_rounds": rounds_per_decision,
                "stage3_candidates_per_round": candidates_per_round,
                "stage3_update_candidates_per_round": candidates_per_round,
                "stage3_include_no_change_control": include_no_change,
                "stage3_no_change_counts_toward_round_budget": no_change_counts,
                "stage3_reuse_no_change_across_rounds": bool(
                    adaptive_method.get("stage3_reuse_no_change_across_rounds", True)
                ),
                "stage3_candidate_review_enabled": review_enabled,
                "stage3_candidate_repair_max_rounds": review_repair_rounds,
                "stage3_min_effective_update_candidates_per_checkpoint": int(
                    adaptive_method.get("stage3_min_effective_update_candidates_per_checkpoint") or 6
                ),
                "stage3_require_nontrivial_config_delta": bool(
                    adaptive_method.get("stage3_require_nontrivial_config_delta", True)
                ),
                "stage3_max_fallback_style_candidates_per_checkpoint": int(
                    adaptive_method.get("stage3_max_fallback_style_candidates_per_checkpoint") or 1
                ),
                "stage3_require_candidate_type_diversity": bool(
                    adaptive_method.get("stage3_require_candidate_type_diversity", True)
                ),
                "policy_guided_stage3_require_behavior_evidence": bool(
                    adaptive_method.get("policy_guided_stage3_require_behavior_evidence", True)
                ),
                "use_winner_branch_promotion": use_winner_branch_promotion,
                "rounds": [
                    {
                        "round_id": round_id,
                        "status": "pending",
                        "candidate_generation_status": "pending",
                        "candidate_review_status": "pending",
                        "critic_analysis_status": "pending",
                        "policy_guidance": {
                            "policy_guidance_used": False,
                            "policy_guidance_source": "reward_only_fallback",
                            "policy_guidance_fallback_to_reward_only": True,
                            "policy_guidance_evidence_keys": [],
                            "policy_guidance_failure_reason": None,
                            "integrated_guidance_card_path": None,
                            "integrated_guidance_card": {},
                            "behavior_summary_available": False,
                            "behavior_summary_source": "unavailable",
                            "intervention_point": "",
                        },
                        "generator_candidates": [],
                        "candidates": [],
                        "analysis": {},
                    }
                    for round_id in range(1, rounds_per_decision + 1)
                ],
                "critic_pre_diagnosis_status": "pending",
                "critic_pre_diagnosis": {},
                "round1_result_summary": {},
                "llm_result_diagnosis_status": None,
                "llm_result_diagnosis_cache_path": None,
                "llm_result_diagnosis_validation_errors": [],
                "llm_recommended_winner": None,
                "llm_ranking": [],
                "winner_candidate_id": None,
                "final_selected_winner": None,
                "final_decision_source": None,
                "decision_source": None,
                "deterministic_winner": None,
                "deterministic_gate_completed": False,
                "no_change_gate_overrode_llm": False,
                "fallback_used": False,
                "fallback_reason": None,
                "winner_config": None,
                "winner_score": None,
                "no_change_score": None,
                "replacement_accepted": None,
                "selected_winner_candidate_id": None,
                "selected_winner_branch_run_id": None,
                "selected_winner_endpoint_checkpoint_path": None,
                "selected_winner_endpoint_checkpoint_step": None,
                "promoted_to_mainline": False,
                "promotion_reason": None,
                "promotion_failure_reason": None,
                "decision_summary_path": None,
                "continuation": {
                    "target_step": int(checkpoints[index + 1].get("step")) if index + 1 < len(checkpoints) else int(t_max),
                    "status": "pending",
                    "run_id": None,
                    "run_dir": None,
                    "result_json": None,
                    "last_seen_sacred_status": None,
                    "last_seen_process_alive": None,
                    "launch_attempts": 0,
                },
                "completed": False,
            }
        )
    return {
        "workflow_id": workflow_id,
        "workflow_kind": "adaptive_checkpoint_replacement",
        "method_version": "phase1_adaptive_replacement_v1",
        "dense_reference_run_dir": dense_reference_run_dir,
        "stage_selection_result": deepcopy(stage_selection_result),
        "branch_budget_steps": int(branch_budget_steps),
        "t_max": int(t_max),
        "initial_config": deepcopy(initial_config),
        "current_config": deepcopy(initial_config),
        "adaptive_method": deepcopy(adaptive_method),
        "first_change_baseline_launched": False,
        "first_change_baseline_run_id": None,
        "initial_mainline_seed": {
            "target_step": int(checkpoints[0].get("step")) if checkpoints else None,
            "status": "pending" if checkpoints else "completed",
            "run_id": None,
            "run_dir": None,
            "result_json": None,
            "last_seen_sacred_status": None,
            "last_seen_process_alive": None,
            "launch_attempts": 0,
            "checkpoint_root_dir": None,
            "available_checkpoint_steps": [],
            "latest_checkpoint_step": None,
        },
        "first_change_baseline": {
            "status": "pending",
            "run_id": None,
            "run_dir": None,
            "result_json": None,
            "last_seen_sacred_status": None,
            "last_seen_process_alive": None,
            "launch_attempts": 0,
            "source_decision_id": None,
            "source_checkpoint_step": None,
            "initial_dense_config": deepcopy(initial_config),
            "baseline_type": "fixed_initial_from_first_change_checkpoint",
            "run_type": "first_change_fixed_initial_baseline",
        },
        "decisions": decisions,
        "readiness": {
            "all_decisions_completed": False,
            "final_spec_ready": False,
            "final_mainline_completed": False,
        },
        "artifacts": {
            "adaptive_final_spec_json": None,
            "final_mainline_run_json": None,
        },
    }


def _effective_llm_stage3_max_candidates(adaptive_method: Dict[str, Any]) -> int:
    configured = adaptive_method.get("llm_stage3_max_candidates")
    if configured is None:
        configured = ((adaptive_method.get("candidate_generation") or {}).get("max_candidates_per_checkpoint"))
    return max(1, int(configured or 9))


def _round_payload(decision: Dict[str, Any], round_id: int) -> Dict[str, Any]:
    rounds = list(decision.get("rounds") or [])
    if not (1 <= round_id <= len(rounds)):
        raise ValueError(f"invalid Stage 3 round_id={round_id}")
    return rounds[round_id - 1]


def _candidate_identity(candidate: Dict[str, Any]) -> tuple[Any, ...]:
    if str(candidate.get("pbrs_version") or "") == "rware_pbrs_v2":
        normalized = _normalize_rware_stage_config(
            {
                "candidate_id": str(candidate.get("candidate_id") or "identity"),
                "candidate_type": str(candidate.get("candidate_type") or "identity"),
                "pbrs_version": "rware_pbrs_v2",
                "mode": candidate.get("mode"),
                "beta": candidate.get("beta"),
                "active_terms": list(candidate.get("active_terms") or []),
                "weights": deepcopy(candidate.get("weights") or {}),
                "gamma": candidate.get("gamma", 0.99),
                "evidence_keys_used": list(candidate.get("evidence_keys_used") or []),
            }
        )
        return (
            "rware_pbrs_v2",
            round(float(normalized["beta"]), 10),
            str(normalized.get("mode") or ""),
            tuple(normalized.get("active_terms") or []),
            tuple(
                round(float((normalized.get("weights") or {}).get(term, 0.0)), 10)
                for term in RWARE_STAGE3_TERMS
            ),
        )
    return (float(candidate["beta"]), float(candidate["wc"]))


def _merge_stage3_unique_candidates(
    *,
    existing_candidates: List[Dict[str, Any]],
    new_candidates: List[Dict[str, Any]],
    validation_errors: List[str],
    max_candidates: int,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = [deepcopy(item) for item in existing_candidates]
    seen = {_candidate_identity(item) for item in merged}
    for candidate in new_candidates:
        identity = _candidate_identity(candidate)
        if identity in seen:
            validation_errors.append(
                f"repair_duplicate_of_existing_valid_candidate:{candidate.get('candidate_id') or '<unknown>'}"
            )
            continue
        seen.add(identity)
        merged.append(deepcopy(candidate))
        if len(merged) >= max_candidates:
            break
    return merged[:max_candidates]


def _round_candidates_completed(round_payload: Dict[str, Any]) -> bool:
    return all(
        str(candidate.get("status")) in {"completed", "completed_invalid", "failed_invalid"}
        for candidate in list(round_payload.get("candidates") or [])
    )


def _round_branch_result_summary(round_payload: Dict[str, Any], decision_rule: Dict[str, Any]) -> Dict[str, Any]:
    ranked = sorted(
        list(round_payload.get("candidates") or []),
        key=lambda item: _candidate_score(item.get("result_record") or {}, decision_rule),
        reverse=True,
    )
    no_change = next((item for item in ranked if bool(item.get("is_no_change"))), None)
    best_non_no_change = next((item for item in ranked if not bool(item.get("is_no_change"))), None)
    return {
        "round_id": round_payload.get("round_id"),
        "ranking": [
            {
                "candidate_id": item.get("candidate_id"),
                "score": list(_candidate_score(item.get("result_record") or {}, decision_rule)),
                "run_id": item.get("run_id"),
            }
            for item in ranked
        ],
        "best_candidate_id": ranked[0].get("candidate_id") if ranked else None,
        "best_non_no_change_candidate_id": best_non_no_change.get("candidate_id") if best_non_no_change else None,
        "no_change_candidate_id": no_change.get("candidate_id") if no_change else None,
    }


def _round_generation_context(
    *,
    decision: Dict[str, Any],
    current_config: Dict[str, float],
    round_id: int,
    decision_rule: Dict[str, Any],
) -> Dict[str, Any]:
    if round_id <= 1:
        return deepcopy(current_config)
    round1 = _round_payload(decision, 1)
    ranked = sorted(
        [item for item in list(round1.get("candidates") or []) if item.get("result_record")],
        key=lambda item: _candidate_score(item.get("result_record") or {}, decision_rule),
        reverse=True,
    )
    anchor = ranked[0] if ranked else None
    if anchor is None:
        return deepcopy(current_config)
    return {
        "beta": float(anchor["beta"]),
        "wc": float(anchor["wc"]),
        "wp": float(anchor["wp"]),
    }


def _review_stage3_candidates(
    *,
    round_payload: Dict[str, Any],
    current_config: Dict[str, float],
    round_id: int,
    decision: Dict[str, Any],
) -> None:
    unique: List[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    duplicate_count = 0
    current_identity = _candidate_identity(
        _build_rware_no_change_candidate(current_config=current_config)
        if is_rware_workflow(
            env_key=decision.get("env_key"),
            pbrs_version=decision.get("pbrs_version") or current_config.get("pbrs_version"),
        )
        else {
            "beta": float(current_config["beta"]),
            "wc": float(current_config["wc"]),
        }
    )
    round_cfg = _decision_round_config(decision)
    update_target = round_cfg["round1_generated"] if round_id == 1 else round_cfg["round2_generated"]
    policy_guidance = deepcopy(round_payload.get("policy_guidance") or {})
    for candidate in list(round_payload.get("generator_candidates") or []):
        key = _candidate_identity(candidate)
        if key == current_identity:
            duplicate_count += 1
            continue
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(candidate)
    if round_id == 1 and bool(decision.get("stage3_include_no_change_control", True)):
        if is_rware_workflow(
            env_key=decision.get("env_key"),
            pbrs_version=decision.get("pbrs_version") or current_config.get("pbrs_version"),
        ):
            no_change = _build_rware_no_change_candidate(current_config=current_config)
        elif is_lbf_workflow(
            env_key=decision.get("env_key"),
            pbrs_version=decision.get("pbrs_version") or current_config.get("pbrs_version"),
        ):
            no_change = _build_lbf_no_change_candidate(current_config=current_config)
        else:
            no_change = _normalize_candidate_config(
                beta=current_config["beta"],
                wc=current_config["wc"],
            )
            assert no_change is not None
            no_change["candidate_id"] = "no_change"
            no_change["candidate_type"] = "no_change_control"
            no_change["source_config"] = "current_mainline"
            no_change["is_no_change"] = True
        _attach_env_specific_pbrs_metadata(
            no_change,
            env_key=decision.get("env_key"),
            pbrs_version=decision.get("pbrs_version"),
        )
        if _candidate_identity(no_change) not in seen:
            unique.insert(0, no_change)
    reviewed: List[Dict[str, Any]] = []
    existing_identities = {
        _candidate_identity(item)
        for item in _all_round_candidates(decision)
        if int(item.get("round_id") or 0) != int(round_id)
    }
    for index, candidate in enumerate(unique, start=1):
        item = deepcopy(candidate)
        item.setdefault("source_config", "generator_round" if not item.get("is_no_change") else "current_mainline")
        _attach_env_specific_pbrs_metadata(
            item,
            env_key=decision.get("env_key"),
            pbrs_version=decision.get("pbrs_version"),
        )
        if not item.get("is_no_change"):
            item.setdefault(
                "candidate_type",
                _default_stage3_candidate_type(
                    decision=decision,
                    round_id=round_id,
                    index=index - 1,
                    policy_guidance_used=bool(policy_guidance.get("policy_guidance_used", False)),
                ),
            )
        item.setdefault("is_no_change", False)
        item.setdefault("evidence_keys_used", _stage3_candidate_evidence_keys(item))
        if item.get("is_no_change"):
            item["effective_update_candidate"] = False
            item["fallback_style_candidate"] = False
            item["candidate_category"] = "no_change_control"
        else:
            if bool(policy_guidance.get("policy_guidance_used", False)) and bool(
                decision.get("policy_guided_stage3_require_behavior_evidence", True)
            ):
                if not item.get("evidence_keys_used"):
                    continue
            if (not bool(policy_guidance.get("policy_guidance_used", False))) and _candidate_mentions_forbidden_policy_evidence(item):
                continue
            if bool(decision.get("stage3_require_nontrivial_config_delta", True)) and not _is_nontrivial_stage3_update(
                current_config=current_config,
                candidate=item,
            ):
                item["effective_update_candidate"] = False
                item["effective_update_reason"] = "near_duplicate_or_non_effective_update"
                continue
            item["effective_update_candidate"] = True
            item["candidate_category"] = _stage3_candidate_category(str(item.get("candidate_type") or ""))
            item["fallback_style_candidate"] = _is_fallback_style_stage3_candidate(item)
            if _candidate_identity(item) in existing_identities:
                continue
            existing_identities.add(_candidate_identity(item))
        item["round_id"] = round_id
        item["branch_id"] = f"round{round_id}_{item['candidate_id']}"
        reviewed.append(item)
    effective_updates = [item for item in reviewed if not bool(item.get("is_no_change"))]
    if len(effective_updates) < update_target:
        reviewed.extend(
            _build_stage3_deterministic_update_topups(
                decision=decision,
                current_config=current_config,
                round_id=round_id,
                existing_identities=existing_identities,
                existing_candidates=effective_updates,
                target_count=update_target,
                policy_guidance=policy_guidance,
            )
        )
        for item in reviewed:
            item.setdefault("round_id", round_id)
            item.setdefault("branch_id", f"round{round_id}_{item['candidate_id']}")
            item.setdefault("effective_update_candidate", not bool(item.get("is_no_change")))
            item.setdefault(
                "candidate_category",
                "no_change_control" if bool(item.get("is_no_change")) else _stage3_candidate_category(str(item.get("candidate_type") or "")),
            )
            item.setdefault("fallback_style_candidate", False)
    round_updates = [item for item in reviewed if not bool(item.get("is_no_change"))][:update_target]
    no_change_items = [item for item in reviewed if bool(item.get("is_no_change"))][:1]
    reviewed = no_change_items + round_updates
    fallback_style_count = sum(
        1 for item in _all_round_candidates(decision) + reviewed if bool(item.get("fallback_style_candidate"))
    )
    round_payload["candidates"] = reviewed
    round_payload["candidate_review_status"] = "completed"
    round_payload["candidate_review"] = {
        "deduplicated_count": duplicate_count,
        "final_candidate_ids": [item.get("candidate_id") for item in reviewed],
        "effective_update_candidates": len(round_updates),
        "no_change_present": any(bool(item.get("is_no_change")) for item in reviewed),
        "fallback_style_candidates_seen": fallback_style_count,
        "policy_guidance_candidate_collapsed": bool(
            duplicate_count > 0
            and bool((round_payload.get("policy_guidance") or {}).get("policy_guidance_used", False))
        ),
        "collapsed_reason": (
            "stage3_policy_guided_candidates_deduplicated_to_same_beta_wc"
            if duplicate_count > 0
            and bool((round_payload.get("policy_guidance") or {}).get("policy_guidance_used", False))
            else ""
        ),
    }


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return _load_json(path)


def _decision_source_fields_missing(decision: Dict[str, Any]) -> bool:
    return not bool(decision.get("source_run_dir")) or not bool(
        decision.get("source_checkpoint_root_dir")
    ) or not bool(normalize_checkpoint_steps(decision.get("source_available_steps")))


def _checkpoint_root_from_path(checkpoint_path: Optional[str]) -> Optional[str]:
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        return None
    return str(Path(checkpoint_path).parent)


def _extract_source_payload_from_continuation_result(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    run_reference = payload.get("run_reference") or {}
    run_metadata = ((payload.get("metrics_summary") or {}).get("run_metadata") or {})
    latest_path = (
        run_reference.get("latest_model_path")
        or payload.get("endpoint_checkpoint_path")
        or run_metadata.get("latest_model_path")
    )
    checkpoint_root_dir = (
        run_reference.get("checkpoint_root_dir")
        or run_metadata.get("checkpoint_root_dir")
        or _checkpoint_root_from_path(latest_path)
    )
    available_steps = normalize_checkpoint_steps(
        run_reference.get("available_checkpoint_steps")
        or run_metadata.get("available_checkpoint_steps")
        or payload.get("available_checkpoint_steps")
        or payload.get("saved_model_steps")
        or []
    )
    latest_step = (
        run_reference.get("latest_checkpoint_step")
        or run_metadata.get("latest_checkpoint_step")
        or payload.get("actual_checkpoint_step")
    )
    try:
        latest_step_int = int(latest_step) if latest_step is not None else None
    except (TypeError, ValueError):
        latest_step_int = None
    if latest_step_int is not None:
        available_steps = sorted(set(available_steps) | {latest_step_int})
    return {
        "source_run_dir": run_reference.get("run_dir") or payload.get("run_dir") or payload.get("source_run_dir"),
        "source_checkpoint_root_dir": checkpoint_root_dir,
        "source_available_steps": available_steps,
        "source_latest_checkpoint_step": latest_step_int,
        "source_latest_checkpoint_path": latest_path,
    }


def _recover_source_payload_from_run_dir(run_dir: Optional[str]) -> Dict[str, Any]:
    if not isinstance(run_dir, str) or not run_dir:
        return {}
    run_path = Path(run_dir)
    if not run_path.exists():
        return {"source_run_dir": run_dir}
    run_reference = _build_run_reference_from_run_dir(run_path)
    latest_step = run_reference.get("latest_checkpoint_step")
    checkpoint_root_dir = run_reference.get("checkpoint_root_dir")
    latest_path = (
        str(Path(str(checkpoint_root_dir)) / str(int(latest_step)))
        if checkpoint_root_dir and latest_step is not None
        else None
    )
    return {
        "source_run_dir": str(run_path),
        "source_checkpoint_root_dir": checkpoint_root_dir,
        "source_available_steps": list(run_reference.get("available_checkpoint_steps") or []),
        "source_latest_checkpoint_step": latest_step,
        "source_latest_checkpoint_path": latest_path,
    }


def _recover_next_decision_source_from_completed_decision(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    continuation = decision.get("continuation") or {}
    available_steps = normalize_checkpoint_steps(continuation.get("available_checkpoint_steps"))
    latest_step = continuation.get("effective_checkpoint_step") or continuation.get("latest_checkpoint_step")
    try:
        latest_step_int = int(latest_step) if latest_step is not None else None
    except (TypeError, ValueError):
        latest_step_int = None
    if latest_step_int is not None:
        available_steps = sorted(set(available_steps) | {latest_step_int})
    checkpoint_root_dir = (
        continuation.get("checkpoint_root_dir")
        or continuation.get("effective_checkpoint_root_dir")
        or _checkpoint_root_from_path(continuation.get("effective_checkpoint_path"))
        or _checkpoint_root_from_path(continuation.get("latest_checkpoint_path"))
    )
    payload = {
        "source_run_dir": continuation.get("run_dir"),
        "source_checkpoint_root_dir": checkpoint_root_dir,
        "source_available_steps": available_steps,
        "source_latest_checkpoint_step": latest_step_int,
        "source_latest_checkpoint_path": (
            continuation.get("effective_checkpoint_path") or continuation.get("latest_checkpoint_path")
        ),
    }
    if payload["source_run_dir"] and payload["source_checkpoint_root_dir"] and payload["source_available_steps"]:
        return payload

    result_path_value = continuation.get("result_json")
    result_path = Path(str(result_path_value)) if isinstance(result_path_value, str) and result_path_value else None
    if result_path is None or not result_path.exists():
        result_path = _continuation_result_path(
            workflow_dir,
            int(decision["decision_index"]),
            str(decision["stage_label"]),
        )
    result_payload = _load_json_if_exists(result_path)
    if result_payload:
        recovered = _extract_source_payload_from_continuation_result(result_payload)
        if recovered.get("source_run_dir") and recovered.get("source_checkpoint_root_dir") and recovered.get(
            "source_available_steps"
        ):
            return recovered
        payload = {
            "source_run_dir": payload.get("source_run_dir") or recovered.get("source_run_dir"),
            "source_checkpoint_root_dir": payload.get("source_checkpoint_root_dir")
            or recovered.get("source_checkpoint_root_dir"),
            "source_available_steps": payload.get("source_available_steps") or recovered.get("source_available_steps"),
            "source_latest_checkpoint_step": payload.get("source_latest_checkpoint_step")
            or recovered.get("source_latest_checkpoint_step"),
            "source_latest_checkpoint_path": payload.get("source_latest_checkpoint_path")
            or recovered.get("source_latest_checkpoint_path"),
        }
        if payload["source_run_dir"] and payload["source_checkpoint_root_dir"] and payload["source_available_steps"]:
            return payload

    recovered_from_run = _recover_source_payload_from_run_dir(payload.get("source_run_dir"))
    return {
        "source_run_dir": payload.get("source_run_dir") or recovered_from_run.get("source_run_dir"),
        "source_checkpoint_root_dir": payload.get("source_checkpoint_root_dir")
        or recovered_from_run.get("source_checkpoint_root_dir"),
        "source_available_steps": payload.get("source_available_steps") or recovered_from_run.get("source_available_steps") or [],
        "source_latest_checkpoint_step": payload.get("source_latest_checkpoint_step")
        or recovered_from_run.get("source_latest_checkpoint_step"),
        "source_latest_checkpoint_path": payload.get("source_latest_checkpoint_path")
        or recovered_from_run.get("source_latest_checkpoint_path"),
    }


def _apply_source_payload_to_decision(decision: Dict[str, Any], source_payload: Dict[str, Any]) -> None:
    run_dir = source_payload.get("source_run_dir")
    checkpoint_root_dir = source_payload.get("source_checkpoint_root_dir")
    available_steps = normalize_checkpoint_steps(source_payload.get("source_available_steps"))
    latest_step = source_payload.get("source_latest_checkpoint_step")
    latest_path = source_payload.get("source_latest_checkpoint_path")
    if run_dir:
        decision["source_run_dir"] = str(run_dir)
    if checkpoint_root_dir:
        decision["source_checkpoint_root_dir"] = str(checkpoint_root_dir)
    if available_steps:
        decision["source_available_steps"] = list(available_steps)
    if latest_step is not None:
        decision["source_latest_checkpoint_step"] = int(latest_step)
    if latest_path:
        decision["source_latest_checkpoint_path"] = str(latest_path)


def _propagate_completed_continuation_to_next_decision(
    *,
    workflow_dir: Path,
    decisions: List[Dict[str, Any]],
    decision_index: int,
) -> None:
    if decision_index + 1 >= len(decisions):
        return
    next_decision = decisions[decision_index + 1]
    if not _decision_source_fields_missing(next_decision):
        return
    source_payload = _recover_next_decision_source_from_completed_decision(
        workflow_dir=workflow_dir,
        decision=decisions[decision_index],
    )
    _apply_source_payload_to_decision(next_decision, source_payload)


def _decision_source_run_reference(
    *,
    decision: Dict[str, Any],
    fallback_run_dir: str,
) -> Dict[str, Any]:
    source_run_dir = decision.get("source_run_dir") or fallback_run_dir
    checkpoint_root_dir = decision.get("source_checkpoint_root_dir")
    available_steps = normalize_checkpoint_steps(decision.get("source_available_steps"))
    latest_checkpoint_step = decision.get("source_latest_checkpoint_step")
    latest_checkpoint_path = decision.get("source_latest_checkpoint_path")
    if checkpoint_root_dir and available_steps:
        if latest_checkpoint_step is None:
            latest_checkpoint_step = available_steps[-1]
        return {
            "run_dir": str(source_run_dir),
            "checkpoint_root_dir": str(checkpoint_root_dir),
            "available_checkpoint_steps": list(available_steps),
            "latest_checkpoint_step": int(latest_checkpoint_step) if latest_checkpoint_step is not None else None,
            "latest_model_path": latest_checkpoint_path,
            "baseline_readiness": {},
        }
    return _build_run_reference_from_run_dir(source_run_dir)


def _decision_dir(workflow_dir: Path, decision_index: int, stage_label: str) -> Path:
    return workflow_dir / f"decision_{decision_index:02d}_{stage_label}"


def _decision_summary_path(workflow_dir: Path, decision_index: int, stage_label: str) -> Path:
    return _decision_dir(workflow_dir, decision_index, stage_label) / "decision_summary.json"


def _continuation_result_path(workflow_dir: Path, decision_index: int, stage_label: str) -> Path:
    return _decision_dir(workflow_dir, decision_index, stage_label) / "mainline_continuation_result.json"


def _candidate_branch_result_path(workflow_dir: Path, decision_index: int, stage_label: str, candidate_id: str) -> Path:
    return _decision_dir(workflow_dir, decision_index, stage_label) / "branches" / candidate_id / "branch_result.json"


def _initial_mainline_seed_result_path(workflow_dir: Path) -> Path:
    return workflow_dir / "initial_mainline_seed_result.json"


def _candidate_train_config_for_decision(
    *,
    base_train_config: Dict[str, Any],
    checkpoint_root_dir: str,
    available_steps: List[int],
    checkpoint_step: int,
    workflow_id: str,
    decision: Dict[str, Any],
    candidate: Dict[str, Any],
    branch_budget_steps: int,
    branch_use_cuda: Optional[bool],
    local_results_path: str,
    launcher: EPyMARLTrainLauncher,
) -> Dict[str, Any]:
    override_env_args = _build_native_pbrs_env_args(candidate)
    interval = _recommended_short_resume_interval(int(branch_budget_steps))
    plan = launcher.build_checkpoint_resume_plan(
        base_train_config=base_train_config,
        run_reference={
            "checkpoint_root_dir": checkpoint_root_dir,
            "available_checkpoint_steps": available_steps,
        },
        requested_step=checkpoint_step,
        label_suffix=(
            f"_{workflow_id}_{decision['stage_label']}_{candidate['candidate_id']}_adaptive_branch"
        ),
        use_dense_reward=False,
        reward_module_path="",
        alpha_policy_config={"type": "constant", "value": 1.0},
        workflow_id=workflow_id,
        workflow_round=int(decision["decision_index"]),
        reward_paradigm="pbrs",
        active_pbrs_field="adaptive_combo",
        active_pbrs_field_source="adaptive_checkpoint_replacement",
        candidate_value=None,
        active_checkpoint_context={
            "name": decision["checkpoint_name"],
            "step": checkpoint_step,
            "stage_label": decision["stage_label"],
        },
        active_field_carryover_context={"current_config": deepcopy(decision["current_config"] or {})},
        candidate_selection_context={
            "candidate_type": "adaptive_replacement_combo",
            "candidate_id": candidate["candidate_id"],
            "candidate_config": (
                {
                    "pbrs_version": "rware_pbrs_v2",
                    "mode": candidate.get("mode"),
                    "beta": candidate["beta"],
                    "active_terms": list(candidate.get("active_terms") or []),
                    "weights": deepcopy(candidate.get("weights") or {}),
                }
                if str(candidate.get("pbrs_version") or "") == "rware_pbrs_v2"
                else {
                    "beta": candidate["beta"],
                    "wc": candidate["wc"],
                    "wp": candidate["wp"],
                }
            ),
            "is_no_change": bool(candidate["is_no_change"]),
        },
        phase_name="adaptive_branch",
        save_model=True,
        save_model_interval=max(1, int(branch_budget_steps)),
        save_final_model=True,
        local_results_path=local_results_path,
        override_env_args=override_env_args,
        override_overrides={
            "t_max": int(checkpoint_step) + int(branch_budget_steps),
            "test_interval": interval,
            "log_interval": interval,
            "runner_log_interval": interval,
            "learner_log_interval": interval,
            **({"use_cuda": bool(branch_use_cuda)} if branch_use_cuda is not None else {}),
        },
    )
    _inject_candidate_metadata_into_plan(plan=plan, candidate=candidate)
    plan["resume_train_config"]["defer_initial_model_save"] = True
    return plan


def _continuation_plan_for_decision(
    *,
    base_train_config: Dict[str, Any],
    checkpoint_root_dir: str,
    available_steps: List[int],
    checkpoint_step: int,
    workflow_id: str,
    decision: Dict[str, Any],
    winner_config: Dict[str, float],
    target_step: int,
    mainline_use_cuda: Optional[bool],
    local_results_path: str,
    save_model_interval: int,
    launcher: EPyMARLTrainLauncher,
) -> Dict[str, Any]:
    override_env_args = _build_native_pbrs_env_args(winner_config)
    interval = _recommended_short_resume_interval(int(target_step) - int(checkpoint_step))
    plan = launcher.build_checkpoint_resume_plan(
        base_train_config=base_train_config,
        run_reference={
            "checkpoint_root_dir": checkpoint_root_dir,
            "available_checkpoint_steps": available_steps,
        },
        requested_step=checkpoint_step,
        label_suffix=f"_{workflow_id}_{decision['stage_label']}_adaptive_mainline",
        use_dense_reward=False,
        reward_module_path="",
        alpha_policy_config={"type": "constant", "value": 1.0},
        workflow_id=f"{workflow_id}_mainline",
        workflow_round=int(decision["decision_index"]),
        reward_paradigm="pbrs",
        active_pbrs_field="adaptive_mainline",
        active_pbrs_field_source="adaptive_checkpoint_replacement",
        candidate_value=None,
        active_checkpoint_context={
            "name": decision["checkpoint_name"],
            "step": checkpoint_step,
            "stage_label": decision["stage_label"],
        },
        active_field_carryover_context={"winner_config": deepcopy(winner_config)},
        candidate_selection_context={
            "candidate_type": "adaptive_mainline_continuation",
            "winner_config": deepcopy(winner_config),
        },
        phase_name="adaptive_mainline",
        save_model=True,
        save_model_interval=int(save_model_interval),
        save_final_model=True,
        local_results_path=local_results_path,
        override_env_args=override_env_args,
        override_overrides={
            "t_max": int(target_step),
            "test_interval": interval,
            "log_interval": interval,
            "runner_log_interval": interval,
            "learner_log_interval": interval,
            **({"use_cuda": bool(mainline_use_cuda)} if mainline_use_cuda is not None else {}),
        },
    )
    _inject_candidate_metadata_into_plan(plan=plan, candidate=winner_config)
    plan["resume_train_config"]["defer_initial_model_save"] = True
    return plan


def _build_initial_mainline_seed_plan(
    *,
    base_train_config: Dict[str, Any],
    checkpoint_root_dir: str,
    available_steps: List[int],
    source_checkpoint_step: int,
    workflow_id: str,
    checkpoint_name: str,
    checkpoint_step: int,
    initial_config: Dict[str, float],
    mainline_use_cuda: Optional[bool],
    local_results_path: str,
    launcher: EPyMARLTrainLauncher,
) -> Dict[str, Any]:
    plan = launcher.build_checkpoint_resume_plan(
        base_train_config=base_train_config,
        run_reference={
            "checkpoint_root_dir": checkpoint_root_dir,
            "available_checkpoint_steps": available_steps,
        },
        requested_step=int(source_checkpoint_step),
        label_suffix=f"_{workflow_id}_initial_mainline_seed",
        use_dense_reward=False,
        reward_module_path="",
        alpha_policy_config={"type": "constant", "value": 1.0},
        workflow_id=f"{workflow_id}_initial_mainline_seed",
        workflow_round=0,
        reward_paradigm="pbrs",
        active_pbrs_field="adaptive_initial_mainline_seed",
        active_pbrs_field_source="adaptive_checkpoint_replacement",
        candidate_value=None,
        active_checkpoint_context={
            "name": checkpoint_name,
            "step": int(checkpoint_step),
            "stage_label": "initial_mainline_seed",
        },
        active_field_carryover_context={"initial_config": deepcopy(initial_config)},
        candidate_selection_context={
            "candidate_type": "initial_mainline_seed",
            "initial_config": deepcopy(initial_config),
            "source_checkpoint_step": int(source_checkpoint_step),
        },
        phase_name="adaptive_initial_mainline_seed",
        save_model=True,
        save_model_interval=max(1, int(checkpoint_step) - int(source_checkpoint_step)),
        save_final_model=True,
        local_results_path=local_results_path,
        override_env_args=_build_native_pbrs_env_args(initial_config),
        override_overrides={
            "t_max": int(checkpoint_step),
            "test_interval": _recommended_short_resume_interval(int(checkpoint_step) - int(source_checkpoint_step)),
            "log_interval": _recommended_short_resume_interval(int(checkpoint_step) - int(source_checkpoint_step)),
            "runner_log_interval": _recommended_short_resume_interval(int(checkpoint_step) - int(source_checkpoint_step)),
            "learner_log_interval": _recommended_short_resume_interval(int(checkpoint_step) - int(source_checkpoint_step)),
            **({"use_cuda": bool(mainline_use_cuda)} if mainline_use_cuda is not None else {}),
        },
    )
    plan["resume_train_config"]["defer_initial_model_save"] = True
    return plan


def _reconcile_initial_mainline_seed(
    *,
    workflow_dir: Path,
    manifest: Dict[str, Any],
    launcher: EPyMARLTrainLauncher,
    base_train_config: Dict[str, Any],
    workflow_id: str,
    initial_config: Dict[str, float],
    checkpoint_name: str,
    checkpoint_step: int,
    checkpoint_root_dir: str,
    available_steps: List[int],
    source_checkpoint_step: int,
    mainline_use_cuda: Optional[bool],
    local_results_path: str,
) -> None:
    seed_state = manifest.get("initial_mainline_seed") or {}
    result_path = _initial_mainline_seed_result_path(workflow_dir)
    seed_state["result_json"] = str(result_path)
    plan = _build_initial_mainline_seed_plan(
        base_train_config=base_train_config,
        workflow_id=workflow_id,
        checkpoint_name=checkpoint_name,
        checkpoint_step=checkpoint_step,
        checkpoint_root_dir=checkpoint_root_dir,
        available_steps=available_steps,
        source_checkpoint_step=source_checkpoint_step,
        initial_config=initial_config,
        mainline_use_cuda=mainline_use_cuda,
        local_results_path=local_results_path,
        launcher=launcher,
    )
    state = _classify_existing_candidate_state(
        launcher=launcher,
        branch_plan=plan,
        branch_result_path=result_path,
    )
    seed_state["status"] = str(state.get("status") or "planned")
    run_reference = state.get("run_reference") or {}
    if seed_state["status"] == "planned" and run_reference.get("run_id") is not None:
        seed_state["status"] = "running"
    seed_state["run_id"] = run_reference.get("run_id")
    seed_state["run_dir"] = run_reference.get("run_dir")
    seed_state["last_seen_sacred_status"] = state.get("last_seen_sacred_status")
    seed_state["last_seen_process_alive"] = state.get("last_seen_process_alive")
    seed_state["checkpoint_root_dir"] = run_reference.get("checkpoint_root_dir")
    seed_state["available_checkpoint_steps"] = list(run_reference.get("available_checkpoint_steps") or [])
    seed_state["latest_checkpoint_step"] = run_reference.get("latest_checkpoint_step")
    if _checkpoint_target_reached(run_reference, checkpoint_step):
        seed_state["status"] = "completed"
    manifest["initial_mainline_seed"] = seed_state


def _dispatch_initial_mainline_seed(
    *,
    workflow_dir: Path,
    manifest: Dict[str, Any],
    launcher: EPyMARLTrainLauncher,
    base_train_config: Dict[str, Any],
    workflow_id: str,
    initial_config: Dict[str, float],
    checkpoint_name: str,
    checkpoint_step: int,
    checkpoint_root_dir: str,
    available_steps: List[int],
    source_checkpoint_step: int,
    mainline_use_cuda: Optional[bool],
    local_results_path: str,
) -> None:
    seed_state = manifest.get("initial_mainline_seed") or {}
    if seed_state.get("status") not in {"pending", "planned"}:
        return
    result_path = _initial_mainline_seed_result_path(workflow_dir)
    plan = _build_initial_mainline_seed_plan(
        base_train_config=base_train_config,
        workflow_id=workflow_id,
        checkpoint_name=checkpoint_name,
        checkpoint_step=checkpoint_step,
        checkpoint_root_dir=checkpoint_root_dir,
        available_steps=available_steps,
        source_checkpoint_step=source_checkpoint_step,
        initial_config=initial_config,
        mainline_use_cuda=mainline_use_cuda,
        local_results_path=local_results_path,
        launcher=launcher,
    )
    launch_result = _launch_candidate_background_and_bind_run(
        launcher=launcher,
        branch_plan=plan,
        branch_result_path=result_path,
    )
    run_reference = launch_result.get("run_reference") or {}
    seed_state["status"] = "running"
    seed_state["run_id"] = run_reference.get("run_id")
    seed_state["run_dir"] = run_reference.get("run_dir")
    seed_state["result_json"] = str(result_path)
    seed_state["last_seen_sacred_status"] = "RUNNING" if run_reference.get("run_id") is not None else None
    seed_state["last_seen_process_alive"] = True if run_reference.get("run_id") is not None else None
    seed_state["launch_attempts"] = int(seed_state.get("launch_attempts") or 0) + 1
    manifest["initial_mainline_seed"] = seed_state


def _continuation_source_for_decision(
    *,
    decision: Dict[str, Any],
    checkpoint_root_dir: str,
    available_steps: List[int],
    checkpoint_step: int,
) -> Dict[str, Any]:
    winner_candidate_id = str(decision.get("winner_candidate_id") or "")
    candidates = list(decision.get("candidates") or [])
    winner = next(
        (candidate for candidate in candidates if str(candidate.get("candidate_id") or "") == winner_candidate_id),
        None,
    )
    if winner is not None:
        winner_root = winner.get("checkpoint_root_dir")
        winner_steps = list(winner.get("available_checkpoint_steps") or [])
        winner_step = winner.get("latest_checkpoint_step")
        winner_run_dir = winner.get("run_dir")
        if winner_root and winner_steps and winner_step:
            return {
                "checkpoint_root_dir": str(winner_root),
                "available_checkpoint_steps": winner_steps,
                "checkpoint_step": int(winner_step),
                "source_run_dir": winner_run_dir,
            }
    return {
        "checkpoint_root_dir": checkpoint_root_dir,
        "available_checkpoint_steps": list(available_steps),
        "checkpoint_step": int(checkpoint_step),
        "source_run_dir": decision.get("source_run_dir"),
    }


def _continuation_target_already_satisfied(source_checkpoint_step: int, target_step: int) -> bool:
    return int(source_checkpoint_step) >= int(target_step)


def _continuation_result_payload(
    *,
    source: Dict[str, Any],
    decision: Dict[str, Any],
    status: str,
    skipped: bool,
    skip_reason: Optional[str],
) -> Dict[str, Any]:
    source_step = int(source.get("checkpoint_step") or 0)
    source_root = str(source.get("checkpoint_root_dir") or "")
    endpoint_path = str(Path(source_root) / str(source_step)) if source_root and source_step else None
    return {
        "candidate_id": f"{decision.get('checkpoint_name')}_adaptive_mainline_continuation",
        "status": status,
        "skipped": bool(skipped),
        "skip_reason": skip_reason,
        "source_checkpoint_step": source_step,
        "source_checkpoint_root_dir": source_root,
        "source_run_dir": source.get("source_run_dir"),
        "target_step": int((decision.get("continuation") or {}).get("target_step") or 0),
        "actual_checkpoint_step": source_step,
        "endpoint_checkpoint_path": endpoint_path,
        "checkpoint_exists": bool(source_root and source_step),
        "winner_config": deepcopy(decision.get("winner_config") or {}),
        "stage_label": decision.get("stage_label"),
        "checkpoint_name": decision.get("checkpoint_name"),
    }


def _mark_continuation_completed_from_source(
    *,
    continuation: Dict[str, Any],
    source: Dict[str, Any],
    result_path: Path,
    decision: Dict[str, Any],
    reason: str,
    existing_run_reference: Optional[Dict[str, Any]] = None,
) -> None:
    payload = _continuation_result_payload(
        source=source,
        decision=decision,
        status="completed",
        skipped=True,
        skip_reason=reason,
    )
    _save_json(result_path, payload)
    continuation["status"] = "completed"
    continuation["skipped"] = True
    continuation["skip_reason"] = reason
    continuation["run_id"] = (existing_run_reference or {}).get("run_id")
    continuation["run_dir"] = (existing_run_reference or {}).get("run_dir") or source.get("source_run_dir")
    continuation["last_seen_sacred_status"] = (
        (existing_run_reference or {}).get("last_seen_sacred_status")
        if (existing_run_reference or {}).get("run_id") is not None
        else "SKIPPED_TARGET_ALREADY_SATISFIED"
    )
    continuation["last_seen_process_alive"] = (
        (existing_run_reference or {}).get("last_seen_process_alive")
        if (existing_run_reference or {}).get("run_id") is not None
        else False
    )
    continuation["checkpoint_root_dir"] = source.get("checkpoint_root_dir")
    continuation["available_checkpoint_steps"] = list(source.get("available_checkpoint_steps") or [])
    continuation["latest_checkpoint_step"] = int(source.get("checkpoint_step") or 0) or None
    continuation["latest_checkpoint_path"] = payload.get("endpoint_checkpoint_path")
    continuation["effective_checkpoint_step"] = int(source.get("checkpoint_step") or 0)
    continuation["effective_checkpoint_root_dir"] = source.get("checkpoint_root_dir")
    continuation["effective_checkpoint_path"] = payload.get("endpoint_checkpoint_path")


def _decision_complete(decision: Dict[str, Any]) -> bool:
    return bool(decision.get("completed"))


def _decision_branch_search_skipped(decision: Dict[str, Any]) -> bool:
    return bool(((decision.get("fork_bootstrap") or {}).get("skip_branch_search", False)))


def _ensure_decision_candidates(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    current_config: Dict[str, float],
    adaptive_method: Dict[str, Any],
    stage_selection_result: Dict[str, Any],
    dense_reference_selection: Dict[str, Any],
    dense_reference_run_summary: Dict[str, Any],
    branch_budget_steps: int,
    use_real_llm: bool,
    api_key_env: str,
    base_url: str,
    model: str,
    temperature: float,
    llm_timeout: float,
    llm_retry_count: int,
    llm_retry_backoff: float,
) -> None:
    if decision.get("candidates"):
        return
    decision["current_config"] = deepcopy(current_config)
    decision["candidate_generation"] = deepcopy(adaptive_method.get("candidate_generation") or {})
    decision["candidate_generation"]["max_candidates_per_checkpoint"] = _effective_llm_stage3_max_candidates(
        adaptive_method
    )
    decision["candidates"] = _generate_candidates_with_llm_or_fallback(
        workflow_dir=workflow_dir,
        decision=decision,
        current_config=current_config,
        adaptive_method=adaptive_method,
        stage_selection_result=stage_selection_result,
        dense_reference_selection=dense_reference_selection,
        dense_reference_run_summary=dense_reference_run_summary,
        branch_budget_steps=branch_budget_steps,
        use_real_llm=use_real_llm,
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        temperature=temperature,
        llm_timeout=llm_timeout,
        llm_retry_count=llm_retry_count,
        llm_retry_backoff=llm_retry_backoff,
    )


def _deterministic_winner_selection(
    *,
    decision: Dict[str, Any],
    decision_rule: Dict[str, Any],
) -> Dict[str, Any]:
    candidates = list(decision.get("candidates") or [])
    no_change = next((candidate for candidate in candidates if bool(candidate.get("is_no_change"))), None)
    if no_change is None:
        raise ValueError("adaptive replacement decision requires one no-change baseline candidate")
    no_change_score = _candidate_score(no_change.get("result_record") or {}, decision_rule)
    deterministic_winner = max(
        candidates,
        key=lambda item: _candidate_score(item.get("result_record") or {}, decision_rule),
    )
    deterministic_score = _candidate_score(deterministic_winner.get("result_record") or {}, decision_rule)
    min_improvement = float(decision_rule.get("min_improvement") or 0.0)
    primary_delta = deterministic_score[0] - no_change_score[0]
    replacement_accepted = deterministic_winner.get("candidate_id") != no_change.get("candidate_id")
    gated_to_no_change = False
    if bool(decision_rule.get("fallback_to_no_change", True)) and primary_delta < min_improvement:
        deterministic_winner = no_change
        deterministic_score = no_change_score
        replacement_accepted = False
        gated_to_no_change = True
    return {
        "no_change": no_change,
        "no_change_score": no_change_score,
        "deterministic_winner": deterministic_winner,
        "deterministic_score": deterministic_score,
        "replacement_accepted": replacement_accepted,
        "gated_to_no_change": gated_to_no_change,
        "min_improvement": min_improvement,
    }


def _reconcile_branch_candidates(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    launcher: EPyMARLTrainLauncher,
    base_train_config: Dict[str, Any],
    checkpoint_root_dir: str,
    available_steps: List[int],
    branch_budget_steps: int,
    branch_use_cuda: Optional[bool],
    local_results_path: str,
    workflow_id: str,
) -> None:
    checkpoint_step = int(decision["checkpoint_step"])
    for candidate in decision.get("candidates") or []:
        branch_result_path = _candidate_branch_result_path(
            workflow_dir,
            int(decision["decision_index"]),
            str(decision["stage_label"]),
            str(candidate["candidate_id"]),
        )
        plan = _candidate_train_config_for_decision(
            base_train_config=base_train_config,
            checkpoint_root_dir=checkpoint_root_dir,
            available_steps=available_steps,
            checkpoint_step=checkpoint_step,
            workflow_id=workflow_id,
            decision=decision,
            candidate=candidate,
            branch_budget_steps=branch_budget_steps,
            branch_use_cuda=branch_use_cuda,
            local_results_path=local_results_path,
            launcher=launcher,
        )
        state = _classify_existing_candidate_state(
            launcher=launcher,
            branch_plan=plan,
            branch_result_path=branch_result_path,
        )
        run_reference = state.get("run_reference") or {}
        run_status_payload = {}
        run_json = run_reference.get("run_json")
        if run_json and Path(str(run_json)).exists():
            try:
                run_status_payload = json.loads(Path(str(run_json)).read_text(encoding="utf-8"))
            except Exception:
                run_status_payload = {}
        run_status = str(run_status_payload.get("status") or state.get("last_seen_sacred_status") or "")
        candidate["status"] = str(state.get("status") or "planned")
        if candidate["status"] == "planned" and run_reference.get("run_id") is not None:
            candidate["status"] = "running" if run_status != "COMPLETED" else "completed_invalid"
        candidate["branch_result_path"] = str(branch_result_path)
        candidate["run_id"] = run_reference.get("run_id")
        candidate["run_dir"] = run_reference.get("run_dir")
        candidate["checkpoint_root_dir"] = run_reference.get("checkpoint_root_dir")
        candidate["available_checkpoint_steps"] = list(run_reference.get("available_checkpoint_steps") or [])
        candidate["latest_checkpoint_step"] = run_reference.get("latest_checkpoint_step")
        candidate["last_seen_sacred_status"] = state.get("last_seen_sacred_status")
        candidate["last_seen_process_alive"] = state.get("last_seen_process_alive")
        target_step = _candidate_endpoint_step(decision, branch_budget_steps)
        terminal_or_checkpointed = run_status == "COMPLETED" or _checkpoint_target_reached(run_reference, target_step)
        if terminal_or_checkpointed:
            branch_result = _materialize_branch_result(
                launcher=launcher,
                branch_plan=plan,
                branch_result_path=branch_result_path,
                candidate=candidate,
                expected_endpoint_step=target_step,
            )
            candidate["status"] = str(branch_result.get("status") or "completed_invalid")
            candidate["candidate_valid"] = bool(branch_result.get("candidate_valid"))
            candidate["invalid_reason"] = branch_result.get("invalid_reason")
            candidate["checkpoint_exists"] = bool(branch_result.get("checkpoint_exists"))
            candidate["endpoint_checkpoint_path"] = branch_result.get("endpoint_checkpoint_path")
            candidate["latest_checkpoint_step"] = branch_result.get("actual_checkpoint_step")
            candidate["distance_from_expected"] = branch_result.get("distance_from_expected")
            candidate["metrics_missing"] = bool(branch_result.get("metrics_missing"))
            candidate["available_metric_keys"] = list(branch_result.get("available_metric_keys") or [])
            candidate["metrics_source"] = branch_result.get("metrics_source")
            candidate["run_reference"] = deepcopy(branch_result.get("run_reference") or {})
            candidate["branch_result"] = deepcopy(branch_result)
            _hydrate_stage3_candidate_from_branch_result(candidate, workflow_dir=workflow_dir)
            if candidate["candidate_valid"]:
                candidate["result_record"] = _build_result_record(
                    field_name="adaptive_combo",
                    stage_label=str(decision["stage_label"]),
                    checkpoint={
                        "name": decision["checkpoint_name"],
                        "step": decision["checkpoint_step"],
                    },
                    candidate_id=str(candidate["candidate_id"]),
                    candidate_value=None,
                    branch_result=branch_result,
                )
            continue
        candidate["candidate_valid"] = None
        _hydrate_stage3_candidate_from_branch_result(candidate, workflow_dir=workflow_dir)


def _dispatch_branch_candidates(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    launcher: EPyMARLTrainLauncher,
    base_train_config: Dict[str, Any],
    checkpoint_root_dir: str,
    available_steps: List[int],
    branch_budget_steps: int,
    branch_use_cuda: Optional[bool],
    local_results_path: str,
    workflow_id: str,
    max_parallel_candidates: int,
) -> None:
    running_count = sum(1 for candidate in decision.get("candidates") or [] if candidate.get("status") == "running")
    dispatch_slots = max(0, int(max_parallel_candidates) - int(running_count))
    if dispatch_slots <= 0:
        return
    checkpoint_step = int(decision["checkpoint_step"])
    launchable = [
        item
        for item in (decision.get("candidates") or [])
        if item.get("status") in {"pending", "planned"} and not item.get("run_id")
    ]
    for candidate in launchable[:dispatch_slots]:
        branch_result_path = _candidate_branch_result_path(
            workflow_dir,
            int(decision["decision_index"]),
            str(decision["stage_label"]),
            str(candidate["candidate_id"]),
        )
        plan = _candidate_train_config_for_decision(
            base_train_config=base_train_config,
            checkpoint_root_dir=checkpoint_root_dir,
            available_steps=available_steps,
            checkpoint_step=checkpoint_step,
            workflow_id=workflow_id,
            decision=decision,
            candidate=candidate,
            branch_budget_steps=branch_budget_steps,
            branch_use_cuda=branch_use_cuda,
            local_results_path=local_results_path,
            launcher=launcher,
        )
        launch_result = _launch_candidate_background_and_bind_run(
            launcher=launcher,
            branch_plan=plan,
            branch_result_path=branch_result_path,
        )
        run_reference = launch_result.get("run_reference") or {}
        candidate["status"] = "running"
        candidate["branch_result_path"] = str(branch_result_path)
        candidate["run_id"] = run_reference.get("run_id")
        candidate["run_dir"] = run_reference.get("run_dir")
        candidate["checkpoint_root_dir"] = run_reference.get("checkpoint_root_dir")
        candidate["available_checkpoint_steps"] = list(run_reference.get("available_checkpoint_steps") or [])
        candidate["latest_checkpoint_step"] = run_reference.get("latest_checkpoint_step")
        candidate["last_seen_sacred_status"] = "RUNNING" if run_reference.get("run_id") is not None else None
        candidate["last_seen_process_alive"] = True if run_reference.get("run_id") is not None else None
        candidate["launch_attempts"] = int(candidate.get("launch_attempts") or 0) + 1


def _all_branch_candidates_completed(decision: Dict[str, Any]) -> bool:
    return all(
        str(candidate.get("status")) in {"completed", "completed_invalid", "failed_invalid"}
        for candidate in (decision.get("candidates") or [])
    )


def _write_decision_summary(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    decision_rule: Dict[str, Any],
) -> Path:
    records = [
        {
            "candidate_id": candidate.get("candidate_id"),
            "beta": candidate.get("beta"),
            "wc": candidate.get("wc"),
            "wp": candidate.get("wp"),
            "is_no_change": candidate.get("is_no_change"),
            "status": candidate.get("status"),
            "run_id": candidate.get("run_id"),
            "score": list(_candidate_score(candidate.get("result_record") or {}, decision_rule)),
            "result_record": deepcopy(candidate.get("result_record") or {}),
        }
        for candidate in (decision.get("candidates") or [])
    ]
    payload = {
        "decision_id": decision.get("decision_id"),
        "decision_index": decision.get("decision_index"),
        "stage_label": decision.get("stage_label"),
        "checkpoint_name": decision.get("checkpoint_name"),
        "checkpoint_step": decision.get("checkpoint_step"),
        "llm_routing": deepcopy(
            llm_routing_artifact_fields(decision.get("llm_routing") or {})
        ),
        "stage3_branch_rounds": decision.get("stage3_branch_rounds"),
        "stage3_update_candidates_per_round": decision.get("stage3_update_candidates_per_round")
        or decision.get("stage3_candidates_per_round"),
        "stage3_gate_mode": decision.get("stage3_gate_mode"),
        "stage3_include_no_change_control": decision.get("stage3_include_no_change_control"),
        "stage3_no_change_counts_toward_round_budget": decision.get(
            "stage3_no_change_counts_toward_round_budget"
        ),
        "stage3_reuse_no_change_across_rounds": decision.get("stage3_reuse_no_change_across_rounds"),
        "stage3_min_effective_update_candidates_per_checkpoint": decision.get(
            "stage3_min_effective_update_candidates_per_checkpoint"
        ),
        "stage3_require_nontrivial_config_delta": decision.get("stage3_require_nontrivial_config_delta"),
        "stage3_max_fallback_style_candidates_per_checkpoint": decision.get(
            "stage3_max_fallback_style_candidates_per_checkpoint"
        ),
        "stage3_require_candidate_type_diversity": decision.get("stage3_require_candidate_type_diversity"),
        "policy_guided_stage3_require_behavior_evidence": decision.get(
            "policy_guided_stage3_require_behavior_evidence"
        ),
        "rounds": deepcopy(decision.get("rounds") or []),
        "current_config": deepcopy(decision.get("current_config") or {}),
        "llm_candidate_generation_status": decision.get("llm_candidate_generation_status"),
        "llm_candidate_generation_cache_path": decision.get("llm_candidate_generation_cache_path"),
        "llm_candidate_generation_validation_errors": deepcopy(
            decision.get("llm_candidate_generation_validation_errors") or []
        ),
        "llm_candidate_generation_underperforming_vs_fixed_reference": decision.get(
            "llm_candidate_generation_underperforming_vs_fixed_reference"
        ),
        "llm_candidate_generation_fixed_reference_config": deepcopy(
            decision.get("llm_candidate_generation_fixed_reference_config") or {}
        ),
        "llm_candidate_generation_requires_reference_or_recovery": bool(
            decision.get("llm_candidate_generation_requires_reference_or_recovery")
        ),
        "llm_candidate_generation_provided_reference_or_recovery": bool(
            decision.get("llm_candidate_generation_provided_reference_or_recovery")
        ),
        "llm_candidate_generation_repair_used": bool(
            decision.get("llm_candidate_generation_repair_used")
        ),
        "llm_candidate_generation_deterministic_recovery_added": bool(
            decision.get("llm_candidate_generation_deterministic_recovery_added")
        ),
        "llm_result_diagnosis_status": decision.get("llm_result_diagnosis_status"),
        "llm_result_diagnosis_cache_path": decision.get("llm_result_diagnosis_cache_path"),
        "llm_result_diagnosis_validation_errors": deepcopy(
            decision.get("llm_result_diagnosis_validation_errors") or []
        ),
        "llm_ranking": deepcopy(decision.get("llm_ranking") or []),
        "llm_recommended_winner": decision.get("llm_recommended_winner"),
        "deterministic_winner": decision.get("deterministic_winner"),
        "winner_candidate_id": decision.get("winner_candidate_id"),
        "final_selected_winner": decision.get("final_selected_winner"),
        "final_decision_source": decision.get("final_decision_source"),
        "decision_source": decision.get("decision_source"),
        "deterministic_gate_completed": decision.get("deterministic_gate_completed"),
        "no_change_gate_overrode_llm": decision.get("no_change_gate_overrode_llm"),
        "fallback_used": decision.get("fallback_used"),
        "fallback_reason": decision.get("fallback_reason"),
        "winner_config": deepcopy(decision.get("winner_config") or {}),
        "winner_score": decision.get("winner_score"),
        "no_change_score": decision.get("no_change_score"),
        "replacement_accepted": decision.get("replacement_accepted"),
        "selected_winner_candidate_id": decision.get("selected_winner_candidate_id"),
        "selected_winner_branch_run_id": decision.get("selected_winner_branch_run_id"),
        "selected_winner_endpoint_checkpoint_path": decision.get("selected_winner_endpoint_checkpoint_path"),
        "selected_winner_endpoint_checkpoint_step": decision.get("selected_winner_endpoint_checkpoint_step"),
        "promoted_to_mainline": decision.get("promoted_to_mainline"),
        "promotion_reason": decision.get("promotion_reason"),
        "promotion_failure_reason": decision.get("promotion_failure_reason"),
        "candidate_metrics": deepcopy(decision.get("candidate_metrics") or {}),
        "reference_context": deepcopy(decision.get("reference_context") or {}),
        "safety_gate": deepcopy(decision.get("safety_gate") or {}),
        "oscillation": deepcopy(decision.get("oscillation") or {}),
        "valid_branch_metric_comparison": decision.get("valid_branch_metric_comparison"),
        "decision_meaningful": decision.get("decision_meaningful"),
        "decision_status": decision.get("decision_status"),
        "candidate_results": records,
    }
    path = _decision_summary_path(
        workflow_dir,
        int(decision["decision_index"]),
        str(decision["stage_label"]),
    )
    _save_json(path, payload)
    return path


def _build_result_diagnosis_payload(
    *,
    decision: Dict[str, Any],
    decision_rule: Dict[str, Any],
) -> Dict[str, Any]:
    rows = []
    candidate_metrics_by_id = deepcopy(decision.get("candidate_metrics") or {})
    for candidate in (decision.get("candidates") or []):
        result_record = deepcopy(candidate.get("result_record") or {})
        metrics = deepcopy(candidate_metrics_by_id.get(str(candidate.get("candidate_id")) or "") or {})
        series_stats = _diagnostic_series_stats(candidate.get("run_dir") or result_record.get("run_dir"))
        rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "beta": candidate.get("beta"),
                "wc": candidate.get("wc"),
                "wp": candidate.get("wp"),
                "is_no_change": bool(candidate.get("is_no_change")),
                "best_test_sparse_return_mean": result_record.get("best_test_sparse_return_mean"),
                "last_test_sparse_return_mean": result_record.get("last_test_sparse_return_mean"),
                "last_k_mean": metrics.get("last_k_mean", series_stats.get("last_k_mean")),
                "auc": metrics.get("auc_mean", series_stats.get("auc")),
                "final": metrics.get("final"),
                "last_k_std": metrics.get("last_k_std"),
                "spike_gap": metrics.get("spike_gap"),
                "late_window_slope": series_stats.get("late_window_slope"),
                "test_sparse_return_std": result_record.get("test_sparse_return_std"),
                "best_test_return_mean": result_record.get("best_test_return_mean"),
                "last_test_return_mean": result_record.get("last_test_return_mean"),
                "grad_norm": result_record.get("last_grad_norm"),
                "loss": result_record.get("last_loss"),
                "td_error_abs": result_record.get("last_td_error_abs"),
                "run_status": candidate.get("status"),
                "run_id": candidate.get("run_id"),
                "metrics_valid": metrics.get("metrics_valid"),
                "stability_gate_score": metrics.get("score"),
                "risk_adjusted_score": metrics.get("risk_adjusted_score"),
                "soft_acceptance_margin": metrics.get("soft_acceptance_margin"),
                "soft_warning_reasons": deepcopy(metrics.get("soft_warning_reasons") or []),
                "hard_rejection_reasons": deepcopy(metrics.get("hard_rejection_reasons") or []),
                "accepted_by_soft_gate": bool(metrics.get("accepted_by_soft_gate")),
                "original_hard_gate_would_reject": bool(metrics.get("original_hard_gate_would_reject")),
                "rejection_reasons": deepcopy(metrics.get("rejection_reasons") or []),
                "score": list(_candidate_score(result_record, decision_rule)),
            }
        )
    return {
        "decision_index": decision.get("decision_index"),
        "stage_label": decision.get("stage_label"),
        "checkpoint_name": decision.get("checkpoint_name"),
        "checkpoint_step": decision.get("checkpoint_step"),
        "current_config": deepcopy(decision.get("current_config") or {}),
        "decision_rule": deepcopy(decision_rule),
        "safety_gate": deepcopy(decision.get("safety_gate") or {}),
        "reference_context": deepcopy(decision.get("reference_context") or {}),
        "initial_config": deepcopy(decision.get("initial_config") or {}),
        "winner_history": deepcopy(decision.get("winner_history") or []),
        "candidate_results": rows,
    }


def _deterministic_mock_result_diagnosis(*, decision: Dict[str, Any], decision_rule: Dict[str, Any]) -> Dict[str, Any]:
    candidates = list(decision.get("candidates") or [])
    winner = max(
        candidates,
        key=lambda item: _candidate_score(item.get("result_record") or {}, decision_rule),
    )
    no_change = next((candidate for candidate in candidates if bool(candidate.get("is_no_change"))), None)
    no_change_id = no_change.get("candidate_id") if no_change else None
    if winner.get("candidate_id") == no_change_id:
        recommended_candidate_id = no_change_id
        decision_label = "no_change"
    else:
        recommended_candidate_id = winner.get("candidate_id")
        decision_label = "update"
    ranking = []
    ranked = sorted(
        candidates,
        key=lambda item: _candidate_score(item.get("result_record") or {}, decision_rule),
        reverse=True,
    )
    for index, item in enumerate(ranked, start=1):
        ranking.append(
            {
                "candidate_id": item.get("candidate_id"),
                "rank": index,
                "evidence_strength": "moderate" if index == 1 else "weak",
                "main_evidence": "mock ranking by deterministic score",
                "main_risk": "mock diagnosis only",
            }
        )
    return {
        "ranking": ranking,
        "recommended_candidate_id": recommended_candidate_id,
        "decision": decision_label,
        "confidence": "low",
        "metric_basis": {
            "beats_no_change_on_best": recommended_candidate_id != no_change_id,
            "beats_no_change_on_last": recommended_candidate_id != no_change_id,
            "beats_no_change_on_last_k_mean": recommended_candidate_id != no_change_id,
            "beats_no_change_on_auc": recommended_candidate_id != no_change_id,
            "stability_concern": False,
        },
        "short_reason": "mock_llm_mode deterministic recommendation",
        "risk_notes": "Mock diagnosis for controller dry-run validation.",
    }


def _normalize_result_diagnosis_payload(
    *,
    parsed: Dict[str, Any],
    decision: Dict[str, Any],
    validation_errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    errors = validation_errors if validation_errors is not None else []
    candidates = list(decision.get("candidates") or [])
    candidate_ids = {str(candidate.get("candidate_id")) for candidate in candidates if candidate.get("candidate_id")}
    no_change = next((candidate for candidate in candidates if bool(candidate.get("is_no_change"))), None)
    no_change_id = str(no_change.get("candidate_id")) if no_change is not None else None

    ranking_payload = parsed.get("ranking")
    normalized_ranking: List[Dict[str, Any]] = []
    seen_ranked: set[str] = set()
    if isinstance(ranking_payload, list):
        for index, item in enumerate(ranking_payload, start=1):
            if not isinstance(item, dict):
                errors.append(f"invalid_ranking_entry:{item!r}")
                continue
            candidate_id = str(item.get("candidate_id") or "").strip()
            if not candidate_id:
                errors.append(f"missing_ranking_candidate_id:{index}")
                continue
            if candidate_id not in candidate_ids:
                errors.append(f"unknown_ranking_candidate_id:{candidate_id}")
                continue
            if candidate_id in seen_ranked:
                errors.append(f"duplicate_ranking_candidate_id:{candidate_id}")
                continue
            seen_ranked.add(candidate_id)
            rank_value = item.get("rank")
            try:
                rank_int = int(rank_value)
            except (TypeError, ValueError):
                rank_int = index
                errors.append(f"invalid_rank_value:{candidate_id}:{rank_value!r}")
            evidence_strength = str(item.get("evidence_strength") or "weak").strip().lower()
            if evidence_strength not in {"strong", "moderate", "weak"}:
                evidence_strength = "weak"
                errors.append(f"invalid_evidence_strength:{candidate_id}")
            normalized_ranking.append(
                {
                    "candidate_id": candidate_id,
                    "rank": rank_int,
                    "evidence_strength": evidence_strength,
                    "main_evidence": str(item.get("main_evidence") or ""),
                    "main_risk": str(item.get("main_risk") or ""),
                }
            )
    else:
        if ranking_payload is not None:
            errors.append("invalid_ranking_payload")

    recommended_candidate_id = parsed.get("recommended_candidate_id")
    if recommended_candidate_id is not None:
        recommended_candidate_id = str(recommended_candidate_id).strip() or None
    if recommended_candidate_id and recommended_candidate_id not in candidate_ids:
        errors.append(f"unknown_recommended_candidate_id:{recommended_candidate_id}")
        recommended_candidate_id = None

    raw_decision = str(parsed.get("decision") or "").strip().lower()
    if raw_decision not in {"update", "no_change"}:
        if raw_decision:
            errors.append(f"invalid_decision_value:{raw_decision}")
        raw_decision = "no_change" if recommended_candidate_id == no_change_id else "update"
    confidence = str(parsed.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
        errors.append("invalid_confidence_value")

    metric_basis_raw = parsed.get("metric_basis")
    metric_basis = metric_basis_raw if isinstance(metric_basis_raw, dict) else {}
    if metric_basis_raw is not None and not isinstance(metric_basis_raw, dict):
        errors.append("invalid_metric_basis_payload")

    return {
        "ranking": normalized_ranking,
        "recommended_candidate_id": recommended_candidate_id,
        "decision": raw_decision,
        "confidence": confidence,
        "metric_basis": {
            "beats_no_change_on_best": bool(metric_basis.get("beats_no_change_on_best", False)),
            "beats_no_change_on_last": bool(metric_basis.get("beats_no_change_on_last", False)),
            "beats_no_change_on_last_k_mean": bool(metric_basis.get("beats_no_change_on_last_k_mean", False)),
            "beats_no_change_on_auc": bool(metric_basis.get("beats_no_change_on_auc", False)),
            "stability_concern": bool(metric_basis.get("stability_concern", False)),
        },
        "short_reason": str(parsed.get("short_reason") or ""),
        "risk_notes": str(parsed.get("risk_notes") or ""),
    }


def _diagnose_results_with_llm_or_fallback(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    decision_rule: Dict[str, Any],
    adaptive_method: Dict[str, Any],
    use_real_llm: bool,
    api_key_env: str,
    base_url: str,
    model: str,
    temperature: float,
    llm_timeout: float,
    llm_retry_count: int,
    llm_retry_backoff: float,
) -> Optional[Dict[str, Any]]:
    llm_enabled = bool(adaptive_method.get("use_llm_stage3_result_diagnosis", True))
    use_cache = bool(adaptive_method.get("llm_stage3_use_cache", True))
    fallback_on_error = bool(adaptive_method.get("fallback_to_deterministic_on_llm_error", True))
    mock_mode = bool(adaptive_method.get("mock_llm_mode", False))
    decision["llm_result_diagnosis_status"] = "pending"
    cache_path = _result_diagnosis_cache_path(
        workflow_dir,
        int(decision["decision_index"]),
        str(decision["stage_label"]),
    )
    decision["llm_result_diagnosis_cache_path"] = str(cache_path)
    payload = _build_result_diagnosis_payload(decision=decision, decision_rule=decision_rule)
    payload["llm_model"] = str(model)
    payload["llm_routing"] = deepcopy(
        llm_routing_artifact_fields(adaptive_method.get("llm_routing") or {})
    )
    payload["prompt_version"] = _prompt_version(DEFAULT_STAGE3_DIAGNOSIS_PROMPT_PATH)
    payload_hash = _stable_payload_hash(payload)
    ledger_context = _stage3_ledger_context(
        workflow_dir=workflow_dir,
        decision=decision,
        adaptive_method=adaptive_method,
        model=model,
        base_url=base_url,
    )
    decision["llm_result_diagnosis_validation_errors"] = []

    if not llm_enabled:
        decision["llm_result_diagnosis_status"] = "disabled"
        return None

    if use_cache and cache_path.exists():
        try:
            cache_payload = _load_json(cache_path)
            if str(cache_payload.get("payload_hash") or "") == payload_hash:
                parsed = cache_payload.get("parsed") or {}
                if isinstance(parsed, dict):
                    validation_errors = list(cache_payload.get("validation_errors") or [])
                    normalized = _normalize_result_diagnosis_payload(
                        parsed=parsed,
                        decision=decision,
                        validation_errors=validation_errors,
                    )
                    decision["llm_result_diagnosis_status"] = str(cache_payload.get("status") or "cache_hit")
                    decision["llm_result_diagnosis_validation_errors"] = validation_errors
                    write_ledger_event(
                        **ledger_context,
                        call_type="cache_hit",
                        reason="stage3 result diagnosis cache hit",
                        cache_key=str(cache_path),
                        cache_hit=True,
                        payload=payload,
                        payload_hash=payload_hash,
                        request_id=extract_request_id(cache_payload),
                        success=True,
                        artifact_path=str(decision.get("llm_candidate_generation_artifact_path") or ""),
                        caller_file=__file__,
                        caller_function="_diagnose_results_with_llm_or_fallback",
                        event_state="cache_hit",
                        metadata={"decision_index": decision.get("decision_index"), "checkpoint_name": decision.get("checkpoint_name")},
                    )
                    return normalized
        except Exception:
            pass

    try:
        if mock_mode:
            parsed = _deterministic_mock_result_diagnosis(decision=decision, decision_rule=decision_rule)
            raw_text = json.dumps(parsed, ensure_ascii=False, indent=2)
            backend_metadata = {"mock_llm_mode": True}
            status = "mock_success"
        else:
            backend = _build_stage3_llm_backend(
                use_real_llm=use_real_llm,
                api_key_env=api_key_env,
                base_url=base_url,
                model=model,
                temperature=temperature,
                llm_timeout=llm_timeout,
                llm_retry_count=llm_retry_count,
                llm_retry_backoff=llm_retry_backoff,
            )
            if backend is None:
                raise ValueError("LLM result diagnosis requested but no backend is available.")
            prompt_spec = DEFAULT_STAGE3_DIAGNOSIS_PROMPT_PATH.read_text(encoding="utf-8")
            prompt = "\n".join(
                [
                    "Prompt specification:",
                    prompt_spec,
                    "",
                    "Adaptive Stage 3 result-diagnosis payload:",
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                    "",
                    "Return strict JSON only.",
                ]
            )
            attempt_event = write_ledger_event(
                **ledger_context,
                call_type="diagnosis",
                reason="stage3 result diagnosis live llm attempt",
                cache_key=str(cache_path),
                cache_hit=False,
                payload=payload,
                payload_hash=payload_hash,
                success=None,
                artifact_path=str(decision.get("llm_candidate_generation_artifact_path") or ""),
                caller_file=__file__,
                caller_function="_diagnose_results_with_llm_or_fallback",
                event_state="attempt",
                metadata={"decision_index": decision.get("decision_index"), "checkpoint_name": decision.get("checkpoint_name")},
            )
            llm_result = backend.generate_text(
                system_prompt=(
                    "You diagnose adaptive checkpoint branch-validation results for MARL reward-parameter updates. "
                    "Return strictly valid JSON."
                ),
                user_prompt=prompt,
                metadata={
                    "role": "adaptive_stage3_result_diagnosis",
                    "decision_index": decision.get("decision_index"),
                    "stage_label": decision.get("stage_label"),
                },
            )
            parsed = _extract_json_object(llm_result["text"])
            raw_text = llm_result["text"]
            backend_metadata = llm_result.get("backend_metadata") or {}
            status = "success"
        validation_errors: List[str] = []
        normalized = _normalize_result_diagnosis_payload(
            parsed=parsed,
            decision=decision,
            validation_errors=validation_errors,
        )
        if use_cache:
            _save_json(
                cache_path,
                {
                    "payload_hash": payload_hash,
                    "status": status,
                    "prompt_version": _prompt_version(DEFAULT_STAGE3_DIAGNOSIS_PROMPT_PATH),
                    "prompt_path": str(DEFAULT_STAGE3_DIAGNOSIS_PROMPT_PATH),
                    "input_payload": payload,
                    "parsed": parsed,
                    "raw_text": raw_text,
                    "backend_metadata": backend_metadata,
                    "validation_errors": validation_errors,
                    "normalized": normalized,
                },
            )
        decision["llm_result_diagnosis_status"] = status
        decision["llm_result_diagnosis_validation_errors"] = validation_errors
        write_ledger_event(
            **ledger_context,
            call_type="diagnosis",
            reason="stage3 result diagnosis completed",
            cache_key=str(cache_path),
            cache_hit=False,
            payload=payload,
            payload_hash=payload_hash,
            usage=None if mock_mode else llm_result.get("usage"),
            request_id=None if mock_mode else extract_request_id(llm_result),
            success=True,
            artifact_path=str(decision.get("llm_candidate_generation_artifact_path") or ""),
            caller_file=__file__,
            caller_function="_diagnose_results_with_llm_or_fallback",
            event_state="success",
            metadata={"decision_index": decision.get("decision_index"), "checkpoint_name": decision.get("checkpoint_name"), "backend_metadata": backend_metadata},
            record_id=None if mock_mode else str(attempt_event.get("record_id") or ""),
        )
        return normalized
    except Exception as exc:
        decision["llm_result_diagnosis_status"] = f"error:{exc.__class__.__name__}"
        decision["llm_result_diagnosis_validation_errors"] = [str(exc)]
        write_ledger_event(
            **ledger_context,
            call_type="fallback",
            reason="stage3 result diagnosis fallback",
            cache_key=str(cache_path),
            cache_hit=False,
            payload=payload,
            payload_hash=payload_hash,
            success=False,
            fallback_used=True,
            fallback_reason=str(exc),
            artifact_path=str(decision.get("llm_candidate_generation_artifact_path") or ""),
            caller_file=__file__,
            caller_function="_diagnose_results_with_llm_or_fallback",
            event_state="fallback",
            metadata={"decision_index": decision.get("decision_index"), "checkpoint_name": decision.get("checkpoint_name")},
            error_type=exc.__class__.__name__,
        )
        if not fallback_on_error:
            raise
        return None


def _select_winner_candidate(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    decision_rule: Dict[str, Any],
    adaptive_method: Dict[str, Any],
    initial_config: Dict[str, Any],
    previous_decisions: List[Dict[str, Any]],
    fixed_reference_run_dir: Optional[str],
    branch_budget_steps: int,
    use_real_llm: bool,
    api_key_env: str,
    base_url: str,
    model: str,
    temperature: float,
    llm_timeout: float,
    llm_retry_count: int,
    llm_retry_backoff: float,
) -> None:
    deterministic = _evaluate_stability_gate(
        workflow_dir=workflow_dir,
        decision=decision,
        adaptive_method=adaptive_method,
        initial_config=initial_config,
        previous_decisions=previous_decisions,
        fixed_reference_run_dir=fixed_reference_run_dir,
        branch_budget_steps=branch_budget_steps,
    )
    no_change = deterministic["no_change"]
    no_change_metrics = deepcopy(deterministic.get("no_change_metrics") or {})
    no_change_score = no_change_metrics.get("score")
    if not isinstance(no_change, dict):
        raise ValueError("adaptive replacement decision requires one no-change baseline candidate")
    deterministic_winner = deterministic.get("deterministic_winner")
    if not isinstance(deterministic_winner, dict):
        deterministic_winner = no_change
    replacement_accepted = deterministic_winner.get("candidate_id") != no_change.get("candidate_id")
    decision["candidate_metrics"] = deepcopy(deterministic.get("candidate_metrics") or {})
    decision["reference_context"] = deepcopy(deterministic.get("reference_context") or {})
    decision["safety_gate"] = deepcopy(deterministic.get("safety_gate") or {})
    decision["valid_branch_metric_comparison"] = bool(
        (decision.get("safety_gate") or {}).get("valid_branch_metric_comparison")
    )
    decision["decision_meaningful"] = bool(
        (decision.get("safety_gate") or {}).get("decision_meaningful")
    )
    decision["stage3_gate_mode"] = str((decision.get("safety_gate") or {}).get("gate_mode") or "hard_conservative")
    decision["decision_status"] = (
        "completed" if decision["decision_meaningful"] else "failed_metrics_reconcile"
    )

    llm_payload = _diagnose_results_with_llm_or_fallback(
        workflow_dir=workflow_dir,
        decision=decision,
        decision_rule=decision_rule,
        adaptive_method=adaptive_method,
        use_real_llm=use_real_llm,
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        temperature=temperature,
        llm_timeout=llm_timeout,
        llm_retry_count=llm_retry_count,
        llm_retry_backoff=llm_retry_backoff,
    )
    llm_candidate_id = None
    if isinstance(llm_payload, dict):
        llm_candidate_id = llm_payload.get("recommended_candidate_id")
        decision["llm_ranking"] = deepcopy(llm_payload.get("ranking") or [])
    decision["llm_recommended_winner"] = llm_candidate_id
    final_candidate = deterministic_winner
    final_source = str(
        (decision.get("safety_gate") or {}).get("final_decision_source") or "deterministic_stability_gate"
    )

    winner = final_candidate
    candidate_metrics = deepcopy(decision.get("candidate_metrics") or {})
    tentative_winner = deterministic.get("tentative_winner")
    decision["tentative_winner_candidate_id"] = (
        tentative_winner.get("candidate_id") if isinstance(tentative_winner, dict) else None
    )
    winner_score = (candidate_metrics.get(str((winner or {}).get("candidate_id")) or "") or {}).get("score")
    replacement_accepted = winner.get("candidate_id") != no_change.get("candidate_id")
    decision["winner_candidate_id"] = winner.get("candidate_id")
    decision["final_selected_winner"] = winner.get("candidate_id")
    decision["final_decision_source"] = final_source
    decision["decision_source"] = final_source
    decision["deterministic_winner"] = deterministic_winner.get("candidate_id")
    decision["deterministic_gate_completed"] = True
    decision["no_change_gate_overrode_llm"] = False
    decision["fallback_used"] = False
    decision["fallback_reason"] = None
    if str(decision.get("pbrs_version") or "") == "lbf_pbrs_v2":
        decision["winner_config"] = _normalize_lbf_stage_config(winner)
    else:
        decision["winner_config"] = {
            "beta": winner.get("beta"),
            "wc": winner.get("wc"),
            "wp": winner.get("wp"),
        }
    if bool(winner.get("is_no_change")) and str(decision.get("pbrs_version") or "") == "lbf_pbrs_v2":
        current_config = deepcopy(decision.get("current_config") or {})
        if not _lbf_stage_configs_match(decision["winner_config"], current_config):
            raise ValueError(
                "LBF Stage3 no_change winner drifted from decision.current_config: "
                f"winner_config={_canonicalize_lbf_stage_config_for_compare(decision['winner_config'])} "
                f"current_config={_canonicalize_lbf_stage_config_for_compare(current_config)}"
            )
    decision["winner_score"] = winner_score
    decision["no_change_score"] = no_change_score
    decision["replacement_accepted"] = bool(replacement_accepted)
    decision["selected_winner_candidate_id"] = winner.get("candidate_id")
    decision["selected_winner_branch_run_id"] = winner.get("run_id")
    decision["selected_winner_endpoint_checkpoint_path"] = (
        str(winner.get("checkpoint_root_dir") or "") + f"/{winner.get('latest_checkpoint_step')}"
        if winner.get("checkpoint_root_dir") and winner.get("latest_checkpoint_step")
        else None
    )
    decision["selected_winner_endpoint_checkpoint_step"] = winner.get("latest_checkpoint_step")
    decision["promoted_to_mainline"] = bool(
        decision.get("use_winner_branch_promotion", True) and winner.get("latest_checkpoint_step")
    )
    decision["promotion_reason"] = (
        "winner_branch_endpoint_selected_for_next_mainline_source"
        if decision["promoted_to_mainline"]
        else "winner_branch_promotion_disabled_or_missing_endpoint"
    )
    oscillation = {
        "revert_to_stage1b_config": False,
        "previous_update_may_be_unnecessary": False,
        "adaptive_oscillation_detected": False,
    }
    if bool(_stage3_gate_config(adaptive_method).get("oscillation_detection")) and _configs_match(
        decision["winner_config"],
        initial_config,
    ) and _previous_non_stage1b_update_exists(previous_decisions=previous_decisions, stage1b_config=initial_config):
        oscillation = {
            "revert_to_stage1b_config": True,
            "previous_update_may_be_unnecessary": True,
            "adaptive_oscillation_detected": True,
        }
    decision["oscillation"] = oscillation
    decision["promotion_failure_reason"] = None if decision["promoted_to_mainline"] else decision["promotion_reason"]


def _first_change_baseline_plan_for_decision(
    *,
    base_train_config: Dict[str, Any],
    checkpoint_root_dir: str,
    available_steps: List[int],
    checkpoint_step: int,
    workflow_id: str,
    decision: Dict[str, Any],
    initial_config: Dict[str, float],
    target_step: int,
    mainline_use_cuda: Optional[bool],
    local_results_path: str,
    save_model_interval: int,
    launcher: EPyMARLTrainLauncher,
) -> Dict[str, Any]:
    override_env_args = _build_native_pbrs_env_args(initial_config)
    interval = _recommended_short_resume_interval(int(target_step) - int(checkpoint_step))
    plan = launcher.build_checkpoint_resume_plan(
        base_train_config=base_train_config,
        run_reference={
            "checkpoint_root_dir": checkpoint_root_dir,
            "available_checkpoint_steps": available_steps,
        },
        requested_step=checkpoint_step,
        label_suffix=f"_{workflow_id}_{decision['stage_label']}_first_change_fixed_initial_baseline",
        use_dense_reward=False,
        reward_module_path="",
        alpha_policy_config={"type": "constant", "value": 1.0},
        workflow_id=f"{workflow_id}_fixed_initial_baseline",
        workflow_round=int(decision["decision_index"]),
        reward_paradigm="pbrs",
        active_pbrs_field="adaptive_first_change_fixed_initial_baseline",
        active_pbrs_field_source="adaptive_checkpoint_replacement",
        candidate_value=None,
        active_checkpoint_context={
            "name": decision["checkpoint_name"],
            "step": checkpoint_step,
            "stage_label": decision["stage_label"],
        },
        active_field_carryover_context={"baseline_config": deepcopy(initial_config)},
        candidate_selection_context={
            "candidate_type": "first_change_fixed_initial_baseline",
            "source_decision_id": decision.get("decision_index"),
            "source_checkpoint_step": checkpoint_step,
            "initial_dense_config": deepcopy(initial_config),
        },
        phase_name="adaptive_first_change_fixed_initial_baseline",
        save_model=True,
        save_model_interval=max(1, int(target_step) - int(checkpoint_step)),
        save_final_model=True,
        local_results_path=local_results_path,
        override_env_args=override_env_args,
        override_overrides={
            "t_max": int(target_step),
            "test_interval": interval,
            "log_interval": interval,
            "runner_log_interval": interval,
            "learner_log_interval": interval,
            **({"use_cuda": bool(mainline_use_cuda)} if mainline_use_cuda is not None else {}),
        },
    )
    _inject_candidate_metadata_into_plan(plan=plan, candidate=initial_config)
    plan["resume_train_config"]["defer_initial_model_save"] = True
    return plan


def _reconcile_first_change_baseline(
    *,
    workflow_dir: Path,
    manifest: Dict[str, Any],
    decision: Dict[str, Any],
    launcher: EPyMARLTrainLauncher,
    base_train_config: Dict[str, Any],
    checkpoint_root_dir: str,
    available_steps: List[int],
    mainline_use_cuda: Optional[bool],
    local_results_path: str,
    save_model_interval: int,
    workflow_id: str,
) -> None:
    baseline = manifest.get("first_change_baseline") or {}
    if not baseline.get("source_decision_id"):
        return
    result_path = (
        _decision_dir(workflow_dir, int(decision["decision_index"]), str(decision["stage_label"]))
        / "first_change_fixed_initial_baseline_result.json"
    )
    baseline["result_json"] = str(result_path)
    plan = _first_change_baseline_plan_for_decision(
        base_train_config=base_train_config,
        checkpoint_root_dir=checkpoint_root_dir,
        available_steps=available_steps,
        checkpoint_step=int(decision["checkpoint_step"]),
        workflow_id=workflow_id,
        decision=decision,
        initial_config=deepcopy(manifest.get("initial_config") or {}),
        target_step=int((decision.get("continuation") or {}).get("target_step") or 0),
        mainline_use_cuda=mainline_use_cuda,
        local_results_path=local_results_path,
        save_model_interval=save_model_interval,
        launcher=launcher,
    )
    state = _classify_existing_candidate_state(
        launcher=launcher,
        branch_plan=plan,
        branch_result_path=result_path,
    )
    baseline["status"] = str(state.get("status") or "planned")
    if baseline["status"] == "planned" and ((state.get("run_reference") or {}).get("run_id") is not None):
        baseline["status"] = "running"
    baseline["run_id"] = ((state.get("run_reference") or {}).get("run_id"))
    baseline["run_dir"] = ((state.get("run_reference") or {}).get("run_dir"))
    baseline["last_seen_sacred_status"] = state.get("last_seen_sacred_status")
    baseline["last_seen_process_alive"] = state.get("last_seen_process_alive")
    if _checkpoint_target_reached(
        state.get("run_reference") or {},
        int((decision.get("continuation") or {}).get("target_step") or 0),
    ):
        baseline["status"] = "completed"
    manifest["first_change_baseline_run_id"] = baseline.get("run_id")
    manifest["first_change_baseline"] = baseline


def _dispatch_first_change_baseline(
    *,
    workflow_dir: Path,
    manifest: Dict[str, Any],
    decision: Dict[str, Any],
    launcher: EPyMARLTrainLauncher,
    base_train_config: Dict[str, Any],
    checkpoint_root_dir: str,
    available_steps: List[int],
    mainline_use_cuda: Optional[bool],
    local_results_path: str,
    save_model_interval: int,
    workflow_id: str,
) -> None:
    baseline = manifest.get("first_change_baseline") or {}
    if baseline.get("status") not in {"pending", "planned"}:
        return
    result_path = (
        _decision_dir(workflow_dir, int(decision["decision_index"]), str(decision["stage_label"]))
        / "first_change_fixed_initial_baseline_result.json"
    )
    plan = _first_change_baseline_plan_for_decision(
        base_train_config=base_train_config,
        checkpoint_root_dir=checkpoint_root_dir,
        available_steps=available_steps,
        checkpoint_step=int(decision["checkpoint_step"]),
        workflow_id=workflow_id,
        decision=decision,
        initial_config=deepcopy(manifest.get("initial_config") or {}),
        target_step=int((decision.get("continuation") or {}).get("target_step") or 0),
        mainline_use_cuda=mainline_use_cuda,
        local_results_path=local_results_path,
        save_model_interval=save_model_interval,
        launcher=launcher,
    )
    launch_result = _launch_candidate_background_and_bind_run(
        launcher=launcher,
        branch_plan=plan,
        branch_result_path=result_path,
    )
    run_reference = launch_result.get("run_reference") or {}
    baseline["status"] = "running"
    baseline["run_id"] = run_reference.get("run_id")
    baseline["run_dir"] = run_reference.get("run_dir")
    baseline["result_json"] = str(result_path)
    baseline["last_seen_sacred_status"] = "RUNNING" if run_reference.get("run_id") is not None else None
    baseline["last_seen_process_alive"] = True if run_reference.get("run_id") is not None else None
    baseline["launch_attempts"] = int(baseline.get("launch_attempts") or 0) + 1
    manifest["first_change_baseline_run_id"] = baseline.get("run_id")
    manifest["first_change_baseline"] = baseline


def _reconcile_continuation(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    launcher: EPyMARLTrainLauncher,
    base_train_config: Dict[str, Any],
    checkpoint_root_dir: str,
    available_steps: List[int],
    mainline_use_cuda: Optional[bool],
    local_results_path: str,
    save_model_interval: int,
    workflow_id: str,
) -> None:
    continuation = decision["continuation"]
    if decision.get("winner_config") is None:
        return
    result_path = _continuation_result_path(
        workflow_dir,
        int(decision["decision_index"]),
        str(decision["stage_label"]),
    )
    continuation["result_json"] = str(result_path)
    source = _continuation_source_for_decision(
        decision=decision,
        checkpoint_root_dir=checkpoint_root_dir,
        available_steps=available_steps,
        checkpoint_step=int(decision["checkpoint_step"]),
    )
    continuation["source_run_dir"] = source.get("source_run_dir")
    continuation["source_checkpoint_step"] = source.get("checkpoint_step")
    if _continuation_target_already_satisfied(
        int(source.get("checkpoint_step") or 0),
        int(continuation["target_step"]),
    ):
        _mark_continuation_completed_from_source(
            continuation=continuation,
            source=source,
            result_path=result_path,
            decision=decision,
            reason="source_checkpoint_already_reaches_or_exceeds_target",
        )
        return
    plan = _continuation_plan_for_decision(
        base_train_config=base_train_config,
        checkpoint_root_dir=str(source["checkpoint_root_dir"]),
        available_steps=list(source["available_checkpoint_steps"]),
        checkpoint_step=int(source["checkpoint_step"]),
        workflow_id=workflow_id,
        decision=decision,
        winner_config=decision["winner_config"],
        target_step=int(continuation["target_step"]),
        mainline_use_cuda=mainline_use_cuda,
        local_results_path=local_results_path,
        save_model_interval=max(1, int(continuation["target_step"]) - int(source["checkpoint_step"])),
        launcher=launcher,
    )
    state = _classify_existing_candidate_state(
        launcher=launcher,
        branch_plan=plan,
        branch_result_path=result_path,
    )
    continuation["status"] = str(state.get("status") or "planned")
    if continuation["status"] == "planned" and ((state.get("run_reference") or {}).get("run_id") is not None):
        continuation["status"] = "running"
    continuation["run_id"] = ((state.get("run_reference") or {}).get("run_id"))
    continuation["run_dir"] = ((state.get("run_reference") or {}).get("run_dir"))
    continuation["last_seen_sacred_status"] = state.get("last_seen_sacred_status")
    continuation["last_seen_process_alive"] = state.get("last_seen_process_alive")
    if _checkpoint_target_reached(
        state.get("run_reference") or {},
        int(continuation["target_step"]),
    ):
        continuation["status"] = "completed"
        continuation["checkpoint_root_dir"] = (state.get("run_reference") or {}).get("checkpoint_root_dir")
        continuation["available_checkpoint_steps"] = list(
            ((state.get("run_reference") or {}).get("available_checkpoint_steps") or [])
        )
        continuation["latest_checkpoint_step"] = (
            (state.get("run_reference") or {}).get("latest_checkpoint_step")
        )
        continuation["latest_checkpoint_path"] = (
            (state.get("run_reference") or {}).get("latest_model_path")
        )
        continuation["effective_checkpoint_step"] = int(
            ((state.get("run_reference") or {}).get("latest_checkpoint_step") or 0)
        )
        continuation["effective_checkpoint_root_dir"] = (
            (state.get("run_reference") or {}).get("checkpoint_root_dir")
        )
        if continuation.get("effective_checkpoint_root_dir") and continuation.get("effective_checkpoint_step"):
            continuation["effective_checkpoint_path"] = str(
                Path(str(continuation["effective_checkpoint_root_dir"])) / str(int(continuation["effective_checkpoint_step"]))
            )
    elif (
        str(continuation.get("status") or "") in {"running", "planned"}
        and not bool(state.get("last_seen_process_alive"))
        and str(state.get("last_seen_sacred_status") or "").upper() in {"COMPLETED"}
    ):
        run_reference = state.get("run_reference") or {}
        if _continuation_target_already_satisfied(
            int(source.get("checkpoint_step") or 0),
            int(continuation["target_step"]),
        ):
            _mark_continuation_completed_from_source(
                continuation=continuation,
                source=source,
                result_path=result_path,
                decision=decision,
                reason="source_checkpoint_already_reaches_or_exceeds_target_after_terminal_reconcile",
                existing_run_reference={
                    "run_id": run_reference.get("run_id"),
                    "run_dir": run_reference.get("run_dir"),
                    "last_seen_sacred_status": state.get("last_seen_sacred_status"),
                    "last_seen_process_alive": state.get("last_seen_process_alive"),
                },
            )
    elif (
        str(continuation.get("status") or "") in {"running", "planned"}
        and not bool(state.get("last_seen_process_alive"))
        and not _checkpoint_target_reached(
            state.get("run_reference") or {},
            int(continuation["target_step"]),
        )
    ):
        continuation["status"] = "pending"
        continuation["run_id"] = None
        continuation["run_dir"] = None
        continuation["last_seen_sacred_status"] = state.get("last_seen_sacred_status")
        continuation["last_seen_process_alive"] = False


def _dispatch_continuation(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    launcher: EPyMARLTrainLauncher,
    base_train_config: Dict[str, Any],
    checkpoint_root_dir: str,
    available_steps: List[int],
    mainline_use_cuda: Optional[bool],
    local_results_path: str,
    save_model_interval: int,
    workflow_id: str,
) -> None:
    continuation = decision["continuation"]
    if continuation.get("status") not in {"pending", "planned"}:
        return
    result_path = _continuation_result_path(
        workflow_dir,
        int(decision["decision_index"]),
        str(decision["stage_label"]),
    )
    source = _continuation_source_for_decision(
        decision=decision,
        checkpoint_root_dir=checkpoint_root_dir,
        available_steps=available_steps,
        checkpoint_step=int(decision["checkpoint_step"]),
    )
    continuation["source_run_dir"] = source.get("source_run_dir")
    continuation["source_checkpoint_step"] = source.get("checkpoint_step")
    if _continuation_target_already_satisfied(
        int(source.get("checkpoint_step") or 0),
        int(continuation["target_step"]),
    ):
        _mark_continuation_completed_from_source(
            continuation=continuation,
            source=source,
            result_path=result_path,
            decision=decision,
            reason="source_checkpoint_already_reaches_or_exceeds_target",
        )
        return
    plan = _continuation_plan_for_decision(
        base_train_config=base_train_config,
        checkpoint_root_dir=str(source["checkpoint_root_dir"]),
        available_steps=list(source["available_checkpoint_steps"]),
        checkpoint_step=int(source["checkpoint_step"]),
        workflow_id=workflow_id,
        decision=decision,
        winner_config=decision["winner_config"],
        target_step=int(continuation["target_step"]),
        mainline_use_cuda=mainline_use_cuda,
        local_results_path=local_results_path,
        save_model_interval=max(1, int(continuation["target_step"]) - int(source["checkpoint_step"])),
        launcher=launcher,
    )
    launch_result = _launch_candidate_background_and_bind_run(
        launcher=launcher,
        branch_plan=plan,
        branch_result_path=result_path,
    )
    run_reference = launch_result.get("run_reference") or {}
    continuation["status"] = "running"
    continuation["run_id"] = run_reference.get("run_id")
    continuation["run_dir"] = run_reference.get("run_dir")
    continuation["last_seen_sacred_status"] = "RUNNING" if run_reference.get("run_id") is not None else None
    continuation["last_seen_process_alive"] = True if run_reference.get("run_id") is not None else None
    continuation["launch_attempts"] = int(continuation.get("launch_attempts") or 0) + 1


def _build_run_reference_from_run_dir(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    readiness = inspect_baseline_checkpoint_readiness(baseline_run_dir=str(run_path), baseline_result_json=None)
    return {
        "run_dir": str(run_path),
        "checkpoint_root_dir": (readiness.get("checkpoint_summary") or {}).get("checkpoint_root_dir"),
        "available_checkpoint_steps": (readiness.get("checkpoint_summary") or {}).get("available_checkpoint_steps") or [],
        "baseline_readiness": readiness,
    }


def _normalize_stage3_manifest_summary(
    manifest: Dict[str, Any],
    decisions: List[Dict[str, Any]],
) -> None:
    completed = [str(d.get("decision_id") or "") for d in decisions if bool(d.get("completed"))]
    pending = [str(d.get("decision_id") or "") for d in decisions if not bool(d.get("completed"))]
    latest_promoted_checkpoint = None
    final_mainline_source_checkpoint = None
    for decision in decisions:
        if decision.get("selected_winner_endpoint_checkpoint_path"):
            latest_promoted_checkpoint = decision.get("selected_winner_endpoint_checkpoint_path")
        continuation = decision.get("continuation") or {}
        if continuation.get("effective_checkpoint_path"):
            final_mainline_source_checkpoint = continuation.get("effective_checkpoint_path")
        elif continuation.get("run_dir"):
            eff_root = continuation.get("effective_checkpoint_root_dir")
            eff_step = continuation.get("effective_checkpoint_step")
            if eff_root and eff_step:
                final_mainline_source_checkpoint = str(Path(str(eff_root)) / str(int(eff_step)))
    current_index = None
    for idx, decision in enumerate(decisions, start=1):
        if not bool(decision.get("completed")):
            current_index = idx
            break
    if current_index is None and decisions:
        current_index = len(decisions)
    all_terminal = bool(decisions) and all(
        bool(d.get("completed"))
        and str((d.get("continuation") or {}).get("status") or "") == "completed"
        for d in decisions
    )
    manifest["stage3_decisions_total"] = len(decisions)
    manifest["stage3_decisions_completed"] = len(completed)
    manifest["stage3_update_candidates_per_round"] = max(
        [
            int(d.get("stage3_update_candidates_per_round") or d.get("stage3_candidates_per_round") or 0)
            for d in decisions
        ]
        or [0]
    )
    manifest["stage3_include_no_change_control"] = any(
        bool(d.get("stage3_include_no_change_control", True)) for d in decisions
    )
    manifest["stage3_no_change_counts_toward_round_budget"] = any(
        bool(d.get("stage3_no_change_counts_toward_round_budget", False)) for d in decisions
    )
    manifest["current_decision_index"] = current_index
    manifest["completed_decision_ids"] = completed
    manifest["pending_decision_ids"] = pending
    manifest["latest_promoted_checkpoint"] = latest_promoted_checkpoint
    manifest["final_mainline_source_checkpoint"] = final_mainline_source_checkpoint or latest_promoted_checkpoint
    manifest["all_decisions_terminal"] = all_terminal
    manifest["ready_for_qmix_single_seed_pilot"] = bool(all_terminal)
    workflow_summary = manifest.setdefault("workflow_summary", {})
    if isinstance(workflow_summary, dict):
        workflow_summary["real_llm_called"] = bool(
            workflow_summary.get("real_llm_called", False)
        )


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
    except Exception:
        return []
    return records


def _adaptive_real_llm_called(workflow_dir: Path) -> bool:
    workflow_id = workflow_dir.name
    for ledger_path in (
        workflow_dir / "llm_api_call_ledger.jsonl",
        workflow_dir.parent.parent / "llm_api_call_ledger.jsonl",
    ):
        for record in _load_jsonl_records(ledger_path):
            if str(record.get("workflow_id") or "") != workflow_id:
                continue
            if bool(record.get("cache_hit", False)):
                continue
            if str(record.get("call_type") or "") in {"cache_hit", "fallback"}:
                continue
            if str(record.get("provider") or "").lower() != "openai":
                continue
            return True
    return False


def _build_adaptive_timeline_events(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    workflow_id = str(manifest.get("workflow_id") or "")
    events: List[Dict[str, Any]] = [
        {
            "event_type": "workflow_initialized",
            "payload": {"workflow_id": workflow_id},
        }
    ]
    for decision in list(manifest.get("decisions") or []):
        decision_payload = {
            "decision_index": int(decision.get("decision_index") or 0),
            "checkpoint_name": str(decision.get("checkpoint_name") or ""),
            "stage_label": str(decision.get("stage_label") or ""),
            "completed": bool(decision.get("completed")),
        }
        if str(decision.get("critic_pre_diagnosis_status") or "") == "completed":
            events.append(
                {
                    "event_type": "stage3_pre_checkpoint_reconciled",
                    "payload": deepcopy(decision_payload),
                }
            )
        for round_payload in list(decision.get("rounds") or []):
            round_id = int(round_payload.get("round_id") or 0)
            if str(round_payload.get("candidate_generation_status") or "pending") != "pending":
                events.append(
                    {
                        "event_type": "stage3_round_candidate_generation_reconciled",
                        "payload": {
                            **deepcopy(decision_payload),
                            "round_id": round_id,
                            "candidate_generation_status": str(
                                round_payload.get("candidate_generation_status") or ""
                            ),
                            "generator_candidate_count": len(
                                list(round_payload.get("generator_candidates") or [])
                            ),
                        },
                    }
                )
            if list(round_payload.get("candidates") or []):
                events.append(
                    {
                        "event_type": "stage3_round_candidates_reviewed",
                        "payload": {
                            **deepcopy(decision_payload),
                            "round_id": round_id,
                            "candidate_count": len(list(round_payload.get("candidates") or [])),
                            "round_status": str(round_payload.get("status") or "pending"),
                        },
                    }
                )
            for candidate in list(round_payload.get("candidates") or []):
                if str(candidate.get("status") or "pending") == "pending":
                    continue
                events.append(
                    {
                        "event_type": "stage3_branch_candidate_reconciled",
                        "payload": {
                            **deepcopy(decision_payload),
                            "round_id": round_id,
                            "candidate_id": str(candidate.get("candidate_id") or ""),
                            "status": str(candidate.get("status") or ""),
                            "run_id": candidate.get("run_id"),
                            "candidate_valid": candidate.get("candidate_valid"),
                        },
                    }
                )
            if round_payload.get("analysis"):
                events.append(
                    {
                        "event_type": "stage3_round_analysis_reconciled",
                        "payload": {
                            **deepcopy(decision_payload),
                            "round_id": round_id,
                            "best_candidate_id": (
                                (round_payload.get("analysis") or {}).get("best_candidate_id")
                            ),
                        },
                    }
                )
        continuation = decision.get("continuation") or {}
        if str(continuation.get("status") or "pending") != "pending":
            events.append(
                {
                    "event_type": "stage3_continuation_reconciled",
                    "payload": {
                        **deepcopy(decision_payload),
                        "status": str(continuation.get("status") or ""),
                        "run_id": continuation.get("run_id"),
                        "effective_checkpoint_step": continuation.get("effective_checkpoint_step"),
                    },
                }
            )
    events.append(
        {
            "event_type": "workflow_reconciled_snapshot",
            "payload": {
                "workflow_id": workflow_id,
                "current_decision_index": manifest.get("current_decision_index"),
                "completed_decision_ids": list(manifest.get("completed_decision_ids") or []),
                "all_decisions_terminal": bool(manifest.get("all_decisions_terminal", False)),
                "real_llm_called": bool(
                    ((manifest.get("workflow_summary") or {}).get("real_llm_called", False))
                ),
            },
        }
    )
    return events


def _rewrite_adaptive_timeline(workflow_dir: Path, manifest: Dict[str, Any]) -> None:
    timeline_path = workflow_dir / "timeline.jsonl"
    events = _build_adaptive_timeline_events(manifest)
    lines = [
        json.dumps(
            {
                "event_type": str(event.get("event_type") or ""),
                "payload": deepcopy(event.get("payload") or {}),
                "timestamp": utc_now_iso(),
            },
            ensure_ascii=False,
        )
        for event in events
    ]
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _recover_stage3_round_from_saved_artifact(
    *,
    workflow_dir: Path,
    decision: Dict[str, Any],
    round_id: int,
    round_anchor: Dict[str, Any],
    candidate_generation: Dict[str, Any],
) -> bool:
    round_payload = _round_payload(decision, round_id)
    artifact_path = _stage3_artifact_path(workflow_dir, decision, round_id)
    artifact = _load_json_if_exists(artifact_path)
    if not artifact:
        return False
    artifact_policy_guidance = deepcopy(
        (artifact.get("input_payload", {}) or {}).get("policy_guidance") or {}
    )
    if artifact_policy_guidance and not bool((round_payload.get("policy_guidance") or {}).get("policy_guidance_used", False)):
        round_payload["policy_guidance"] = artifact_policy_guidance
    recovered_generator_candidates = list(artifact.get("final_candidates") or [])
    replay_validation_errors = list(artifact.get("validation_errors") or [])
    parsed = artifact.get("parsed") or {}
    if not recovered_generator_candidates and isinstance(parsed, dict):
        replay_input_config = artifact.get("input_payload", {}).get("current_mainline_config")
        sanitize_current_config = (
            replay_input_config if isinstance(replay_input_config, dict) and replay_input_config else round_anchor
        )
        recovered_generator_candidates, replay_errors = _sanitize_llm_candidate_payload(
            llm_payload=parsed,
            current_config=sanitize_current_config,
            candidate_generation=candidate_generation,
            round_policy_guidance=artifact_policy_guidance or round_payload.get("policy_guidance") or {},
            decision=decision,
        )
        replay_validation_errors.extend(replay_errors)
    if not recovered_generator_candidates:
        return False
    round_payload["generator_candidates"] = deepcopy(recovered_generator_candidates)
    round_payload["candidate_generation_status"] = str(
        artifact.get("status")
        or round_payload.get("candidate_generation_status")
        or "recovered_from_saved_artifact"
    )
    round_payload["candidate_generation_source"] = str(
        artifact.get("candidate_generation_source")
        or artifact.get("final_source")
        or round_payload.get("candidate_generation_source")
        or "saved_artifact_replay"
    )
    round_payload["candidate_generation_replay_validation_errors"] = replay_validation_errors
    if round_id == int(decision.get("active_round_id") or 1) or decision.get("llm_candidate_generation_status") is None:
        decision["llm_candidate_generation_status"] = str(artifact.get("status") or "")
        decision["candidate_generation_source"] = str(
            artifact.get("candidate_generation_source") or artifact.get("final_source") or "saved_artifact_replay"
        )
        decision["llm_candidate_generation_validation_errors"] = replay_validation_errors
        decision["llm_candidate_generation_repair_used"] = bool(artifact.get("repair_retry_used", False))
        decision["llm_candidate_generation_deterministic_recovery_added"] = bool(
            artifact.get("deterministic_recovery_candidate_added", False)
        )
        decision["llm_candidate_generation_cache_path"] = str(artifact_path)
    return True


def _reconcile_stage3_manifest_from_saved_artifacts(
    *,
    workflow_dir: Path,
    manifest: Dict[str, Any],
    decision_rule: Dict[str, Any],
    adaptive_method: Dict[str, Any],
) -> None:
    decisions = list(manifest.get("decisions") or [])
    current_config = deepcopy(manifest.get("initial_config") or manifest.get("current_config") or {})
    for decision in decisions:
        decision.setdefault("current_config", deepcopy(current_config))
        rounds = list(decision.get("rounds") or [])
        for round_payload in rounds:
            round_id = int(round_payload.get("round_id") or 1)
            round_anchor = _round_generation_context(
                decision=decision,
                current_config=current_config,
                round_id=round_id,
                decision_rule=decision_rule,
            )
            round_cfg = _decision_round_config(decision)
            candidate_generation = deepcopy(adaptive_method.get("candidate_generation") or {})
            candidate_generation["include_no_change"] = False
            candidate_generation["max_candidates_per_checkpoint"] = (
                round_cfg["round1_generated"] if round_id == 1 else round_cfg["round2_generated"]
            )
            if not list(round_payload.get("generator_candidates") or []):
                _recover_stage3_round_from_saved_artifact(
                    workflow_dir=workflow_dir,
                    decision=decision,
                    round_id=round_id,
                    round_anchor=round_anchor,
                    candidate_generation=candidate_generation,
                )
            if list(round_payload.get("generator_candidates") or []) and not list(round_payload.get("candidates") or []):
                _review_stage3_candidates(
                    round_payload=round_payload,
                    current_config=current_config,
                    round_id=round_id,
                    decision=decision,
                )
            for candidate in list(round_payload.get("candidates") or []):
                branch_result_path = _candidate_branch_result_path(
                    workflow_dir,
                    int(decision.get("decision_index") or 0),
                    str(decision.get("stage_label") or ""),
                    str(candidate.get("candidate_id") or ""),
                )
                if branch_result_path.exists():
                    candidate["branch_result_path"] = str(branch_result_path)
                    _hydrate_stage3_candidate_from_branch_result(
                        candidate,
                        workflow_dir=workflow_dir,
                    )
                    if bool(candidate.get("candidate_valid")) and not candidate.get("result_record"):
                        candidate["result_record"] = _build_result_record(
                            field_name="adaptive_combo",
                            stage_label=str(decision.get("stage_label") or ""),
                            checkpoint={
                                "name": decision.get("checkpoint_name"),
                                "step": decision.get("checkpoint_step"),
                            },
                            candidate_id=str(candidate.get("candidate_id") or ""),
                            candidate_value=None,
                            branch_result=deepcopy(candidate.get("branch_result") or {}),
                        )
            if list(round_payload.get("candidates") or []) and not round_payload.get("analysis") and _round_candidates_completed(round_payload):
                round_payload["analysis"] = _round_branch_result_summary(round_payload, decision_rule)
                round_payload["critic_analysis_status"] = "completed"
            if list(round_payload.get("candidates") or []):
                round_payload["status"] = (
                    "completed" if _round_candidates_completed(round_payload) else "running"
                )
        decision["candidates"] = _all_round_candidates(decision)
        if rounds and not decision.get("round1_result_summary") and (rounds[0].get("analysis") or {}):
            decision["round1_result_summary"] = deepcopy(rounds[0].get("analysis") or {})
        if decision.get("winner_config"):
            current_config = deepcopy(decision.get("winner_config") or current_config)
    manifest["decisions"] = decisions


def _checkpoint_target_reached(run_reference: Dict[str, Any], target_step: int) -> bool:
    latest = run_reference.get("latest_checkpoint_step")
    try:
        return int(latest) >= int(target_step)
    except (TypeError, ValueError):
        return False


def _candidate_endpoint_step(decision: Dict[str, Any], branch_budget_steps: int) -> int:
    return int(decision.get("checkpoint_step") or 0) + int(branch_budget_steps)


def _discover_endpoint_checkpoint(
    *,
    run_reference: Dict[str, Any],
    expected_endpoint_step: int,
) -> Dict[str, Any]:
    searched_paths: List[str] = []
    checkpoint_root_dir = run_reference.get("checkpoint_root_dir")
    available_steps = normalize_checkpoint_steps(run_reference.get("available_checkpoint_steps"))
    latest_model_path = run_reference.get("latest_model_path")
    checkpoint_root = Path(str(checkpoint_root_dir)).resolve() if checkpoint_root_dir else None
    if checkpoint_root is not None:
        searched_paths.append(str(checkpoint_root))
    if checkpoint_root is not None and checkpoint_root.exists():
        discovered_steps: List[int] = []
        for child in checkpoint_root.iterdir():
            if child.is_dir() and child.name.isdigit():
                discovered_steps.append(int(child.name))
                searched_paths.append(str(child))
        available_steps = sorted(set(available_steps) | set(discovered_steps))
    if isinstance(latest_model_path, str) and latest_model_path:
        latest_model = Path(latest_model_path).resolve()
        searched_paths.append(str(latest_model))
        if latest_model.name.isdigit():
            available_steps = sorted(set(available_steps) | {int(latest_model.name)})
            if checkpoint_root is None:
                checkpoint_root = latest_model.parent
    selected_step: Optional[int] = None
    if available_steps:
        le_steps = [step for step in available_steps if int(step) <= int(expected_endpoint_step)]
        if le_steps:
            selected_step = max(le_steps)
        else:
            selected_step = min(available_steps, key=lambda step: abs(int(step) - int(expected_endpoint_step)))
    endpoint_path = None
    if checkpoint_root is not None and selected_step is not None:
        endpoint_path = checkpoint_root / str(selected_step)
    checkpoint_exists = bool(endpoint_path is not None and endpoint_path.exists())
    return {
        "expected_endpoint_step": int(expected_endpoint_step),
        "actual_checkpoint_step": selected_step,
        "endpoint_checkpoint_path": str(endpoint_path) if endpoint_path is not None else None,
        "checkpoint_exists": checkpoint_exists,
        "distance_from_expected": (
            None if selected_step is None else abs(int(selected_step) - int(expected_endpoint_step))
        ),
        "searched_paths": searched_paths,
    }


def _materialize_branch_result(
    *,
    launcher: EPyMARLTrainLauncher,
    branch_plan: Dict[str, Any],
    branch_result_path: Path,
    candidate: Dict[str, Any],
    expected_endpoint_step: int,
) -> Dict[str, Any]:
    recovered = launcher.recover_existing_result(branch_plan["resume_train_config"])
    run_reference = recovered["run_reference"]
    metrics_summary = recovered.get("metrics_summary") or {}
    metric_summary = metrics_summary.get("metric_summary") if isinstance(metrics_summary, dict) else {}
    available_metric_keys = sorted(metric_summary.keys()) if isinstance(metric_summary, dict) else []
    endpoint = _discover_endpoint_checkpoint(
        run_reference=run_reference,
        expected_endpoint_step=expected_endpoint_step,
    )
    metrics_missing = not bool(available_metric_keys)
    metrics_source = "metrics_json" if available_metric_keys else "missing_metrics_json"
    candidate_valid = bool(endpoint["checkpoint_exists"])
    invalid_reason = None
    status = "completed"
    if not endpoint["checkpoint_exists"]:
        candidate_valid = False
        status = "completed_invalid"
        invalid_reason = "completed_run_missing_endpoint_checkpoint"
    branch_result = {
        "candidate_id": candidate.get("candidate_id"),
        "round_id": candidate.get("round_id"),
        "run_id": run_reference.get("run_id"),
        "status": status,
        "candidate_valid": candidate_valid,
        "pbrs_config": deepcopy(
            _normalize_rware_stage_config(
                {
                    "candidate_id": str(candidate.get("candidate_id") or "branch_result"),
                    "candidate_type": str(candidate.get("candidate_type") or "branch_result"),
                    "pbrs_version": "rware_pbrs_v2",
                    "mode": candidate.get("mode"),
                    "beta": candidate.get("beta"),
                    "active_terms": list(candidate.get("active_terms") or []),
                    "weights": deepcopy(candidate.get("weights") or {}),
                    "gamma": candidate.get("gamma", 0.99),
                    "evidence_keys_used": list(candidate.get("evidence_keys_used") or []),
                }
            )
            if str(candidate.get("pbrs_version") or "") == "rware_pbrs_v2"
            else
            normalize_lbf_pbrs_v2_config(candidate)
            if str(candidate.get("pbrs_version") or "") == "lbf_pbrs_v2"
            else {
                "beta": candidate.get("beta"),
                "wc": candidate.get("wc"),
                "wp": candidate.get("wp"),
            }
        ),
        "expected_endpoint_step": int(expected_endpoint_step),
        "actual_checkpoint_step": endpoint["actual_checkpoint_step"],
        "endpoint_checkpoint_path": endpoint["endpoint_checkpoint_path"],
        "checkpoint_exists": endpoint["checkpoint_exists"],
        "distance_from_expected": endpoint["distance_from_expected"],
        "metrics": metric_summary if isinstance(metric_summary, dict) else {},
        "metrics_summary": metrics_summary,
        "metrics_missing": metrics_missing,
        "metrics_source": metrics_source,
        "available_metric_keys": available_metric_keys,
        "invalid_reason": invalid_reason,
        "searched_paths": endpoint["searched_paths"],
        "train_config": recovered["train_config"],
        "run_reference": run_reference,
    }
    branch_result_path.parent.mkdir(parents=True, exist_ok=True)
    branch_result_path.write_text(
        json.dumps(branch_result, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return branch_result


def build_adaptive_checkpoint_replacement_manifest(
    *,
    workflow_id: str,
    stage_selection_result: Dict[str, Any],
    dense_reference_run_dir: str,
    dense_reference_selection: Dict[str, Any],
    python_executable: str,
    stage1b_source_metadata: Optional[Dict[str, Any]] = None,
    stage3_fork_spec_json: Optional[str] = None,
    execute: bool,
    storage_root: Optional[str] = None,
    branch_budget_steps: int = 205000,
    max_parallel_candidates: int = 2,
    branch_use_cuda: Optional[bool] = None,
    mainline_use_cuda: Optional[bool] = None,
    continuation_save_model_interval: int = 25000,
    t_max: int = 2050000,
    adaptive_method: Optional[Dict[str, Any]] = None,
    use_real_llm: bool = True,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-5.2",
    temperature: float = 0.2,
    llm_timeout: float = 60.0,
    llm_retry_count: int = 2,
    llm_retry_backoff: float = 5.0,
) -> Dict[str, Any]:
    adaptive_method = deepcopy(adaptive_method or {})
    adaptive_method["use_real_llm"] = bool(use_real_llm)
    adaptive_method["api_key_env"] = str(api_key_env)
    adaptive_method["base_url"] = str(base_url)
    adaptive_method["model"] = str(model)
    adaptive_method["llm_routing"] = deepcopy(adaptive_method.get("llm_routing") or {})
    adaptive_method["pbrs_version"] = str(
        adaptive_method.get("pbrs_version")
        or (stage1b_source_metadata.get("adaptive_mainline_initial_pbrs_config") or {}).get("pbrs_version")
        or (stage1b_source_metadata.get("selected_initial_dense_config") or {}).get("pbrs_version")
        or ""
    )
    adaptive_method["temperature"] = float(temperature)
    adaptive_method["llm_timeout"] = float(llm_timeout)
    adaptive_method["llm_max_retries"] = int(llm_retry_count)
    adaptive_method["llm_retry_backoff"] = float(llm_retry_backoff)
    stage3_fork_spec = _load_stage3_fork_spec_json(stage3_fork_spec_json)
    stage1b_source_metadata = _validate_stage1b_source_metadata(stage1b_source_metadata or {})
    dense_reference_selection = deepcopy(dense_reference_selection or {})
    if stage3_fork_spec is not None:
        needs_fork_dense_selection = any(
            dense_reference_selection.get(key) is None
            for key in ("selected_beta", "selected_wc", "selected_wp")
        )
        if needs_fork_dense_selection:
            fork_winner_config = _normalize_stage3_fork_winner_config(
                deepcopy(stage3_fork_spec.get("winner_config") or {}),
                env_key=((stage_selection_result.get("sparse_summary") or {}).get("env_key")),
                pbrs_version=adaptive_method.get("pbrs_version"),
            )
            for target_key, winner_key in (
                ("selected_beta", "beta"),
                ("selected_wc", "wc"),
                ("selected_wp", "wp"),
            ):
                if dense_reference_selection.get(target_key) is None:
                    dense_reference_selection[target_key] = fork_winner_config.get(winner_key)
    selected_checkpoints, checkpoint_schedule_source = _resolve_selected_checkpoints(
        stage_selection_result=stage_selection_result,
        adaptive_method=adaptive_method,
    )
    if not selected_checkpoints:
        raise ValueError(
            "Stage3 adaptive decision initialization requires selected_checkpoints from "
            "stage_2_selection.json or adaptive_method.fixed_intervention_checkpoints."
        )
    effective_stage_selection_result = _stage_selection_with_selected_checkpoints(
        stage_selection_result,
        selected_checkpoints,
        schedule_source=checkpoint_schedule_source,
    )
    initial_config = {
        "beta": float(dense_reference_selection["selected_beta"]),
        "wc": float(dense_reference_selection["selected_wc"]),
        "wp": float(dense_reference_selection["selected_wp"]),
    }
    if adaptive_method["pbrs_version"] == "lbf_pbrs_v2":
        initial_config = _normalize_lbf_stage_config(
            deepcopy(
                stage1b_source_metadata.get("adaptive_mainline_initial_pbrs_config")
                or stage1b_source_metadata.get("selected_initial_dense_config")
                or initial_config
            )
        )
    storage = WorkflowStorage.create(
        workflow_id,
        root_dir=storage_root or DEFAULT_RESULTS_ROOT,
        manifest_data={"status": "running", "workflow_kind": "adaptive_checkpoint_replacement"},
    )
    workflow_dir = storage.workflow_dir
    manifest_path = _adaptive_manifest_alias_path(workflow_dir)
    manifest = _load_existing_adaptive_manifest(workflow_dir)
    if _manifest_requires_decision_initialization(
        manifest,
        selected_checkpoints=selected_checkpoints,
    ):
        manifest = _build_initial_manifest(
            workflow_id=workflow_id,
            dense_reference_run_dir=dense_reference_run_dir,
            stage_selection_result=effective_stage_selection_result,
            initial_config=initial_config,
            branch_budget_steps=branch_budget_steps,
            t_max=t_max,
            adaptive_method=adaptive_method,
        )
        manifest["stage1b_source_metadata"] = deepcopy(stage1b_source_metadata)
        manifest["status"] = "running"
        manifest["initialization_status"] = "recovered_from_incomplete_manifest"
        manifest["checkpoint_schedule_source"] = checkpoint_schedule_source
        manifest["initialization_recovery_reason"] = (
            "existing adaptive manifest had no decisions despite available checkpoint schedule"
        )
    elif manifest:
        manifest.setdefault("stage1b_source_metadata", deepcopy(stage1b_source_metadata))
        manifest.setdefault("checkpoint_schedule_source", checkpoint_schedule_source)
        manifest.setdefault(
            "stage_selection_result",
            deepcopy(effective_stage_selection_result),
        )
    else:
        manifest = _build_initial_manifest(
            workflow_id=workflow_id,
            dense_reference_run_dir=dense_reference_run_dir,
            stage_selection_result=effective_stage_selection_result,
            initial_config=initial_config,
            branch_budget_steps=branch_budget_steps,
            t_max=t_max,
            adaptive_method=adaptive_method,
        )
        manifest["stage1b_source_metadata"] = deepcopy(stage1b_source_metadata)
        manifest["checkpoint_schedule_source"] = checkpoint_schedule_source
    if stage3_fork_spec is not None and not bool((manifest.get("fork_bootstrap") or {}).get("enabled")):
        manifest = _apply_stage3_fork_bootstrap(
            manifest=manifest,
            stage3_fork_spec=stage3_fork_spec,
        )

    launcher = EPyMARLTrainLauncher(
        repo_root=Path(__file__).resolve().parents[2],
        python_executable=python_executable,
    )
    current_config = deepcopy(manifest.get("current_config") or manifest.get("initial_config") or {})
    decisions = list(manifest.get("decisions") or [])
    local_results_path = "results"
    decision_rule = deepcopy((adaptive_method.get("decision_rule") or {}))
    dense_reference_run_summary = _load_json(Path(dense_reference_run_dir) / "info.json").get("result") or {}
    memory = load_or_init_memory(
        workflow_dir,
        _stage3_memory_static_context(
            workflow_id=workflow_id,
            dense_reference_run_summary=dense_reference_run_summary,
            branch_budget_steps=branch_budget_steps,
            llm_routing=adaptive_method.get("llm_routing") or {},
        ),
    )
    memory = update_after_fixed_checkpoint_plan(
        memory,
        deepcopy(selected_checkpoints),
    )
    save_memory(workflow_dir, memory)
    launch_first_change_baseline = bool(
        adaptive_method.get("launch_first_change_fixed_initial_baseline", True)
    )
    first_change_baseline_non_blocking = bool(
        adaptive_method.get("first_change_baseline_non_blocking", True)
    )

    if decisions and not decisions[0].get("source_run_dir"):
        first_decision = decisions[0]
        dense_base_train_config = _load_json(Path(dense_reference_run_dir) / "config.json")
        local_results_path = str(dense_base_train_config.get("local_results_path") or "results")
        adaptive_source_path = Path(str(stage1b_source_metadata["adaptive_mainline_source_checkpoint_path"]))
        adaptive_source_step = int(stage1b_source_metadata["selected_endpoint_checkpoint_step"])
        adaptive_checkpoint_root_dir = str(adaptive_source_path.parent)
        adaptive_available_steps = [adaptive_source_step]
        _reconcile_initial_mainline_seed(
            workflow_dir=workflow_dir,
            manifest=manifest,
            launcher=launcher,
            base_train_config=dense_base_train_config,
            workflow_id=workflow_id,
            initial_config=deepcopy(manifest.get("initial_config") or {}),
            checkpoint_name=str(first_decision.get("checkpoint_name") or "early"),
            checkpoint_step=int(first_decision.get("checkpoint_step") or 0),
            checkpoint_root_dir=adaptive_checkpoint_root_dir,
            available_steps=adaptive_available_steps,
            source_checkpoint_step=adaptive_source_step,
            mainline_use_cuda=mainline_use_cuda,
            local_results_path=local_results_path,
        )
        seed_state = manifest.get("initial_mainline_seed") or {}
        if execute and str(seed_state.get("status")) in {"pending", "planned"}:
            _dispatch_initial_mainline_seed(
                workflow_dir=workflow_dir,
                manifest=manifest,
                launcher=launcher,
                base_train_config=dense_base_train_config,
                workflow_id=workflow_id,
                initial_config=deepcopy(manifest.get("initial_config") or {}),
                checkpoint_name=str(first_decision.get("checkpoint_name") or "early"),
                checkpoint_step=int(first_decision.get("checkpoint_step") or 0),
                checkpoint_root_dir=adaptive_checkpoint_root_dir,
                available_steps=adaptive_available_steps,
                source_checkpoint_step=adaptive_source_step,
                mainline_use_cuda=mainline_use_cuda,
                local_results_path=local_results_path,
            )
            manifest["decisions"] = decisions
            _normalize_stage3_manifest_summary(manifest, decisions)
            _save_adaptive_manifest(workflow_dir, manifest)
            return manifest
        if str(seed_state.get("status")) != "completed":
            manifest["decisions"] = decisions
            _normalize_stage3_manifest_summary(manifest, decisions)
            _save_adaptive_manifest(workflow_dir, manifest)
            return manifest
        if seed_state.get("run_dir"):
            first_decision["source_run_dir"] = str(seed_state.get("run_dir"))
            first_decision["source_checkpoint_root_dir"] = seed_state.get("checkpoint_root_dir")
            first_decision["source_available_steps"] = list(seed_state.get("available_checkpoint_steps") or [])
            first_decision["source_latest_checkpoint_step"] = seed_state.get("latest_checkpoint_step")

    _reconcile_stage3_manifest_from_saved_artifacts(
        workflow_dir=workflow_dir,
        manifest=manifest,
        decision_rule=decision_rule,
        adaptive_method=adaptive_method,
    )
    decisions = list(manifest.get("decisions") or [])

    for index, decision in enumerate(decisions):
        stop_after_checkpoint = str(
            adaptive_method.get("stop_after_checkpoint_name") or ""
        ).strip().upper()
        if stop_after_checkpoint and any(
            str(previous.get("checkpoint_name") or "").strip().upper() == stop_after_checkpoint
            and bool(previous.get("completed"))
            for previous in decisions[:index]
        ):
            manifest["current_config"] = deepcopy(current_config)
            manifest["decisions"] = decisions
            _normalize_stage3_manifest_summary(manifest, decisions)
            _save_adaptive_manifest(workflow_dir, manifest)
            return manifest
        if _decision_complete(decision):
            _propagate_completed_continuation_to_next_decision(
                workflow_dir=workflow_dir,
                decisions=decisions,
                decision_index=index,
            )
            if (
                bool(manifest.get("first_change_baseline_launched"))
                and (manifest.get("first_change_baseline") or {}).get("source_decision_id") == decision.get("decision_index")
            ):
                source_run_dir = decision.get("source_run_dir") or dense_reference_run_dir
                source_run_reference = _build_run_reference_from_run_dir(source_run_dir)
                base_train_config = _load_json(Path(source_run_dir) / "config.json")
                local_results_path = str(base_train_config.get("local_results_path") or "results")
                _reconcile_first_change_baseline(
                    workflow_dir=workflow_dir,
                    manifest=manifest,
                    decision=decision,
                    launcher=launcher,
                    base_train_config=base_train_config,
                    checkpoint_root_dir=str(source_run_reference["checkpoint_root_dir"]),
                    available_steps=list(source_run_reference["available_checkpoint_steps"] or []),
                    mainline_use_cuda=mainline_use_cuda,
                    local_results_path=local_results_path,
                    save_model_interval=continuation_save_model_interval,
                    workflow_id=workflow_id,
                )
            current_config = deepcopy(decision.get("winner_config") or current_config)
            continue

        if index > 0 and _decision_source_fields_missing(decision):
            _propagate_completed_continuation_to_next_decision(
                workflow_dir=workflow_dir,
                decisions=decisions,
                decision_index=index - 1,
            )
        source_run_dir = decision.get("source_run_dir") or dense_reference_run_dir
        source_run_reference = _decision_source_run_reference(
            decision=decision,
            fallback_run_dir=dense_reference_run_dir,
        )
        readiness = source_run_reference["baseline_readiness"]
        base_train_config = _load_json(Path(source_run_dir) / "config.json")
        local_results_path = str(base_train_config.get("local_results_path") or "results")
        checkpoint_root_dir = str(source_run_reference["checkpoint_root_dir"])
        available_steps = list(source_run_reference["available_checkpoint_steps"] or [])
        decision["source_checkpoint_root_dir"] = checkpoint_root_dir
        decision["source_available_steps"] = available_steps
        decision["source_latest_checkpoint_step"] = source_run_reference.get("latest_checkpoint_step")
        decision["source_latest_checkpoint_path"] = source_run_reference.get("latest_model_path")
        decision["current_config"] = deepcopy(current_config)
        decision["critic_pre_diagnosis_status"] = "completed"
        decision["critic_pre_diagnosis"] = {
            "source_checkpoint_step": int(decision["checkpoint_step"]),
            "source_run_dir": source_run_dir,
            "adaptive_mainline_source_checkpoint_path": manifest.get("stage1b_source_metadata", {}).get("adaptive_mainline_source_checkpoint_path"),
            "dense_reference_source_checkpoint_path": manifest.get("stage1b_source_metadata", {}).get("dense_reference_source_checkpoint_path"),
        }
        if not _decision_branch_search_skipped(decision):
            for round_id in range(1, max(1, int(decision.get("stage3_branch_rounds") or 2)) + 1):
                round_payload = _round_payload(decision, round_id)
                memory = _maybe_attach_stage3_policy_guidance(
                    workflow_dir=workflow_dir,
                    decision=decision,
                    round_payload=round_payload,
                    round_id=round_id,
                    current_config=current_config,
                    adaptive_method=adaptive_method,
                    dense_reference_run_summary=dense_reference_run_summary,
                    memory=memory,
                )
                save_memory(workflow_dir, memory)
                if not list(round_payload.get("generator_candidates") or []):
                    round_anchor = _round_generation_context(
                        decision=decision,
                        current_config=current_config,
                        round_id=round_id,
                        decision_rule=decision_rule,
                    )
                    candidate_generation = deepcopy(adaptive_method.get("candidate_generation") or {})
                    round_cfg = _decision_round_config(decision)
                    candidate_generation["include_no_change"] = False
                    candidate_generation["max_candidates_per_checkpoint"] = (
                        round_cfg["round1_generated"] if round_id == 1 else round_cfg["round2_generated"]
                    )
                    decision["active_round_id"] = round_id
                    generated = _generate_candidates_with_llm_or_fallback(
                        workflow_dir=workflow_dir,
                        decision=decision,
                        current_config=round_anchor,
                        adaptive_method={**adaptive_method, "candidate_generation": candidate_generation},
                        stage_selection_result=effective_stage_selection_result,
                        dense_reference_selection=dense_reference_selection,
                        dense_reference_run_summary=dense_reference_run_summary,
                        branch_budget_steps=branch_budget_steps,
                        use_real_llm=use_real_llm,
                        api_key_env=api_key_env,
                        base_url=base_url,
                        model=model,
                        temperature=temperature,
                        llm_timeout=llm_timeout,
                        llm_retry_count=llm_retry_count,
                        llm_retry_backoff=llm_retry_backoff,
                    )
                    generated = [item for item in generated if not bool(item.get("is_no_change"))]
                    if round_id > 1:
                        seen_prior = {_candidate_identity(item) for item in _all_round_candidates(decision)}
                        generated = [item for item in generated if _candidate_identity(item) not in seen_prior]
                    round_payload["generator_candidates"] = generated
                    round_payload["candidate_generation_status"] = str(decision.get("llm_candidate_generation_status") or "completed")
                    round_payload["candidate_generation_source"] = str(
                        decision.get("candidate_generation_source") or "deterministic_fallback"
                    )
                    round_payload["candidate_generation_backend_probe"] = deepcopy(
                        decision.get("llm_candidate_generation_backend_probe") or {}
                    )
                    append_stage3_policy_guided_candidate_generation(
                        memory,
                        {
                            "checkpoint_id": decision.get("checkpoint_name"),
                            "round_id": round_id,
                            "policy_guidance_used": bool(
                                (round_payload.get("policy_guidance") or {}).get("policy_guidance_used", False)
                            ),
                            "candidate_generation_basis": {
                                "evidence_keys": list(
                                    ((round_payload.get("policy_guidance") or {}).get("policy_guidance_evidence_keys"))
                                    or []
                                ),
                            },
                            "candidates": generated,
                        },
                    )
                    save_memory(workflow_dir, memory)
                if decision.get("stage3_candidate_review_enabled", True) and round_payload.get("candidate_review_status") != "completed":
                    _review_stage3_candidates(
                        round_payload=round_payload,
                        current_config=current_config,
                        round_id=round_id,
                        decision=decision,
                    )
                decision["candidates"] = _all_round_candidates(decision)
                _reconcile_branch_candidates(
                    workflow_dir=workflow_dir,
                    decision=decision,
                    launcher=launcher,
                    base_train_config=base_train_config,
                    checkpoint_root_dir=checkpoint_root_dir,
                    available_steps=available_steps,
                    branch_budget_steps=branch_budget_steps,
                    branch_use_cuda=branch_use_cuda,
                    local_results_path=local_results_path,
                    workflow_id=workflow_id,
                )
                round_payload["status"] = "completed" if _round_candidates_completed(round_payload) else "running"
                if execute and not _round_candidates_completed(round_payload):
                    _dispatch_branch_candidates(
                        workflow_dir=workflow_dir,
                        decision=decision,
                        launcher=launcher,
                        base_train_config=base_train_config,
                        checkpoint_root_dir=checkpoint_root_dir,
                        available_steps=available_steps,
                        branch_budget_steps=branch_budget_steps,
                        branch_use_cuda=branch_use_cuda,
                        local_results_path=local_results_path,
                        workflow_id=workflow_id,
                        max_parallel_candidates=max_parallel_candidates,
                    )
                    manifest["current_config"] = deepcopy(current_config)
                    manifest["decisions"] = decisions
                    _normalize_stage3_manifest_summary(manifest, decisions)
                    _save_adaptive_manifest(workflow_dir, manifest)
                    return manifest
                if not _round_candidates_completed(round_payload):
                    manifest["current_config"] = deepcopy(current_config)
                    manifest["decisions"] = decisions
                    _normalize_stage3_manifest_summary(manifest, decisions)
                    _save_adaptive_manifest(workflow_dir, manifest)
                    return manifest
                round_payload["analysis"] = _round_branch_result_summary(round_payload, decision_rule)
                round_payload["critic_analysis_status"] = "completed"
                round_payload["status"] = "completed"
                if round_id == 1:
                    decision["round1_result_summary"] = deepcopy(round_payload["analysis"])
            decision["candidates"] = _all_round_candidates(decision)
        else:
            decision["decision_source"] = str(decision.get("decision_source") or "stage3_fork_bootstrap")
            decision["final_decision_source"] = str(
                decision.get("final_decision_source") or decision.get("decision_source") or "stage3_fork_bootstrap"
            )
            if not decision.get("decision_summary_path"):
                summary_path = _write_decision_summary(
                    workflow_dir=workflow_dir,
                    decision=decision,
                    decision_rule=decision_rule,
                )
                decision["decision_summary_path"] = str(summary_path)

        if decision.get("winner_candidate_id") is None:
            _select_winner_candidate(
                workflow_dir=workflow_dir,
                decision=decision,
                decision_rule=decision_rule,
                adaptive_method=adaptive_method,
                initial_config=deepcopy(manifest.get("initial_config") or {}),
                previous_decisions=deepcopy(decisions[:index]),
                fixed_reference_run_dir=str(manifest.get("dense_reference_run_dir") or ""),
                branch_budget_steps=int(branch_budget_steps),
                use_real_llm=use_real_llm,
                api_key_env=api_key_env,
                base_url=base_url,
                model=model,
                temperature=temperature,
                llm_timeout=llm_timeout,
                llm_retry_count=llm_retry_count,
                llm_retry_backoff=llm_retry_backoff,
            )
            summary_path = _write_decision_summary(
                workflow_dir=workflow_dir,
                decision=decision,
                decision_rule=decision_rule,
            )
            decision["decision_summary_path"] = str(summary_path)
            memory = update_after_stage3_decision(memory, deepcopy(decision))
            save_memory(workflow_dir, memory)

        first_real_update = (
            launch_first_change_baseline
            and not bool(manifest.get("first_change_baseline_launched"))
            and bool(decision.get("replacement_accepted"))
            and deepcopy(decision.get("winner_config") or {}) != deepcopy(current_config)
        )
        if first_real_update:
            baseline = manifest.get("first_change_baseline") or {}
            baseline.update(
                {
                    "status": "planned",
                    "source_decision_id": decision.get("decision_index"),
                    "source_checkpoint_step": decision.get("checkpoint_step"),
                    "initial_dense_config": deepcopy(manifest.get("initial_config") or {}),
                }
            )
            manifest["first_change_baseline_launched"] = True
            manifest["first_change_baseline"] = baseline

        if bool(manifest.get("first_change_baseline_launched")) and (
            (manifest.get("first_change_baseline") or {}).get("source_decision_id") == decision.get("decision_index")
        ):
            _reconcile_first_change_baseline(
                workflow_dir=workflow_dir,
                manifest=manifest,
                decision=decision,
                launcher=launcher,
                base_train_config=base_train_config,
                checkpoint_root_dir=checkpoint_root_dir,
                available_steps=available_steps,
                mainline_use_cuda=mainline_use_cuda,
                local_results_path=local_results_path,
                save_model_interval=continuation_save_model_interval,
                workflow_id=workflow_id,
            )
            if execute and str((manifest.get("first_change_baseline") or {}).get("status")) in {"pending", "planned"}:
                _dispatch_first_change_baseline(
                    workflow_dir=workflow_dir,
                    manifest=manifest,
                    decision=decision,
                    launcher=launcher,
                    base_train_config=base_train_config,
                    checkpoint_root_dir=checkpoint_root_dir,
                    available_steps=available_steps,
                    mainline_use_cuda=mainline_use_cuda,
                    local_results_path=local_results_path,
                    save_model_interval=continuation_save_model_interval,
                    workflow_id=workflow_id,
                )

        _reconcile_continuation(
            workflow_dir=workflow_dir,
            decision=decision,
            launcher=launcher,
            base_train_config=base_train_config,
            checkpoint_root_dir=checkpoint_root_dir,
            available_steps=available_steps,
            mainline_use_cuda=mainline_use_cuda,
            local_results_path=local_results_path,
            save_model_interval=continuation_save_model_interval,
            workflow_id=workflow_id,
        )
        if execute and str((decision.get("continuation") or {}).get("status")) not in {"completed"}:
            _dispatch_continuation(
                workflow_dir=workflow_dir,
                decision=decision,
                launcher=launcher,
                base_train_config=base_train_config,
                checkpoint_root_dir=checkpoint_root_dir,
                available_steps=available_steps,
                mainline_use_cuda=mainline_use_cuda,
                local_results_path=local_results_path,
                save_model_interval=continuation_save_model_interval,
                workflow_id=workflow_id,
            )
            manifest["current_config"] = deepcopy(current_config)
            manifest["decisions"] = decisions
            _normalize_stage3_manifest_summary(manifest, decisions)
            _save_adaptive_manifest(workflow_dir, manifest)
            return manifest

        if str((decision.get("continuation") or {}).get("status")) != "completed":
            manifest["current_config"] = deepcopy(current_config)
            manifest["decisions"] = decisions
            _normalize_stage3_manifest_summary(manifest, decisions)
            _save_adaptive_manifest(workflow_dir, manifest)
            return manifest

        decision["completed"] = True
        _propagate_completed_continuation_to_next_decision(
            workflow_dir=workflow_dir,
            decisions=decisions,
            decision_index=index,
        )
        current_config = deepcopy(decision.get("winner_config") or current_config)
        manifest["current_config"] = deepcopy(current_config)
        manifest["decisions"] = decisions
        _normalize_stage3_manifest_summary(manifest, decisions)
        _save_adaptive_manifest(workflow_dir, manifest)

        if (
            bool(manifest.get("first_change_baseline_launched"))
            and not first_change_baseline_non_blocking
            and (manifest.get("first_change_baseline") or {}).get("source_decision_id") == decision.get("decision_index")
        ):
            baseline_status = str((manifest.get("first_change_baseline") or {}).get("status") or "")
            if baseline_status != "completed":
                return manifest

    manifest["current_config"] = deepcopy(current_config)
    manifest["decisions"] = decisions
    _normalize_stage3_manifest_summary(manifest, decisions)
    all_decisions_completed = bool(decisions) and all(
        bool(decision.get("completed")) for decision in decisions
    )
    final_spec_ready = bool(decisions) and all(
        bool(decision.get("completed")) and bool(decision.get("winner_config"))
        for decision in decisions
    )
    manifest["readiness"] = {
        "all_decisions_completed": all_decisions_completed,
        "final_spec_ready": final_spec_ready,
        "final_mainline_completed": bool(decisions)
        and str((decisions[-1].get("continuation") or {}).get("status") or "") == "completed",
    }
    if manifest["readiness"]["final_spec_ready"]:
        final_spec = _build_adaptive_final_spec(
            selected_checkpoints=deepcopy(selected_checkpoints),
            decisions=decisions,
        )
        final_spec_path = workflow_dir / "final_adaptive_reward_spec.json"
        _save_json(final_spec_path, final_spec)
        manifest["artifacts"]["adaptive_final_spec_json"] = str(final_spec_path)
    final_run_dir = ((decisions[-1].get("continuation") or {}).get("run_dir")) if decisions else None
    if final_run_dir:
        final_run_payload = {
            "result": {
                "run_reference": {
                    "run_dir": final_run_dir,
                    "run_id": (decisions[-1].get("continuation") or {}).get("run_id"),
                }
            }
        }
        final_run_json = workflow_dir / "final_mainline_run.json"
        _save_json(final_run_json, final_run_payload)
        manifest["artifacts"]["final_mainline_run_json"] = str(final_run_json)
    _save_adaptive_manifest(workflow_dir, manifest)
    return manifest
