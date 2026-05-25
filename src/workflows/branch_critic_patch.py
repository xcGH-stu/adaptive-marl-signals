from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from workflows.checkpoint_selector_schema import SUPPORTED_STAGE_LABELS
from workflows.storage import WorkflowStorage


DEFAULT_BRANCH_PLAN_ROOT = (
    Path(__file__).resolve().parents[2] / "results" / "pbrs_branch_plans"
)

ROUND_RANKING_FIELDS = [
    "best_test_sparse_return_mean",
    "last_test_sparse_return_mean",
    "best_test_return_mean",
    "last_test_return_mean",
    "last_mixed_return_mean",
    "last_sparse_return_mean",
    "last_return_mean",
]


def is_stage_key(key: str) -> bool:
    return key in SUPPORTED_STAGE_LABELS


def deepcopy_workflow_spec(workflow_spec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return deepcopy(workflow_spec) if isinstance(workflow_spec, dict) else None


def ensure_nested_dict(root: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        value = {}
        root[key] = value
    return value


def _extract_metric_last(summary: Optional[Dict[str, Any]], metric_name: str) -> Any:
    if not isinstance(summary, dict):
        return None
    metric_summary = summary.get("metric_summary", {})
    payload = metric_summary.get(metric_name)
    if not isinstance(payload, dict):
        return None
    return payload.get("last_value")


def _extract_metric_best(summary: Optional[Dict[str, Any]], metric_name: str) -> Any:
    if not isinstance(summary, dict):
        return None
    metric_summary = summary.get("metric_summary", {})
    payload = metric_summary.get(metric_name)
    if not isinstance(payload, dict):
        return None
    return payload.get("best_value")


def _choose_ranking_metric(records: list[Dict[str, Any]]) -> Optional[str]:
    for field_name in ROUND_RANKING_FIELDS:
        values = [record.get(field_name) for record in records if record.get(field_name) is not None]
        if len(values) < 2:
            continue
        distinct = {round(float(value), 12) for value in values}
        if len(distinct) > 1:
            return field_name
    for field_name in ROUND_RANKING_FIELDS:
        if any(record.get(field_name) is not None for record in records):
            return field_name
    return None


def _summarize_branch_candidate_record(
    *,
    branch_plan_id: str,
    round_id: int,
    checkpoint: Dict[str, Any],
    resolved_round_state: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    branch_result = candidate.get("branch_result") or {}
    metrics_summary = branch_result.get("metrics_summary") or {}
    run_reference = branch_result.get("run_reference") or {}
    status = candidate.get("branch_result_status")
    if status not in {"planned_only", "executed", "reused"}:
        status = "finished" if metrics_summary else "planned_only"
    return {
        "branch_plan_id": branch_plan_id,
        "round_id": round_id,
        "checkpoint_name": checkpoint.get("name"),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_stage_label": checkpoint.get("stage_label"),
        "active_pbrs_field": resolved_round_state.get("active_pbrs_field"),
        "active_pbrs_field_source": resolved_round_state.get("active_pbrs_field_source"),
        "candidate_value_source": (
            (resolved_round_state.get("candidate_selection_context") or {}).get(
                "candidate_value_source"
            )
        ),
        "candidate_id": candidate.get("candidate_id"),
        "candidate_value": candidate.get("candidate_value"),
        "run_id": run_reference.get("run_id"),
        "status": status,
        "last_test_sparse_return_mean": _extract_metric_last(
            metrics_summary, "test_sparse_return_mean"
        ),
        "best_test_sparse_return_mean": _extract_metric_best(
            metrics_summary, "test_sparse_return_mean"
        ),
        "last_sparse_return_mean": _extract_metric_last(metrics_summary, "sparse_return_mean"),
        "last_mixed_return_mean": _extract_metric_last(metrics_summary, "mixed_return_mean"),
        "last_test_return_mean": _extract_metric_last(metrics_summary, "test_return_mean"),
        "best_test_return_mean": _extract_metric_best(metrics_summary, "test_return_mean"),
        "last_return_mean": _extract_metric_last(metrics_summary, "return_mean"),
    }


def build_branch_plan_summary_from_manifest(
    manifest: Dict[str, Any],
    *,
    branch_plan_dir: Optional[str] = None,
) -> Dict[str, Any]:
    branch_plan_id = manifest.get("branch_plan_id")
    round_summaries = []
    checkpoint_comparisons = []
    aggregate_best_candidates = []
    for round_payload in manifest.get("rounds", []):
        round_id = int(round_payload["round_id"])
        checkpoint = round_payload.get("checkpoint") or {}
        resolved_round_state = round_payload.get("resolved_round_state") or {}
        records = [
            _summarize_branch_candidate_record(
                branch_plan_id=branch_plan_id,
                round_id=round_id,
                checkpoint=checkpoint,
                resolved_round_state=resolved_round_state,
                candidate=candidate,
            )
            for candidate in round_payload.get("branch_candidates", [])
        ]
        ranking_metric = _choose_ranking_metric(records)
        ranked_records = list(records)
        if ranking_metric is not None:
            ranked_records = sorted(
                records,
                key=lambda record: (
                    record.get(ranking_metric) is not None,
                    record.get(ranking_metric) or float("-inf"),
                ),
                reverse=True,
            )
        best_candidate = ranked_records[0] if ranked_records else None
        round_summary = {
            "round_id": round_id,
            "checkpoint": checkpoint,
            "resolved_round_state": resolved_round_state,
            "ranking_metric": ranking_metric,
            "best_candidate": best_candidate,
            "records": ranked_records,
        }
        round_summaries.append(round_summary)
        checkpoint_payload = {
            "round_id": round_id,
            "checkpoint_name": checkpoint.get("name"),
            "checkpoint_step": checkpoint.get("step"),
            "checkpoint_stage_label": checkpoint.get("stage_label"),
            "checkpoint_reason": checkpoint.get("reason"),
            "recommended_focus": checkpoint.get("recommended_focus"),
            "active_pbrs_field": resolved_round_state.get("active_pbrs_field"),
            "active_pbrs_field_source": resolved_round_state.get("active_pbrs_field_source"),
            "candidate_value_source": (
                (resolved_round_state.get("candidate_selection_context") or {}).get(
                    "candidate_value_source"
                )
            ),
            "effective_candidate_values": (
                (resolved_round_state.get("candidate_selection_context") or {}).get(
                    "effective_candidate_values",
                    [],
                )
            ),
            "ranking_metric": ranking_metric,
            "planned_only": all(record.get("status") == "planned_only" for record in ranked_records),
            "best_candidate": best_candidate if best_candidate else None,
            "candidate_records": ranked_records,
        }
        checkpoint_comparisons.append(checkpoint_payload)
        if best_candidate:
            aggregate_best_candidates.append(
                {
                    "round_id": round_id,
                    "checkpoint_name": checkpoint.get("name"),
                    "checkpoint_stage_label": checkpoint.get("stage_label"),
                    "active_pbrs_field": resolved_round_state.get("active_pbrs_field"),
                    "ranking_metric": ranking_metric,
                    "candidate_id": best_candidate.get("candidate_id"),
                    "candidate_value": best_candidate.get("candidate_value"),
                    "score": (
                        best_candidate.get(ranking_metric)
                        if isinstance(ranking_metric, str)
                        else None
                    ),
                    "planned_only": checkpoint_payload["planned_only"],
                }
            )
    return {
        "branch_plan_id": branch_plan_id,
        "branch_plan_dir": branch_plan_dir,
        "checkpoint_selection": manifest.get("checkpoint_selection"),
        "branch_execution_summary": manifest.get("branch_execution_summary") or {},
        "branch_resume_summary": manifest.get("branch_resume_summary") or {},
        "rounds": round_summaries,
        "critic_payload": {
            "branch_plan_id": branch_plan_id,
            "checkpoint_selection": manifest.get("checkpoint_selection") or {},
            "checkpoint_comparisons": checkpoint_comparisons,
            "aggregate_best_candidates": aggregate_best_candidates,
        },
    }


def select_branch_evidence_for_checkpoint_context(
    branch_plan_summary: Optional[Dict[str, Any]],
    active_checkpoint_context: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(branch_plan_summary, dict) or not isinstance(active_checkpoint_context, dict):
        return None
    checkpoint_name = active_checkpoint_context.get("name")
    stage_label = active_checkpoint_context.get("stage_label")
    comparisons = ((branch_plan_summary.get("critic_payload") or {}).get("checkpoint_comparisons")) or []
    if not isinstance(comparisons, list):
        return None
    match_by_name = None
    match_by_stage = None
    for item in comparisons:
        if not isinstance(item, dict):
            continue
        if checkpoint_name is not None and str(item.get("checkpoint_name")) == str(checkpoint_name):
            match_by_name = deepcopy(item)
            break
        if stage_label is not None and str(item.get("checkpoint_stage_label")) == str(stage_label):
            match_by_stage = deepcopy(item)
    selected = match_by_name or match_by_stage
    if not isinstance(selected, dict):
        return None
    selected["selection_basis"] = "checkpoint_name" if match_by_name is not None else "stage_label"
    return selected


def build_design_transition_application(
    *,
    recommended_design_transition: Any,
    target_next_round_id: Optional[int] = None,
) -> Dict[str, Any]:
    if not isinstance(recommended_design_transition, dict):
        return {}

    transition_action = recommended_design_transition.get("transition_action")
    target_scope = recommended_design_transition.get("target_scope")
    target_stage = recommended_design_transition.get("target_stage")
    allowed_fields = recommended_design_transition.get("allowed_structural_pbrs_fields")
    normalized_fields = (
        [str(field_name) for field_name in allowed_fields]
        if isinstance(allowed_fields, list)
        else []
    )

    policy_patch = {
        "allowed_structural_pbrs_fields": normalized_fields,
        "allow_variant_updates": "variant" in normalized_fields,
        "allow_gate_updates": any(
            field_name in {"gate_mode", "gate_radius"} for field_name in normalized_fields
        ),
        "allow_closeness_updates": "closeness_mode" in normalized_fields,
    }
    if transition_action == "keep_coefficients_only":
        policy_patch = {
            "allowed_structural_pbrs_fields": [],
            "allow_variant_updates": False,
            "allow_gate_updates": False,
            "allow_closeness_updates": False,
        }

    application = {
        "transition_action": transition_action,
        "target_scope": target_scope,
        "target_stage": target_stage,
        "target_round_id": None,
        "policy_patch": policy_patch,
        "applied": False,
    }
    if target_scope == "global":
        application["applied"] = True
    elif target_scope == "next_stage" and isinstance(target_stage, str):
        application["applied"] = True
    elif target_scope == "next_round":
        next_round_id = (
            int(target_next_round_id)
            if isinstance(target_next_round_id, int) and target_next_round_id > 0
            else 1
        )
        application["target_round_id"] = next_round_id
        application["applied"] = True
    return application


def derive_workflow_patch(
    *,
    critique_payload: Dict[str, Any],
    base_workflow_spec: Optional[Dict[str, Any]] = None,
    target_next_round_id: Optional[int] = None,
) -> Dict[str, Any]:
    parsed = (
        ((critique_payload.get("branch_critic_result") or {}).get("response") or {}).get("parsed")
        or {}
    )
    final_recommendation = parsed.get("final_recommendation") or {}
    recommended_strategy = final_recommendation.get("recommended_strategy")
    recommended_next_field = final_recommendation.get("recommended_next_field")
    recommended_global_value = final_recommendation.get("recommended_global_value")
    recommended_candidate_values = final_recommendation.get("recommended_candidate_values")
    recommended_values_by_stage = final_recommendation.get("recommended_values_by_stage") or {}
    recommended_fields_by_stage = final_recommendation.get("recommended_fields_by_stage") or {}
    recommended_design_transition = (
        final_recommendation.get("recommended_design_transition") or {}
    )
    design_transition_application = build_design_transition_application(
        recommended_design_transition=recommended_design_transition,
        target_next_round_id=target_next_round_id,
    )

    patch = {
        "branch_plan_id": critique_payload.get("branch_plan_summary", {}).get("branch_plan_id"),
        "analysis_summary": parsed.get("analysis_summary"),
        "structured_diagnosis": parsed.get("structured_diagnosis"),
        "recommended_strategy": recommended_strategy,
        "recommended_next_field": recommended_next_field,
        "recommended_global_value": recommended_global_value,
        "recommended_candidate_values": recommended_candidate_values,
        "recommended_values_by_stage": recommended_values_by_stage,
        "recommended_fields_by_stage": recommended_fields_by_stage,
        "recommended_design_transition": recommended_design_transition,
        "design_transition_application": design_transition_application,
        "recommended_followup": parsed.get("recommended_followup"),
        "rationale": final_recommendation.get("rationale"),
    }

    updated_workflow_spec = apply_branch_critic_patch_to_workflow_spec(
        workflow_spec=base_workflow_spec,
        workflow_patch=patch,
        final_recommendation=final_recommendation,
        target_next_round_id=target_next_round_id,
    )
    return {
        "patch": patch,
        "updated_workflow_spec": updated_workflow_spec,
    }


def apply_branch_critic_patch_to_workflow_spec(
    *,
    workflow_spec: Optional[Dict[str, Any]],
    workflow_patch: Dict[str, Any],
    final_recommendation: Optional[Dict[str, Any]] = None,
    target_next_round_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    updated_workflow_spec = deepcopy_workflow_spec(workflow_spec)
    if updated_workflow_spec is None:
        return None

    tuning = ensure_nested_dict(updated_workflow_spec, "pbrs_tuning")
    base_pbrs = ensure_nested_dict(tuning, "base_pbrs")
    candidate_values = ensure_nested_dict(tuning, "candidate_values")
    per_stage_candidate_values = ensure_nested_dict(tuning, "per_stage_candidate_values")
    per_checkpoint_candidate_values = ensure_nested_dict(tuning, "per_checkpoint_candidate_values")
    per_stage_active_fields = ensure_nested_dict(tuning, "per_stage_active_fields")

    recommendation = final_recommendation or {
        "recommended_strategy": workflow_patch.get("recommended_strategy"),
        "recommended_next_field": workflow_patch.get("recommended_next_field"),
        "recommended_global_value": workflow_patch.get("recommended_global_value"),
        "recommended_candidate_values": workflow_patch.get("recommended_candidate_values"),
        "recommended_values_by_stage": workflow_patch.get("recommended_values_by_stage") or {},
        "recommended_fields_by_stage": workflow_patch.get("recommended_fields_by_stage") or {},
        "recommended_design_transition": workflow_patch.get("recommended_design_transition") or {},
        "design_transition_application": workflow_patch.get("design_transition_application")
        or {},
        "rationale": workflow_patch.get("rationale"),
    }
    updated_workflow_spec["branch_critic_recommendation"] = deepcopy(recommendation)
    updated_workflow_spec["branch_design_transition_application"] = deepcopy(
        workflow_patch.get("design_transition_application") or {}
    )

    recommended_strategy = workflow_patch.get("recommended_strategy")
    recommended_next_field = workflow_patch.get("recommended_next_field")
    recommended_global_value = workflow_patch.get("recommended_global_value")
    recommended_candidate_values = workflow_patch.get("recommended_candidate_values")
    recommended_values_by_stage = workflow_patch.get("recommended_values_by_stage") or {}
    recommended_fields_by_stage = workflow_patch.get("recommended_fields_by_stage") or {}
    recommended_design_transition = workflow_patch.get("recommended_design_transition") or {}
    design_transition_application = workflow_patch.get("design_transition_application") or {}

    if (
        isinstance(recommended_next_field, str)
        and isinstance(recommended_candidate_values, list)
        and recommended_candidate_values
    ):
        candidate_values[recommended_next_field] = list(recommended_candidate_values)

    if (
        recommended_strategy == "global_value"
        and isinstance(recommended_next_field, str)
        and recommended_global_value is not None
    ):
        base_pbrs[recommended_next_field] = float(recommended_global_value)
        if not (isinstance(recommended_candidate_values, list) and recommended_candidate_values):
            candidate_values[recommended_next_field] = [float(recommended_global_value)]

    if (
        recommended_strategy == "stage_specific"
        and isinstance(recommended_values_by_stage, dict)
    ):
        for key, value in recommended_values_by_stage.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            stage_field = (
                str(recommended_fields_by_stage.get(key))
                if isinstance(recommended_fields_by_stage, dict)
                and isinstance(recommended_fields_by_stage.get(key), str)
                else (
                    str(recommended_next_field)
                    if isinstance(recommended_next_field, str)
                    else None
                )
            )
            if not isinstance(stage_field, str) or not stage_field:
                continue
            if is_stage_key(str(key)):
                stage_patch = ensure_nested_dict(per_stage_candidate_values, str(key))
                stage_patch[stage_field] = [numeric_value]
                per_stage_active_fields[str(key)] = stage_field
            else:
                checkpoint_patch = ensure_nested_dict(per_checkpoint_candidate_values, str(key))
                checkpoint_patch[stage_field] = [numeric_value]

    design_policy = ensure_nested_dict(tuning, "design_update_policy")
    if isinstance(recommended_design_transition, dict):
        transition_action = recommended_design_transition.get("transition_action")
        target_scope = recommended_design_transition.get("target_scope")
        target_stage = recommended_design_transition.get("target_stage")
        policy_patch = (
            design_transition_application.get("policy_patch")
            if isinstance(design_transition_application, dict)
            else None
        )
        if not isinstance(policy_patch, dict):
            policy_patch = build_design_transition_application(
                recommended_design_transition=recommended_design_transition,
                target_next_round_id=target_next_round_id,
            ).get("policy_patch", {})

        if target_scope == "global":
            design_policy.update(policy_patch)
        elif target_scope == "next_stage" and isinstance(target_stage, str):
            per_stage_policy = ensure_nested_dict(design_policy, "per_stage_policy")
            stage_policy = ensure_nested_dict(per_stage_policy, target_stage)
            stage_policy.update(policy_patch)
        elif target_scope == "next_round":
            per_round_policy = ensure_nested_dict(design_policy, "per_round_policy")
            next_round_id = 1
            if isinstance(target_next_round_id, int) and target_next_round_id > 0:
                next_round_id = int(target_next_round_id)
            per_round_policy[str(next_round_id)] = policy_patch
    return updated_workflow_spec


def infer_target_next_round_id_from_workflow_manifest(
    manifest: Optional[Dict[str, Any]],
) -> Optional[int]:
    if not isinstance(manifest, dict):
        return None
    current_round = manifest.get("current_round")
    try:
        current_round_value = int(current_round)
    except (TypeError, ValueError):
        current_round_value = None

    if current_round_value is not None and current_round_value > 0:
        return current_round_value + 1

    completed_rounds = manifest.get("completed_rounds")
    try:
        completed_rounds_value = int(completed_rounds)
    except (TypeError, ValueError):
        completed_rounds_value = None

    if completed_rounds_value is not None and completed_rounds_value >= 0:
        return completed_rounds_value + 1
    return 1


def load_attached_branch_critic_patch(
    *,
    workflow_id: str,
    storage_root: str | Path,
) -> Optional[Dict[str, Any]]:
    workflow_dir = Path(storage_root) / workflow_id
    patch_path = workflow_dir / "branch_critic_patch.json"
    if patch_path.exists():
        with patch_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        workflow_patch = payload.get("workflow_patch")
        if isinstance(workflow_patch, dict):
            if "design_transition_application" not in workflow_patch:
                workflow_patch["design_transition_application"] = build_design_transition_application(
                    recommended_design_transition=workflow_patch.get(
                        "recommended_design_transition"
                    )
                )
            return workflow_patch
    manifest_path = workflow_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        workflow_patch = manifest.get("branch_critic_patch")
        if isinstance(workflow_patch, dict):
            if "design_transition_application" not in workflow_patch:
                workflow_patch["design_transition_application"] = build_design_transition_application(
                    recommended_design_transition=workflow_patch.get(
                        "recommended_design_transition"
                    )
                )
            return workflow_patch
    return None


def attach_branch_critic_payload_to_workflow(
    *,
    workflow_id: str,
    storage_root: str | Path,
    payload: Dict[str, Any],
    updated_workflow_spec: Optional[Dict[str, Any]],
) -> None:
    storage = WorkflowStorage.load(workflow_id, root_dir=storage_root)
    storage.save_workflow_json("branch_critic_patch.json", payload)
    manifest = storage.load_manifest()
    if isinstance(updated_workflow_spec, dict):
        manifest["workflow_spec"] = updated_workflow_spec
    manifest["branch_critic_patch"] = payload.get("workflow_patch")
    storage.save_manifest(manifest)


def load_workflow_patch_from_branch_plan(
    *,
    branch_plan_id: str,
    branch_results_root: str | Path = DEFAULT_BRANCH_PLAN_ROOT,
) -> Optional[Dict[str, Any]]:
    branch_plan_dir = Path(branch_results_root) / branch_plan_id
    workflow_patch_path = branch_plan_dir / "workflow_patch.json"
    if not workflow_patch_path.exists():
        return None
    with workflow_patch_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    workflow_patch = payload.get("workflow_patch")
    if isinstance(workflow_patch, dict):
        if "design_transition_application" not in workflow_patch:
            workflow_patch["design_transition_application"] = build_design_transition_application(
                recommended_design_transition=workflow_patch.get(
                    "recommended_design_transition"
                )
            )
        return workflow_patch
    return None


def load_branch_plan_manifest(
    *,
    branch_plan_id: str,
    branch_results_root: str | Path = DEFAULT_BRANCH_PLAN_ROOT,
) -> Optional[Dict[str, Any]]:
    branch_plan_dir = Path(branch_results_root) / branch_plan_id
    manifest_path = branch_plan_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest if isinstance(manifest, dict) else None
