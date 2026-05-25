from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.plan_pbrs_baseline_branching import (
    _align_selected_checkpoints_to_available_steps,
)
from scripts.run_reward_workflow import build_default_workflow_spec
from scripts.select_pbrs_checkpoints import build_baseline_summary
from workflows.baseline_run_support import build_baseline_checkpoint_candidates
from workflows.clients.checkpoint_selector_client import (
    TemplateBasedCheckpointSelectorClient,
)
from workflows.clients.openai_backend import OpenAIChatBackend


DEFAULT_PREFERRED_METRICS = [
    "test_sparse_return_mean",
    "sparse_return_mean",
    "test_return_mean",
    "return_mean",
]
STAGE_LABELS = ("early_exploration", "mid_progress", "late_plateau")
PROGRESS_METRIC_PRIORITY = [
    "test_sparse_return_mean",
    "test_return_mean",
    "sparse_return_mean",
    "return_mean",
]
EP_LENGTH_METRIC_PRIORITY = [
    "test_ep_length_mean",
    "ep_length_mean",
]


class StageSelectionError(ValueError):
    def __init__(self, message: str, *, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.payload = payload or {}


def build_fixed_intervention_stage_selection_result(
    *,
    workflow_id: str,
    checkpoint_steps: List[int],
    objective_metric: str = "test_sparse_return_mean",
) -> Dict[str, Any]:
    fixed_plan = [
        {
            "name": "C1",
            "step": int(checkpoint_steps[0]),
            "stage_label": "post_initialization_transition",
            "reason": (
                "Fixed intervention checkpoint after the selected 800k dense initialization phase "
                "to test whether the early-selected PBRS configuration still matches the active learning regime."
            ),
            "recommended_focus": "beta_down,beta_up,wc_down_wp_up,reference_like,recovery",
            "intervention_question": (
                "Does the early-selected dense reward configuration remain suitable after the initial 800k dense search phase?"
            ),
            "recommended_candidate_focus": [
                "beta_down",
                "beta_up",
                "wc_down_wp_up",
                "reference_like",
                "recovery",
            ],
            "branch_budget_feasible": True,
            "adaptation_window_quality": "fixed",
        },
        {
            "name": "C2",
            "step": int(checkpoint_steps[1]),
            "stage_label": "late_stability",
            "reason": (
                "Fixed late intervention checkpoint to test whether the current PBRS configuration remains stable and task-aligned."
            ),
            "recommended_focus": "beta_down,wc_down_wp_up,reference_like,recovery",
            "intervention_question": (
                "Does the current reward configuration remain stable and task-aligned in the later training phase?"
            ),
            "recommended_candidate_focus": [
                "beta_down",
                "wc_down_wp_up",
                "reference_like",
                "recovery",
            ],
            "branch_budget_feasible": True,
            "adaptation_window_quality": "fixed",
        },
    ]
    return {
        "workflow_id": workflow_id,
        "reward_paradigm": "pbrs",
        "selection_mode": "fixed_intervention_schedule",
        "selection_constraints": {
            "checkpoint_count": len(fixed_plan),
            "fixed_intervention_checkpoints": list(checkpoint_steps),
        },
        "preferred_metrics": [objective_metric],
        "baseline_summary": {
            "selection_mode": "fixed_intervention_schedule",
            "objective_metric": objective_metric,
        },
        "checkpoint_candidates": {
            "checkpoint_candidates": [
                {
                    "checkpoint_step": item["step"],
                    "stage_label": item["stage_label"],
                    "intervention_question": item["intervention_question"],
                    "recommended_candidate_focus": item["recommended_candidate_focus"],
                }
                for item in fixed_plan
            ]
        },
        "selection": {
            "analysis_summary": "Deterministic fixed checkpoint schedule for the dual-LLM adaptive workflow profile.",
            "selected_checkpoints": fixed_plan,
            "meets_stage_requirements": True,
            "requirement_notes": "Fixed checkpoints intentionally replace LLM checkpoint selection for this profile.",
        },
        "selected_checkpoint_evidence": [],
        "semantic_validation": {
            "checks": {
                "fixed_schedule": True,
            },
            "meets_semantic_requirements": True,
            "failure_reason": None,
            "progress_metric": objective_metric,
        },
        "response": {
            "model": "fixed-schedule",
            "parsed": {
                "selected_checkpoints": fixed_plan,
            },
            "raw_text": json.dumps({"selected_checkpoints": fixed_plan}, ensure_ascii=False),
            "raw_response": {"fixed_schedule": True},
            "usage": None,
        },
        "baseline_checkpoint_root_dir": None,
        "baseline_available_checkpoint_steps": list(checkpoint_steps),
        "baseline_run_config_json": None,
        "baseline_run_info_json": None,
        "baseline_metrics_json": None,
    }


def build_stage_selection_result(
    *,
    workflow_id: str,
    baseline_metrics_json: str,
    baseline_run_config_json: str,
    baseline_run_info_json: str,
    reward_paradigm: str = "pbrs",
    max_rounds: int = 3,
    min_count: int = 3,
    max_count: int = 3,
    min_step_gap: int = 50000,
    max_points_per_metric: int = 15,
    preferred_metrics: Optional[List[str]] = None,
    checkpoint_optional_metrics: Optional[List[str]] = None,
    use_real_llm: bool = False,
    api_key_env: str = "IUSEAPI_API_KEY",
    base_url: str = "https://www.iuseapi.com/v1",
    model: str = "gpt-5.5",
    temperature: float = 0.2,
    llm_timeout: float = 60.0,
    llm_max_retries: int = 3,
    llm_retry_backoff: float = 5.0,
) -> Dict[str, Any]:
    if not use_real_llm:
        raise ValueError(
            "Stage-conditioned checkpoint selection must use real LLM selection. "
            "Heuristic fallback is disabled for the new workflow path."
        )

    metrics_path = Path(baseline_metrics_json)
    run_config_path = Path(baseline_run_config_json)
    run_info_path = Path(baseline_run_info_json)
    metrics = preferred_metrics or DEFAULT_PREFERRED_METRICS

    baseline_summary = build_baseline_summary(
        metrics_json_path=metrics_path,
        run_config_path=run_config_path,
        preferred_metrics=metrics,
        max_points_per_metric=max_points_per_metric,
    )

    workflow_spec = build_default_workflow_spec(max_rounds, reward_paradigm=reward_paradigm)
    workflow_spec["workflow_id"] = workflow_id

    selection_constraints = {
        "min_count": int(min_count),
        "max_count": int(max_count),
        "min_step_gap": int(min_step_gap),
    }

    baseline_run_config_payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    baseline_run_info_payload = json.loads(run_info_path.read_text(encoding="utf-8"))
    checkpoint_candidates = build_baseline_checkpoint_candidates(
        baseline_run_dir=str(run_config_path.parent),
        baseline_result_json=None,
        optional_metric_names=checkpoint_optional_metrics,
    )
    available_steps = sorted(
        {
            int(item.get("checkpoint_step"))
            for item in (checkpoint_candidates.get("checkpoint_candidates") or [])
            if item.get("checkpoint_step") is not None
        }
    )
    checkpoint_root_dir = baseline_run_info_payload.get("model_root_path")

    enriched_baseline_summary = deepcopy(baseline_summary)
    enriched_baseline_summary["available_checkpoint_steps"] = list(available_steps)
    enriched_baseline_summary["train_budget"] = {
        "t_max": int(baseline_run_config_payload.get("t_max", 0) or 0),
    }

    llm_backend = OpenAIChatBackend(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=llm_timeout,
        max_retries=llm_max_retries,
        retry_backoff_seconds=llm_retry_backoff,
    )
    selector = TemplateBasedCheckpointSelectorClient(llm_backend=llm_backend)
    llm_result = selector.select_checkpoints(
        workflow_spec=workflow_spec,
        baseline_summary=enriched_baseline_summary,
        selection_constraints=selection_constraints,
        checkpoint_candidates=checkpoint_candidates,
    )
    llm_selected = list((llm_result.get("selection") or {}).get("selected_checkpoints", []))
    aligned_llm_selected = _align_selected_checkpoints_to_available_steps(
        selected_checkpoints=llm_selected,
        available_steps=available_steps,
    )
    if not _aligned_selection_is_valid(aligned_llm_selected):
        raise StageSelectionError(
            "LLM checkpoint selection is invalid for the stage-conditioned workflow. "
            "The selector must produce exactly three ordered checkpoints with "
            "stage labels early_exploration, mid_progress, late_plateau.",
            payload={
                "baseline_summary": enriched_baseline_summary,
                "checkpoint_candidates": checkpoint_candidates,
                "selection_constraints": selection_constraints,
                "response": llm_result.get("response"),
                "selection": llm_result.get("selection"),
            },
        )
    llm_selection_payload = deepcopy(llm_result.get("selection") or {})
    if not bool(llm_selection_payload.get("meets_stage_requirements", True)):
        raise StageSelectionError(
            "LLM checkpoint selection explicitly reports that the baseline does not "
            "satisfy the required early/mid/late stage semantics: "
            + str(llm_selection_payload.get("requirement_notes") or "no details provided"),
            payload={
                "baseline_summary": enriched_baseline_summary,
                "checkpoint_candidates": checkpoint_candidates,
                "selection_constraints": selection_constraints,
                "response": llm_result.get("response"),
                "selection": llm_selection_payload,
            },
        )
    selection_result = {
        "selection": {
            **llm_selection_payload,
            "selected_checkpoints": aligned_llm_selected,
        },
        "response": llm_result.get("response"),
    }
    selection_mode = "real_llm"

    selection_payload = deepcopy(selection_result.get("selection") or {})
    selected_checkpoint_evidence = _attach_checkpoint_evidence(
        selected_checkpoints=aligned_llm_selected,
        checkpoint_candidates=checkpoint_candidates,
    )
    semantic_validation = _evaluate_stage_semantics(
        selected_checkpoints=aligned_llm_selected,
        selected_checkpoint_evidence=selected_checkpoint_evidence,
        preferred_metrics=metrics,
    )
    if not bool(semantic_validation.get("meets_semantic_requirements")):
        raise StageSelectionError(
            "LLM-selected checkpoints do not satisfy strict Stage 2 semantics after metric validation: "
            + str(semantic_validation.get("failure_reason") or "unknown_reason"),
            payload={
                "baseline_summary": enriched_baseline_summary,
                "checkpoint_candidates": checkpoint_candidates,
                "selection_constraints": selection_constraints,
                "response": llm_result.get("response"),
                "selection": selection_payload,
                "selected_checkpoint_evidence": selected_checkpoint_evidence,
                "semantic_validation": semantic_validation,
            },
        )

    return {
        "workflow_id": workflow_id,
        "reward_paradigm": reward_paradigm,
        "selection_mode": selection_mode,
        "selection_constraints": selection_constraints,
        "preferred_metrics": metrics,
        "baseline_summary": enriched_baseline_summary,
        "checkpoint_candidates": checkpoint_candidates,
        "selection": selection_payload,
        "selected_checkpoint_evidence": selected_checkpoint_evidence,
        "semantic_validation": semantic_validation,
        "response": selection_result.get("response"),
        "baseline_checkpoint_root_dir": checkpoint_root_dir,
        "baseline_available_checkpoint_steps": available_steps,
        "baseline_run_config_json": str(run_config_path),
        "baseline_run_info_json": str(run_info_path),
        "baseline_metrics_json": str(metrics_path),
    }


def build_dual_source_stage_selection_result(
    *,
    workflow_id: str,
    sparse_metrics_json: str,
    sparse_run_config_json: str,
    dense_metrics_json: str,
    dense_run_config_json: str,
    dense_run_info_json: str,
    reward_paradigm: str = "pbrs",
    max_rounds: int = 3,
    min_count: int = 3,
    max_count: int = 3,
    min_step_gap: int = 50000,
    branch_budget_steps: Optional[int] = None,
    max_points_per_metric: int = 15,
    preferred_metrics: Optional[List[str]] = None,
    checkpoint_optional_metrics: Optional[List[str]] = None,
    use_real_llm: bool = False,
    api_key_env: str = "IUSEAPI_API_KEY",
    base_url: str = "https://www.iuseapi.com/v1",
    model: str = "gpt-5.5",
    temperature: float = 0.2,
    llm_timeout: float = 60.0,
    llm_max_retries: int = 3,
    llm_retry_backoff: float = 5.0,
) -> Dict[str, Any]:
    if not use_real_llm:
        raise ValueError(
            "Formal checkpoint selection must use real LLM selection in the Phase 1 workflow."
        )

    metrics = preferred_metrics or DEFAULT_PREFERRED_METRICS
    sparse_metrics_path = Path(sparse_metrics_json)
    sparse_run_config_path = Path(sparse_run_config_json)
    dense_metrics_path = Path(dense_metrics_json)
    dense_run_config_path = Path(dense_run_config_json)
    dense_run_info_path = Path(dense_run_info_json)

    sparse_summary = build_baseline_summary(
        metrics_json_path=sparse_metrics_path,
        run_config_path=sparse_run_config_path,
        preferred_metrics=metrics,
        max_points_per_metric=max_points_per_metric,
    )
    dense_summary = build_baseline_summary(
        metrics_json_path=dense_metrics_path,
        run_config_path=dense_run_config_path,
        preferred_metrics=metrics,
        max_points_per_metric=max_points_per_metric,
    )

    workflow_spec = build_default_workflow_spec(max_rounds, reward_paradigm=reward_paradigm)
    workflow_spec["workflow_id"] = workflow_id
    workflow_spec["task_description"] = (
        (workflow_spec.get("task_description") or "")
        + "\nThis is a formal checkpoint selection step for phase-conditioned PBRS validation."
    ).strip()
    selection_constraints = {
        "min_count": int(min_count),
        "max_count": int(max_count),
        "min_step_gap": int(min_step_gap),
    }
    if branch_budget_steps is not None:
        selection_constraints["branch_budget_steps"] = int(branch_budget_steps)

    dense_run_config_payload = json.loads(dense_run_config_path.read_text(encoding="utf-8"))
    dense_run_info_payload = json.loads(dense_run_info_path.read_text(encoding="utf-8"))
    checkpoint_candidates = build_baseline_checkpoint_candidates(
        baseline_run_dir=str(dense_run_config_path.parent),
        baseline_result_json=None,
        optional_metric_names=checkpoint_optional_metrics,
    )
    available_steps = sorted(
        {
            int(item.get("checkpoint_step"))
            for item in (checkpoint_candidates.get("checkpoint_candidates") or [])
            if item.get("checkpoint_step") is not None
        }
    )
    if not available_steps:
        raise StageSelectionError(
            "dense reference trajectory does not expose any usable checkpoint candidate steps from "
            "saved_model_steps or metrics.json",
            payload={
                "dense_run_dir": str(dense_run_config_path.parent),
                "checkpoint_candidates": checkpoint_candidates,
            },
        )
    checkpoint_root_dir = dense_run_info_payload.get("model_root_path")
    backend = OpenAIChatBackend(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=llm_timeout,
        max_retries=llm_max_retries,
        retry_backoff_seconds=llm_retry_backoff,
    )
    selector = TemplateBasedCheckpointSelectorClient(llm_backend=backend)
    prompt_spec = selector.prompt_spec_path.read_text(encoding="utf-8")
    env_reference = selector.env_reference_path.read_text(encoding="utf-8")
    prompt = "\n".join(
        [
            "Prompt specification:",
            prompt_spec,
            "",
            "Task description:",
            workflow_spec.get("task_description", "No task description provided."),
            "",
            "Environment description:",
            workflow_spec.get("environment_description", "No environment description provided."),
            "",
            "LBF environment reference:",
            env_reference,
            "",
            "Checkpoint selection constraints:",
            json.dumps(selection_constraints, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Sparse reference trajectory summary (coarse phase reference only):",
            json.dumps(sparse_summary, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Dense reference trajectory summary (formal checkpoint source):",
            json.dumps(
                {
                    **deepcopy(dense_summary),
                    "available_checkpoint_steps": list(available_steps),
                    "train_budget": {
                        "t_max": int(dense_run_config_payload.get("t_max", 0) or 0),
                    },
                    "branch_validation_budget": {
                        "branch_budget_steps": (
                            int(branch_budget_steps) if branch_budget_steps is not None else None
                        ),
                        "selection_goal": (
                            "Choose checkpoints that are not only stage-representative but also "
                            "good starting points for a fixed-budget short-horizon branch validation run."
                        ),
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Checkpoint candidates from the dense reference run:",
            json.dumps(
                checkpoint_candidates or {"checkpoint_candidates": []},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Select exactly three formal checkpoints from the dense reference run. "
            "Use the sparse trajectory only as auxiliary context for coarse training stages. "
            "The returned checkpoints must come from the dense reference candidate set. "
            "When selecting mid_progress, avoid choosing a brittle local spike if a nearby checkpoint better represents the transition regime, and prefer a nearby pullback/trough point if it gives a more meaningful branch-validation start. "
            "When selecting late_plateau, prefer a checkpoint that still leaves enough remaining training room for the fixed-budget branch validation run, and avoid picking the single local peak if a nearby non-peak point would make continuation comparisons more informative.",
        ]
    )
    llm_result = backend.generate_text(
        system_prompt=(
            "You select three formal training checkpoints for phase-conditioned PBRS validation. "
            "Use sparse results only as coarse context, but choose checkpoints strictly from the "
            "dense-reference run and return strictly valid JSON. "
            "Optimize for later branch-validation usefulness, not just semantic phase labels or peak metric values."
        ),
        user_prompt=prompt,
        metadata={
            "workflow_id": workflow_id,
            "role": "dual_source_checkpoint_selector",
        },
    )
    parsed = selector.parse_response(
        llm_result["text"],
        selection_constraints=selection_constraints,
        checkpoint_candidates=checkpoint_candidates,
    )
    llm_selected = list((parsed or {}).get("selected_checkpoints", []))
    aligned_llm_selected = _align_selected_checkpoints_to_available_steps(
        selected_checkpoints=llm_selected,
        available_steps=available_steps,
    )
    if not _aligned_selection_is_valid(aligned_llm_selected):
        raise StageSelectionError(
            "Formal checkpoint selection is invalid. The selector must produce exactly three ordered dense checkpoints "
            "with stage labels early_exploration, mid_progress, late_plateau.",
            payload={
                "sparse_summary": sparse_summary,
                "dense_summary": dense_summary,
                "checkpoint_candidates": checkpoint_candidates,
                "selection_constraints": selection_constraints,
                "response": llm_result,
                "selection": parsed,
            },
        )
    if not bool(parsed.get("meets_stage_requirements", True)):
        raise StageSelectionError(
            "LLM reported that the dense reference trajectory does not satisfy the required formal checkpoint semantics: "
            + str(parsed.get("requirement_notes") or "no details provided"),
            payload={
                "sparse_summary": sparse_summary,
                "dense_summary": dense_summary,
                "checkpoint_candidates": checkpoint_candidates,
                "selection_constraints": selection_constraints,
                "response": llm_result,
                "selection": parsed,
            },
        )

    selection_payload = {
        **deepcopy(parsed),
        "selected_checkpoints": aligned_llm_selected,
    }
    selected_checkpoint_evidence = _attach_checkpoint_evidence(
        selected_checkpoints=aligned_llm_selected,
        checkpoint_candidates=checkpoint_candidates,
    )
    semantic_validation = _evaluate_stage_semantics(
        selected_checkpoints=aligned_llm_selected,
        selected_checkpoint_evidence=selected_checkpoint_evidence,
        preferred_metrics=metrics,
    )
    if not bool(semantic_validation.get("meets_semantic_requirements")):
        raise StageSelectionError(
            "Formal dense checkpoint selection does not satisfy strict Stage 2 semantics after metric validation: "
            + str(semantic_validation.get("failure_reason") or "unknown_reason"),
            payload={
                "sparse_summary": sparse_summary,
                "dense_summary": dense_summary,
                "checkpoint_candidates": checkpoint_candidates,
                "selection_constraints": selection_constraints,
                "selection": selection_payload,
                "selected_checkpoint_evidence": selected_checkpoint_evidence,
                "semantic_validation": semantic_validation,
            },
        )

    return {
        "workflow_id": workflow_id,
        "reward_paradigm": reward_paradigm,
        "selection_mode": "real_llm_post_dense_dual_source",
        "selection_source": "dense_reference_dual_source",
        "selection_constraints": selection_constraints,
        "preferred_metrics": metrics,
        "sparse_reference_summary": sparse_summary,
        "dense_reference_summary": {
            **deepcopy(dense_summary),
            "available_checkpoint_steps": list(available_steps),
        },
        "checkpoint_candidates": checkpoint_candidates,
        "selection": selection_payload,
        "selected_checkpoint_evidence": selected_checkpoint_evidence,
        "semantic_validation": semantic_validation,
        "response": {
            "raw_text": llm_result.get("text"),
            "raw_response": llm_result.get("raw_response"),
            "model": llm_result.get("model"),
            "usage": llm_result.get("usage"),
            "parsed": parsed,
        },
        "baseline_checkpoint_root_dir": checkpoint_root_dir,
        "baseline_available_checkpoint_steps": available_steps,
        "sparse_metrics_json": str(sparse_metrics_path),
        "sparse_run_config_json": str(sparse_run_config_path),
        "dense_run_config_json": str(dense_run_config_path),
        "dense_run_info_json": str(dense_run_info_path),
        "dense_metrics_json": str(dense_metrics_path),
    }


def _aligned_selection_is_valid(selected_checkpoints: List[Dict[str, Any]]) -> bool:
    if len(selected_checkpoints) != 3:
        return False
    seen_steps = []
    seen_stages = []
    for item in selected_checkpoints:
        try:
            seen_steps.append(int(item.get("step")))
        except (TypeError, ValueError):
            return False
        stage_label = item.get("stage_label")
        if stage_label not in STAGE_LABELS:
            return False
        seen_stages.append(stage_label)
    return len(set(seen_steps)) == 3 and tuple(seen_stages) == STAGE_LABELS and seen_steps == sorted(seen_steps)


def _attach_checkpoint_evidence(
    *,
    selected_checkpoints: List[Dict[str, Any]],
    checkpoint_candidates: Dict[str, Any],
) -> List[Dict[str, Any]]:
    candidates_by_step = {}
    for item in checkpoint_candidates.get("checkpoint_candidates") or []:
        try:
            candidates_by_step[int(item.get("checkpoint_step"))] = item
        except (TypeError, ValueError):
            continue
    enriched = []
    for checkpoint in selected_checkpoints:
        step = int(checkpoint["step"])
        candidate_payload = deepcopy(candidates_by_step.get(step) or {})
        enriched.append(
            {
                **deepcopy(checkpoint),
                "candidate_payload": candidate_payload,
                "metric_snapshot": deepcopy(candidate_payload.get("metric_snapshot") or {}),
            }
        )
    return enriched


def _metric_value(
    metric_snapshot: Dict[str, Any],
    metric_name: str,
    key: str,
) -> Optional[float]:
    payload = metric_snapshot.get(metric_name)
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _pick_progress_metric(
    selected_checkpoint_evidence: List[Dict[str, Any]],
    preferred_metrics: List[str],
) -> Optional[str]:
    priority = []
    for metric_name in preferred_metrics:
        if metric_name not in priority:
            priority.append(metric_name)
    for metric_name in PROGRESS_METRIC_PRIORITY:
        if metric_name not in priority:
            priority.append(metric_name)
    for metric_name in priority:
        values = [
            _metric_value(item.get("metric_snapshot") or {}, metric_name, "historical_best_value_to_checkpoint")
            for item in selected_checkpoint_evidence
        ]
        if all(value is not None for value in values):
            return metric_name
    return None


def _pick_ep_length_metric(
    selected_checkpoint_evidence: List[Dict[str, Any]],
) -> Optional[str]:
    for metric_name in EP_LENGTH_METRIC_PRIORITY:
        values = [
            _metric_value(item.get("metric_snapshot") or {}, metric_name, "latest_value_at_or_before_checkpoint")
            for item in selected_checkpoint_evidence
        ]
        if all(value is not None for value in values):
            return metric_name
    return None


def _evaluate_stage_semantics(
    *,
    selected_checkpoints: List[Dict[str, Any]],
    selected_checkpoint_evidence: List[Dict[str, Any]],
    preferred_metrics: List[str],
) -> Dict[str, Any]:
    progress_metric = _pick_progress_metric(selected_checkpoint_evidence, preferred_metrics)
    if progress_metric is None:
        return {
            "meets_semantic_requirements": False,
            "failure_reason": "no_common_progress_metric_available_for_selected_checkpoints",
            "progress_metric": None,
        }

    progress_history = [
        _metric_value(item.get("metric_snapshot") or {}, progress_metric, "historical_best_value_to_checkpoint")
        for item in selected_checkpoint_evidence
    ]
    progress_latest = [
        _metric_value(item.get("metric_snapshot") or {}, progress_metric, "latest_value_at_or_before_checkpoint")
        for item in selected_checkpoint_evidence
    ]
    early_best, mid_best, late_best = progress_history
    steps = [int(item.get("step")) for item in selected_checkpoints]

    checks = {
        "strict_step_order": steps[0] < steps[1] < steps[2],
        "strict_progress_order": early_best < mid_best < late_best,
        "early_has_low_signal_relative_to_late": (
            late_best > 0 and early_best <= 0.35 * late_best
        ),
        "mid_shows_meaningful_gain_over_early": (
            (mid_best - early_best) > max(0.05, 0.2 * max(late_best - early_best, 0.0))
        ),
        "late_is_distinctly_better_than_mid": (
            (late_best - mid_best) > max(0.03, 0.1 * max(late_best - early_best, 0.0))
        ),
    }

    ep_length_metric = _pick_ep_length_metric(selected_checkpoint_evidence)
    if ep_length_metric is not None:
        ep_lengths = [
            _metric_value(
                item.get("metric_snapshot") or {},
                ep_length_metric,
                "latest_value_at_or_before_checkpoint",
            )
            for item in selected_checkpoint_evidence
        ]
        if all(value is not None for value in ep_lengths):
            early_len, _, late_len = ep_lengths
            checks["late_episode_length_not_worse_than_early"] = late_len <= early_len
    else:
        ep_lengths = None

    meets = all(checks.values())
    failure_reason = None
    if not meets:
        failed = [name for name, ok in checks.items() if not ok]
        failure_reason = ", ".join(failed)

    return {
        "meets_semantic_requirements": meets,
        "failure_reason": failure_reason,
        "progress_metric": progress_metric,
        "progress_history_best_values": progress_history,
        "progress_latest_values": progress_latest,
        "episode_length_metric": ep_length_metric,
        "episode_length_latest_values": ep_lengths,
        "checks": checks,
    }
