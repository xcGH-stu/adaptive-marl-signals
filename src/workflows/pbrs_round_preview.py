from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from scripts.inspect_reward_workflow_state import DEFAULT_RESULTS_ROOT, inspect_workflow
from scripts.run_reward_workflow import build_workflow_spec_from_args
from workflows.reward_workflow import RewardWorkflow
from workflows.storage import WorkflowStorage


def fmt_preview_value(value: Any) -> str:
    return "-" if value is None else str(value)


def build_read_only_workflow(args: argparse.Namespace) -> RewardWorkflow:
    workflow_spec = build_workflow_spec_from_args(args)
    storage = WorkflowStorage(
        workflow_id=args.workflow_id,
        root_dir=args.storage_root,
    )
    workflow = object.__new__(RewardWorkflow)
    workflow.workflow_id = args.workflow_id
    workflow.workflow_spec = workflow_spec
    workflow.generator_client = None
    workflow.critic_client = None
    workflow.train_launcher = None
    workflow.storage_root = args.storage_root
    workflow.storage = storage
    return workflow


def workflow_dir(storage_root: Optional[str], workflow_id: str) -> Path:
    root = Path(storage_root) if storage_root is not None else DEFAULT_RESULTS_ROOT
    return root / workflow_id


def determine_preview_round(
    args: argparse.Namespace,
    workflow: RewardWorkflow,
) -> int:
    preview_round = getattr(args, "preview_round", None)
    if preview_round is not None:
        return int(preview_round)

    resolved_workflow_dir = workflow_dir(args.storage_root, args.workflow_id)
    if resolved_workflow_dir.exists():
        report = inspect_workflow(resolved_workflow_dir)
        resume_plan = report.get("resume_plan") or {}
        suggested_start_round = resume_plan.get("suggested_start_round")
        if suggested_start_round is not None:
            return int(suggested_start_round)
        current_round = report.get("current_round")
        if current_round is not None:
            return int(current_round)

    return 1


def load_previous_round(
    workflow: RewardWorkflow,
    round_id: int,
) -> Optional[Dict[str, Any]]:
    if not workflow.storage.workflow_dir.exists():
        return None
    return workflow.storage.get_previous_round_artifacts(round_id)


def ensure_previous_round(
    previous_round: Optional[Dict[str, Any]],
    round_id: int,
) -> Dict[str, Any]:
    if isinstance(previous_round, dict):
        return previous_round
    return {
        "round_id": int(round_id) - 1,
        "status": {},
        "critic_response": {"parsed": {}},
    }


def set_nested_search_strategy(
    previous_round: Dict[str, Any],
    recommendation: Dict[str, Any],
) -> None:
    critic_response = previous_round.setdefault("critic_response", {})
    if not isinstance(critic_response, dict):
        critic_response = {}
        previous_round["critic_response"] = critic_response
    parsed = critic_response.setdefault("parsed", {})
    if not isinstance(parsed, dict):
        parsed = {}
        critic_response["parsed"] = parsed
    parsed["search_strategy_recommendation"] = recommendation


def set_previous_active_field(previous_round: Dict[str, Any], field_name: str) -> None:
    status = previous_round.setdefault("status", {})
    if not isinstance(status, dict):
        status = {}
        previous_round["status"] = status
    status["active_pbrs_field"] = field_name


def load_round_state_from_storage(
    storage: WorkflowStorage,
    round_id: int,
) -> Optional[Dict[str, Any]]:
    round_dir = storage.get_round_dir(round_id)
    if not round_dir.exists():
        return None

    status = None
    status_path = storage.get_round_paths(round_id).status
    if status_path.exists():
        status = storage.load_round_status(round_id)
    if not isinstance(status, dict):
        status = {"round_id": int(round_id), "status": "missing"}

    candidate_manifest = None
    candidate_manifest_path = storage.get_round_paths(round_id).candidate_manifest
    if candidate_manifest_path.exists():
        candidate_manifest = storage.load_json(round_id, candidate_manifest_path.name)
    if not isinstance(candidate_manifest, dict):
        candidate_manifest = {}

    candidate_metadata = candidate_manifest.get("metadata") or {}
    candidates = candidate_manifest.get("candidates")
    first_candidate = None
    if isinstance(candidates, list):
        first_candidate = next(
            (item for item in candidates if isinstance(item, dict)),
            None,
        )

    active_pbrs_field = status.get("active_pbrs_field")
    active_pbrs_field_source = status.get("active_pbrs_field_source")
    if active_pbrs_field is None:
        active_pbrs_field = candidate_metadata.get("active_pbrs_field")
    if active_pbrs_field_source is None:
        active_pbrs_field_source = candidate_metadata.get("active_pbrs_field_source")
    if active_pbrs_field is None and isinstance(first_candidate, dict):
        active_pbrs_field = first_candidate.get("active_pbrs_field")
    if active_pbrs_field_source is None and isinstance(first_candidate, dict):
        active_pbrs_field_source = first_candidate.get("active_pbrs_field_source")

    active_checkpoint_context = status.get("active_checkpoint_context")
    if active_checkpoint_context is None:
        active_checkpoint_context = candidate_metadata.get("active_checkpoint_context")

    active_field_carryover_context = status.get("active_field_carryover_context")
    if active_field_carryover_context is None:
        active_field_carryover_context = candidate_metadata.get("active_field_carryover_context")

    candidate_selection_context = status.get("candidate_selection_context")
    if candidate_selection_context is None:
        candidate_selection_context = candidate_metadata.get("candidate_selection_context")

    return {
        "round_id": int(round_id),
        "status": status.get("status"),
        "stage": status.get("stage"),
        "reason": status.get("reason"),
        "active_pbrs_field": active_pbrs_field,
        "active_pbrs_field_source": active_pbrs_field_source,
        "active_checkpoint_context": active_checkpoint_context,
        "active_field_carryover_context": active_field_carryover_context,
        "candidate_selection_context": candidate_selection_context,
        "raw_status": status,
        "candidate_manifest": candidate_manifest,
    }


def build_candidate_previews(
    workflow: RewardWorkflow,
    *,
    round_id: int,
    active_pbrs_field: Optional[str],
    candidate_selection_context: Dict[str, Any],
) -> list[Dict[str, Any]]:
    candidate_values = list(candidate_selection_context.get("effective_candidate_values", []))
    if active_pbrs_field is None:
        return [{"candidate_id": "candidate_01", "candidate_value": None}]
    return [
        {
            "candidate_id": workflow._format_candidate_id(
                index=index,
                field_name=active_pbrs_field,
                candidate_value=candidate_value,
            ),
            "candidate_value": candidate_value,
            "round_id": int(round_id),
            "active_pbrs_field": active_pbrs_field,
        }
        for index, candidate_value in enumerate(candidate_values, start=1)
    ]


def resolve_round_preview_state(
    workflow: RewardWorkflow,
    *,
    round_id: int,
    previous_round: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    active_checkpoint_context = workflow._get_active_pbrs_checkpoint_context(round_id)
    active_pbrs_field, active_pbrs_field_source = workflow._resolve_active_pbrs_field(
        round_id=round_id,
        active_checkpoint_context=active_checkpoint_context,
        previous_round=previous_round,
    )
    active_field_carryover_context = workflow._build_active_field_carryover_context(
        active_pbrs_field=active_pbrs_field,
        active_pbrs_field_source=active_pbrs_field_source,
        previous_round=previous_round,
    )
    candidate_selection_context = workflow._resolve_candidate_selection_context(
        round_id=round_id,
        active_pbrs_field=active_pbrs_field,
        active_checkpoint_context=active_checkpoint_context,
        previous_round=previous_round,
    )
    search_strategy_context_summary = workflow._build_search_strategy_context_summary(
        round_id=round_id,
        active_pbrs_field=active_pbrs_field,
        active_pbrs_field_source=active_pbrs_field_source,
        active_field_carryover_context=active_field_carryover_context,
        candidate_selection_context=candidate_selection_context,
        active_checkpoint_context=active_checkpoint_context,
        previous_round=previous_round,
    )
    validation_config = workflow._get_effective_pbrs_validation_config(round_id)
    candidate_previews = build_candidate_previews(
        workflow,
        round_id=round_id,
        active_pbrs_field=active_pbrs_field,
        candidate_selection_context=candidate_selection_context,
    )
    return {
        "round_id": int(round_id),
        "reward_paradigm": workflow._get_reward_paradigm(),
        "active_pbrs_field": active_pbrs_field,
        "active_pbrs_field_source": active_pbrs_field_source,
        "active_checkpoint_context": active_checkpoint_context,
        "active_field_carryover_context": active_field_carryover_context,
        "candidate_selection_context": candidate_selection_context,
        "search_strategy_context_summary": search_strategy_context_summary,
        "validation_config": validation_config,
        "candidate_previews": candidate_previews,
    }


def namespace_for_existing_workflow(
    *,
    workflow_id: str,
    results_root: str,
    preview_round: Optional[int] = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        workflow_id=workflow_id,
        max_rounds=None,
        start_round=1,
        reward_paradigm=None,
        python_executable=None,
        storage_root=results_root,
        use_real_llm=False,
        api_key_env="IUSEAPI_API_KEY",
        base_url=None,
        model=None,
        temperature=0.2,
        llm_timeout=60.0,
        llm_max_retries=3,
        llm_retry_backoff=5.0,
        t_max=None,
        test_interval=None,
        runner_log_interval=None,
        learner_log_interval=None,
        env_key=None,
        train_config=None,
        time_limit=None,
        use_existing_workflow_spec=True,
        preview_round=preview_round,
    )
