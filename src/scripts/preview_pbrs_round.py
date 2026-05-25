from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.inspect_reward_workflow_state import DEFAULT_RESULTS_ROOT
from scripts.run_reward_workflow import build_arg_parser, build_workflow_spec_from_args
from workflows.clients.critic_client import StubCriticBackend, TemplateBasedCriticClient
from workflows.clients.generator_client import StubGeneratorBackend, TemplateBasedGeneratorClient
from workflows.pbrs_round_preview import (
    build_read_only_workflow as _build_read_only_workflow,
    determine_preview_round as _determine_preview_round,
    fmt_preview_value as _fmt,
    load_previous_round as _load_previous_round,
    resolve_round_preview_state,
)


def _build_preview_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Preview the next PBRS/heuristic workflow round without running training."
    parser.set_defaults(use_existing_workflow_spec=True)
    parser.set_defaults(apply_attached_branch_critic_patch=True)
    parser.add_argument(
        "--preview-round",
        type=int,
        default=None,
        help="Round index to preview. Defaults to the suggested resume round or 1.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Include Generator and Critic prompt previews.",
    )
    parser.add_argument(
        "--prompt-max-chars",
        type=int,
        default=2000,
        help="Maximum number of characters to print for each prompt in text mode.",
    )
    parser.add_argument(
        "--mock-checkpoint-stage-sequence",
        default=None,
        help=(
            "Optional comma-separated PBRS checkpoint stage sequence used only for preview, "
            "for example 'early_exploration,mid_progress,late_plateau'."
        ),
    )
    return parser


def _maybe_apply_mock_checkpoint_stage_sequence(
    workflow,
    stage_sequence: Optional[str],
) -> None:
    if not isinstance(stage_sequence, str) or not stage_sequence.strip():
        return
    stages = [item.strip() for item in stage_sequence.split(",") if item.strip()]
    if not stages:
        return
    checkpoint_plan = dict(workflow.workflow_spec.get("pbrs_checkpoint_plan") or {})
    selected_checkpoints = []
    round_to_checkpoint: Dict[str, str] = {}
    for index, stage_label in enumerate(stages, start=1):
        checkpoint_name = f"mock_{stage_label}_{index:02d}"
        selected_checkpoints.append(
            {
                "name": checkpoint_name,
                "step": index * 100000,
                "stage_label": stage_label,
                "reason": f"Preview-only mock checkpoint for {stage_label}.",
                "recommended_focus": stage_label,
            }
        )
        round_to_checkpoint[str(index)] = checkpoint_name
    checkpoint_plan["selected_checkpoints"] = selected_checkpoints
    checkpoint_plan["round_to_checkpoint"] = round_to_checkpoint
    workflow.workflow_spec["pbrs_checkpoint_plan"] = checkpoint_plan

def _build_preview_current_round(
    *,
    round_id: int,
    reward_paradigm: str,
    active_pbrs_field: Optional[str],
    active_pbrs_field_source: Optional[str],
    active_field_carryover_context: Dict[str, Any],
    candidate_selection_context: Dict[str, Any],
    active_checkpoint_context: Optional[Dict[str, Any]],
    effective_pbrs_design_update_policy: Dict[str, Any],
    branch_design_transition: Dict[str, Any],
    search_strategy_context_summary: Dict[str, Any],
    validation_config: Dict[str, Any],
    candidate_previews: list[Dict[str, Any]],
    base_reward_spec: Optional[Dict[str, Any]],
    previous_round: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "round_id": int(round_id),
        "preview_mode": True,
        "reward_paradigm": reward_paradigm,
        "active_pbrs_field": active_pbrs_field,
        "active_pbrs_field_source": active_pbrs_field_source,
        "active_field_carryover_context": active_field_carryover_context,
        "candidate_selection_context": candidate_selection_context,
        "active_checkpoint_context": active_checkpoint_context,
        "effective_pbrs_design_update_policy": effective_pbrs_design_update_policy,
        "branch_design_transition": branch_design_transition,
        "search_strategy_context_summary": search_strategy_context_summary,
        "validation_plan": validation_config,
        "candidate_previews": candidate_previews,
        "base_reward_spec": base_reward_spec,
        "previous_round_status": (previous_round or {}).get("status"),
    }


def build_preview_report(args: argparse.Namespace) -> Dict[str, Any]:
    workflow = _build_read_only_workflow(args)
    _maybe_apply_mock_checkpoint_stage_sequence(
        workflow,
        getattr(args, "mock_checkpoint_stage_sequence", None),
    )
    round_id = _determine_preview_round(args, workflow)
    previous_round = _load_previous_round(workflow, round_id)
    resolved = resolve_round_preview_state(
        workflow,
        round_id=round_id,
        previous_round=previous_round,
    )
    active_checkpoint_context = resolved["active_checkpoint_context"]
    active_pbrs_field = resolved["active_pbrs_field"]
    active_pbrs_field_source = resolved["active_pbrs_field_source"]
    active_field_carryover_context = resolved["active_field_carryover_context"]
    candidate_selection_context = resolved["candidate_selection_context"]
    search_strategy_context_summary = resolved["search_strategy_context_summary"]
    validation_config = resolved["validation_config"]
    candidate_previews = resolved["candidate_previews"]
    effective_pbrs_design_update_policy = workflow._get_effective_pbrs_design_update_policy(
        round_id=round_id,
        active_checkpoint_context=active_checkpoint_context,
    )
    base_reward_spec = workflow._get_base_reward_spec_for_round(
        round_id=round_id,
        previous_round=previous_round,
    )
    branch_critic_recommendation = workflow.workflow_spec.get("branch_critic_recommendation") or {}
    branch_critic_patch = workflow.workflow_spec.get("branch_critic_patch") or {}
    branch_design_transition = (
        branch_critic_recommendation.get("recommended_design_transition")
        or branch_critic_patch.get("recommended_design_transition")
        or {}
    )
    branch_design_transition_application = (
        workflow.workflow_spec.get("branch_design_transition_application") or {}
    )

    current_round = _build_preview_current_round(
        round_id=round_id,
        reward_paradigm=workflow._get_reward_paradigm(),
        active_pbrs_field=active_pbrs_field,
        active_pbrs_field_source=active_pbrs_field_source,
        active_field_carryover_context=active_field_carryover_context,
        candidate_selection_context=candidate_selection_context,
        active_checkpoint_context=active_checkpoint_context,
        effective_pbrs_design_update_policy=effective_pbrs_design_update_policy,
        branch_design_transition=branch_design_transition,
        search_strategy_context_summary=search_strategy_context_summary,
        validation_config=validation_config,
        candidate_previews=candidate_previews,
        base_reward_spec=base_reward_spec,
        previous_round=previous_round,
    )

    report: Dict[str, Any] = {
        "workflow_id": workflow.workflow_id,
        "workflow_dir": str(workflow.storage.workflow_dir),
        "workflow_exists": workflow.storage.workflow_dir.exists(),
        "reward_paradigm": workflow._get_reward_paradigm(),
        "branch_critic_recommendation": branch_critic_recommendation,
        "branch_critic_patch": branch_critic_patch,
        "branch_design_transition": branch_design_transition,
        "branch_design_transition_application": branch_design_transition_application,
        "preview_round": round_id,
        "active_pbrs_field": active_pbrs_field,
        "active_pbrs_field_source": active_pbrs_field_source,
        "active_checkpoint_context": active_checkpoint_context,
        "active_field_carryover_context": active_field_carryover_context,
        "candidate_selection_context": candidate_selection_context,
        "effective_pbrs_design_update_policy": effective_pbrs_design_update_policy,
        "search_strategy_context_summary": search_strategy_context_summary,
        "validation_config": validation_config,
        "candidate_previews": candidate_previews,
        "base_reward_spec": base_reward_spec,
        "previous_round": previous_round,
        "current_round_preview": current_round,
    }

    if (
        workflow.workflow_spec.get("branch_critic_patch")
        or workflow.workflow_spec.get("branch_critic_recommendation")
    ):
        baseline_args = deepcopy(args)
        baseline_args.use_existing_workflow_spec = False
        baseline_args.apply_attached_branch_critic_patch = False
        baseline_args.branch_plan_id = None
        baseline_workflow = _build_read_only_workflow(baseline_args)
        _maybe_apply_mock_checkpoint_stage_sequence(
            baseline_workflow,
            getattr(args, "mock_checkpoint_stage_sequence", None),
        )
        baseline_active_checkpoint_context = baseline_workflow._get_active_pbrs_checkpoint_context(
            round_id=round_id
        )
        baseline_effective_design_policy = (
            baseline_workflow._get_effective_pbrs_design_update_policy(
                round_id=round_id,
                active_checkpoint_context=baseline_active_checkpoint_context,
            )
        )
        report["baseline_effective_pbrs_design_update_policy"] = baseline_effective_design_policy
        report["design_policy_changed_by_branch_patch"] = (
            baseline_effective_design_policy != effective_pbrs_design_update_policy
        )

    if args.include_prompts:
        generator_client = TemplateBasedGeneratorClient(llm_backend=StubGeneratorBackend())
        critic_client = TemplateBasedCriticClient(llm_backend=StubCriticBackend())
        report["generator_prompt_preview"] = generator_client.build_prompt(
            round_id=round_id,
            workflow_spec=workflow.workflow_spec,
            previous_round=previous_round,
        )
        report["critic_prompt_preview"] = critic_client.build_prompt(
            round_id=round_id,
            workflow_spec=workflow.workflow_spec,
            current_round=current_round,
            previous_round=previous_round,
        )

    return report

def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n... [truncated]"


def print_text_report(report: Dict[str, Any], *, prompt_max_chars: int) -> None:
    print(
        f"{report['workflow_id']}: preview_round={report['preview_round']} "
        f"paradigm={report['reward_paradigm']} active={_fmt(report['active_pbrs_field'])} "
        f"active_source={_fmt(report['active_pbrs_field_source'])}"
    )

    checkpoint = report.get("active_checkpoint_context") or {}
    print(
        "  "
        f"checkpoint={_fmt(checkpoint.get('name'))} "
        f"stage={_fmt(checkpoint.get('stage_label'))} "
        f"step={_fmt(checkpoint.get('step'))}"
    )

    field_carryover = report.get("active_field_carryover_context") or {}
    print(
        "  "
        f"field_carryover={_fmt(field_carryover.get('previous_switch_field_applied'))} "
        f"field_carryover_source_round={_fmt(field_carryover.get('source_round_id'))} "
        f"suggested_next_field={_fmt(field_carryover.get('suggested_next_active_field'))}"
    )

    candidate_context = report.get("candidate_selection_context") or {}
    print(
        "  "
        f"candidate_source={_fmt(candidate_context.get('candidate_value_source'))} "
        f"carryover={_fmt(candidate_context.get('previous_search_strategy_applied'))} "
        f"carryover_source_round={_fmt(candidate_context.get('previous_search_strategy_source_round'))}"
    )
    print(
        "  "
        f"effective_candidate_values={json.dumps(candidate_context.get('effective_candidate_values', []), ensure_ascii=False)}"
    )

    validation_config = report.get("validation_config") or {}
    if validation_config:
        print(
            "  "
            f"validation_enabled={_fmt(validation_config.get('enabled'))} "
            f"shortlist_size={_fmt(validation_config.get('shortlist_size'))}"
        )

    print("  candidate_previews:")
    for candidate in report.get("candidate_previews", []):
        print(
            "    "
            f"{candidate.get('candidate_id')}: value={_fmt(candidate.get('candidate_value'))}"
        )

    search_summary = report.get("search_strategy_context_summary") or {}
    if search_summary:
        print(
            "  "
            f"search_summary_previous_field={_fmt(search_summary.get('previous_active_pbrs_field'))} "
            f"search_summary_previous_applied={_fmt(search_summary.get('previous_search_strategy_applied'))}"
        )

    branch_recommendation = report.get("branch_critic_recommendation") or {}
    if branch_recommendation:
        print(
            "  "
            f"branch_recommended_field={_fmt(branch_recommendation.get('recommended_next_field'))} "
            f"branch_strategy={_fmt(branch_recommendation.get('recommended_strategy'))}"
        )
    branch_design_transition = report.get("branch_design_transition") or {}
    if branch_design_transition:
        print(
            "  "
            f"branch_design_transition_action={_fmt(branch_design_transition.get('transition_action'))} "
            f"scope={_fmt(branch_design_transition.get('target_scope'))} "
            f"target_stage={_fmt(branch_design_transition.get('target_stage'))}"
        )
        print(
            "  "
            f"branch_allowed_structural_fields={json.dumps(branch_design_transition.get('allowed_structural_pbrs_fields') or [], ensure_ascii=False)}"
        )
    branch_design_transition_application = report.get("branch_design_transition_application") or {}
    if branch_design_transition_application:
        print(
            "  "
            f"branch_transition_applied={_fmt(branch_design_transition_application.get('applied'))} "
            f"branch_transition_scope={_fmt(branch_design_transition_application.get('target_scope'))} "
            f"branch_transition_stage={_fmt(branch_design_transition_application.get('target_stage'))} "
            f"branch_transition_round={_fmt(branch_design_transition_application.get('target_round_id'))}"
        )
    effective_design_policy = report.get("effective_pbrs_design_update_policy") or {}
    if effective_design_policy:
        print(
            "  "
            f"effective_design_policy={json.dumps(effective_design_policy, ensure_ascii=False, sort_keys=True)}"
        )
    baseline_effective_design_policy = (
        report.get("baseline_effective_pbrs_design_update_policy") or {}
    )
    if baseline_effective_design_policy:
        print(
            "  "
            f"baseline_effective_design_policy={json.dumps(baseline_effective_design_policy, ensure_ascii=False, sort_keys=True)}"
        )
        print(
            "  "
            f"design_policy_changed_by_branch_patch={_fmt(report.get('design_policy_changed_by_branch_patch'))}"
        )

    generator_prompt = report.get("generator_prompt_preview")
    if isinstance(generator_prompt, str):
        print("")
        print("Generator prompt preview:")
        print(_truncate(generator_prompt, prompt_max_chars))

    critic_prompt = report.get("critic_prompt_preview")
    if isinstance(critic_prompt, str):
        print("")
        print("Critic prompt preview:")
        print(_truncate(critic_prompt, prompt_max_chars))


def main() -> None:
    parser = _build_preview_parser()
    args = parser.parse_args()
    report = build_preview_report(args)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print_text_report(report, prompt_max_chars=max(200, int(args.prompt_max_chars)))


if __name__ == "__main__":
    main()
