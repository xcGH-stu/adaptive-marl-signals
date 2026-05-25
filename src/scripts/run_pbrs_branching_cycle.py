from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.critique_pbrs_branch_plan import build_branch_critique
from scripts.inspect_pbrs_branch_plan_state import inspect_branch_plan
from scripts.plan_pbrs_baseline_branching import build_branching_plan
from scripts.resume_pbrs_branch_plan import resume_branch_plan
from scripts.run_reward_workflow import load_existing_workflow_spec
from workflows.branch_critic_patch import (
    attach_branch_critic_payload_to_workflow,
    derive_workflow_patch,
    infer_target_next_round_id_from_workflow_manifest,
)
from workflows.storage import WorkflowStorage


DEFAULT_BRANCH_PLAN_ROOT = ROOT / "results" / "pbrs_branch_plans"
DEFAULT_WORKFLOW_ROOT = ROOT / "results" / "llm_reward_workflows"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-plan-id", required=True)
    parser.add_argument(
        "--reuse-existing-branch-plan",
        action="store_true",
        help="Skip branch planning and critique an already existing branch plan directory.",
    )
    parser.add_argument("--baseline-metrics-json", default=None)
    parser.add_argument("--baseline-run-config-json", default=None)
    parser.add_argument("--baseline-run-info-json", default=None)
    parser.add_argument("--reward-paradigm", choices=["pbrs"], default="pbrs")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-count", type=int, default=4)
    parser.add_argument("--min-step-gap", type=int, default=50000)
    parser.add_argument("--max-points-per-metric", type=int, default=15)
    parser.add_argument(
        "--preferred-metric",
        action="append",
        default=None,
        help="Metric names to provide to the selector; can be passed multiple times.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--branch-results-root", default=str(DEFAULT_BRANCH_PLAN_ROOT))
    parser.add_argument(
        "--workflow-results-root",
        default=str(DEFAULT_WORKFLOW_ROOT),
        help="Root directory containing llm_reward_workflows results.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch checkpoint branch training runs. With --reuse-existing-branch-plan, this resumes missing branch candidates.",
    )
    parser.add_argument(
        "--round-id",
        type=int,
        default=None,
        help="Optional round filter for branch-plan resume execution.",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Optional candidate filter for branch-plan resume execution.",
    )
    parser.add_argument(
        "--reuse-completed-candidates",
        action="store_true",
        help="Reuse existing branch_result.json files instead of rerunning completed candidates.",
    )
    parser.add_argument(
        "--target-workflow-id",
        default=None,
        help="Optional workflow id to receive the derived branch-critic patch.",
    )
    parser.add_argument(
        "--attach-to-workflow",
        action="store_true",
        help="Attach the derived branch patch to the target workflow manifest and artifact root.",
    )
    parser.add_argument("--use-real-llm", action="store_true")
    parser.add_argument("--api-key-env", default="IUSEAPI_API_KEY")
    parser.add_argument("--base-url", default="https://www.iuseapi.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=5.0)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional file path to save the combined branching-cycle payload.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.attach_to_workflow and not args.target_workflow_id:
        raise ValueError("--attach-to-workflow requires --target-workflow-id")
    if args.reuse_existing_branch_plan:
        return
    required = [
        "baseline_metrics_json",
        "baseline_run_config_json",
        "baseline_run_info_json",
    ]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError(
            "building a new branch plan requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )


def _build_plan_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        branch_plan_id=args.branch_plan_id,
        baseline_metrics_json=args.baseline_metrics_json,
        baseline_run_config_json=args.baseline_run_config_json,
        baseline_run_info_json=args.baseline_run_info_json,
        reward_paradigm=args.reward_paradigm,
        max_rounds=args.max_rounds,
        min_count=args.min_count,
        max_count=args.max_count,
        min_step_gap=args.min_step_gap,
        max_points_per_metric=args.max_points_per_metric,
        preferred_metric=args.preferred_metric,
        python_executable=args.python_executable,
        results_root=args.branch_results_root,
        workflow_results_root=args.workflow_results_root,
        execute=args.execute,
        reuse_completed_candidates=args.reuse_completed_candidates,
        use_real_llm=args.use_real_llm,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff=args.llm_retry_backoff,
        format="json",
    )


def _build_critique_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        branch_plan_id=args.branch_plan_id,
        results_root=args.branch_results_root,
        use_real_llm=args.use_real_llm,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff=args.llm_retry_backoff,
        output_path=None,
        format="json",
    )


def run_branching_cycle(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)

    branch_plan_summary: Optional[Dict[str, Any]] = None
    branch_resume_payload: Optional[Dict[str, Any]] = None
    if not args.reuse_existing_branch_plan:
        branch_plan_summary = build_branching_plan(_build_plan_args(args))
    elif args.execute:
        branch_resume_payload = resume_branch_plan(
            argparse.Namespace(
                branch_plan_id=args.branch_plan_id,
                results_root=args.branch_results_root,
                python_executable=args.python_executable,
                round_id=args.round_id,
                candidate_id=args.candidate_id,
                reuse_completed_candidates=args.reuse_completed_candidates,
                format="json",
            )
        )

    critique_payload = build_branch_critique(_build_critique_args(args))
    base_workflow_spec = None
    target_next_round_id = None
    if args.target_workflow_id is not None:
        base_workflow_spec = load_existing_workflow_spec(
            args.target_workflow_id,
            storage_root=args.workflow_results_root,
        )
        try:
            workflow_storage = WorkflowStorage.load(
                args.target_workflow_id,
                root_dir=args.workflow_results_root,
            )
            target_next_round_id = infer_target_next_round_id_from_workflow_manifest(
                workflow_storage.load_manifest()
            )
        except FileNotFoundError:
            target_next_round_id = None
    derived = derive_workflow_patch(
        critique_payload=critique_payload,
        base_workflow_spec=base_workflow_spec,
        target_next_round_id=target_next_round_id,
    )

    payload = {
        "branch_plan_id": args.branch_plan_id,
        "plan_built": not args.reuse_existing_branch_plan,
        "branch_plan_summary": branch_plan_summary,
        "branch_resume_summary": branch_resume_payload,
        "branch_plan_state": inspect_branch_plan(
            Path(args.branch_results_root) / args.branch_plan_id
        ),
        "critique_payload": critique_payload,
        "workflow_patch": derived["patch"],
        "updated_workflow_spec": derived["updated_workflow_spec"],
        "target_workflow_id": args.target_workflow_id,
        "attached_to_workflow": False,
    }

    branch_plan_dir = Path(args.branch_results_root) / args.branch_plan_id
    branch_plan_dir.mkdir(parents=True, exist_ok=True)
    (branch_plan_dir / "branch_critic_result.json").write_text(
        json.dumps(critique_payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (branch_plan_dir / "workflow_patch.json").write_text(
        json.dumps(
            {
                "branch_plan_id": args.branch_plan_id,
                "target_workflow_id": args.target_workflow_id,
                "workflow_patch": derived["patch"],
                "updated_workflow_spec": derived["updated_workflow_spec"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if args.attach_to_workflow:
        attach_branch_critic_payload_to_workflow(
            workflow_id=args.target_workflow_id,
            storage_root=args.workflow_results_root,
            payload={
                "branch_plan_id": args.branch_plan_id,
                "target_workflow_id": args.target_workflow_id,
                "critique_payload": critique_payload,
                "workflow_patch": derived["patch"],
                "updated_workflow_spec": derived["updated_workflow_spec"],
            },
            updated_workflow_spec=derived["updated_workflow_spec"],
        )
        payload["attached_to_workflow"] = True

    return payload


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def print_text_payload(payload: Dict[str, Any]) -> None:
    patch = payload.get("workflow_patch") or {}
    critique = (
        ((payload.get("critique_payload") or {}).get("branch_critic_result") or {})
        .get("response", {})
        .get("parsed", {})
    )
    print(
        f"{payload['branch_plan_id']}: plan_built={payload.get('plan_built')} "
        f"attached_to_workflow={payload.get('attached_to_workflow')} "
        f"target_workflow={_fmt(payload.get('target_workflow_id'))}"
    )
    branch_resume_summary = payload.get("branch_resume_summary") or {}
    if branch_resume_summary:
        resume_stats = branch_resume_summary.get("branch_resume_summary") or {}
        print(
            "  "
            f"resume_resumed={_fmt(resume_stats.get('resumed_candidate_count'))} "
            f"resume_reused={_fmt(resume_stats.get('reused_candidate_count'))} "
            f"resume_skipped={_fmt(resume_stats.get('skipped_candidate_count'))}"
        )
    branch_plan_state = payload.get("branch_plan_state") or {}
    resume_plan = branch_plan_state.get("resume_plan") or {}
    print(
        "  "
        f"branch_completed={_fmt(branch_plan_state.get('completed_candidates'))}/"
        f"{_fmt(branch_plan_state.get('total_candidates'))} "
        f"next_round={_fmt(resume_plan.get('suggested_resume_round'))} "
        f"next_candidate={_fmt(resume_plan.get('suggested_resume_candidate'))}"
    )
    print(f"  analysis_summary={_fmt(critique.get('analysis_summary'))}")
    print(
        "  "
        f"recommended_strategy={_fmt(patch.get('recommended_strategy'))} "
        f"recommended_next_field={_fmt(patch.get('recommended_next_field'))} "
        f"recommended_global_value={_fmt(patch.get('recommended_global_value'))}"
    )
    if patch.get("recommended_candidate_values") is not None:
        print(
            "  "
            f"recommended_candidate_values={json.dumps(patch.get('recommended_candidate_values'), ensure_ascii=False)}"
        )
    design_transition = patch.get("recommended_design_transition") or {}
    if design_transition:
        print(
            "  "
            f"design_transition_action={_fmt(design_transition.get('transition_action'))} "
            f"scope={_fmt(design_transition.get('target_scope'))} "
            f"target_stage={_fmt(design_transition.get('target_stage'))}"
        )
        print(
            "  "
            f"allowed_structural_pbrs_fields={json.dumps(design_transition.get('allowed_structural_pbrs_fields') or [], ensure_ascii=False)}"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload = run_branching_cycle(args)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print_text_payload(payload)


if __name__ == "__main__":
    main()
