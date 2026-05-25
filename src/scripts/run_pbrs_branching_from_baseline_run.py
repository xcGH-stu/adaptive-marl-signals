from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.run_pbrs_branching_cycle import run_branching_cycle
from workflows.baseline_run_support import (
    inspect_baseline_checkpoint_readiness,
    resolve_baseline_paths,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-plan-id", required=True)
    parser.add_argument(
        "--baseline-run-dir",
        default=None,
        help="Path to a Sacred baseline run directory containing config.json, info.json, and metrics.json.",
    )
    parser.add_argument(
        "--baseline-result-json",
        default=None,
        help="Optional JSON emitted by run_sparse_baseline_with_checkpoints.py containing result.run_reference paths.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
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
    parser.add_argument("--branch-results-root", default=str(ROOT / "results" / "pbrs_branch_plans"))
    parser.add_argument(
        "--workflow-results-root",
        default=str(ROOT / "results" / "llm_reward_workflows"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the branch runs after building the branch plan.",
    )
    parser.add_argument(
        "--allow-unready-baseline",
        action="store_true",
        help="Allow branch planning/execution even when baseline checkpoint readiness is incomplete.",
    )
    parser.add_argument(
        "--reuse-completed-candidates",
        action="store_true",
        help="Reuse existing branch_result.json files during branch execution/resume.",
    )
    parser.add_argument("--target-workflow-id", default=None)
    parser.add_argument("--attach-to-workflow", action="store_true")
    parser.add_argument("--use-real-llm", action="store_true")
    parser.add_argument("--api-key-env", default="IUSEAPI_API_KEY")
    parser.add_argument("--base-url", default="https://www.iuseapi.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=5.0)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


def _build_cycle_args(args: argparse.Namespace, baseline_paths: Dict[str, str]) -> argparse.Namespace:
    return argparse.Namespace(
        branch_plan_id=args.branch_plan_id,
        reuse_existing_branch_plan=False,
        baseline_metrics_json=baseline_paths["baseline_metrics_json"],
        baseline_run_config_json=baseline_paths["baseline_run_config_json"],
        baseline_run_info_json=baseline_paths["baseline_run_info_json"],
        reward_paradigm=args.reward_paradigm,
        max_rounds=args.max_rounds,
        min_count=args.min_count,
        max_count=args.max_count,
        min_step_gap=args.min_step_gap,
        max_points_per_metric=args.max_points_per_metric,
        preferred_metric=args.preferred_metric,
        python_executable=args.python_executable,
        branch_results_root=args.branch_results_root,
        workflow_results_root=args.workflow_results_root,
        execute=args.execute,
        round_id=None,
        candidate_id=None,
        reuse_completed_candidates=args.reuse_completed_candidates,
        target_workflow_id=args.target_workflow_id,
        attach_to_workflow=args.attach_to_workflow,
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


def run_branching_from_baseline(args: argparse.Namespace) -> Dict[str, Any]:
    readiness = inspect_baseline_checkpoint_readiness(
        baseline_run_dir=args.baseline_run_dir,
        baseline_result_json=args.baseline_result_json,
    )
    branch_plan_dir = Path(args.branch_results_root) / args.branch_plan_id
    branch_plan_dir.mkdir(parents=True, exist_ok=True)
    (branch_plan_dir / "baseline_readiness.json").write_text(
        json.dumps(readiness, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.execute and not readiness.get("branching_ready") and not args.allow_unready_baseline:
        blocked_payload = {
            "reason": (
                "baseline run is not ready for branching execution; "
                "checkpoint artifacts are incomplete or missing."
            ),
            "branching_ready": readiness.get("branching_ready"),
            "checkpoint_summary": readiness.get("checkpoint_summary"),
            "baseline_source": readiness.get("baseline_source"),
        }
        (branch_plan_dir / "branching_blocked.json").write_text(
            json.dumps(blocked_payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        raise ValueError(
            "baseline run is not ready for branching execution; "
            "use inspect_sparse_baseline_run.py to inspect missing readiness signals, "
            "or pass --allow-unready-baseline to override."
        )
    baseline_paths = readiness["baseline_source"]
    payload = run_branching_cycle(_build_cycle_args(args, baseline_paths))
    payload["baseline_source"] = baseline_paths
    payload["baseline_readiness"] = readiness
    return payload


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def print_text_payload(payload: Dict[str, Any]) -> None:
    baseline_source = payload.get("baseline_source") or {}
    baseline_readiness = payload.get("baseline_readiness") or {}
    checkpoint_summary = baseline_readiness.get("checkpoint_summary") or {}
    blockers = baseline_readiness.get("blockers") or []
    print(
        f"{payload['branch_plan_id']}: baseline_run_dir={_fmt(baseline_source.get('baseline_run_dir'))}"
    )
    print(
        "  "
        f"branching_ready={_fmt(baseline_readiness.get('branching_ready'))} "
        f"latest_checkpoint_step={_fmt(checkpoint_summary.get('latest_checkpoint_step'))}"
    )
    if blockers:
        print("  " f"blockers={json.dumps(blockers, ensure_ascii=False)}")
    branch_plan_state = payload.get("branch_plan_state") or {}
    resume_plan = branch_plan_state.get("resume_plan") or {}
    print(
        "  "
        f"branch_completed={_fmt(branch_plan_state.get('completed_candidates'))}/"
        f"{_fmt(branch_plan_state.get('total_candidates'))} "
        f"next_round={_fmt(resume_plan.get('suggested_resume_round'))} "
        f"next_candidate={_fmt(resume_plan.get('suggested_resume_candidate'))}"
    )
    patch = payload.get("workflow_patch") or {}
    print(
        "  "
        f"recommended_strategy={_fmt(patch.get('recommended_strategy'))} "
        f"recommended_next_field={_fmt(patch.get('recommended_next_field'))}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        payload = run_branching_from_baseline(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
