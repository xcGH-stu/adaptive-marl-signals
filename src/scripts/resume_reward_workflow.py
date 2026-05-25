from __future__ import annotations

from pathlib import Path

from scripts.inspect_reward_workflow_state import inspect_workflow
from scripts.run_reward_workflow import build_arg_parser, build_workflow_from_args


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "results" / "llm_reward_workflows"


def main() -> None:
    parser = build_arg_parser()
    parser.description = "Resume an existing reward workflow from a suggested or explicit round."
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing llm_reward_workflows results.",
    )
    parser.add_argument(
        "--resume-round",
        type=int,
        default=None,
        help="Explicit round to resume from. If omitted, use the suggested start round.",
    )
    parser.set_defaults(use_existing_workflow_spec=True)
    parser.set_defaults(apply_attached_branch_critic_patch=True)
    args = parser.parse_args()

    workflow_dir = Path(args.results_root) / args.workflow_id
    report = inspect_workflow(workflow_dir)
    suggested_round = (report.get("resume_plan") or {}).get("suggested_start_round")
    if args.resume_round is not None:
        args.start_round = int(args.resume_round)
    elif suggested_round is not None:
        args.start_round = int(suggested_round)
    else:
        raise ValueError(
            f"No suggested resume round for workflow {args.workflow_id}; "
            "pass --resume-round explicitly if you want to rerun a round."
        )

    workflow = build_workflow_from_args(args)
    result = workflow.run(start_round=args.start_round)

    print("reward workflow resume completed")
    print(f"workflow_id={result.workflow_id}")
    print(f"resumed_from_round={args.start_round}")
    print(f"completed_rounds={result.completed_rounds}")
    print(f"manifest_status={result.manifest.get('status')}")
    print(f"last_round_status={result.last_round_status.get('status')}")
    print(
        "candidate_reuse="
        f"{result.last_round_status.get('reused_candidate_count', '-')}"
    )
    print(
        "candidate_rerun="
        f"{result.last_round_status.get('rerun_candidate_count', '-')}"
    )


if __name__ == "__main__":
    main()
