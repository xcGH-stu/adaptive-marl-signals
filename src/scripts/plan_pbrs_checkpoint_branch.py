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

from scripts.inspect_reward_workflow_state import DEFAULT_RESULTS_ROOT
from workflows.train_launcher import EPyMARLTrainLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Candidate id for candidate-based rounds. If omitted, use the round-level train artifacts.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=0,
        help="Checkpoint step to resume from. 0 means latest available checkpoint.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing llm_reward_workflows results.",
    )
    parser.add_argument(
        "--branch-use-dense-reward",
        choices=["true", "false"],
        default=None,
        help="Override whether the branch run should use dense reward.",
    )
    parser.add_argument(
        "--reward-module-path",
        default=None,
        help="Optional override reward module path for the branch run.",
    )
    parser.add_argument(
        "--alpha-policy-json",
        default=None,
        help="Optional JSON file for alpha_policy_config override.",
    )
    parser.add_argument(
        "--label-suffix",
        default="_branch",
        help="Suffix appended to the base label for the branch run.",
    )
    parser.add_argument(
        "--save-model",
        choices=["true", "false"],
        default=None,
        help="Override save_model in the branch run.",
    )
    parser.add_argument(
        "--save-model-interval",
        type=int,
        default=None,
        help="Override save_model_interval in the branch run.",
    )
    parser.add_argument(
        "--local-results-path",
        default=None,
        help="Override local_results_path in the branch run.",
    )
    parser.add_argument(
        "--phase-name",
        default="checkpoint_branch",
        help="Phase name metadata for the branch run.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional file path to save the generated plan as JSON.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_optional_bool(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    return raw.lower() == "true"


def _load_alpha_policy(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    return _load_json(Path(path))


def _load_train_artifacts(
    *,
    workflow_dir: Path,
    round_id: int,
    candidate_id: Optional[str],
) -> Dict[str, Any]:
    round_dir = workflow_dir / f"round_{int(round_id):02d}"
    if candidate_id is not None:
        candidate_dir = round_dir / "candidates" / str(candidate_id)
        train_config_path = candidate_dir / "train_config.json"
        train_run_ref_path = candidate_dir / "train_run_ref.json"
        if not train_config_path.exists() or not train_run_ref_path.exists():
            raise FileNotFoundError(
                f"candidate artifacts not found for round {round_id} candidate {candidate_id}"
            )
        return {
            "scope": "candidate",
            "scope_id": candidate_id,
            "train_config": _load_json(train_config_path),
            "run_reference": _load_json(train_run_ref_path),
        }

    train_config_path = round_dir / "train_config.json"
    train_run_ref_path = round_dir / "train_run_ref.json"
    if not train_config_path.exists() or not train_run_ref_path.exists():
        raise FileNotFoundError(f"round-level train artifacts not found for round {round_id}")
    return {
        "scope": "round",
        "scope_id": round_id,
        "train_config": _load_json(train_config_path),
        "run_reference": _load_json(train_run_ref_path),
    }


def build_branch_plan(args: argparse.Namespace) -> Dict[str, Any]:
    workflow_dir = Path(args.results_root) / args.workflow_id
    artifacts = _load_train_artifacts(
        workflow_dir=workflow_dir,
        round_id=args.round_id,
        candidate_id=args.candidate_id,
    )
    train_config = artifacts["train_config"]
    run_reference = artifacts["run_reference"]

    launcher = EPyMARLTrainLauncher(repo_root=ROOT, python_executable=sys.executable)
    run_reference = launcher._augment_run_reference_with_checkpoint_info(
        train_config=train_config,
        run_reference=dict(run_reference),
    )
    plan = launcher.build_checkpoint_resume_plan(
        base_train_config=train_config,
        run_reference=run_reference,
        requested_step=args.checkpoint_step,
        label_suffix=args.label_suffix,
        use_dense_reward=_parse_optional_bool(args.branch_use_dense_reward),
        reward_module_path=args.reward_module_path,
        alpha_policy_config=_load_alpha_policy(args.alpha_policy_json),
        workflow_id=train_config.get("workflow_id"),
        workflow_round=train_config.get("workflow_round"),
        reward_paradigm=train_config.get("reward_paradigm"),
        active_pbrs_field=train_config.get("active_pbrs_field"),
        active_pbrs_field_source=train_config.get("active_pbrs_field_source"),
        candidate_value=train_config.get("candidate_value"),
        active_checkpoint_context=train_config.get("active_checkpoint_context"),
        active_field_carryover_context=train_config.get("active_field_carryover_context"),
        candidate_selection_context=train_config.get("candidate_selection_context"),
        phase_name=args.phase_name,
        save_model=_parse_optional_bool(args.save_model),
        save_model_interval=args.save_model_interval,
        local_results_path=args.local_results_path,
    )
    return {
        "workflow_id": args.workflow_id,
        "round_id": args.round_id,
        "candidate_id": args.candidate_id,
        "artifact_scope": artifacts["scope"],
        "artifact_scope_id": artifacts["scope_id"],
        "base_train_config": train_config,
        "base_run_reference": run_reference,
        "branch_plan": plan,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def print_text_plan(plan: Dict[str, Any]) -> None:
    branch_plan = plan["branch_plan"]
    resume_train_config = branch_plan["resume_train_config"]
    print(
        f"{plan['workflow_id']}: round={plan['round_id']} "
        f"candidate={_fmt(plan['candidate_id'])} scope={plan['artifact_scope']}"
    )
    print(
        "  "
        f"checkpoint_root={_fmt(branch_plan.get('checkpoint_root_dir'))} "
        f"selected_step={_fmt(branch_plan.get('selected_checkpoint_step'))}"
    )
    print(
        "  "
        f"available_steps={json.dumps(branch_plan.get('available_checkpoint_steps', []), ensure_ascii=False)}"
    )
    print(
        "  "
        f"label={_fmt(resume_train_config.get('label'))} "
        f"use_dense_reward={_fmt(resume_train_config.get('use_dense_reward'))} "
        f"checkpoint_path={_fmt(resume_train_config.get('checkpoint_path'))} "
        f"load_step={_fmt(resume_train_config.get('load_step'))}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        plan = build_branch_plan(args)
    except ValueError as exc:
        raise SystemExit(
            f"Unable to build checkpoint branch plan: {exc}. "
            "This source run may not have checkpoint metadata or saved model checkpoints yet."
        ) from exc
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print_text_plan(plan)


if __name__ == "__main__":
    main()
