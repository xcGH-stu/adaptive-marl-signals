from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.qmix_hparam_presets import list_qmix_hparam_presets
from scripts.run_reward_workflow import build_default_workflow_spec
from workflows.train_launcher import EPyMARLTrainLauncher
from workflows.baseline_run_support import inspect_baseline_checkpoint_readiness
from workflows.baseline_run_support import build_baseline_checkpoint_candidates


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--reward-paradigm", choices=["pbrs", "heuristic"], default="pbrs")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--env-key", default="lbforaging:Foraging-15x15-4p-3f-v3")
    parser.add_argument("--train-config", default="qmix")
    parser.add_argument(
        "--qmix-hparam-preset",
        choices=list_qmix_hparam_presets(),
        default="default",
    )
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false")
    parser.set_defaults(use_cuda=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--time-limit", type=int, default=50)
    parser.add_argument("--t-max", type=int, default=2050000)
    parser.add_argument("--test-interval", type=int, default=50000)
    parser.add_argument("--runner-log-interval", type=int, default=50000)
    parser.add_argument("--learner-log-interval", type=int, default=50000)
    parser.add_argument("--save-model-interval", type=int, default=50000)
    parser.add_argument("--local-results-path", default="results")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--load-step", type=int, default=0)
    parser.add_argument(
        "--enable-checkpointing",
        action="store_true",
        help="Enable checkpoint saving/resume metadata for this sparse baseline run.",
    )
    parser.add_argument(
        "--label-suffix",
        default="_sparse_baseline",
        help="Suffix appended to the generated baseline label.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch the sparse baseline training run. Default behavior is plan-only.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional file path to save the baseline plan/result as JSON.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    parser.add_argument("--experiment-family", default="sparse_baseline")
    parser.add_argument("--budget-profile", default="formal")
    parser.add_argument("--experiment-tag", default=None)
    parser.add_argument(
        "--checkpoint-optional-metric",
        action="append",
        default=None,
        help=(
            "Optional additional metrics to store in baseline_checkpoint_candidates.json. "
            "Sparse return metrics are always included."
        ),
    )
    return parser


def build_sparse_baseline_workflow_spec(args: argparse.Namespace) -> Dict[str, Any]:
    workflow_spec = deepcopy(
        build_default_workflow_spec(args.max_rounds, reward_paradigm=args.reward_paradigm)
    )
    train_spec = workflow_spec["train"]
    train_spec["config"] = args.train_config
    train_spec["qmix_hparam_preset"] = args.qmix_hparam_preset
    train_spec["seed"] = args.seed
    train_spec["env_args"]["key"] = args.env_key
    train_spec["env_args"]["time_limit"] = args.time_limit
    train_spec["use_dense_reward"] = False
    train_spec["overrides"]["t_max"] = args.t_max
    train_spec["overrides"]["test_interval"] = args.test_interval
    train_spec["overrides"]["runner_log_interval"] = args.runner_log_interval
    train_spec["overrides"]["learner_log_interval"] = args.learner_log_interval
    train_spec["overrides"]["use_cuda"] = bool(args.use_cuda)
    train_spec["checkpointing"] = {
        "enabled": bool(args.enable_checkpointing),
        "save_model_interval": args.save_model_interval,
        "checkpoint_path": args.checkpoint_path,
        "load_step": args.load_step,
        "local_results_path": args.local_results_path,
    }
    workflow_spec["workflow_id"] = args.baseline_id
    workflow_spec["experiment_metadata"] = {
        "experiment_family": args.experiment_family,
        "budget_profile": args.budget_profile,
        "seed": args.seed,
        "experiment_tag": args.experiment_tag,
        "qmix_hparam_preset": args.qmix_hparam_preset,
    }
    return workflow_spec


def build_sparse_baseline_plan(args: argparse.Namespace) -> Dict[str, Any]:
    workflow_spec = build_sparse_baseline_workflow_spec(args)
    launcher = EPyMARLTrainLauncher(repo_root=ROOT, python_executable=args.python_executable)
    label_override = f"workflow_{args.baseline_id}_round_0{args.label_suffix}"
    training_plan = launcher.build_training_plan(
        round_id=0,
        workflow_spec=workflow_spec,
        reward_module_path="",
        alpha_policy=None,
        policy_guidance_spec=workflow_spec.get("base_policy_guidance_spec"),
        workflow_id=args.baseline_id,
        candidate_id=None,
        reward_paradigm=args.reward_paradigm,
        active_pbrs_field=None,
        active_pbrs_field_source=None,
        candidate_value=None,
        active_checkpoint_context=None,
        active_field_carryover_context=None,
        candidate_selection_context=None,
        phase_name="sparse_baseline",
        train_overrides_override=None,
        label_override=label_override,
    )
    train_config = training_plan["train_config"]
    command = training_plan["command"]
    plan = {
        "baseline_id": args.baseline_id,
        "reward_paradigm": args.reward_paradigm,
        "workflow_spec": workflow_spec,
        "train_config": train_config,
        "command": command,
    }

    if not args.execute:
        return plan

    result = launcher.run_training(
        round_id=0,
        workflow_spec=workflow_spec,
        reward_module_path="",
        alpha_policy=None,
        policy_guidance_spec=workflow_spec.get("base_policy_guidance_spec"),
        workflow_id=args.baseline_id,
        candidate_id=None,
        reward_paradigm=args.reward_paradigm,
        active_pbrs_field=None,
        active_pbrs_field_source=None,
        candidate_value=None,
        active_checkpoint_context=None,
        active_field_carryover_context=None,
        candidate_selection_context=None,
        phase_name="sparse_baseline",
        train_overrides_override=None,
        label_override=label_override,
    )
    plan["result"] = result
    run_reference = result.get("run_reference") or {}
    baseline_run_dir = run_reference.get("run_dir")
    if isinstance(baseline_run_dir, str) and baseline_run_dir and args.enable_checkpointing:
        plan["baseline_readiness"] = inspect_baseline_checkpoint_readiness(
            baseline_run_dir=baseline_run_dir,
            baseline_result_json=None,
        )
        checkpoint_candidates = build_baseline_checkpoint_candidates(
            baseline_run_dir=baseline_run_dir,
            baseline_result_json=None,
            optional_metric_names=args.checkpoint_optional_metric,
        )
        plan["baseline_checkpoint_candidates"] = checkpoint_candidates
        baseline_run_path = Path(baseline_run_dir)
        (baseline_run_path / "baseline_checkpoint_candidates.json").write_text(
            json.dumps(checkpoint_candidates, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        plan["branching_command"] = _build_branching_command(
            python_executable=args.python_executable,
            baseline_run_dir=baseline_run_dir,
            baseline_id=args.baseline_id,
        )
    return plan


def _build_branching_command(
    *,
    python_executable: str,
    baseline_run_dir: str,
    baseline_id: str,
) -> list[str]:
    return [
        python_executable,
        "src/scripts/run_pbrs_branching_from_baseline_run.py",
        "--branch-plan-id",
        f"{baseline_id}_branching",
        "--baseline-run-dir",
        str(baseline_run_dir),
        "--format",
        "text",
    ]


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def print_text_plan(plan: Dict[str, Any]) -> None:
    train_config = plan["train_config"]
    print(
        f"{plan['baseline_id']}: paradigm={plan['reward_paradigm']} "
        f"algorithm={_fmt(train_config.get('config'))} "
        f"env={_fmt((train_config.get('env_args') or {}).get('key'))}"
    )
    print(
        "  "
        f"label={_fmt(train_config.get('label'))} "
        f"use_dense_reward={_fmt(train_config.get('use_dense_reward'))} "
        f"save_model={_fmt(train_config.get('save_model'))} "
        f"save_model_interval={_fmt(train_config.get('save_model_interval'))}"
    )
    print(
        "  "
        f"t_max={_fmt((train_config.get('overrides') or {}).get('t_max'))} "
        f"test_interval={_fmt((train_config.get('overrides') or {}).get('test_interval'))} "
        f"local_results_path={_fmt(train_config.get('local_results_path'))}"
    )
    print("  command:")
    print("    " + " ".join(str(item) for item in plan["command"]))
    if "result" in plan:
        result = plan["result"]
        print(
            "  "
            f"run_id={_fmt((result.get('run_reference') or {}).get('run_id'))} "
            f"checkpoint_root_dir={_fmt((result.get('run_reference') or {}).get('checkpoint_root_dir'))} "
            f"latest_checkpoint_step={_fmt((result.get('run_reference') or {}).get('latest_checkpoint_step'))}"
        )
    readiness = plan.get("baseline_readiness") or {}
    if readiness:
        checkpoint_summary = readiness.get("checkpoint_summary") or {}
        print(
            "  "
            f"branching_ready={_fmt(readiness.get('branching_ready'))} "
            f"available_checkpoint_steps={json.dumps(checkpoint_summary.get('available_checkpoint_steps', []), ensure_ascii=False)}"
        )
    branching_command = plan.get("branching_command")
    if isinstance(branching_command, list) and branching_command:
        print("  branching_command:")
        print("    " + " ".join(str(item) for item in branching_command))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    plan = build_sparse_baseline_plan(args)
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
