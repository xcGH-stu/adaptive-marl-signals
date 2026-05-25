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
from rewarding.spec_renderer import render_reward_code_from_spec
from rewarding.spec_schema import get_default_reward_spec, validate_reward_spec
from scripts.run_reward_workflow import build_default_workflow_spec
from workflows.baseline_run_support import build_baseline_checkpoint_candidates
from workflows.baseline_run_support import inspect_baseline_checkpoint_readiness
from workflows.policy_guidance import get_default_policy_guidance_spec
from workflows.train_launcher import EPyMARLTrainLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reward-paradigm", choices=["pbrs", "heuristic"], default="pbrs")
    parser.add_argument("--native-original-pbrs", action="store_true")
    parser.add_argument("--native-stage-conditioned-pbrs", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--env-key", default="lbforaging:Foraging-8x8-2p-1f-v3")
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
    parser.add_argument("--t-max", type=int, default=60000)
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
        help="Enable checkpoint saving/resume metadata for this run.",
    )
    parser.add_argument(
        "--save-final-model",
        action="store_true",
        help="Request one final checkpoint save at training exit when checkpointing is enabled.",
    )
    parser.add_argument("--reward-spec-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--pbrs-beta", type=float, default=0.3)
    parser.add_argument("--pbrs-wc", type=float, default=0.6)
    parser.add_argument("--pbrs-wp", type=float, default=0.4)
    parser.add_argument("--pbrs-gamma", type=float, default=0.99)
    parser.add_argument("--pbrs-variant", default="original")
    parser.add_argument("--pbrs-version", default="")
    parser.add_argument("--pbrs-mode", default="")
    parser.add_argument("--pbrs-active-terms", default=None)
    parser.add_argument("--pbrs-weights", default=None)
    parser.add_argument("--eval-use-pbrs", action="store_true")
    parser.add_argument("--pbrs-stage-boundary-1", type=int, default=875923)
    parser.add_argument("--pbrs-stage-boundary-2", type=int, default=1451234)
    parser.add_argument("--pbrs-stage-beta-early", type=float, default=0.3)
    parser.add_argument("--pbrs-stage-beta-mid", type=float, default=1.0)
    parser.add_argument("--pbrs-stage-beta-late", type=float, default=0.5)
    parser.add_argument("--pbrs-stage-wc-early", type=float, default=0.1)
    parser.add_argument("--pbrs-stage-wc-mid", type=float, default=1.0)
    parser.add_argument("--pbrs-stage-wc-late", type=float, default=0.7)
    parser.add_argument("--pbrs-stage-wp-early", type=float, default=0.7)
    parser.add_argument("--pbrs-stage-wp-mid", type=float, default=0.1)
    parser.add_argument("--pbrs-stage-wp-late", type=float, default=0.3)
    execute_group = parser.add_mutually_exclusive_group()
    execute_group.add_argument("--execute", dest="execute", action="store_true")
    execute_group.add_argument("--dry-run", dest="execute", action="store_false")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--experiment-family", default="pbrs_fixed")
    parser.add_argument("--budget-profile", default="sanity")
    parser.add_argument("--experiment-tag", default=None)
    parser.set_defaults(execute=False)
    return parser


def _load_reward_spec(path: str | None, reward_paradigm: str, workflow_spec: Dict[str, Any]) -> Dict[str, Any]:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return validate_reward_spec(payload)
    reward_spec = get_default_reward_spec()
    if reward_paradigm == "pbrs":
        reward_spec["pbrs"] = deepcopy(
            ((workflow_spec.get("pbrs_tuning") or {}).get("base_pbrs") or {})
        )
        reward_spec["alpha_policy"] = {"type": "constant", "value": 1.0}
    return validate_reward_spec(reward_spec)


def build_fixed_reward_baseline_plan(args: argparse.Namespace) -> Dict[str, Any]:
    workflow_spec = deepcopy(build_default_workflow_spec(1, reward_paradigm=args.reward_paradigm))
    train_spec = workflow_spec["train"]
    train_spec["config"] = args.train_config
    train_spec["qmix_hparam_preset"] = args.qmix_hparam_preset
    train_spec["seed"] = args.seed
    train_spec["env_args"]["key"] = args.env_key
    train_spec["env_args"]["time_limit"] = args.time_limit
    train_spec["use_dense_reward"] = not (
        args.native_original_pbrs or args.native_stage_conditioned_pbrs
    )
    train_spec["overrides"]["t_max"] = args.t_max
    train_spec["overrides"]["test_interval"] = args.test_interval
    train_spec["overrides"]["runner_log_interval"] = args.runner_log_interval
    train_spec["overrides"]["learner_log_interval"] = args.learner_log_interval
    train_spec["overrides"]["use_cuda"] = bool(args.use_cuda)
    train_spec["checkpointing"] = {
        "enabled": bool(args.enable_checkpointing),
        "save_model_interval": args.save_model_interval,
        "save_final_model": bool(getattr(args, "save_final_model", False)),
        "checkpoint_path": args.checkpoint_path,
        "load_step": args.load_step,
        "local_results_path": args.local_results_path,
    }
    workflow_spec["workflow_id"] = args.run_id
    workflow_spec["experiment_metadata"] = {
        "experiment_family": args.experiment_family,
        "budget_profile": args.budget_profile,
        "seed": args.seed,
        "experiment_tag": args.experiment_tag,
        "native_original_pbrs": bool(args.native_original_pbrs),
        "native_stage_conditioned_pbrs": bool(args.native_stage_conditioned_pbrs),
        "qmix_hparam_preset": args.qmix_hparam_preset,
    }
    reward_spec = None
    reward_spec_path = None
    reward_module_path = ""
    if args.native_original_pbrs or args.native_stage_conditioned_pbrs:
        train_spec["env_args"]["use_pbrs"] = True
        train_spec["env_args"]["eval_use_pbrs"] = bool(args.eval_use_pbrs)
        train_spec["env_args"]["pbrs_beta"] = float(args.pbrs_beta)
        train_spec["env_args"]["pbrs_wc"] = float(args.pbrs_wc)
        train_spec["env_args"]["pbrs_wp"] = float(args.pbrs_wp)
        train_spec["env_args"]["pbrs_version"] = str(getattr(args, "pbrs_version", "") or "")
        train_spec["env_args"]["pbrs_mode"] = str(getattr(args, "pbrs_mode", "") or "")
        train_spec["env_args"]["pbrs_active_terms"] = deepcopy(
            getattr(args, "pbrs_active_terms", None) or []
        )
        train_spec["env_args"]["pbrs_weights"] = deepcopy(
            getattr(args, "pbrs_weights", None) or {}
        )
        train_spec["env_args"]["pbrs_gamma"] = float(args.pbrs_gamma)
        train_spec["env_args"]["pbrs_variant"] = str(args.pbrs_variant)
        if args.native_stage_conditioned_pbrs:
            train_spec["env_args"]["pbrs_stage_schedule_enabled"] = True
            train_spec["env_args"]["pbrs_stage_boundary_1"] = int(args.pbrs_stage_boundary_1)
            train_spec["env_args"]["pbrs_stage_boundary_2"] = int(args.pbrs_stage_boundary_2)
            train_spec["env_args"]["pbrs_stage_beta_early"] = float(args.pbrs_stage_beta_early)
            train_spec["env_args"]["pbrs_stage_beta_mid"] = float(args.pbrs_stage_beta_mid)
            train_spec["env_args"]["pbrs_stage_beta_late"] = float(args.pbrs_stage_beta_late)
            train_spec["env_args"]["pbrs_stage_wc_early"] = float(args.pbrs_stage_wc_early)
            train_spec["env_args"]["pbrs_stage_wc_mid"] = float(args.pbrs_stage_wc_mid)
            train_spec["env_args"]["pbrs_stage_wc_late"] = float(args.pbrs_stage_wc_late)
            train_spec["env_args"]["pbrs_stage_wp_early"] = float(args.pbrs_stage_wp_early)
            train_spec["env_args"]["pbrs_stage_wp_mid"] = float(args.pbrs_stage_wp_mid)
            train_spec["env_args"]["pbrs_stage_wp_late"] = float(args.pbrs_stage_wp_late)
    else:
        reward_spec = _load_reward_spec(args.reward_spec_path, args.reward_paradigm, workflow_spec)
        reward_dir = ROOT / "results" / "fixed_reward_baselines" / args.run_id
        reward_dir.mkdir(parents=True, exist_ok=True)
        reward_spec_path = reward_dir / "reward_spec.json"
        reward_module_path = reward_dir / "reward_function.py"
        reward_spec_path.write_text(
            json.dumps(reward_spec, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        reward_module_path.write_text(
            render_reward_code_from_spec(reward_spec),
            encoding="utf-8",
        )

    launcher = EPyMARLTrainLauncher(repo_root=ROOT, python_executable=args.python_executable)
    training_plan = launcher.build_training_plan(
        round_id=0,
        workflow_spec=workflow_spec,
        reward_module_path=str(reward_module_path),
        alpha_policy=None if reward_spec is None else reward_spec.get("alpha_policy"),
        policy_guidance_spec=get_default_policy_guidance_spec(),
        workflow_id=args.run_id,
        candidate_id=None,
        reward_paradigm=args.reward_paradigm,
        active_pbrs_field=None,
        active_pbrs_field_source=None,
        candidate_value=None,
        active_checkpoint_context=None,
        active_field_carryover_context=None,
        candidate_selection_context=None,
        phase_name="fixed_reward_baseline",
        train_overrides_override=None,
        label_override=f"fixed_reward_{args.run_id}",
    )
    plan = {
        "run_id": args.run_id,
        "reward_paradigm": args.reward_paradigm,
        "native_original_pbrs": bool(args.native_original_pbrs),
        "native_stage_conditioned_pbrs": bool(args.native_stage_conditioned_pbrs),
        "experiment_metadata": workflow_spec["experiment_metadata"],
        "reward_spec_path": None if reward_spec_path is None else str(reward_spec_path),
        "reward_module_path": str(reward_module_path),
        "workflow_spec": workflow_spec,
        "train_config": training_plan["train_config"],
        "command": training_plan["command"],
    }
    if args.execute:
        result = launcher.execute_training_plan(training_plan)
        plan["result"] = result
        run_reference = result.get("run_reference") or {}
        run_dir = run_reference.get("run_dir")
        if isinstance(run_dir, str) and run_dir and args.enable_checkpointing:
            plan["baseline_readiness"] = inspect_baseline_checkpoint_readiness(
                baseline_run_dir=run_dir,
                baseline_result_json=None,
            )
            checkpoint_candidates = build_baseline_checkpoint_candidates(
                baseline_run_dir=run_dir,
                baseline_result_json=None,
                optional_metric_names=[
                    "return_mean",
                    "test_return_mean",
                    "ep_length_mean",
                    "test_ep_length_mean",
                ],
            )
            plan["baseline_checkpoint_candidates"] = checkpoint_candidates
            run_dir_path = Path(run_dir)
            (run_dir_path / "baseline_checkpoint_candidates.json").write_text(
                json.dumps(checkpoint_candidates, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
    return plan


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _print_text(plan: Dict[str, Any]) -> None:
    train_config = plan["train_config"]
    print(
        f"{plan['run_id']}: family={_fmt((plan.get('experiment_metadata') or {}).get('experiment_family'))} "
        f"seed={_fmt((plan.get('experiment_metadata') or {}).get('seed'))} "
        f"algorithm={_fmt(train_config.get('config'))} "
        f"env={_fmt((train_config.get('env_args') or {}).get('key'))}"
    )
    print(
        "  "
        f"reward_spec_path={_fmt(plan.get('reward_spec_path'))} "
        f"reward_module_path={_fmt(plan.get('reward_module_path'))}"
    )
    print("  command:")
    print("    " + " ".join(str(item) for item in plan["command"]))
    if "result" in plan:
        run_ref = (plan["result"] or {}).get("run_reference") or {}
        print(
            "  "
            f"run_id={_fmt(run_ref.get('run_id'))} "
            f"run_dir={_fmt(run_ref.get('run_dir'))}"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    plan = build_fixed_reward_baseline_plan(args)
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
    _print_text(plan)


if __name__ == "__main__":
    main()
