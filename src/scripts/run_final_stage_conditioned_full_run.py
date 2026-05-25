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

from scripts.run_reward_workflow import build_default_workflow_spec
from workflows.policy_guidance import get_default_policy_guidance_spec
from workflows.stage_conditioned_renderer import render_stage_conditioned_reward_bundle
from workflows.train_launcher import EPyMARLTrainLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--final-spec-path", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--env-key", default="lbforaging:Foraging-8x8-2p-1f-v3")
    parser.add_argument("--train-config", default="qmix")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--time-limit", type=int, default=50)
    parser.add_argument("--t-max", type=int, default=2050000)
    parser.add_argument("--test-interval", type=int, default=25000)
    parser.add_argument("--runner-log-interval", type=int, default=25000)
    parser.add_argument("--learner-log-interval", type=int, default=25000)
    parser.add_argument("--reward-paradigm", choices=["pbrs"], default="pbrs")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--experiment-family", default="stage_conditioned_full_run")
    parser.add_argument("--budget-profile", default="formal")
    parser.add_argument("--experiment-tag", default=None)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false")
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true")
    parser.set_defaults(use_cuda=None)
    return parser


def build_final_stage_conditioned_full_run_plan(args: argparse.Namespace) -> Dict[str, Any]:
    final_spec = json.loads(Path(args.final_spec_path).read_text(encoding="utf-8"))
    native_stage_schedule = final_spec.get("native_original_pbrs_stage_schedule") or None

    workflow_spec = deepcopy(build_default_workflow_spec(1, reward_paradigm=args.reward_paradigm))
    workflow_spec["workflow_id"] = args.run_id
    workflow_spec["experiment_metadata"] = {
        "experiment_family": args.experiment_family,
        "budget_profile": args.budget_profile,
        "seed": args.seed,
        "experiment_tag": args.experiment_tag,
    }

    train_spec = workflow_spec["train"]
    train_spec["config"] = args.train_config
    train_spec["seed"] = args.seed
    train_spec["env_args"]["key"] = args.env_key
    train_spec["env_args"]["time_limit"] = args.time_limit
    train_spec["use_dense_reward"] = not bool(native_stage_schedule)
    train_spec["checkpointing"]["enabled"] = False
    train_spec["overrides"]["t_max"] = args.t_max
    train_spec["overrides"]["test_interval"] = args.test_interval
    train_spec["overrides"]["runner_log_interval"] = args.runner_log_interval
    train_spec["overrides"]["learner_log_interval"] = args.learner_log_interval
    if args.use_cuda is not None:
        train_spec["overrides"]["use_cuda"] = bool(args.use_cuda)

    output_dir = ROOT / "results" / "stage_conditioned_full_runs" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    final_spec_copy_path = output_dir / "final_stage_conditioned_reward_spec.json"
    final_spec_copy_path.write_text(
        json.dumps(final_spec, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    bundle = None
    alpha_policy = None
    if native_stage_schedule:
        env_args = train_spec["env_args"]
        env_args["use_pbrs"] = True
        env_args["eval_use_pbrs"] = bool(native_stage_schedule.get("eval_use_pbrs", False))
        env_args["pbrs_variant"] = str(native_stage_schedule.get("variant", "original"))
        env_args["pbrs_gamma"] = float(native_stage_schedule.get("gamma", 0.99))
        env_args["pbrs_stage_schedule_enabled"] = True
        env_args["pbrs_stage_boundary_1"] = int(native_stage_schedule["stage_boundary_1"])
        env_args["pbrs_stage_boundary_2"] = int(native_stage_schedule["stage_boundary_2"])
        env_args["pbrs_stage_beta_early"] = float(native_stage_schedule["beta"]["early"])
        env_args["pbrs_stage_beta_mid"] = float(native_stage_schedule["beta"]["mid"])
        env_args["pbrs_stage_beta_late"] = float(native_stage_schedule["beta"]["late"])
        env_args["pbrs_stage_wc_early"] = float(native_stage_schedule["wc"]["early"])
        env_args["pbrs_stage_wc_mid"] = float(native_stage_schedule["wc"]["mid"])
        env_args["pbrs_stage_wc_late"] = float(native_stage_schedule["wc"]["late"])
        env_args["pbrs_stage_wp_early"] = float(native_stage_schedule["wp"]["early"])
        env_args["pbrs_stage_wp_mid"] = float(native_stage_schedule["wp"]["mid"])
        env_args["pbrs_stage_wp_late"] = float(native_stage_schedule["wp"]["late"])
        reward_module_path = ""
    else:
        bundle = render_stage_conditioned_reward_bundle(final_spec, output_dir=output_dir)
        stage_reward_specs = final_spec.get("stage_reward_specs") or {}
        if stage_reward_specs:
            first_stage_spec = next(iter(stage_reward_specs.values()))
            alpha_policy = (first_stage_spec or {}).get("alpha_policy")
        reward_module_path = str(bundle["master_reward_module_path"])

    launcher = EPyMARLTrainLauncher(repo_root=ROOT, python_executable=args.python_executable)
    training_plan = launcher.build_training_plan(
        round_id=0,
        workflow_spec=workflow_spec,
        reward_module_path=reward_module_path,
        alpha_policy=alpha_policy,
        policy_guidance_spec=get_default_policy_guidance_spec(),
        workflow_id=args.run_id,
        candidate_id=None,
        reward_paradigm=args.reward_paradigm,
        active_pbrs_field=None,
        active_pbrs_field_source="stage_conditioned_final_spec",
        candidate_value=None,
        active_checkpoint_context=None,
        active_field_carryover_context=None,
        candidate_selection_context={
            "stage_conditioned_reward_spec_path": str(final_spec_copy_path),
            "stage_boundaries": (
                native_stage_schedule
                if native_stage_schedule is not None
                else bundle["stage_boundaries"]
            ),
        },
        phase_name="stage_conditioned_final_full_run",
        train_overrides_override=None,
        label_override=f"stage_conditioned_final_{args.run_id}",
    )

    plan = {
        "run_id": args.run_id,
        "reward_paradigm": args.reward_paradigm,
        "experiment_metadata": workflow_spec["experiment_metadata"],
        "final_stage_conditioned_reward_spec_path": str(final_spec_copy_path),
        "native_original_pbrs_stage_schedule": native_stage_schedule,
        "master_reward_module_path": None if bundle is None else str(bundle["master_reward_module_path"]),
        "bundle_metadata_path": None if bundle is None else str(bundle["bundle_metadata_path"]),
        "stage_module_paths": {} if bundle is None else bundle["stage_module_paths"],
        "workflow_spec": workflow_spec,
        "train_config": training_plan["train_config"],
        "command": training_plan["command"],
    }
    if args.execute:
        result = launcher.execute_training_plan(training_plan)
        plan["result"] = result
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
        f"final_spec_path={_fmt(plan.get('final_stage_conditioned_reward_spec_path'))} "
        f"master_reward_module_path={_fmt(plan.get('master_reward_module_path'))}"
    )
    print("  command:")
    print("    " + " ".join(str(item) for item in plan["command"]))
    if "result" in plan:
        run_ref = (plan["result"] or {}).get("run_reference") or {}
        print("  " f"run_id={_fmt(run_ref.get('run_id'))} run_dir={_fmt(run_ref.get('run_dir'))}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    plan = build_final_stage_conditioned_full_run_plan(args)
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
