from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workflows.train_launcher import EPyMARLTrainLauncher


DEFAULT_CANDIDATE_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--stage-selection-result-json", required=True)
    parser.add_argument("--field-name", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--branch-budget-steps", type=int, required=True)
    parser.add_argument(
        "--stage-label",
        action="append",
        default=None,
        help="Optional stage label filter. Repeat to launch multiple stages.",
    )
    parser.add_argument(
        "--candidate-values",
        nargs="*",
        type=float,
        default=None,
        help="Optional explicit candidate values. Defaults to 0.1 0.3 0.5 0.7 0.9 1.0.",
    )
    parser.add_argument(
        "--launch-mode",
        choices=["background", "blocking"],
        default="background",
        help="Whether to detach candidate runs or execute them synchronously.",
    )
    parser.add_argument("--native-original-pbrs-branching", action="store_true")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false")
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true")
    parser.set_defaults(use_cuda=None)
    return parser


def _candidate_id(stage_label: str, field_name: str, value: Any) -> str:
    if isinstance(value, float):
        suffix = str(value).replace(".", "p")
    else:
        suffix = str(value)
    return f"{stage_label}_{field_name}_{suffix}"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _launch_missing_candidates(args: argparse.Namespace) -> Dict[str, Any]:
    workflow_dir = ROOT / "results" / "stage_conditioned_workflows" / args.workflow_id
    stage_selection_result = _load_json(Path(args.stage_selection_result_json))
    baseline_run_config = _load_json(Path(stage_selection_result["baseline_run_config_json"]))
    checkpoint_root_dir = stage_selection_result["baseline_checkpoint_root_dir"]
    available_steps = stage_selection_result["baseline_available_checkpoint_steps"]
    selected_checkpoints = list((stage_selection_result.get("selection") or {}).get("selected_checkpoints", []))
    selected_by_label = {str(item["stage_label"]): item for item in selected_checkpoints}
    requested_stage_labels: List[str] = list(args.stage_label or selected_by_label.keys())
    candidate_values = list(args.candidate_values or DEFAULT_CANDIDATE_VALUES)

    launcher = EPyMARLTrainLauncher(
        repo_root=ROOT,
        python_executable=args.python_executable,
    )

    launched = []
    skipped = []

    for stage_label in requested_stage_labels:
        checkpoint = selected_by_label.get(stage_label)
        if checkpoint is None:
            raise ValueError(f"Unknown stage label: {stage_label}")
        checkpoint_name = str(checkpoint["name"])
        stage_dir = workflow_dir / f"round_{args.round_id:02d}_{args.field_name}" / checkpoint_name
        for candidate_value in candidate_values:
            candidate_id = _candidate_id(stage_label, args.field_name, candidate_value)
            candidate_dir = stage_dir / candidate_id
            reward_function_path = candidate_dir / "reward_function.py"
            native_pbrs_config_path = candidate_dir / "native_pbrs_config.json"
            branch_result_path = candidate_dir / "branch_result.json"
            if branch_result_path.exists():
                skipped.append({"stage_label": stage_label, "candidate_value": candidate_value, "reason": "branch_result_exists"})
                continue
            if args.native_original_pbrs_branching:
                if not native_pbrs_config_path.exists():
                    skipped.append(
                        {
                            "stage_label": stage_label,
                            "candidate_value": candidate_value,
                            "reason": "missing_native_pbrs_config",
                        }
                    )
                    continue
                native_pbrs_config = _load_json(native_pbrs_config_path)
                override_env_args = dict(native_pbrs_config)
                reward_module_path = ""
                use_dense_reward = False
            else:
                if not reward_function_path.exists():
                    skipped.append({"stage_label": stage_label, "candidate_value": candidate_value, "reason": "missing_reward_function"})
                    continue
                override_env_args = None
                reward_module_path = str(reward_function_path)
                use_dense_reward = True
            override_overrides = {
                "t_max": int(checkpoint["step"]) + int(args.branch_budget_steps),
            }
            if args.use_cuda is not None:
                override_overrides["use_cuda"] = bool(args.use_cuda)
            plan = launcher.build_checkpoint_resume_plan(
                base_train_config=baseline_run_config,
                run_reference={
                    "checkpoint_root_dir": checkpoint_root_dir,
                    "available_checkpoint_steps": available_steps,
                },
                requested_step=int(checkpoint["step"]),
                label_suffix=f"_{args.workflow_id}_{args.field_name}_{checkpoint_name}_{candidate_id}",
                use_dense_reward=use_dense_reward,
                reward_module_path=reward_module_path,
                alpha_policy_config={"type": "constant", "value": 1.0},
                workflow_id=args.workflow_id,
                workflow_round=int(args.round_id),
                reward_paradigm="pbrs",
                active_pbrs_field=args.field_name,
                active_pbrs_field_source="field_round",
                candidate_value=candidate_value,
                active_checkpoint_context=checkpoint,
                active_field_carryover_context=None,
                candidate_selection_context={
                    "field_round": args.field_name,
                    "stage_label": stage_label,
                    "candidate_values": candidate_values,
                },
                phase_name="stage_conditioned_branch",
                save_model=False,
                save_model_interval=baseline_run_config.get("save_model_interval"),
                local_results_path=baseline_run_config.get("local_results_path"),
                override_env_args=override_env_args,
                override_overrides=override_overrides,
            )
            if args.launch_mode == "blocking":
                result = launcher.run_prebuilt_train_config(plan["resume_train_config"])
                branch_result = {
                    "train_config": result["train_config"],
                    "run_reference": result["run_reference"],
                    "metrics_summary": result.get("metrics_summary"),
                }
                branch_result_path.write_text(
                    json.dumps(branch_result, indent=2, sort_keys=True, ensure_ascii=False),
                    encoding="utf-8",
                )
                launched.append(
                    {
                        "stage_label": stage_label,
                        "candidate_value": candidate_value,
                        "candidate_id": candidate_id,
                        "mode": "blocking",
                        "run_id": (result.get("run_reference") or {}).get("run_id"),
                    }
                )
            else:
                launch_result = launcher.launch_prebuilt_train_config_background(
                    plan["resume_train_config"],
                    stdout_path=candidate_dir / "manual_launch.log",
                )
                launched.append(
                    {
                        "stage_label": stage_label,
                        "candidate_value": candidate_value,
                        "candidate_id": candidate_id,
                        "mode": "background",
                        "pid": launch_result["pid"],
                        "stdout_path": launch_result["stdout_path"],
                    }
                )

    payload = {
        "workflow_id": args.workflow_id,
        "field_name": args.field_name,
        "round_id": int(args.round_id),
        "launched": launched,
        "skipped": skipped,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _launch_missing_candidates(args)


if __name__ == "__main__":
    main()
