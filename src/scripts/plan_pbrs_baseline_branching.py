from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rewarding.spec_renderer import render_reward_code_from_spec
from rewarding.spec_schema import validate_reward_spec
from scripts.run_reward_workflow import build_default_workflow_spec
from scripts.select_pbrs_checkpoints import build_baseline_summary
from workflows.clients.checkpoint_selector_client import (
    StubCheckpointSelectorBackend,
    TemplateBasedCheckpointSelectorClient,
)
from workflows.clients.openai_backend import OpenAIChatBackend
from workflows.pbrs_round_preview import resolve_round_preview_state
from workflows.reward_workflow import RewardWorkflow
from workflows.storage import WorkflowStorage
from workflows.train_launcher import EPyMARLTrainLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-plan-id", required=True)
    parser.add_argument("--baseline-metrics-json", required=True)
    parser.add_argument("--baseline-run-config-json", required=True)
    parser.add_argument("--baseline-run-info-json", required=True)
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
    parser.add_argument("--results-root", default=str(ROOT / "results" / "pbrs_branch_plans"))
    parser.add_argument(
        "--workflow-results-root",
        default=str(ROOT / "results" / "llm_reward_workflows"),
        help="Artifact root for any future workflow-aligned outputs.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch the checkpoint branch training runs. Default behavior is plan-only.",
    )
    parser.add_argument(
        "--reuse-completed-candidates",
        action="store_true",
        help="When executing, reuse existing branch_result.json files instead of rerunning completed candidates.",
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
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser


def _build_llm_backend(args: argparse.Namespace):
    if args.use_real_llm:
        return OpenAIChatBackend(
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            retry_backoff_seconds=args.llm_retry_backoff,
        )
    return StubCheckpointSelectorBackend()


def _load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_read_only_workflow_for_spec(
    *,
    workflow_id: str,
    workflow_spec: Dict[str, Any],
    storage_root: str,
) -> RewardWorkflow:
    workflow = object.__new__(RewardWorkflow)
    workflow.workflow_id = workflow_id
    workflow.workflow_spec = workflow_spec
    workflow.generator_client = None
    workflow.critic_client = None
    workflow.train_launcher = None
    workflow.storage_root = storage_root
    workflow.storage = WorkflowStorage(workflow_id=workflow_id, root_dir=storage_root)
    return workflow


def _checkpoint_root_and_steps_from_baseline(
    *,
    baseline_run_config: Dict[str, Any],
    baseline_run_info: Dict[str, Any],
) -> tuple[str, List[int]]:
    model_root_path = baseline_run_info.get("model_root_path")
    if not isinstance(model_root_path, str) or not model_root_path:
        raise ValueError("baseline run info does not contain model_root_path")
    saved_model_steps = baseline_run_info.get("saved_model_steps")
    steps = []
    if isinstance(saved_model_steps, list):
        for value in saved_model_steps:
            try:
                steps.append(int(value))
            except (TypeError, ValueError):
                continue
    steps = sorted(set(steps))
    if not steps:
        raise ValueError("baseline run info does not contain any saved_model_steps")
    return model_root_path, steps


def _round_to_checkpoint_mapping(selected_checkpoints: List[Dict[str, Any]]) -> Dict[str, str]:
    return {
        str(index): str(checkpoint["name"])
        for index, checkpoint in enumerate(selected_checkpoints, start=1)
    }


def _align_selected_checkpoints_to_available_steps(
    *,
    selected_checkpoints: List[Dict[str, Any]],
    available_steps: List[int],
) -> List[Dict[str, Any]]:
    if not selected_checkpoints:
        return []
    sorted_steps = sorted({int(step) for step in available_steps})
    if not sorted_steps:
        raise ValueError("no available checkpoint steps were provided")

    if len(selected_checkpoints) == 1:
        aligned = deepcopy(selected_checkpoints)
        aligned[0]["requested_step"] = aligned[0].get("step")
        aligned[0]["step"] = int(sorted_steps[-1])
        aligned[0]["resolved_checkpoint_step"] = int(sorted_steps[-1])
        return aligned

    use_rank_alignment = False
    requested_steps: List[int] = []
    for checkpoint in selected_checkpoints:
        try:
            requested_steps.append(int(checkpoint.get("step")))
        except (TypeError, ValueError):
            requested_steps.append(0)
    if requested_steps and max(requested_steps) > max(sorted_steps) * 2:
        use_rank_alignment = True

    aligned = []
    used_indices = set()
    for index, checkpoint in enumerate(selected_checkpoints):
        aligned_checkpoint = deepcopy(checkpoint)
        requested_step = requested_steps[index]
        if use_rank_alignment:
            if len(selected_checkpoints) == 1:
                target_index = len(sorted_steps) - 1
            else:
                fraction = index / max(1, len(selected_checkpoints) - 1)
                target_index = round(fraction * (len(sorted_steps) - 1))
        else:
            target_index = min(
                range(len(sorted_steps)),
                key=lambda step_index: abs(sorted_steps[step_index] - requested_step),
            )
        while target_index in used_indices and target_index < len(sorted_steps) - 1:
            target_index += 1
        while target_index in used_indices and target_index > 0:
            target_index -= 1
        used_indices.add(target_index)
        resolved_step = int(sorted_steps[target_index])
        aligned_checkpoint["requested_step"] = aligned_checkpoint.get("step")
        aligned_checkpoint["step"] = resolved_step
        aligned_checkpoint["resolved_checkpoint_step"] = resolved_step
        aligned.append(aligned_checkpoint)
    return aligned


def _build_candidate_reward_spec(
    *,
    workflow: RewardWorkflow,
    active_pbrs_field: str,
    candidate_value: float,
) -> Dict[str, Any]:
    reward_spec = workflow._build_initial_reward_spec()
    reward_spec = deepcopy(reward_spec)
    reward_spec.setdefault("pbrs", {})
    reward_spec["pbrs"][active_pbrs_field] = float(candidate_value)
    return validate_reward_spec(reward_spec)


def _write_candidate_reward_artifacts(
    *,
    output_dir: Path,
    candidate_id: str,
    reward_spec: Dict[str, Any],
) -> Dict[str, str]:
    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    reward_spec_path = candidate_dir / "reward_spec.json"
    reward_function_path = candidate_dir / "reward_function.py"
    reward_spec_path.write_text(
        json.dumps(reward_spec, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    reward_function_path.write_text(
        render_reward_code_from_spec(reward_spec),
        encoding="utf-8",
    )
    return {
        "candidate_dir": str(candidate_dir),
        "reward_spec_path": str(reward_spec_path),
        "reward_function_path": str(reward_function_path),
    }


def _load_existing_branch_result(candidate_dir: str) -> Optional[Dict[str, Any]]:
    branch_result_path = Path(candidate_dir) / "branch_result.json"
    if not branch_result_path.exists():
        return None
    with branch_result_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    train_config = payload.get("train_config")
    run_reference = payload.get("run_reference")
    metrics_summary = payload.get("metrics_summary")
    if not isinstance(train_config, dict):
        return None
    if not isinstance(run_reference, dict):
        return None
    return {
        "payload": payload,
        "path": str(branch_result_path),
        "metrics_summary": metrics_summary,
    }


def build_branching_plan(args: argparse.Namespace) -> Dict[str, Any]:
    preferred_metrics = args.preferred_metric or [
        "test_sparse_return_mean",
        "sparse_return_mean",
        "test_return_mean",
        "return_mean",
    ]
    baseline_summary = build_baseline_summary(
        metrics_json_path=Path(args.baseline_metrics_json),
        run_config_path=Path(args.baseline_run_config_json),
        preferred_metrics=preferred_metrics,
        max_points_per_metric=args.max_points_per_metric,
    )
    workflow_spec = build_default_workflow_spec(args.max_rounds, reward_paradigm="pbrs")
    workflow_spec["workflow_id"] = args.branch_plan_id

    selection_constraints = {
        "min_count": args.min_count,
        "max_count": args.max_count,
        "min_step_gap": args.min_step_gap,
    }
    selector = TemplateBasedCheckpointSelectorClient(llm_backend=_build_llm_backend(args))
    selection_result = selector.select_checkpoints(
        workflow_spec=workflow_spec,
        baseline_summary=baseline_summary,
        selection_constraints=selection_constraints,
    )
    selection = selection_result["selection"]
    selected_checkpoints = list(selection.get("selected_checkpoints", []))

    workflow_spec["pbrs_checkpoint_plan"] = {
        "selected_checkpoints": selected_checkpoints,
        "round_to_checkpoint": _round_to_checkpoint_mapping(selected_checkpoints),
    }

    workflow = _build_read_only_workflow_for_spec(
        workflow_id=args.branch_plan_id,
        workflow_spec=workflow_spec,
        storage_root=args.workflow_results_root,
    )
    launcher = EPyMARLTrainLauncher(repo_root=ROOT, python_executable=args.python_executable)
    baseline_run_config = _load_json(args.baseline_run_config_json)
    baseline_run_info = _load_json(args.baseline_run_info_json)
    checkpoint_root_dir, available_steps = _checkpoint_root_and_steps_from_baseline(
        baseline_run_config=baseline_run_config,
        baseline_run_info=baseline_run_info,
    )
    selected_checkpoints = _align_selected_checkpoints_to_available_steps(
        selected_checkpoints=selected_checkpoints,
        available_steps=available_steps,
    )
    workflow_spec["pbrs_checkpoint_plan"] = {
        "selected_checkpoints": selected_checkpoints,
        "round_to_checkpoint": _round_to_checkpoint_mapping(selected_checkpoints),
    }
    baseline_run_reference = {
        "checkpoint_root_dir": checkpoint_root_dir,
        "available_checkpoint_steps": available_steps,
    }

    output_root = Path(args.results_root) / args.branch_plan_id
    output_root.mkdir(parents=True, exist_ok=True)

    rounds: List[Dict[str, Any]] = []
    executed_branch_candidate_count = 0
    reused_branch_candidate_count = 0
    planned_only_branch_candidate_count = 0
    for round_index, checkpoint in enumerate(selected_checkpoints[: args.max_rounds], start=1):
        resolved = resolve_round_preview_state(
            workflow,
            round_id=round_index,
            previous_round=None,
        )
        round_output_dir = output_root / f"round_{round_index:02d}"
        round_output_dir.mkdir(parents=True, exist_ok=True)
        branch_candidates = []
        for candidate in resolved["candidate_previews"]:
            candidate_id = str(candidate["candidate_id"])
            candidate_value = candidate.get("candidate_value")
            active_pbrs_field = resolved["active_pbrs_field"]
            reward_spec = _build_candidate_reward_spec(
                workflow=workflow,
                active_pbrs_field=active_pbrs_field,
                candidate_value=float(candidate_value),
            )
            reward_artifacts = _write_candidate_reward_artifacts(
                output_dir=round_output_dir,
                candidate_id=candidate_id,
                reward_spec=reward_spec,
            )
            branch_plan = launcher.build_checkpoint_resume_plan(
                base_train_config=baseline_run_config,
                run_reference=baseline_run_reference,
                requested_step=int(checkpoint["step"]),
                label_suffix=f"_{checkpoint['name']}_{candidate_id}",
                use_dense_reward=True,
                reward_module_path=reward_artifacts["reward_function_path"],
                alpha_policy_config={"type": "constant", "value": 1.0},
                workflow_id=args.branch_plan_id,
                workflow_round=round_index,
                reward_paradigm="pbrs",
                active_pbrs_field=resolved["active_pbrs_field"],
                active_pbrs_field_source=resolved["active_pbrs_field_source"],
                candidate_value=float(candidate_value),
                active_checkpoint_context=resolved["active_checkpoint_context"],
                active_field_carryover_context=resolved["active_field_carryover_context"],
                candidate_selection_context=resolved["candidate_selection_context"],
                phase_name="checkpoint_branch",
                save_model=True,
                save_model_interval=baseline_run_config.get("save_model_interval"),
                local_results_path=baseline_run_config.get("local_results_path"),
            )
            candidate_payload = {
                "candidate_id": candidate_id,
                "candidate_value": candidate_value,
                "reward_artifacts": reward_artifacts,
                "branch_plan": branch_plan,
            }
            if args.execute:
                existing_branch_result = None
                if args.reuse_completed_candidates:
                    existing_branch_result = _load_existing_branch_result(
                        reward_artifacts["candidate_dir"]
                    )
                if existing_branch_result is not None:
                    candidate_payload["branch_result"] = existing_branch_result["payload"]
                    candidate_payload["branch_result_path"] = existing_branch_result["path"]
                    candidate_payload["branch_result_status"] = "reused"
                    reused_branch_candidate_count += 1
                else:
                    result = launcher.run_prebuilt_train_config(
                        branch_plan["resume_train_config"]
                    )
                    candidate_result_payload = {
                        "train_config": result["train_config"],
                        "run_reference": result["run_reference"],
                        "metrics_summary": result.get("metrics_summary"),
                    }
                    candidate_result_path = (
                        Path(reward_artifacts["candidate_dir"]) / "branch_result.json"
                    )
                    candidate_result_path.write_text(
                        json.dumps(
                            candidate_result_payload,
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    candidate_payload["branch_result"] = candidate_result_payload
                    candidate_payload["branch_result_path"] = str(candidate_result_path)
                    candidate_payload["branch_result_status"] = "executed"
                    executed_branch_candidate_count += 1
            else:
                candidate_payload["branch_result_status"] = "planned_only"
                planned_only_branch_candidate_count += 1
            branch_candidates.append(
                {
                    **candidate_payload,
                }
            )

        rounds.append(
            {
                "round_id": round_index,
                "checkpoint": checkpoint,
                "resolved_round_state": resolved,
                "branch_candidates": branch_candidates,
            }
        )

    manifest = {
        "branch_plan_id": args.branch_plan_id,
        "selection_constraints": selection_constraints,
        "baseline_summary": baseline_summary,
        "checkpoint_selection": selection,
        "workflow_spec": workflow_spec,
        "baseline_run_config": baseline_run_config,
        "baseline_run_info": baseline_run_info,
        "baseline_checkpoint_root_dir": checkpoint_root_dir,
        "baseline_available_checkpoint_steps": available_steps,
        "branch_execution_summary": {
            "execute_requested": bool(args.execute),
            "reuse_completed_candidates": bool(args.reuse_completed_candidates),
            "executed_branch_candidate_count": executed_branch_candidate_count,
            "reused_branch_candidate_count": reused_branch_candidate_count,
            "planned_only_branch_candidate_count": planned_only_branch_candidate_count,
        },
        "rounds": rounds,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def print_text_manifest(manifest: Dict[str, Any]) -> None:
    print(
        f"{manifest['branch_plan_id']}: checkpoints={len(manifest.get('checkpoint_selection', {}).get('selected_checkpoints', []))} "
        f"rounds={len(manifest.get('rounds', []))}"
    )
    print(
        "  "
        f"baseline_checkpoint_root={_fmt(manifest.get('baseline_checkpoint_root_dir'))} "
        f"available_steps={json.dumps(manifest.get('baseline_available_checkpoint_steps', []), ensure_ascii=False)}"
    )
    execution_summary = manifest.get("branch_execution_summary") or {}
    print(
        "  "
        f"execute_requested={_fmt(execution_summary.get('execute_requested'))} "
        f"executed={_fmt(execution_summary.get('executed_branch_candidate_count'))} "
        f"reused={_fmt(execution_summary.get('reused_branch_candidate_count'))} "
        f"planned_only={_fmt(execution_summary.get('planned_only_branch_candidate_count'))}"
    )
    for round_summary in manifest.get("rounds", []):
        resolved = round_summary.get("resolved_round_state") or {}
        checkpoint = round_summary.get("checkpoint") or {}
        candidate_context = resolved.get("candidate_selection_context") or {}
        print(
            "  "
            f"round_{round_summary['round_id']:02d}: "
            f"checkpoint={_fmt(checkpoint.get('name'))}@{_fmt(checkpoint.get('step'))} "
            f"stage={_fmt(checkpoint.get('stage_label'))} "
            f"active={_fmt(resolved.get('active_pbrs_field'))} "
            f"candidate_source={_fmt(candidate_context.get('candidate_value_source'))} "
            f"candidates={len(round_summary.get('branch_candidates', []))}"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    manifest = build_branching_plan(args)
    if args.format == "json":
        print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print_text_manifest(manifest)


if __name__ == "__main__":
    main()
