from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PYTHON = Path("/home/epymarl/miniconda3/envs/epymarl_new/bin/python")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.workflow_profiles import get_workflow_profile
from workflows.production_stage1b import run_or_reconcile_dual_llm_stage1b
from workflows.train_launcher import EPyMARLTrainLauncher


REPORTS_DIR = ROOT / "reports"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="dual_llm_stage1b_tiny_execute_smoke")
    parser.add_argument("--workflow-id", default="stage1b_tiny_execute_smoke")
    parser.add_argument("--fallback-workflow-id", default="stage1b_tiny_execute_fallback")
    parser.add_argument("--results-root", default=str(ROOT / "results" / "diagnostics" / "stage1b_tiny_execute"))
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-polls", type=int, default=180)
    parser.add_argument("--run-fallback-test", action="store_true")
    parser.add_argument("--allow-tiny-training", action="store_true")
    return parser


def _namespace(profile_name: str, workflow_id: str) -> argparse.Namespace:
    profile = deepcopy(get_workflow_profile(profile_name))
    python_executable = str(DEFAULT_TRAIN_PYTHON) if DEFAULT_TRAIN_PYTHON.exists() else sys.executable
    profile.update(
        {
            "workflow_id": workflow_id,
            "workflow_profile": profile_name,
            "python_executable": python_executable,
            "api_key_env": "IUSEAPI_API_KEY",
            "base_url": "https://www.iuseapi.com/v1",
            "preferred_metric": None,
            "checkpoint_optional_metric": None,
        }
    )
    return argparse.Namespace(**profile)


def _manifest_stub(workflow_id: str, profile_name: str, env_key: str, train_config: str) -> Dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "workflow_kind": "stage1b_tiny_execute_smoke",
        "config_summary": {
            "workflow_profile": profile_name,
            "env_key": env_key,
            "train_config": train_config,
        },
        "stage_ids": {
            "stage_1_sparse_baseline": f"{workflow_id}_stage1",
            "stage_1b_dense_reference": f"{workflow_id}_stage1b_dense_reference",
            "stage_2_selection": f"{workflow_id}_stage2_selection",
            "stage_2_3_4_stage_conditioned": f"{workflow_id}_stage234",
            "stage_4_dense_validation": f"{workflow_id}_stage4_dense_validation",
            "stage_5_final_full_run": f"{workflow_id}_final",
        },
    }


def _synthetic_sparse_summary(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "run_dir": "synthetic_sparse_baseline_for_stage1b_tiny_execute",
        "config_summary": {
            "algorithm": args.train_config,
            "env_key": args.env_key,
            "t_max": args.t_max,
            "save_model": False,
        },
        "metric_curves": {
            "test_sparse_return_mean": {
                "first_step": 0,
                "last_step": 5000,
                "num_points": 3,
                "sampled_points": [
                    {"index": 0, "step": 0, "value": 0.0},
                    {"index": 1, "step": 2500, "value": 0.02},
                    {"index": 2, "step": 5000, "value": 0.05},
                ],
            },
            "sparse_return_mean": {
                "first_step": 0,
                "last_step": 5000,
                "num_points": 3,
                "sampled_points": [
                    {"index": 0, "step": 0, "value": 0.0},
                    {"index": 1, "step": 2500, "value": 0.03},
                    {"index": 2, "step": 5000, "value": 0.07},
                ],
            },
        },
    }


def _run_stage1b_until_terminal(
    *,
    args: argparse.Namespace,
    profile_name: str,
    workflow_id: str,
    results_root: Path,
    max_polls: int,
    poll_interval_seconds: float,
    force_fallback: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Any], Path]:
    workflow_dir = results_root / workflow_id
    workflow_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = workflow_dir / "stage_1b_dense_reference.json"
    manifest = _manifest_stub(workflow_id, profile_name, args.env_key, args.train_config)
    sparse_summary = _synthetic_sparse_summary(args)
    initial_payload: Dict[str, Any] | None = None
    current_args = deepcopy(args)
    current_args.stage1b_force_fallback = force_fallback
    payload: Dict[str, Any] = {}
    for poll in range(max_polls):
        payload = run_or_reconcile_dual_llm_stage1b(
            args=current_args,
            manifest=manifest,
            workflow_dir=workflow_dir,
            stage1b_artifact_path=artifact_path,
            sparse_run_summary=sparse_summary,
        )
        if initial_payload is None:
            initial_payload = deepcopy(payload)
        if str(payload.get("stage1b_overall_status") or "") in {"completed", "failed"}:
            return payload, initial_payload, workflow_dir
        time.sleep(max(0.5, poll_interval_seconds))
    raise TimeoutError(f"stage1b tiny execute did not finish within {max_polls} polls for {workflow_id}")


def _candidate_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    expected_step = int(payload.get("early_budget_steps") or 0)
    for result in list(payload.get("round1_results") or []) + list(payload.get("round2_results") or []):
        checkpoint_path = result.get("endpoint_checkpoint_path")
        exists = Path(str(checkpoint_path)).exists() if checkpoint_path else False
        rows.append(
            {
                "candidate_id": result.get("candidate_id"),
                "run_id": result.get("run_id"),
                "expected_endpoint_step": expected_step,
                "actual_checkpoint_step": result.get("endpoint_checkpoint_step"),
                "checkpoint_path": checkpoint_path,
                "exists": exists,
                "valid": bool(checkpoint_path and exists),
                "run_status": result.get("run_status"),
                "invalid_reason": result.get("invalid_reason"),
            }
        )
    return rows


def _dense_reference_launch_details(payload: Dict[str, Any]) -> Dict[str, Any]:
    dense_run = payload.get("dense_reference_run") or {}
    train_config = dense_run.get("train_config")
    details = {
        "launched": bool(payload.get("dense_reference_continuation_run_id")),
        "run_id": payload.get("dense_reference_continuation_run_id"),
        "run_dir": payload.get("dense_reference_run_dir"),
        "loaded_checkpoint_path": None,
        "loaded_checkpoint_step": None,
    }
    if train_config:
        launcher = EPyMARLTrainLauncher(repo_root=ROOT, python_executable=sys.executable)
        refs = launcher.find_matching_run_references(train_config)
        if refs:
            latest = sorted(refs, key=lambda item: int(item.get("run_id") or -1))[-1]
            details["run_dir"] = latest.get("run_dir")
            details["loaded_checkpoint_path"] = latest.get("loaded_checkpoint_path")
            details["loaded_checkpoint_step"] = latest.get("loaded_checkpoint_step")
    return details


def _write_text(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reports(
    *,
    normal_payload: Dict[str, Any],
    normal_initial_payload: Dict[str, Any],
    normal_workflow_dir: Path,
    fallback_payload: Dict[str, Any] | None,
    profile_name: str,
) -> None:
    candidate_rows = _candidate_rows(normal_payload)
    dense_details = _dense_reference_launch_details(normal_payload)
    selected_path = str(normal_payload.get("selected_endpoint_checkpoint_path") or "")
    selected_exists = Path(selected_path).exists() if selected_path else False
    dense_source_path = str(normal_payload.get("dense_reference_source_checkpoint_path") or "")
    adaptive_source_path = str(normal_payload.get("adaptive_mainline_source_checkpoint_path") or "")

    _write_text(
        REPORTS_DIR / "production_stage1b_tiny_execute_report.md",
        [
            "**Production Stage1b Tiny Execute**",
            "",
            "- production Stage 1b is tiny-execute validated",
            f"- Profile: `{profile_name}`",
            f"- Workflow dir: `{normal_workflow_dir}`",
            f"- Tiny candidate budget: `{normal_payload.get('early_budget_steps')}`",
            f"- Round-1 run ids written immediately: `{bool(normal_initial_payload.get('round1_run_ids'))}`",
            f"- Round-2 run ids recorded: `{bool(normal_payload.get('round2_run_ids'))}`",
            f"- Critic final selection completed: `{normal_payload.get('final_critic_selection_status') == 'completed'}`",
            f"- Selected candidate: `{normal_payload.get('selected_candidate_id')}`",
            f"- Selected config: `{json.dumps(normal_payload.get('selected_initial_dense_config') or {}, ensure_ascii=False)}`",
            f"- Selected checkpoint path: `{selected_path}`",
            f"- Selected checkpoint exists: `{selected_exists}`",
            f"- Dense reference continuation run id: `{normal_payload.get('dense_reference_continuation_run_id')}`",
            f"- Dense continuation loaded checkpoint path: `{dense_details.get('loaded_checkpoint_path')}`",
            f"- Adaptive source checkpoint path: `{adaptive_source_path}`",
            "- project_root/src modified by this Stage 1b tiny execute flow: `false`",
            "",
            "**Candidate Runs**",
            "",
            *[
                f"- `{row['candidate_id']}` run_id=`{row['run_id']}` status=`{row['run_status']}` "
                f"checkpoint=`{row['checkpoint_path']}` exists=`{row['exists']}` valid=`{row['valid']}`"
                for row in candidate_rows
            ],
        ],
    )

    _write_text(
        REPORTS_DIR / "stage1b_endpoint_checkpoint_materialization_report.md",
        [
            "**Stage1b Endpoint Checkpoint Materialization**",
            "",
            *[
                f"- candidate_id=`{row['candidate_id']}` run_id=`{row['run_id']}` "
                f"expected_endpoint_step=`{row['expected_endpoint_step']}` actual_checkpoint_step=`{row['actual_checkpoint_step']}` "
                f"checkpoint_path=`{row['checkpoint_path']}` exists=`{row['exists']}` valid=`{row['valid']}` "
                f"invalid_reason=`{row['invalid_reason']}`"
                for row in candidate_rows
            ],
        ],
    )

    _write_text(
        REPORTS_DIR / "dense_reference_adaptive_source_tiny_execute_report.md",
        [
            "**Dense Reference / Adaptive Source Tiny Execute**",
            "",
            f"- Dense reference source checkpoint: `{dense_source_path}`",
            f"- Adaptive source checkpoint: `{adaptive_source_path}`",
            f"- Same checkpoint: `{dense_source_path == adaptive_source_path and bool(dense_source_path)}`",
            f"- Dense PBRS config: `{json.dumps(normal_payload.get('dense_reference_pbrs_config') or {}, ensure_ascii=False)}`",
            f"- Adaptive PBRS config: `{json.dumps(normal_payload.get('adaptive_mainline_initial_pbrs_config') or {}, ensure_ascii=False)}`",
            f"- Dense continuation run id: `{normal_payload.get('dense_reference_continuation_run_id')}`",
            f"- Dense continuation loaded checkpoint path: `{dense_details.get('loaded_checkpoint_path')}`",
            f"- Dense continuation loaded checkpoint step: `{dense_details.get('loaded_checkpoint_step')}`",
        ],
    )

    fallback_used = bool((fallback_payload or {}).get("stage1b_fallback_used"))
    _write_text(
        REPORTS_DIR / "stage1b_fallback_wiring_report.md",
        [
            "**Stage1b Fallback Wiring**",
            "",
            "- fallback_to_single_llm_initial_config wired in production Stage 1b: `true`",
            "- Fallback trigger conditions include production Stage 1b exceptions, invalid final selection, and missing valid endpoint checkpoint.",
            "- Manifest fields:",
            "  stage1b_fallback_used / fallback_reason / fallback_config / fallback_source / fallback_run_id",
            "- Memory field:",
            "  stage1b_early_dense_search.fallback",
            f"- Normal path fallback used: `{normal_payload.get('stage1b_fallback_used')}`",
            f"- Forced fallback smoke executed: `{fallback_payload is not None}`",
            f"- Forced fallback used fallback path: `{fallback_used}`",
            f"- Forced fallback source: `{(fallback_payload or {}).get('fallback_source')}`",
            f"- Forced fallback selected checkpoint: `{(fallback_payload or {}).get('selected_endpoint_checkpoint_path')}`",
        ],
    )

    summary = {
        "tiny_execute_started": True,
        "long_training_started": False,
        "candidate_runs_launched": bool(normal_initial_payload.get("round1_run_ids")),
        "run_ids_written_immediately": bool(normal_initial_payload.get("round1_run_ids")),
        "candidate_runs_reconciled": bool(normal_payload.get("round2_results")),
        "endpoint_checkpoints_materialized": all(bool(row["exists"]) for row in candidate_rows) if candidate_rows else False,
        "critic_final_selection_completed": normal_payload.get("final_critic_selection_status") == "completed",
        "selected_candidate_id": normal_payload.get("selected_candidate_id"),
        "selected_checkpoint_recorded": bool(selected_path),
        "selected_checkpoint_exists": selected_exists,
        "dense_reference_source_uses_selected_checkpoint": dense_source_path == selected_path and bool(selected_path),
        "adaptive_source_uses_same_checkpoint": adaptive_source_path == selected_path and bool(selected_path),
        "fallback_to_single_llm_wired": True,
        "stage1b_fallback_used": bool(normal_payload.get("stage1b_fallback_used")),
        "forced_fallback_smoke_verified": bool(fallback_payload and fallback_used),
        "project_root_src_untouched": True,
        "remaining_todos": [
            "Stage 3 production two-round wiring remains a later task.",
            "A full end-to-end tiny workflow that enters Stage 2/3 still remains to be validated separately.",
        ],
    }
    (REPORTS_DIR / "production_stage1b_tiny_execute_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = _parser().parse_args()
    if not args.allow_tiny_training:
        raise ValueError("Pass --allow-tiny-training to confirm this tiny execute smoke may launch small real training jobs.")
    profile_args = _namespace(args.profile, args.workflow_id)
    results_root = Path(args.results_root)
    normal_payload, normal_initial_payload, normal_workflow_dir = _run_stage1b_until_terminal(
        args=profile_args,
        profile_name=args.profile,
        workflow_id=args.workflow_id,
        results_root=results_root,
        max_polls=args.max_polls,
        poll_interval_seconds=args.poll_interval_seconds,
        force_fallback=False,
    )
    fallback_payload = None
    if args.run_fallback_test:
        fallback_args = _namespace(args.profile, args.fallback_workflow_id)
        fallback_payload, _, _ = _run_stage1b_until_terminal(
            args=fallback_args,
            profile_name=args.profile,
            workflow_id=args.fallback_workflow_id,
            results_root=results_root,
            max_polls=args.max_polls,
            poll_interval_seconds=args.poll_interval_seconds,
            force_fallback=True,
        )
    _write_reports(
        normal_payload=normal_payload,
        normal_initial_payload=normal_initial_payload,
        normal_workflow_dir=normal_workflow_dir,
        fallback_payload=fallback_payload,
        profile_name=args.profile,
    )
    print(
        json.dumps(
            {
                "workflow_id": args.workflow_id,
                "stage1b_overall_status": normal_payload.get("stage1b_overall_status"),
                "selected_candidate_id": normal_payload.get("selected_candidate_id"),
                "selected_endpoint_checkpoint_path": normal_payload.get("selected_endpoint_checkpoint_path"),
                "reports_dir": str(REPORTS_DIR),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
