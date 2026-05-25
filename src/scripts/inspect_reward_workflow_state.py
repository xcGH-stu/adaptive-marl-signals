from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "results" / "llm_reward_workflows"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _infer_reward_paradigm(manifest: Dict[str, Any]) -> str:
    reward_paradigm = manifest.get("reward_paradigm")
    if isinstance(reward_paradigm, str):
        return reward_paradigm
    workflow_spec = manifest.get("workflow_spec") or {}
    reward_paradigm = workflow_spec.get("reward_paradigm")
    if isinstance(reward_paradigm, str):
        return reward_paradigm
    if isinstance(workflow_spec.get("pbrs_tuning"), dict):
        return "pbrs"
    return "heuristic"


def _collect_round_state(workflow_dir: Path, round_id: int) -> Dict[str, Any]:
    round_dir = workflow_dir / f"round_{round_id:02d}"
    status = _load_json(round_dir / "status.json") or {
        "round_id": round_id,
        "status": "missing",
    }
    candidate_manifest = _load_json(round_dir / "candidate_manifest.json") or {}
    candidate_results = _load_json(round_dir / "candidate_results.json") or {}
    train_metrics = _load_json(round_dir / "train_metrics_summary.json")
    validation_payload = _load_json(round_dir / "validation_candidate_results.json") or {}
    candidate_metadata = candidate_manifest.get("metadata") or {}

    active_pbrs_field = status.get("active_pbrs_field")
    active_pbrs_field_source = status.get("active_pbrs_field_source")
    if active_pbrs_field is None:
        active_pbrs_field = candidate_metadata.get("active_pbrs_field")
    if active_pbrs_field_source is None:
        active_pbrs_field_source = candidate_metadata.get("active_pbrs_field_source")
    if active_pbrs_field is None and isinstance(candidate_manifest.get("candidates"), list):
        first_candidate = next(
            (item for item in candidate_manifest["candidates"] if isinstance(item, dict)),
            None,
        )
        if first_candidate is not None:
            active_pbrs_field = first_candidate.get("active_pbrs_field")
            if active_pbrs_field_source is None:
                active_pbrs_field_source = first_candidate.get("active_pbrs_field_source")

    active_checkpoint_context = status.get("active_checkpoint_context")
    if active_checkpoint_context is None:
        active_checkpoint_context = candidate_metadata.get("active_checkpoint_context")
    active_field_carryover_context = status.get("active_field_carryover_context")
    candidate_selection_context = status.get("candidate_selection_context")
    if candidate_selection_context is None:
        candidate_selection_context = candidate_metadata.get("candidate_selection_context")

    candidate_count = 0
    completed_candidate_count = 0
    if isinstance(candidate_manifest.get("candidates"), list):
        candidate_count = len(candidate_manifest["candidates"])
    if isinstance(candidate_results.get("candidate_results"), list):
        completed_candidate_count = len(candidate_results["candidate_results"])
        if candidate_count == 0:
            candidate_count = completed_candidate_count
    elif train_metrics is not None:
        completed_candidate_count = 1
        if candidate_count == 0:
            candidate_count = 1

    return {
        "round_id": round_id,
        "status": status.get("status"),
        "reason": status.get("reason"),
        "stage": status.get("stage"),
        "reward_paradigm": status.get("reward_paradigm"),
        "active_pbrs_field": active_pbrs_field,
        "active_pbrs_field_source": active_pbrs_field_source,
        "active_checkpoint_context": active_checkpoint_context,
        "active_field_carryover_context": active_field_carryover_context,
        "candidate_selection_context": candidate_selection_context,
        "critic_structured_diagnosis": status.get("critic_structured_diagnosis"),
        "critic_verdict": status.get("critic_verdict"),
        "critic_best_candidate": status.get("critic_best_candidate"),
        "critic_whether_to_full_run": status.get("critic_whether_to_full_run"),
        "search_strategy_recommendation": status.get("search_strategy_recommendation"),
        "validation_skipped": status.get("validation_skipped"),
        "validation_skip_reason": status.get("validation_skip_reason"),
        "validation_summary": validation_payload.get("validation_summary"),
        "candidate_count": candidate_count,
        "completed_candidate_count": completed_candidate_count,
        "reused_candidate_count": status.get("reused_candidate_count"),
        "rerun_candidate_count": status.get("rerun_candidate_count"),
        "has_generator_response": (round_dir / "generator_response.json").exists(),
        "has_critic_response": (round_dir / "critic_response.json").exists(),
        "has_train_metrics_summary": train_metrics is not None,
    }


def inspect_workflow(workflow_dir: Path) -> Dict[str, Any]:
    manifest = _load_json(workflow_dir / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"manifest.json not found under {workflow_dir}")

    workflow_id = manifest.get("workflow_id", workflow_dir.name)
    round_count = int(manifest.get("round_count", 0) or 0)
    rounds = [
        _collect_round_state(workflow_dir, round_id)
        for round_id in range(1, round_count + 1)
    ]

    latest_completed_round = 0
    latest_existing_round = 0
    failed_round = None
    for round_state in rounds:
        latest_existing_round = max(latest_existing_round, int(round_state["round_id"]))
        if round_state.get("status") == "completed":
            latest_completed_round = max(latest_completed_round, int(round_state["round_id"]))
        if round_state.get("status") == "failed" and failed_round is None:
            failed_round = round_state

    current_round = manifest.get("current_round")
    if current_round is None and failed_round is not None:
        current_round = failed_round["round_id"]

    resume_round = None
    resume_reason = None
    if failed_round is not None:
        resume_round = int(failed_round["round_id"])
        resume_reason = (
            f"workflow failed in round {resume_round} stage={failed_round.get('stage') or '-'}"
        )
    elif manifest.get("status") != "completed" and current_round is not None:
        resume_round = int(current_round)
        resume_reason = f"workflow status is {manifest.get('status')}"
    elif latest_completed_round > 0 and latest_completed_round < int(manifest.get("max_rounds", 0) or 0):
        resume_round = latest_completed_round + 1
        resume_reason = "workflow stopped before max_rounds completed"

    return {
        "workflow_id": workflow_id,
        "workflow_dir": str(workflow_dir),
        "status": manifest.get("status"),
        "failure_reason": manifest.get("failure_reason"),
        "reward_paradigm": _infer_reward_paradigm(manifest),
        "current_round": current_round,
        "round_count": round_count,
        "max_rounds": manifest.get("max_rounds"),
        "latest_completed_round": latest_completed_round,
        "resume_plan": {
            "suggested_start_round": resume_round,
            "reason": resume_reason,
        },
        "rounds": rounds,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def print_text_report(report: Dict[str, Any]) -> None:
    print(
        f"{report['workflow_id']}: status={report['status']} "
        f"paradigm={report['reward_paradigm']} current_round={_fmt(report['current_round'])} "
        f"latest_completed_round={report['latest_completed_round']}"
    )
    if report.get("failure_reason"):
        print(f"  failure_reason={report['failure_reason']}")
    resume_plan = report.get("resume_plan") or {}
    if resume_plan.get("suggested_start_round") is not None:
        print(
            "  "
            f"suggested_start_round={resume_plan['suggested_start_round']} "
            f"reason={resume_plan.get('reason') or '-'}"
        )
    for round_state in report["rounds"]:
        print(
            "  "
            f"round_{round_state['round_id']:02d}: "
            f"status={_fmt(round_state['status'])} "
            f"stage={_fmt(round_state['stage'])} "
            f"active={_fmt(round_state['active_pbrs_field'])} "
            f"active_source={_fmt(round_state['active_pbrs_field_source'])} "
            f"field_carryover={_fmt((round_state['active_field_carryover_context'] or {}).get('previous_switch_field_applied') if isinstance(round_state['active_field_carryover_context'], dict) else None)} "
            f"candidate_source={_fmt((round_state['candidate_selection_context'] or {}).get('candidate_value_source') if isinstance(round_state['candidate_selection_context'], dict) else None)} "
            f"carryover={_fmt((round_state['candidate_selection_context'] or {}).get('previous_search_strategy_applied') if isinstance(round_state['candidate_selection_context'], dict) else None)} "
            f"checkpoint={_fmt((round_state['active_checkpoint_context'] or {}).get('name') if isinstance(round_state['active_checkpoint_context'], dict) else None)} "
            f"verdict={_fmt(round_state['critic_verdict'])} "
            f"best_candidate={_fmt((round_state['critic_best_candidate'] or {}).get('candidate_id') if isinstance(round_state['critic_best_candidate'], dict) else None)} "
            f"full_run={_fmt(round_state['critic_whether_to_full_run'])} "
            f"search_action={_fmt((round_state['search_strategy_recommendation'] or {}).get('candidate_value_action') if isinstance(round_state['search_strategy_recommendation'], dict) else None)} "
            f"validation_skipped={_fmt(round_state['validation_skipped'])} "
            f"validation_skip_reason={_fmt(round_state['validation_skip_reason'] or ((round_state['validation_summary'] or {}).get('skip_reason') if isinstance(round_state['validation_summary'], dict) else None))} "
            f"candidates={round_state['completed_candidate_count']}/{round_state['candidate_count']} "
            f"reused={_fmt(round_state['reused_candidate_count'])} "
            f"rerun={_fmt(round_state['rerun_candidate_count'])} "
            f"generator={round_state['has_generator_response']} "
            f"critic={round_state['has_critic_response']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing llm_reward_workflows results.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    args = parser.parse_args()

    workflow_dir = Path(args.results_root) / args.workflow_id
    report = inspect_workflow(workflow_dir)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print_text_report(report)


if __name__ == "__main__":
    main()
