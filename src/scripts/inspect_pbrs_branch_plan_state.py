from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.summarize_pbrs_branch_plan import summarize_branch_plan


DEFAULT_BRANCH_PLAN_ROOT = ROOT / "results" / "pbrs_branch_plans"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-plan-id", required=True)
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_BRANCH_PLAN_ROOT),
        help="Root directory containing pbrs_branch_plans.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser


def _candidate_status(record: Dict[str, Any]) -> str:
    status = record.get("status")
    return str(status) if status is not None else "planned_only"


def inspect_branch_plan(branch_plan_dir: Path) -> Dict[str, Any]:
    baseline_readiness_path = branch_plan_dir / "baseline_readiness.json"
    baseline_readiness = {}
    if baseline_readiness_path.exists():
        with baseline_readiness_path.open("r", encoding="utf-8") as handle:
            baseline_readiness = json.load(handle)
    blocked_path = branch_plan_dir / "branching_blocked.json"
    blocked_payload = {}
    if blocked_path.exists():
        with blocked_path.open("r", encoding="utf-8") as handle:
            blocked_payload = json.load(handle)
    manifest_path = branch_plan_dir / "manifest.json"
    if not manifest_path.exists():
        return {
            "branch_plan_id": branch_plan_dir.name,
            "branch_plan_dir": str(branch_plan_dir),
            "baseline_readiness": baseline_readiness,
            "blocked_payload": blocked_payload,
            "branch_execution_summary": {},
            "branch_resume_summary": {},
            "total_candidates": 0,
            "completed_candidates": 0,
            "incomplete_candidates": 0,
            "rounds": [],
            "resume_plan": {
                "suggested_resume_round": None,
                "suggested_resume_candidate": None,
                "has_incomplete_candidates": False,
            },
            "status": "readiness_only",
        }

    summary = summarize_branch_plan(branch_plan_dir)
    round_reports: List[Dict[str, Any]] = []
    first_incomplete: Optional[Dict[str, Any]] = None

    total_candidates = 0
    completed_candidates = 0
    incomplete_candidates = 0

    for round_summary in summary.get("rounds", []):
        records = round_summary.get("records", [])
        round_total = len(records)
        round_completed = 0
        round_incomplete = 0
        missing_candidate_ids: List[str] = []

        for record in records:
            total_candidates += 1
            status = _candidate_status(record)
            if status in {"executed", "reused"}:
                completed_candidates += 1
                round_completed += 1
            else:
                incomplete_candidates += 1
                round_incomplete += 1
                missing_candidate_ids.append(str(record.get("candidate_id")))
                if first_incomplete is None:
                    first_incomplete = {
                        "round_id": round_summary.get("round_id"),
                        "candidate_id": record.get("candidate_id"),
                    }

        round_reports.append(
            {
                "round_id": round_summary.get("round_id"),
                "checkpoint": round_summary.get("checkpoint") or {},
                "active_pbrs_field": (
                    (round_summary.get("resolved_round_state") or {}).get("active_pbrs_field")
                ),
                "ranking_metric": round_summary.get("ranking_metric"),
                "total_candidates": round_total,
                "completed_candidates": round_completed,
                "incomplete_candidates": round_incomplete,
                "missing_candidate_ids": missing_candidate_ids,
                "status": "complete" if round_incomplete == 0 else "incomplete",
            }
        )

    resume_plan = {
        "suggested_resume_round": (
            first_incomplete.get("round_id") if isinstance(first_incomplete, dict) else None
        ),
        "suggested_resume_candidate": (
            first_incomplete.get("candidate_id") if isinstance(first_incomplete, dict) else None
        ),
        "has_incomplete_candidates": first_incomplete is not None,
    }

    return {
        "branch_plan_id": summary.get("branch_plan_id"),
        "branch_plan_dir": str(branch_plan_dir),
        "baseline_readiness": baseline_readiness,
        "blocked_payload": blocked_payload,
        "branch_execution_summary": summary.get("branch_execution_summary") or {},
        "branch_resume_summary": summary.get("branch_resume_summary") or {},
        "total_candidates": total_candidates,
        "completed_candidates": completed_candidates,
        "incomplete_candidates": incomplete_candidates,
        "rounds": round_reports,
        "resume_plan": resume_plan,
        "status": "planned" if total_candidates > 0 else "empty",
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def print_text_report(report: Dict[str, Any]) -> None:
    baseline_readiness = report.get("baseline_readiness") or {}
    checkpoint_summary = baseline_readiness.get("checkpoint_summary") or {}
    blocked_payload = report.get("blocked_payload") or {}
    print(
        f"{report['branch_plan_id']}: total={report.get('total_candidates')} "
        f"completed={report.get('completed_candidates')} "
        f"incomplete={report.get('incomplete_candidates')}"
    )
    print(f"  status={_fmt(report.get('status'))}")
    if baseline_readiness:
        print(
            "  "
            f"baseline_branching_ready={_fmt(baseline_readiness.get('branching_ready'))} "
            f"baseline_latest_checkpoint_step={_fmt(checkpoint_summary.get('latest_checkpoint_step'))}"
        )
    if blocked_payload:
        print(f"  blocked_reason={_fmt(blocked_payload.get('reason'))}")
    execution_summary = report.get("branch_execution_summary") or {}
    if execution_summary:
        print(
            "  "
            f"execute_requested={_fmt(execution_summary.get('execute_requested'))} "
            f"executed={_fmt(execution_summary.get('executed_branch_candidate_count'))} "
            f"reused={_fmt(execution_summary.get('reused_branch_candidate_count'))} "
            f"planned_only={_fmt(execution_summary.get('planned_only_branch_candidate_count'))}"
        )
    for round_report in report.get("rounds", []):
        checkpoint = round_report.get("checkpoint") or {}
        print(
            "  "
            f"round_{int(round_report.get('round_id')):02d}: "
            f"checkpoint={_fmt(checkpoint.get('name'))}@{_fmt(checkpoint.get('step'))} "
            f"active={_fmt(round_report.get('active_pbrs_field'))} "
            f"completed={_fmt(round_report.get('completed_candidates'))}/{_fmt(round_report.get('total_candidates'))} "
            f"status={_fmt(round_report.get('status'))}"
        )
        missing = round_report.get("missing_candidate_ids") or []
        if missing:
            print(f"    missing={json.dumps(missing, ensure_ascii=False)}")
    resume_plan = report.get("resume_plan") or {}
    print(
        "  "
        f"suggested_resume_round={_fmt(resume_plan.get('suggested_resume_round'))} "
        f"suggested_resume_candidate={_fmt(resume_plan.get('suggested_resume_candidate'))}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    branch_plan_dir = Path(args.results_root) / args.branch_plan_id
    report = inspect_branch_plan(branch_plan_dir)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print_text_report(report)


if __name__ == "__main__":
    main()
