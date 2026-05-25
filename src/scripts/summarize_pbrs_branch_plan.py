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


DEFAULT_BRANCH_PLAN_ROOT = ROOT / "results" / "pbrs_branch_plans"
ROUND_RANKING_FIELDS = [
    "best_test_sparse_return_mean",
    "last_test_sparse_return_mean",
    "best_test_return_mean",
    "last_test_return_mean",
    "last_mixed_return_mean",
    "last_sparse_return_mean",
    "last_return_mean",
]


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


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_metric_last(summary: Optional[Dict[str, Any]], metric_name: str) -> Any:
    if not isinstance(summary, dict):
        return None
    metric_summary = summary.get("metric_summary", {})
    payload = metric_summary.get(metric_name)
    if not isinstance(payload, dict):
        return None
    return payload.get("last_value")


def _extract_metric_best(summary: Optional[Dict[str, Any]], metric_name: str) -> Any:
    if not isinstance(summary, dict):
        return None
    metric_summary = summary.get("metric_summary", {})
    payload = metric_summary.get(metric_name)
    if not isinstance(payload, dict):
        return None
    return payload.get("best_value")


def _build_candidate_record(
    *,
    branch_plan_id: str,
    round_id: int,
    checkpoint: Dict[str, Any],
    resolved_round_state: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    branch_result = candidate.get("branch_result") or {}
    metrics_summary = branch_result.get("metrics_summary") or {}
    run_reference = branch_result.get("run_reference") or {}
    branch_result_status = candidate.get("branch_result_status")
    if branch_result_status not in {"planned_only", "executed", "reused"}:
        branch_result_status = "finished" if metrics_summary else "planned_only"
    return {
        "branch_plan_id": branch_plan_id,
        "round_id": round_id,
        "checkpoint_name": checkpoint.get("name"),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_requested_step": checkpoint.get("requested_step"),
        "checkpoint_resolved_step": checkpoint.get("resolved_checkpoint_step"),
        "checkpoint_stage_label": checkpoint.get("stage_label"),
        "active_pbrs_field": resolved_round_state.get("active_pbrs_field"),
        "active_pbrs_field_source": resolved_round_state.get("active_pbrs_field_source"),
        "candidate_value_source": (
            (resolved_round_state.get("candidate_selection_context") or {}).get(
                "candidate_value_source"
            )
        ),
        "candidate_id": candidate.get("candidate_id"),
        "candidate_value": candidate.get("candidate_value"),
        "run_id": run_reference.get("run_id"),
        "status": branch_result_status,
        "last_test_sparse_return_mean": _extract_metric_last(
            metrics_summary, "test_sparse_return_mean"
        ),
        "best_test_sparse_return_mean": _extract_metric_best(
            metrics_summary, "test_sparse_return_mean"
        ),
        "last_sparse_return_mean": _extract_metric_last(
            metrics_summary, "sparse_return_mean"
        ),
        "last_mixed_return_mean": _extract_metric_last(
            metrics_summary, "mixed_return_mean"
        ),
        "last_test_return_mean": _extract_metric_last(metrics_summary, "test_return_mean"),
        "best_test_return_mean": _extract_metric_best(metrics_summary, "test_return_mean"),
        "last_return_mean": _extract_metric_last(metrics_summary, "return_mean"),
    }


def _choose_ranking_metric(records: List[Dict[str, Any]]) -> Optional[str]:
    for field_name in ROUND_RANKING_FIELDS:
        values = [record.get(field_name) for record in records if record.get(field_name) is not None]
        if len(values) < 2:
            continue
        distinct = {round(float(value), 12) for value in values}
        if len(distinct) > 1:
            return field_name
    for field_name in ROUND_RANKING_FIELDS:
        if any(record.get(field_name) is not None for record in records):
            return field_name
    return None


def _build_critic_payload(
    *,
    branch_plan_id: str,
    checkpoint_selection: Optional[Dict[str, Any]],
    round_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    checkpoint_comparisons = []
    aggregate_best_candidates = []

    for round_summary in round_summaries:
        checkpoint = round_summary.get("checkpoint") or {}
        resolved_round_state = round_summary.get("resolved_round_state") or {}
        best_candidate = round_summary.get("best_candidate") or {}
        ranking_metric = round_summary.get("ranking_metric")
        planned_only = all(
            record.get("status") == "planned_only"
            for record in round_summary.get("records", [])
        )

        checkpoint_payload = {
            "round_id": round_summary.get("round_id"),
            "checkpoint_name": checkpoint.get("name"),
            "checkpoint_step": checkpoint.get("step"),
            "checkpoint_stage_label": checkpoint.get("stage_label"),
            "checkpoint_reason": checkpoint.get("reason"),
            "recommended_focus": checkpoint.get("recommended_focus"),
            "active_pbrs_field": resolved_round_state.get("active_pbrs_field"),
            "active_pbrs_field_source": resolved_round_state.get("active_pbrs_field_source"),
            "candidate_value_source": (
                (resolved_round_state.get("candidate_selection_context") or {}).get(
                    "candidate_value_source"
                )
            ),
            "effective_candidate_values": (
                (resolved_round_state.get("candidate_selection_context") or {}).get(
                    "effective_candidate_values", []
                )
            ),
            "ranking_metric": ranking_metric,
            "planned_only": planned_only,
            "best_candidate": best_candidate if best_candidate else None,
            "candidate_records": round_summary.get("records", []),
        }
        checkpoint_comparisons.append(checkpoint_payload)

        if best_candidate:
            aggregate_best_candidates.append(
                {
                    "round_id": round_summary.get("round_id"),
                    "checkpoint_name": checkpoint.get("name"),
                    "checkpoint_stage_label": checkpoint.get("stage_label"),
                    "active_pbrs_field": resolved_round_state.get("active_pbrs_field"),
                    "ranking_metric": ranking_metric,
                    "candidate_id": best_candidate.get("candidate_id"),
                    "candidate_value": best_candidate.get("candidate_value"),
                    "score": (
                        best_candidate.get(ranking_metric)
                        if isinstance(ranking_metric, str)
                        else None
                    ),
                    "planned_only": planned_only,
                }
            )

    return {
        "branch_plan_id": branch_plan_id,
        "checkpoint_selection": checkpoint_selection or {},
        "checkpoint_comparisons": checkpoint_comparisons,
        "aggregate_best_candidates": aggregate_best_candidates,
    }


def summarize_branch_plan(branch_plan_dir: Path) -> Dict[str, Any]:
    manifest = _load_json(branch_plan_dir / "manifest.json")
    branch_plan_id = manifest.get("branch_plan_id", branch_plan_dir.name)
    round_summaries = []

    for round_payload in manifest.get("rounds", []):
        round_id = int(round_payload["round_id"])
        checkpoint = round_payload.get("checkpoint") or {}
        resolved_round_state = round_payload.get("resolved_round_state") or {}
        records = [
            _build_candidate_record(
                branch_plan_id=branch_plan_id,
                round_id=round_id,
                checkpoint=checkpoint,
                resolved_round_state=resolved_round_state,
                candidate=candidate,
            )
            for candidate in round_payload.get("branch_candidates", [])
        ]
        ranking_metric = _choose_ranking_metric(records)
        ranked_records = list(records)
        if ranking_metric is not None:
            ranked_records = sorted(
                records,
                key=lambda record: (
                    record.get(ranking_metric) is not None,
                    record.get(ranking_metric) or float("-inf"),
                ),
                reverse=True,
            )
        best_candidate = ranked_records[0] if ranked_records else None
        round_summaries.append(
            {
                "round_id": round_id,
                "checkpoint": checkpoint,
                "resolved_round_state": resolved_round_state,
                "ranking_metric": ranking_metric,
                "best_candidate": best_candidate,
                "records": ranked_records,
            }
        )

    critic_payload = _build_critic_payload(
        branch_plan_id=branch_plan_id,
        checkpoint_selection=manifest.get("checkpoint_selection"),
        round_summaries=round_summaries,
    )

    return {
        "branch_plan_id": branch_plan_id,
        "branch_plan_dir": str(branch_plan_dir),
        "checkpoint_selection": manifest.get("checkpoint_selection"),
        "branch_execution_summary": manifest.get("branch_execution_summary") or {},
        "branch_resume_summary": manifest.get("branch_resume_summary") or {},
        "rounds": round_summaries,
        "critic_payload": critic_payload,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def print_text_summary(summary: Dict[str, Any]) -> None:
    print(f"{summary['branch_plan_id']}: rounds={len(summary.get('rounds', []))}")
    execution_summary = summary.get("branch_execution_summary") or {}
    if execution_summary:
        print(
            "  "
            f"execute_requested={_fmt(execution_summary.get('execute_requested'))} "
            f"executed={_fmt(execution_summary.get('executed_branch_candidate_count'))} "
            f"reused={_fmt(execution_summary.get('reused_branch_candidate_count'))} "
            f"planned_only={_fmt(execution_summary.get('planned_only_branch_candidate_count'))}"
        )
    resume_summary = summary.get("branch_resume_summary") or {}
    if resume_summary:
        print(
            "  "
            f"last_resume_resumed={_fmt(resume_summary.get('resumed_candidate_count'))} "
            f"last_resume_reused={_fmt(resume_summary.get('reused_candidate_count'))} "
            f"last_resume_skipped={_fmt(resume_summary.get('skipped_candidate_count'))}"
        )
    for round_summary in summary.get("rounds", []):
        checkpoint = round_summary.get("checkpoint") or {}
        best_candidate = round_summary.get("best_candidate") or {}
        print(
            "  "
            f"round_{round_summary['round_id']:02d}: "
            f"checkpoint={_fmt(checkpoint.get('name'))}@{_fmt(checkpoint.get('step'))} "
            f"requested_step={_fmt(checkpoint.get('requested_step'))} "
            f"stage={_fmt(checkpoint.get('stage_label'))} "
            f"active={_fmt((round_summary.get('resolved_round_state') or {}).get('active_pbrs_field'))} "
            f"ranking_metric={_fmt(round_summary.get('ranking_metric'))} "
            f"best={_fmt(best_candidate.get('candidate_id'))} "
            f"value={_fmt(best_candidate.get('candidate_value'))} "
            f"status={_fmt(best_candidate.get('status'))} "
            f"score={_fmt(best_candidate.get(round_summary.get('ranking_metric') or ''))}"
        )
    critic_payload = summary.get("critic_payload") or {}
    aggregate_best = critic_payload.get("aggregate_best_candidates") or []
    if aggregate_best:
        print("  critic_payload:")
        for item in aggregate_best:
            print(
                "    "
                f"checkpoint={_fmt(item.get('checkpoint_name'))} "
                f"stage={_fmt(item.get('checkpoint_stage_label'))} "
                f"active={_fmt(item.get('active_pbrs_field'))} "
                f"best={_fmt(item.get('candidate_id'))} "
                f"value={_fmt(item.get('candidate_value'))} "
                f"planned_only={_fmt(item.get('planned_only'))} "
                f"score={_fmt(item.get('score'))}"
            )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    branch_plan_dir = Path(args.results_root) / args.branch_plan_id
    summary = summarize_branch_plan(branch_plan_dir)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print_text_summary(summary)


if __name__ == "__main__":
    main()
