from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from workflows.spec_diff import summarize_reward_spec_diff


def attach_candidate_proposals_to_runs(
    *,
    candidate_runs: list[Dict[str, Any]],
    candidate_proposals: list[Dict[str, Any]],
    base_reward_spec: Dict[str, Any],
) -> list[Dict[str, Any]]:
    if not isinstance(candidate_proposals, list) or not candidate_proposals:
        return candidate_runs
    attached_runs: List[Dict[str, Any]] = []
    for index, candidate_run in enumerate(candidate_runs):
        attached = dict(candidate_run)
        proposal = candidate_proposals[index % len(candidate_proposals)]
        attached["candidate_proposal"] = deepcopy(proposal)
        attached["candidate_diff"] = summarize_reward_spec_diff(
            base_reward_spec,
            attached.get("reward_spec"),
        )
        attached_runs.append(attached)
    return attached_runs


def build_candidate_diff_table(
    candidate_runs: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    diff_table = []
    for candidate_run in candidate_runs:
        diff_table.append(
            {
                "candidate_id": candidate_run.get("candidate_id"),
                "candidate_value": candidate_run.get("candidate_value"),
                "candidate_proposal": deepcopy(candidate_run.get("candidate_proposal")),
                "candidate_diff": deepcopy(candidate_run.get("candidate_diff")),
            }
        )
    return diff_table


def build_candidate_execution_plan(
    candidate_runs: list[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "candidate_count": len(candidate_runs),
        "candidates": [
            {
                "candidate_id": candidate_run.get("candidate_id"),
                "candidate_value": candidate_run.get("candidate_value"),
                "active_pbrs_field": candidate_run.get("active_pbrs_field"),
                "proposal_id": (
                    (candidate_run.get("candidate_proposal") or {}).get("candidate_id")
                    if isinstance(candidate_run.get("candidate_proposal"), dict)
                    else None
                ),
                "proposal_index": (
                    (candidate_run.get("candidate_proposal") or {}).get("proposal_index")
                    if isinstance(candidate_run.get("candidate_proposal"), dict)
                    else None
                ),
                "expected_effect": (
                    (candidate_run.get("candidate_proposal") or {}).get("expected_effect")
                    if isinstance(candidate_run.get("candidate_proposal"), dict)
                    else None
                ),
                "risk_hypothesis": (
                    (candidate_run.get("candidate_proposal") or {}).get("risk_hypothesis")
                    if isinstance(candidate_run.get("candidate_proposal"), dict)
                    else None
                ),
            }
            for candidate_run in candidate_runs
        ],
    }


def build_candidate_comparison_table(
    *,
    validation_candidate_results: Any,
    candidate_results: list[Dict[str, Any]],
) -> Dict[str, Any]:
    def _record_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
        metrics_summary = result.get("train_metrics_summary") or {}
        metric_summary = metrics_summary.get("metric_summary") or {}
        return {
            "candidate_id": result.get("candidate_id"),
            "candidate_value": result.get("candidate_value"),
            "best_test_sparse_return_mean": (
                ((metric_summary.get("test_sparse_return_mean") or {}).get("best_value"))
                if isinstance(metric_summary, dict)
                else None
            ),
            "last_test_sparse_return_mean": (
                ((metric_summary.get("test_sparse_return_mean") or {}).get("last_value"))
                if isinstance(metric_summary, dict)
                else None
            ),
            "best_test_return_mean": (
                ((metric_summary.get("test_return_mean") or {}).get("best_value"))
                if isinstance(metric_summary, dict)
                else None
            ),
            "last_mixed_return_mean": (
                ((metric_summary.get("mixed_return_mean") or {}).get("last_value"))
                if isinstance(metric_summary, dict)
                else None
            ),
        }

    validation_records = []
    if isinstance(validation_candidate_results, list):
        validation_records = [_record_from_result(item) for item in validation_candidate_results]
    final_records = [_record_from_result(item) for item in candidate_results]
    return {
        "validation_records": validation_records,
        "final_candidate_records": final_records,
    }
