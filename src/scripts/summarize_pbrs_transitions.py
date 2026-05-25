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

from scripts.inspect_reward_workflow_state import DEFAULT_RESULTS_ROOT, inspect_workflow
from scripts.summarize_pbrs_round_transition import build_transition_summary
from workflows.pbrs_round_preview import fmt_preview_value


def _build_parser() -> argparse.ArgumentParser:
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
    return parser


def _fmt(value: Any) -> str:
    return fmt_preview_value(value)


def build_transitions_summary(args: argparse.Namespace) -> Dict[str, Any]:
    workflow_dir = Path(args.results_root) / args.workflow_id
    workflow_report = inspect_workflow(workflow_dir)
    latest_completed_round = int(workflow_report.get("latest_completed_round") or 0)

    transitions: List[Dict[str, Any]] = []
    for from_round in range(1, latest_completed_round + 1):
        transition_args = argparse.Namespace(
            workflow_id=args.workflow_id,
            from_round=from_round,
            results_root=args.results_root,
            format="json",
        )
        transitions.append(build_transition_summary(transition_args))

    return {
        "workflow_id": args.workflow_id,
        "results_root": args.results_root,
        "latest_completed_round": latest_completed_round,
        "transition_count": len(transitions),
        "transitions": transitions,
    }


def print_text_summary(summary: Dict[str, Any]) -> None:
    print(
        f"{summary['workflow_id']}: latest_completed_round={summary['latest_completed_round']} "
        f"transition_count={summary['transition_count']}"
    )
    for item in summary.get("transitions", []):
        predicted = item.get("predicted_next_round") or {}
        predicted_candidate_context = predicted.get("candidate_selection_context") or {}
        transition = item.get("transition") or {}
        previous_recommendation = item.get("previous_round_search_strategy_recommendation") or {}
        print(
            "  "
            f"round_{item['from_round']:02d}->round_{item['to_round']:02d}: "
            f"active={_fmt((item.get('from_round_status') or {}).get('active_pbrs_field'))}"
            f"->{_fmt(predicted.get('active_pbrs_field'))} "
            f"source={_fmt(predicted.get('active_pbrs_field_source'))} "
            f"candidate_source={_fmt(predicted_candidate_context.get('candidate_value_source'))} "
            f"search_action={_fmt(previous_recommendation.get('candidate_value_action'))} "
            f"next_field_action={_fmt(previous_recommendation.get('next_active_field_action'))} "
            f"search_applied={_fmt(transition.get('search_strategy_applied'))} "
            f"switch_applied={_fmt(transition.get('switch_field_applied'))}"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    summary = build_transitions_summary(args)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print_text_summary(summary)


if __name__ == "__main__":
    main()
