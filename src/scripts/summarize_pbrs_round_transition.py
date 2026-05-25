from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.inspect_reward_workflow_state import DEFAULT_RESULTS_ROOT, inspect_workflow
from workflows.pbrs_round_preview import (
    build_read_only_workflow,
    fmt_preview_value,
    load_round_state_from_storage,
    namespace_for_existing_workflow,
    resolve_round_preview_state,
)
from workflows.storage import WorkflowStorage


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument(
        "--from-round",
        type=int,
        default=None,
        help="Source round for the transition summary. Defaults to latest completed round.",
    )
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


def _determine_from_round(workflow_dir: Path, explicit_from_round: Optional[int]) -> int:
    if explicit_from_round is not None:
        return int(explicit_from_round)
    report = inspect_workflow(workflow_dir)
    latest_completed_round = int(report.get("latest_completed_round") or 0)
    if latest_completed_round <= 0:
        raise ValueError("no completed round available for transition summary")
    return latest_completed_round


def _candidate_values_from_status(status: Optional[Dict[str, Any]]) -> list[float]:
    if not isinstance(status, dict):
        return []
    candidate_selection_context = status.get("candidate_selection_context")
    if not isinstance(candidate_selection_context, dict):
        return []
    values = candidate_selection_context.get("effective_candidate_values")
    if not isinstance(values, list):
        return []
    normalized = []
    for value in values:
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _extract_previous_recommendation(previous_round: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(previous_round, dict):
        return None
    critic_response = previous_round.get("critic_response")
    if not isinstance(critic_response, dict):
        return None
    parsed = critic_response.get("parsed")
    if not isinstance(parsed, dict):
        return None
    recommendation = parsed.get("search_strategy_recommendation")
    return recommendation if isinstance(recommendation, dict) else None


def build_transition_summary(args: argparse.Namespace) -> Dict[str, Any]:
    workflow_dir = Path(args.results_root) / args.workflow_id
    from_round = _determine_from_round(workflow_dir, args.from_round)
    to_round = from_round + 1

    preview_args = namespace_for_existing_workflow(
        workflow_id=args.workflow_id,
        results_root=args.results_root,
        preview_round=to_round,
    )
    workflow = build_read_only_workflow(preview_args)
    storage = WorkflowStorage.load(args.workflow_id, root_dir=args.results_root)

    previous_round = storage.get_previous_round_artifacts(to_round)
    predicted_next = resolve_round_preview_state(
        workflow,
        round_id=to_round,
        previous_round=previous_round,
    )
    from_status = load_round_state_from_storage(storage, from_round)
    to_status = load_round_state_from_storage(storage, to_round)
    actual_to = None
    if isinstance(to_status, dict):
        actual_to = {
            "active_pbrs_field": to_status.get("active_pbrs_field"),
            "active_pbrs_field_source": to_status.get("active_pbrs_field_source"),
            "active_checkpoint_context": to_status.get("active_checkpoint_context"),
            "active_field_carryover_context": to_status.get("active_field_carryover_context"),
            "candidate_selection_context": to_status.get("candidate_selection_context"),
        }

    previous_recommendation = _extract_previous_recommendation(previous_round)
    from_values = _candidate_values_from_status(from_status)
    predicted_values = list(
        (predicted_next.get("candidate_selection_context") or {}).get(
            "effective_candidate_values", []
        )
    )
    actual_values = _candidate_values_from_status(to_status)

    transition = {
        "field_changed": (
            (from_status or {}).get("active_pbrs_field") != predicted_next.get("active_pbrs_field")
            if isinstance(from_status, dict)
            else None
        ),
        "candidate_values_changed": (
            from_values != predicted_values if from_values else None
        ),
        "search_strategy_applied": (
            (predicted_next.get("candidate_selection_context") or {}).get(
                "previous_search_strategy_applied"
            )
        ),
        "switch_field_applied": (
            (predicted_next.get("active_field_carryover_context") or {}).get(
                "previous_switch_field_applied"
            )
        ),
    }

    actual_match = None
    if isinstance(actual_to, dict):
        actual_match = {
            "field": (
                actual_to.get("active_pbrs_field") == predicted_next.get("active_pbrs_field")
                if actual_to.get("active_pbrs_field") is not None
                else None
            ),
            "field_source": (
                actual_to.get("active_pbrs_field_source")
                == predicted_next.get("active_pbrs_field_source")
                if actual_to.get("active_pbrs_field_source") is not None
                else None
            ),
            "candidate_values": actual_values == predicted_values if actual_values else None,
            "candidate_source": (
                ((actual_to.get("candidate_selection_context") or {}).get("candidate_value_source"))
                == ((predicted_next.get("candidate_selection_context") or {}).get("candidate_value_source"))
                if ((actual_to.get("candidate_selection_context") or {}).get("candidate_value_source"))
                is not None
                else None
            ),
        }

    return {
        "workflow_id": args.workflow_id,
        "results_root": args.results_root,
        "from_round": from_round,
        "to_round": to_round,
        "from_round_status": from_status,
        "previous_round_search_strategy_recommendation": previous_recommendation,
        "predicted_next_round": predicted_next,
        "actual_next_round": actual_to,
        "transition": transition,
        "actual_match": actual_match,
    }


def _fmt(value: Any) -> str:
    return fmt_preview_value(value)


def print_text_summary(summary: Dict[str, Any]) -> None:
    from_status = summary.get("from_round_status") or {}
    predicted = summary.get("predicted_next_round") or {}
    actual = summary.get("actual_next_round") or {}
    transition = summary.get("transition") or {}
    previous_recommendation = summary.get("previous_round_search_strategy_recommendation") or {}
    predicted_candidate_context = predicted.get("candidate_selection_context") or {}
    actual_candidate_context = actual.get("candidate_selection_context") or {}

    print(
        f"{summary['workflow_id']}: round_{summary['from_round']:02d} -> "
        f"round_{summary['to_round']:02d}"
    )
    print(
        "  "
        f"from_active={_fmt(from_status.get('active_pbrs_field'))} "
        f"predicted_active={_fmt(predicted.get('active_pbrs_field'))} "
        f"predicted_active_source={_fmt(predicted.get('active_pbrs_field_source'))}"
    )
    print(
        "  "
        f"search_action={_fmt(previous_recommendation.get('candidate_value_action'))} "
        f"next_field_action={_fmt(previous_recommendation.get('next_active_field_action'))} "
        f"suggested_next_field={_fmt(previous_recommendation.get('suggested_next_active_field'))}"
    )
    print(
        "  "
        f"predicted_candidate_source={_fmt(predicted_candidate_context.get('candidate_value_source'))} "
        f"predicted_values={json.dumps(predicted_candidate_context.get('effective_candidate_values', []), ensure_ascii=False)}"
    )
    print(
        "  "
        f"field_changed={_fmt(transition.get('field_changed'))} "
        f"candidate_values_changed={_fmt(transition.get('candidate_values_changed'))} "
        f"search_strategy_applied={_fmt(transition.get('search_strategy_applied'))} "
        f"switch_field_applied={_fmt(transition.get('switch_field_applied'))}"
    )
    if actual:
        print(
            "  "
            f"actual_active={_fmt(actual.get('active_pbrs_field'))} "
            f"actual_active_source={_fmt(actual.get('active_pbrs_field_source'))} "
            f"actual_candidate_source={_fmt(actual_candidate_context.get('candidate_value_source'))}"
        )
        print(
            "  "
            f"actual_values={json.dumps(actual_candidate_context.get('effective_candidate_values', []), ensure_ascii=False)}"
        )
        actual_match = summary.get("actual_match") or {}
        print(
            "  "
            f"match_field={_fmt(actual_match.get('field'))} "
            f"match_field_source={_fmt(actual_match.get('field_source'))} "
            f"match_candidate_source={_fmt(actual_match.get('candidate_source'))} "
            f"match_candidate_values={_fmt(actual_match.get('candidate_values'))}"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    summary = build_transition_summary(args)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print_text_summary(summary)


if __name__ == "__main__":
    main()
