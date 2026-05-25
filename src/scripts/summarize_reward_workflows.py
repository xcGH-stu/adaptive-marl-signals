from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "results" / "llm_reward_workflows"


KEY_METRICS = [
    "test_sparse_return_mean",
    "sparse_return_mean",
    "test_return_mean",
    "return_mean",
    "test_mixed_return_mean",
    "mixed_return_mean",
]

ROUND_RANKING_FIELDS = [
    "best_test_sparse_return_mean",
    "last_test_sparse_return_mean",
    "best_test_return_mean",
    "last_test_return_mean",
    "last_sparse_return_mean",
    "last_return_mean",
    "last_test_mixed_return_mean",
    "last_mixed_return_mean",
]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_workflow_dirs(results_root: Path, workflow_id: Optional[str]) -> Iterable[Path]:
    if workflow_id:
        target = results_root / workflow_id
        if target.exists():
            yield target
        return

    for path in sorted(results_root.iterdir()):
        if path.is_dir():
            yield path


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
    workflow_id: str,
    round_id: int,
    reward_paradigm: Optional[str],
    active_pbrs_field: Optional[str],
    active_pbrs_field_source: Optional[str],
    active_checkpoint_context: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    metrics_summary = candidate.get("train_metrics_summary") or {}
    train_config = candidate.get("train_config") or {}
    run_metadata = metrics_summary.get("run_metadata") or {}
    return {
        "workflow_id": workflow_id,
        "round_id": round_id,
        "reward_paradigm": reward_paradigm or run_metadata.get("reward_paradigm"),
        "active_pbrs_field": active_pbrs_field or run_metadata.get("active_pbrs_field"),
        "active_pbrs_field_source": active_pbrs_field_source
        or run_metadata.get("active_pbrs_field_source"),
        "active_checkpoint_context": active_checkpoint_context
        or run_metadata.get("active_checkpoint_context"),
        "candidate_selection_context": candidate.get("candidate_selection_context")
        or run_metadata.get("candidate_selection_context"),
        "candidate_id": candidate.get("candidate_id"),
        "candidate_value": candidate.get("candidate_value", run_metadata.get("candidate_value")),
        "algorithm": train_config.get("config", run_metadata.get("algorithm")),
        "env_key": (train_config.get("env_args") or run_metadata.get("env_args") or {}).get("key"),
        "run_id": (candidate.get("run_reference") or {}).get("run_id"),
        "status": "finished" if metrics_summary else "missing_metrics",
        "last_test_sparse_return_mean": _extract_metric_last(
            metrics_summary, "test_sparse_return_mean"
        ),
        "best_test_sparse_return_mean": _extract_metric_best(
            metrics_summary, "test_sparse_return_mean"
        ),
        "last_sparse_return_mean": _extract_metric_last(
            metrics_summary, "sparse_return_mean"
        ),
        "last_test_return_mean": _extract_metric_last(metrics_summary, "test_return_mean"),
        "best_test_return_mean": _extract_metric_best(metrics_summary, "test_return_mean"),
        "last_return_mean": _extract_metric_last(metrics_summary, "return_mean"),
        "last_test_mixed_return_mean": _extract_metric_last(
            metrics_summary, "test_mixed_return_mean"
        ),
        "last_mixed_return_mean": _extract_metric_last(metrics_summary, "mixed_return_mean"),
    }


def _build_single_run_record(
    workflow_id: str,
    round_id: int,
    reward_paradigm: Optional[str],
    active_pbrs_field: Optional[str],
    active_pbrs_field_source: Optional[str],
    active_checkpoint_context: Optional[Dict[str, Any]],
    train_metrics_summary: Dict[str, Any],
    train_config: Dict[str, Any],
    train_run_ref: Dict[str, Any],
) -> Dict[str, Any]:
    run_metadata = train_metrics_summary.get("run_metadata") or {}
    return {
        "workflow_id": workflow_id,
        "round_id": round_id,
        "reward_paradigm": reward_paradigm or run_metadata.get("reward_paradigm"),
        "active_pbrs_field": active_pbrs_field or run_metadata.get("active_pbrs_field"),
        "active_pbrs_field_source": active_pbrs_field_source
        or run_metadata.get("active_pbrs_field_source"),
        "active_checkpoint_context": active_checkpoint_context
        or run_metadata.get("active_checkpoint_context"),
        "candidate_selection_context": run_metadata.get("candidate_selection_context"),
        "candidate_id": None,
        "candidate_value": run_metadata.get("candidate_value"),
        "algorithm": train_config.get("config", run_metadata.get("algorithm")),
        "env_key": (train_config.get("env_args") or run_metadata.get("env_args") or {}).get("key"),
        "run_id": train_run_ref.get("run_id"),
        "status": "finished",
        "last_test_sparse_return_mean": _extract_metric_last(
            train_metrics_summary, "test_sparse_return_mean"
        ),
        "best_test_sparse_return_mean": _extract_metric_best(
            train_metrics_summary, "test_sparse_return_mean"
        ),
        "last_sparse_return_mean": _extract_metric_last(
            train_metrics_summary, "sparse_return_mean"
        ),
        "last_test_return_mean": _extract_metric_last(
            train_metrics_summary, "test_return_mean"
        ),
        "best_test_return_mean": _extract_metric_best(
            train_metrics_summary, "test_return_mean"
        ),
        "last_return_mean": _extract_metric_last(train_metrics_summary, "return_mean"),
        "last_test_mixed_return_mean": _extract_metric_last(
            train_metrics_summary, "test_mixed_return_mean"
        ),
        "last_mixed_return_mean": _extract_metric_last(
            train_metrics_summary, "mixed_return_mean"
        ),
    }


def summarize_workflow(workflow_dir: Path) -> Dict[str, Any]:
    manifest = _load_json(workflow_dir / "manifest.json") or {}
    workflow_id = manifest.get("workflow_id", workflow_dir.name)
    reward_paradigm = manifest.get("reward_paradigm")
    if reward_paradigm is None:
        reward_paradigm = (manifest.get("workflow_spec") or {}).get("reward_paradigm")
    if reward_paradigm is None:
        reward_paradigm = (
            "pbrs"
            if isinstance((manifest.get("workflow_spec") or {}).get("pbrs_tuning"), dict)
            else "heuristic"
        )

    rounds = []
    for round_dir in sorted(workflow_dir.glob("round_*")):
        status = _load_json(round_dir / "status.json") or {}
        round_id = int(status.get("round_id") or round_dir.name.split("_")[-1])
        candidate_manifest = _load_json(round_dir / "candidate_manifest.json") or {}
        candidate_metadata = candidate_manifest.get("metadata") or {}
        active_pbrs_field = status.get("active_pbrs_field")
        active_pbrs_field_source = status.get("active_pbrs_field_source")
        if active_pbrs_field is None:
            active_pbrs_field = candidate_metadata.get("active_pbrs_field")
        if active_pbrs_field_source is None:
            active_pbrs_field_source = candidate_metadata.get("active_pbrs_field_source")
        active_checkpoint_context = status.get("active_checkpoint_context")
        if active_checkpoint_context is None:
            active_checkpoint_context = candidate_metadata.get("active_checkpoint_context")
        active_field_carryover_context = status.get("active_field_carryover_context")
        candidate_selection_context = status.get("candidate_selection_context")
        if candidate_selection_context is None:
            candidate_selection_context = candidate_metadata.get("candidate_selection_context")
        if active_pbrs_field is None and isinstance(candidate_manifest.get("candidates"), list):
            first_candidate = next(
                (item for item in candidate_manifest["candidates"] if isinstance(item, dict)),
                None,
            )
            if first_candidate is not None:
                active_pbrs_field = first_candidate.get("active_pbrs_field")
                if active_pbrs_field_source is None:
                    active_pbrs_field_source = first_candidate.get("active_pbrs_field_source")

        round_summary: Dict[str, Any] = {
            "round_id": round_id,
            "status": status.get("status"),
            "reward_paradigm": status.get("reward_paradigm") or reward_paradigm,
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
            "validation_summary": (_load_json(round_dir / "validation_candidate_results.json") or {}).get(
                "validation_summary"
            ),
            "validation_skipped": status.get("validation_skipped"),
            "validation_skip_reason": status.get("validation_skip_reason"),
            "records": [],
        }

        candidate_results = _load_json(round_dir / "candidate_results.json") or {}
        if isinstance(candidate_results.get("candidate_results"), list):
            for candidate in candidate_results["candidate_results"]:
                round_summary["records"].append(
                    _build_candidate_record(
                        workflow_id=workflow_id,
                        round_id=round_id,
                        reward_paradigm=round_summary["reward_paradigm"],
                        active_pbrs_field=active_pbrs_field,
                        active_pbrs_field_source=active_pbrs_field_source,
                        active_checkpoint_context=active_checkpoint_context,
                        candidate=candidate,
                    )
                )
        else:
            train_metrics_summary = _load_json(round_dir / "train_metrics_summary.json")
            train_config = _load_json(round_dir / "train_config.json")
            train_run_ref = _load_json(round_dir / "train_run_ref.json")
            if train_metrics_summary and train_config and train_run_ref:
                round_summary["records"].append(
                    _build_single_run_record(
                        workflow_id=workflow_id,
                        round_id=round_id,
                        reward_paradigm=round_summary["reward_paradigm"],
                        active_pbrs_field=active_pbrs_field,
                        active_pbrs_field_source=active_pbrs_field_source,
                        active_checkpoint_context=active_checkpoint_context,
                        train_metrics_summary=train_metrics_summary,
                        train_config=train_config,
                        train_run_ref=train_run_ref,
                    )
                )
        _annotate_round_records(round_summary)
        rounds.append(round_summary)

    return {
        "workflow_id": workflow_id,
        "status": manifest.get("status"),
        "current_round": manifest.get("current_round"),
        "reward_paradigm": reward_paradigm,
        "algorithm": ((manifest.get("workflow_spec") or {}).get("train") or {}).get("config"),
        "env_key": ((((manifest.get("workflow_spec") or {}).get("train") or {}).get("env_args") or {}).get("key")),
        "rounds": rounds,
    }


def _record_sort_key(record: Dict[str, Any], ranking_field: str) -> tuple[int, float]:
    value = record.get(ranking_field)
    if value is None:
        return (1, float("-inf"))
    return (0, float(value))


def _annotate_round_records(round_summary: Dict[str, Any]) -> None:
    records = round_summary.get("records") or []
    ranking_field = None
    for field_name in ROUND_RANKING_FIELDS:
        if any(record.get(field_name) is not None for record in records):
            ranking_field = field_name
            break

    round_summary["ranking_metric"] = ranking_field
    if ranking_field is None:
        round_summary["best_record"] = None
        return

    sorted_records = sorted(
        records,
        key=lambda record: _record_sort_key(record, ranking_field),
        reverse=True,
    )
    round_summary["records"] = sorted_records
    best_record = next(
        (record for record in sorted_records if record.get(ranking_field) is not None),
        None,
    )
    round_summary["best_record"] = (
        {
            "candidate_id": best_record.get("candidate_id"),
            "candidate_value": best_record.get("candidate_value"),
            "run_id": best_record.get("run_id"),
            "ranking_metric": ranking_field,
            "ranking_value": best_record.get(ranking_field),
        }
        if best_record is not None
        else None
    )
    for record in sorted_records:
        record["is_best"] = (
            best_record is not None
            and record.get("candidate_id") == best_record.get("candidate_id")
            and record.get("run_id") == best_record.get("run_id")
        )


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_text_summary(workflow_summary: Dict[str, Any]) -> None:
    print(
        f"{workflow_summary['workflow_id']}: status={workflow_summary['status']} "
        f"paradigm={workflow_summary['reward_paradigm']} "
        f"algo={workflow_summary['algorithm']} env={workflow_summary['env_key']}"
    )
    for round_summary in workflow_summary["rounds"]:
        print(
            f"  round_{round_summary['round_id']:02d}: "
            f"status={round_summary['status']} "
            f"active={round_summary['active_pbrs_field'] or '-'} "
            f"active_source={round_summary.get('active_pbrs_field_source') or '-'} "
            f"field_carryover={((round_summary.get('active_field_carryover_context') or {}).get('previous_switch_field_applied') if isinstance(round_summary.get('active_field_carryover_context'), dict) else '-')!s} "
            f"candidate_source={((round_summary.get('candidate_selection_context') or {}).get('candidate_value_source') if isinstance(round_summary.get('candidate_selection_context'), dict) else '-') or '-'} "
            f"carryover={((round_summary.get('candidate_selection_context') or {}).get('previous_search_strategy_applied') if isinstance(round_summary.get('candidate_selection_context'), dict) else '-')!s} "
            f"checkpoint={((round_summary.get('active_checkpoint_context') or {}).get('name') if isinstance(round_summary.get('active_checkpoint_context'), dict) else '-') or '-'} "
            f"verdict={round_summary.get('critic_verdict') or '-'} "
            f"best_candidate={((round_summary.get('critic_best_candidate') or {}).get('candidate_id') if isinstance(round_summary.get('critic_best_candidate'), dict) else '-') or '-'} "
            f"full_run={_format_value(round_summary.get('critic_whether_to_full_run'))} "
            f"search_action={((round_summary.get('search_strategy_recommendation') or {}).get('candidate_value_action') if isinstance(round_summary.get('search_strategy_recommendation'), dict) else '-') or '-'} "
            f"validation_skipped={_format_value(round_summary.get('validation_skipped'))} "
            f"records={len(round_summary['records'])}"
        )
        validation_skip_reason = round_summary.get("validation_skip_reason")
        if validation_skip_reason is None and isinstance(round_summary.get("validation_summary"), dict):
            validation_skip_reason = round_summary["validation_summary"].get("skip_reason")
        if validation_skip_reason is not None:
            print(f"    validation_skip_reason={validation_skip_reason}")
        if round_summary.get("best_record") is not None:
            best_record = round_summary["best_record"]
            print(
                "    "
                f"best={best_record['candidate_id'] or 'single'} "
                f"value={_format_value(best_record['candidate_value'])} "
                f"{best_record['ranking_metric']}={_format_value(best_record['ranking_value'])}"
            )
        for record in round_summary["records"]:
            print(
                "    "
                f"{'*' if record.get('is_best') else '-'} "
                f"candidate={record['candidate_id'] or 'single'} "
                f"value={_format_value(record['candidate_value'])} "
                f"last_test_sparse={_format_value(record['last_test_sparse_return_mean'])} "
                f"best_test_sparse={_format_value(record['best_test_sparse_return_mean'])} "
                f"last_test_return={_format_value(record['last_test_return_mean'])} "
                f"run_id={_format_value(record['run_id'])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing llm_reward_workflows results.",
    )
    parser.add_argument(
        "--workflow-id",
        default=None,
        help="Optional single workflow to summarize.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    summaries = [
        summarize_workflow(workflow_dir)
        for workflow_dir in _iter_workflow_dirs(results_root, args.workflow_id)
        if (workflow_dir / "manifest.json").exists()
    ]

    if args.format == "json":
        print(json.dumps({"workflows": summaries}, indent=2, sort_keys=True))
        return

    for index, workflow_summary in enumerate(summaries):
        if index:
            print()
        print_text_summary(workflow_summary)


if __name__ == "__main__":
    main()
