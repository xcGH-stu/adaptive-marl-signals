from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.summarize_reward_workflows import summarize_workflow


RANKING_FIELDS = [
    "best_test_sparse_return_mean",
    "last_test_sparse_return_mean",
    "best_test_return_mean",
    "last_test_return_mean",
    "last_test_mixed_return_mean",
    "last_mixed_return_mean",
]


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _score_record(record: Dict[str, Any]) -> tuple[float, str]:
    for field in RANKING_FIELDS:
        value = record.get(field)
        if isinstance(value, (int, float)):
            return float(value), field
    return float("-inf"), "none"


def _best_final_round_record(workflow_dir: Path) -> Optional[Dict[str, Any]]:
    summary = summarize_workflow(workflow_dir)
    rounds = summary.get("rounds") or []
    if not rounds:
        return None
    final_round = max(rounds, key=lambda item: int(item.get("round_id", 0)))
    records = final_round.get("records") or []
    if not records:
        return None
    best = max(records, key=lambda item: _score_record(item)[0])
    score, ranking_field = _score_record(best)
    metadata = (summary.get("manifest") or {}).get("workflow_spec", {}).get("experiment_metadata") or {}
    return {
        "source_type": "workflow",
        "source_path": str(workflow_dir),
        "workflow_id": summary.get("workflow_id"),
        "experiment_family": metadata.get("experiment_family", "unknown"),
        "budget_profile": metadata.get("budget_profile", "unknown"),
        "seed": metadata.get("seed"),
        "algorithm": best.get("algorithm"),
        "env_key": best.get("env_key"),
        "round_id": final_round.get("round_id"),
        "candidate_id": best.get("candidate_id"),
        "candidate_value": best.get("candidate_value"),
        "ranking_metric": ranking_field,
        "ranking_score": score,
        "best_test_sparse_return_mean": best.get("best_test_sparse_return_mean"),
        "last_test_sparse_return_mean": best.get("last_test_sparse_return_mean"),
        "best_test_return_mean": best.get("best_test_return_mean"),
        "last_test_return_mean": best.get("last_test_return_mean"),
        "last_test_mixed_return_mean": best.get("last_test_mixed_return_mean"),
        "last_mixed_return_mean": best.get("last_mixed_return_mean"),
        "status": summary.get("manifest", {}).get("status"),
    }


def _baseline_record_from_result_json(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    result = payload.get("result") or {}
    train_config = result.get("train_config") or payload.get("train_config") or {}
    metrics_summary = result.get("metrics_summary") or {}
    run_metadata = metrics_summary.get("run_metadata") or {}
    experiment_metadata = payload.get("experiment_metadata") or train_config.get("experiment_metadata") or {}
    metric_summary = metrics_summary.get("metric_summary") or {}

    def _metric(metric_name: str, key: str) -> Any:
        item = metric_summary.get(metric_name) or {}
        return item.get(key)

    return {
        "source_type": "baseline_result_json",
        "source_path": str(path),
        "workflow_id": payload.get("baseline_id") or payload.get("run_id"),
        "experiment_family": experiment_metadata.get("experiment_family", "unknown"),
        "budget_profile": experiment_metadata.get("budget_profile", "unknown"),
        "seed": experiment_metadata.get("seed", run_metadata.get("seed")),
        "algorithm": train_config.get("config", run_metadata.get("algorithm")),
        "env_key": (train_config.get("env_args") or run_metadata.get("env_args") or {}).get("key"),
        "round_id": 0,
        "candidate_id": None,
        "candidate_value": None,
        "ranking_metric": "best_test_sparse_return_mean",
        "ranking_score": _metric("test_sparse_return_mean", "best_value"),
        "best_test_sparse_return_mean": _metric("test_sparse_return_mean", "best_value"),
        "last_test_sparse_return_mean": _metric("test_sparse_return_mean", "last_value"),
        "best_test_return_mean": _metric("test_return_mean", "best_value"),
        "last_test_return_mean": _metric("test_return_mean", "last_value"),
        "last_test_mixed_return_mean": _metric("test_mixed_return_mean", "last_value"),
        "last_mixed_return_mean": _metric("mixed_return_mean", "last_value"),
        "status": "completed" if result else "planned",
    }


def _group_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for record in records:
        key = (
            record.get("experiment_family"),
            record.get("budget_profile"),
            record.get("algorithm"),
            record.get("env_key"),
        )
        grouped.setdefault(key, []).append(record)

    rows: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        scores = [item.get("best_test_sparse_return_mean") for item in items if isinstance(item.get("best_test_sparse_return_mean"), (int, float))]
        last_scores = [item.get("last_test_sparse_return_mean") for item in items if isinstance(item.get("last_test_sparse_return_mean"), (int, float))]
        row = {
            "experiment_family": key[0],
            "budget_profile": key[1],
            "algorithm": key[2],
            "env_key": key[3],
            "num_seeds": len({item.get("seed") for item in items}),
            "num_runs": len(items),
            "mean_best_test_sparse_return_mean": statistics.mean(scores) if scores else None,
            "std_best_test_sparse_return_mean": statistics.pstdev(scores) if len(scores) > 1 else 0.0 if scores else None,
            "mean_last_test_sparse_return_mean": statistics.mean(last_scores) if last_scores else None,
            "std_last_test_sparse_return_mean": statistics.pstdev(last_scores) if len(last_scores) > 1 else 0.0 if last_scores else None,
            "seeds": sorted(item.get("seed") for item in items if item.get("seed") is not None),
        }
        rows.append(row)
    return rows


def _print_text(records: List[Dict[str, Any]], grouped: List[Dict[str, Any]]) -> None:
    print("per-run:")
    for record in records:
        print(
            f"  family={record.get('experiment_family')} budget={record.get('budget_profile')} "
            f"seed={record.get('seed')} alg={record.get('algorithm')} env={record.get('env_key')} "
            f"best_test_sparse={record.get('best_test_sparse_return_mean')} source={record.get('source_path')}"
        )
    print("grouped:")
    for row in grouped:
        print(
            f"  family={row.get('experiment_family')} budget={row.get('budget_profile')} "
            f"alg={row.get('algorithm')} env={row.get('env_key')} "
            f"seeds={row.get('seeds')} mean_best_test_sparse={row.get('mean_best_test_sparse_return_mean')} "
            f"std={row.get('std_best_test_sparse_return_mean')}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-run-dir", action="append", default=[])
    parser.add_argument("--baseline-result-json", action="append", default=[])
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--format", choices=["text", "json", "csv"], default="text")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    records: List[Dict[str, Any]] = []
    for path in args.workflow_run_dir:
        record = _best_final_round_record(Path(path))
        if record is not None:
            records.append(record)
    for path in args.baseline_result_json:
        records.append(_baseline_record_from_result_json(Path(path)))

    grouped = _group_records(records)
    payload = {"per_run_records": records, "grouped_summary": grouped}

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "csv":
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(grouped[0].keys()) if grouped else [])
                if grouped:
                    writer.writeheader()
                    writer.writerows(grouped)
        else:
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(grouped[0].keys()) if grouped else [])
        if grouped:
            writer.writeheader()
            writer.writerows(grouped)
        return
    _print_text(records, grouped)


if __name__ == "__main__":
    main()
