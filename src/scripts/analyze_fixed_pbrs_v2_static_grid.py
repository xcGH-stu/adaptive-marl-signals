from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.fixed_pbrs_v2_static_grid import DEFAULT_ENV_KEY
from experiments.fixed_pbrs_v2_static_grid import DEFAULT_T_MAX
from experiments.fixed_pbrs_v2_static_grid import FIXED_PBRS_V2_STATIC_CONFIGS
from experiments.fixed_pbrs_v2_static_grid import build_expected_summary
from experiments.fixed_pbrs_v2_static_grid import build_workflow_id
from experiments.fixed_pbrs_v2_static_grid import get_static_config
from experiments.fixed_pbrs_v2_static_grid import list_config_ids


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _trapezoid_area(points: List[Tuple[int, float]]) -> float:
    if len(points) < 2:
        return 0.0
    area = 0.0
    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        delta = float(x1 - x0)
        if delta <= 0:
            continue
        area += 0.5 * (float(y0) + float(y1)) * delta
    return float(area)


def _series_from_metric_payload(metric_payload: Any) -> Tuple[List[int], List[float]]:
    if not isinstance(metric_payload, dict):
        return [], []
    steps = [_safe_int(item) for item in list(metric_payload.get("steps") or [])]
    values = [_safe_float(item) for item in list(metric_payload.get("values") or [])]
    clean = [
        (int(step), float(value))
        for step, value in zip(steps, values)
        if step is not None and value is not None
    ]
    return [step for step, _ in clean], [value for _, value in clean]


def _choose_eval_metric(metrics_payload: Dict[str, Any]) -> Tuple[str, List[int], List[float]]:
    for metric_name in ("test_sparse_return_mean", "test_return_mean"):
        steps, values = _series_from_metric_payload(metrics_payload.get(metric_name))
        if steps and values:
            return metric_name, steps, values
    return "", [], []


def _metric_last(metrics_payload: Dict[str, Any], metric_name: str) -> Optional[float]:
    _steps, values = _series_from_metric_payload(metrics_payload.get(metric_name))
    if not values:
        return None
    return float(values[-1])


def _metric_best(metrics_payload: Dict[str, Any], metric_name: str) -> Optional[float]:
    _steps, values = _series_from_metric_payload(metrics_payload.get(metric_name))
    if not values:
        return None
    return float(max(values))


def _metric_last_k_mean(metrics_payload: Dict[str, Any], metric_name: str, k: int = 5) -> Optional[float]:
    _steps, values = _series_from_metric_payload(metrics_payload.get(metric_name))
    if not values:
        return None
    tail = values[-k:]
    return float(sum(tail) / float(len(tail)))


def _scan_run_dirs(results_root: Path) -> List[Path]:
    sacred_root = results_root / "sacred"
    if not sacred_root.exists():
        return []
    return sorted(path.parent for path in sacred_root.rglob("run.json"))


def _read_run_dir(run_dir: Path) -> Optional[Dict[str, Any]]:
    config_path = run_dir / "config.json"
    run_path = run_dir / "run.json"
    metrics_path = run_dir / "metrics.json"
    info_path = run_dir / "info.json"
    if not config_path.exists() or not run_path.exists() or not metrics_path.exists():
        return None
    config = _load_json(config_path)
    run_json = _load_json(run_path)
    metrics = _load_json(metrics_path)
    info = _load_json(info_path) if info_path.exists() else {}
    if not isinstance(config, dict) or not isinstance(run_json, dict) or not isinstance(metrics, dict):
        return None
    return {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "run_path": str(run_path),
        "metrics_path": str(metrics_path),
        "info_path": str(info_path) if info_path.exists() else None,
        "config": config,
        "run_json": run_json,
        "metrics": metrics,
        "info": info if isinstance(info, dict) else {},
    }


def _find_static_baseline_run(results_root: Path, workflow_id: str) -> Optional[Dict[str, Any]]:
    for run_dir in _scan_run_dirs(results_root):
        payload = _read_run_dir(run_dir)
        if payload is None:
            continue
        config = payload["config"]
        run_json = payload["run_json"]
        workflow_values = {
            str(config.get("workflow_id") or ""),
            str((run_json.get("config") or {}).get("workflow_id") or ""),
        }
        if workflow_id in workflow_values:
            return payload
    return None


@dataclass
class RunSummary:
    workflow_id: str
    config_id: str
    seed: int
    run_dir: Optional[str]
    run_id: Optional[int]
    metric_name_used: Optional[str]
    final_test_sparse_return: Optional[float]
    max_test_sparse_return: Optional[float]
    last_k_mean_test_sparse_return: Optional[float]
    selected_path_auc_mean: Optional[float]
    selected_path_auc_raw: Optional[float]
    selected_path_auc_mode: str
    pbrs_version_ok: bool
    pbrs_runtime_metric_present: bool
    pbrs_weights_metrics_present: bool
    eval_use_pbrs_false: bool
    test_eval_use_pbrs_mean: Optional[float]
    llm_free: bool
    warnings: List[str]
    config_json: Optional[str]
    metrics_json: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "config_id": self.config_id,
            "seed": self.seed,
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "metric_name_used": self.metric_name_used,
            "final_test_sparse_return": self.final_test_sparse_return,
            "max_test_sparse_return": self.max_test_sparse_return,
            "last_k_mean_test_sparse_return": self.last_k_mean_test_sparse_return,
            "selected_path_auc_mean": self.selected_path_auc_mean,
            "selected_path_auc_raw": self.selected_path_auc_raw,
            "selected_path_auc_mode": self.selected_path_auc_mode,
            "pbrs_version_ok": self.pbrs_version_ok,
            "pbrs_runtime_metric_present": self.pbrs_runtime_metric_present,
            "pbrs_weights_metrics_present": self.pbrs_weights_metrics_present,
            "eval_use_pbrs_false": self.eval_use_pbrs_false,
            "test_eval_use_pbrs_mean": self.test_eval_use_pbrs_mean,
            "llm_free": self.llm_free,
            "warnings": list(self.warnings),
            "config_json": self.config_json,
            "metrics_json": self.metrics_json,
        }


def summarize_run(results_root: Path, *, config_id: str, seed: int, run_index: int = 1) -> RunSummary:
    workflow_id = build_workflow_id(config_id, seed, run_index=run_index)
    payload = _find_static_baseline_run(results_root, workflow_id)
    if payload is None:
        return RunSummary(
            workflow_id=workflow_id,
            config_id=config_id,
            seed=seed,
            run_dir=None,
            run_id=None,
            metric_name_used=None,
            final_test_sparse_return=None,
            max_test_sparse_return=None,
            last_k_mean_test_sparse_return=None,
            selected_path_auc_mean=None,
            selected_path_auc_raw=None,
            selected_path_auc_mode="missing_run",
            pbrs_version_ok=False,
            pbrs_runtime_metric_present=False,
            pbrs_weights_metrics_present=False,
            eval_use_pbrs_false=False,
            test_eval_use_pbrs_mean=None,
            llm_free=True,
            warnings=[f"workflow_id not found under {results_root}: {workflow_id}"],
            config_json=None,
            metrics_json=None,
        )

    config = payload["config"]
    metrics = payload["metrics"]
    env_args = config.get("env_args") or {}
    warnings: List[str] = []
    metric_name, steps, values = _choose_eval_metric(metrics)
    if not metric_name:
        warnings.append("no test_sparse_return_mean or test_return_mean metric found")
    combined = list(zip(steps, values))
    auc_raw = _trapezoid_area(combined) if combined else None
    auc_mean = (float(auc_raw) / float(DEFAULT_T_MAX)) if auc_raw is not None else None
    test_eval_use_pbrs = _metric_last(metrics, "reward_breakdown__eval_use_pbrs_mean")
    if test_eval_use_pbrs is None:
        test_eval_use_pbrs = _metric_last(metrics, "test_eval_use_pbrs_mean")
    expected_weights = ["alloc", "app", "col", "cov", "ready", "stab"]
    pbrs_weights_metrics_present = all(
        f"pbrs_weights__{name}_mean" in metrics for name in expected_weights
    )
    if not pbrs_weights_metrics_present:
        warnings.append("one or more pbrs_weights__*_mean metrics missing")
    pbrs_runtime_metric_present = "pbrs_v2_runtime_used_mean" in metrics
    if not pbrs_runtime_metric_present:
        warnings.append("pbrs_v2_runtime_used_mean missing")
    pbrs_version_ok = str(env_args.get("pbrs_version") or "") == "lbf_pbrs_v2"
    eval_use_pbrs_false = bool(env_args.get("eval_use_pbrs") is False)
    if test_eval_use_pbrs is not None and abs(float(test_eval_use_pbrs)) > 1e-9:
        warnings.append(f"test eval PBRS metric is not zero: {test_eval_use_pbrs}")
    llm_free = True
    experiment_metadata_raw = config.get("experiment_metadata")
    if isinstance(experiment_metadata_raw, str):
        try:
            experiment_metadata = json.loads(experiment_metadata_raw)
        except Exception:
            experiment_metadata = {}
    else:
        experiment_metadata = experiment_metadata_raw if isinstance(experiment_metadata_raw, dict) else {}
    if experiment_metadata.get("native_original_pbrs") is not True:
        warnings.append("native_original_pbrs flag not recorded as true in experiment_metadata")
    if str(config.get("policy_guidance_config") or "") not in {"{}", "null", ""}:
        warnings.append("policy_guidance_config is not empty in sacred config")
    return RunSummary(
        workflow_id=workflow_id,
        config_id=config_id,
        seed=seed,
        run_dir=payload["run_dir"],
        run_id=_safe_int(Path(payload["run_dir"]).name),
        metric_name_used=metric_name or None,
        final_test_sparse_return=float(values[-1]) if values else None,
        max_test_sparse_return=float(max(values)) if values else None,
        last_k_mean_test_sparse_return=_metric_last_k_mean(metrics, metric_name, k=5) if metric_name else None,
        selected_path_auc_mean=auc_mean,
        selected_path_auc_raw=auc_raw,
        selected_path_auc_mode="single_run_curve_trapezoid" if combined else "missing_metric",
        pbrs_version_ok=pbrs_version_ok,
        pbrs_runtime_metric_present=pbrs_runtime_metric_present,
        pbrs_weights_metrics_present=pbrs_weights_metrics_present,
        eval_use_pbrs_false=eval_use_pbrs_false,
        test_eval_use_pbrs_mean=test_eval_use_pbrs,
        llm_free=llm_free,
        warnings=warnings,
        config_json=payload["config_path"],
        metrics_json=payload["metrics_path"],
    )


def _score_key(summary: RunSummary) -> Tuple[float, float, float]:
    auc = summary.selected_path_auc_mean if summary.selected_path_auc_mean is not None else float("-inf")
    best = summary.max_test_sparse_return if summary.max_test_sparse_return is not None else float("-inf")
    final = summary.final_test_sparse_return if summary.final_test_sparse_return is not None else float("-inf")
    return (auc, best, final)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / float(len(clean)))


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    mu = sum(clean) / float(len(clean))
    return float((sum((value - mu) ** 2 for value in clean) / float(len(clean))) ** 0.5)


def _seed1_comparison_md(rows: List[RunSummary], best_config_id: Optional[str]) -> str:
    lines = [
        "# Fixed PBRS-v2 Static Grid Seed1 Comparison",
        "",
        f"- Generated at: `{_utc_now_iso()}`",
        f"- Env: `{DEFAULT_ENV_KEY}`",
        f"- Budget: `{DEFAULT_T_MAX}`",
        f"- Best config id: `{best_config_id}`",
        "",
        "| Config | Workflow | Metric | AUC Mean | Max Test Sparse | Final Test Sparse | Last-5 Mean | Run ID | Warnings |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {config_id} | {workflow_id} | {metric} | {auc} | {best} | {final} | {lastk} | {run_id} | {warnings} |".format(
                config_id=row.config_id,
                workflow_id=row.workflow_id,
                metric=row.metric_name_used or "-",
                auc=row.selected_path_auc_mean,
                best=row.max_test_sparse_return,
                final=row.final_test_sparse_return,
                lastk=row.last_k_mean_test_sparse_return,
                run_id=row.run_id,
                warnings="; ".join(row.warnings),
            )
        )
    return "\n".join(lines) + "\n"


def _selection_md(rows: List[RunSummary], best_config_id: Optional[str]) -> str:
    best_row = next((row for row in rows if row.config_id == best_config_id), None)
    return "\n".join(
        [
            "# Fixed PBRS-v2 Static Grid Selection",
            "",
            f"- Selected config: `{best_config_id}`",
            "- Primary metric: `selected_path_auc_mean`",
            "- Secondary metric: `max_test_sparse_return`",
            "- Tertiary metric: `final_test_sparse_return`",
            "- Selection note: for this fixed baseline, selected-path AUC is computed from the single full-budget sparse-eval curve.",
            f"- Winner workflow: `{best_row.workflow_id if best_row else '-'}`",
            f"- Winner run id: `{best_row.run_id if best_row else '-'}`",
        ]
    ) + "\n"


def _best_3seed_md(best_config_id: str, rows: List[RunSummary], aggregate: Dict[str, Any]) -> str:
    lines = [
        "# Fixed PBRS-v2 Static Grid Best 3-Seed Summary",
        "",
        f"- Best config id: `{best_config_id}`",
        f"- Seeds: `{aggregate.get('seeds')}`",
        f"- Final test sparse return mean/std: `{aggregate.get('final_test_sparse_return_mean')}` / `{aggregate.get('final_test_sparse_return_std')}`",
        f"- Max test sparse return mean/std: `{aggregate.get('max_test_sparse_return_mean')}` / `{aggregate.get('max_test_sparse_return_std')}`",
        f"- Selected-path AUC mean/std: `{aggregate.get('selected_path_auc_mean_mean')}` / `{aggregate.get('selected_path_auc_mean_std')}`",
        "",
        "| Seed | Workflow | Run ID | Metric | Final | Max | Last-5 Mean | AUC Mean | Warnings |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {workflow_id} | {run_id} | {metric} | {final} | {best} | {lastk} | {auc} | {warnings} |".format(
                seed=row.seed,
                workflow_id=row.workflow_id,
                run_id=row.run_id,
                metric=row.metric_name_used or "-",
                final=row.final_test_sparse_return,
                best=row.max_test_sparse_return,
                lastk=row.last_k_mean_test_sparse_return,
                auc=row.selected_path_auc_mean,
                warnings="; ".join(row.warnings),
            )
        )
    return "\n".join(lines) + "\n"


def _seed1_mode(results_root: Path, output_dir: Path) -> Dict[str, Any]:
    rows = [summarize_run(results_root, config_id=config_id, seed=1) for config_id in list_config_ids()]
    complete_rows = [row for row in rows if row.metric_name_used]
    best_row = max(complete_rows, key=_score_key) if complete_rows else None
    best_config_id = best_row.config_id if best_row else None
    comparison_summary = {
        **build_expected_summary(),
        "generated_at_utc": _utc_now_iso(),
        "seed1_comparison_completed": True,
        "candidate_configs": list_config_ids(),
        "selection_metric_priority": [
            "selected_path_auc_mean",
            "max_test_sparse_return",
            "final_test_sparse_return",
            "last_k_mean_test_sparse_return",
        ],
        "seed1_rows": [row.to_dict() for row in rows],
        "best_config_id": best_config_id,
        "selection_metric": "selected_path_auc_mean_then_max_then_final",
        "best_config_selected_for_seed2_seed3": best_config_id is not None,
        "ready_to_launch_best_config_seed2_seed3": best_config_id is not None,
    }
    selection_summary = {
        **comparison_summary,
        "seed1_selection_completed": True,
    }
    _write_text(output_dir / "fixed_pbrs_v2_static_grid_seed1_comparison.md", _seed1_comparison_md(rows, best_config_id))
    _write_json(output_dir / "fixed_pbrs_v2_static_grid_seed1_comparison_summary.json", comparison_summary)
    _write_text(output_dir / "fixed_pbrs_v2_static_grid_selection.md", _selection_md(rows, best_config_id))
    _write_json(output_dir / "fixed_pbrs_v2_static_grid_selection_summary.json", selection_summary)
    return selection_summary


def _final_mode(results_root: Path, output_dir: Path, best_config_id: str, seeds: List[int]) -> Dict[str, Any]:
    rows = [summarize_run(results_root, config_id=best_config_id, seed=seed) for seed in seeds]
    aggregate = {
        "best_config_id": best_config_id,
        "config_details": get_static_config(best_config_id),
        "seeds": list(seeds),
        "final_test_sparse_return_mean": _mean(row.final_test_sparse_return for row in rows),
        "final_test_sparse_return_std": _std(row.final_test_sparse_return for row in rows),
        "max_test_sparse_return_mean": _mean(row.max_test_sparse_return for row in rows),
        "max_test_sparse_return_std": _std(row.max_test_sparse_return for row in rows),
        "selected_path_auc_mean_mean": _mean(row.selected_path_auc_mean for row in rows),
        "selected_path_auc_mean_std": _std(row.selected_path_auc_mean for row in rows),
        "provenance_checks": {
            "pbrs_version_lbf_pbrs_v2_all": all(row.pbrs_version_ok for row in rows),
            "pbrs_v2_runtime_used_mean_exists_all": all(row.pbrs_runtime_metric_present for row in rows),
            "pbrs_weights_metrics_exist_all": all(row.pbrs_weights_metrics_present for row in rows),
            "config_eval_use_pbrs_false_all": all(row.eval_use_pbrs_false for row in rows),
            "test_eval_use_pbrs_mean_all_zero_when_present": all(
                row.test_eval_use_pbrs_mean is None or abs(float(row.test_eval_use_pbrs_mean)) <= 1e-9
                for row in rows
            ),
        },
        "runs": [row.to_dict() for row in rows],
        "llm_free_all": all(row.llm_free for row in rows),
    }
    _write_text(output_dir / "fixed_pbrs_v2_static_best_3seed_summary.md", _best_3seed_md(best_config_id, rows, aggregate))
    _write_json(output_dir / "fixed_pbrs_v2_static_best_3seed_summary.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fixed PBRS-v2 static-grid results")
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    parser.add_argument("--output-dir", default=str(ROOT / "reports"))
    parser.add_argument("--mode", choices=["seed1", "final"], required=True)
    parser.add_argument("--best-config-id", choices=list_config_ids(), default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.mode == "seed1":
        payload = _seed1_mode(results_root, output_dir)
    else:
        if not args.best_config_id:
            raise ValueError("--best-config-id is required for --mode final")
        payload = _final_mode(results_root, output_dir, args.best_config_id, list(args.seeds))
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
