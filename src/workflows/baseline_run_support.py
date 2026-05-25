from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_baseline_paths(
    *,
    baseline_run_dir: Optional[str],
    baseline_result_json: Optional[str],
) -> Dict[str, str]:
    if baseline_run_dir:
        run_dir = Path(baseline_run_dir)
        config_json = run_dir / "config.json"
        info_json = run_dir / "info.json"
        metrics_json = run_dir / "metrics.json"
        missing = [
            str(path.name)
            for path in (config_json, info_json, metrics_json)
            if not path.exists()
        ]
        if missing:
            raise ValueError(
                "baseline run directory is missing required files: " + ", ".join(missing)
            )
        return {
            "baseline_run_dir": str(run_dir),
            "baseline_run_config_json": str(config_json),
            "baseline_run_info_json": str(info_json),
            "baseline_metrics_json": str(metrics_json),
        }

    if baseline_result_json:
        payload = load_json(Path(baseline_result_json))
        result = payload.get("result") or {}
        run_reference = result.get("run_reference") or {}
        config_json = run_reference.get("config_json")
        info_json = run_reference.get("info_json")
        metrics_json = run_reference.get("metrics_json")
        run_dir = run_reference.get("run_dir")
        if not (config_json and info_json and metrics_json):
            raise ValueError(
                "baseline result JSON does not contain result.run_reference config/info/metrics paths"
            )
        return {
            "baseline_run_dir": str(run_dir) if run_dir else "",
            "baseline_run_config_json": str(config_json),
            "baseline_run_info_json": str(info_json),
            "baseline_metrics_json": str(metrics_json),
        }

    raise ValueError("provide either --baseline-run-dir or --baseline-result-json")


def inspect_baseline_checkpoint_readiness(
    *,
    baseline_run_dir: Optional[str],
    baseline_result_json: Optional[str],
) -> Dict[str, Any]:
    resolved = resolve_baseline_paths(
        baseline_run_dir=baseline_run_dir,
        baseline_result_json=baseline_result_json,
    )
    config_payload = load_json(Path(resolved["baseline_run_config_json"]))
    info_payload = load_json(Path(resolved["baseline_run_info_json"]))
    metrics_payload = load_json(Path(resolved["baseline_metrics_json"]))

    checkpoint_root = info_payload.get("model_root_path")
    available_steps = []
    saved_model_steps = info_payload.get("saved_model_steps")
    if isinstance(saved_model_steps, list):
        for value in saved_model_steps:
            try:
                available_steps.append(int(value))
            except (TypeError, ValueError):
                continue
    available_steps = sorted(set(available_steps))

    checkpoint_root_exists = False
    discovered_steps = []
    if isinstance(checkpoint_root, str) and checkpoint_root:
        checkpoint_root_path = Path(checkpoint_root)
        checkpoint_root_exists = checkpoint_root_path.exists()
        if checkpoint_root_exists:
            for child in checkpoint_root_path.iterdir():
                if child.is_dir() and child.name.isdigit():
                    discovered_steps.append(int(child.name))
    discovered_steps = sorted(set(discovered_steps))
    all_steps = sorted(set(available_steps) | set(discovered_steps))

    metric_names = sorted(
        key for key, value in metrics_payload.items() if isinstance(value, dict)
    )
    blockers = []
    if not resolved["baseline_run_config_json"]:
        blockers.append("missing_config_json")
    if not resolved["baseline_run_info_json"]:
        blockers.append("missing_info_json")
    if not resolved["baseline_metrics_json"]:
        blockers.append("missing_metrics_json")
    if not checkpoint_root:
        blockers.append("missing_checkpoint_root_dir")
    elif not checkpoint_root_exists:
        blockers.append("checkpoint_root_missing_on_disk")
    if not available_steps:
        blockers.append("no_saved_model_steps_in_info")
    if checkpoint_root_exists and not discovered_steps:
        blockers.append("no_checkpoint_directories_found")
    if not metric_names:
        blockers.append("no_metric_payloads_found")

    branching_ready = bool(
        resolved["baseline_run_config_json"]
        and resolved["baseline_run_info_json"]
        and resolved["baseline_metrics_json"]
        and checkpoint_root_exists
        and all_steps
    )
    recommended_actions = []
    if "checkpoint_root_missing_on_disk" in blockers:
        recommended_actions.append(
            "verify local_results_path/models/<unique_token> exists on disk for this baseline run"
        )
    if "no_saved_model_steps_in_info" in blockers or "no_checkpoint_directories_found" in blockers:
        recommended_actions.append(
            "rerun sparse baseline with checkpoint saving enabled and verify save_model_interval is active"
        )
    if "no_metric_payloads_found" in blockers:
        recommended_actions.append(
            "verify Sacred metrics.json was written correctly before checkpoint selection"
        )
    return {
        "baseline_source": resolved,
        "config_summary": {
            "algorithm": config_payload.get("name") or config_payload.get("config"),
            "env_key": (
                (config_payload.get("env_args") or {}).get("key")
                if isinstance(config_payload.get("env_args"), dict)
                else None
            ),
            "t_max": config_payload.get("t_max"),
            "save_model": config_payload.get("save_model"),
            "save_model_interval": config_payload.get("save_model_interval"),
        },
        "checkpoint_summary": {
            "checkpoint_root_dir": checkpoint_root,
            "checkpoint_root_exists": checkpoint_root_exists,
            "saved_model_steps": available_steps,
            "discovered_checkpoint_steps": discovered_steps,
            "available_checkpoint_steps": all_steps,
            "latest_checkpoint_step": all_steps[-1] if all_steps else None,
        },
        "metrics_summary": {
            "metric_count": len(metric_names),
            "metric_names": metric_names,
        },
        "blockers": blockers,
        "recommended_actions": recommended_actions,
        "branching_ready": branching_ready,
    }


MANDATORY_CHECKPOINT_CANDIDATE_METRICS = [
    "sparse_return_mean",
    "test_sparse_return_mean",
]

OPTIONAL_CHECKPOINT_CANDIDATE_METRICS = [
    "return_mean",
    "test_return_mean",
    "ep_length_mean",
    "test_ep_length_mean",
]


def _candidate_steps_from_metrics_payload(
    *,
    metrics_payload: Dict[str, Any],
    selected_metrics: List[str],
) -> List[int]:
    candidate_steps: List[int] = []
    for metric_name in selected_metrics:
        payload = metrics_payload.get(metric_name)
        if not isinstance(payload, dict):
            continue
        for value in payload.get("steps") or []:
            try:
                step = int(value)
            except (TypeError, ValueError):
                continue
            if step >= 0:
                candidate_steps.append(step)
    return sorted(set(candidate_steps))


def build_baseline_checkpoint_candidates(
    *,
    baseline_run_dir: Optional[str],
    baseline_result_json: Optional[str],
    optional_metric_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    resolved = resolve_baseline_paths(
        baseline_run_dir=baseline_run_dir,
        baseline_result_json=baseline_result_json,
    )
    config_payload = load_json(Path(resolved["baseline_run_config_json"]))
    info_payload = load_json(Path(resolved["baseline_run_info_json"]))
    metrics_payload = load_json(Path(resolved["baseline_metrics_json"]))

    selected_metrics = list(MANDATORY_CHECKPOINT_CANDIDATE_METRICS)
    for metric_name in optional_metric_names or []:
        if metric_name in OPTIONAL_CHECKPOINT_CANDIDATE_METRICS and metric_name not in selected_metrics:
            selected_metrics.append(metric_name)
    checkpoint_steps = []
    for value in info_payload.get("saved_model_steps") or []:
        try:
            checkpoint_steps.append(int(value))
        except (TypeError, ValueError):
            continue
    checkpoint_steps = sorted(set(checkpoint_steps))
    candidate_source = "saved_model_steps"
    if not checkpoint_steps:
        checkpoint_steps = _candidate_steps_from_metrics_payload(
            metrics_payload=metrics_payload,
            selected_metrics=selected_metrics,
        )
        candidate_source = "metric_steps_fallback"
    candidates = []
    for step in checkpoint_steps:
        metric_snapshot = {}
        for metric_name in selected_metrics:
            payload = metrics_payload.get(metric_name)
            if not isinstance(payload, dict):
                continue
            snapshot = _summarize_metric_at_step(payload, step)
            if snapshot is not None:
                metric_snapshot[metric_name] = snapshot
        candidates.append(
            {
                "checkpoint_step": step,
                "metric_snapshot": metric_snapshot,
            }
        )

    return {
        "baseline_source": resolved,
        "config_summary": {
            "algorithm": config_payload.get("name") or config_payload.get("config"),
            "env_key": (
                (config_payload.get("env_args") or {}).get("key")
                if isinstance(config_payload.get("env_args"), dict)
                else None
            ),
            "t_max": config_payload.get("t_max"),
            "save_model": config_payload.get("save_model"),
            "save_model_interval": config_payload.get("save_model_interval"),
            "test_interval": config_payload.get("test_interval"),
            "runner_log_interval": config_payload.get("runner_log_interval"),
            "learner_log_interval": config_payload.get("learner_log_interval"),
        },
        "candidate_metric_names": selected_metrics,
        "mandatory_metric_names": list(MANDATORY_CHECKPOINT_CANDIDATE_METRICS),
        "optional_metric_names": [
            metric_name
            for metric_name in selected_metrics
            if metric_name not in MANDATORY_CHECKPOINT_CANDIDATE_METRICS
        ],
        "candidate_step_source": candidate_source,
        "checkpoint_candidates": candidates,
    }


def _summarize_metric_at_step(metric_payload: Dict[str, Any], checkpoint_step: int) -> Optional[Dict[str, Any]]:
    values = metric_payload.get("values")
    steps = metric_payload.get("steps")
    if not isinstance(values, list) or not isinstance(steps, list) or not values or not steps:
        return None

    paired = []
    for step, value in zip(steps, values):
        try:
            paired.append((int(step), float(value)))
        except (TypeError, ValueError):
            continue
    if not paired:
        return None

    latest_step = None
    latest_value = None
    hist_best = None
    hist_worst = None
    for step, value in paired:
        if step <= checkpoint_step:
            latest_step = step
            latest_value = value
            hist_best = value if hist_best is None else max(hist_best, value)
            hist_worst = value if hist_worst is None else min(hist_worst, value)
        else:
            break

    nearest_step, nearest_value = min(paired, key=lambda item: abs(item[0] - checkpoint_step))

    return {
        "latest_step_at_or_before_checkpoint": latest_step,
        "latest_value_at_or_before_checkpoint": latest_value,
        "historical_best_value_to_checkpoint": hist_best,
        "historical_worst_value_to_checkpoint": hist_worst,
        "nearest_logged_step": nearest_step,
        "nearest_logged_value": nearest_value,
        "step_gap_to_nearest_log": abs(nearest_step - checkpoint_step),
    }
