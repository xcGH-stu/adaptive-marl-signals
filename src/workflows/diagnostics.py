from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from workflows.baseline_run_support import inspect_baseline_checkpoint_readiness
from workflows.stage_conditioned_branching import (
    _classify_existing_candidate_state,
    _launch_candidate_background_and_bind_run,
)
from workflows.train_checkpointing import normalize_base_train_config
from workflows.train_launcher import EPyMARLTrainLauncher


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGNOSTICS_ROOT = ROOT / "results" / "diagnostics"
DEFAULT_END_TO_END_WORKFLOW_DIR = ROOT / "results" / "end_to_end_workflows" / "adaptive_10x10_fullrun_02"
DEFAULT_ADAPTIVE_MANIFEST_PATH = (
    ROOT
    / "results"
    / "adaptive_checkpoint_replacement_workflows"
    / "adaptive_10x10_fullrun_02_stage234"
    / "adaptive_checkpoint_replacement_manifest.json"
)
DEFAULT_PYTHON = Path("/home/epymarl/miniconda3/envs/epymarl_new/bin/python")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def default_python_executable() -> str:
    if DEFAULT_PYTHON.exists():
        return str(DEFAULT_PYTHON)
    return "python"


def parse_seeds(seeds_arg: Optional[str], max_seeds: Optional[int], default_seed: int = 1) -> List[int]:
    seeds: List[int] = []
    if seeds_arg:
        for token in str(seeds_arg).split(","):
            token = token.strip()
            if not token:
                continue
            seeds.append(int(token))
    if not seeds:
        seeds = [int(default_seed)]
    if max_seeds is not None:
        seeds = seeds[: max(1, int(max_seeds))]
    deduped: List[int] = []
    for seed in seeds:
        if seed not in deduped:
            deduped.append(seed)
    return deduped


def load_default_dense_reference_context(
    workflow_dir: Path = DEFAULT_END_TO_END_WORKFLOW_DIR,
) -> Dict[str, Any]:
    stage1b = load_json(workflow_dir / "stage_1b_dense_reference.json")
    stage2 = load_json(workflow_dir / "stage_2_selection.json")
    dense_reference_run_dir = str(
        (((stage1b.get("dense_reference_run") or {}).get("result") or {}).get("run_reference") or {}).get("run_dir")
    )
    if not dense_reference_run_dir:
        raise ValueError(f"unable to resolve dense reference run dir from {workflow_dir / 'stage_1b_dense_reference.json'}")
    dense_config = load_json(Path(dense_reference_run_dir) / "config.json")
    env_args = deepcopy(dense_config.get("env_args") or {})
    selected_checkpoints = list(((stage2.get("selection") or {}).get("selected_checkpoints")) or [])
    return {
        "workflow_dir": str(workflow_dir),
        "dense_reference_run_dir": dense_reference_run_dir,
        "train_config_json": str(Path(dense_reference_run_dir) / "config.json"),
        "env_key": env_args.get("key"),
        "train_config_name": dense_config.get("name") or dense_config.get("config") or "qmix",
        "time_limit": env_args.get("time_limit"),
        "t_max": dense_config.get("t_max"),
        "test_interval": dense_config.get("test_interval"),
        "runner_log_interval": dense_config.get("runner_log_interval"),
        "learner_log_interval": dense_config.get("learner_log_interval"),
        "save_model_interval": dense_config.get("save_model_interval"),
        "seed": dense_config.get("seed"),
        "pbrs_config": {
            "beta": env_args.get("pbrs_beta"),
            "wc": env_args.get("pbrs_wc"),
            "wp": env_args.get("pbrs_wp"),
            "gamma": env_args.get("pbrs_gamma"),
            "variant": env_args.get("pbrs_variant"),
        },
        "selected_checkpoints": selected_checkpoints,
    }


def normalize_source_run_train_config(run_dir: Path) -> Dict[str, Any]:
    return normalize_base_train_config(load_json(run_dir / "config.json"))


def build_fixed_pbrs_train_config_from_source_run(
    *,
    source_run_dir: Path,
    workflow_id: str,
    workflow_round: int,
    label: str,
    seed: Optional[int] = None,
    t_max: Optional[int] = None,
    checkpoint_path: str = "",
    load_step: int = 0,
    save_model: Optional[bool] = None,
    save_model_interval: Optional[int] = None,
    save_final_model: Optional[bool] = None,
    local_results_path: Optional[str] = None,
    override_env_args: Optional[Dict[str, Any]] = None,
    override_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    train_config = normalize_source_run_train_config(source_run_dir)
    train_config["workflow_id"] = str(workflow_id)
    train_config["workflow_round"] = int(workflow_round)
    train_config["label"] = str(label)
    train_config["checkpoint_path"] = str(checkpoint_path)
    train_config["load_step"] = int(load_step)
    if seed is not None:
        train_config["seed"] = int(seed)
        metadata = deepcopy(train_config.get("experiment_metadata") or {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["seed"] = int(seed)
        train_config["experiment_metadata"] = metadata
    if save_model is not None:
        train_config["save_model"] = bool(save_model)
    if save_model_interval is not None:
        train_config["save_model_interval"] = int(save_model_interval)
    if local_results_path is not None:
        train_config["local_results_path"] = str(local_results_path)

    env_args = deepcopy(train_config.get("env_args") or {})
    if isinstance(override_env_args, dict):
        env_args.update(deepcopy(override_env_args))
    train_config["env_args"] = env_args

    overrides = deepcopy(train_config.get("overrides") or {})
    if t_max is not None:
        overrides["t_max"] = int(t_max)
    if save_final_model is not None:
        overrides["save_final_model"] = bool(save_final_model)
    if isinstance(override_overrides, dict):
        overrides.update(deepcopy(override_overrides))
    train_config["overrides"] = overrides
    return train_config


def build_generic_branch_plan(train_config: Dict[str, Any]) -> Dict[str, Any]:
    return {"resume_train_config": train_config}


def reconcile_background_task(
    *,
    launcher: EPyMARLTrainLauncher,
    train_config: Dict[str, Any],
    result_json_path: Path,
) -> Dict[str, Any]:
    state = _classify_existing_candidate_state(
        launcher=launcher,
        branch_plan=build_generic_branch_plan(train_config),
        branch_result_path=result_json_path,
    )
    payload: Dict[str, Any] = {
        "status": str(state.get("status") or "planned"),
        "result_json": str(result_json_path),
        "run_id": ((state.get("run_reference") or {}).get("run_id")),
        "run_dir": ((state.get("run_reference") or {}).get("run_dir")),
        "last_seen_sacred_status": state.get("last_seen_sacred_status"),
        "last_seen_process_alive": state.get("last_seen_process_alive"),
    }
    run_reference = state.get("run_reference") or {}
    run_dir = run_reference.get("run_dir")
    if payload["status"] == "running" and run_dir:
        run_path = Path(run_dir)
        run_json = run_path / "run.json"
        metrics_json = run_path / "metrics.json"
        info_json = run_path / "info.json"
        cout_txt = run_path / "cout.txt"
        heartbeat = run_reference.get("heartbeat")
        has_runtime_artifacts = (
            metrics_json.exists()
            or info_json.exists()
            or (cout_txt.exists() and cout_txt.stat().st_size > 0)
        )
        if heartbeat is None and not has_runtime_artifacts:
            payload["status"] = "planned"
            payload["run_id"] = None
            payload["run_dir"] = None
            payload["last_seen_sacred_status"] = None
            payload["last_seen_process_alive"] = None
    if state.get("branch_result") is not None:
        payload["branch_result"] = state.get("branch_result")
    return payload


def launch_background_task(
    *,
    launcher: EPyMARLTrainLauncher,
    train_config: Dict[str, Any],
    result_json_path: Path,
) -> Dict[str, Any]:
    launch_result = _launch_candidate_background_and_bind_run(
        launcher=launcher,
        branch_plan=build_generic_branch_plan(train_config),
        branch_result_path=result_json_path,
    )
    run_reference = launch_result.get("run_reference") or {}
    return {
        "status": "running",
        "result_json": str(result_json_path),
        "run_id": run_reference.get("run_id"),
        "run_dir": run_reference.get("run_dir"),
        "last_seen_sacred_status": "RUNNING" if run_reference.get("run_id") is not None else None,
        "last_seen_process_alive": True if run_reference.get("run_id") is not None else None,
        "launch_payload": launch_result.get("launch_payload"),
    }


def resolve_run_reference_from_run_dir(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    readiness = inspect_baseline_checkpoint_readiness(
        baseline_run_dir=str(run_path),
        baseline_result_json=None,
    )
    run_json = load_json(run_path / "run.json")
    return {
        "run_dir": str(run_path),
        "run_id": run_json.get("_id"),
        "run_json": str(run_path / "run.json"),
        "info_json": str(run_path / "info.json"),
        "metrics_json": str(run_path / "metrics.json"),
        "config_json": str(run_path / "config.json"),
        "checkpoint_root_dir": (readiness.get("checkpoint_summary") or {}).get("checkpoint_root_dir"),
        "available_checkpoint_steps": (readiness.get("checkpoint_summary") or {}).get("available_checkpoint_steps") or [],
        "latest_checkpoint_step": (readiness.get("checkpoint_summary") or {}).get("latest_checkpoint_step"),
        "loaded_checkpoint_path": (load_json(run_path / "info.json")).get("loaded_checkpoint_path"),
        "loaded_checkpoint_step": (load_json(run_path / "info.json")).get("loaded_checkpoint_step"),
        "status": run_json.get("status"),
        "heartbeat": run_json.get("heartbeat"),
        "stop_time": run_json.get("stop_time"),
    }


def load_metric_series(run_dir: str | Path, metric_name: str = "test_sparse_return_mean") -> Dict[str, Any]:
    run_path = Path(run_dir)
    metrics = load_json(run_path / "metrics.json")
    metric_payload = metrics.get(metric_name) or {}
    steps = [int(step) for step in (metric_payload.get("steps") or [])]
    values = [float(value) for value in (metric_payload.get("values") or [])]
    timestamps = list(metric_payload.get("timestamps") or [])
    return {
        "run_dir": str(run_path),
        "metric_name": metric_name,
        "steps": steps,
        "values": values,
        "timestamps": timestamps,
    }


def merge_run_series(run_dirs: Iterable[str | Path], metric_name: str = "test_sparse_return_mean") -> Dict[str, Any]:
    merged: Dict[int, float] = {}
    for run_dir in run_dirs:
        series = load_metric_series(run_dir, metric_name=metric_name)
        for step, value in zip(series["steps"], series["values"]):
            merged[int(step)] = float(value)
    ordered_steps = sorted(merged.keys())
    return {
        "run_dirs": [str(Path(run_dir)) for run_dir in run_dirs],
        "metric_name": metric_name,
        "steps": ordered_steps,
        "values": [merged[step] for step in ordered_steps],
    }


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def series_summary(series: Dict[str, Any], last_k: int = 5) -> Dict[str, Any]:
    steps = list(series.get("steps") or [])
    values = list(series.get("values") or [])
    if not steps or not values:
        return {
            "num_points": 0,
            "final": None,
            "best": None,
            "last_5_mean": None,
            "auc": None,
        }
    tail = values[-max(1, min(last_k, len(values))) :]
    if len(values) >= 2:
        integration_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        auc = float(integration_fn(np.array(values, dtype=float), np.array(steps, dtype=float)))
    else:
        auc = 0.0
    return {
        "num_points": len(values),
        "start_step": int(steps[0]),
        "end_step": int(steps[-1]),
        "final": float(values[-1]),
        "best": float(max(values)),
        "last_5_mean": _safe_mean([float(v) for v in tail]),
        "auc": auc,
    }


def aligned_curve_gap(series_a: Dict[str, Any], series_b: Dict[str, Any]) -> Dict[str, Any]:
    steps_a = np.array(series_a.get("steps") or [], dtype=float)
    values_a = np.array(series_a.get("values") or [], dtype=float)
    steps_b = np.array(series_b.get("steps") or [], dtype=float)
    values_b = np.array(series_b.get("values") or [], dtype=float)
    if len(steps_a) < 2 or len(steps_b) < 2:
        return {"max_pointwise_gap": None, "aligned_steps": [], "gaps": []}
    union_steps = np.array(sorted(set(steps_a.tolist()) | set(steps_b.tolist())), dtype=float)
    interp_a = np.interp(union_steps, steps_a, values_a)
    interp_b = np.interp(union_steps, steps_b, values_b)
    gaps = np.abs(interp_a - interp_b)
    return {
        "max_pointwise_gap": float(np.max(gaps)) if len(gaps) else None,
        "aligned_steps": union_steps.tolist(),
        "gaps": gaps.tolist(),
    }


def transient_drop_after_boundary(series: Dict[str, Any], boundary_step: int, k: int = 5) -> Dict[str, Any]:
    steps = [int(step) for step in (series.get("steps") or [])]
    values = [float(value) for value in (series.get("values") or [])]
    before_indices = [idx for idx, step in enumerate(steps) if step <= int(boundary_step)]
    after_indices = [idx for idx, step in enumerate(steps) if step > int(boundary_step)]
    if not before_indices or not after_indices:
        return {
            "boundary_step": int(boundary_step),
            "boundary_value": None,
            "next_k_mean": None,
            "next_k_min": None,
            "delta_mean": None,
            "delta_min": None,
        }
    boundary_value = float(values[before_indices[-1]])
    next_values = [values[idx] for idx in after_indices[: max(1, int(k))]]
    next_mean = _safe_mean([float(v) for v in next_values])
    next_min = float(min(next_values))
    return {
        "boundary_step": int(boundary_step),
        "boundary_value": boundary_value,
        "next_k_mean": next_mean,
        "next_k_min": next_min,
        "delta_mean": None if next_mean is None else float(next_mean - boundary_value),
        "delta_min": float(next_min - boundary_value),
    }


def improvement_slope(series: Dict[str, Any], start_step: Optional[int] = None, points: int = 5) -> Optional[float]:
    steps = [int(step) for step in (series.get("steps") or [])]
    values = [float(value) for value in (series.get("values") or [])]
    selected = [
        (step, value)
        for step, value in zip(steps, values)
        if start_step is None or step >= int(start_step)
    ][: max(2, int(points))]
    if len(selected) < 2:
        return None
    x0, y0 = selected[0]
    x1, y1 = selected[-1]
    if x1 == x0:
        return None
    return float((y1 - y0) / (x1 - x0))


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def maybe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clone_task(task: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(task)
