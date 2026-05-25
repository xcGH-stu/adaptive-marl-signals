from __future__ import annotations

import argparse
from copy import deepcopy
import json
from datetime import datetime, timezone
from pathlib import Path
import time
import sys
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.workflow_profiles import get_workflow_profile
from experiments.workflow_profiles import is_deprecated_workflow_profile
from experiments.workflow_profiles import is_full_clean_runnable_workflow_profile
from experiments.workflow_profiles import is_summary_only_sidecar_profile
from experiments.workflow_profiles import list_launchable_workflow_profiles
from experiments.workflow_profiles import list_workflow_profiles
from scripts.run_fixed_reward_baseline import build_fixed_reward_baseline_plan
from scripts.run_final_stage_conditioned_full_run import build_final_stage_conditioned_full_run_plan
from workflows.adaptive_checkpoint_replacement import build_adaptive_checkpoint_replacement_manifest
from workflows.clients.llm_routing import llm_routing_artifact_fields
from workflows.clients.llm_routing import resolve_llm_model_tier
from scripts.run_sparse_baseline_with_checkpoints import build_sparse_baseline_plan
from scripts.run_stage_conditioned_branching import run_stage_conditioned_branching
from workflows.baseline_run_support import build_baseline_checkpoint_candidates
from workflows.baseline_run_support import inspect_baseline_checkpoint_readiness
from workflows.phase1_method import build_dense_reference_artifact
from workflows.phase1_method import build_dense_validation_candidates
from workflows.phase1_method import build_run_summary_from_run_dir
from workflows.phase1_method import materialize_dense_validation_final_specs
from workflows.phase1_method import select_dense_reference_with_llm
from workflows.policy_guidance import generate_stage1_behavior_summary_artifact
from workflows.production_stage1b import is_dual_llm_stage1b_enabled
from workflows.production_stage1b import run_or_reconcile_dual_llm_stage1b
from workflows.stage_selection import build_dual_source_stage_selection_result
from workflows.stage_selection import build_fixed_intervention_stage_selection_result
from workflows.stage_selection import build_stage_selection_result
from workflows.stage_selection import StageSelectionError


DEFAULT_RESULTS_ROOT = ROOT / "results" / "end_to_end_workflows"
DEFAULT_TRAIN_PYTHON = Path("/home/epymarl/miniconda3/envs/epymarl_new/bin/python")
PORTABLE_STAGE1_BUNDLE_ROOT = SRC_DIR / "reuse_artifacts" / "stage1_sparse_baselines"
PORTABLE_STAGE1_BUNDLE_INDEX = PORTABLE_STAGE1_BUNDLE_ROOT / "index.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_python_executable() -> str:
    if DEFAULT_TRAIN_PYTHON.exists():
        return str(DEFAULT_TRAIN_PYTHON)
    return sys.executable


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_portable_stage1_bundle_index() -> Dict[str, Any]:
    if not PORTABLE_STAGE1_BUNDLE_INDEX.exists():
        raise FileNotFoundError(
            f"portable Stage1 bundle index missing: {PORTABLE_STAGE1_BUNDLE_INDEX}"
        )
    payload = _load_json(PORTABLE_STAGE1_BUNDLE_INDEX)
    bundles = payload.get("bundles")
    if not isinstance(bundles, list):
        raise ValueError("portable Stage1 bundle index must contain a 'bundles' list")
    return payload


def _resolve_portable_stage1_bundle(
    *,
    args: argparse.Namespace,
    reuse: Dict[str, Any],
    policy_guidance_enabled: bool,
) -> Dict[str, Any]:
    bundle_id = str(reuse.get("bundle_id") or "").strip()
    if not bundle_id:
        raise ValueError("stage1_reuse.bundle_id must be set for portable_stage1_bundle")

    index_payload = _load_portable_stage1_bundle_index()
    bundle_entry = next(
        (
            item
            for item in (index_payload.get("bundles") or [])
            if str((item or {}).get("bundle_id") or "") == bundle_id
        ),
        None,
    )
    if not isinstance(bundle_entry, dict):
        raise FileNotFoundError(f"portable Stage1 bundle not found: {bundle_id}")

    bundle_rel = str(bundle_entry.get("bundle_relative_path") or "").strip()
    if not bundle_rel:
        raise ValueError(f"portable Stage1 bundle missing relative path: {bundle_id}")
    bundle_dir = (PORTABLE_STAGE1_BUNDLE_ROOT / bundle_rel).resolve()
    bundle_manifest_path = bundle_dir / "stage1_reuse_manifest.json"
    if not bundle_manifest_path.exists():
        raise FileNotFoundError(
            f"portable Stage1 bundle manifest missing: {bundle_manifest_path}"
        )
    bundle_manifest = _load_json(bundle_manifest_path)
    if str(bundle_manifest.get("bundle_id") or "") != bundle_id:
        raise ValueError(f"portable Stage1 bundle id mismatch: {bundle_id}")

    algorithm = str(bundle_manifest.get("algorithm") or "")
    env_key = str(bundle_manifest.get("env_key") or "")
    source_seed = int(bundle_manifest.get("seed") or 0)
    time_limit = int(bundle_manifest.get("time_limit") or -1)
    if algorithm != str(args.train_config):
        raise ValueError("portable Stage1 bundle algorithm mismatch")
    if env_key != str(args.env_key):
        raise ValueError("portable Stage1 bundle env_key mismatch")
    if time_limit != int(args.time_limit):
        raise ValueError("portable Stage1 bundle time_limit mismatch")
    if source_seed != int(args.seed):
        raise ValueError("portable Stage1 bundle seed mismatch")
    if not bool(bundle_manifest.get("stage1_completed", False)):
        raise ValueError("portable Stage1 bundle stage1_completed must be true")
    if not bool(bundle_manifest.get("reusable_for_stage1b", False)):
        raise ValueError("portable Stage1 bundle not marked reusable_for_stage1b")

    paths = bundle_manifest.get("paths") or {}
    stage1_artifact_rel = str(paths.get("stage1_artifact") or "stage_1_sparse_baseline.json")
    source_stage1_artifact_path = (bundle_dir / stage1_artifact_rel).resolve()
    if not source_stage1_artifact_path.exists():
        raise FileNotFoundError(
            f"portable Stage1 bundle stage1 artifact missing: {source_stage1_artifact_path}"
        )
    source_stage1_payload = _load_json(source_stage1_artifact_path)

    behavior_summary_rel = str(paths.get("stage1_behavior_summary") or "").strip()
    source_behavior_summary_path = (
        str((bundle_dir / behavior_summary_rel).resolve()) if behavior_summary_rel else ""
    )
    if behavior_summary_rel and not Path(source_behavior_summary_path).exists():
        raise FileNotFoundError(
            f"portable Stage1 bundle behavior summary missing: {source_behavior_summary_path}"
        )
    if not policy_guidance_enabled:
        source_behavior_summary_path = ""

    sacred_run_dir_rel = str(paths.get("sacred_run_dir") or "sacred_run").strip()
    source_baseline_run_dir = str((bundle_dir / sacred_run_dir_rel).resolve())
    if not Path(source_baseline_run_dir).exists():
        raise FileNotFoundError(
            f"portable Stage1 bundle sacred run dir missing: {source_baseline_run_dir}"
        )

    return {
        "enabled": True,
        "reuse": reuse,
        "reuse_mode": "portable_stage1_bundle",
        "bundle_id": bundle_id,
        "bundle_dir": bundle_dir,
        "bundle_manifest": bundle_manifest,
        "source_type": "portable_stage1_bundle",
        "source_workflow_id": str(bundle_manifest.get("source_workflow_id") or ""),
        "source_seed": source_seed,
        "source_stage": "stage1",
        "source_results_root": str(bundle_dir),
        "allow_partial_source_workflow": False,
        "require_source_stage1_completed": True,
        "reuse_behavior_summary": bool(source_behavior_summary_path),
        "reuse_sparse_metrics": True,
        "source_workflow_dir": bundle_dir,
        "source_manifest": bundle_manifest,
        "source_stage1_payload": source_stage1_payload,
        "source_behavior_summary_path": source_behavior_summary_path,
        "source_stage1_artifact_path": source_stage1_artifact_path,
        "source_baseline_run_dir": source_baseline_run_dir,
    }


def _sacred_env_root(train_config: str, env_key: str) -> Path:
    return ROOT / "results" / "sacred" / train_config / f'"{env_key}"'


def _parse_iso_maybe(value: Any) -> str:
    return "" if value is None else str(value)


def _find_completed_stage1_run_dir(
    *,
    stage1_workflow_id: str,
    env_key: str,
    train_config: str,
) -> Optional[Path]:
    env_root = _sacred_env_root(train_config, env_key)
    if not env_root.exists():
        return None

    matches: list[tuple[str, Path]] = []
    for run_dir in env_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.isdigit():
            continue
        run_json_path = run_dir / "run.json"
        config_json_path = run_dir / "config.json"
        if not run_json_path.exists():
            continue
        try:
            run_payload = _load_json(run_json_path)
        except Exception:
            continue
        try:
            config_payload = _load_json(config_json_path) if config_json_path.exists() else (run_payload.get("config") or {})
        except Exception:
            config_payload = (run_payload.get("config") or {})
        workflow_id = str(config_payload.get("workflow_id") or "")
        status = str(run_payload.get("status") or "")
        if not workflow_id.startswith(stage1_workflow_id):
            continue
        if status != "COMPLETED":
            continue
        stop_time = _parse_iso_maybe(run_payload.get("stop_time") or run_payload.get("heartbeat"))
        matches.append((stop_time, run_dir))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def _find_completed_run_dir_for_workflow(
    *,
    workflow_id_prefix: str,
    env_key: str,
    train_config: str,
) -> Optional[Path]:
    env_root = _sacred_env_root(train_config, env_key)
    if not env_root.exists():
        return None

    matches: list[tuple[str, Path]] = []
    for run_dir in env_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.isdigit():
            continue
        run_json_path = run_dir / "run.json"
        config_json_path = run_dir / "config.json"
        if not run_json_path.exists():
            continue
        try:
            run_payload = _load_json(run_json_path)
        except Exception:
            continue
        try:
            config_payload = _load_json(config_json_path) if config_json_path.exists() else (run_payload.get("config") or {})
        except Exception:
            config_payload = (run_payload.get("config") or {})
        workflow_id = str(config_payload.get("workflow_id") or "")
        status = str(run_payload.get("status") or "")
        if not workflow_id.startswith(workflow_id_prefix):
            continue
        if status != "COMPLETED":
            continue
        stop_time = _parse_iso_maybe(run_payload.get("stop_time") or run_payload.get("heartbeat"))
        matches.append((stop_time, run_dir))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def _find_latest_run_dir_for_workflow(
    *,
    workflow_id_prefix: str,
    env_key: str,
    train_config: str,
) -> Optional[Path]:
    env_root = _sacred_env_root(train_config, env_key)
    if not env_root.exists():
        return None

    matches: list[tuple[int, str, Path]] = []
    for run_dir in env_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.isdigit():
            continue
        run_json_path = run_dir / "run.json"
        config_json_path = run_dir / "config.json"
        if not run_json_path.exists():
            continue
        try:
            run_payload = _load_json(run_json_path)
            config_payload = _load_json(config_json_path) if config_json_path.exists() else (run_payload.get("config") or {})
        except Exception:
            continue
        workflow_id = str(config_payload.get("workflow_id") or "")
        if not workflow_id.startswith(workflow_id_prefix):
            continue
        status = str(run_payload.get("status") or "")
        status_rank = 2 if status == "COMPLETED" else 1 if status == "RUNNING" else 0
        heartbeat = _parse_iso_maybe(run_payload.get("heartbeat") or run_payload.get("stop_time"))
        matches.append((status_rank, heartbeat, run_dir))

    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches[-1][2]


def _build_recovered_stage1_payload(
    *,
    baseline_run_dir: Path,
    optional_metric_names: list[str] | None,
) -> Dict[str, Any]:
    readiness = inspect_baseline_checkpoint_readiness(
        baseline_run_dir=str(baseline_run_dir),
        baseline_result_json=None,
    )
    checkpoint_candidates = build_baseline_checkpoint_candidates(
        baseline_run_dir=str(baseline_run_dir),
        baseline_result_json=None,
        optional_metric_names=optional_metric_names,
    )
    (baseline_run_dir / "baseline_checkpoint_candidates.json").write_text(
        json.dumps(checkpoint_candidates, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    run_json = _load_json(baseline_run_dir / "run.json")
    config_json = _load_json(baseline_run_dir / "config.json")
    return {
        "recovered": True,
        "baseline_readiness": readiness,
        "baseline_checkpoint_candidates": checkpoint_candidates,
        "result": {
            "run_reference": {
                "run_id": run_json.get("_id"),
                "run_dir": str(baseline_run_dir),
                "checkpoint_root_dir": (readiness.get("checkpoint_summary") or {}).get("checkpoint_root_dir"),
                "latest_checkpoint_step": (readiness.get("checkpoint_summary") or {}).get("latest_checkpoint_step"),
                "workflow_id": (config_json.get("workflow_id") or run_json.get("experiment", {}).get("name")),
            }
        },
    }


def _build_existing_stage1_payload(
    *,
    baseline_run_dir: Path,
) -> Dict[str, Any]:
    run_json = _load_json(baseline_run_dir / "run.json")
    config_json = _load_json(baseline_run_dir / "config.json")
    payload: Dict[str, Any] = {
        "recovered": True,
        "result": {
            "run_reference": {
                "run_id": run_json.get("_id"),
                "run_dir": str(baseline_run_dir),
                "workflow_id": (config_json.get("workflow_id") or run_json.get("experiment", {}).get("name")),
            }
        },
    }
    try:
        readiness = inspect_baseline_checkpoint_readiness(
            baseline_run_dir=str(baseline_run_dir),
            baseline_result_json=None,
        )
        payload["baseline_readiness"] = readiness
    except Exception:
        pass
    return payload


def _build_recovered_stage5_payload(*, final_run_dir: Path) -> Dict[str, Any]:
    run_json = _load_json(final_run_dir / "run.json")
    config_json = _load_json(final_run_dir / "config.json")
    info_json_path = final_run_dir / "info.json"
    info_payload = _load_json(info_json_path) if info_json_path.exists() else {}
    return {
        "recovered": True,
        "result": {
            "run_reference": {
                "run_id": run_json.get("_id"),
                "run_dir": str(final_run_dir),
                "workflow_id": (config_json.get("workflow_id") or run_json.get("experiment", {}).get("name")),
            },
            "metrics_summary": info_payload,
        },
    }


def _run_dir_completed(run_dir: str | Path | None) -> bool:
    if not run_dir:
        return False
    run_json_path = Path(str(run_dir)) / "run.json"
    if not run_json_path.exists():
        return False
    try:
        run_payload = _load_json(run_json_path)
    except Exception:
        return False
    return str(run_payload.get("status") or "") == "COMPLETED"


def _stage1_reuse_config(args: argparse.Namespace) -> Dict[str, Any]:
    return deepcopy(getattr(args, "stage1_reuse", None) or {})


def _stage1_sparse_stage_satisfied(manifest: Dict[str, Any]) -> bool:
    status = str((((manifest.get("stages") or {}).get("stage_1_sparse_baseline") or {}).get("status")) or "")
    return status in {"completed", "skipped_reused"}


def _set_stage1_sparse_reuse_summary(
    manifest: Dict[str, Any],
    *,
    training_started: bool,
    reuse_enabled: bool,
    reuse_validated: bool,
    count_as_evaluation_baseline: bool,
) -> None:
    manifest["workflow_summary"] = {
        **deepcopy(manifest.get("workflow_summary") or {}),
        "stage1_sparse_training_started": bool(training_started),
        "stage1_sparse_reuse_enabled": bool(reuse_enabled),
        "stage1_sparse_reuse_validated": bool(reuse_validated),
        "stage1_sparse_reuse_count_as_evaluation_baseline": bool(
            count_as_evaluation_baseline
        ),
    }


def _configure_stage1_behavior_summary_reuse(
    args: argparse.Namespace,
    *,
    summary_path: str,
) -> None:
    setattr(args, "policy_guidance_stage1_behavior_summary_path", str(summary_path))
    setattr(args, "policy_guidance_stage1_behavior_summary_reuse_allowed", True)
    setattr(args, "allow_reuse_stage1_behavior_summary", True)
    setattr(args, "policy_guidance_stage1_behavior_summary_must_be_fresh", False)
    phase1_method = deepcopy(getattr(args, "phase1_method", None) or {})
    phase1_method["policy_guidance_stage1_behavior_summary_path"] = str(summary_path)
    phase1_method["policy_guidance_stage1_behavior_summary_reuse_allowed"] = True
    phase1_method["allow_reuse_stage1_behavior_summary"] = True
    phase1_method["policy_guidance_stage1_behavior_summary_must_be_fresh"] = False
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["policy_guidance_stage1_behavior_summary_path"] = str(summary_path)
    adaptive_replacement["policy_guidance_stage1_behavior_summary_reuse_allowed"] = True
    adaptive_replacement["allow_reuse_stage1_behavior_summary"] = True
    adaptive_replacement["policy_guidance_stage1_behavior_summary_must_be_fresh"] = False
    phase1_method["adaptive_replacement"] = adaptive_replacement
    setattr(args, "phase1_method", phase1_method)


def _record_stage1_reuse_blocker(
    manifest: Dict[str, Any],
    *,
    workflow_dir: Path,
    message: str,
) -> None:
    blockers = list(manifest.get("blockers") or [])
    blockers.append(
        {
            "stage": "stage_1_sparse_baseline",
            "kind": "stage1_sparse_reuse_validation_failed",
            "message": str(message),
        }
    )
    manifest["blockers"] = blockers
    _append_timeline(
        workflow_dir,
        "stage1_sparse_reuse_blocked",
        {"message": str(message)},
    )


def _inspect_stage1_reuse_source(args: argparse.Namespace) -> Dict[str, Any]:
    reuse = _stage1_reuse_config(args)
    if not bool(reuse.get("enabled", False)):
        return {"enabled": False}

    reuse_mode = str(reuse.get("reuse_mode") or "")
    count_as_evaluation_baseline = bool(reuse.get("count_as_evaluation_baseline", True))
    reuse_behavior_summary = bool(reuse.get("reuse_behavior_summary", False))
    reuse_sparse_metrics = bool(reuse.get("reuse_sparse_metrics_for_context", False))
    skip_sparse_training = bool(reuse.get("skip_sparse_training", False))
    policy_guidance_enabled = bool(_policy_guidance_setting(args, "enable_policy_guidance", False))

    if reuse_mode not in {
        "shared_sparse_diagnosis",
        "stage1_completed_artifact_reuse",
        "portable_stage1_bundle",
    }:
        raise ValueError(f"unsupported stage1_reuse mode: {reuse_mode!r}")
    if not skip_sparse_training:
        raise ValueError("stage1_reuse.skip_sparse_training must be true")
    if count_as_evaluation_baseline:
        raise ValueError(
            "stage1_reuse.count_as_evaluation_baseline must be false for shared_sparse_diagnosis"
        )
    if not policy_guidance_enabled and reuse_behavior_summary:
        raise ValueError(
            "reward-only workflows must not load shared stage1 behavior summaries as policy evidence"
        )

    if reuse_mode == "portable_stage1_bundle":
        return _resolve_portable_stage1_bundle(
            args=args,
            reuse=reuse,
            policy_guidance_enabled=policy_guidance_enabled,
        )

    source_workflow_id = str(reuse.get("source_workflow_id") or "")
    source_seed = int(reuse.get("source_seed") or 0)
    source_stage = str(reuse.get("source_stage") or "stage1").strip()
    source_results_root = str(reuse.get("source_results_root") or getattr(args, "results_root"))
    allow_partial_source_workflow = bool(reuse.get("allow_partial_source_workflow", False))
    require_source_stage1_completed = bool(reuse.get("require_source_stage1_completed", True))
    if not source_workflow_id:
        raise ValueError("stage1_reuse.source_workflow_id must be set")
    if source_seed <= 0:
        raise ValueError("stage1_reuse.source_seed must be a positive integer")
    if source_stage != "stage1":
        raise ValueError("stage1_reuse.source_stage must be 'stage1'")

    source_workflow_dir = Path(source_results_root) / source_workflow_id
    source_manifest_path = source_workflow_dir / "workflow_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(
            f"shared sparse reuse source workflow does not exist: {source_manifest_path}"
        )
    source_manifest = _load_json(source_manifest_path)
    source_manifest_status = str(source_manifest.get("status") or "")
    if not allow_partial_source_workflow and source_manifest_status != "completed":
        raise ValueError(
            f"shared sparse reuse source workflow is not completed: {source_workflow_id}"
        )
    source_stage1 = (source_manifest.get("stages") or {}).get("stage_1_sparse_baseline") or {}
    if require_source_stage1_completed and str(source_stage1.get("status") or "") != "completed":
        raise ValueError(
            f"shared sparse reuse source stage1 sparse baseline is not completed: {source_workflow_id}"
        )

    source_config = source_manifest.get("config_summary") or {}
    if str(source_config.get("env_key") or "") != str(args.env_key):
        raise ValueError("shared sparse reuse env_key mismatch")
    if str(source_config.get("train_config") or "") != str(args.train_config):
        raise ValueError("shared sparse reuse algorithm mismatch")
    if int(source_config.get("time_limit") or -1) != int(args.time_limit):
        raise ValueError("shared sparse reuse time_limit mismatch")

    manifest_behavior_summary_path = str(
        ((source_manifest.get("artifacts") or {}).get("stage1_behavior_summary_path")) or ""
    ).strip()
    local_behavior_summary_path = source_workflow_dir / "stage1_sparse_policy_behavior_summary.json"
    source_behavior_summary_path = (
        manifest_behavior_summary_path
        if manifest_behavior_summary_path and Path(manifest_behavior_summary_path).exists()
        else str(local_behavior_summary_path)
    )
    if reuse_behavior_summary and not Path(source_behavior_summary_path).exists():
        raise FileNotFoundError(
            f"shared sparse reuse behavior summary missing: {source_behavior_summary_path}"
        )

    source_stage1_artifact_path = source_workflow_dir / "stage_1_sparse_baseline.json"
    source_baseline_run_dir = str(
        ((source_manifest.get("artifacts") or {}).get("baseline_run_dir")) or ""
    )
    if reuse_sparse_metrics and not source_baseline_run_dir:
        raise ValueError("shared sparse reuse source baseline_run_dir missing")
    if require_source_stage1_completed and not source_stage1_artifact_path.exists():
        raise FileNotFoundError(
            f"shared sparse reuse source stage1 artifact missing: {source_stage1_artifact_path}"
        )

    source_stage1_payload = (
        _load_json(source_stage1_artifact_path)
        if source_stage1_artifact_path.exists()
        else {}
    )
    return {
        "enabled": True,
        "reuse": reuse,
        "reuse_mode": reuse_mode,
        "source_workflow_id": source_workflow_id,
        "source_seed": source_seed,
        "source_stage": source_stage,
        "source_results_root": source_results_root,
        "allow_partial_source_workflow": allow_partial_source_workflow,
        "require_source_stage1_completed": require_source_stage1_completed,
        "reuse_behavior_summary": reuse_behavior_summary,
        "reuse_sparse_metrics": reuse_sparse_metrics,
        "source_workflow_dir": source_workflow_dir,
        "source_manifest": source_manifest,
        "source_stage1_payload": source_stage1_payload,
        "source_behavior_summary_path": source_behavior_summary_path,
        "source_stage1_artifact_path": source_stage1_artifact_path,
        "source_baseline_run_dir": source_baseline_run_dir,
    }


def _resolve_shared_sparse_reuse(
    *,
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    paths: Dict[str, Path],
    workflow_dir: Path,
) -> Optional[Dict[str, Any]]:
    reuse = _stage1_reuse_config(args)
    if not bool(reuse.get("enabled", False)):
        return None
    inspection = _inspect_stage1_reuse_source(args)
    reuse_mode = str(inspection["reuse_mode"])
    source_workflow_id = str(inspection["source_workflow_id"])
    source_seed = int(inspection["source_seed"])
    source_stage = str(inspection["source_stage"])
    source_results_root = str(inspection["source_results_root"])
    allow_partial_source_workflow = bool(inspection["allow_partial_source_workflow"])
    require_source_stage1_completed = bool(inspection["require_source_stage1_completed"])
    reuse_behavior_summary = bool(inspection["reuse_behavior_summary"])
    reuse_sparse_metrics = bool(inspection["reuse_sparse_metrics"])
    bundle_id = str(inspection.get("bundle_id") or "")
    source_type = str(inspection.get("source_type") or "workflow")
    source_stage1_payload = deepcopy(inspection["source_stage1_payload"])
    source_behavior_summary_path = str(inspection["source_behavior_summary_path"])
    source_stage1_artifact_path = Path(inspection["source_stage1_artifact_path"])
    source_baseline_run_dir = str(inspection["source_baseline_run_dir"])
    policy_guidance_enabled = bool(_policy_guidance_setting(args, "enable_policy_guidance", False))
    target_stage1_payload = {
        "status": "skipped_reused",
        "reuse_mode": reuse_mode,
        "source_workflow_id": source_workflow_id,
        "source_seed": source_seed,
        "source_stage": source_stage,
        "source_results_root": source_results_root,
        "source_type": source_type,
        "bundle_id": bundle_id or None,
        "allow_partial_source_workflow": allow_partial_source_workflow,
        "require_source_stage1_completed": require_source_stage1_completed,
        "shared_calibration": {
            "source_seed": source_seed,
            "target_seed": int(args.seed),
            "seed_differs": int(args.seed) != int(source_seed),
        },
        "count_as_evaluation_baseline": False,
        "result": deepcopy(source_stage1_payload.get("result") or {}),
        "reuse_artifacts": {
            "stage1_behavior_summary_path": (
                source_behavior_summary_path if reuse_behavior_summary else None
            ),
            "stage1_sparse_metrics_path": (
                str(source_stage1_artifact_path) if reuse_sparse_metrics else None
            ),
        },
    }
    _save_json(paths["stage1_json"], target_stage1_payload)
    manifest["artifacts"]["baseline_run_dir"] = source_baseline_run_dir or None
    manifest["artifacts"]["stage1_behavior_summary_path"] = (
        source_behavior_summary_path if reuse_behavior_summary else None
    )
    manifest["artifacts"]["stage1_sparse_metrics_path"] = (
        str(source_stage1_artifact_path) if reuse_sparse_metrics else None
    )
    manifest["stages"]["stage_1_sparse_baseline"] = {
        **deepcopy(manifest["stages"]["stage_1_sparse_baseline"]),
        "status": "skipped_reused",
        "started_at": manifest["stages"]["stage_1_sparse_baseline"].get("started_at")
        or _utc_now_iso(),
        "completed_at": _utc_now_iso(),
        "artifact_json": str(paths["stage1_json"]),
        "error": None,
        "reuse_mode": reuse_mode,
        "reuse_source_workflow_id": source_workflow_id,
        "reuse_source_seed": source_seed,
        "reuse_source_stage": source_stage,
        "source_results_root": source_results_root,
        "source_type": source_type,
        "bundle_id": bundle_id or None,
        "reuse_artifacts": {
            "stage1_behavior_summary_path": (
                source_behavior_summary_path if reuse_behavior_summary else None
            ),
            "stage1_sparse_metrics_path": (
                str(source_stage1_artifact_path) if reuse_sparse_metrics else None
            ),
        },
        "count_as_evaluation_baseline": False,
        "shared_calibration": {
            "source_seed": source_seed,
            "target_seed": int(args.seed),
            "seed_differs": int(args.seed) != int(source_seed),
        },
    }
    _set_stage1_sparse_reuse_summary(
        manifest,
        training_started=False,
        reuse_enabled=True,
        reuse_validated=True,
        count_as_evaluation_baseline=False,
    )
    if policy_guidance_enabled and reuse_behavior_summary:
        _configure_stage1_behavior_summary_reuse(
            args,
            summary_path=source_behavior_summary_path,
        )
    _append_timeline(
        workflow_dir,
        "stage1_sparse_reused",
        {
            "reuse_mode": reuse_mode,
            "source_workflow_id": source_workflow_id,
            "source_seed": source_seed,
            "source_stage": source_stage,
            "source_results_root": source_results_root,
            "source_type": source_type,
            "bundle_id": bundle_id,
            "allow_partial_source_workflow": allow_partial_source_workflow,
            "require_source_stage1_completed": require_source_stage1_completed,
            "target_seed": int(args.seed),
            "count_as_evaluation_baseline": False,
            "stage1_behavior_summary_path": (
                source_behavior_summary_path if reuse_behavior_summary else ""
            ),
            "stage1_sparse_metrics_path": (
                str(source_stage1_artifact_path) if reuse_sparse_metrics else ""
            ),
        },
    )
    return target_stage1_payload


def _build_recovered_stage1b_payload(
    *,
    dense_reference_run_dir: Path,
    sparse_run_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    run_json = _load_json(dense_reference_run_dir / "run.json")
    config_json = _load_json(dense_reference_run_dir / "config.json")
    env_args = config_json.get("env_args") or {}
    selection_payload = {
        "selected_beta": env_args.get("pbrs_beta"),
        "selected_wc": env_args.get("pbrs_wc"),
        "selected_wp": env_args.get("pbrs_wp"),
        "selection_mode": "recovered_from_existing_run",
        "reason": "Recovered from an existing dense reference run.",
        "selection_goal": "Reuse an existing dense reference run as the boundary-calibration trajectory.",
    }
    return {
        "workflow_id": config_json.get("workflow_id") or run_json.get("experiment", {}).get("name"),
        "source_sparse_run_summary": deepcopy(sparse_run_summary or {}),
        "dense_reference_selection": selection_payload,
        "dense_reference_run": {
            "recovered": True,
            "result": {
                "run_reference": {
                    "run_id": run_json.get("_id"),
                    "run_dir": str(dense_reference_run_dir),
                    "workflow_id": config_json.get("workflow_id") or run_json.get("experiment", {}).get("name"),
                }
            },
        },
    }


def _stage2_requires_post_dense_dual_source(manifest: Dict[str, Any]) -> bool:
    phase1_method = (manifest.get("config_summary") or {}).get("phase1_method") or {}
    return str(phase1_method.get("checkpoint_selection_mode") or "") == "post_dense_dual_source"


def _is_adaptive_replacement_method(manifest: Dict[str, Any]) -> bool:
    phase1_method = (manifest.get("config_summary") or {}).get("phase1_method") or {}
    return str(phase1_method.get("method_version") or "") == "phase1_adaptive_replacement_v1"


def _is_adaptive_phase1_method(args: argparse.Namespace) -> bool:
    phase1_method = getattr(args, "phase1_method", None) or {}
    return str(phase1_method.get("method_version") or "") == "phase1_adaptive_replacement_v1"


def _adaptive_manifest_path(workflow_id: str) -> Path:
    return ROOT / "results" / "adaptive_checkpoint_replacement_workflows" / workflow_id / "adaptive_checkpoint_replacement_manifest.json"


def _adaptive_manifest_compat_path(workflow_id: str) -> Path:
    return ROOT / "results" / "adaptive_checkpoint_replacement_workflows" / workflow_id / "manifest.json"


def _load_adaptive_payload_from_wrapper(stage234_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(stage234_payload, dict):
        return None
    artifacts = stage234_payload.get("artifacts") or {}
    adaptive_manifest_json = artifacts.get("adaptive_checkpoint_replacement_manifest_json")
    if not adaptive_manifest_json:
        return None
    path = Path(str(adaptive_manifest_json))
    if not path.exists():
        compat_path = _adaptive_manifest_compat_path(str(stage234_payload.get("workflow_id") or ""))
        if compat_path.exists():
            path = compat_path
        else:
            return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _adaptive_progress_status(adaptive_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(adaptive_payload, dict):
        return {}
    decisions = list(adaptive_payload.get("decisions") or [])
    for decision in decisions:
        if bool(decision.get("completed")):
            continue
        running_candidates = [
            candidate for candidate in (decision.get("candidates") or []) if str(candidate.get("status") or "") == "running"
        ]
        planned_candidates = [
            candidate for candidate in (decision.get("candidates") or []) if str(candidate.get("status") or "") in {"pending", "planned"}
        ]
        continuation = decision.get("continuation") or {}
        continuation_status = str(continuation.get("status") or "")
        if running_candidates or planned_candidates:
            return {
                "decision_index": decision.get("decision_index"),
                "stage_label": decision.get("stage_label"),
                "active_phase": "branch_candidates",
                "running_candidates": len(running_candidates),
                "planned_candidates": len(planned_candidates),
            }
        if continuation_status and continuation_status != "completed":
            return {
                "decision_index": decision.get("decision_index"),
                "stage_label": decision.get("stage_label"),
                "active_phase": "mainline_continuation",
                "continuation_status": continuation_status,
                "continuation_run_id": continuation.get("run_id"),
            }
    return {}


def _adaptive_completion_status(adaptive_payload: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    readiness = ((adaptive_payload or {}).get("readiness") or {}) if isinstance(adaptive_payload, dict) else {}
    final_spec_json = (((adaptive_payload or {}).get("artifacts") or {}).get("adaptive_final_spec_json")) if isinstance(adaptive_payload, dict) else None
    final_run_json = (((adaptive_payload or {}).get("artifacts") or {}).get("final_mainline_run_json")) if isinstance(adaptive_payload, dict) else None
    return {
        "stage_2_selection": True,
        "stage_3_branching": bool(readiness.get("all_decisions_completed")),
        "stage_4_final_spec": bool(final_spec_json),
        "stage_5_final_full_run": bool(readiness.get("final_mainline_completed")) and bool(final_run_json),
    }


def _stage2_artifact_is_compatible(
    payload: Optional[Dict[str, Any]],
    *,
    manifest: Dict[str, Any],
) -> bool:
    if not isinstance(payload, dict):
        return False
    if not _stage2_requires_post_dense_dual_source(manifest):
        return True
    return str(payload.get("selection_source") or "") == "dense_reference_dual_source"


def _native_stage_schedule_signature(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    schedule = payload.get("native_original_pbrs_stage_schedule")
    if not isinstance(schedule, dict):
        return None
    return {
        "stage_boundary_1": int(schedule.get("stage_boundary_1")),
        "stage_boundary_2": int(schedule.get("stage_boundary_2")),
        "beta": {
            "early": float(schedule["beta"]["early"]),
            "mid": float(schedule["beta"]["mid"]),
            "late": float(schedule["beta"]["late"]),
        },
        "wc": {
            "early": float(schedule["wc"]["early"]),
            "mid": float(schedule["wc"]["mid"]),
            "late": float(schedule["wc"]["late"]),
        },
        "wp": {
            "early": float(schedule["wp"]["early"]),
            "mid": float(schedule["wp"]["mid"]),
            "late": float(schedule["wp"]["late"]),
        },
    }


def _run_config_stage_schedule_signature(config_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    env_args = config_payload.get("env_args") or {}
    if not env_args.get("pbrs_stage_schedule_enabled"):
        return None
    try:
        return {
            "stage_boundary_1": int(env_args.get("pbrs_stage_boundary_1")),
            "stage_boundary_2": int(env_args.get("pbrs_stage_boundary_2")),
            "beta": {
                "early": float(env_args.get("pbrs_stage_beta_early")),
                "mid": float(env_args.get("pbrs_stage_beta_mid")),
                "late": float(env_args.get("pbrs_stage_beta_late")),
            },
            "wc": {
                "early": float(env_args.get("pbrs_stage_wc_early")),
                "mid": float(env_args.get("pbrs_stage_wc_mid")),
                "late": float(env_args.get("pbrs_stage_wc_late")),
            },
            "wp": {
                "early": float(env_args.get("pbrs_stage_wp_early")),
                "mid": float(env_args.get("pbrs_stage_wp_mid")),
                "late": float(env_args.get("pbrs_stage_wp_late")),
            },
        }
    except (TypeError, ValueError):
        return None


def _recover_stage234_payload(
    *,
    branching_workflow_id: str,
    results_root: str,
) -> Optional[Dict[str, Any]]:
    workflow_dir = Path(results_root) / branching_workflow_id
    if not workflow_dir.exists():
        return None

    selection_json = workflow_dir / "stage_selection_result.json"
    branching_manifest_json = workflow_dir / "stage_conditioned_branching_manifest.json"
    final_spec_json = workflow_dir / "final_stage_conditioned_reward_spec.json"
    if not selection_json.exists() and not branching_manifest_json.exists() and not final_spec_json.exists():
        return None

    payload: Dict[str, Any] = {
        "workflow_id": branching_workflow_id,
        "recovered": True,
        "artifacts": {
            "workflow_dir": str(workflow_dir),
            "stage_selection_result_json": str(selection_json) if selection_json.exists() else None,
            "branching_manifest_json": str(branching_manifest_json) if branching_manifest_json.exists() else None,
            "final_stage_conditioned_reward_spec_json": str(final_spec_json) if final_spec_json.exists() else None,
        },
    }
    if selection_json.exists():
        payload["stage_selection_result"] = _load_json(selection_json)
    if branching_manifest_json.exists():
        payload["stage_conditioned_branching_manifest"] = _load_json(branching_manifest_json)
    if final_spec_json.exists():
        payload["final_stage_conditioned_reward_spec"] = _load_json(final_spec_json)
    return payload


def _stage234_payload_rank(
    payload: Optional[Dict[str, Any]],
    *,
    expected_field_rounds: list[str],
) -> tuple[int, int, int]:
    if not isinstance(payload, dict):
        return (0, 0, 0)
    completion = _stage234_completion_status(
        stage234_payload=payload,
        expected_field_rounds=expected_field_rounds,
    )
    artifacts = payload.get("artifacts") or {}
    has_final_spec = 1 if completion.get("stage_4_final_spec") else 0
    has_branching = 1 if completion.get("stage_3_branching") else 0
    has_selection = 1 if completion.get("stage_2_selection") else 0
    # Prefer more complete payloads, then prefer payloads with explicit final spec artifact.
    return (
        has_final_spec + has_branching + has_selection,
        1 if artifacts.get("final_stage_conditioned_reward_spec_json") else 0,
        1 if "native" in str(payload.get("workflow_id") or "") else 0,
    )


def _recover_best_stage234_payload(
    *,
    branching_workflow_id: str,
    results_root: str,
    expected_field_rounds: list[str],
) -> Optional[Dict[str, Any]]:
    root = Path(results_root)
    candidate_ids = [branching_workflow_id]
    if root.exists():
        sibling_ids = sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith(f"{branching_workflow_id}_native")
        )
        candidate_ids.extend(sibling_ids)

    best_payload: Optional[Dict[str, Any]] = None
    best_rank = (0, 0, 0)
    for candidate_id in candidate_ids:
        payload = _recover_stage234_payload(
            branching_workflow_id=candidate_id,
            results_root=results_root,
        )
        rank = _stage234_payload_rank(payload, expected_field_rounds=expected_field_rounds)
        if rank > best_rank:
            best_rank = rank
            best_payload = payload
    return best_payload


def _find_completed_final_run_dir_for_spec(
    *,
    final_spec_path: Optional[str],
    env_key: str,
    train_config: str,
    workflow_id_prefixes: list[str],
) -> Optional[Path]:
    env_root = _sacred_env_root(train_config, env_key)
    if not env_root.exists():
        return None

    final_spec_payload = None
    final_signature = None
    if final_spec_path:
        final_spec_file = Path(final_spec_path)
        if final_spec_file.exists():
            try:
                final_spec_payload = _load_json(final_spec_file)
                final_signature = _native_stage_schedule_signature(final_spec_payload)
            except Exception:
                final_signature = None

    matches: list[tuple[int, str, Path]] = []
    for run_dir in env_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.isdigit():
            continue
        run_json_path = run_dir / "run.json"
        config_json_path = run_dir / "config.json"
        if not run_json_path.exists() or not config_json_path.exists():
            continue
        try:
            run_payload = _load_json(run_json_path)
            config_payload = _load_json(config_json_path)
        except Exception:
            continue
        if str(run_payload.get("status") or "") != "COMPLETED":
            continue
        workflow_id = str(config_payload.get("workflow_id") or "")
        score = 0
        matched_prefix = any(workflow_id.startswith(prefix) for prefix in workflow_id_prefixes if prefix)
        if matched_prefix:
            score += 10
        if matched_prefix and "native_final_full_run" in workflow_id:
            score += 2
        run_signature = _run_config_stage_schedule_signature(config_payload)
        if final_signature is not None and run_signature == final_signature:
            score += 20
        if final_spec_payload is not None and run_signature is None:
            continue
        if score <= 0:
            continue
        stop_time = _parse_iso_maybe(run_payload.get("stop_time") or run_payload.get("heartbeat"))
        matches.append((score, stop_time, run_dir))

    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches[-1][2]


def _get_expected_field_rounds(manifest: Dict[str, Any]) -> list[str]:
    return list((manifest.get("config_summary") or {}).get("field_rounds") or ["beta", "wc", "wp"])


def _recover_stage234_round_manifests(*, workflow_dir: Path, expected_field_rounds: list[str]) -> list[Dict[str, Any]]:
    recovered: list[Dict[str, Any]] = []
    for round_index, field_name in enumerate(expected_field_rounds, start=1):
        round_manifest_path = workflow_dir / f"round_{round_index:02d}_{field_name}_manifest.json"
        if not round_manifest_path.exists():
            continue
        try:
            payload = _load_json(round_manifest_path)
        except Exception:
            continue
        recovered.append(payload)
    return recovered


def _stage234_progress_status(
    *,
    stage234_payload: Dict[str, Any],
    expected_field_rounds: list[str],
) -> Dict[str, Any]:
    artifacts = stage234_payload.get("artifacts") or {}
    workflow_dir_value = (artifacts.get("workflow_dir") or "")
    round_manifests: list[Dict[str, Any]] = []
    if workflow_dir_value:
        round_manifests = _recover_stage234_round_manifests(
            workflow_dir=Path(str(workflow_dir_value)),
            expected_field_rounds=expected_field_rounds,
        )
    if not round_manifests:
        branching_manifest = stage234_payload.get("stage_conditioned_branching_manifest") or {}
        rounds = list(branching_manifest.get("rounds") or [])
        if rounds:
            round_manifests = rounds

    completed_fields: list[str] = []
    active_field: Optional[str] = None
    active_gate: Optional[str] = None
    for payload in round_manifests:
        field_name = str(payload.get("field_name") or "")
        gates = payload.get("gates") or {}
        if bool(gates.get("ready_to_advance")):
            completed_fields.append(field_name)
            continue
        if active_field is None:
            active_field = field_name
            if not gates.get("dispatch_complete"):
                active_gate = "dispatch"
            elif not gates.get("collection_complete"):
                active_gate = "collection"
            elif not gates.get("summary_complete"):
                active_gate = "summary"
            else:
                active_gate = "advance"
            break
    return {
        "completed_fields": completed_fields,
        "active_field": active_field,
        "active_gate": active_gate,
        "round_manifest_count": len(round_manifests),
    }


def _stage234_completion_status(
    *,
    stage234_payload: Dict[str, Any],
    expected_field_rounds: list[str],
) -> Dict[str, bool]:
    artifacts = stage234_payload.get("artifacts") or {}
    stage2_complete = bool(artifacts.get("stage_selection_result_json"))

    branching_manifest = stage234_payload.get("stage_conditioned_branching_manifest") or {}
    field_round_summaries = list(branching_manifest.get("field_round_summaries") or [])
    if not field_round_summaries:
        workflow_dir_value = (artifacts.get("workflow_dir") or "")
        if workflow_dir_value:
            workflow_dir = Path(str(workflow_dir_value))
            recovered_summaries = []
            for round_index, field_name in enumerate(expected_field_rounds, start=1):
                summary_path = workflow_dir / f"field_round_{round_index:02d}_{field_name}_summary.json"
                if not summary_path.exists():
                    continue
                try:
                    recovered_summaries.append(_load_json(summary_path))
                except Exception:
                    continue
            if recovered_summaries:
                field_round_summaries = recovered_summaries
    observed_fields = [
        str((summary or {}).get("field_recommendation", {}).get("field_name") or "")
        for summary in field_round_summaries
        if (summary or {}).get("field_recommendation")
    ]
    readiness = branching_manifest.get("readiness") or {}
    if not readiness and field_round_summaries:
        readiness = {
            "ready_for_final_synthesis": all(
                bool((summary or {}).get("readiness", {}).get("ready_for_final_synthesis"))
                for summary in field_round_summaries
            )
        }
    stage3_complete = (
        bool(artifacts.get("branching_manifest_json"))
        and len(field_round_summaries) >= len(expected_field_rounds)
        and all(field_name in observed_fields for field_name in expected_field_rounds)
        and bool(readiness.get("ready_for_final_synthesis"))
    )

    stage4_complete = bool(artifacts.get("final_stage_conditioned_reward_spec_json"))
    return {
        "stage_2_selection": stage2_complete,
        "stage_3_branching": stage3_complete,
        "stage_4_final_spec": stage4_complete,
    }


def _append_timeline(workflow_dir: Path, event_type: str, payload: Dict[str, Any] | None = None) -> None:
    timeline_path = workflow_dir / "timeline.jsonl"
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": _utc_now_iso(),
        "event_type": event_type,
        "payload": payload or {},
    }
    with timeline_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")


def _default_stage_status() -> Dict[str, Any]:
    return {
        "status": "pending",
        "started_at": None,
        "completed_at": None,
        "artifact_json": None,
        "error": None,
    }


def _apply_planned_stage1_reuse_manifest(
    manifest: Dict[str, Any],
    *,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    planned = deepcopy(manifest)
    reuse = deepcopy(getattr(args, "stage1_reuse", None) or {})
    if not bool(reuse.get("enabled", False)):
        return planned
    validation_error = None
    inspection: Dict[str, Any] | None = None
    try:
        inspection = _inspect_stage1_reuse_source(args)
    except Exception as exc:
        validation_error = str(exc)
    stage = deepcopy((planned.get("stages") or {}).get("stage_1_sparse_baseline") or {})
    inspected_source_workflow_id = ""
    inspected_source_seed = 0
    if inspection is not None:
        inspected_source_workflow_id = str(inspection.get("source_workflow_id") or "")
        inspected_source_seed = int(inspection.get("source_seed") or 0)
    stage["status"] = "skipped_reused" if validation_error is None else "failed"
    stage["reuse_mode"] = str(reuse.get("reuse_mode") or "")
    stage["reuse_source_workflow_id"] = (
        inspected_source_workflow_id or str(reuse.get("source_workflow_id") or "")
    )
    stage["reuse_source_seed"] = inspected_source_seed or int(reuse.get("source_seed") or 0)
    stage["bundle_id"] = str(
        reuse.get("bundle_id")
        or (inspection or {}).get("bundle_id")
        or ""
    )
    stage["reuse_artifacts"] = {
        "stage1_behavior_summary_path": (
            None if inspection is None else inspection.get("source_behavior_summary_path")
        ),
        "stage1_sparse_metrics_path": (
            None
            if inspection is None
            else str(inspection.get("source_stage1_artifact_path") or "")
        ),
    }
    stage["count_as_evaluation_baseline"] = bool(
        reuse.get("count_as_evaluation_baseline", True)
    )
    stage["error"] = validation_error
    planned["stages"]["stage_1_sparse_baseline"] = stage
    if inspection is not None:
        planned["artifacts"]["baseline_run_dir"] = inspection.get("source_baseline_run_dir") or None
        planned["artifacts"]["stage1_behavior_summary_path"] = (
            inspection.get("source_behavior_summary_path")
            if bool(reuse.get("reuse_behavior_summary", False))
            else None
        )
        planned["artifacts"]["stage1_sparse_metrics_path"] = (
            str(inspection.get("source_stage1_artifact_path") or "")
            if bool(reuse.get("reuse_sparse_metrics_for_context", False))
            else None
        )
    planned["workflow_summary"] = {
        **deepcopy(planned.get("workflow_summary") or {}),
        "stage1_sparse_training_started": False,
        "stage1_sparse_reuse_enabled": True,
        "stage1_sparse_reuse_validated": validation_error is None,
        "stage1_sparse_reuse_count_as_evaluation_baseline": bool(
            reuse.get("count_as_evaluation_baseline", True)
        ),
        "blocked_for_execute": validation_error is not None,
        "safe_for_execute_after_api_recharge": validation_error is None,
    }
    if validation_error is not None:
        blockers = list(planned.get("blockers") or [])
        blockers.append(
            {
                "stage": "stage_1_sparse_baseline",
                "kind": "stage1_sparse_reuse_validation_failed",
                "message": validation_error,
            }
        )
        planned["blockers"] = blockers
    return planned


def _phase1_method_budget_value(
    args: argparse.Namespace,
    key: str,
) -> Optional[int]:
    phase1_method = deepcopy(getattr(args, "phase1_method", None) or {})
    value = phase1_method.get(key)
    if value is None:
        return None
    return int(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--python-executable", default=_default_python_executable())
    parser.add_argument(
        "--workflow-profile",
        choices=list_workflow_profiles(),
        default=None,
        help="Predefined reusable end-to-end workflow profile for a specific LBF configuration.",
    )

    parser.add_argument("--env-key", default="lbforaging:Foraging-8x8-2p-1f-v3")
    parser.add_argument("--train-config", default="qmix")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--time-limit", type=int, default=50)
    parser.add_argument("--t-max", type=int, default=2050000)
    parser.add_argument("--test-interval", type=int, default=25000)
    parser.add_argument("--runner-log-interval", type=int, default=25000)
    parser.add_argument("--learner-log-interval", type=int, default=25000)
    parser.add_argument("--save-model-interval", type=int, default=25000)
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false")
    parser.add_argument("--stage1-use-cuda", dest="stage1_use_cuda", action="store_true")
    parser.add_argument("--stage1-cpu", dest="stage1_use_cuda", action="store_false")
    parser.add_argument("--stage3-use-cuda", dest="stage3_use_cuda", action="store_true")
    parser.add_argument("--stage3-cpu", dest="stage3_use_cuda", action="store_false")
    parser.add_argument("--stage5-use-cuda", dest="stage5_use_cuda", action="store_true")
    parser.add_argument("--stage5-cpu", dest="stage5_use_cuda", action="store_false")
    parser.add_argument(
        "--native-original-pbrs-branching",
        dest="native_original_pbrs_branching",
        action="store_true",
    )
    parser.add_argument(
        "--legacy-stage3-reward-module-branching",
        dest="native_original_pbrs_branching",
        action="store_false",
    )
    parser.set_defaults(
        use_cuda=False,
        stage1_use_cuda=None,
        stage3_use_cuda=None,
        stage5_use_cuda=None,
        native_original_pbrs_branching=True,
    )

    parser.add_argument("--reward-paradigm", choices=["pbrs"], default="pbrs")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-count", type=int, default=3)
    parser.add_argument("--min-step-gap", type=int, default=50000)
    parser.add_argument("--max-points-per-metric", type=int, default=15)
    parser.add_argument("--preferred-metric", action="append", default=None)
    parser.add_argument("--checkpoint-optional-metric", action="append", default=None)
    parser.add_argument("--field-round", action="append", default=None)
    parser.add_argument("--branch-budget-steps", type=int, default=None)
    parser.add_argument("--max-parallel-candidates", type=int, default=2)
    parser.add_argument("--reuse-completed-candidates", action="store_true")

    parser.add_argument("--use-real-llm", action="store_true")
    parser.add_argument("--use-real-llm-for-field-analysis", action="store_true")
    parser.add_argument("--api-key-env", default="IUSEAPI_API_KEY")
    parser.add_argument("--base-url", default="https://www.iuseapi.com/v1")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--allow-formal-execute", action="store_true")
    parser.add_argument("--llm-model-tier-override", default=None)
    parser.add_argument("--llm-validation-provider", default=None)
    parser.add_argument("--llm-validation-model", default=None)
    parser.add_argument("--llm-validation-api-key-env", default=None)
    parser.add_argument("--llm-validation-base-url", default=None)
    parser.add_argument("--llm-formal-provider", default=None)
    parser.add_argument("--llm-formal-model", default=None)
    parser.add_argument("--llm-formal-api-key-env", default=None)
    parser.add_argument("--llm-formal-base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=5.0)
    parser.add_argument("--stage234-max-attempts", type=int, default=None)

    parser.add_argument("--baseline-id", default=None)
    parser.add_argument("--branching-workflow-id", default=None)
    parser.add_argument("--final-run-id", default=None)
    parser.add_argument("--existing-sparse-run-dir", default=None)
    parser.add_argument("--stage3-fork-spec-json", default=None)

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the full end-to-end workflow. Default behavior is plan-only.",
    )
    parser.add_argument("--loop-until-complete", action="store_true")
    parser.add_argument("--single-tick", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--max-supervisor-errors", type=int, default=10)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


def _apply_workflow_profile(args: argparse.Namespace) -> argparse.Namespace:
    if args.workflow_profile is None:
        if getattr(args, "stage1_use_cuda", None) is None:
            args.stage1_use_cuda = bool(args.use_cuda)
        if getattr(args, "stage5_use_cuda", None) is None:
            args.stage5_use_cuda = bool(args.use_cuda)
        if not hasattr(args, "checkpointing_strategy") or args.checkpointing_strategy is None:
            args.checkpointing_strategy = {
                "stage1_enable_checkpointing": False,
                "stage3_enable_checkpointing": False,
                "stage5_enable_checkpointing": False,
            }
        if not hasattr(args, "recovery_policy") or args.recovery_policy is None:
            args.recovery_policy = {"stage234_max_attempts": 3}
        if not hasattr(args, "resource_policy") or args.resource_policy is None:
            args.resource_policy = {
                "stage1_use_cuda": bool(args.stage1_use_cuda),
                "stage3_use_cuda": bool(args.stage3_use_cuda) if args.stage3_use_cuda is not None else None,
                "stage3_max_parallel_candidates": int(args.max_parallel_candidates or 0),
                "stage5_use_cuda": bool(args.stage5_use_cuda),
            }
        return args
    expects_full_clean = "full_clean" in str(args.workflow_profile or "").lower()
    if expects_full_clean and not is_full_clean_runnable_workflow_profile(str(args.workflow_profile)):
        raise ValueError(
            f"workflow profile {args.workflow_profile!r} is not configured as a full clean final-continuation run"
        )
    if expects_full_clean and is_summary_only_sidecar_profile(str(args.workflow_profile)):
        raise ValueError(
            f"workflow profile {args.workflow_profile!r} is summary-only/sidecar and cannot be launched as full clean"
        )
    profile = get_workflow_profile(args.workflow_profile)
    for key, value in profile.items():
        if key == "profile_name":
            continue
        setattr(args, key, deepcopy(value) if isinstance(value, (dict, list)) else value)
    resource_policy = deepcopy(getattr(args, "resource_policy", None) or {})
    if getattr(args, "stage1_use_cuda", None) is None:
        args.stage1_use_cuda = bool(resource_policy.get("stage1_use_cuda", args.use_cuda))
    if getattr(args, "stage5_use_cuda", None) is None:
        args.stage5_use_cuda = bool(resource_policy.get("stage5_use_cuda", args.use_cuda))
    if getattr(args, "stage3_use_cuda", None) is None:
        stage3_policy = resource_policy.get("stage3_use_cuda")
        args.stage3_use_cuda = None if stage3_policy is None else bool(stage3_policy)
    if getattr(args, "max_parallel_candidates", None) is None:
        args.max_parallel_candidates = int(resource_policy.get("stage3_max_parallel_candidates") or 0) or None
    if getattr(args, "recovery_policy", None) is None:
        args.recovery_policy = {"stage234_max_attempts": 3}
    if getattr(args, "stage234_max_attempts", None) is None:
        args.stage234_max_attempts = int(
            ((getattr(args, "recovery_policy", None) or {}).get("stage234_max_attempts") or 3)
        )
    return args


def _manifest_paths(args: argparse.Namespace) -> Dict[str, Path]:
    workflow_dir = Path(args.results_root) / args.workflow_id
    return {
        "workflow_dir": workflow_dir,
        "manifest_path": workflow_dir / "workflow_manifest.json",
        "stage1_json": workflow_dir / "stage_1_sparse_baseline.json",
        "stage1b_json": workflow_dir / "stage_1b_dense_reference.json",
        "stage2_json": workflow_dir / "stage_2_selection.json",
        "stage234_json": workflow_dir / "stage_2_3_4_stage_conditioned.json",
        "stage4_dense_json": workflow_dir / "stage_4_dense_validation.json",
        "stage5_json": workflow_dir / "stage_5_final_full_run.json",
    }


def _apply_llm_routing(args: argparse.Namespace) -> argparse.Namespace:
    routing = resolve_llm_model_tier(
        {
            "workflow_profile": getattr(args, "workflow_profile", None),
            "profile_name": getattr(args, "profile_name", None),
            "workflow_id": getattr(args, "workflow_id", None),
            "stage_name": "end_to_end_workflow",
            "script_name": Path(__file__).name,
            "planned_only": not bool(getattr(args, "execute", False)),
            "execute": bool(getattr(args, "execute", False)),
            "allow_formal_execute": bool(getattr(args, "allow_formal_execute", False)),
            "formal_execute_guard_required": bool(
                getattr(args, "formal_execute_guard_required", False)
            ),
            "budget_profile": getattr(args, "budget_profile", None),
            "llm_model_routing": deepcopy(getattr(args, "llm_model_routing", None) or {}),
            "llm_model_tier_override": getattr(args, "llm_model_tier_override", None),
            "llm_validation_provider": getattr(args, "llm_validation_provider", None),
            "llm_validation_model": getattr(args, "llm_validation_model", None),
            "llm_validation_api_key_env": getattr(args, "llm_validation_api_key_env", None),
            "llm_validation_base_url": getattr(args, "llm_validation_base_url", None),
            "llm_formal_provider": getattr(args, "llm_formal_provider", None),
            "llm_formal_model": getattr(args, "llm_formal_model", None),
            "llm_formal_api_key_env": getattr(args, "llm_formal_api_key_env", None),
            "llm_formal_base_url": getattr(args, "llm_formal_base_url", None),
            "api_key_env": getattr(args, "api_key_env", None),
            "base_url": getattr(args, "base_url", None),
            "model": getattr(args, "model", None),
        }
    )
    args.llm_routing = routing
    args.llm_provider = routing["provider"]
    args.api_key_env = routing["api_key_env"]
    args.base_url = routing["base_url"]
    args.model = routing["model"]
    return args


def _default_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    baseline_id = args.baseline_id or f"{args.workflow_id}_stage1"
    dense_reference_id = f"{args.workflow_id}_stage1b_dense_reference"
    branching_workflow_id = args.branching_workflow_id or f"{args.workflow_id}_stage234"
    dense_validation_id = f"{args.workflow_id}_stage4_dense_validation"
    final_run_id = args.final_run_id or f"{args.workflow_id}_final"
    manifest = {
        "workflow_id": args.workflow_id,
        "workflow_kind": "end_to_end_stage_conditioned_workflow",
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "status": "initialized",
        "current_stage": None,
        "config_summary": {
            "workflow_profile": args.workflow_profile,
            "env_key": args.env_key,
            "train_config": args.train_config,
            "policy_guidance_enabled": bool(getattr(args, "enable_policy_guidance", False)),
            "pbrs_version": str(getattr(args, "pbrs_version", "lbf_pbrs_v2") or "lbf_pbrs_v2"),
            "dense_reference_sidecar_enabled": bool(
                getattr(args, "dense_reference_sidecar_enabled", False)
            ),
            "eval_use_pbrs": bool(getattr(args, "eval_use_pbrs", False)),
            "sparse_eval_only": bool(getattr(args, "sparse_eval_only", True)),
            "seed": args.seed,
            "time_limit": args.time_limit,
            "t_max": args.t_max,
            "stage1_sparse_t_max": int(getattr(args, "stage1_sparse_t_max", args.t_max)),
            "dense_reference_t_max": int(
                getattr(
                    args,
                    "dense_reference_t_max",
                    _phase1_method_budget_value(args, "dense_reference_target_t_max")
                    or args.t_max,
                )
            ),
            "adaptive_mainline_target_t_max": int(
                getattr(
                    args,
                    "adaptive_mainline_target_t_max",
                    _phase1_method_budget_value(args, "adaptive_mainline_target_t_max")
                    or args.t_max,
                )
            ),
            "use_cuda": bool(args.use_cuda),
            "stage1_use_cuda": bool(getattr(args, "stage1_use_cuda", args.use_cuda)),
            "stage3_use_cuda": getattr(args, "stage3_use_cuda", args.use_cuda),
            "stage5_use_cuda": bool(getattr(args, "stage5_use_cuda", args.use_cuda)),
            "native_original_pbrs_branching": bool(args.native_original_pbrs_branching),
            "max_rounds": args.max_rounds,
            "field_rounds": list(args.field_round or ["beta", "wc"]),
            "branch_budget_steps": args.branch_budget_steps,
            "resource_policy": deepcopy(getattr(args, "resource_policy", None) or {}),
            "checkpointing_strategy": deepcopy(getattr(args, "checkpointing_strategy", None) or {}),
            "recovery_policy": deepcopy(getattr(args, "recovery_policy", None) or {}),
            "phase1_method": deepcopy(getattr(args, "phase1_method", None) or {}),
            "stage1_reuse": deepcopy(getattr(args, "stage1_reuse", None) or {}),
            "stage234_max_attempts": args.stage234_max_attempts,
            "llm_routing": deepcopy(
                llm_routing_artifact_fields(getattr(args, "llm_routing", None) or {})
            ),
            "stage3_fork_spec_json": getattr(args, "stage3_fork_spec_json", None),
        },
        "workflow_summary": {
            "stage1_sparse_training_started": False,
            "formal_training_started": False,
            "real_llm_called": False,
            "stage1_sparse_reuse_enabled": bool(
                ((getattr(args, "stage1_reuse", None) or {}).get("enabled", False))
            ),
            "stage1_sparse_reuse_validated": False,
            "stage1_sparse_reuse_count_as_evaluation_baseline": bool(
                ((getattr(args, "stage1_reuse", None) or {}).get(
                    "count_as_evaluation_baseline",
                    True,
                ))
            ),
        },
        "stage_ids": {
            "stage_1_sparse_baseline": baseline_id,
            "stage_1b_dense_reference": dense_reference_id,
            "stage_2_selection": f"{args.workflow_id}_stage2_selection",
            "stage_2_3_4_stage_conditioned": branching_workflow_id,
            "stage_4_dense_validation": dense_validation_id,
            "stage_5_final_full_run": final_run_id,
        },
        "artifacts": {
            "baseline_run_dir": None,
            "dense_reference_run_dir": None,
            "stage1_behavior_summary_path": None,
            "stage1_sparse_metrics_path": None,
            "stage_selection_result_json": None,
            "stage_conditioned_branching_manifest_json": None,
            "final_stage_conditioned_reward_spec_json": None,
            "dense_validation_json": None,
            "dense_validation_winner_final_spec_json": None,
            "final_full_run_dir": None,
        },
        "stages": {
            "stage_1_sparse_baseline": _default_stage_status(),
            "stage_1b_dense_reference": _default_stage_status(),
            "stage_2_selection": _default_stage_status(),
            "stage_3_branching": _default_stage_status(),
            "stage_4_dense_validation": _default_stage_status(),
            "stage_4_final_spec": _default_stage_status(),
            "stage_5_final_full_run": _default_stage_status(),
        },
    }
    return _apply_planned_stage1_reuse_manifest(manifest, args=args)


def _load_or_init_manifest(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Path], bool]:
    paths = _manifest_paths(args)
    workflow_dir = paths["workflow_dir"]
    workflow_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths["manifest_path"]
    if manifest_path.exists():
        return _load_json(manifest_path), paths, False
    manifest = _default_manifest(args)
    _save_json(manifest_path, manifest)
    _append_timeline(workflow_dir, "workflow_initialized", {"workflow_id": args.workflow_id})
    return manifest, paths, True


def _save_manifest(paths: Dict[str, Path], manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now_iso()
    _save_json(paths["manifest_path"], manifest)


def _load_stage3_fork_spec_json(stage3_fork_spec_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(stage3_fork_spec_json, str) or not stage3_fork_spec_json.strip():
        return None
    payload = _load_json(Path(stage3_fork_spec_json))
    if not isinstance(payload, dict):
        raise ValueError("stage3 fork spec json must decode to an object")
    payload = deepcopy(payload)
    payload["fork_spec_json"] = str(stage3_fork_spec_json)
    return payload


def _infer_checkpoint_step_from_path(checkpoint_path: Any) -> Optional[int]:
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        return None
    try:
        return int(Path(checkpoint_path).name)
    except (TypeError, ValueError):
        return None


def _stage3_fork_dense_reference_run_dir(fork_spec: Dict[str, Any]) -> str:
    for key in ("dense_reference_run_dir", "source_run_dir", "endpoint_run_dir"):
        value = str(fork_spec.get(key) or "").strip()
        if value:
            return value
    raise ValueError("stage3 fork spec must include dense_reference_run_dir or source_run_dir")


def _build_stage3_fork_stage1_artifact(
    *,
    manifest: Dict[str, Any],
    fork_spec: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_run_dir = _stage3_fork_dense_reference_run_dir(fork_spec)
    return {
        "workflow_id": manifest["stage_ids"]["stage_1_sparse_baseline"],
        "status": "skipped_reused",
        "reuse_mode": "stage3_fork_bootstrap",
        "source_workflow_id": str(fork_spec.get("source_workflow_id") or ""),
        "count_as_evaluation_baseline": False,
        "reuse_artifacts": {},
        "result": {
            "run_reference": {
                "run_dir": baseline_run_dir,
            }
        },
    }


def _build_stage3_fork_stage1b_artifact(
    *,
    manifest: Dict[str, Any],
    fork_spec: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    endpoint_checkpoint_path = str(fork_spec.get("endpoint_checkpoint_path") or "").strip()
    endpoint_checkpoint_step = int(
        fork_spec.get("endpoint_checkpoint_step")
        or _infer_checkpoint_step_from_path(endpoint_checkpoint_path)
        or 0
    )
    winner_config = deepcopy(fork_spec.get("winner_config") or {})
    source_config = deepcopy(fork_spec.get("source_config") or winner_config)
    dense_reference_run_dir = _stage3_fork_dense_reference_run_dir(fork_spec)
    return {
        "workflow_id": manifest["stage_ids"]["stage_1b_dense_reference"],
        "stage1b_overall_status": "completed",
        "dense_reference_run_dir": dense_reference_run_dir,
        "dense_reference_selection": {
            "selected_beta": winner_config.get("beta"),
            "selected_wc": winner_config.get("wc"),
            "selected_wp": winner_config.get("wp"),
            "selection_source": "stage3_fork_bootstrap",
        },
        "selected_candidate_id": str(
            fork_spec.get("winner_candidate_id")
            or f"beta_{str(float(winner_config.get('beta', 0.0))).replace('.', 'p')}_wc_{str(float(winner_config.get('wc', 0.0))).replace('.', 'p')}"
        ),
        "selected_initial_dense_config": deepcopy(source_config),
        "selected_endpoint_checkpoint_path": endpoint_checkpoint_path,
        "selected_endpoint_checkpoint_step": endpoint_checkpoint_step,
        "dense_reference_source_checkpoint_path": str(
            fork_spec.get("dense_reference_source_checkpoint_path") or endpoint_checkpoint_path
        ),
        "adaptive_mainline_source_checkpoint_path": str(
            fork_spec.get("adaptive_mainline_source_checkpoint_path") or endpoint_checkpoint_path
        ),
        "adaptive_mainline_initial_pbrs_config": deepcopy(source_config),
        "fork_bootstrap": {
            "enabled": True,
            "fork_spec_json": fork_spec.get("fork_spec_json"),
            "workflow_profile": args.workflow_profile,
        },
    }


def _build_stage3_fork_stage2_artifact(
    *,
    manifest: Dict[str, Any],
    fork_spec: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    endpoint_checkpoint_path = str(fork_spec.get("endpoint_checkpoint_path") or "").strip()
    endpoint_checkpoint_step = int(
        fork_spec.get("endpoint_checkpoint_step")
        or _infer_checkpoint_step_from_path(endpoint_checkpoint_path)
        or 0
    )
    next_checkpoint_step = int(fork_spec.get("next_checkpoint_step") or 0)
    return {
        "workflow_id": manifest["stage_ids"]["stage_2_selection"],
        "selection_source": "stage3_fork_bootstrap",
        "selection_mode": "fixed_intervention_schedule",
        "sparse_summary": {
            "env_key": args.env_key,
            "train_config": args.train_config,
            "pbrs_version": str(getattr(args, "pbrs_version", "") or ""),
        },
        "selection": {
            "selected_checkpoints": [
                {
                    "name": "C1",
                    "step": endpoint_checkpoint_step,
                    "stage_label": "post_initialization_transition",
                    "reason": "Fork bootstrap from completed C1 branch endpoint.",
                    "branch_budget_feasible": True,
                },
                {
                    "name": "C2",
                    "step": next_checkpoint_step,
                    "stage_label": "late_stability",
                    "reason": "Continue Stage3 from fork bootstrap continuation target.",
                    "branch_budget_feasible": True,
                },
            ]
        },
        "selection_constraints": {
            "checkpoint_count": 2,
            "fixed_intervention_checkpoints": [endpoint_checkpoint_step, next_checkpoint_step],
        },
        "fork_bootstrap": {
            "enabled": True,
            "fork_spec_json": fork_spec.get("fork_spec_json"),
        },
    }


def _apply_stage3_fork_bootstrap_end_to_end(
    *,
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    paths: Dict[str, Path],
    workflow_dir: Path,
) -> Dict[str, Any]:
    fork_spec = _load_stage3_fork_spec_json(getattr(args, "stage3_fork_spec_json", None))
    if fork_spec is None:
        return manifest
    stage1_payload = _build_stage3_fork_stage1_artifact(manifest=manifest, fork_spec=fork_spec)
    stage1b_payload = _build_stage3_fork_stage1b_artifact(manifest=manifest, fork_spec=fork_spec, args=args)
    stage2_payload = _build_stage3_fork_stage2_artifact(manifest=manifest, fork_spec=fork_spec, args=args)
    _save_json(paths["stage1_json"], stage1_payload)
    _save_json(paths["stage1b_json"], stage1b_payload)
    _save_json(paths["stage2_json"], stage2_payload)
    manifest["artifacts"]["baseline_run_dir"] = (
        (((stage1_payload.get("result") or {}).get("run_reference") or {}).get("run_dir"))
    )
    manifest["artifacts"]["dense_reference_run_dir"] = str(stage1b_payload.get("dense_reference_run_dir") or "")
    manifest["artifacts"]["stage_selection_result_json"] = str(paths["stage2_json"])
    manifest["stages"]["stage_1_sparse_baseline"] = {
        **deepcopy(manifest["stages"]["stage_1_sparse_baseline"]),
        "status": "skipped_reused",
        "completed_at": _utc_now_iso(),
        "artifact_json": str(paths["stage1_json"]),
        "error": None,
        "reuse_mode": "stage3_fork_bootstrap",
        "reuse_source_workflow_id": str(fork_spec.get("source_workflow_id") or ""),
        "reuse_source_seed": int(args.seed or 0),
        "reuse_artifacts": {},
        "count_as_evaluation_baseline": False,
    }
    _mark_stage_completed(
        manifest,
        "stage_1b_dense_reference",
        artifact_json=paths["stage1b_json"],
        workflow_dir=workflow_dir,
    )
    _mark_stage_completed(
        manifest,
        "stage_2_selection",
        artifact_json=paths["stage2_json"],
        workflow_dir=workflow_dir,
    )
    _set_stage1_sparse_reuse_summary(
        manifest,
        training_started=False,
        reuse_enabled=True,
        reuse_validated=True,
        count_as_evaluation_baseline=False,
    )
    manifest.setdefault("workflow_summary", {})
    manifest["workflow_summary"]["real_llm_called"] = False
    manifest["workflow_summary"]["formal_training_started"] = False
    manifest["config_summary"]["stage3_fork_spec_json"] = fork_spec.get("fork_spec_json")
    manifest["config_summary"]["stage3_fork_bootstrap"] = True
    return manifest


def _refresh_manifest_config_summary(
    *,
    manifest: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    refreshed = deepcopy(manifest)
    refreshed["config_summary"] = {
        "workflow_profile": args.workflow_profile,
        "env_key": args.env_key,
        "train_config": args.train_config,
        "policy_guidance_enabled": bool(getattr(args, "enable_policy_guidance", False)),
        "pbrs_version": str(getattr(args, "pbrs_version", "lbf_pbrs_v2") or "lbf_pbrs_v2"),
        "dense_reference_sidecar_enabled": bool(
            getattr(args, "dense_reference_sidecar_enabled", False)
        ),
        "eval_use_pbrs": bool(getattr(args, "eval_use_pbrs", False)),
        "sparse_eval_only": bool(getattr(args, "sparse_eval_only", True)),
        "seed": args.seed,
        "time_limit": args.time_limit,
        "t_max": args.t_max,
        "stage1_sparse_t_max": int(getattr(args, "stage1_sparse_t_max", args.t_max)),
        "dense_reference_t_max": int(
            getattr(
                args,
                "dense_reference_t_max",
                _phase1_method_budget_value(args, "dense_reference_target_t_max")
                or args.t_max,
            )
        ),
        "adaptive_mainline_target_t_max": int(
            getattr(
                args,
                "adaptive_mainline_target_t_max",
                _phase1_method_budget_value(args, "adaptive_mainline_target_t_max")
                or args.t_max,
            )
        ),
        "use_cuda": bool(args.use_cuda),
        "stage1_use_cuda": bool(getattr(args, "stage1_use_cuda", args.use_cuda)),
        "stage3_use_cuda": getattr(args, "stage3_use_cuda", args.use_cuda),
        "stage5_use_cuda": bool(getattr(args, "stage5_use_cuda", args.use_cuda)),
        "native_original_pbrs_branching": bool(args.native_original_pbrs_branching),
        "max_rounds": args.max_rounds,
        "field_rounds": list(args.field_round or ["beta", "wc"]),
        "branch_budget_steps": args.branch_budget_steps,
        "resource_policy": deepcopy(getattr(args, "resource_policy", None) or {}),
        "checkpointing_strategy": deepcopy(getattr(args, "checkpointing_strategy", None) or {}),
        "recovery_policy": deepcopy(getattr(args, "recovery_policy", None) or {}),
        "phase1_method": deepcopy(getattr(args, "phase1_method", None) or {}),
        "stage1_reuse": deepcopy(getattr(args, "stage1_reuse", None) or {}),
        "stage234_max_attempts": args.stage234_max_attempts,
        "llm_routing": deepcopy(
            llm_routing_artifact_fields(getattr(args, "llm_routing", None) or {})
        ),
        "stage3_fork_spec_json": getattr(args, "stage3_fork_spec_json", None),
    }
    refreshed["workflow_summary"] = {
        "stage1_sparse_training_started": bool(
            ((manifest.get("workflow_summary") or {}).get("stage1_sparse_training_started", False))
        ),
        "formal_training_started": bool(
            ((manifest.get("workflow_summary") or {}).get("formal_training_started", False))
        ),
        "real_llm_called": bool(
            ((manifest.get("workflow_summary") or {}).get("real_llm_called", False))
        ),
        "stage1_sparse_reuse_enabled": bool(
            ((getattr(args, "stage1_reuse", None) or {}).get("enabled", False))
        ),
        "stage1_sparse_reuse_validated": bool(
            ((manifest.get("workflow_summary") or {}).get("stage1_sparse_reuse_validated", False))
        ),
        "stage1_sparse_reuse_count_as_evaluation_baseline": bool(
            ((getattr(args, "stage1_reuse", None) or {}).get(
                "count_as_evaluation_baseline",
                True,
            ))
        ),
    }
    return _apply_planned_stage1_reuse_manifest(refreshed, args=args)


def _normalize_manifest_from_local_artifacts(
    *,
    manifest: Dict[str, Any],
    paths: Dict[str, Path],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    normalized = deepcopy(manifest)
    stages = normalized.setdefault("stages", {})
    for stage_name in (
        "stage_1_sparse_baseline",
        "stage_1b_dense_reference",
        "stage_2_selection",
        "stage_3_branching",
        "stage_4_dense_validation",
        "stage_4_final_spec",
        "stage_5_final_full_run",
    ):
        stages.setdefault(stage_name, _default_stage_status())

    stage1_artifact = paths["stage1_json"]
    stage1b_artifact = paths["stage1b_json"]
    stage2_artifact = paths["stage2_json"]
    stage234_artifact = paths["stage234_json"]
    stage4_dense_artifact = paths["stage4_dense_json"]
    stage5_artifact = paths["stage5_json"]

    if stage1_artifact.exists():
        stages["stage_1_sparse_baseline"]["artifact_json"] = str(stage1_artifact)
        try:
            stage1_payload = _load_json(stage1_artifact)
            if str(stage1_payload.get("status") or "") == "skipped_reused":
                stages["stage_1_sparse_baseline"]["status"] = "skipped_reused"
                stages["stage_1_sparse_baseline"]["reuse_mode"] = str(
                    stage1_payload.get("reuse_mode") or ""
                )
                stages["stage_1_sparse_baseline"]["reuse_source_workflow_id"] = str(
                    stage1_payload.get("source_workflow_id") or ""
                )
                stages["stage_1_sparse_baseline"]["reuse_source_seed"] = int(
                    stage1_payload.get("source_seed") or 0
                )
                stages["stage_1_sparse_baseline"]["reuse_artifacts"] = deepcopy(
                    stage1_payload.get("reuse_artifacts") or {}
                )
                stages["stage_1_sparse_baseline"]["count_as_evaluation_baseline"] = bool(
                    stage1_payload.get("count_as_evaluation_baseline", False)
                )
                normalized["workflow_summary"] = {
                    **deepcopy(normalized.get("workflow_summary") or {}),
                    "stage1_sparse_training_started": False,
                    "stage1_sparse_reuse_enabled": True,
                    "stage1_sparse_reuse_validated": True,
                    "stage1_sparse_reuse_count_as_evaluation_baseline": bool(
                        stage1_payload.get("count_as_evaluation_baseline", False)
                    ),
                }
            else:
                stages["stage_1_sparse_baseline"]["status"] = "completed"
            normalized["artifacts"]["baseline_run_dir"] = (
                ((stage1_payload.get("result") or {}).get("run_reference") or {}).get("run_dir")
            )
            normalized["artifacts"]["stage1_behavior_summary_path"] = (
                (stage1_payload.get("reuse_artifacts") or {}).get("stage1_behavior_summary_path")
                or normalized["artifacts"].get("stage1_behavior_summary_path")
            )
            normalized["artifacts"]["stage1_sparse_metrics_path"] = (
                (stage1_payload.get("reuse_artifacts") or {}).get("stage1_sparse_metrics_path")
                or normalized["artifacts"].get("stage1_sparse_metrics_path")
            )
        except Exception:
            pass

    if stage2_artifact.exists():
        try:
            stage2_payload = _load_json(stage2_artifact)
        except Exception:
            stage2_payload = None
        if _stage2_artifact_is_compatible(stage2_payload, manifest=normalized):
            stages["stage_2_selection"]["status"] = "completed"
            stages["stage_2_selection"]["artifact_json"] = str(stage2_artifact)
            normalized["artifacts"]["stage_selection_result_json"] = str(stage2_artifact)
        else:
            stages["stage_2_selection"]["status"] = "pending"
            stages["stage_2_selection"]["artifact_json"] = None
            normalized["artifacts"]["stage_selection_result_json"] = None

    if stage1b_artifact.exists():
        try:
            stage1b_payload = _load_json(stage1b_artifact)
            stages["stage_1b_dense_reference"]["artifact_json"] = str(stage1b_artifact)
            if str(stage1b_payload.get("stage1b_mode") or "") == "dual_llm_initial_dense_search_v1":
                dense_reference_run_dir = str(stage1b_payload.get("dense_reference_run_dir") or "")
                if dense_reference_run_dir:
                    normalized["artifacts"]["dense_reference_run_dir"] = dense_reference_run_dir
                overall_status = str(stage1b_payload.get("stage1b_overall_status") or "running")
                stages["stage_1b_dense_reference"]["status"] = (
                    "completed" if overall_status == "completed" else overall_status
                )
            else:
                dense_reference_run_dir = (
                    (((stage1b_payload.get("dense_reference_run") or {}).get("result") or {}).get("run_reference") or {}).get("run_dir")
                )
                if dense_reference_run_dir:
                    normalized["artifacts"]["dense_reference_run_dir"] = dense_reference_run_dir
                    stages["stage_1b_dense_reference"]["status"] = (
                        "completed" if _run_dir_completed(dense_reference_run_dir) else "running"
                    )
        except Exception:
            pass

    if _stage2_requires_post_dense_dual_source(normalized) and stages["stage_1b_dense_reference"].get("status") != "completed":
        stages["stage_2_selection"]["status"] = "pending"
        stages["stage_2_selection"]["artifact_json"] = None
        stages["stage_3_branching"]["status"] = "pending"
        stages["stage_3_branching"]["artifact_json"] = None
        stages["stage_4_final_spec"]["status"] = "pending"
        stages["stage_4_final_spec"]["artifact_json"] = None
        stages["stage_4_dense_validation"]["status"] = "pending"
        stages["stage_4_dense_validation"]["artifact_json"] = None
        stages["stage_5_final_full_run"]["status"] = "pending"
        stages["stage_5_final_full_run"]["artifact_json"] = None
        normalized["artifacts"]["stage_selection_result_json"] = None
        normalized["artifacts"]["stage_conditioned_branching_manifest_json"] = None
        normalized["artifacts"]["final_stage_conditioned_reward_spec_json"] = None
        normalized["artifacts"]["dense_validation_json"] = None
        normalized["artifacts"]["dense_validation_winner_final_spec_json"] = None
        normalized["artifacts"]["final_full_run_dir"] = None

    if _is_adaptive_replacement_method(normalized):
        adaptive_wrapper_payload = None
        if stage234_artifact.exists():
            try:
                adaptive_wrapper_payload = _load_json(stage234_artifact)
            except Exception:
                adaptive_wrapper_payload = None
        adaptive_payload = _load_adaptive_payload_from_wrapper(adaptive_wrapper_payload)
        if adaptive_payload is None:
            adaptive_manifest_path = _adaptive_manifest_path(str(normalized.get("stage_ids", {}).get("stage_2_3_4_stage_conditioned", "")))
            if adaptive_manifest_path.exists():
                try:
                    adaptive_payload = _load_json(adaptive_manifest_path)
                except Exception:
                    adaptive_payload = None
        if isinstance(adaptive_payload, dict):
            adaptive_completion = _adaptive_completion_status(adaptive_payload)
            adaptive_progress = _adaptive_progress_status(adaptive_payload)
            adaptive_artifacts = adaptive_payload.get("artifacts") or {}
            adaptive_manifest_json = str(_adaptive_manifest_path(str(adaptive_payload.get("workflow_id") or "")))
            normalized["artifacts"]["stage_conditioned_branching_manifest_json"] = adaptive_manifest_json
            normalized["artifacts"]["final_stage_conditioned_reward_spec_json"] = adaptive_artifacts.get("adaptive_final_spec_json")
            final_mainline_json = adaptive_artifacts.get("final_mainline_run_json")
            normalized["artifacts"]["final_full_run_dir"] = None
            if final_mainline_json and Path(str(final_mainline_json)).exists():
                try:
                    final_mainline_payload = _load_json(Path(str(final_mainline_json)))
                except Exception:
                    final_mainline_payload = None
                normalized["artifacts"]["final_full_run_dir"] = (
                    (((final_mainline_payload or {}).get("result") or {}).get("run_reference") or {}).get("run_dir")
                )
            if adaptive_completion["stage_3_branching"]:
                stages["stage_3_branching"]["status"] = "completed"
                stages["stage_3_branching"]["artifact_json"] = adaptive_manifest_json
                stages["stage_3_branching"]["error"] = None
            else:
                stages["stage_3_branching"]["status"] = "running"
                stages["stage_3_branching"]["artifact_json"] = adaptive_manifest_json
                stages["stage_3_branching"]["error"] = None
                normalized["stage3_progress"] = adaptive_progress
            if adaptive_completion["stage_4_final_spec"]:
                stages["stage_4_final_spec"]["status"] = "completed"
                stages["stage_4_final_spec"]["artifact_json"] = adaptive_artifacts.get("adaptive_final_spec_json")
                stages["stage_4_final_spec"]["error"] = None
            else:
                stages["stage_4_final_spec"]["status"] = "pending"
                stages["stage_4_final_spec"]["artifact_json"] = None
                stages["stage_4_final_spec"]["error"] = None
            stages["stage_4_dense_validation"]["status"] = (
                "completed" if adaptive_completion["stage_3_branching"] else "pending"
            )
            stages["stage_4_dense_validation"]["artifact_json"] = adaptive_manifest_json
            stages["stage_4_dense_validation"]["error"] = None
            if adaptive_completion["stage_5_final_full_run"]:
                stages["stage_5_final_full_run"]["status"] = "completed"
                stages["stage_5_final_full_run"]["artifact_json"] = final_mainline_json
                stages["stage_5_final_full_run"]["error"] = None
            elif adaptive_artifacts.get("final_mainline_run_json"):
                stages["stage_5_final_full_run"]["status"] = "running"
                stages["stage_5_final_full_run"]["artifact_json"] = final_mainline_json
                stages["stage_5_final_full_run"]["error"] = None
            else:
                stages["stage_5_final_full_run"]["status"] = "pending"
                stages["stage_5_final_full_run"]["artifact_json"] = None
                stages["stage_5_final_full_run"]["error"] = None

    stage234_payload = None
    allow_stage234_recovery = (
        not _is_adaptive_replacement_method(normalized)
        and not (
            _stage2_requires_post_dense_dual_source(normalized)
            and stages["stage_1b_dense_reference"].get("status") != "completed"
        )
    )
    if allow_stage234_recovery:
        if stage234_artifact.exists():
            try:
                stage234_payload = _load_json(stage234_artifact)
            except Exception:
                stage234_payload = None
        best_recovered_stage234_payload = _recover_best_stage234_payload(
            branching_workflow_id=(normalized.get("stage_ids") or {}).get("stage_2_3_4_stage_conditioned", ""),
            results_root=str(ROOT / "results" / "stage_conditioned_workflows"),
            expected_field_rounds=_get_expected_field_rounds(normalized),
        )
        if _stage234_payload_rank(
            best_recovered_stage234_payload,
            expected_field_rounds=_get_expected_field_rounds(normalized),
        ) > _stage234_payload_rank(
            stage234_payload,
            expected_field_rounds=_get_expected_field_rounds(normalized),
        ):
            stage234_payload = best_recovered_stage234_payload
            if stage234_payload is not None:
                _save_json(stage234_artifact, stage234_payload)
    if isinstance(stage234_payload, dict):
        expected_field_rounds = _get_expected_field_rounds(normalized)
        completion_status = _stage234_completion_status(
            stage234_payload=stage234_payload,
            expected_field_rounds=expected_field_rounds,
        )
        progress_status = _stage234_progress_status(
            stage234_payload=stage234_payload,
            expected_field_rounds=expected_field_rounds,
        )
        artifacts = stage234_payload.get("artifacts") or {}
        normalized["artifacts"]["stage_selection_result_json"] = artifacts.get("stage_selection_result_json")
        normalized["artifacts"]["stage_conditioned_branching_manifest_json"] = artifacts.get("branching_manifest_json")
        normalized["artifacts"]["final_stage_conditioned_reward_spec_json"] = artifacts.get("final_stage_conditioned_reward_spec_json")
        if completion_status["stage_2_selection"]:
            stages["stage_2_selection"]["status"] = "completed"
            stages["stage_2_selection"]["artifact_json"] = (
                artifacts.get("stage_selection_result_json") or str(stage234_artifact)
            )
        if completion_status["stage_3_branching"]:
            stages["stage_3_branching"]["status"] = "completed"
            stages["stage_3_branching"]["artifact_json"] = (
                artifacts.get("branching_manifest_json") or str(stage234_artifact)
            )
        if completion_status["stage_4_final_spec"]:
            stages["stage_4_final_spec"]["status"] = "completed"
            stages["stage_4_final_spec"]["artifact_json"] = artifacts.get("final_stage_conditioned_reward_spec_json")
        elif stages["stage_4_final_spec"].get("status") != "completed":
            stages["stage_4_final_spec"]["status"] = "pending"
            stages["stage_4_final_spec"]["artifact_json"] = None
        if not completion_status["stage_3_branching"] and stages["stage_3_branching"].get("status") != "completed":
            stages["stage_3_branching"]["status"] = (
                "running" if progress_status.get("active_field") else "pending"
            )
            stages["stage_3_branching"]["artifact_json"] = (
                artifacts.get("branching_manifest_json") or stages["stage_3_branching"].get("artifact_json")
            )
            normalized.setdefault("stage3_progress", {})
            normalized["stage3_progress"] = progress_status
    else:
        if stages["stage_3_branching"].get("status") != "completed":
            stages["stage_3_branching"]["status"] = "pending"
            stages["stage_3_branching"]["artifact_json"] = None
        if stages["stage_4_final_spec"].get("status") != "completed":
            stages["stage_4_final_spec"]["status"] = "pending"
            stages["stage_4_final_spec"]["artifact_json"] = None

    if stages["stage_4_final_spec"].get("status") != "completed" and stages["stage_4_dense_validation"].get("status") != "completed":
        stages["stage_4_dense_validation"]["status"] = "pending"
        stages["stage_4_dense_validation"]["artifact_json"] = None

    if stage4_dense_artifact.exists():
        try:
            stage4_dense_payload = _load_json(stage4_dense_artifact)
            winner_final_spec = ((stage4_dense_payload.get("winner") or {}).get("final_spec_path"))
            if winner_final_spec:
                stages["stage_4_dense_validation"]["status"] = "completed"
                stages["stage_4_dense_validation"]["artifact_json"] = str(stage4_dense_artifact)
                normalized["artifacts"]["dense_validation_json"] = str(stage4_dense_artifact)
                normalized["artifacts"]["dense_validation_winner_final_spec_json"] = winner_final_spec
        except Exception:
            pass

    allow_stage5_recovery = (
        stages["stage_4_dense_validation"].get("status") == "completed"
        and not (
            _stage2_requires_post_dense_dual_source(normalized)
            and stages["stage_1b_dense_reference"].get("status") != "completed"
        )
    )
    if allow_stage5_recovery:
        if stage5_artifact.exists():
            stages["stage_5_final_full_run"]["status"] = "completed"
            stages["stage_5_final_full_run"]["artifact_json"] = str(stage5_artifact)
            try:
                stage5_payload = _load_json(stage5_artifact)
                normalized["artifacts"]["final_full_run_dir"] = (
                    ((stage5_payload.get("result") or {}).get("run_reference") or {}).get("run_dir")
                )
            except Exception:
                pass
        else:
            recovered_final_run_dir = _find_completed_final_run_dir_for_spec(
                final_spec_path=(
                    normalized.get("artifacts", {}).get("dense_validation_winner_final_spec_json")
                    or normalized.get("artifacts", {}).get("final_stage_conditioned_reward_spec_json")
                ),
                env_key=str((normalized.get("config_summary") or {}).get("env_key") or ""),
                train_config=str((normalized.get("config_summary") or {}).get("train_config") or ""),
                workflow_id_prefixes=[
                    str((normalized.get("stage_ids") or {}).get("stage_5_final_full_run") or ""),
                    str(normalized.get("workflow_id") or ""),
                ],
            )
            if recovered_final_run_dir is not None:
                stage5_payload = _build_recovered_stage5_payload(final_run_dir=recovered_final_run_dir)
                _save_json(stage5_artifact, stage5_payload)
                stages["stage_5_final_full_run"]["status"] = "completed"
                stages["stage_5_final_full_run"]["artifact_json"] = str(stage5_artifact)
                normalized["artifacts"]["final_full_run_dir"] = str(recovered_final_run_dir)

    failed_stages = [name for name, payload in stages.items() if payload.get("status") == "failed"]
    for payload in stages.values():
        if payload.get("status") != "failed":
            payload["error"] = None
    if stages["stage_5_final_full_run"]["status"] == "completed":
        normalized["status"] = "completed"
        normalized["current_stage"] = None
    elif failed_stages:
        normalized["status"] = "failed"
        normalized["current_stage"] = failed_stages[-1]
    else:
        active_stage = None
        for stage_name in (
            "stage_1_sparse_baseline",
            "stage_1b_dense_reference",
            "stage_2_selection",
            "stage_3_branching",
            "stage_4_final_spec",
            "stage_4_dense_validation",
            "stage_5_final_full_run",
        ):
            if stages[stage_name].get("status") != "completed":
                active_stage = stage_name
                break
        normalized["current_stage"] = active_stage
        normalized["status"] = "running" if active_stage is not None else "completed"
    return normalized


def _mark_stage_running(
    manifest: Dict[str, Any],
    stage_name: str,
    *,
    workflow_dir: Path,
) -> None:
    stage = manifest["stages"][stage_name]
    was_running = stage.get("status") == "running"
    stage["status"] = "running"
    stage["started_at"] = stage["started_at"] or _utc_now_iso()
    stage["error"] = None
    manifest["status"] = "running"
    manifest["current_stage"] = stage_name
    if not was_running:
        _append_timeline(workflow_dir, "stage_started", {"stage_name": stage_name})


def _mark_stage_completed(
    manifest: Dict[str, Any],
    stage_name: str,
    *,
    artifact_json: Path | None,
    workflow_dir: Path,
) -> None:
    stage = manifest["stages"][stage_name]
    stage["status"] = "completed"
    stage["completed_at"] = _utc_now_iso()
    stage["artifact_json"] = str(artifact_json) if artifact_json is not None else stage.get("artifact_json")
    stage["error"] = None
    _append_timeline(
        workflow_dir,
        "stage_completed",
        {"stage_name": stage_name, "artifact_json": stage["artifact_json"]},
    )


def _mark_stage_failed(
    manifest: Dict[str, Any],
    stage_name: str,
    *,
    error: str,
    workflow_dir: Path,
) -> None:
    stage = manifest["stages"][stage_name]
    stage["status"] = "failed"
    stage["completed_at"] = _utc_now_iso()
    stage["error"] = error
    manifest["status"] = "failed"
    manifest["current_stage"] = stage_name
    _append_timeline(workflow_dir, "stage_failed", {"stage_name": stage_name, "error": error})


def _build_stage1_args(args: argparse.Namespace, manifest: Dict[str, Any]) -> argparse.Namespace:
    baseline_id = manifest["stage_ids"]["stage_1_sparse_baseline"]
    return argparse.Namespace(
        baseline_id=baseline_id,
        reward_paradigm=args.reward_paradigm,
        max_rounds=args.max_rounds,
        python_executable=args.python_executable,
        env_key=args.env_key,
        train_config=args.train_config,
        use_cuda=bool(args.stage1_use_cuda),
        seed=args.seed,
        time_limit=args.time_limit,
        t_max=int(getattr(args, "stage1_sparse_t_max", args.t_max)),
        test_interval=args.test_interval,
        runner_log_interval=args.runner_log_interval,
        learner_log_interval=args.learner_log_interval,
        save_model_interval=args.save_model_interval,
        local_results_path="results",
        checkpoint_path="",
        load_step=0,
        enable_checkpointing=bool(
            ((getattr(args, "checkpointing_strategy", None) or {}).get("stage1_enable_checkpointing", False))
        ),
        label_suffix="_sparse_baseline",
        execute=True,
        output_path=None,
        format="json",
        experiment_family="stage1_sparse_baseline",
        budget_profile=getattr(args, "budget_profile", "formal"),
        experiment_tag=args.workflow_id,
        checkpoint_optional_metric=args.checkpoint_optional_metric,
    )


def _policy_guidance_setting(args: argparse.Namespace, key: str, default: Any) -> Any:
    direct_value = getattr(args, key, None)
    if direct_value is not None:
        return direct_value
    phase1_method = getattr(args, "phase1_method", None) or {}
    adaptive_replacement = phase1_method.get("adaptive_replacement") or {}
    if key in adaptive_replacement:
        return adaptive_replacement.get(key)
    if key in phase1_method:
        return phase1_method.get(key)
    return default


def _update_policy_guidance_stage1_summary_config(
    args: argparse.Namespace,
    *,
    summary_path: str,
) -> None:
    setattr(args, "policy_guidance_stage1_behavior_summary_path", str(summary_path))
    setattr(args, "policy_guidance_stage1_behavior_summary_reuse_allowed", False)
    setattr(args, "allow_reuse_stage1_behavior_summary", False)
    setattr(args, "policy_guidance_stage1_behavior_summary_must_be_fresh", True)
    phase1_method = deepcopy(getattr(args, "phase1_method", None) or {})
    phase1_method["policy_guidance_stage1_behavior_summary_path"] = str(summary_path)
    phase1_method["policy_guidance_stage1_behavior_summary_reuse_allowed"] = False
    phase1_method["allow_reuse_stage1_behavior_summary"] = False
    phase1_method["policy_guidance_stage1_behavior_summary_must_be_fresh"] = True
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["policy_guidance_stage1_behavior_summary_path"] = str(summary_path)
    adaptive_replacement["policy_guidance_stage1_behavior_summary_reuse_allowed"] = False
    adaptive_replacement["allow_reuse_stage1_behavior_summary"] = False
    adaptive_replacement["policy_guidance_stage1_behavior_summary_must_be_fresh"] = True
    phase1_method["adaptive_replacement"] = adaptive_replacement
    setattr(args, "phase1_method", phase1_method)


def _ensure_workflow_local_stage1_behavior_summary(
    *,
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    workflow_dir: Path,
) -> Optional[Dict[str, Any]]:
    if not bool(_policy_guidance_setting(args, "enable_policy_guidance", False)):
        return None
    if not bool(
        _policy_guidance_setting(
            args,
            "policy_guidance_stage1_behavior_summary_must_be_fresh",
            True,
        )
    ):
        return None

    workflow_summary_path = workflow_dir / "stage1_sparse_policy_behavior_summary.json"
    rewrite_existing = False
    if workflow_summary_path.exists():
        try:
            existing = _load_json(workflow_summary_path)
            existing_workflow_id = str(
                existing.get("workflow_id")
                or (existing.get("metadata") or {}).get("workflow_id")
                or ""
            )
            if (not existing_workflow_id) or existing_workflow_id != str(args.workflow_id):
                rewrite_existing = True
        except Exception:
            rewrite_existing = True

    generated = generate_stage1_behavior_summary_artifact(
        workflow_dir=workflow_dir,
        workflow_id=str(args.workflow_id),
        env_key=str(args.env_key),
        algorithm=str(args.train_config),
        seed=int(args.seed),
        milestone_steps=list(
            _policy_guidance_setting(args, "policy_guidance_stage1_milestones", [])
            or []
        ),
        eval_episodes_per_milestone=int(
            _policy_guidance_setting(
                args,
                "policy_guidance_eval_episodes_stage1_per_milestone",
                2,
            )
        ),
        source_run_ref={
            "workflow_id": manifest["stage_ids"]["stage_1_sparse_baseline"],
            "run_dir": str((manifest.get("artifacts") or {}).get("baseline_run_dir") or ""),
        },
        force_rewrite=bool(rewrite_existing),
    )
    manifest["artifacts"]["stage1_behavior_summary_path"] = generated["summary_path"]
    _update_policy_guidance_stage1_summary_config(
        args,
        summary_path=str(generated["summary_path"]),
    )
    return generated


def _build_stage234_args(args: argparse.Namespace, manifest: Dict[str, Any]) -> argparse.Namespace:
    baseline_run_dir = (
        manifest["artifacts"].get("dense_reference_run_dir")
        or manifest["artifacts"]["baseline_run_dir"]
    )
    return argparse.Namespace(
        workflow_id=manifest["stage_ids"]["stage_2_3_4_stage_conditioned"],
        stage_selection_result_json=manifest["artifacts"].get("stage_selection_result_json"),
        baseline_run_dir=baseline_run_dir,
        baseline_result_json=None,
        branch_source_run_dir=manifest["artifacts"].get("dense_reference_run_dir"),
        branch_source_result_json=None,
        python_executable=args.python_executable,
        results_root=str(ROOT / "results" / "stage_conditioned_workflows"),
        reward_paradigm=args.reward_paradigm,
        max_rounds=args.max_rounds,
        min_count=args.min_count,
        max_count=args.max_count,
        min_step_gap=args.min_step_gap,
        require_full_baseline=True,
        expected_full_t_max=args.t_max,
        max_points_per_metric=args.max_points_per_metric,
        field_round=args.field_round,
        branch_budget_steps=args.branch_budget_steps,
        max_parallel_candidates=args.max_parallel_candidates,
        branch_use_cuda=args.stage3_use_cuda,
        native_original_pbrs_branching=args.native_original_pbrs_branching,
        eval_use_pbrs=False,
        preferred_metric=args.preferred_metric,
        checkpoint_optional_metric=args.checkpoint_optional_metric,
        execute=True,
        reuse_completed_candidates=args.reuse_completed_candidates,
        use_real_llm=args.use_real_llm,
        use_real_llm_for_field_analysis=args.use_real_llm_for_field_analysis,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff=args.llm_retry_backoff,
        output_path=None,
        format="json",
    )


def _build_stage1b_dense_reference_args(
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    dense_reference_selection: Dict[str, Any],
) -> argparse.Namespace:
    return argparse.Namespace(
        run_id=manifest["stage_ids"]["stage_1b_dense_reference"],
        reward_paradigm=args.reward_paradigm,
        native_original_pbrs=True,
        native_stage_conditioned_pbrs=False,
        python_executable=args.python_executable,
        env_key=args.env_key,
        train_config=args.train_config,
        use_cuda=bool(args.stage1_use_cuda),
        seed=args.seed,
        time_limit=args.time_limit,
        t_max=args.t_max,
        test_interval=args.test_interval,
        runner_log_interval=args.runner_log_interval,
        learner_log_interval=args.learner_log_interval,
        save_model_interval=args.save_model_interval,
        local_results_path="results",
        checkpoint_path="",
        load_step=0,
        enable_checkpointing=False,
        reward_spec_path=None,
        output_path=None,
        pbrs_beta=float(dense_reference_selection["selected_beta"]),
        pbrs_wc=float(dense_reference_selection["selected_wc"]),
        pbrs_wp=float(dense_reference_selection["selected_wp"]),
        pbrs_gamma=float(((getattr(args, "phase1_method", None) or {}).get("dense_reference") or {}).get("gamma", 0.99)),
        pbrs_variant=str(((getattr(args, "phase1_method", None) or {}).get("dense_reference") or {}).get("variant", "original")),
        eval_use_pbrs=False,
        pbrs_stage_boundary_1=875923,
        pbrs_stage_boundary_2=1451234,
        pbrs_stage_beta_early=0.3,
        pbrs_stage_beta_mid=1.0,
        pbrs_stage_beta_late=0.5,
        pbrs_stage_wc_early=0.1,
        pbrs_stage_wc_mid=1.0,
        pbrs_stage_wc_late=0.7,
        pbrs_stage_wp_early=0.7,
        pbrs_stage_wp_mid=0.1,
        pbrs_stage_wp_late=0.3,
        execute=True,
        format="json",
        experiment_family="dense_reference_native_original_pbrs",
        budget_profile=getattr(args, "budget_profile", "formal"),
        experiment_tag=args.workflow_id,
    )


def _build_stage5_args(args: argparse.Namespace, manifest: Dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        run_id=manifest["stage_ids"]["stage_5_final_full_run"],
        final_spec_path=(
            manifest["artifacts"].get("dense_validation_winner_final_spec_json")
            or manifest["artifacts"]["final_stage_conditioned_reward_spec_json"]
        ),
        python_executable=args.python_executable,
        env_key=args.env_key,
        train_config=args.train_config,
        seed=args.seed,
        time_limit=args.time_limit,
        t_max=args.t_max,
        test_interval=args.test_interval,
        runner_log_interval=args.runner_log_interval,
        learner_log_interval=args.learner_log_interval,
        reward_paradigm=args.reward_paradigm,
        output_path=None,
        execute=True,
        format="json",
        experiment_family="stage5_final_full_run",
        budget_profile=getattr(args, "budget_profile", "formal"),
        experiment_tag=args.workflow_id,
        use_cuda=bool(args.stage5_use_cuda),
    )


def _metric_summary_value(metrics_summary: Dict[str, Any], metric_name: str, key: str) -> Optional[float]:
    metric_payload = ((metrics_summary.get("metric_summary") or {}).get(metric_name) or {})
    value = metric_payload.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_dense_validation_candidate(result_payload: Dict[str, Any]) -> tuple[float, float]:
    metrics_summary = result_payload.get("metrics_summary") or {}
    best_test_sparse = _metric_summary_value(metrics_summary, "test_sparse_return_mean", "best_value")
    last_test_sparse = _metric_summary_value(metrics_summary, "test_sparse_return_mean", "last_value")
    return (
        best_test_sparse if best_test_sparse is not None else float("-inf"),
        last_test_sparse if last_test_sparse is not None else float("-inf"),
    )


def run_end_to_end_workflow(args: argparse.Namespace) -> Dict[str, Any]:
    args = _apply_workflow_profile(args)
    args = _apply_llm_routing(args)
    if getattr(args, "workflow_profile", None) and is_deprecated_workflow_profile(str(args.workflow_profile)):
        launchable = ", ".join(list_launchable_workflow_profiles())
        raise ValueError(
            f"workflow profile {args.workflow_profile!r} is deprecated for new end-to-end launches because it uses "
            "the legacy stage-conditioned branching path that depends on dense-reference checkpoints. "
            f"Use one of: {launchable}"
        )
    if not _is_adaptive_phase1_method(args):
        launchable = ", ".join(list_launchable_workflow_profiles())
        raise ValueError(
            "new end-to-end launches must use phase1_adaptive_replacement_v1; "
            f"available adaptive profiles: {launchable}"
        )
    manifest, paths, created_new_manifest = _load_or_init_manifest(args)
    manifest = _refresh_manifest_config_summary(manifest=manifest, args=args)
    manifest = _normalize_manifest_from_local_artifacts(manifest=manifest, paths=paths, args=args)
    workflow_dir = paths["workflow_dir"]
    if getattr(args, "stage3_fork_spec_json", None):
        manifest = _apply_stage3_fork_bootstrap_end_to_end(
            args=args,
            manifest=manifest,
            paths=paths,
            workflow_dir=workflow_dir,
        )

    if not args.execute:
        if created_new_manifest:
            manifest["status"] = "planned"
            manifest["current_stage"] = None
            _append_timeline(workflow_dir, "workflow_planned", {"workflow_id": args.workflow_id})
            _save_manifest(paths, manifest)
        return {
            "workflow_id": args.workflow_id,
            "workflow_dir": str(workflow_dir),
            "manifest_json": str(paths["manifest_path"]),
            "planned_only": True,
            "manifest": manifest,
        }

    stage1_artifact = paths["stage1_json"]
    if not _stage1_sparse_stage_satisfied(manifest):
        if stage1_artifact.exists():
            stage1_payload = _load_json(stage1_artifact)
            manifest["artifacts"]["baseline_run_dir"] = (
                ((stage1_payload.get("result") or {}).get("run_reference") or {}).get("run_dir")
            )
            if str(stage1_payload.get("status") or "") == "skipped_reused":
                manifest["artifacts"]["stage1_behavior_summary_path"] = (
                    (stage1_payload.get("reuse_artifacts") or {}).get("stage1_behavior_summary_path")
                )
                manifest["artifacts"]["stage1_sparse_metrics_path"] = (
                    (stage1_payload.get("reuse_artifacts") or {}).get("stage1_sparse_metrics_path")
                )
                manifest["stages"]["stage_1_sparse_baseline"] = {
                    **deepcopy(manifest["stages"]["stage_1_sparse_baseline"]),
                    "status": "skipped_reused",
                    "completed_at": _utc_now_iso(),
                    "artifact_json": str(stage1_artifact),
                    "error": None,
                    "reuse_mode": str(stage1_payload.get("reuse_mode") or ""),
                    "reuse_source_workflow_id": str(
                        stage1_payload.get("source_workflow_id") or ""
                    ),
                    "reuse_source_seed": int(stage1_payload.get("source_seed") or 0),
                    "reuse_artifacts": deepcopy(
                        stage1_payload.get("reuse_artifacts") or {}
                    ),
                    "count_as_evaluation_baseline": bool(
                        stage1_payload.get("count_as_evaluation_baseline", False)
                    ),
                }
                _set_stage1_sparse_reuse_summary(
                    manifest,
                    training_started=False,
                    reuse_enabled=True,
                    reuse_validated=True,
                    count_as_evaluation_baseline=bool(
                        stage1_payload.get("count_as_evaluation_baseline", False)
                    ),
                )
                if bool(_policy_guidance_setting(args, "enable_policy_guidance", False)):
                    summary_path = (
                        (stage1_payload.get("reuse_artifacts") or {}).get(
                            "stage1_behavior_summary_path"
                        )
                        or ""
                    )
                    if summary_path:
                        _configure_stage1_behavior_summary_reuse(
                            args,
                            summary_path=str(summary_path),
                        )
                manifest = _refresh_manifest_config_summary(manifest=manifest, args=args)
            else:
                _mark_stage_completed(
                    manifest,
                    "stage_1_sparse_baseline",
                    artifact_json=stage1_artifact,
                    workflow_dir=workflow_dir,
                )
            _save_manifest(paths, manifest)
        else:
            try:
                reused_stage1 = _resolve_shared_sparse_reuse(
                    args=args,
                    manifest=manifest,
                    paths=paths,
                    workflow_dir=workflow_dir,
                )
            except Exception as exc:
                _record_stage1_reuse_blocker(
                    manifest,
                    workflow_dir=workflow_dir,
                    message=str(exc),
                )
                _mark_stage_failed(
                    manifest,
                    "stage_1_sparse_baseline",
                    error=str(exc),
                    workflow_dir=workflow_dir,
                )
                _save_manifest(paths, manifest)
                raise
            if reused_stage1 is not None:
                manifest = _refresh_manifest_config_summary(manifest=manifest, args=args)
                _save_manifest(paths, manifest)
            else:
                existing_sparse_run_dir = getattr(args, "existing_sparse_run_dir", None)
                if existing_sparse_run_dir:
                    existing_sparse_path = Path(str(existing_sparse_run_dir))
                    if not (existing_sparse_path / "run.json").exists():
                        raise FileNotFoundError(
                            f"existing sparse run dir does not contain run.json: {existing_sparse_path}"
                        )
                    stage1_payload = _build_existing_stage1_payload(
                        baseline_run_dir=existing_sparse_path,
                    )
                    _save_json(stage1_artifact, stage1_payload)
                    manifest["artifacts"]["baseline_run_dir"] = str(existing_sparse_path)
                    _append_timeline(
                        workflow_dir,
                        "stage_recovered",
                        {
                            "stage_name": "stage_1_sparse_baseline",
                            "baseline_run_dir": str(existing_sparse_path),
                            "recovery_mode": "existing_sparse_run_dir",
                        },
                    )
                    _mark_stage_completed(
                        manifest,
                        "stage_1_sparse_baseline",
                        artifact_json=stage1_artifact,
                        workflow_dir=workflow_dir,
                    )
                    _save_manifest(paths, manifest)
                else:
                    recovered_run_dir = _find_completed_stage1_run_dir(
                        stage1_workflow_id=manifest["stage_ids"]["stage_1_sparse_baseline"],
                        env_key=args.env_key,
                        train_config=args.train_config,
                    )
                    if recovered_run_dir is not None:
                        stage1_payload = _build_recovered_stage1_payload(
                            baseline_run_dir=recovered_run_dir,
                            optional_metric_names=args.checkpoint_optional_metric,
                        )
                        _save_json(stage1_artifact, stage1_payload)
                        manifest["artifacts"]["baseline_run_dir"] = str(recovered_run_dir)
                        _append_timeline(
                            workflow_dir,
                            "stage_recovered",
                            {
                                "stage_name": "stage_1_sparse_baseline",
                                "baseline_run_dir": str(recovered_run_dir),
                            },
                        )
                        _mark_stage_completed(
                            manifest,
                            "stage_1_sparse_baseline",
                            artifact_json=stage1_artifact,
                            workflow_dir=workflow_dir,
                        )
                        _save_manifest(paths, manifest)
                    else:
                        try:
                            _set_stage1_sparse_reuse_summary(
                                manifest,
                                training_started=True,
                                reuse_enabled=False,
                                reuse_validated=False,
                                count_as_evaluation_baseline=True,
                            )
                            _mark_stage_running(
                                manifest,
                                "stage_1_sparse_baseline",
                                workflow_dir=workflow_dir,
                            )
                            _save_manifest(paths, manifest)
                            stage1_payload = build_sparse_baseline_plan(
                                _build_stage1_args(args, manifest)
                            )
                            _save_json(stage1_artifact, stage1_payload)
                            manifest["artifacts"]["baseline_run_dir"] = (
                                ((stage1_payload.get("result") or {}).get("run_reference") or {}).get("run_dir")
                            )
                            _mark_stage_completed(
                                manifest,
                                "stage_1_sparse_baseline",
                                artifact_json=stage1_artifact,
                                workflow_dir=workflow_dir,
                            )
                            _save_manifest(paths, manifest)
                        except Exception as exc:
                            _mark_stage_failed(
                                manifest,
                                "stage_1_sparse_baseline",
                                error=str(exc),
                                workflow_dir=workflow_dir,
                            )
                            _save_manifest(paths, manifest)
                            raise

    stage1b_artifact = paths["stage1b_json"]
    if manifest["stages"]["stage_1b_dense_reference"]["status"] != "completed":
        if is_dual_llm_stage1b_enabled(args):
            try:
                _mark_stage_running(manifest, "stage_1b_dense_reference", workflow_dir=workflow_dir)
                _save_manifest(paths, manifest)
                stage1_behavior_summary_generation = _ensure_workflow_local_stage1_behavior_summary(
                    args=args,
                    manifest=manifest,
                    workflow_dir=workflow_dir,
                )
                if stage1_behavior_summary_generation is not None:
                    _save_manifest(paths, manifest)
                sparse_run_summary = build_run_summary_from_run_dir(
                    run_dir=str(manifest["artifacts"]["baseline_run_dir"]),
                    preferred_metrics=args.preferred_metric,
                    max_points_per_metric=args.max_points_per_metric,
                )
                stage1b_payload = run_or_reconcile_dual_llm_stage1b(
                    args=args,
                    manifest=manifest,
                    workflow_dir=workflow_dir,
                    stage1b_artifact_path=stage1b_artifact,
                    sparse_run_summary=sparse_run_summary,
                )
                dense_reference_run_dir = str(stage1b_payload.get("dense_reference_run_dir") or "")
                manifest["artifacts"]["dense_reference_run_dir"] = dense_reference_run_dir or None
                if str(stage1b_payload.get("stage1b_overall_status") or "") == "completed":
                    _mark_stage_completed(
                        manifest,
                        "stage_1b_dense_reference",
                        artifact_json=stage1b_artifact,
                        workflow_dir=workflow_dir,
                    )
                else:
                    _mark_stage_running(manifest, "stage_1b_dense_reference", workflow_dir=workflow_dir)
                _save_manifest(paths, manifest)
            except Exception as exc:
                _mark_stage_failed(manifest, "stage_1b_dense_reference", error=str(exc), workflow_dir=workflow_dir)
                _save_manifest(paths, manifest)
                raise
        else:
            if stage1b_artifact.exists():
                stage1b_payload = _load_json(stage1b_artifact)
                dense_reference_run_dir = (
                    (((stage1b_payload.get("dense_reference_run") or {}).get("result") or {}).get("run_reference") or {}).get("run_dir")
                )
                manifest["artifacts"]["dense_reference_run_dir"] = dense_reference_run_dir
                if _run_dir_completed(dense_reference_run_dir):
                    _mark_stage_completed(manifest, "stage_1b_dense_reference", artifact_json=stage1b_artifact, workflow_dir=workflow_dir)
                    _save_manifest(paths, manifest)
                else:
                    _mark_stage_running(manifest, "stage_1b_dense_reference", workflow_dir=workflow_dir)
                    _save_manifest(paths, manifest)
            else:
                recovered_dense_reference_run_dir = _find_completed_run_dir_for_workflow(
                    workflow_id_prefix=manifest["stage_ids"]["stage_1b_dense_reference"],
                    env_key=args.env_key,
                    train_config=args.train_config,
                )
                latest_dense_reference_run_dir = recovered_dense_reference_run_dir or _find_latest_run_dir_for_workflow(
                    workflow_id_prefix=manifest["stage_ids"]["stage_1b_dense_reference"],
                    env_key=args.env_key,
                    train_config=args.train_config,
                )
                if latest_dense_reference_run_dir is not None:
                    sparse_run_summary = build_run_summary_from_run_dir(
                        run_dir=str(manifest["artifacts"]["baseline_run_dir"]),
                        preferred_metrics=args.preferred_metric,
                        max_points_per_metric=args.max_points_per_metric,
                    )
                    stage1b_payload = _build_recovered_stage1b_payload(
                        dense_reference_run_dir=latest_dense_reference_run_dir,
                        sparse_run_summary=sparse_run_summary,
                    )
                    _save_json(stage1b_artifact, stage1b_payload)
                    manifest["artifacts"]["dense_reference_run_dir"] = str(latest_dense_reference_run_dir)
                    _append_timeline(
                        workflow_dir,
                        "stage_recovered",
                        {
                            "stage_name": "stage_1b_dense_reference",
                            "dense_reference_run_dir": str(latest_dense_reference_run_dir),
                        },
                    )
                    if _run_dir_completed(latest_dense_reference_run_dir):
                        _mark_stage_completed(manifest, "stage_1b_dense_reference", artifact_json=stage1b_artifact, workflow_dir=workflow_dir)
                    else:
                        _mark_stage_running(manifest, "stage_1b_dense_reference", workflow_dir=workflow_dir)
                    _save_manifest(paths, manifest)
                else:
                    try:
                        _mark_stage_running(manifest, "stage_1b_dense_reference", workflow_dir=workflow_dir)
                        _save_manifest(paths, manifest)
                        sparse_run_summary = build_run_summary_from_run_dir(
                            run_dir=str(manifest["artifacts"]["baseline_run_dir"]),
                            preferred_metrics=args.preferred_metric,
                            max_points_per_metric=args.max_points_per_metric,
                        )
                        dense_reference_config = (
                            ((getattr(args, "phase1_method", None) or {}).get("dense_reference") or {})
                        )
                        dense_reference_selection = select_dense_reference_with_llm(
                            workflow_id=manifest["stage_ids"]["stage_1b_dense_reference"],
                            sparse_run_summary=sparse_run_summary,
                            dense_reference_config=dense_reference_config,
                            api_key_env=args.api_key_env,
                            base_url=args.base_url,
                            model=args.model,
                            temperature=args.temperature,
                            llm_timeout=args.llm_timeout,
                            llm_max_retries=args.llm_max_retries,
                            llm_retry_backoff=args.llm_retry_backoff,
                        )
                        dense_reference_run_plan = build_fixed_reward_baseline_plan(
                            _build_stage1b_dense_reference_args(args, manifest, dense_reference_selection)
                        )
                        stage1b_payload = build_dense_reference_artifact(
                            workflow_id=manifest["stage_ids"]["stage_1b_dense_reference"],
                            sparse_run_summary=sparse_run_summary,
                            selection_result=dense_reference_selection,
                            run_plan=dense_reference_run_plan,
                        )
                        _save_json(stage1b_artifact, stage1b_payload)
                        manifest["artifacts"]["dense_reference_run_dir"] = (
                            (((dense_reference_run_plan.get("result") or {}).get("run_reference") or {}).get("run_dir"))
                        )
                        _mark_stage_running(manifest, "stage_1b_dense_reference", workflow_dir=workflow_dir)
                        _save_manifest(paths, manifest)
                    except Exception as exc:
                        _mark_stage_failed(manifest, "stage_1b_dense_reference", error=str(exc), workflow_dir=workflow_dir)
                        _save_manifest(paths, manifest)
                        raise

    stage2_artifact = paths["stage2_json"]
    if manifest["stages"]["stage_2_selection"]["status"] != "completed":
        if _stage2_requires_post_dense_dual_source(manifest) and manifest["stages"]["stage_1b_dense_reference"]["status"] != "completed":
            manifest["stages"]["stage_2_selection"]["status"] = "pending"
            manifest["stages"]["stage_2_selection"]["artifact_json"] = None
            manifest["artifacts"]["stage_selection_result_json"] = None
            _save_manifest(paths, manifest)
        else:
            stage2_payload = None
            stage2_artifact_compatible = False
            if stage2_artifact.exists():
                try:
                    stage2_payload = _load_json(stage2_artifact)
                except Exception:
                    stage2_payload = None
                stage2_artifact_compatible = _stage2_artifact_is_compatible(stage2_payload, manifest=manifest)
                if stage2_artifact_compatible:
                    manifest["artifacts"]["stage_selection_result_json"] = str(stage2_artifact)
                    _mark_stage_completed(manifest, "stage_2_selection", artifact_json=stage2_artifact, workflow_dir=workflow_dir)
                    _save_manifest(paths, manifest)
            if not stage2_artifact.exists() or not stage2_artifact_compatible:
                dense_reference_run_dir = manifest["artifacts"].get("dense_reference_run_dir")
                fixed_schedule_mode = str(
                    ((getattr(args, "phase1_method", None) or {}).get("checkpoint_selection_mode") or "")
                ) == "fixed_intervention_schedule"
                if dense_reference_run_dir and not fixed_schedule_mode and not _run_dir_completed(dense_reference_run_dir):
                    _mark_stage_running(manifest, "stage_1b_dense_reference", workflow_dir=workflow_dir)
                    _save_manifest(paths, manifest)
                else:
                    try:
                        _mark_stage_running(manifest, "stage_2_selection", workflow_dir=workflow_dir)
                        _save_manifest(paths, manifest)
                        sparse_run_path = Path(str(manifest["artifacts"]["baseline_run_dir"]))
                        dense_run_path = Path(str(manifest["artifacts"]["dense_reference_run_dir"]))
                        phase1_method = getattr(args, "phase1_method", None) or {}
                        checkpoint_selection_mode = str(phase1_method.get("checkpoint_selection_mode") or "")
                        if checkpoint_selection_mode == "fixed_intervention_schedule":
                            checkpoint_steps = list(
                                phase1_method.get("fixed_intervention_checkpoints")
                                or getattr(args, "fixed_intervention_checkpoints", None)
                                or [1000000, 1500000]
                            )
                            stage2_payload = build_fixed_intervention_stage_selection_result(
                                workflow_id=manifest["stage_ids"]["stage_2_selection"],
                                checkpoint_steps=[int(step) for step in checkpoint_steps],
                                objective_metric="test_sparse_return_mean",
                            )
                        elif checkpoint_selection_mode == "post_dense_dual_source":
                            stage2_payload = build_dual_source_stage_selection_result(
                                workflow_id=manifest["stage_ids"]["stage_2_selection"],
                                sparse_metrics_json=str(sparse_run_path / "metrics.json"),
                                sparse_run_config_json=str(sparse_run_path / "config.json"),
                                dense_metrics_json=str(dense_run_path / "metrics.json"),
                                dense_run_config_json=str(dense_run_path / "config.json"),
                                dense_run_info_json=str(dense_run_path / "info.json"),
                                reward_paradigm=args.reward_paradigm,
                                max_rounds=args.max_rounds,
                                min_count=args.min_count,
                                max_count=args.max_count,
                                min_step_gap=args.min_step_gap,
                                branch_budget_steps=args.branch_budget_steps,
                                max_points_per_metric=args.max_points_per_metric,
                                preferred_metrics=args.preferred_metric,
                                checkpoint_optional_metrics=args.checkpoint_optional_metric,
                                use_real_llm=args.use_real_llm,
                                api_key_env=args.api_key_env,
                                base_url=args.base_url,
                                model=args.model,
                                temperature=args.temperature,
                                llm_timeout=args.llm_timeout,
                                llm_max_retries=args.llm_max_retries,
                                llm_retry_backoff=args.llm_retry_backoff,
                            )
                        else:
                            stage2_payload = build_stage_selection_result(
                                workflow_id=manifest["stage_ids"]["stage_2_selection"],
                                baseline_metrics_json=str(sparse_run_path / "metrics.json"),
                                baseline_run_config_json=str(sparse_run_path / "config.json"),
                                baseline_run_info_json=str(sparse_run_path / "info.json"),
                                reward_paradigm=args.reward_paradigm,
                                max_rounds=args.max_rounds,
                                min_count=args.min_count,
                                max_count=args.max_count,
                                min_step_gap=args.min_step_gap,
                                max_points_per_metric=args.max_points_per_metric,
                                preferred_metrics=args.preferred_metric,
                                checkpoint_optional_metrics=args.checkpoint_optional_metric,
                                use_real_llm=args.use_real_llm,
                                api_key_env=args.api_key_env,
                                base_url=args.base_url,
                                model=args.model,
                                temperature=args.temperature,
                                llm_timeout=args.llm_timeout,
                                llm_max_retries=args.llm_max_retries,
                                llm_retry_backoff=args.llm_retry_backoff,
                            )
                        _save_json(stage2_artifact, stage2_payload)
                        manifest["artifacts"]["stage_selection_result_json"] = str(stage2_artifact)
                        _mark_stage_completed(manifest, "stage_2_selection", artifact_json=stage2_artifact, workflow_dir=workflow_dir)
                        _save_manifest(paths, manifest)
                    except StageSelectionError as exc:
                        _mark_stage_failed(manifest, "stage_2_selection", error=str(exc), workflow_dir=workflow_dir)
                        _save_manifest(paths, manifest)
                        raise
                    except Exception as exc:
                        _mark_stage_failed(manifest, "stage_2_selection", error=str(exc), workflow_dir=workflow_dir)
                        _save_manifest(paths, manifest)
                        raise

    stage234_artifact = paths["stage234_json"]
    expected_field_rounds = _get_expected_field_rounds(manifest)
    if (
        _is_adaptive_replacement_method(manifest)
        and manifest["stages"]["stage_1b_dense_reference"]["status"] == "completed"
        and manifest["stages"]["stage_2_selection"]["status"] == "completed"
        and manifest["stages"]["stage_5_final_full_run"]["status"] != "completed"
    ):
        try:
            _mark_stage_running(manifest, "stage_3_branching", workflow_dir=workflow_dir)
            _save_manifest(paths, manifest)
            stage2_payload = _load_json(paths["stage2_json"])
            stage1b_payload = _load_json(paths["stage1b_json"])
            dense_reference_selection = deepcopy(stage1b_payload.get("dense_reference_selection") or {})
            adaptive_payload = build_adaptive_checkpoint_replacement_manifest(
                workflow_id=manifest["stage_ids"]["stage_2_3_4_stage_conditioned"],
                stage_selection_result=deepcopy(stage2_payload),
                dense_reference_run_dir=str(manifest["artifacts"]["dense_reference_run_dir"]),
                dense_reference_selection=dense_reference_selection,
                stage1b_source_metadata={
                    "selected_candidate_id": stage1b_payload.get("selected_candidate_id"),
                    "selected_initial_dense_config": deepcopy(stage1b_payload.get("selected_initial_dense_config") or {}),
                    "selected_endpoint_checkpoint_path": stage1b_payload.get("selected_endpoint_checkpoint_path"),
                    "selected_endpoint_checkpoint_step": stage1b_payload.get("selected_endpoint_checkpoint_step"),
                    "dense_reference_source_checkpoint_path": stage1b_payload.get("dense_reference_source_checkpoint_path"),
                    "adaptive_mainline_source_checkpoint_path": stage1b_payload.get("adaptive_mainline_source_checkpoint_path"),
                    "adaptive_mainline_initial_pbrs_config": deepcopy(stage1b_payload.get("adaptive_mainline_initial_pbrs_config") or {}),
                },
                python_executable=args.python_executable,
                stage3_fork_spec_json=getattr(args, "stage3_fork_spec_json", None),
                execute=True,
                branch_budget_steps=int(args.branch_budget_steps),
                max_parallel_candidates=int(args.max_parallel_candidates),
                branch_use_cuda=args.stage3_use_cuda,
                mainline_use_cuda=bool(args.stage5_use_cuda),
                continuation_save_model_interval=int(args.save_model_interval),
                t_max=int(args.t_max),
                adaptive_method={
                    **deepcopy(
                        ((getattr(args, "phase1_method", None) or {}).get(
                            "adaptive_replacement"
                        ) or {})
                    ),
                    "llm_routing": deepcopy(getattr(args, "llm_routing", None) or {}),
                },
                use_real_llm=bool(args.use_real_llm),
                api_key_env=str(args.api_key_env),
                base_url=str(args.base_url),
                model=str(args.model),
                temperature=float(args.temperature),
                llm_timeout=float(args.llm_timeout),
                llm_retry_count=int(((getattr(args, "phase1_method", None) or {}).get("adaptive_replacement") or {}).get("llm_stage3_retry_count", args.llm_max_retries)),
                llm_retry_backoff=float(args.llm_retry_backoff),
            )
            adaptive_manifest_json = str(_adaptive_manifest_path(manifest["stage_ids"]["stage_2_3_4_stage_conditioned"]))
            stage234_payload = {
                "workflow_id": manifest["stage_ids"]["stage_2_3_4_stage_conditioned"],
                "workflow_kind": "adaptive_checkpoint_replacement_wrapper",
                "artifacts": {
                    "workflow_dir": str(Path(adaptive_manifest_json).parent),
                    "stage_selection_result_json": str(paths["stage2_json"]),
                    "branching_manifest_json": adaptive_manifest_json,
                    "final_stage_conditioned_reward_spec_json": ((adaptive_payload.get("artifacts") or {}).get("adaptive_final_spec_json")),
                    "adaptive_checkpoint_replacement_manifest_json": adaptive_manifest_json,
                    "final_mainline_run_json": ((adaptive_payload.get("artifacts") or {}).get("final_mainline_run_json")),
                },
                "adaptive_checkpoint_replacement_manifest": adaptive_payload,
            }
            _save_json(stage234_artifact, stage234_payload)
            manifest["artifacts"]["stage_conditioned_branching_manifest_json"] = adaptive_manifest_json
            manifest["artifacts"]["final_stage_conditioned_reward_spec_json"] = (
                (stage234_payload.get("artifacts") or {}).get("final_stage_conditioned_reward_spec_json")
            )
            adaptive_completion = _adaptive_completion_status(adaptive_payload)
            adaptive_progress = _adaptive_progress_status(adaptive_payload)
            if adaptive_completion["stage_3_branching"]:
                _mark_stage_completed(manifest, "stage_3_branching", artifact_json=Path(adaptive_manifest_json), workflow_dir=workflow_dir)
            else:
                manifest.setdefault("stage3_progress", {})
                manifest["stage3_progress"] = adaptive_progress
            if adaptive_completion["stage_4_final_spec"]:
                _mark_stage_completed(
                    manifest,
                    "stage_4_final_spec",
                    artifact_json=Path(str((adaptive_payload.get("artifacts") or {}).get("adaptive_final_spec_json"))),
                    workflow_dir=workflow_dir,
                )
            if adaptive_completion["stage_3_branching"]:
                _mark_stage_completed(manifest, "stage_4_dense_validation", artifact_json=stage234_artifact, workflow_dir=workflow_dir)
            else:
                _mark_stage_running(manifest, "stage_4_dense_validation", workflow_dir=workflow_dir)
            final_mainline_run_json = (adaptive_payload.get("artifacts") or {}).get("final_mainline_run_json")
            manifest["artifacts"]["final_full_run_dir"] = None
            if final_mainline_run_json and Path(str(final_mainline_run_json)).exists():
                final_mainline_payload = _load_json(Path(str(final_mainline_run_json)))
                manifest["artifacts"]["final_full_run_dir"] = (
                    (((final_mainline_payload.get("result") or {}).get("run_reference") or {}).get("run_dir"))
                )
            if adaptive_completion["stage_5_final_full_run"]:
                _mark_stage_completed(
                    manifest,
                    "stage_5_final_full_run",
                    artifact_json=Path(str(final_mainline_run_json)) if final_mainline_run_json else None,
                    workflow_dir=workflow_dir,
                )
            elif final_mainline_run_json:
                _mark_stage_running(manifest, "stage_5_final_full_run", workflow_dir=workflow_dir)
            _save_manifest(paths, manifest)
        except Exception as exc:
            _mark_stage_failed(manifest, "stage_3_branching", error=str(exc), workflow_dir=workflow_dir)
            _save_manifest(paths, manifest)
            raise

    if (
        not _is_adaptive_replacement_method(manifest)
        and manifest["stages"]["stage_1b_dense_reference"]["status"] == "completed"
        and manifest["stages"]["stage_2_selection"]["status"] == "completed"
        and any(
        manifest["stages"][stage_name]["status"] != "completed"
        for stage_name in ("stage_3_branching", "stage_4_final_spec")
        )
    ):
        stage234_max_attempts = max(1, int(args.stage234_max_attempts or 1))
        last_stage234_error: Optional[str] = None
        completion_status = {
            "stage_3_branching": False,
            "stage_4_final_spec": False,
        }
        for attempt_index in range(1, stage234_max_attempts + 1):
            stage234_payload = None
            if stage234_artifact.exists():
                stage234_payload = _load_json(stage234_artifact)
            best_recovered_stage234_payload = _recover_best_stage234_payload(
                branching_workflow_id=manifest["stage_ids"]["stage_2_3_4_stage_conditioned"],
                results_root=_build_stage234_args(args, manifest).results_root,
                expected_field_rounds=expected_field_rounds,
            )
            if _stage234_payload_rank(
                best_recovered_stage234_payload,
                expected_field_rounds=expected_field_rounds,
            ) > _stage234_payload_rank(
                stage234_payload,
                expected_field_rounds=expected_field_rounds,
            ):
                stage234_payload = best_recovered_stage234_payload
                if stage234_payload is not None:
                    _save_json(stage234_artifact, stage234_payload)

            if stage234_payload is not None:
                branching_manifest_json = ((stage234_payload.get("artifacts") or {}).get("branching_manifest_json"))
                final_spec_json = ((stage234_payload.get("artifacts") or {}).get("final_stage_conditioned_reward_spec_json"))
                manifest["artifacts"]["stage_conditioned_branching_manifest_json"] = branching_manifest_json
                manifest["artifacts"]["final_stage_conditioned_reward_spec_json"] = final_spec_json
                completion_status = _stage234_completion_status(
                    stage234_payload=stage234_payload,
                    expected_field_rounds=expected_field_rounds,
                )
                if completion_status["stage_3_branching"]:
                    _mark_stage_completed(
                        manifest,
                        "stage_3_branching",
                        artifact_json=Path(branching_manifest_json) if branching_manifest_json else stage234_artifact,
                        workflow_dir=workflow_dir,
                    )
                if completion_status["stage_4_final_spec"] and final_spec_json:
                    _mark_stage_completed(
                        manifest,
                        "stage_4_final_spec",
                        artifact_json=Path(final_spec_json),
                        workflow_dir=workflow_dir,
                    )
                _save_manifest(paths, manifest)

            if all(completion_status.values()):
                break

            try:
                for stage_name in ("stage_3_branching", "stage_4_final_spec"):
                    if not completion_status[stage_name]:
                        _mark_stage_running(manifest, stage_name, workflow_dir=workflow_dir)
                _append_timeline(
                    workflow_dir,
                    "stage234_resume_attempt",
                    {
                        "attempt": attempt_index,
                        "max_attempts": stage234_max_attempts,
                        "workflow_id": manifest["stage_ids"]["stage_2_3_4_stage_conditioned"],
                    },
                )
                _save_manifest(paths, manifest)
                stage234_payload = run_stage_conditioned_branching(_build_stage234_args(args, manifest))
                _save_json(stage234_artifact, stage234_payload)
            except Exception as exc:
                last_stage234_error = str(exc)
                if attempt_index >= stage234_max_attempts:
                    break
                time.sleep(1.0)
                continue

            branching_manifest_json = ((stage234_payload.get("artifacts") or {}).get("branching_manifest_json"))
            final_spec_json = ((stage234_payload.get("artifacts") or {}).get("final_stage_conditioned_reward_spec_json"))
            manifest["artifacts"]["stage_conditioned_branching_manifest_json"] = branching_manifest_json
            manifest["artifacts"]["final_stage_conditioned_reward_spec_json"] = final_spec_json
            completion_status = _stage234_completion_status(
                stage234_payload=stage234_payload,
                expected_field_rounds=expected_field_rounds,
            )
            if completion_status["stage_3_branching"]:
                _mark_stage_completed(
                    manifest,
                    "stage_3_branching",
                    artifact_json=Path(branching_manifest_json) if branching_manifest_json else stage234_artifact,
                    workflow_dir=workflow_dir,
                )
            if completion_status["stage_4_final_spec"] and final_spec_json:
                _mark_stage_completed(
                    manifest,
                    "stage_4_final_spec",
                    artifact_json=Path(final_spec_json),
                    workflow_dir=workflow_dir,
                )
            _save_manifest(paths, manifest)
            if all(completion_status.values()):
                break

        if not all(completion_status.values()):
            error = last_stage234_error or "stage234 did not reach a completed state"
            for stage_name in ("stage_3_branching", "stage_4_final_spec"):
                if not completion_status[stage_name]:
                    _mark_stage_failed(manifest, stage_name, error=str(error), workflow_dir=workflow_dir)
            _save_manifest(paths, manifest)
            raise RuntimeError(f"Stage 3/4-final-spec did not complete after {stage234_max_attempts} attempt(s): {error}")

    stage4_dense_artifact = paths["stage4_dense_json"]
    if (
        not _is_adaptive_replacement_method(manifest)
        and manifest["stages"]["stage_4_final_spec"]["status"] == "completed"
        and manifest["stages"]["stage_4_dense_validation"]["status"] != "completed"
    ):
        if stage4_dense_artifact.exists():
            stage4_dense_payload = _load_json(stage4_dense_artifact)
            winner_final_spec_path = ((stage4_dense_payload.get("winner") or {}).get("final_spec_path"))
            if winner_final_spec_path:
                manifest["artifacts"]["dense_validation_json"] = str(stage4_dense_artifact)
                manifest["artifacts"]["dense_validation_winner_final_spec_json"] = winner_final_spec_path
                _mark_stage_completed(manifest, "stage_4_dense_validation", artifact_json=stage4_dense_artifact, workflow_dir=workflow_dir)
                _save_manifest(paths, manifest)
        else:
            try:
                _mark_stage_running(manifest, "stage_4_dense_validation", workflow_dir=workflow_dir)
                _save_manifest(paths, manifest)
                phase1_method = getattr(args, "phase1_method", None) or {}
                dense_validation_config = (phase1_method.get("dense_validation") or {})
                final_spec_path = manifest["artifacts"]["final_stage_conditioned_reward_spec_json"]
                branching_manifest_path = manifest["artifacts"]["stage_conditioned_branching_manifest_json"]
                final_spec_payload = _load_json(Path(final_spec_path))
                branching_manifest_payload = _load_json(Path(branching_manifest_path))
                dense_validation_candidates = build_dense_validation_candidates(
                    final_spec=final_spec_payload,
                    field_round_summaries=list(branching_manifest_payload.get("field_round_summaries") or []),
                    top_k_schedules=int(dense_validation_config.get("top_k_schedules") or 3),
                )
                materialized_specs = materialize_dense_validation_final_specs(
                    base_final_spec=final_spec_payload,
                    dense_validation_candidates=dense_validation_candidates,
                )
                dense_validation_dir = workflow_dir / "stage_4_dense_validation_candidates"
                dense_validation_dir.mkdir(parents=True, exist_ok=True)
                candidate_runs = []
                for candidate_payload in materialized_specs:
                    candidate_name = str(candidate_payload["candidate_name"])
                    candidate_spec_path = dense_validation_dir / f"{candidate_name}.json"
                    _save_json(candidate_spec_path, candidate_payload["final_spec"])
                    candidate_run_id = f"{manifest['stage_ids']['stage_4_dense_validation']}_{candidate_name}"
                    recovered_candidate_dir = _find_completed_final_run_dir_for_spec(
                        final_spec_path=str(candidate_spec_path),
                        env_key=args.env_key,
                        train_config=args.train_config,
                        workflow_id_prefixes=[candidate_run_id],
                    )
                    if recovered_candidate_dir is not None:
                        run_plan = _build_recovered_stage5_payload(final_run_dir=recovered_candidate_dir)
                    else:
                        candidate_args = argparse.Namespace(
                            run_id=candidate_run_id,
                            final_spec_path=str(candidate_spec_path),
                            python_executable=args.python_executable,
                            env_key=args.env_key,
                            train_config=args.train_config,
                            seed=args.seed,
                            time_limit=args.time_limit,
                            t_max=args.t_max,
                            test_interval=args.test_interval,
                            runner_log_interval=args.runner_log_interval,
                            learner_log_interval=args.learner_log_interval,
                            reward_paradigm=args.reward_paradigm,
                            output_path=None,
                            execute=True,
                            format="json",
                            experiment_family="dense_validation_stage_conditioned_full_run",
                            budget_profile=getattr(args, "budget_profile", "formal"),
                            experiment_tag=args.workflow_id,
                            use_cuda=bool(args.stage5_use_cuda),
                        )
                        run_plan = build_final_stage_conditioned_full_run_plan(candidate_args)
                    candidate_runs.append(
                        {
                            "candidate_name": candidate_name,
                            "candidate_rank": candidate_payload["candidate_rank"],
                            "final_spec_path": str(candidate_spec_path),
                            "run_plan": run_plan,
                            "score": _score_dense_validation_candidate(run_plan.get("result") or {}),
                        }
                    )
                winner = max(candidate_runs, key=lambda item: item["score"]) if candidate_runs else None
                stage4_dense_payload = {
                    "workflow_id": manifest["stage_ids"]["stage_4_dense_validation"],
                    "source_final_spec_path": final_spec_path,
                    "source_branching_manifest_path": branching_manifest_path,
                    "dense_validation_candidates": dense_validation_candidates,
                    "materialized_specs": [
                        {
                            "candidate_name": item["candidate_name"],
                            "candidate_rank": item["candidate_rank"],
                            "final_spec_path": item["final_spec_path"],
                            "score": list(item["score"]),
                            "run_dir": (((item.get("run_plan") or {}).get("result") or {}).get("run_reference") or {}).get("run_dir"),
                        }
                        for item in candidate_runs
                    ],
                    "winner": None
                    if winner is None
                    else {
                        "candidate_name": winner["candidate_name"],
                        "candidate_rank": winner["candidate_rank"],
                        "final_spec_path": winner["final_spec_path"],
                        "score": list(winner["score"]),
                        "run_dir": (((winner.get("run_plan") or {}).get("result") or {}).get("run_reference") or {}).get("run_dir"),
                    },
                }
                _save_json(stage4_dense_artifact, stage4_dense_payload)
                manifest["artifacts"]["dense_validation_json"] = str(stage4_dense_artifact)
                manifest["artifacts"]["dense_validation_winner_final_spec_json"] = (
                    (stage4_dense_payload.get("winner") or {}).get("final_spec_path")
                )
                _mark_stage_completed(manifest, "stage_4_dense_validation", artifact_json=stage4_dense_artifact, workflow_dir=workflow_dir)
                _save_manifest(paths, manifest)
            except Exception as exc:
                _mark_stage_failed(manifest, "stage_4_dense_validation", error=str(exc), workflow_dir=workflow_dir)
                _save_manifest(paths, manifest)
                raise

    stage5_artifact = paths["stage5_json"]
    if (
        not _is_adaptive_replacement_method(manifest)
        and manifest["stages"]["stage_4_dense_validation"]["status"] == "completed"
        and manifest["stages"]["stage_5_final_full_run"]["status"] != "completed"
    ):
        if stage5_artifact.exists():
            stage5_payload = _load_json(stage5_artifact)
            manifest["artifacts"]["final_full_run_dir"] = (
                ((stage5_payload.get("result") or {}).get("run_reference") or {}).get("run_dir")
            )
            _mark_stage_completed(manifest, "stage_5_final_full_run", artifact_json=stage5_artifact, workflow_dir=workflow_dir)
            _save_manifest(paths, manifest)
        else:
            recovered_run_dir = _find_completed_run_dir_for_workflow(
                workflow_id_prefix=manifest["stage_ids"]["stage_5_final_full_run"],
                env_key=args.env_key,
                train_config=args.train_config,
            )
            if recovered_run_dir is None:
                recovered_run_dir = _find_completed_final_run_dir_for_spec(
                    final_spec_path=manifest["artifacts"]["final_stage_conditioned_reward_spec_json"],
                    env_key=args.env_key,
                    train_config=args.train_config,
                    workflow_id_prefixes=[
                        manifest["stage_ids"]["stage_5_final_full_run"],
                        manifest["workflow_id"],
                    ],
                )
            if recovered_run_dir is not None:
                stage5_payload = _build_recovered_stage5_payload(final_run_dir=recovered_run_dir)
                _save_json(stage5_artifact, stage5_payload)
                manifest["artifacts"]["final_full_run_dir"] = str(recovered_run_dir)
                _append_timeline(
                    workflow_dir,
                    "stage_recovered",
                    {
                        "stage_name": "stage_5_final_full_run",
                        "final_full_run_dir": str(recovered_run_dir),
                    },
                )
                _mark_stage_completed(
                    manifest,
                    "stage_5_final_full_run",
                    artifact_json=stage5_artifact,
                    workflow_dir=workflow_dir,
                )
                _save_manifest(paths, manifest)
            else:
                try:
                    _mark_stage_running(manifest, "stage_5_final_full_run", workflow_dir=workflow_dir)
                    _save_manifest(paths, manifest)
                    stage5_payload = build_final_stage_conditioned_full_run_plan(_build_stage5_args(args, manifest))
                    _save_json(stage5_artifact, stage5_payload)
                    manifest["artifacts"]["final_full_run_dir"] = (
                        ((stage5_payload.get("result") or {}).get("run_reference") or {}).get("run_dir")
                    )
                    _mark_stage_completed(manifest, "stage_5_final_full_run", artifact_json=stage5_artifact, workflow_dir=workflow_dir)
                    _save_manifest(paths, manifest)
                except Exception as exc:
                    _mark_stage_failed(manifest, "stage_5_final_full_run", error=str(exc), workflow_dir=workflow_dir)
                    _save_manifest(paths, manifest)
                    raise

    manifest = _normalize_manifest_from_local_artifacts(manifest=manifest, paths=paths, args=args)
    if manifest.get("status") == "completed":
        _append_timeline(workflow_dir, "workflow_completed", {"workflow_id": args.workflow_id})
    _save_manifest(paths, manifest)

    payload = {
        "workflow_id": args.workflow_id,
        "workflow_dir": str(workflow_dir),
        "manifest_json": str(paths["manifest_path"]),
        "stage_1_sparse_baseline_json": str(stage1_artifact),
        "stage_2_3_4_stage_conditioned_json": str(stage234_artifact),
        "stage_5_final_full_run_json": str(stage5_artifact),
        "manifest": manifest,
    }
    return payload


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _print_text(payload: Dict[str, Any]) -> None:
    manifest = payload["manifest"]
    print(
        f"{payload['workflow_id']}: status={_fmt(manifest.get('status'))} "
        f"env={_fmt((manifest.get('config_summary') or {}).get('env_key'))} "
        f"algorithm={_fmt((manifest.get('config_summary') or {}).get('train_config'))}"
    )
    for stage_name, stage_payload in (manifest.get("stages") or {}).items():
        print(
            "  "
            f"{stage_name}: status={_fmt(stage_payload.get('status'))} "
            f"artifact={_fmt(stage_payload.get('artifact_json'))}"
        )
    print(
        "  "
        f"baseline_run_dir={_fmt((manifest.get('artifacts') or {}).get('baseline_run_dir'))} "
        f"final_spec={_fmt((manifest.get('artifacts') or {}).get('final_stage_conditioned_reward_spec_json'))} "
        f"final_full_run_dir={_fmt((manifest.get('artifacts') or {}).get('final_full_run_dir'))}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload = run_end_to_end_workflow(args)
    supervise = (bool(args.execute) and not bool(args.single_tick)) or (args.loop_until_complete and args.execute)
    consecutive_errors = 0
    if supervise:
        while True:
            manifest = payload.get("manifest") or {}
            status = str(manifest.get("status") or "")
            if status in {"completed", "failed"}:
                break
            time.sleep(max(1.0, float(args.poll_interval_seconds)))
            try:
                payload = run_end_to_end_workflow(args)
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                manifest_path = Path(args.results_root) / args.workflow_id / "workflow_manifest.json"
                if manifest_path.exists():
                    manifest = _load_json(manifest_path)
                    payload = {"workflow_id": args.workflow_id, "manifest": manifest}
                    status = str(manifest.get("status") or "")
                    if status in {"completed", "failed"}:
                        break
                if consecutive_errors >= max(1, int(args.max_supervisor_errors)):
                    raise
    if args.output_path:
        _save_json(Path(args.output_path), payload)
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    _print_text(payload)


if __name__ == "__main__":
    main()
