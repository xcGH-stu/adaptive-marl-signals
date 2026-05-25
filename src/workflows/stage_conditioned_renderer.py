from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from rewarding.spec_renderer import render_reward_code_from_spec


def render_stage_conditioned_reward_bundle(
    final_stage_conditioned_spec: Dict[str, Any],
    output_dir: Path | str,
) -> Dict[str, Any]:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    stage_reward_specs = dict(final_stage_conditioned_spec.get("stage_reward_specs") or {})
    if not stage_reward_specs:
        raise ValueError("final_stage_conditioned_spec.stage_reward_specs is required")

    stage_boundaries = dict(final_stage_conditioned_spec.get("stage_boundaries") or {})
    if not stage_boundaries:
        raise ValueError("final_stage_conditioned_spec.stage_boundaries is required")

    stage_module_paths: Dict[str, str] = {}
    for stage_label, reward_spec in stage_reward_specs.items():
        stage_module_path = output_path / f"reward_function_{stage_label}.py"
        stage_module_path.write_text(
            render_reward_code_from_spec(reward_spec),
            encoding="utf-8",
        )
        stage_module_paths[stage_label] = str(stage_module_path)

    bundle_metadata = {
        "workflow_kind": "stage_conditioned_reward_bundle",
        "selected_checkpoints": final_stage_conditioned_spec.get("selected_checkpoints") or [],
        "stage_boundaries": stage_boundaries,
        "stage_module_paths": stage_module_paths,
    }
    metadata_path = output_path / "stage_conditioned_bundle.json"
    metadata_path.write_text(
        json.dumps(bundle_metadata, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    master_module_path = output_path / "reward_function_stage_conditioned.py"
    master_module_path.write_text(
        _render_master_stage_conditioned_module(
            stage_boundaries=stage_boundaries,
            stage_module_paths=stage_module_paths,
        ),
        encoding="utf-8",
    )

    return {
        "bundle_metadata_path": str(metadata_path),
        "master_reward_module_path": str(master_module_path),
        "stage_module_paths": stage_module_paths,
        "stage_boundaries": stage_boundaries,
    }


def _render_master_stage_conditioned_module(
    *,
    stage_boundaries: Dict[str, Any],
    stage_module_paths: Dict[str, str],
) -> str:
    return f"""\
\"\"\"
Auto-generated stage-conditioned reward module.

This file dispatches reward computation to one of several stage-specific
reward modules based on the current training step (t_env).
\"\"\"

from __future__ import annotations

import importlib.util
from pathlib import Path

STAGE_BOUNDARIES = {stage_boundaries!r}
STAGE_MODULE_PATHS = {stage_module_paths!r}

_STAGE_MODULES = {{}}


def _load_stage_module(stage_label):
    module = _STAGE_MODULES.get(stage_label)
    if module is not None:
        return module
    module_path = Path(STAGE_MODULE_PATHS[stage_label])
    spec = importlib.util.spec_from_file_location(
        f"stage_conditioned_reward_{{stage_label}}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reward module for stage {{stage_label}} from {{module_path}}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _STAGE_MODULES[stage_label] = module
    return module


def _resolve_stage_label(t_env):
    t_env = int(t_env)
    resolved_label = None
    for stage_label, boundary in STAGE_BOUNDARIES.items():
        start_step = int(boundary.get("start_step", 0) or 0)
        end_step = boundary.get("end_step")
        if end_step is None:
            if t_env >= start_step:
                resolved_label = stage_label
        else:
            if start_step <= t_env <= int(end_step):
                resolved_label = stage_label
    if resolved_label is not None:
        return resolved_label
    ordered = sorted(
        STAGE_BOUNDARIES.items(),
        key=lambda item: int((item[1] or {{}}).get("start_step", 0) or 0),
    )
    if not ordered:
        raise RuntimeError("No stage boundaries configured")
    if t_env < int((ordered[0][1] or {{}}).get("start_step", 0) or 0):
        return ordered[0][0]
    return ordered[-1][0]


def compute_dense_reward(context):
    stage_label = _resolve_stage_label(getattr(context, "t_env", 0))
    module = _load_stage_module(stage_label)
    result = module.compute_dense_reward(context)
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
        breakdown = dict(result[1])
        breakdown.setdefault("stage_conditioned", {{}})
        breakdown["stage_conditioned"]["stage_label"] = stage_label
        breakdown["stage_conditioned"]["t_env"] = int(getattr(context, "t_env", 0))
        return result[0], breakdown
    return result
"""
