from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-final-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _constantize_spec(
    final_spec: Dict[str, Any],
    *,
    source_stage_label: str,
) -> Dict[str, Any]:
    stage_reward_specs = dict(final_spec.get("stage_reward_specs") or {})
    if source_stage_label not in stage_reward_specs:
        raise ValueError(f"stage {source_stage_label!r} not found in final spec")

    source_reward_spec = deepcopy(stage_reward_specs[source_stage_label])
    selected_checkpoints = deepcopy(final_spec.get("selected_checkpoints") or [])
    if not selected_checkpoints:
        raise ValueError("final spec does not contain selected_checkpoints")

    source_boundary = (
        dict(final_spec.get("stage_boundaries") or {}).get(source_stage_label) or {}
    )
    anchor_step = int(source_boundary.get("anchor_step", 0) or 0)

    constant_stage_reward_specs = {}
    stage_boundaries = {}
    for index, checkpoint in enumerate(selected_checkpoints):
        stage_label = str(checkpoint.get("stage_label"))
        constant_stage_reward_specs[stage_label] = deepcopy(source_reward_spec)
        stage_boundaries[stage_label] = {
            "start_step": 0 if index == 0 else 999999999,
            "end_step": None if index == 0 else 999999998,
            "anchor_step": anchor_step,
        }

    payload = deepcopy(final_spec)
    payload["workflow_kind"] = "constant_stage_conditioned_final_spec"
    payload["constant_source_stage_label"] = source_stage_label
    payload["stage_reward_specs"] = constant_stage_reward_specs
    payload["stage_boundaries"] = stage_boundaries
    return payload


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input_final_spec)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_spec = json.loads(input_path.read_text(encoding="utf-8"))
    for source_stage_label in ["early_exploration", "mid_progress", "late_plateau"]:
        payload = _constantize_spec(final_spec, source_stage_label=source_stage_label)
        output_path = output_dir / f"constant_{source_stage_label}_full_spec.json"
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        print(output_path)


if __name__ == "__main__":
    main()
