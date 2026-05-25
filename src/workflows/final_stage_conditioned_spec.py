from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from rewarding.spec_schema import get_default_reward_spec, validate_reward_spec


def synthesize_final_stage_conditioned_spec(
    stage_conditioned_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    pbrs_version = str(stage_conditioned_manifest.get("pbrs_version") or "")
    stage_pbrs_configs = deepcopy(stage_conditioned_manifest.get("stage_pbrs_configs") or {})
    selected_checkpoints = (
        (stage_conditioned_manifest.get("stage_selection_result") or {}).get("selection") or {}
    ).get("selected_checkpoints") or []
    if pbrs_version in {"lbf_pbrs_v2", "rware_pbrs_v2"} or stage_pbrs_configs:
        normalized_stage_configs: Dict[str, Dict[str, Any]] = {}
        expected_stage_labels = [str(checkpoint.get("stage_label") or "") for checkpoint in selected_checkpoints]
        missing_stage_configs = [
            label for label in expected_stage_labels if label and label not in stage_pbrs_configs
        ]
        if expected_stage_labels and missing_stage_configs:
            raise ValueError(
                "cannot synthesize final stage-conditioned reward spec because "
                "the following v2 stage configs are missing: "
                + ", ".join(missing_stage_configs)
            )
        for stage_label, config in stage_pbrs_configs.items():
            cfg = deepcopy(config or {})
            if pbrs_version == "lbf_pbrs_v2":
                required_keys = ["mode", "beta", "active_terms", "weights"]
                missing_keys = [key for key in required_keys if cfg.get(key) in (None, [], {})]
                if missing_keys:
                    raise ValueError(
                        "cannot synthesize final stage-conditioned reward spec because "
                        f"lbf_pbrs_v2 stage {stage_label!r} is missing keys: {', '.join(missing_keys)}"
                    )
            normalized_stage_configs[str(stage_label)] = cfg
        return {
            "workflow_kind": "stage_conditioned_final_spec",
            "source_manifest_workflow_id": stage_conditioned_manifest.get("workflow_id"),
            "pbrs_version": pbrs_version or None,
            "selected_checkpoints": deepcopy(selected_checkpoints),
            "stage_pbrs_configs": normalized_stage_configs,
            "field_recommendations": {},
            "field_round_summaries": deepcopy(stage_conditioned_manifest.get("field_round_summaries") or []),
            "stage_boundaries": {},
            "stage_reward_specs": {},
            "native_original_pbrs_stage_schedule": None,
        }
    manifest_readiness = stage_conditioned_manifest.get("readiness") or {}
    if not bool(manifest_readiness.get("ready_for_final_synthesis")):
        raise ValueError(
            "cannot synthesize final stage-conditioned reward spec because Stage 3 "
            "field-round validation is not yet marked ready_for_final_synthesis"
        )
    workflow_spec = {"pbrs_tuning": {"base_pbrs": {}}}
    base_pbrs = deepcopy(
        (
            ((stage_conditioned_manifest.get("workflow_spec") or {}).get("pbrs_tuning") or {}).get("base_pbrs")
            or workflow_spec["pbrs_tuning"]["base_pbrs"]
        )
    )
    if not base_pbrs:
        base_pbrs = {"enabled": True, "variant": "original", "beta": 0.3, "gamma": 0.99, "wc": 0.6, "wp": 0.4}

    configured_field_rounds = list(stage_conditioned_manifest.get("field_rounds") or [])
    recommendations_by_field: Dict[str, Dict[str, Any]] = {}
    for round_payload in stage_conditioned_manifest.get("rounds", []):
        if not isinstance(round_payload, dict):
            continue
        field_name = round_payload.get("field_name")
        field_recommendation = round_payload.get("field_recommendation") or {}
        if isinstance(field_name, str):
            recommendations_by_field[field_name] = deepcopy(
                field_recommendation.get("recommended_values_by_stage") or {}
            )

    required_fields = ["beta", "wc"]
    missing_recommendations = []
    for checkpoint in selected_checkpoints:
        stage_label = str(checkpoint.get("stage_label"))
        for field_name in required_fields:
            stage_values = recommendations_by_field.get(field_name) or {}
            if stage_label not in stage_values:
                missing_recommendations.append(f"{field_name}:{stage_label}")
    if missing_recommendations:
        raise ValueError(
            "cannot synthesize final stage-conditioned reward spec because "
            "the following field/stage recommendations are missing: "
            + ", ".join(missing_recommendations)
        )

    stage_reward_specs: Dict[str, Dict[str, Any]] = {}
    for checkpoint in selected_checkpoints:
        stage_label = str(checkpoint.get("stage_label"))
        reward_spec = get_default_reward_spec()
        reward_spec = deepcopy(reward_spec)
        reward_spec.setdefault("pbrs", {})
        reward_spec["pbrs"].update(deepcopy(base_pbrs))
        reward_spec["alpha_policy"] = {"type": "constant", "value": 1.0}
        for field_name, stage_values in recommendations_by_field.items():
            if stage_label in stage_values:
                reward_spec["pbrs"][field_name] = stage_values[stage_label]
        if "wc" in reward_spec["pbrs"]:
            reward_spec["pbrs"]["wp"] = 1.0 - float(reward_spec["pbrs"]["wc"])
        stage_reward_specs[stage_label] = validate_reward_spec(reward_spec)

    stage_boundaries = {}
    for index, checkpoint in enumerate(selected_checkpoints):
        stage_label = str(checkpoint.get("stage_label"))
        current_step = int(checkpoint.get("step"))
        next_step = (
            int(selected_checkpoints[index + 1].get("step"))
            if index + 1 < len(selected_checkpoints)
            else None
        )
        stage_boundaries[stage_label] = {
            "start_step": current_step if index > 0 else 0,
            "end_step": (next_step - 1) if next_step is not None else None,
            "anchor_step": current_step,
        }

    derive_wp_from_wc = ("wp" not in configured_field_rounds) and ("wc" in recommendations_by_field)
    if derive_wp_from_wc:
        recommendations_by_field["wp"] = {
            stage_label: 1.0 - float(value)
            for stage_label, value in (recommendations_by_field.get("wc") or {}).items()
        }

    stage_labels = [str(checkpoint.get("stage_label")) for checkpoint in selected_checkpoints]
    native_stage_schedule = None
    if stage_labels == ["early_exploration", "mid_progress", "late_plateau"]:
        wp_by_stage = {
            "early_exploration": float(recommendations_by_field["wp"]["early_exploration"]),
            "mid_progress": float(recommendations_by_field["wp"]["mid_progress"]),
            "late_plateau": float(recommendations_by_field["wp"]["late_plateau"]),
        }
        native_stage_schedule = {
            "enabled": True,
            "variant": str(base_pbrs.get("variant", "original")),
            "gamma": float(base_pbrs.get("gamma", 0.99)),
            "eval_use_pbrs": False,
            "stage_boundary_1": int(selected_checkpoints[1].get("step")),
            "stage_boundary_2": int(selected_checkpoints[2].get("step")),
            "beta": {
                "early": recommendations_by_field["beta"]["early_exploration"],
                "mid": recommendations_by_field["beta"]["mid_progress"],
                "late": recommendations_by_field["beta"]["late_plateau"],
            },
            "wc": {
                "early": recommendations_by_field["wc"]["early_exploration"],
                "mid": recommendations_by_field["wc"]["mid_progress"],
                "late": recommendations_by_field["wc"]["late_plateau"],
            },
            "wp": {
                "early": wp_by_stage["early_exploration"],
                "mid": wp_by_stage["mid_progress"],
                "late": wp_by_stage["late_plateau"],
            },
        }

    return {
        "workflow_kind": "stage_conditioned_final_spec",
        "source_manifest_workflow_id": stage_conditioned_manifest.get("workflow_id"),
        "selected_checkpoints": deepcopy(selected_checkpoints),
        "field_recommendations": recommendations_by_field,
        "field_round_summaries": deepcopy(stage_conditioned_manifest.get("field_round_summaries") or []),
        "stage_boundaries": stage_boundaries,
        "stage_reward_specs": stage_reward_specs,
        "native_original_pbrs_stage_schedule": native_stage_schedule,
    }
