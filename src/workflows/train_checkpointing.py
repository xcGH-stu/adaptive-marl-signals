from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from experiments.qmix_hparam_presets import resolve_qmix_hparam_preset


_DIRECT_TRAIN_CONFIG_KEYS = {
    "config",
    "name",
    "env_config",
    "env",
    "env_args",
    "overrides",
    "label",
    "use_dense_reward",
    "reward_module_path",
    "alpha_policy_config",
    "policy_guidance_config",
    "workflow_id",
    "workflow_round",
    "reward_paradigm",
    "active_pbrs_field",
    "active_pbrs_field_source",
    "candidate_value",
    "active_checkpoint_context",
    "active_field_carryover_context",
    "candidate_selection_context",
    "phase_name",
    "save_model",
    "save_model_interval",
    "save_final_model",
    "checkpoint_path",
    "load_step",
    "local_results_path",
    "apply_dense_reward_in_eval",
    "log_dense_reward_details",
    "alg_config",
    "qmix_hparam_preset",
}


def normalize_base_train_config(base_train_config: Dict[str, Any]) -> Dict[str, Any]:
    resolved_config, qmix_hparam_preset = resolve_qmix_hparam_preset(
        base_train_config.get("alg_config")
        or base_train_config.get("config")
        or base_train_config.get("name")
        or "qmix",
        base_train_config.get("qmix_hparam_preset"),
    )
    if (
        isinstance(base_train_config.get("config"), str)
        and isinstance(base_train_config.get("env_config"), str)
        and isinstance(base_train_config.get("overrides"), dict)
    ):
        normalized = deepcopy(base_train_config)
        normalized["alg_config"] = resolved_config
        normalized["qmix_hparam_preset"] = qmix_hparam_preset
        normalized.setdefault("seed", int(base_train_config.get("seed", 1) or 1))
        normalized.setdefault("experiment_metadata", deepcopy(base_train_config.get("experiment_metadata")))
        return normalized

    normalized: Dict[str, Any] = {
        "config": resolved_config,
        "env_config": str(
            base_train_config.get("env_config")
            or base_train_config.get("env")
            or "gymma"
        ),
        "env_args": deepcopy(base_train_config.get("env_args", {}))
        if isinstance(base_train_config.get("env_args"), dict)
        else {},
        "overrides": {},
        "label": str(base_train_config.get("label", "checkpoint_branch")),
        "use_dense_reward": bool(base_train_config.get("use_dense_reward", True)),
        "reward_module_path": str(base_train_config.get("reward_module_path", "") or ""),
        "seed": int(base_train_config.get("seed", 1) or 1),
        "experiment_metadata": deepcopy(base_train_config.get("experiment_metadata")),
        "alpha_policy_config": deepcopy(base_train_config.get("alpha_policy_config")),
        "policy_guidance_config": deepcopy(base_train_config.get("policy_guidance_config")),
        "workflow_id": base_train_config.get("workflow_id"),
        "workflow_round": base_train_config.get("workflow_round"),
        "reward_paradigm": base_train_config.get("reward_paradigm"),
        "active_pbrs_field": base_train_config.get("active_pbrs_field"),
        "active_pbrs_field_source": base_train_config.get("active_pbrs_field_source"),
        "candidate_value": base_train_config.get("candidate_value"),
        "active_checkpoint_context": deepcopy(
            base_train_config.get("active_checkpoint_context")
        ),
        "active_field_carryover_context": deepcopy(
            base_train_config.get("active_field_carryover_context")
        ),
        "candidate_selection_context": deepcopy(
            base_train_config.get("candidate_selection_context")
        ),
        "phase_name": str(base_train_config.get("phase_name", "main")),
        "save_model": bool(base_train_config.get("save_model", False)),
        "save_model_interval": int(base_train_config.get("save_model_interval", 50000)),
        "save_final_model": bool(base_train_config.get("save_final_model", False)),
        "checkpoint_path": str(base_train_config.get("checkpoint_path", "") or ""),
        "load_step": int(base_train_config.get("load_step", 0) or 0),
        "local_results_path": str(base_train_config.get("local_results_path", "results") or "results"),
        "alg_config": resolved_config,
        "qmix_hparam_preset": qmix_hparam_preset,
        "apply_dense_reward_in_eval": bool(
            base_train_config.get("apply_dense_reward_in_eval", False)
        ),
        "log_dense_reward_details": bool(
            base_train_config.get("log_dense_reward_details", True)
        ),
    }

    overrides = {}
    for key, value in base_train_config.items():
        if key in _DIRECT_TRAIN_CONFIG_KEYS:
            continue
        overrides[key] = deepcopy(value)
    normalized["overrides"] = overrides
    return normalized


def normalize_checkpoint_steps(values: Any) -> List[int]:
    if not isinstance(values, list):
        return []
    normalized: List[int] = []
    for value in values:
        try:
            step = int(value)
        except (TypeError, ValueError):
            continue
        if step >= 0 and step not in normalized:
            normalized.append(step)
    normalized.sort()
    return normalized


def resolve_checkpoint_step(
    available_steps: List[int],
    requested_step: Optional[int],
) -> int:
    if not available_steps:
        raise ValueError("no checkpoint steps are available")
    if requested_step is None or int(requested_step) == 0:
        return int(max(available_steps))
    target = int(requested_step)
    return min(available_steps, key=lambda step: abs(int(step) - target))


def build_resume_train_config(
    *,
    base_train_config: Dict[str, Any],
    checkpoint_root_dir: str,
    checkpoint_step: int,
    label_suffix: str = "_resume",
    use_dense_reward: Optional[bool] = None,
    reward_module_path: Optional[str] = None,
    alpha_policy_config: Optional[Dict[str, Any]] = None,
    policy_guidance_config: Optional[Dict[str, Any]] = None,
    workflow_id: Optional[str] = None,
    workflow_round: Optional[int] = None,
    reward_paradigm: Optional[str] = None,
    active_pbrs_field: Optional[str] = None,
    active_pbrs_field_source: Optional[str] = None,
    candidate_value: Optional[float] = None,
    active_checkpoint_context: Optional[Dict[str, Any]] = None,
    active_field_carryover_context: Optional[Dict[str, Any]] = None,
    candidate_selection_context: Optional[Dict[str, Any]] = None,
    phase_name: Optional[str] = None,
    save_model: Optional[bool] = None,
    save_model_interval: Optional[int] = None,
    save_final_model: Optional[bool] = None,
    local_results_path: Optional[str] = None,
    override_env_args: Optional[Dict[str, Any]] = None,
    override_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    train_config = normalize_base_train_config(base_train_config)
    label = str(train_config.get("label", "checkpoint_branch"))
    if label_suffix:
        label = f"{label}{label_suffix}"
    train_config["label"] = label
    train_config["checkpoint_path"] = str(checkpoint_root_dir)
    train_config["load_step"] = int(checkpoint_step)

    if use_dense_reward is not None:
        train_config["use_dense_reward"] = bool(use_dense_reward)
    if reward_module_path is not None:
        train_config["reward_module_path"] = str(reward_module_path)
    if alpha_policy_config is not None:
        train_config["alpha_policy_config"] = deepcopy(alpha_policy_config)
    if policy_guidance_config is not None:
        train_config["policy_guidance_config"] = deepcopy(policy_guidance_config)
    if workflow_id is not None:
        train_config["workflow_id"] = workflow_id
    if workflow_round is not None:
        train_config["workflow_round"] = int(workflow_round)
    if reward_paradigm is not None:
        train_config["reward_paradigm"] = reward_paradigm
    if active_pbrs_field is not None:
        train_config["active_pbrs_field"] = active_pbrs_field
    if active_pbrs_field_source is not None:
        train_config["active_pbrs_field_source"] = active_pbrs_field_source
    if candidate_value is not None:
        train_config["candidate_value"] = float(candidate_value)
    if active_checkpoint_context is not None:
        train_config["active_checkpoint_context"] = deepcopy(active_checkpoint_context)
    if active_field_carryover_context is not None:
        train_config["active_field_carryover_context"] = deepcopy(
            active_field_carryover_context
        )
    if candidate_selection_context is not None:
        train_config["candidate_selection_context"] = deepcopy(
            candidate_selection_context
        )
    if phase_name is not None:
        train_config["phase_name"] = str(phase_name)
    if save_model is not None:
        train_config["save_model"] = bool(save_model)
    if save_model_interval is not None:
        train_config["save_model_interval"] = int(save_model_interval)
    if save_final_model is not None:
        train_config["save_final_model"] = bool(save_final_model)
    if local_results_path is not None:
        train_config["local_results_path"] = str(local_results_path)
    if isinstance(override_env_args, dict):
        merged_env_args = dict(train_config.get("env_args", {}))
        merged_env_args.update(deepcopy(override_env_args))
        train_config["env_args"] = merged_env_args
    if isinstance(override_overrides, dict):
        merged_overrides = dict(train_config.get("overrides", {}))
        merged_overrides.update(override_overrides)
        train_config["overrides"] = merged_overrides

    return train_config
