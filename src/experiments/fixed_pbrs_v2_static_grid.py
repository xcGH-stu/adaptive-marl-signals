from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Dict

from experiments.qmix_hparam_presets import list_qmix_hparam_presets
from rewarding.lbf_pbrs_v2 import DEFAULT_MODE as DEFAULT_LBF_PBRS_V2_MODE
from rewarding.lbf_pbrs_v2 import DEFAULT_MODE_CONFIGS as LBF_PBRS_V2_MODE_CONFIGS

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TIME_LIMIT = 50
DEFAULT_TEST_INTERVAL = 50000
DEFAULT_RUNNER_LOG_INTERVAL = 50000
DEFAULT_LEARNER_LOG_INTERVAL = 50000
# Keep the save interval beyond the full budget so fixed baselines avoid
# intermediate checkpoints while preserving compatibility with the launcher CLI.
DEFAULT_SAVE_MODEL_INTERVAL = 10000001
DEFAULT_LOCAL_RESULTS_PATH = "results"
DEFAULT_EXPERIMENT_FAMILY = "fixed_pbrs_v2_static_grid"
DEFAULT_BUDGET_PROFILE = "fixed_pbrs_v2_static_grid"
DEFAULT_PBRS_VERSION = "lbf_pbrs_v2"
DEFAULT_PBRS_GAMMA = 0.99
DEFAULT_PBRS_VARIANT = "original"
DEFAULT_PBRS_WC = 0.5
DEFAULT_PBRS_WP = 0.5
DEFAULT_RUNTIME_PBRS_MODE = DEFAULT_LBF_PBRS_V2_MODE
SUPPORTED_LBF_PBRS_V2_MODES = tuple(LBF_PBRS_V2_MODE_CONFIGS.keys())

DEFAULT_T_MAX_BY_ALGORITHM: Dict[str, int] = {
    "qmix": 2050000,
    "mappo": 10000000,
}

LBF_ENV_SPECS: Dict[str, Dict[str, Any]] = {
    "8x8-2p-1f": {
        "env_label": "8x8-2p-1f",
        "env_key": "lbforaging:Foraging-8x8-2p-1f-v3",
        "time_limit": 50,
        "evidence": [
            "policy_method/results/sacred/qmix/\"lbforaging:Foraging-8x8-2p-1f-v3\"/*",
            "policy_method/src/scripts/run_fixed_reward_baseline.py default",
        ],
    },
    "8x8-2p-2f": {
        "env_label": "8x8-2p-2f",
        "env_key": "lbforaging:Foraging-8x8-2p-2f-coop-v3",
        "time_limit": 50,
        "evidence": [
            "policy_method/src/scripts/diagnostics/check_lbf_env_support.py candidate order",
            "policy_method/src/scripts/diagnostics/check_qmix_model52_four_env_stage1_reuse_rerun.py",
            "policy_method/reports/policy_method_mappo_qmix_status_summary.json",
        ],
    },
    "2s-8x8-2p-2f": {
        "env_label": "2s-8x8-2p-2f",
        "env_key": "lbforaging:Foraging-2s-8x8-2p-2f-coop-v3",
        "time_limit": 50,
        "evidence": [
            "policy_method/src/scripts/diagnostics/check_lbf_env_support.py candidate order",
            "policy_method/src/scripts/diagnostics/check_qmix_model52_four_env_stage1_reuse_rerun.py",
            "policy_method/reports/policy_method_mappo_qmix_status_summary.json",
        ],
    },
    "10x10-2p-1f": {
        "env_label": "10x10-2p-1f",
        "env_key": "lbforaging:Foraging-10x10-2p-1f-v3",
        "time_limit": 50,
        "evidence": [
            "policy_method/src/scripts/diagnostics/check_lbf_policy_guided_multienv_multialgo_support.py",
            "policy_method/src/scripts/diagnostics/check_qmix_model52_four_env_stage1_reuse_rerun.py",
            "policy_method/results/sacred/qmix/\"lbforaging:Foraging-10x10-2p-1f-v3\"/*",
        ],
    },
    "15x15-3p-4f": {
        "env_label": "15x15-3p-4f",
        "env_key": "lbforaging:Foraging-15x15-3p-4f-v3",
        "time_limit": 50,
        "evidence": [
            "policy_method/src/scripts/diagnostics/check_lbf_policy_guided_multienv_multialgo_support.py",
            "policy_method/src/scripts/diagnostics/check_qmix_model52_four_env_stage1_reuse_rerun.py",
            "policy_method/reports/policy_method_mappo_qmix_status_summary.json",
        ],
    },
}


FIXED_PBRS_V2_STATIC_CONFIGS: Dict[str, Dict[str, Any]] = {
    "balanced_progress": {
        "config_id": "balanced_progress",
        "pbrs_version": "lbf_pbrs_v2",
        "runtime_mode": DEFAULT_RUNTIME_PBRS_MODE,
        "beta": 0.3,
        "active_terms": ["col", "app", "cov", "ready"],
        "weights": {
            "col": 0.25,
            "app": 0.25,
            "cov": 0.25,
            "ready": 0.25,
            "alloc": 0.0,
            "stab": 0.0,
        },
    },
    "early_discovery": {
        "config_id": "early_discovery",
        "pbrs_version": "lbf_pbrs_v2",
        "runtime_mode": DEFAULT_RUNTIME_PBRS_MODE,
        "beta": 0.4,
        "active_terms": ["col", "app", "cov"],
        "weights": {
            "col": 0.15,
            "app": 0.25,
            "cov": 0.60,
            "ready": 0.0,
            "alloc": 0.0,
            "stab": 0.0,
        },
    },
    "collection_readiness": {
        "config_id": "collection_readiness",
        "pbrs_version": "lbf_pbrs_v2",
        "runtime_mode": DEFAULT_RUNTIME_PBRS_MODE,
        "beta": 0.5,
        "active_terms": ["col", "app", "cov", "ready"],
        "weights": {
            "col": 0.20,
            "app": 0.20,
            "cov": 0.20,
            "ready": 0.40,
            "alloc": 0.0,
            "stab": 0.0,
        },
    },
}

DEFAULT_ENV_KEY = str(LBF_ENV_SPECS["8x8-2p-1f"]["env_key"])
DEFAULT_T_MAX = int(DEFAULT_T_MAX_BY_ALGORITHM["qmix"])


def list_config_ids() -> list[str]:
    return sorted(FIXED_PBRS_V2_STATIC_CONFIGS.keys())


def list_algorithm_ids() -> list[str]:
    return sorted(DEFAULT_T_MAX_BY_ALGORITHM.keys())


def list_lbf_env_labels() -> list[str]:
    return list(LBF_ENV_SPECS.keys())


def list_supported_lbf_pbrs_v2_modes() -> list[str]:
    return sorted(SUPPORTED_LBF_PBRS_V2_MODES)


def get_static_config(config_id: str) -> Dict[str, Any]:
    try:
        return deepcopy(FIXED_PBRS_V2_STATIC_CONFIGS[config_id])
    except KeyError as exc:
        available = ", ".join(list_config_ids())
        raise ValueError(f"unknown fixed PBRS-v2 static config: {config_id}. available: {available}") from exc


def get_lbf_env_spec(env_label: str) -> Dict[str, Any]:
    try:
        return deepcopy(LBF_ENV_SPECS[env_label])
    except KeyError as exc:
        available = ", ".join(list_lbf_env_labels())
        raise ValueError(f"unknown LBF env label: {env_label}. available: {available}") from exc


def normalize_algorithm(algorithm: str) -> str:
    value = str(algorithm or "").strip().lower()
    if value not in DEFAULT_T_MAX_BY_ALGORITHM:
        available = ", ".join(list_algorithm_ids())
        raise ValueError(f"unknown algorithm: {algorithm}. available: {available}")
    return value


def default_t_max_for_algorithm(algorithm: str) -> int:
    return int(DEFAULT_T_MAX_BY_ALGORITHM[normalize_algorithm(algorithm)])


def default_env_key() -> str:
    return DEFAULT_ENV_KEY


def default_time_limit_for_env_key(env_key: str) -> int:
    for spec in LBF_ENV_SPECS.values():
        if str(spec["env_key"]) == str(env_key):
            return int(spec.get("time_limit") or DEFAULT_TIME_LIMIT)
    return DEFAULT_TIME_LIMIT


def _slugify_env_key(env_key: str) -> str:
    text = str(env_key)
    if text.startswith("lbforaging:Foraging-"):
        text = text.split("lbforaging:Foraging-", 1)[1]
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "env"


def build_workflow_id(
    config_id: str,
    seed: int,
    *,
    algorithm: str = "qmix",
    env_key: str | None = None,
    run_index: int = 1,
) -> str:
    get_static_config(config_id)
    algo = normalize_algorithm(algorithm)
    resolved_env_key = str(env_key or default_env_key())
    if algo == "qmix" and resolved_env_key == default_env_key():
        return f"qmix_fixed_pbrs_v2_static_{config_id}_seed{int(seed)}_run{int(run_index)}"
    env_slug = _slugify_env_key(resolved_env_key)
    return f"{algo}_fixed_pbrs_v2_static_{env_slug}_{config_id}_seed{int(seed)}_run{int(run_index)}"


def build_launcher_output_path(
    config_id: str,
    seed: int,
    *,
    algorithm: str = "qmix",
    env_key: str | None = None,
    run_index: int = 1,
) -> str:
    workflow_id = build_workflow_id(
        config_id,
        seed,
        algorithm=algorithm,
        env_key=env_key,
        run_index=run_index,
    )
    return str(ROOT / "reports" / f"{workflow_id}.launcher_output.json")


def build_baseline_kwargs(
    *,
    config_id: str,
    algorithm: str,
    env_key: str,
    seed: int,
    t_max: int | None,
    run_index: int = 1,
    python_executable: str,
    use_cuda: bool,
    execute: bool,
    qmix_hparam_preset: str = "default",
    output_path: str | None = None,
    result_root: str = DEFAULT_LOCAL_RESULTS_PATH,
    format_name: str = "json",
) -> Dict[str, Any]:
    static_config = get_static_config(config_id)
    algo = normalize_algorithm(algorithm)
    if qmix_hparam_preset not in list_qmix_hparam_presets():
        available = ", ".join(list_qmix_hparam_presets())
        raise ValueError(
            f"unknown qmix_hparam_preset: {qmix_hparam_preset}. available: {available}"
        )
    resolved_env_key = str(env_key)
    resolved_t_max = int(t_max) if t_max is not None else default_t_max_for_algorithm(algo)
    workflow_id = build_workflow_id(
        config_id,
        seed,
        algorithm=algo,
        env_key=resolved_env_key,
        run_index=run_index,
    )
    return {
        "run_id": workflow_id,
        "reward_paradigm": "pbrs",
        "native_original_pbrs": True,
        "native_stage_conditioned_pbrs": False,
        "python_executable": python_executable,
        "env_key": resolved_env_key,
        "train_config": algo,
        "qmix_hparam_preset": str(qmix_hparam_preset),
        "use_cuda": bool(use_cuda),
        "seed": int(seed),
        "time_limit": default_time_limit_for_env_key(resolved_env_key),
        "t_max": resolved_t_max,
        "test_interval": DEFAULT_TEST_INTERVAL,
        "runner_log_interval": DEFAULT_RUNNER_LOG_INTERVAL,
        "learner_log_interval": DEFAULT_LEARNER_LOG_INTERVAL,
        "save_model_interval": max(DEFAULT_SAVE_MODEL_INTERVAL, resolved_t_max + 1),
        "local_results_path": result_root,
        "checkpoint_path": "",
        "load_step": 0,
        "enable_checkpointing": False,
        "save_final_model": False,
        "reward_spec_path": None,
        "output_path": output_path
        or build_launcher_output_path(
            config_id,
            seed,
            algorithm=algo,
            env_key=resolved_env_key,
            run_index=run_index,
        ),
        "pbrs_beta": float(static_config["beta"]),
        "pbrs_wc": DEFAULT_PBRS_WC,
        "pbrs_wp": DEFAULT_PBRS_WP,
        "pbrs_gamma": DEFAULT_PBRS_GAMMA,
        "pbrs_variant": DEFAULT_PBRS_VARIANT,
        "pbrs_version": DEFAULT_PBRS_VERSION,
        "pbrs_mode": str(static_config["runtime_mode"]),
        "pbrs_active_terms": list(static_config["active_terms"]),
        "pbrs_weights": deepcopy(static_config["weights"]),
        "eval_use_pbrs": False,
        "pbrs_stage_boundary_1": 875923,
        "pbrs_stage_boundary_2": 1451234,
        "pbrs_stage_beta_early": 0.3,
        "pbrs_stage_beta_mid": 1.0,
        "pbrs_stage_beta_late": 0.5,
        "pbrs_stage_wc_early": 0.1,
        "pbrs_stage_wc_mid": 1.0,
        "pbrs_stage_wc_late": 0.7,
        "pbrs_stage_wp_early": 0.7,
        "pbrs_stage_wp_mid": 0.1,
        "pbrs_stage_wp_late": 0.3,
        "execute": bool(execute),
        "format": format_name,
        "experiment_family": DEFAULT_EXPERIMENT_FAMILY,
        "budget_profile": DEFAULT_BUDGET_PROFILE,
        "experiment_tag": workflow_id,
    }


def build_expected_summary() -> Dict[str, Any]:
    return {
        "config_ids": list_config_ids(),
        "algorithm_ids": list_algorithm_ids(),
        "lbf_env_specs": deepcopy(LBF_ENV_SPECS),
        "default_env_key": default_env_key(),
        "default_t_max_by_algorithm": deepcopy(DEFAULT_T_MAX_BY_ALGORITHM),
        "supported_lbf_pbrs_v2_modes": list_supported_lbf_pbrs_v2_modes(),
        "default_runtime_pbrs_mode": DEFAULT_RUNTIME_PBRS_MODE,
        "default_test_interval": DEFAULT_TEST_INTERVAL,
        "default_eval_use_pbrs": False,
        "default_use_dense_reward": False,
        "default_apply_dense_reward_in_eval": False,
        "static_grid_uses_llm": False,
        "static_grid_uses_policy_guidance": False,
        "static_grid_uses_stage1b_candidate_search": False,
        "static_grid_uses_stage3_adaptive_replacement": False,
        "rware_implemented": False,
        "lbf_only": True,
    }


def static_config_json(config_id: str) -> str:
    return json.dumps(get_static_config(config_id), sort_keys=True, ensure_ascii=False)
