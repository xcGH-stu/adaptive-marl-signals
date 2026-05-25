from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rewarding.lbf_pbrs_v2 import normalize_lbf_pbrs_v2_config
from rewarding.rware_pbrs_v2 import normalize_rware_pbrs_v2_config
from scripts.run_reward_workflow import build_default_workflow_spec
from scripts.run_fixed_reward_baseline import build_fixed_reward_baseline_plan
from workflows.train_launcher import EPyMARLTrainLauncher


PROMPT_VERSION = "v1"
ROOT = Path(__file__).resolve().parents[2]


def llm_cache_key_suffix(*, model: str, prompt_version: str = PROMPT_VERSION) -> str:
    model_slug = "".join(
        ch if ch.isalnum() else "_"
        for ch in str(model or "unknown")
    ).strip("_")
    if not model_slug:
        model_slug = "unknown"
    prompt_slug = "".join(
        ch if ch.isalnum() else "_"
        for ch in str(prompt_version or PROMPT_VERSION)
    ).strip("_")
    if not prompt_slug:
        prompt_slug = "unknown"
    return f"{model_slug}__{prompt_slug}"


def _coerce_checkpoint_step(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_checkpoint_resume_source(
    *,
    checkpoint_path: str,
    checkpoint_step: Any,
) -> Dict[str, Any]:
    raw_path = str(checkpoint_path or "").strip()
    step = _coerce_checkpoint_step(checkpoint_step)
    source_path = Path(raw_path) if raw_path else None
    source_name = source_path.name if source_path is not None else ""
    inferred_leaf_step = int(source_name) if source_name.isdigit() else None

    if source_path is None:
        return {
            "source_checkpoint_path": raw_path,
            "source_checkpoint_root_dir": None,
            "source_checkpoint_step": step,
            "checkpoint_path_argument": None,
            "load_step_argument": step,
            "source_checkpoint_path_exists": False,
            "checkpoint_root_valid": False,
            "load_step_resolved": bool(step is not None),
            "planning_error": "selected_checkpoint_path_missing",
            "planning_warning": None,
            "source_is_mock": False,
        }

    resolved_step = step if step is not None else inferred_leaf_step
    if inferred_leaf_step is not None:
        checkpoint_root = source_path.parent
    else:
        checkpoint_root = source_path

    lower_path = raw_path.lower()
    source_is_mock = (
        not source_path.is_absolute()
        or "/mock_" in lower_path
        or "/mock/" in lower_path
        or "synthetic" in lower_path
    )

    if resolved_step is None:
        return {
            "source_checkpoint_path": raw_path,
            "source_checkpoint_root_dir": str(checkpoint_root),
            "source_checkpoint_step": None,
            "checkpoint_path_argument": str(checkpoint_root),
            "load_step_argument": None,
            "source_checkpoint_path_exists": source_path.exists(),
            "checkpoint_root_valid": False,
            "load_step_resolved": False,
            "planning_error": "selected_checkpoint_step_unresolved",
            "planning_warning": None,
            "source_is_mock": source_is_mock,
        }

    planning_warning = None
    if inferred_leaf_step is not None and step is not None and inferred_leaf_step != step:
        planning_warning = (
            f"selected checkpoint path leaf step {inferred_leaf_step} did not match "
            f"selected_checkpoint_step={step}; using selected_checkpoint_step"
        )

    checkpoint_root_valid = bool(checkpoint_root.exists() and checkpoint_root.is_dir())
    return {
        "source_checkpoint_path": raw_path,
        "source_checkpoint_root_dir": str(checkpoint_root),
        "source_checkpoint_step": int(resolved_step),
        "checkpoint_path_argument": str(checkpoint_root),
        "load_step_argument": int(resolved_step),
        "source_checkpoint_path_exists": source_path.exists(),
        "checkpoint_root_valid": checkpoint_root_valid,
        "load_step_resolved": True,
        "planning_error": None,
        "planning_warning": planning_warning,
        "source_is_mock": source_is_mock,
    }


def ensure_json_cache(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return payload


def build_llm_call_record(
    *,
    workflow_dir: Path,
    cache_name: str,
    agent_role: str,
    agent_name: str,
    model: str = "gpt-5.2",
    prompt_path: Path,
    input_payload: Dict[str, Any],
    parsed_response: Dict[str, Any],
    validation_errors: Optional[List[str]] = None,
    llm_routing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prompt_version = PROMPT_VERSION
    cache_path = workflow_dir / "llm_cache" / f"{cache_name}__{llm_cache_key_suffix(model=model, prompt_version=prompt_version)}.json"
    raw_response = {
        "mock": True,
        "agent_role": agent_role,
        "agent_name": agent_name,
    }
    payload = {
        "agent_role": agent_role,
        "agent_name": agent_name,
        "model": str(model),
        "llm_routing": deepcopy(llm_routing or {}),
        "prompt_path": str(prompt_path),
        "prompt_version": prompt_version,
        "input_payload": deepcopy(input_payload),
        "raw_response": raw_response,
        "parsed_response": deepcopy(parsed_response),
        "validation_errors": list(validation_errors or []),
        "cache_path": str(cache_path),
        "status": "cache_seed",
    }
    existing = ensure_json_cache(cache_path, payload)
    if existing is not payload:
        existing["status"] = "cache_hit"
        return existing
    return payload


def build_fixed_checkpoint_plan() -> List[Dict[str, Any]]:
    return [
        {
            "name": "C1",
            "step": 1000000,
            "stage_label": "post_initialization_transition",
            "intervention_question": (
                "Does the early-selected dense reward configuration remain suitable after the initial 800k dense search phase?"
            ),
            "recommended_candidate_focus": [
                "beta_down",
                "beta_up",
                "wc_down_wp_up",
                "reference_like",
                "recovery",
            ],
            "branch_budget_feasible": True,
        },
        {
            "name": "C2",
            "step": 1500000,
            "stage_label": "late_stability",
            "intervention_question": (
                "Does the current reward configuration remain stable and task-aligned in the later training phase?"
            ),
            "recommended_candidate_focus": [
                "beta_down",
                "wc_down_wp_up",
                "reference_like",
                "recovery",
            ],
            "branch_budget_feasible": True,
        },
    ]


def _baseline_args(
    *,
    run_id: str,
    env_key: str,
    train_config: str,
    seed: int,
    time_limit: int,
    t_max: int,
    use_cuda: bool,
    local_results_path: str,
    experiment_family: str,
    budget_profile: str,
    experiment_tag: str,
    pbrs_config: Dict[str, Any],
    checkpoint_path: str = "",
    load_step: int = 0,
    save_model_interval: int = 50000,
    enable_checkpointing: bool = True,
    save_final_model: bool = False,
) -> argparse.Namespace:
    pbrs_version = str(pbrs_config.get("pbrs_version") or "")
    normalized_pbrs = (
        normalize_lbf_pbrs_v2_config(pbrs_config)
        if pbrs_version == "lbf_pbrs_v2"
        else normalize_rware_pbrs_v2_config(pbrs_config)
        if pbrs_version == "rware_pbrs_v2"
        else deepcopy(pbrs_config)
    )
    beta_value = float(normalized_pbrs.get("beta", 0.3))
    wc_value = float(normalized_pbrs.get("wc", 0.0))
    wp_value = float(normalized_pbrs.get("wp", 0.0))
    return argparse.Namespace(
        run_id=run_id,
        reward_paradigm="pbrs",
        native_original_pbrs=True,
        native_stage_conditioned_pbrs=False,
        python_executable="python",
        env_key=env_key,
        train_config=train_config,
        use_cuda=bool(use_cuda),
        seed=int(seed),
        time_limit=int(time_limit),
        t_max=int(t_max),
        test_interval=50000,
        runner_log_interval=50000,
        learner_log_interval=50000,
        save_model_interval=int(save_model_interval),
        local_results_path=local_results_path,
        checkpoint_path=checkpoint_path,
        load_step=int(load_step),
        enable_checkpointing=bool(enable_checkpointing),
        save_final_model=bool(save_final_model),
        reward_spec_path=None,
        output_path=None,
        pbrs_beta=beta_value,
        pbrs_wc=wc_value,
        pbrs_wp=wp_value,
        pbrs_version=str(normalized_pbrs.get("pbrs_version") or ""),
        pbrs_mode=str(normalized_pbrs.get("mode") or ""),
        pbrs_active_terms=list(normalized_pbrs.get("active_terms") or []),
        pbrs_weights=deepcopy(normalized_pbrs.get("weights") or {}),
        pbrs_gamma=float(normalized_pbrs.get("gamma", 0.99)),
        pbrs_variant=str(normalized_pbrs.get("variant", "original")),
        eval_use_pbrs=False,
        pbrs_stage_boundary_1=875923,
        pbrs_stage_boundary_2=1451234,
        pbrs_stage_beta_early=beta_value,
        pbrs_stage_beta_mid=beta_value,
        pbrs_stage_beta_late=beta_value,
        pbrs_stage_wc_early=wc_value,
        pbrs_stage_wc_mid=wc_value,
        pbrs_stage_wc_late=wc_value,
        pbrs_stage_wp_early=wp_value,
        pbrs_stage_wp_mid=wp_value,
        pbrs_stage_wp_late=wp_value,
        execute=False,
        format="json",
        experiment_family=experiment_family,
        budget_profile=budget_profile,
        experiment_tag=experiment_tag,
    )


def _rware_stage1b_candidate_reward_module_path(*, workflow_id: str, candidate_id: str) -> Path:
    modules_dir = ROOT / "results" / "dual_llm_stage1b_rware_reward_modules" / workflow_id
    modules_dir.mkdir(parents=True, exist_ok=True)
    return modules_dir / f"{candidate_id}.py"


def _render_rware_stage1b_candidate_reward_module(candidate: Dict[str, Any]) -> str:
    config_json = json.dumps(candidate, indent=2, sort_keys=True, ensure_ascii=False)
    return (
        "from copy import deepcopy\n"
        "import importlib.util\n"
        "from pathlib import Path\n"
        "import sys\n\n"
        "ROOT = Path(__file__).resolve().parents[3]\n"
        "SRC_DIR = ROOT / 'src'\n"
        "if str(SRC_DIR) not in sys.path:\n"
        "    sys.path.insert(0, str(SRC_DIR))\n\n"
        "MODULE_PATH = SRC_DIR / 'rewarding' / 'rware_pbrs_v2.py'\n"
        "SPEC = importlib.util.spec_from_file_location('dual_llm_stage1b_rware_runtime_module', MODULE_PATH)\n"
        "if SPEC is None or SPEC.loader is None:\n"
        "    raise RuntimeError(f'Unable to load RWARE runtime module from {MODULE_PATH}')\n"
        "RWARE_RUNTIME_MODULE = importlib.util.module_from_spec(SPEC)\n"
        "SPEC.loader.exec_module(RWARE_RUNTIME_MODULE)\n"
        "RWAREPBRSV2Runtime = RWARE_RUNTIME_MODULE.RWAREPBRSV2Runtime\n\n"
        f"CANDIDATE_CONFIG = {config_json}\n"
        "_RUNTIME = RWAREPBRSV2Runtime(deepcopy(CANDIDATE_CONFIG))\n\n"
        "def compute_dense_reward(context):\n"
        "    return _RUNTIME.compute_dense_reward(context)\n"
    )


def _build_rware_stage1b_candidate_plan(
    *,
    workflow_id: str,
    round_id: int,
    candidate: Dict[str, Any],
    env_key: str,
    train_config: str,
    seed: int,
    time_limit: int,
    local_results_path: str,
    budget_steps: int,
    use_cuda: bool,
    budget_profile: str,
    experiment_tag: str,
) -> Dict[str, Any]:
    normalized = normalize_rware_pbrs_v2_config(candidate)
    reward_module_path = _rware_stage1b_candidate_reward_module_path(
        workflow_id=workflow_id,
        candidate_id=str(candidate["candidate_id"]),
    )
    reward_module_path.write_text(
        _render_rware_stage1b_candidate_reward_module(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_type": str(candidate.get("candidate_type") or "balanced"),
                "pbrs_version": "rware_pbrs_v2",
                "mode": str(normalized.get("mode") or ""),
                "beta": float(normalized.get("beta", 0.0)),
                "active_terms": list(normalized.get("active_terms") or []),
                "weights": deepcopy(normalized.get("weights") or {}),
                "gamma": float(normalized.get("gamma", 0.99)),
            }
        ),
        encoding="utf-8",
    )
    workflow_spec = deepcopy(build_default_workflow_spec(1, reward_paradigm="pbrs"))
    train_spec = workflow_spec["train"]
    train_spec["config"] = train_config
    train_spec["seed"] = int(seed)
    train_spec["env_config"] = "gymma"
    train_spec["env_args"]["key"] = env_key
    train_spec["env_args"]["time_limit"] = int(time_limit)
    train_spec["env_args"]["eval_use_pbrs"] = False
    train_spec["env_args"]["pbrs_version"] = "rware_pbrs_v2"
    train_spec["env_args"]["pbrs_mode"] = str(normalized.get("mode") or "")
    train_spec["env_args"]["pbrs_beta"] = float(normalized.get("beta", 0.0))
    train_spec["env_args"]["pbrs_active_terms"] = list(normalized.get("active_terms") or [])
    train_spec["env_args"]["pbrs_weights"] = deepcopy(normalized.get("weights") or {})
    train_spec["env_args"]["pbrs_gamma"] = float(normalized.get("gamma", 0.99))
    train_spec["use_dense_reward"] = True
    train_spec["overrides"]["t_max"] = int(budget_steps)
    train_spec["overrides"]["test_interval"] = 50000
    train_spec["overrides"]["runner_log_interval"] = 50000
    train_spec["overrides"]["learner_log_interval"] = 50000
    train_spec["overrides"]["use_cuda"] = bool(use_cuda)
    train_spec["checkpointing"] = {
        "enabled": True,
        "save_model_interval": int(budget_steps),
        "save_final_model": False,
        "checkpoint_path": "",
        "load_step": 0,
        "local_results_path": local_results_path,
    }
    train_spec["apply_dense_reward_in_eval"] = False
    train_spec["log_dense_reward_details"] = True
    workflow_spec["workflow_id"] = workflow_id
    workflow_spec["experiment_metadata"] = {
        "experiment_family": "dual_llm_stage1b_initial_dense_search",
        "budget_profile": budget_profile,
        "seed": int(seed),
        "experiment_tag": experiment_tag,
        "native_original_pbrs": False,
        "native_stage_conditioned_pbrs": False,
    }
    launcher = EPyMARLTrainLauncher(repo_root=ROOT, python_executable="python")
    training_plan = launcher.build_training_plan(
        round_id=0,
        workflow_spec=workflow_spec,
        reward_module_path=str(reward_module_path),
        alpha_policy={"type": "constant", "value": 1.0},
        policy_guidance_spec={},
        workflow_id=workflow_id,
        candidate_id=None,
        reward_paradigm="pbrs",
        active_pbrs_field=None,
        active_pbrs_field_source=None,
        candidate_value=None,
        active_checkpoint_context=None,
        active_field_carryover_context=None,
        candidate_selection_context={"stage": "stage1b", "mode": "rware_guarded_formal_execute_smoke"},
        phase_name="fixed_reward_baseline",
        train_overrides_override=None,
        label_override=f"fixed_reward_{workflow_id}",
    )
    return {
        "run_id": workflow_id,
        "reward_paradigm": "pbrs",
        "native_original_pbrs": False,
        "native_stage_conditioned_pbrs": False,
        "experiment_metadata": deepcopy(workflow_spec["experiment_metadata"]),
        "reward_spec_path": None,
        "reward_module_path": str(reward_module_path),
        "workflow_spec": workflow_spec,
        "train_config": training_plan["train_config"],
        "command": training_plan["command"],
    }


def _build_rware_dense_resume_plan(
    *,
    workflow_id: str,
    selected_candidate: Dict[str, Any],
    checkpoint_path_argument: str,
    load_step_argument: int,
    env_key: str,
    train_config: str,
    seed: int,
    time_limit: int,
    local_results_path: str,
    target_t_max: int,
    use_cuda: bool,
    budget_profile: str,
    experiment_family: str,
    experiment_tag: str,
) -> Dict[str, Any]:
    normalized = normalize_rware_pbrs_v2_config(selected_candidate)
    reward_module_path = _rware_stage1b_candidate_reward_module_path(
        workflow_id=workflow_id,
        candidate_id=str(selected_candidate["candidate_id"]),
    )
    reward_module_path.write_text(
        _render_rware_stage1b_candidate_reward_module(
            {
                "candidate_id": str(selected_candidate["candidate_id"]),
                "candidate_type": str(selected_candidate.get("candidate_type") or "reference_like"),
                "pbrs_version": "rware_pbrs_v2",
                "mode": str(normalized.get("mode") or ""),
                "beta": float(normalized.get("beta", 0.0)),
                "active_terms": list(normalized.get("active_terms") or []),
                "weights": deepcopy(normalized.get("weights") or {}),
                "gamma": float(normalized.get("gamma", 0.99)),
            }
        ),
        encoding="utf-8",
    )
    workflow_spec = deepcopy(build_default_workflow_spec(1, reward_paradigm="pbrs"))
    train_spec = workflow_spec["train"]
    train_spec["config"] = train_config
    train_spec["seed"] = int(seed)
    train_spec["env_config"] = "gymma"
    train_spec["env_args"]["key"] = env_key
    train_spec["env_args"]["time_limit"] = int(time_limit)
    train_spec["env_args"]["eval_use_pbrs"] = False
    train_spec["env_args"]["pbrs_version"] = "rware_pbrs_v2"
    train_spec["env_args"]["pbrs_mode"] = str(normalized.get("mode") or "")
    train_spec["env_args"]["pbrs_beta"] = float(normalized.get("beta", 0.0))
    train_spec["env_args"]["pbrs_active_terms"] = list(normalized.get("active_terms") or [])
    train_spec["env_args"]["pbrs_weights"] = deepcopy(normalized.get("weights") or {})
    train_spec["env_args"]["pbrs_gamma"] = float(normalized.get("gamma", 0.99))
    train_spec["use_dense_reward"] = True
    train_spec["overrides"]["t_max"] = int(target_t_max)
    train_spec["overrides"]["test_interval"] = 50000
    train_spec["overrides"]["runner_log_interval"] = 50000
    train_spec["overrides"]["learner_log_interval"] = 50000
    train_spec["overrides"]["use_cuda"] = bool(use_cuda)
    train_spec["checkpointing"] = {
        "enabled": False,
        "save_model_interval": int(target_t_max),
        "save_final_model": False,
        "checkpoint_path": str(checkpoint_path_argument or ""),
        "load_step": int(load_step_argument or 0),
        "local_results_path": local_results_path,
    }
    train_spec["apply_dense_reward_in_eval"] = False
    train_spec["log_dense_reward_details"] = True
    workflow_spec["workflow_id"] = workflow_id
    workflow_spec["experiment_metadata"] = {
        "experiment_family": experiment_family,
        "budget_profile": budget_profile,
        "seed": int(seed),
        "experiment_tag": experiment_tag,
        "native_original_pbrs": False,
        "native_stage_conditioned_pbrs": False,
    }
    launcher = EPyMARLTrainLauncher(repo_root=ROOT, python_executable="python")
    training_plan = launcher.build_training_plan(
        round_id=0,
        workflow_spec=workflow_spec,
        reward_module_path=str(reward_module_path),
        alpha_policy={"type": "constant", "value": 1.0},
        policy_guidance_spec={},
        workflow_id=workflow_id,
        candidate_id=None,
        reward_paradigm="pbrs",
        active_pbrs_field=None,
        active_pbrs_field_source=None,
        candidate_value=None,
        active_checkpoint_context=None,
        active_field_carryover_context=None,
        candidate_selection_context={"stage": "stage1b", "mode": "rware_guarded_formal_execute_smoke"},
        phase_name="fixed_reward_baseline",
        train_overrides_override=None,
        label_override=f"fixed_reward_{workflow_id}",
    )
    return {
        "run_id": workflow_id,
        "reward_paradigm": "pbrs",
        "native_original_pbrs": False,
        "native_stage_conditioned_pbrs": False,
        "experiment_metadata": deepcopy(workflow_spec["experiment_metadata"]),
        "reward_spec_path": None,
        "reward_module_path": str(reward_module_path),
        "workflow_spec": workflow_spec,
        "train_config": training_plan["train_config"],
        "command": training_plan["command"],
    }


def build_stage1b_candidate_plan(
    *,
    workflow_id: str,
    round_id: int,
    candidate: Dict[str, Any],
    env_key: str,
    train_config: str,
    seed: int,
    time_limit: int,
    local_results_path: str,
    budget_steps: int = 850000,
    target_checkpoint_step: Optional[int] = None,
    use_cuda: bool = True,
    budget_profile: str = "mock",
    experiment_tag: Optional[str] = None,
    endpoint_tolerance_steps: Optional[int] = None,
    endpoint_selection_policy: str = "prefer_ge_then_le_nearest",
) -> Dict[str, Any]:
    target_checkpoint = int(target_checkpoint_step or budget_steps)
    save_model_interval = max(1, min(int(target_checkpoint), int(budget_steps)))
    run_id = f"{workflow_id}_stage1b_round{round_id}_{candidate['candidate_id']}"
    if str(candidate.get("pbrs_version") or "") == "rware_pbrs_v2":
        plan = _build_rware_stage1b_candidate_plan(
            workflow_id=run_id,
            round_id=round_id,
            candidate=candidate,
            env_key=env_key,
            train_config=train_config,
            seed=seed,
            time_limit=time_limit,
            local_results_path=local_results_path,
            budget_steps=int(budget_steps),
            use_cuda=bool(use_cuda),
            budget_profile=budget_profile,
            experiment_tag=experiment_tag or workflow_id,
        )
    else:
        plan = build_fixed_reward_baseline_plan(
            _baseline_args(
                run_id=run_id,
                env_key=env_key,
                train_config=train_config,
                seed=seed,
                time_limit=time_limit,
                t_max=int(budget_steps),
                use_cuda=bool(use_cuda),
                local_results_path=local_results_path,
                experiment_family="dual_llm_stage1b_initial_dense_search",
                budget_profile=budget_profile,
                experiment_tag=experiment_tag or workflow_id,
                pbrs_config=candidate,
                save_model_interval=save_model_interval,
                enable_checkpointing=True,
                save_final_model=False,
            )
        )
    plan["candidate_id"] = candidate["candidate_id"]
    plan["round_id"] = int(round_id)
    plan["candidate_training_t_max"] = int(budget_steps)
    plan["target_endpoint_step"] = int(target_checkpoint)
    plan["endpoint_tolerance_steps"] = (
        None if endpoint_tolerance_steps is None else int(endpoint_tolerance_steps)
    )
    plan["endpoint_selection_policy"] = str(endpoint_selection_policy)
    plan["endpoint_checkpoint_step"] = int(target_checkpoint)
    plan["endpoint_checkpoint_path"] = (
        f"{local_results_path}/models/{run_id}/{int(target_checkpoint)}"
    )
    return plan


def build_dense_reference_continuation_plan(
    *,
    workflow_id: str,
    selected_candidate: Dict[str, Any],
    selected_checkpoint_path: str,
    selected_checkpoint_step: int,
    env_key: str,
    train_config: str,
    seed: int,
    time_limit: int,
    local_results_path: str,
    target_t_max: int,
    use_cuda: bool = True,
    budget_profile: str = "mock",
    experiment_tag: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_source = _resolve_checkpoint_resume_source(
        checkpoint_path=selected_checkpoint_path,
        checkpoint_step=selected_checkpoint_step,
    )
    selected_step = resolved_source.get("source_checkpoint_step")
    target_t_max_value = int(target_t_max)
    planning_error = resolved_source.get("planning_error")
    if selected_step is not None and target_t_max_value <= int(selected_step):
        planning_error = (
            f"dense_reference_target_t_max_not_greater_than_source_step:"
            f"{target_t_max_value}<={int(selected_step)}"
        )
    continuation_run_id = run_id or f"{workflow_id}_fixed_dense_continuation_reference"
    if str(selected_candidate.get("pbrs_version") or "") == "rware_pbrs_v2":
        plan = _build_rware_dense_resume_plan(
            workflow_id=continuation_run_id,
            selected_candidate=selected_candidate,
            checkpoint_path_argument=str(resolved_source.get("checkpoint_path_argument") or ""),
            load_step_argument=int(resolved_source.get("load_step_argument") or 0),
            env_key=env_key,
            train_config=train_config,
            seed=seed,
            time_limit=time_limit,
            local_results_path=local_results_path,
            target_t_max=target_t_max_value,
            use_cuda=bool(use_cuda),
            budget_profile=budget_profile,
            experiment_family="selected_checkpoint_dense_reference",
            experiment_tag=experiment_tag or workflow_id,
        )
    else:
        plan = build_fixed_reward_baseline_plan(
            _baseline_args(
                run_id=continuation_run_id,
                env_key=env_key,
                train_config=train_config,
                seed=seed,
                time_limit=time_limit,
                t_max=target_t_max_value,
                use_cuda=bool(use_cuda),
                local_results_path=local_results_path,
                experiment_family="selected_checkpoint_dense_reference",
                budget_profile=budget_profile,
                experiment_tag=experiment_tag or workflow_id,
                pbrs_config=selected_candidate,
                checkpoint_path=str(resolved_source.get("checkpoint_path_argument") or ""),
                load_step=int(resolved_source.get("load_step_argument") or 0),
                save_model_interval=max(1, target_t_max_value),
                enable_checkpointing=False,
                save_final_model=False,
            )
        )
    plan["dense_reference_source_candidate_id"] = selected_candidate["candidate_id"]
    plan["dense_reference_source_checkpoint_path"] = selected_checkpoint_path
    plan["dense_reference_source_checkpoint_root_dir"] = resolved_source.get(
        "source_checkpoint_root_dir"
    )
    plan["dense_reference_source_checkpoint_step"] = int(selected_step or 0)
    plan["dense_reference_checkpoint_path_argument"] = resolved_source.get(
        "checkpoint_path_argument"
    )
    plan["dense_reference_load_step_argument"] = resolved_source.get("load_step_argument")
    plan["dense_reference_source_checkpoint_exists"] = bool(
        resolved_source.get("source_checkpoint_path_exists")
    )
    plan["dense_reference_checkpoint_root_valid"] = bool(
        resolved_source.get("checkpoint_root_valid")
    )
    plan["dense_reference_load_step_resolved"] = bool(
        resolved_source.get("load_step_resolved")
    )
    plan["dense_reference_planning_error"] = planning_error
    plan["dense_reference_planning_warning"] = resolved_source.get("planning_warning")
    plan["dense_reference_source_is_mock"] = bool(resolved_source.get("source_is_mock"))
    plan["dense_reference_pbrs_config"] = deepcopy(selected_candidate)
    plan["dense_reference_target_t_max"] = target_t_max_value
    return plan


def build_adaptive_mainline_source(
    *,
    selected_candidate: Dict[str, Any],
    selected_checkpoint_path: str,
    selected_checkpoint_step: int,
) -> Dict[str, Any]:
    resolved_source = _resolve_checkpoint_resume_source(
        checkpoint_path=selected_checkpoint_path,
        checkpoint_step=selected_checkpoint_step,
    )
    return {
        "adaptive_mainline_source_candidate_id": selected_candidate["candidate_id"],
        "adaptive_mainline_source_checkpoint_path": selected_checkpoint_path,
        "adaptive_mainline_source_checkpoint_step": int(selected_checkpoint_step),
        "adaptive_mainline_source_checkpoint_root_dir": resolved_source.get(
            "source_checkpoint_root_dir"
        ),
        "adaptive_mainline_checkpoint_path_argument": resolved_source.get(
            "checkpoint_path_argument"
        ),
        "adaptive_mainline_load_step_argument": resolved_source.get("load_step_argument"),
        "adaptive_mainline_load_step_resolved": bool(
            resolved_source.get("load_step_resolved")
        ),
        "adaptive_mainline_initial_pbrs_config": deepcopy(selected_candidate),
    }


def build_stage3_branch_plan(
    *,
    workflow_id: str,
    checkpoint_plan: Dict[str, Any],
    candidate: Dict[str, Any],
    env_key: str,
    train_config: str,
    seed: int,
    time_limit: int,
    local_results_path: str,
    source_checkpoint_path: str,
    source_checkpoint_step: int,
    branch_budget_steps: int = 300000,
) -> Dict[str, Any]:
    target_t_max = int(source_checkpoint_step) + int(branch_budget_steps)
    resolved_source = _resolve_checkpoint_resume_source(
        checkpoint_path=source_checkpoint_path,
        checkpoint_step=source_checkpoint_step,
    )
    run_id = f"{workflow_id}_{checkpoint_plan['name']}_{candidate['candidate_id']}"
    plan = build_fixed_reward_baseline_plan(
        _baseline_args(
            run_id=run_id,
            env_key=env_key,
            train_config=train_config,
            seed=seed,
            time_limit=time_limit,
            t_max=target_t_max,
            use_cuda=False,
            local_results_path=local_results_path,
            experiment_family="dual_llm_stage3_branch",
            budget_profile="mock",
            experiment_tag=workflow_id,
            pbrs_config=candidate,
            checkpoint_path=str(resolved_source.get("checkpoint_path_argument") or ""),
            load_step=int(resolved_source.get("load_step_argument") or 0),
        )
    )
    plan["source_checkpoint_path"] = source_checkpoint_path
    plan["source_checkpoint_root_dir"] = resolved_source.get("source_checkpoint_root_dir")
    plan["source_checkpoint_step"] = int(source_checkpoint_step)
    plan["checkpoint_path_argument"] = resolved_source.get("checkpoint_path_argument")
    plan["load_step_argument"] = resolved_source.get("load_step_argument")
    plan["branch_budget_steps"] = int(branch_budget_steps)
    plan["endpoint_checkpoint_step"] = target_t_max
    plan["endpoint_checkpoint_path"] = f"{local_results_path}/models/{run_id}/{target_t_max}"
    return plan


def build_stage3_two_round_mock_decision(
    *,
    workflow_id: str,
    checkpoint_plan: Dict[str, Any],
    env_key: str,
    train_config: str,
    seed: int,
    time_limit: int,
    local_results_path: str,
    current_config: Dict[str, Any],
    source_checkpoint_path: str,
    source_checkpoint_step: int,
    use_winner_branch_promotion: bool,
    seven_branch_mode: bool = False,
) -> Dict[str, Any]:
    round1_generated = [
        {
            "candidate_id": "beta_down",
            "beta": max(0.0, round(float(current_config["beta"]) - 0.2, 2)),
            "wc": float(current_config["wc"]),
            "wp": round(1.0 - float(current_config["wc"]), 10),
            "candidate_type": "beta_down",
        },
        {
            "candidate_id": "wc_down_wp_up",
            "beta": float(current_config["beta"]),
            "wc": max(0.0, round(float(current_config["wc"]) - 0.2, 2)),
            "wp": round(1.0 - max(0.0, round(float(current_config["wc"]) - 0.2, 2)), 10),
            "candidate_type": "wc_down_wp_up",
        },
        {
            "candidate_id": "reference_like_r1",
            "beta": float(current_config["beta"]),
            "wc": float(current_config["wc"]),
            "wp": float(current_config["wp"]),
            "candidate_type": "reference_like",
        },
    ]
    round2_generated = [
        {
            "candidate_id": "recovery",
            "beta": max(0.0, round(float(current_config["beta"]) - 0.1, 2)),
            "wc": max(0.0, round(float(current_config["wc"]) - 0.1, 2)),
            "wp": round(1.0 - max(0.0, round(float(current_config["wc"]) - 0.1, 2)), 10),
            "candidate_type": "recovery",
        },
        {
            "candidate_id": "beta_up",
            "beta": min(1.0, round(float(current_config["beta"]) + 0.2, 2)),
            "wc": float(current_config["wc"]),
            "wp": float(current_config["wp"]),
            "candidate_type": "beta_up",
        },
        {
            "candidate_id": "reference_like_r2",
            "beta": float(current_config["beta"]),
            "wc": float(current_config["wc"]),
            "wp": float(current_config["wp"]),
            "candidate_type": "reference_like",
        },
    ]
    no_change = {
        "candidate_id": "no_change",
        "beta": float(current_config["beta"]),
        "wc": float(current_config["wc"]),
        "wp": float(current_config["wp"]),
        "candidate_type": "no_change_control",
        "is_no_change": True,
    }
    round1_candidates = [deepcopy(no_change)] + round1_generated[: (3 if seven_branch_mode else 2)]
    round2_candidates = round2_generated[:3]
    for candidate in round1_candidates + round2_candidates:
        candidate.setdefault("is_no_change", False)
        candidate["plan"] = build_stage3_branch_plan(
            workflow_id=workflow_id,
            checkpoint_plan=checkpoint_plan,
            candidate=candidate,
            env_key=env_key,
            train_config=train_config,
            seed=seed,
            time_limit=time_limit,
            local_results_path=local_results_path,
            source_checkpoint_path=source_checkpoint_path,
            source_checkpoint_step=source_checkpoint_step,
            branch_budget_steps=300000,
        )
        candidate["mock_metrics"] = {
            "best_test_sparse_return_mean": 0.42 if candidate["candidate_id"] == "recovery" else 0.31,
            "last_test_sparse_return_mean": 0.39 if candidate["candidate_id"] == "recovery" else 0.3,
            "auc_test_sparse_return_mean": 0.34 if candidate["candidate_id"] == "recovery" else 0.27,
            "slope_test_sparse_return_mean": 0.08 if candidate["candidate_id"] == "recovery" else 0.02,
        }
    all_candidates = round1_candidates + round2_candidates
    ranked = sorted(
        all_candidates,
        key=lambda item: (
            item["mock_metrics"]["best_test_sparse_return_mean"],
            item["mock_metrics"]["last_test_sparse_return_mean"],
            item["mock_metrics"]["auc_test_sparse_return_mean"],
        ),
        reverse=True,
    )
    winner = ranked[0]
    return {
        "decision_id": f"{workflow_id}_{checkpoint_plan['name']}",
        "checkpoint": deepcopy(checkpoint_plan),
        "stage3_branch_rounds": 2,
        "stage3_candidates_per_round": 3,
        "stage3_include_no_change_control": True,
        "stage3_no_change_counts_toward_round_budget": True,
        "stage3_candidate_review_enabled": True,
        "stage3_candidate_repair_max_rounds": 1,
        "rounds": [
            {"round_id": 1, "candidates": round1_candidates},
            {"round_id": 2, "candidates": round2_candidates},
        ],
        "llm_recommendation": {
            "decision": "update_to_candidate",
            "recommended_candidate_id": winner["candidate_id"],
        },
        "deterministic_winner": winner["candidate_id"],
        "final_selected_winner": winner["candidate_id"],
        "decision_source": "llm_plus_gate",
        "no_change_gate_overrode_llm": False,
        "selected_winner_branch_run_id": winner["plan"]["run_id"],
        "selected_winner_endpoint_checkpoint_path": winner["plan"]["endpoint_checkpoint_path"],
        "selected_winner_endpoint_checkpoint_step": winner["plan"]["endpoint_checkpoint_step"],
        "promoted_to_mainline": bool(use_winner_branch_promotion),
        "ranking": [
            {
                "candidate_id": item["candidate_id"],
                "best_test_sparse_return_mean": item["mock_metrics"]["best_test_sparse_return_mean"],
                "last_test_sparse_return_mean": item["mock_metrics"]["last_test_sparse_return_mean"],
                "auc_test_sparse_return_mean": item["mock_metrics"]["auc_test_sparse_return_mean"],
            }
            for item in ranked
        ],
    }
