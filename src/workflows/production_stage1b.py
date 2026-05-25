from __future__ import annotations

import argparse
from copy import deepcopy
import json.decoder
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.run_fixed_reward_baseline import build_fixed_reward_baseline_plan
from rewarding.lbf_pbrs_v2 import legacy_lbf_candidate_to_v2
from rewarding.lbf_pbrs_v2 import normalize_lbf_pbrs_v2_config
from rewarding.rware_pbrs_v2 import normalize_rware_pbrs_v2_config
from workflows.clients.llm_routing import llm_routing_artifact_fields
from workflows.clients.openai_backend import OpenAIChatBackend
from workflows.dual_llm_workflow_support import build_adaptive_mainline_source
from workflows.dual_llm_workflow_support import build_dense_reference_continuation_plan
from workflows.dual_llm_workflow_support import build_llm_call_record
from workflows.dual_llm_workflow_support import build_stage1b_candidate_plan
from workflows.dual_llm_workflow_support import llm_cache_key_suffix
from workflows.diagnostics import resolve_run_reference_from_run_dir
from workflows.llm_experiment_memory import load_or_init_memory
from workflows.llm_experiment_memory import save_memory
from workflows.llm_experiment_memory import update_after_critic_sparse_diagnosis
from workflows.llm_experiment_memory import update_after_dense_reference_launch
from workflows.llm_experiment_memory import update_after_initial_candidate_generation
from workflows.llm_experiment_memory import update_after_initial_candidate_results
from workflows.llm_experiment_memory import update_after_initial_final_selection
from workflows.llm_experiment_memory import update_after_initial_round_analysis
from workflows.llm_api_ledger import extract_request_id
from workflows.llm_api_ledger import stable_payload_hash
from workflows.llm_api_ledger import write_ledger_event
from workflows.phase1_method import build_run_summary_from_run_dir
from workflows.phase1_method import _extract_initial_dense_config
from workflows.phase1_method import select_dense_reference_with_llm
from workflows.policy_guidance import build_fallback_integrated_guidance_card
from workflows.policy_guidance import append_policy_guidance_call_record
from workflows.policy_guidance import append_stage1b_policy_guided_candidate_generation
from workflows.policy_guidance import build_policy_guided_stage1b_candidate_batch
from workflows.policy_guidance import maybe_build_stage1b_generator_payload_with_policy_guidance
from workflows.policy_guidance import maybe_build_stage1b_integrated_guidance
from workflows.train_launcher import EPyMARLTrainLauncher


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ROOT = Path(__file__).resolve().parents[2]
CRITIC_SPARSE_PROMPT = PROMPTS_DIR / "critic_initial_sparse_diagnosis_prompt.md"
GENERATOR_R1_PROMPT = PROMPTS_DIR / "generator_initial_dense_candidates_prompt.md"
POLICY_GUIDED_GENERATOR_R1_PROMPT = (
    Path(__file__).resolve().parent
    / "prompts"
    / "policy_guidance"
    / "policy_guided_stage1b_generator_prompt.md"
)
POLICY_GUIDED_GENERATOR_R2_PROMPT = (
    Path(__file__).resolve().parent
    / "prompts"
    / "policy_guidance"
    / "policy_guided_stage1b_revision_prompt.md"
)
CRITIC_R1_PROMPT = PROMPTS_DIR / "critic_initial_round_result_analysis_prompt.md"
GENERATOR_R2_PROMPT = PROMPTS_DIR / "generator_initial_dense_revision_prompt.md"
CRITIC_FINAL_PROMPT = PROMPTS_DIR / "critic_initial_final_selection_prompt.md"

ALLOWED_CANDIDATE_TYPES = {
    "conservative",
    "balanced",
    "exploration_boost",
    "progress_shift",
    "collection_or_coordination_focus",
    "recovery",
    "reference_like",
}


class Stage1bRecoverableFallbackError(Exception):
    """A recoverable Stage 1b runtime problem that may use the configured fallback path."""


class Stage1bImplementationError(Exception):
    """An internal Stage 1b implementation bug that must not be converted into fallback."""


def is_dual_llm_stage1b_enabled(args: argparse.Namespace) -> bool:
    phase1_method = getattr(args, "phase1_method", None) or {}
    return bool(
        getattr(args, "use_dual_llm_stage1b_initial_search", None)
        if getattr(args, "use_dual_llm_stage1b_initial_search", None) is not None
        else phase1_method.get("use_dual_llm_stage1b_initial_search")
    )


def initial_dense_budget_steps(args: argparse.Namespace) -> int:
    phase1_method = getattr(args, "phase1_method", None) or {}
    initial_search = phase1_method.get("initial_dense_search") or {}
    return int(
        getattr(args, "initial_dense_candidate_budget_steps", None)
        or initial_search.get("candidate_budget_steps")
        or 800000
    )


def initial_dense_target_checkpoint_step(args: argparse.Namespace) -> int:
    phase1_method = getattr(args, "phase1_method", None) or {}
    initial_search = phase1_method.get("initial_dense_search") or {}
    return int(
        getattr(args, "initial_dense_candidate_target_checkpoint_step", None)
        or initial_search.get("candidate_target_checkpoint_step")
        or getattr(args, "stage1b_selected_endpoint_target_step", None)
        or initial_search.get("selected_endpoint_target_step")
        or 800000
    )


def stage1b_endpoint_tolerance_steps(args: argparse.Namespace) -> int:
    phase1_method = getattr(args, "phase1_method", None) or {}
    initial_search = phase1_method.get("initial_dense_search") or {}
    return int(
        getattr(args, "stage1b_endpoint_tolerance_steps", None)
        or initial_search.get("endpoint_tolerance_steps")
        or 50000
    )


def stage1b_endpoint_selection_policy(args: argparse.Namespace) -> str:
    phase1_method = getattr(args, "phase1_method", None) or {}
    initial_search = phase1_method.get("initial_dense_search") or {}
    return str(
        getattr(args, "stage1b_endpoint_selection_policy", None)
        or initial_search.get("endpoint_selection_policy")
        or "prefer_ge_then_le_nearest"
    )


def initial_dense_candidates_per_round(args: argparse.Namespace) -> int:
    phase1_method = getattr(args, "phase1_method", None) or {}
    initial_search = phase1_method.get("initial_dense_search") or {}
    return int(
        getattr(args, "initial_dense_candidates_per_round", None)
        or initial_search.get("candidates_per_round")
        or 3
    )


def dense_reference_target_t_max(args: argparse.Namespace) -> int:
    phase1_method = getattr(args, "phase1_method", None) or {}
    explicit_target = phase1_method.get("dense_reference_target_t_max")
    if explicit_target is not None:
        return int(explicit_target)
    if str(getattr(args, "train_config", "") or "") == "mappo":
        return min(int(getattr(args, "t_max", 2000000) or 2000000), 2000000)
    return int(getattr(args, "t_max", 2050000) or 2050000)


def _policy_guidance_setting(
    args: argparse.Namespace,
    key: str,
    default: Any,
) -> Any:
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


def _stage1_sparse_scalar_metrics(sparse_run_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "best_test_sparse_return_mean": _metric_value(
            sparse_run_summary, "test_sparse_return_mean", "best"
        ),
        "final_test_sparse_return_mean": _metric_value(
            sparse_run_summary, "test_sparse_return_mean", "final"
        ),
        "train_return_mean": _metric_value(
            sparse_run_summary, "train_return_mean", "final"
        ),
        "sample_efficiency_status": (
            "available"
            if (sparse_run_summary.get("metric_summary") or {})
            else "summary_missing"
        ),
        "performance_issue": "stage1 sparse baseline summary for stage1b candidate generation",
    }


def _policy_guidance_search_roots(
    *,
    workflow_dir: Path,
    local_results_path: str,
) -> List[Path]:
    return [
        workflow_dir,
        ROOT / local_results_path / "policy_guidance",
        ROOT / "results" / "policy_guidance",
    ]


def _write_policy_guidance_card(
    *,
    workflow_dir: Path,
    card: Dict[str, Any] | None,
) -> str | None:
    if not card:
        return None
    path = workflow_dir / "policy_guidance" / "stage1_after_sparse_baseline_integrated_guidance_card.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(card, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(path)


def default_dual_llm_stage1b_payload(
    *,
    workflow_id: str,
    workflow_dir: Path,
    early_budget_steps: int,
) -> Dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "llm_routing": {},
        "stage1b_mode": "dual_llm_initial_dense_search_v1",
        "stage1b_overall_status": "running",
        "critic_sparse_diagnosis_status": "pending",
        "critic_sparse_diagnosis_cache_path": None,
        "critic_sparse_diagnosis": {},
        "round1_generator_status": "pending",
        "round1_candidates": [],
        "round1_run_ids": [],
        "round1_results": [],
        "round1_endpoint_checkpoints": [],
        "round1_critic_analysis_status": "pending",
        "round1_critic_analysis_cache_path": None,
        "round1_critic_analysis": {},
        "round2_generator_status": "pending",
        "round2_candidates": [],
        "round2_run_ids": [],
        "round2_results": [],
        "round2_endpoint_checkpoints": [],
        "round2_critic_analysis_status": "pending",
        "round2_critic_analysis_cache_path": None,
        "round2_critic_analysis": {},
        "final_critic_selection_status": "pending",
        "final_critic_selection_cache_path": None,
        "final_critic_selection": {},
        "selected_candidate_id": None,
        "selected_initial_dense_config": {},
        "selected_endpoint_checkpoint_path": None,
        "selected_endpoint_checkpoint_step": None,
        "selected_checkpoint_root_dir": None,
        "selected_endpoint_checkpoint_path_argument": None,
        "selected_load_step": None,
        "selected_source_run_id": None,
        "selected_source_workflow_role": None,
        "selected_endpoint_distance_from_target": None,
        "selected_endpoint_warning": None,
        "stage1b_candidate_training_t_max": int(early_budget_steps),
        "stage1b_target_endpoint_step": None,
        "stage1b_endpoint_tolerance_steps": None,
        "stage1b_endpoint_selection_policy": None,
        "selection_reason": None,
        "supported_reward_hypotheses": [],
        "rejected_reward_hypotheses": [],
        "search_priors_for_stage2_and_stage3": {},
        "dense_reference_source_candidate_id": None,
        "dense_reference_source_checkpoint_path": None,
        "dense_reference_source_checkpoint_root_dir": None,
        "dense_reference_source_checkpoint_step": None,
        "dense_reference_checkpoint_path_argument": None,
        "dense_reference_load_step_argument": None,
        "dense_reference_pbrs_config": {},
        "dense_reference_target_t_max": None,
        "dense_reference_continuation_run_id": None,
        "dense_reference_run_id": None,
        "dense_reference_run": {},
        "dense_reference_run_dir": None,
        "adaptive_mainline_source_candidate_id": None,
        "adaptive_mainline_source_checkpoint_path": None,
        "adaptive_mainline_source_checkpoint_step": None,
        "adaptive_mainline_source_checkpoint_root_dir": None,
        "adaptive_mainline_checkpoint_path_argument": None,
        "adaptive_mainline_load_step_argument": None,
        "adaptive_mainline_load_step_resolved": False,
        "adaptive_mainline_initial_pbrs_config": {},
        "adaptive_mainline_source": {},
        "stage1b_fallback_used": False,
        "fallback_reason": None,
        "fallback_config": {},
        "fallback_source": None,
        "fallback_run_id": None,
        "fallback_candidates_launched": False,
        "dense_reference_status": "pending",
        "dense_reference_valid": False,
        "dense_reference_metrics_available": False,
        "dense_reference_failure_reason": None,
        "dense_reference_failure_excerpt": None,
        "dense_reference_checkpoint_root_valid": False,
        "dense_reference_load_step_resolved": False,
        "dense_reference_source_checkpoint_exists": False,
        "dense_reference_run_json_status": None,
        "final_vs_dense_reference_valid": False,
        "final_vs_sparse_valid": False,
        "dense_reference_selection": {},
        "policy_guidance_enabled": False,
        "policy_guidance_used": False,
        "policy_guidance_intervention_point": "stage1_after_sparse_baseline",
        "policy_guidance_source": "reward_only_fallback",
        "policy_guidance_failure_reason": None,
        "policy_guidance_fallback_to_reward_only": True,
        "policy_guidance_stage1_behavior_summary_path": None,
        "policy_guidance_stage1_behavior_summary_abs_path": None,
        "policy_guidance_stage1_behavior_summary_exists": False,
        "policy_guidance_stage1_behavior_summary_workflow_local": False,
        "policy_guidance_stage1_behavior_summary_reused_from_prior_artifact": False,
        "policy_guidance_stage1_behavior_summary_workflow_id": None,
        "policy_guidance_evidence_keys": [],
        "integrated_guidance_card_path": None,
        "integrated_guidance_card": {},
        "llm_memory_path": str(workflow_dir / "llm_experiment_memory.json"),
        "early_budget_steps": int(early_budget_steps),
        "errors": [],
    }


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _decode_json_object_at(text: str, start_index: int) -> tuple[Dict[str, Any], int]:
    decoder = json.JSONDecoder()
    payload, end_index = decoder.raw_decode(text, start_index)
    if not isinstance(payload, dict):
        raise ValueError("expected top-level JSON object")
    return payload, end_index


def _extract_json_dict(text: str) -> Dict[str, Any]:
    stripped = _strip_markdown_fence(text)
    if not stripped:
        raise ValueError("response does not contain a JSON object")
    decoder = json.JSONDecoder()
    candidate_objects: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(stripped):
        brace_index = stripped.find("{", idx)
        if brace_index == -1:
            break
        try:
            payload, end_index = _decode_json_object_at(stripped, brace_index)
        except (ValueError, json.JSONDecodeError, json.decoder.JSONDecodeError):
            idx = brace_index + 1
            continue
        candidate_objects.append(payload)
        if (
            "selected_candidate_id" in payload
            or "selected_initial_dense_config" in payload
            or "initial_config" in payload
        ):
            return payload
        idx = max(end_index, brace_index + 1)
    if candidate_objects:
        return candidate_objects[0]
    raise ValueError("response does not contain a JSON object")


def _normalize_selected_initial_dense_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    selected_config = deepcopy(config)
    if str(selected_config.get("pbrs_version") or "") == "lbf_pbrs_v2":
        return normalize_lbf_pbrs_v2_config(selected_config)
    if str(selected_config.get("pbrs_version") or "") == "rware_pbrs_v2":
        return normalize_rware_pbrs_v2_config(selected_config)
    if selected_config.get("beta") is None and selected_config.get("pbrs_beta") is not None:
        selected_config["beta"] = selected_config.get("pbrs_beta")
    if selected_config.get("wc") is None and selected_config.get("pbrs_wc") is not None:
        selected_config["wc"] = selected_config.get("pbrs_wc")
    if selected_config.get("wp") is None and selected_config.get("pbrs_wp") is not None:
        selected_config["wp"] = selected_config.get("pbrs_wp")
    return selected_config


def _stage1b_env_family(args: argparse.Namespace) -> str:
    env_text = str(getattr(args, "env_key", "") or "").strip().lower()
    version_text = str(getattr(args, "pbrs_version", "") or "").strip().lower()
    if "rware" in env_text or version_text == "rware_pbrs_v2":
        return "rware"
    if "lbforaging" in env_text or env_text.startswith("lbf") or version_text == "lbf_pbrs_v2":
        return "lbf"
    return "unknown"


def _stage1b_policy_guidance_search_space(args: argparse.Namespace) -> Dict[str, Any]:
    if _stage1b_env_family(args) == "rware":
        return {
            "parameters": ["pbrs_version", "mode", "beta", "active_terms", "weights"],
            "constraints": {
                "beta": [0.0, 1.0],
                "pbrs_version": ["rware_pbrs_v2"],
                "supported_modes": [
                    "requested_shelf_acquisition",
                    "balanced_delivery_progress",
                    "carrying_to_goal",
                    "traffic_conservative",
                    "late_stability",
                ],
                "active_terms_subset_of": ["shelf", "pickup", "goal", "deliv", "traffic", "stab"],
                "weights_sum_to_one": True,
                "requires_evidence_keys_used": True,
            },
        }
    return {
        "parameters": ["pbrs_version", "mode", "beta", "active_terms", "weights"],
        "constraints": {
            "beta": [0.0, 1.0],
            "pbrs_version": ["lbf_pbrs_v2"],
            "supported_modes": [
                "balanced_collection_ready",
                "coverage_ready_balance",
                "approach_collection_push",
                "allocation_stability_support",
            ],
            "active_terms_subset_of": ["col", "app", "cov", "ready", "alloc", "stab"],
            "weights_sum_to_one": True,
            "requires_evidence_keys_used": True,
        },
    }


def _load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def _llm_json_call(
    *,
    workflow_dir: Path,
    workflow_id: str,
    profile: str,
    env_key: str,
    algorithm: str,
    cache_name: str,
    agent_role: str,
    agent_name: str,
    call_type: str,
    stage: str,
    reason: str,
    prompt_path: Path,
    input_payload: Dict[str, Any],
    system_prompt: str,
    use_real_llm: bool,
    api_key_env: str,
    base_url: str,
    model: str,
    temperature: float,
    llm_timeout: float,
    llm_max_retries: int,
    llm_retry_backoff: float,
    fallback_response: Dict[str, Any],
    llm_routing: Dict[str, Any] | None = None,
    checkpoint: str | None = None,
    round_id: int | None = None,
    artifact_path: str | None = None,
) -> Dict[str, Any]:
    prompt_version = "v1"
    payload_hash = stable_payload_hash(input_payload)
    cache_path = (
        workflow_dir
        / "llm_cache"
        / f"{cache_name}__{llm_cache_key_suffix(model=model, prompt_version=prompt_version)}.json"
    )
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["status"] = "cache_hit"
        payload["ledger_event"] = write_ledger_event(
            workflow_dir=workflow_dir,
            workflow_id=workflow_id,
            profile=profile,
            env_key=env_key,
            algorithm=algorithm,
            stage=stage,
            call_type="cache_hit",
            reason=reason,
            checkpoint=checkpoint,
            round_id=round_id,
            model=payload.get("model") or model,
            provider=(llm_routing or {}).get("provider"),
            base_url=(llm_routing or {}).get("base_url") or base_url,
            cache_key=str(cache_path),
            cache_hit=True,
            payload=input_payload,
            payload_hash=payload_hash,
            usage=payload.get("usage"),
            request_id=extract_request_id(payload),
            success=True,
            repair_used=bool(payload.get("repair_retry_used", False)),
            repair_retry_count=0,
            fallback_used=bool(payload.get("validation_errors")),
            fallback_reason="cache_hit",
            artifact_path=artifact_path,
            caller_file=__file__,
            caller_function="_llm_json_call",
            event_state="cache_hit",
            metadata={"agent_role": agent_role, "agent_name": agent_name, "prompt_path": str(prompt_path)},
        )
        return payload
    if not use_real_llm:
        payload = build_llm_call_record(
            workflow_dir=workflow_dir,
            cache_name=cache_name,
            agent_role=agent_role,
            agent_name=agent_name,
            model=model,
            prompt_path=prompt_path,
            input_payload=input_payload,
            parsed_response=fallback_response,
            llm_routing=llm_routing,
        )
        payload["ledger_event"] = write_ledger_event(
            workflow_dir=workflow_dir,
            workflow_id=workflow_id,
            profile=profile,
            env_key=env_key,
            algorithm=algorithm,
            stage=stage,
            call_type="fallback",
            reason=reason,
            checkpoint=checkpoint,
            round_id=round_id,
            model=model,
            provider=(llm_routing or {}).get("provider"),
            base_url=(llm_routing or {}).get("base_url") or base_url,
            cache_key=str(cache_path),
            cache_hit=False,
            payload=input_payload,
            payload_hash=payload_hash,
            usage=None,
            request_id=None,
            success=None,
            repair_used=False,
            repair_retry_count=0,
            fallback_used=True,
            fallback_reason="real_llm_disabled",
            artifact_path=artifact_path,
            caller_file=__file__,
            caller_function="_llm_json_call",
            event_state="fallback",
            metadata={"agent_role": agent_role, "agent_name": agent_name, "prompt_path": str(prompt_path)},
        )
        return payload

    prompt = "\n".join(
        [
            "Prompt specification:",
            _load_prompt(prompt_path),
            "",
            "Payload:",
            json.dumps(input_payload, indent=2, sort_keys=True, ensure_ascii=False),
            "",
            "Return strict JSON only.",
        ]
    )
    backend = OpenAIChatBackend(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=llm_timeout,
        max_retries=llm_max_retries,
        retry_backoff_seconds=llm_retry_backoff,
    )
    validation_errors: List[str] = []
    attempt_event = write_ledger_event(
        workflow_dir=workflow_dir,
        workflow_id=workflow_id,
        profile=profile,
        env_key=env_key,
        algorithm=algorithm,
        stage=stage,
        call_type=call_type,
        reason=reason,
        checkpoint=checkpoint,
        round_id=round_id,
        model=model,
        provider=(llm_routing or {}).get("provider"),
        base_url=(llm_routing or {}).get("base_url") or base_url,
        cache_key=str(cache_path),
        cache_hit=False,
        payload=input_payload,
        payload_hash=payload_hash,
        usage=None,
        request_id=None,
        success=None,
        repair_used=False,
        repair_retry_count=0,
        fallback_used=False,
        fallback_reason=None,
        artifact_path=artifact_path,
        caller_file=__file__,
        caller_function="_llm_json_call",
        event_state="attempt",
        metadata={"agent_role": agent_role, "agent_name": agent_name, "prompt_path": str(prompt_path)},
    )
    try:
        llm_result = backend.generate_text(
            system_prompt=system_prompt,
            user_prompt=prompt,
            metadata={
                "agent_role": agent_role,
                "agent_name": agent_name,
            },
        )
        parsed = _extract_json_dict(llm_result.get("text") or "")
    except Exception as exc:
        validation_errors.append(str(exc))
        parsed = deepcopy(fallback_response)
        llm_result = {
            "text": json.dumps(parsed, ensure_ascii=False),
            "raw_response": {"fallback_triggered": True, "error": str(exc)},
            "model": model,
            "usage": None,
            "request_id": None,
        }
    payload = {
        "agent_role": agent_role,
        "agent_name": agent_name,
        "model": str(model),
        "llm_routing": deepcopy(llm_routing or {}),
        "prompt_path": str(prompt_path),
        "prompt_version": prompt_version,
        "input_payload": deepcopy(input_payload),
        "raw_response": llm_result.get("raw_response"),
        "parsed_response": deepcopy(parsed),
        "validation_errors": validation_errors,
        "cache_path": str(cache_path),
        "status": "executed",
        "model": llm_result.get("model"),
        "usage": llm_result.get("usage"),
    }
    payload["ledger_event"] = write_ledger_event(
        workflow_dir=workflow_dir,
        workflow_id=workflow_id,
        profile=profile,
        env_key=env_key,
        algorithm=algorithm,
        stage=stage,
        call_type=call_type,
        reason=reason,
        checkpoint=checkpoint,
        round_id=round_id,
        model=llm_result.get("model") or model,
        provider=(llm_routing or {}).get("provider"),
        base_url=(llm_routing or {}).get("base_url") or base_url,
        cache_key=str(cache_path),
        cache_hit=False,
        payload=input_payload,
        payload_hash=payload_hash,
        usage=llm_result.get("usage"),
        request_id=extract_request_id(llm_result),
        success=not validation_errors,
        repair_used=False,
        repair_retry_count=0,
        fallback_used=bool(validation_errors),
        fallback_reason=validation_errors[-1] if validation_errors else None,
        artifact_path=artifact_path,
        caller_file=__file__,
        caller_function="_llm_json_call",
        error_type="Stage1bFallbackError" if validation_errors else None,
        event_state="success" if not validation_errors else "fallback",
        record_id=str(attempt_event.get("record_id") or ""),
        metadata={"agent_role": agent_role, "agent_name": agent_name, "prompt_path": str(prompt_path)},
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return payload


def _deterministic_sparse_diagnosis(*, args: argparse.Namespace, sparse_run_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "goal": "early_budget_selection",
        "observed_sparse_failures": [
            "slow sparse reward discovery",
            "unstable transition from exploration to reliable collection",
        ],
        "reward_design_needs": [
            "denser early shaping without permanently overwhelming sparse objective",
            "support for coordination and collection conversion",
        ],
        "initial_search_priors": {
            "candidate_budget_steps": initial_dense_budget_steps(args),
            "prefer_diverse_round1_candidates": True,
            "stability_preferred_over_single_spike": True,
        },
        "stage_hypotheses": [
            "moderate beta with balanced wc/wp should provide stable early support",
            "more exploratory settings may help discovery but can overshape later behavior",
        ],
        "sparse_summary_excerpt": {
            "run_dir": sparse_run_summary.get("run_dir"),
            "preferred_metrics": list((sparse_run_summary.get("metric_curves") or {}).keys())[:4],
        },
    }


def _candidate(
    candidate_id: str,
    beta: float,
    wc: float,
    candidate_type: str,
    hypothesis: str,
    effect: str,
    risk: str,
) -> Dict[str, Any]:
    wc_value = min(1.0, max(0.0, round(float(wc), 2)))
    return {
        "candidate_id": candidate_id,
        "beta": min(1.0, max(0.0, round(float(beta), 2))),
        "wc": wc_value,
        "wp": round(1.0 - wc_value, 10),
        "candidate_type": candidate_type if candidate_type in ALLOWED_CANDIDATE_TYPES else "balanced",
        "hypothesis": hypothesis,
        "expected_early_effect": effect,
        "risk": risk,
    }


def _deterministic_rware_candidate(
    candidate_id: str,
    mode: str,
    beta: float,
    weights: Dict[str, float],
    candidate_type: str,
    hypothesis: str,
    effect: str,
    risk: str,
) -> Dict[str, Any]:
    normalized = normalize_rware_pbrs_v2_config(
        {
            "pbrs_version": "rware_pbrs_v2",
            "mode": mode,
            "beta": beta,
            "weights": deepcopy(weights),
        }
    )
    return {
        "candidate_id": candidate_id,
        "candidate_type": candidate_type if candidate_type in ALLOWED_CANDIDATE_TYPES else "balanced",
        "pbrs_version": "rware_pbrs_v2",
        "mode": str(normalized.get("mode") or mode),
        "beta": float(normalized.get("beta", beta)),
        "weights": deepcopy(normalized.get("weights") or {}),
        "active_terms": list(normalized.get("active_terms") or []),
        "evidence_keys_used": [],
        "hypothesis": hypothesis,
        "expected_early_effect": effect,
        "risk": risk,
    }


def _deterministic_round1_candidates(count: int, *, env_family: str = "unknown") -> Dict[str, Any]:
    if env_family == "rware":
        base = [
            _deterministic_rware_candidate(
                "r1_balanced_delivery",
                "balanced_delivery_progress",
                0.30,
                {
                    "shelf": 0.25,
                    "pickup": 0.15,
                    "goal": 0.35,
                    "deliv": 0.15,
                    "traffic": 0.10,
                    "stab": 0.00,
                },
                "balanced",
                "Balanced delivery-progress shaping should improve early coordination without overshooting.",
                "steadier early sparse-return growth",
                "may be too conservative if exploration is the primary bottleneck",
            ),
            _deterministic_rware_candidate(
                "r1_traffic_conservative",
                "traffic_conservative",
                0.20,
                {
                    "shelf": 0.15,
                    "pickup": 0.10,
                    "goal": 0.25,
                    "deliv": 0.10,
                    "traffic": 0.35,
                    "stab": 0.05,
                },
                "recovery",
                "Traffic-aware shaping should reduce blocking and keep early warehouse motion stable.",
                "fewer blocked-forward stalls during early exploration",
                "can underweight delivery progress if traffic shaping dominates",
            ),
            _deterministic_rware_candidate(
                "r1_goal_progress",
                "carrying_to_goal",
                0.30,
                {
                    "shelf": 0.10,
                    "pickup": 0.15,
                    "goal": 0.50,
                    "deliv": 0.15,
                    "traffic": 0.10,
                    "stab": 0.00,
                },
                "goal_progress",
                "Goal-focused shaping should convert successful pickups into faster delivery completion.",
                "stronger early delivery completion signal",
                "can destabilize if agents have not yet learned clean carrying routes",
            ),
        ]
        return {
            "goal": "early_budget_selection",
            "candidates": base[: max(1, count)],
        }
    base = [
        _candidate(
            "r1_balanced",
            0.3,
            0.5,
            "balanced",
            "Balanced shaping should improve early coordination without overshooting.",
            "steadier early sparse-return growth",
            "may be too conservative if exploration is the primary bottleneck",
        ),
        _candidate(
            "r1_exploration_boost",
            0.5,
            0.3,
            "exploration_boost",
            "Higher beta with more wp may accelerate discovery and movement shaping.",
            "faster first successes",
            "can create noisy or unstable policy updates",
        ),
        _candidate(
            "r1_reference_like",
            0.3,
            0.6,
            "reference_like",
            "Reference-like shaping preserves a stable baseline while still being dense.",
            "stable but not overly aggressive improvement",
            "may leave performance headroom unused",
        ),
    ]
    return {
        "goal": "early_budget_selection",
        "candidates": base[: max(1, count)],
    }


def _score_result(result: Dict[str, Any]) -> tuple[float, float]:
    best_value = result.get("best_test_sparse_return_mean")
    final_value = result.get("final_test_sparse_return_mean")
    try:
        best = float(best_value)
    except (TypeError, ValueError):
        best = float("-inf")
    try:
        final = float(final_value)
    except (TypeError, ValueError):
        final = float("-inf")
    return (best, final)


def _deterministic_round1_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ranked = sorted(results, key=_score_result, reverse=True)
    best = ranked[0] if ranked else {}
    return {
        "goal": "early_budget_selection",
        "candidate_reviews": [
            {
                "candidate_id": item.get("candidate_id"),
                "run_status": item.get("run_status"),
                "supports_hypothesis": bool(item.get("endpoint_checkpoint_path")),
                "risk_note": item.get("invalid_reason") or item.get("run_status"),
            }
            for item in results
        ],
        "supported_hypotheses": [f"{best.get('candidate_id')} currently leads the early budget ranking."] if best else [],
        "rejected_or_uncertain_hypotheses": [
            item.get("candidate_id")
            for item in results
            if not item.get("endpoint_checkpoint_path")
        ],
        "risky_patterns": [
            item.get("candidate_id")
            for item in results
            if str(item.get("run_status") or "") not in {"COMPLETED", "running", "RUNNING"}
        ],
        "generator_revision_advice": {
            "preserve_reference_like": True,
            "center_next_round_near": best.get("candidate_id"),
        },
    }


def _deterministic_round2_candidates(
    results: List[Dict[str, Any]],
    count: int,
    *,
    env_family: str = "unknown",
) -> Dict[str, Any]:
    if env_family == "rware":
        ranked = sorted(results, key=_score_result, reverse=True)
        anchor = ranked[0] if ranked else {}
        anchor_config = normalize_rware_pbrs_v2_config(anchor if anchor else None)
        anchor_beta = float(anchor_config.get("beta", 0.30))
        anchor_mode = str(anchor_config.get("mode") or "balanced_delivery_progress")
        anchor_weights = deepcopy(anchor_config.get("weights") or {})
        beta_down_weights = deepcopy(anchor_weights)
        beta_down_weights["traffic"] = min(1.0, float(beta_down_weights.get("traffic", 0.0)) + 0.05)
        beta_down_weights["goal"] = max(0.0, float(beta_down_weights.get("goal", 0.0)) - 0.05)
        traffic_shift_weights = deepcopy(anchor_weights)
        traffic_shift_weights["traffic"] = min(1.0, float(traffic_shift_weights.get("traffic", 0.0)) + 0.10)
        traffic_shift_weights["shelf"] = max(0.0, float(traffic_shift_weights.get("shelf", 0.0)) - 0.05)
        traffic_shift_weights["pickup"] = max(0.0, float(traffic_shift_weights.get("pickup", 0.0)) - 0.05)
        candidates = [
            _deterministic_rware_candidate(
                "r2_reference_preserve",
                anchor_mode,
                anchor_beta,
                anchor_weights,
                "reference_like",
                "Retain the current best stable RWARE PBRS-v2 setting as a control-like revised candidate.",
                "checks whether round-1 leader is already good enough",
                "may duplicate performance ceiling",
            ),
            _deterministic_rware_candidate(
                "r2_beta_down",
                anchor_mode,
                max(0.0, anchor_beta - 0.10),
                beta_down_weights,
                "conservative",
                "A slightly lower beta may preserve sparse alignment after early initialization.",
                "better stability with smaller spikes",
                "may under-shape and slow learning",
            ),
            _deterministic_rware_candidate(
                "r2_traffic_shift",
                "traffic_conservative",
                anchor_beta,
                traffic_shift_weights,
                "progress_shift",
                "A traffic-aware revision may improve conversion of dense support into clean sparse deliveries.",
                "better avoidance of blocked-forward stalls during late early-budget training",
                "can over-penalize movement and reduce delivery pressure",
            ),
        ]
        return {
            "goal": "early_budget_selection",
            "candidates": candidates[: max(1, count)],
        }
    ranked = sorted(results, key=_score_result, reverse=True)
    anchor = ranked[0] if ranked else {"beta": 0.3, "wc": 0.5, "candidate_id": "anchor"}
    beta = float(anchor.get("beta") or 0.3)
    wc = float(anchor.get("wc") or 0.5)
    candidates = [
        _candidate(
            "r2_reference_preserve",
            beta,
            wc,
            "reference_like",
            "Retain the current best stable setting as a control-like revised candidate.",
            "checks whether round-1 leader is already good enough",
            "may duplicate performance ceiling",
        ),
        _candidate(
            "r2_beta_down",
            max(0.0, beta - 0.2),
            wc,
            "conservative",
            "A slightly lower beta may preserve sparse alignment after early initialization.",
            "better stability with smaller spikes",
            "may under-shape and slow learning",
        ),
        _candidate(
            "r2_wc_down",
            beta,
            max(0.0, wc - 0.2),
            "progress_shift",
            "Lower wc and higher wp may improve transition from shaping to sparse collection.",
            "better conversion of dense support into sparse success",
            "can destabilize if underweighted on coordination",
        ),
    ]
    return {
        "goal": "early_budget_selection",
        "candidates": candidates[: max(1, count)],
    }


def _sanitize_candidates(candidates: List[Dict[str, Any]], *, prefix: str) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for index, item in enumerate(candidates, start=1):
        if str(item.get("pbrs_version") or "") == "lbf_pbrs_v2":
            normalized = normalize_lbf_pbrs_v2_config(item)
            candidate_id = str(item.get("candidate_id") or f"{prefix}_{index}")
            candidate_type = str(item.get("candidate_type") or "balanced")
            signature = (
                str(normalized.get("mode") or ""),
                round(float(normalized.get("beta", 0.0)), 3),
                tuple(
                    (term, round(float((normalized.get("weights") or {}).get(term, 0.0)), 3))
                    for term in sorted((normalized.get("weights") or {}).keys())
                ),
            )
            if signature in seen:
                continue
            seen.add(signature)
            sanitized.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_type": candidate_type,
                    "pbrs_version": "lbf_pbrs_v2",
                    "mode": str(normalized.get("mode") or ""),
                    "beta": float(normalized.get("beta", 0.5)),
                    "wc": float(normalized.get("wc", 0.0)),
                    "wp": float(normalized.get("wp", 0.0)),
                    "weights": deepcopy(normalized.get("weights") or {}),
                    "active_terms": list(normalized.get("active_terms") or []),
                    "evidence_keys_used": list(item.get("evidence_keys_used") or []),
                    "hypothesis": str(item.get("hypothesis") or ""),
                    "expected_early_effect": str(item.get("expected_early_effect") or ""),
                    "risk": str(item.get("risk") or ""),
                    "rationale": str(item.get("rationale") or ""),
                }
            )
            continue
        if str(item.get("pbrs_version") or "") == "rware_pbrs_v2":
            normalized = normalize_rware_pbrs_v2_config(item)
            candidate_id = str(item.get("candidate_id") or f"{prefix}_{index}")
            candidate_type = str(item.get("candidate_type") or "balanced")
            signature = (
                str(normalized.get("mode") or ""),
                round(float(normalized.get("beta", 0.3)), 3),
                tuple(
                    (term, round(float((normalized.get("weights") or {}).get(term, 0.0)), 3))
                    for term in ("shelf", "pickup", "goal", "deliv", "traffic", "stab")
                ),
            )
            if signature in seen:
                continue
            seen.add(signature)
            sanitized.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_type": candidate_type,
                    "pbrs_version": "rware_pbrs_v2",
                    "mode": str(normalized.get("mode") or ""),
                    "beta": float(normalized.get("beta", 0.3)),
                    "weights": deepcopy(normalized.get("weights") or {}),
                    "active_terms": list(normalized.get("active_terms") or []),
                    "evidence_keys_used": list(item.get("evidence_keys_used") or []),
                    "hypothesis": str(item.get("hypothesis") or ""),
                    "expected_early_effect": str(item.get("expected_early_effect") or ""),
                    "risk": str(item.get("risk") or ""),
                    "rationale": str(item.get("rationale") or ""),
                }
            )
            continue
        beta = min(1.0, max(0.0, float(item.get("beta", 0.3))))
        wc = min(1.0, max(0.0, float(item.get("wc", 0.5))))
        signature = (round(beta, 3), round(wc, 3))
        if signature in seen:
            continue
        seen.add(signature)
        candidate_id = str(item.get("candidate_id") or f"{prefix}_{index}")
        candidate_type = str(item.get("candidate_type") or "balanced")
        sanitized.append(
            _candidate(
                candidate_id,
                beta,
                wc,
                candidate_type,
                str(item.get("hypothesis") or ""),
                str(item.get("expected_early_effect") or ""),
                str(item.get("risk") or ""),
            )
        )
    return sanitized


def _enforce_formal_v2_candidates(
    *,
    candidates: List[Dict[str, Any]],
    args: argparse.Namespace,
    stage_label: str,
) -> None:
    pbrs_version = str(getattr(args, "pbrs_version", "") or "")
    if pbrs_version not in {"lbf_pbrs_v2", "rware_pbrs_v2"}:
        return
    invalid = [
        str(item.get("candidate_id") or "<unknown>")
        for item in candidates
        if str(item.get("pbrs_version") or "") != pbrs_version
    ]
    if invalid:
        raise Stage1bImplementationError(
            f"{stage_label} produced legacy/non-v2 candidates under formal {pbrs_version}: {', '.join(invalid)}"
        )


def _maybe_attach_stage1b_policy_guidance(
    *,
    args: argparse.Namespace,
    workflow_dir: Path,
    payload: Dict[str, Any],
    memory: Dict[str, Any],
    sparse_run_summary: Dict[str, Any],
    local_results_path: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    enabled = bool(_policy_guidance_setting(args, "enable_policy_guidance", False))
    stage1b_use_guidance = bool(
        _policy_guidance_setting(
            args,
            "policy_guidance_stage1b_use_integrated_guidance",
            True,
        )
    )
    summary_required = bool(
        _policy_guidance_setting(
            args,
            "policy_guidance_stage1_behavior_summary_required",
            False,
        )
    )
    fallback_to_reward_only = bool(
        _policy_guidance_setting(
            args,
            "policy_guidance_stage1b_fallback_to_reward_only",
            True,
        )
    )
    explicit_summary_path = _policy_guidance_setting(
        args,
        "policy_guidance_stage1_behavior_summary_path",
        None,
    )
    allow_reuse_stage1_behavior_summary = bool(
        _policy_guidance_setting(
            args,
            "policy_guidance_stage1_behavior_summary_reuse_allowed",
            False,
        )
        or _policy_guidance_setting(args, "allow_reuse_stage1_behavior_summary", False)
    )
    require_workflow_local_stage1_summary = bool(
        _policy_guidance_setting(
            args,
            "policy_guidance_stage1_behavior_summary_must_be_fresh",
            True,
        )
    )
    hook_result = maybe_build_stage1b_integrated_guidance(
        task_metadata={
            "workflow_id": payload.get("workflow_id"),
            "env_key": getattr(args, "env_key", ""),
            "algorithm": getattr(args, "train_config", ""),
            "seed": getattr(args, "seed", 1),
        },
        sparse_baseline_metrics=_stage1_sparse_scalar_metrics(sparse_run_summary),
        policy_guidance_enabled=enabled and stage1b_use_guidance,
        stage1_behavior_summary_required=summary_required,
        stage1_behavior_summary_path=explicit_summary_path,
        workflow_dir=workflow_dir,
        search_roots=_policy_guidance_search_roots(
            workflow_dir=workflow_dir,
            local_results_path=local_results_path,
        ),
        fallback_to_reward_only=fallback_to_reward_only,
        target_step=initial_dense_target_checkpoint_step(args),
        mock_llm_mode=not bool(getattr(args, "use_real_llm", False)),
        use_real_llm=bool(getattr(args, "use_real_llm", False)),
        api_key_env=str(getattr(args, "api_key_env", "IUSEAPI_API_KEY")),
        base_url=str(getattr(args, "base_url", "https://www.iuseapi.com/v1")),
        model=str(getattr(args, "model", "gpt-5.2")),
        temperature=float(getattr(args, "temperature", 0.2)),
        llm_timeout=float(getattr(args, "llm_timeout", 60.0)),
        llm_max_retries=int(getattr(args, "llm_max_retries", 3)),
        llm_retry_backoff=float(getattr(args, "llm_retry_backoff", 5.0)),
        reuse_policy_guidance_cache=bool(
            _policy_guidance_setting(args, "reuse_policy_guidance_cache", False)
        ),
        policy_guidance_cache_root=_policy_guidance_setting(
            args,
            "policy_guidance_cache_root",
            None,
        ),
        policy_guidance_cache_lookup_mode=str(
            _policy_guidance_setting(
                args,
                "policy_guidance_cache_lookup_mode",
                "exact_or_latest_by_intervention",
            )
        ),
        allow_reuse_stage1_behavior_summary=allow_reuse_stage1_behavior_summary,
        require_workflow_local_stage1_behavior_summary=require_workflow_local_stage1_summary,
    )
    card = deepcopy(hook_result.get("integrated_guidance_card") or {})
    card_path = _write_policy_guidance_card(workflow_dir=workflow_dir, card=card)
    hook_result["integrated_guidance_card_path"] = card_path
    payload["policy_guidance_enabled"] = bool(enabled)
    hook_used = bool(hook_result.get("policy_guidance_used", False))
    payload["policy_guidance_used"] = hook_used
    payload["policy_guidance_intervention_point"] = "stage1_after_sparse_baseline"
    payload["policy_guidance_source"] = str(
        hook_result.get("policy_guidance_source") or "reward_only_fallback"
    )
    payload["policy_guidance_failure_reason"] = hook_result.get("policy_guidance_failure_reason")
    payload["policy_guidance_fallback_to_reward_only"] = not hook_used
    payload["policy_guidance_stage1_behavior_summary_path"] = hook_result.get(
        "stage1_behavior_summary_path"
    )
    payload["policy_guidance_stage1_behavior_summary_abs_path"] = hook_result.get(
        "stage1_behavior_summary_abs_path"
    )
    payload["policy_guidance_stage1_behavior_summary_exists"] = bool(
        hook_result.get("stage1_behavior_summary_exists", False)
    )
    payload["policy_guidance_stage1_behavior_summary_workflow_local"] = bool(
        hook_result.get("stage1_behavior_summary_workflow_local", False)
    )
    payload["policy_guidance_stage1_behavior_summary_reused_from_prior_artifact"] = bool(
        hook_result.get("stage1_behavior_summary_reused_from_prior_artifact", False)
    )
    payload["policy_guidance_stage1_behavior_summary_workflow_id"] = str(
        hook_result.get("stage1_behavior_summary_workflow_id") or ""
    )
    payload["policy_guidance_evidence_keys"] = list(
        hook_result.get("policy_guidance_evidence_keys") or []
    )
    payload["integrated_guidance_card_path"] = card_path
    payload["integrated_guidance_card"] = card

    append_policy_guidance_call_record(
        memory,
        {
            "guidance_id": "pg_stage1_after_sparse_baseline_production_stage1b",
            "intervention_point": "stage1_after_sparse_baseline",
            "input_behavior_summary_path": str(
                hook_result.get("stage1_behavior_summary_abs_path")
                or hook_result.get("stage1_behavior_summary_path")
                or ""
            ),
            "integrated_guidance_card": card
            if card
            else build_fallback_integrated_guidance_card(
                intervention_point="stage1_after_sparse_baseline",
                failure_reason=str(
                    hook_result.get("policy_guidance_failure_reason")
                    or "policy guidance unavailable"
                ),
                reward_diagnosis=hook_result.get("reward_diagnosis") or {},
            ),
            "used_by_critic": True,
            "used_by_generator": True,
            "fallback_used": not bool(hook_result.get("policy_guidance_used", False)),
            "failure_reason": str(hook_result.get("policy_guidance_failure_reason") or ""),
            "metadata": {
                "production_hook": True,
                "integrated_guidance_card_path": card_path or "",
                "evidence_keys": list(hook_result.get("policy_guidance_evidence_keys") or []),
            },
        },
    )
    return payload, memory, hook_result


def maybe_attach_stage1b_policy_guidance(
    *,
    args: argparse.Namespace,
    workflow_dir: Path,
    payload: Dict[str, Any],
    memory: Dict[str, Any],
    sparse_run_summary: Dict[str, Any],
    local_results_path: str = "results",
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return _maybe_attach_stage1b_policy_guidance(
        args=args,
        workflow_dir=workflow_dir,
        payload=payload,
        memory=memory,
        sparse_run_summary=sparse_run_summary,
        local_results_path=local_results_path,
    )


def _status_from_run_dir(run_dir: Path) -> str:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return "missing_run_json"
    try:
        payload = json.loads(run_json.read_text(encoding="utf-8"))
    except Exception:
        return "invalid_run_json"
    return str(payload.get("status") or "unknown")


def _metric_value(summary: Dict[str, Any], metric_name: str, key: str) -> Optional[float]:
    metric_summary = ((summary.get("metric_summary") or {}).get(metric_name) or {})
    value = metric_summary.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_stage1b_endpoint_step(
    *,
    available_steps: List[int],
    target_step: int,
) -> Optional[int]:
    if not available_steps:
        return None
    ge_steps = [int(step) for step in available_steps if int(step) >= int(target_step)]
    if ge_steps:
        return min(ge_steps, key=lambda step: abs(int(step) - int(target_step)))
    le_steps = [int(step) for step in available_steps if int(step) <= int(target_step)]
    if le_steps:
        return min(le_steps, key=lambda step: abs(int(step) - int(target_step)))
    return min([int(step) for step in available_steps], key=lambda step: abs(int(step) - int(target_step)))


def _candidate_result_from_run_reference(
    *,
    candidate: Dict[str, Any],
    run_id: str,
    run_reference: Dict[str, Any],
    target_endpoint_step: int,
    endpoint_tolerance_steps: int,
    endpoint_selection_policy: str,
) -> Dict[str, Any]:
    run_dir = Path(str(run_reference["run_dir"]))
    run_status = _status_from_run_dir(run_dir)
    resolved_run_reference = deepcopy(run_reference)
    summary: Dict[str, Any] = {}
    summary_error = None
    try:
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            summary = build_run_summary_from_run_dir(run_dir=run_dir)
    except Exception as exc:
        summary_error = str(exc)
    available_steps = [
        int(step)
        for step in list(resolved_run_reference.get("available_checkpoint_steps") or [])
        if str(step).isdigit()
    ]
    checkpoint_root_dir = resolved_run_reference.get("checkpoint_root_dir")
    if _terminal_run_status(run_status) and (not checkpoint_root_dir or not available_steps):
        try:
            recovered_reference = resolve_run_reference_from_run_dir(run_dir)
            resolved_run_reference.update(
                {
                    key: value
                    for key, value in recovered_reference.items()
                    if value not in (None, "", [])
                }
            )
            checkpoint_root_dir = resolved_run_reference.get("checkpoint_root_dir")
            available_steps = [
                int(step)
                for step in list(resolved_run_reference.get("available_checkpoint_steps") or [])
                if str(step).isdigit()
            ]
        except Exception as exc:
            if summary_error is None:
                summary_error = f"endpoint metadata recovery failed: {exc}"
    endpoint_step = _select_stage1b_endpoint_step(
        available_steps=available_steps,
        target_step=int(target_endpoint_step),
    )
    checkpoint_path = None
    invalid_reason = None
    endpoint_distance = None
    endpoint_warning = None
    if checkpoint_root_dir and endpoint_step is not None:
        candidate_path = Path(str(checkpoint_root_dir)) / str(int(endpoint_step))
        if candidate_path.exists():
            checkpoint_path = str(candidate_path)
            endpoint_distance = abs(int(endpoint_step) - int(target_endpoint_step))
            if endpoint_distance > int(endpoint_tolerance_steps):
                endpoint_warning = (
                    f"selected endpoint step {endpoint_step} is {endpoint_distance} away from target "
                    f"{int(target_endpoint_step)} under policy {endpoint_selection_policy}"
                )
        else:
            invalid_reason = f"endpoint checkpoint missing at {candidate_path}"
    else:
        if _terminal_run_status(run_status):
            invalid_reason = "endpoint checkpoint metadata missing"
    if summary_error and _terminal_run_status(run_status):
        invalid_reason = summary_error
    return {
        "candidate_id": candidate["candidate_id"],
        "beta": float(candidate["beta"]),
        "wc": float(candidate["wc"]) if candidate.get("wc") is not None else None,
        "wp": float(candidate["wp"]) if candidate.get("wp") is not None else None,
        "pbrs_config": deepcopy(
            normalize_lbf_pbrs_v2_config(candidate)
            if str(candidate.get("pbrs_version") or "") == "lbf_pbrs_v2"
            else normalize_rware_pbrs_v2_config(candidate)
            if str(candidate.get("pbrs_version") or "") == "rware_pbrs_v2"
            else candidate.get("pbrs_config")
            or {
                "beta": float(candidate["beta"]),
                "wc": float(candidate["wc"]),
                "wp": float(candidate["wp"]),
            }
        ),
        "run_id": run_id,
        "run_status": run_status,
        "run_dir": str(run_dir),
        "final_test_sparse_return_mean": _metric_value(summary, "test_sparse_return_mean", "last_value"),
        "best_test_sparse_return_mean": _metric_value(summary, "test_sparse_return_mean", "best_value"),
        "last_k_mean": None,
        "auc": None,
        "endpoint_checkpoint_path": checkpoint_path,
        "endpoint_checkpoint_step": int(endpoint_step) if endpoint_step is not None else None,
        "endpoint_target_step": int(target_endpoint_step),
        "candidate_training_t_max": int(candidate.get("candidate_training_t_max") or target_endpoint_step),
        "distance_from_target": endpoint_distance,
        "endpoint_selection_policy": str(endpoint_selection_policy),
        "endpoint_tolerance_steps": int(endpoint_tolerance_steps),
        "endpoint_warning": endpoint_warning,
        "available_checkpoint_steps": available_steps,
        "invalid_reason": invalid_reason,
        "checkpoint_root_dir": checkpoint_root_dir,
        "endpoint_checkpoint_path_argument": str(checkpoint_root_dir) if checkpoint_root_dir else None,
        "load_step": int(endpoint_step) if endpoint_step is not None else None,
        "source_run_id": run_id,
        "source_workflow_role": "stage1b_candidate",
        "metrics_summary": summary,
    }


def _terminal_run_status(status: str) -> bool:
    return status in {"COMPLETED", "FAILED", "INTERRUPTED", "KILLED"}


def _select_best_result(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [item for item in results if item.get("endpoint_checkpoint_path") and item.get("run_status") == "COMPLETED"]
    if not valid:
        return None
    ranked = sorted(valid, key=_score_result, reverse=True)
    return ranked[0]


def _build_final_selection_fallback(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected = _select_best_result(results)
    if selected is None:
        raise Stage1bRecoverableFallbackError(
            "no valid Stage 1b candidate has a completed run with endpoint checkpoint"
        )
    selected_config = deepcopy(selected.get("pbrs_config") or {})
    if str(selected_config.get("pbrs_version") or "") == "lbf_pbrs_v2":
        selected_config = normalize_lbf_pbrs_v2_config(selected_config)
    elif not selected_config:
        selected_config = {
            "beta": float(selected["beta"]),
            "wc": float(selected["wc"]),
            "wp": float(selected["wp"]),
        }
    return {
        "selected_candidate_id": selected["candidate_id"],
        "selected_initial_dense_config": selected_config,
        "selected_endpoint_checkpoint_path": selected["endpoint_checkpoint_path"],
        "selected_endpoint_checkpoint_step": selected["endpoint_checkpoint_step"],
        "selected_checkpoint_root_dir": selected.get("checkpoint_root_dir"),
        "selected_endpoint_checkpoint_path_argument": selected.get(
            "endpoint_checkpoint_path_argument"
        ),
        "selected_load_step": selected.get("load_step"),
        "selected_source_run_id": selected.get("source_run_id"),
        "selected_source_workflow_role": selected.get("source_workflow_role"),
        "selection_reason": (
            "Deterministic fallback selected the valid candidate with the strongest "
            "best_test_sparse_return_mean and final_test_sparse_return_mean under the early budget."
        ),
        "supported_reward_hypotheses": [
            f"{selected['candidate_id']} delivered the strongest stable early-budget result among valid candidates."
        ],
        "rejected_reward_hypotheses": [
            item["candidate_id"]
            for item in results
            if item["candidate_id"] != selected["candidate_id"]
        ],
        "search_priors_for_stage2_and_stage3": {
            "anchor_candidate_id": selected["candidate_id"],
            "anchor_beta": float(selected_config["beta"]),
            "anchor_wc": float(selected_config.get("wc", 0.0)),
            "anchor_mode": selected_config.get("mode"),
            "anchor_active_terms": list(selected_config.get("active_terms") or []),
            "anchor_weights": deepcopy(selected_config.get("weights") or {}),
            "fixed_checkpoint_strategy": True,
        },
    }


def _clamp_probability(name: str, value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage1bImplementationError(f"{name} could not be parsed as float: {value!r}") from exc
    return max(0.0, min(1.0, round(numeric, 10)))


def _candidate_result_lookup(candidate_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in candidate_results:
        candidate_id = str(item.get("candidate_id") or "").strip()
        if candidate_id:
            lookup[candidate_id] = item
    return lookup


def _normalize_stage1b_final_selection(
    *,
    final_selection: Dict[str, Any],
    candidate_results: List[Dict[str, Any]],
    manifest_context: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], List[str], List[str]]:
    warnings: List[str] = []
    validation_errors: List[str] = []
    candidate_lookup = _candidate_result_lookup(candidate_results)
    manifest_context = manifest_context or {}

    selected_candidate_id = str(
        final_selection.get("selected_candidate_id")
        or final_selection.get("recommended_candidate_id")
        or final_selection.get("winner_candidate_id")
        or ""
    ).strip()
    if not selected_candidate_id:
        validation_errors.append("final selection did not provide a candidate id")
    selected_result = candidate_lookup.get(selected_candidate_id)
    if not selected_result:
        validation_errors.append(
            f"selected candidate id {selected_candidate_id!r} was not found in candidate results"
        )
        raise Stage1bRecoverableFallbackError("; ".join(validation_errors))

    selected_config: Dict[str, Any] = {}
    if isinstance(final_selection.get("selected_initial_dense_config"), dict):
        selected_config = _normalize_selected_initial_dense_config_dict(
            final_selection["selected_initial_dense_config"]
        )
    elif isinstance(final_selection.get("initial_config"), dict):
        selected_config = _normalize_selected_initial_dense_config_dict(
            final_selection["initial_config"]
        )
    else:
        beta, wc, wp = _extract_initial_dense_config(final_selection)
        if beta is not None or wc is not None or wp is not None:
            selected_config = {"beta": beta, "wc": wc, "wp": wp}

    if not selected_config:
        pbrs_config = selected_result.get("pbrs_config")
        if isinstance(pbrs_config, dict):
            selected_config = deepcopy(pbrs_config)
        else:
            selected_config = {
                "beta": selected_result.get("beta"),
                "wc": selected_result.get("wc"),
                "wp": selected_result.get("wp"),
            }

    is_lbf_v2 = str(selected_config.get("pbrs_version") or "") == "lbf_pbrs_v2"
    is_rware_v2 = str(selected_config.get("pbrs_version") or "") == "rware_pbrs_v2"
    if is_lbf_v2:
        selected_config = normalize_lbf_pbrs_v2_config(selected_config)
        beta_value = float(selected_config["beta"])
        wc_value = float(selected_config["wc"])
        wp_value = float(selected_config["wp"])
    elif is_rware_v2:
        selected_config = normalize_rware_pbrs_v2_config(selected_config)
        beta_value = float(selected_config["beta"])
        wc_value = 0.0
        wp_value = 0.0
    else:
        beta = selected_config.get("beta")
        wc = selected_config.get("wc")
        wp = selected_config.get("wp")
        if beta is None or wc is None:
            raise Stage1bRecoverableFallbackError(
                "selected candidate did not resolve a valid beta/wc initial dense config"
            )
        beta_value = _clamp_probability("beta", beta)
        wc_value = _clamp_probability("wc", wc)
        derived_wp = round(1.0 - wc_value, 10)
        if wp is None:
            wp_value = derived_wp
        else:
            wp_value = _clamp_probability("wp", wp)
            if abs(wp_value - derived_wp) > 1e-9:
                warnings.append(
                    f"selected wp={wp_value} disagreed with derived 1-wc={derived_wp}; using derived value"
                )
                wp_value = derived_wp

    endpoint_checkpoint_path = str(
        selected_result.get("endpoint_checkpoint_path")
        or final_selection.get("selected_endpoint_checkpoint_path")
        or ""
    ).strip()
    endpoint_checkpoint_step = selected_result.get("endpoint_checkpoint_step")
    if endpoint_checkpoint_step is None:
        endpoint_checkpoint_step = final_selection.get("selected_endpoint_checkpoint_step")
    try:
        endpoint_checkpoint_step = int(endpoint_checkpoint_step)
    except (TypeError, ValueError) as exc:
        raise Stage1bRecoverableFallbackError(
            f"selected candidate {selected_candidate_id} does not have a valid endpoint checkpoint step"
        ) from exc
    if not endpoint_checkpoint_path:
        raise Stage1bRecoverableFallbackError(
            f"selected candidate {selected_candidate_id} does not have a valid endpoint checkpoint path"
        )

    final_endpoint_path = str(final_selection.get("selected_endpoint_checkpoint_path") or "").strip()
    if final_endpoint_path and final_endpoint_path != endpoint_checkpoint_path:
        warnings.append(
            "final selection endpoint path did not match selected candidate result; "
            f"using candidate result path {endpoint_checkpoint_path}"
        )

    normalized = {
        "selected_candidate_id": selected_candidate_id,
        "selected_initial_dense_config": (
            deepcopy(selected_config)
            if is_lbf_v2 or is_rware_v2
            else {
                "beta": beta_value,
                "wc": wc_value,
                "wp": wp_value,
            }
        ),
        "selected_endpoint_checkpoint_path": endpoint_checkpoint_path,
        "selected_endpoint_checkpoint_step": endpoint_checkpoint_step,
        "selected_checkpoint_root_dir": selected_result.get("checkpoint_root_dir"),
        "selected_endpoint_checkpoint_path_argument": selected_result.get(
            "endpoint_checkpoint_path_argument"
        ),
        "selected_load_step": selected_result.get("load_step"),
        "selected_source_run_id": selected_result.get("source_run_id"),
        "selected_source_workflow_role": selected_result.get("source_workflow_role"),
        "selection_reason": final_selection.get("selection_reason"),
        "supported_reward_hypotheses": list(final_selection.get("supported_reward_hypotheses") or []),
        "rejected_reward_hypotheses": list(final_selection.get("rejected_reward_hypotheses") or []),
        "search_priors_for_stage2_and_stage3": deepcopy(
            final_selection.get("search_priors_for_stage2_and_stage3") or {}
        ),
        "normalization_warnings": warnings,
        "validation_errors": validation_errors,
        "selected_result_candidate_id": selected_result.get("candidate_id"),
        "selected_result_run_id": selected_result.get("run_id"),
        "manifest_context": deepcopy(manifest_context),
    }
    return normalized, warnings, validation_errors


def _should_fallback_for_stage1b_exception(
    *,
    args: argparse.Namespace,
    exc: BaseException,
) -> bool:
    if not isinstance(exc, Stage1bRecoverableFallbackError):
        return False
    if not _fallback_enabled(args):
        return False
    phase1_method = getattr(args, "phase1_method", None) or {}
    adaptive_replacement = phase1_method.get("adaptive_replacement") or {}
    if (
        str(getattr(args, "pbrs_version", "") or "") == "lbf_pbrs_v2"
        and bool(
            getattr(args, "full_clean_runnable", False)
            or (
                bool(phase1_method.get("enable_stage5_final_full_run", False))
                and not bool(phase1_method.get("final_spec_is_summary_only", True))
            )
        )
        and not bool(
            getattr(args, "fallback_to_single_llm_initial_config", None)
            if getattr(args, "fallback_to_single_llm_initial_config", None) is not None
            else phase1_method.get("fallback_to_single_llm_initial_config", True)
        )
        and not bool(adaptive_replacement.get("fallback_to_deterministic_on_llm_error", True))
        and not bool(adaptive_replacement.get("add_deterministic_recovery_fallback", True))
    ):
        return False
    return True


def _dense_reference_metrics_available(metrics_summary: Dict[str, Any]) -> bool:
    if metrics_summary.get("metric_summary"):
        return True
    metric_curves = metrics_summary.get("metric_curves") or {}
    for curve in metric_curves.values():
        if not isinstance(curve, dict):
            continue
        if int(curve.get("num_points") or 0) > 0:
            return True
        if curve.get("sampled_points"):
            return True
    return False


def _dense_reference_state_from_run_reference(
    *,
    run_reference: Dict[str, Any],
    existing_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_dir = Path(str(run_reference.get("run_dir") or ""))
    run_status = _status_from_run_dir(run_dir) if run_dir else "missing_run_dir"
    metrics_summary: Dict[str, Any] = {}
    metrics_available = False
    failure_reason = None
    failure_excerpt = None
    prior_state = existing_state or {}
    checkpoint_root_valid = bool(
        prior_state.get("dense_reference_checkpoint_root_valid", False)
        or prior_state.get("dense_reference_checkpoint_path_argument")
        or ((prior_state.get("dense_reference_run") or {}).get("dense_reference_checkpoint_path_argument"))
    )
    load_step_resolved = bool(
        prior_state.get("dense_reference_load_step_resolved", False)
        or prior_state.get("dense_reference_load_step_argument") is not None
        or ((prior_state.get("dense_reference_run") or {}).get("dense_reference_load_step_argument")) is not None
    )
    if run_dir and (run_dir / "metrics.json").exists():
        try:
            metrics_summary = build_run_summary_from_run_dir(run_dir=run_dir)
            metrics_available = _dense_reference_metrics_available(metrics_summary)
        except Exception as exc:
            failure_reason = f"metrics_summary_failed:{exc}"
    elif _terminal_run_status(run_status):
        failure_reason = "metrics_json_missing"
    dense_status = "running"
    if _terminal_run_status(run_status):
        dense_status = "completed" if run_status == "COMPLETED" else "failed"
        if dense_status == "failed" and failure_reason is None:
            failure_reason, failure_excerpt = _extract_dense_reference_failure_reason(
                run_dir=run_dir,
                run_status=run_status,
            )
    return {
        "dense_reference_status": dense_status,
        "dense_reference_valid": bool(run_status == "COMPLETED" and metrics_available),
        "dense_reference_metrics_available": bool(metrics_available),
        "dense_reference_failure_reason": failure_reason,
        "dense_reference_failure_excerpt": failure_excerpt,
        "dense_reference_checkpoint_root_valid": checkpoint_root_valid,
        "dense_reference_load_step_resolved": load_step_resolved,
        "dense_reference_run_json_status": run_status,
        "dense_reference_metrics_summary": metrics_summary,
    }


def _extract_dense_reference_failure_reason(
    *,
    run_dir: Path,
    run_status: str,
) -> tuple[str, Optional[str]]:
    run_json_path = run_dir / "run.json"
    fail_trace = ""
    if run_json_path.exists():
        try:
            run_json = json.loads(run_json_path.read_text(encoding="utf-8"))
            fail_trace_value = run_json.get("fail_trace") or ""
            if isinstance(fail_trace_value, list):
                fail_trace = "".join(str(item) for item in fail_trace_value)
            else:
                fail_trace = str(fail_trace_value)
        except Exception:
            fail_trace = ""
    if "min() arg is an empty sequence" in fail_trace:
        config_path = run_dir / "config.json"
        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config_payload = {}
        checkpoint_path = Path(str(config_payload.get("checkpoint_path") or ""))
        if checkpoint_path.name.isdigit():
            return (
                "checkpoint_path_leaf_dir_passed_to_run_py",
                "run.py resume scan saw a checkpoint leaf dir instead of checkpoint root dir",
            )
        return (
            "checkpoint_step_discovery_empty_sequence",
            "run.py resume scan found no checkpoint steps for the requested load_step",
        )
    if fail_trace:
        last_line = fail_trace.strip().splitlines()[-1]
        return (f"run_failed:{last_line}", last_line[:400])
    if _terminal_run_status(run_status):
        return (f"terminal_run_status={run_status}", None)
    return ("dense_reference_failure_unknown", None)


def _static_context(args: argparse.Namespace, workflow_id: str) -> Dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "env_key": args.env_key,
        "algorithm": args.train_config,
        "seed": args.seed,
        "objective_metric": "test_sparse_return_mean",
        "early_budget_steps": initial_dense_budget_steps(args),
        "llm_routing": deepcopy(
            llm_routing_artifact_fields(getattr(args, "llm_routing", None) or {})
        ),
        "task_summary": {
            "env_key": args.env_key,
            "time_limit": args.time_limit,
        },
        "pbrs_definition": {
            "beta": "overall shaping scale",
            "wc": "collection/coordination weighting",
            "wp": "derived as 1 - wc",
        },
    }


def _run_or_reconcile_dual_llm_stage1b_core(
    *,
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    workflow_dir: Path,
    stage1b_artifact_path: Path,
    sparse_run_summary: Dict[str, Any],
) -> Dict[str, Any]:
    early_budget_steps = initial_dense_budget_steps(args)
    target_endpoint_step = initial_dense_target_checkpoint_step(args)
    endpoint_tolerance = stage1b_endpoint_tolerance_steps(args)
    endpoint_policy = stage1b_endpoint_selection_policy(args)
    if stage1b_artifact_path.exists():
        payload = json.loads(stage1b_artifact_path.read_text(encoding="utf-8"))
    else:
        payload = default_dual_llm_stage1b_payload(
            workflow_id=manifest["stage_ids"]["stage_1b_dense_reference"],
            workflow_dir=workflow_dir,
            early_budget_steps=early_budget_steps,
        )
    payload["stage1b_candidate_training_t_max"] = int(early_budget_steps)
    payload["stage1b_target_endpoint_step"] = int(target_endpoint_step)
    payload["stage1b_endpoint_tolerance_steps"] = int(endpoint_tolerance)
    payload["stage1b_endpoint_selection_policy"] = str(endpoint_policy)
    payload["llm_routing"] = deepcopy(
        llm_routing_artifact_fields(getattr(args, "llm_routing", None) or {})
    )

    memory = load_or_init_memory(
        workflow_dir,
        _static_context(args, manifest["workflow_id"]),
    )
    launcher = EPyMARLTrainLauncher(repo_root=ROOT, python_executable=args.python_executable)
    local_results_path = "results"
    dense_reference_config = (((getattr(args, "phase1_method", None) or {}).get("dense_reference")) or {})

    if payload["critic_sparse_diagnosis_status"] != "completed":
        diagnosis = _deterministic_sparse_diagnosis(args=args, sparse_run_summary=sparse_run_summary)
        call = _llm_json_call(
            workflow_dir=workflow_dir,
            workflow_id=manifest["workflow_id"],
            profile=str(getattr(args, "workflow_profile", "")),
            env_key=str(args.env_key),
            algorithm=str(args.train_config),
            cache_name="stage1b_critic_sparse_diagnosis",
            agent_role="critic",
            agent_name="stage1b_sparse_diagnosis",
            call_type="diagnosis",
            stage="stage1b",
            reason="stage1b sparse diagnosis",
            prompt_path=CRITIC_SPARSE_PROMPT,
            input_payload={
                "goal": "early_budget_selection",
                "env_key": args.env_key,
                "algorithm": args.train_config,
                "sparse_run_summary": sparse_run_summary,
                "pbrs_definition": memory.get("pbrs_definition") or {},
            },
            system_prompt="Diagnose sparse-reward MARL failures for early dense candidate search. Return strict JSON only.",
            use_real_llm=bool(getattr(args, "use_real_llm", False)),
            api_key_env=str(args.api_key_env),
            base_url=str(args.base_url),
            model=str(args.model),
            temperature=float(args.temperature),
            llm_timeout=float(args.llm_timeout),
            llm_max_retries=int(args.llm_max_retries),
            llm_retry_backoff=float(args.llm_retry_backoff),
            fallback_response=diagnosis,
            round_id=0,
            artifact_path=str(stage1b_artifact_path),
        )
        payload["critic_sparse_diagnosis_status"] = "completed"
        payload["critic_sparse_diagnosis_cache_path"] = call["cache_path"]
        payload["critic_sparse_diagnosis"] = deepcopy(call["parsed_response"])
        memory = update_after_critic_sparse_diagnosis(memory, payload["critic_sparse_diagnosis"])
        save_memory(workflow_dir, memory)
        stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    payload, memory, policy_guidance_result = _maybe_attach_stage1b_policy_guidance(
        args=args,
        workflow_dir=workflow_dir,
        payload=payload,
        memory=memory,
        sparse_run_summary=sparse_run_summary,
        local_results_path=local_results_path,
    )
    save_memory(workflow_dir, memory)
    stage1b_artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    if payload["round1_generator_status"] != "completed":
        fallback = _deterministic_round1_candidates(
            initial_dense_candidates_per_round(args),
            env_family=_stage1b_env_family(args),
        )
        if payload.get("policy_guidance_used") and payload.get("integrated_guidance_card"):
            fallback = build_policy_guided_stage1b_candidate_batch(
                integrated_guidance_card=payload["integrated_guidance_card"],
                max_candidates=initial_dense_candidates_per_round(args),
                env_family="rware" if "rware" in str(args.env_key or "").lower() else "lbf",
            )
        generator_input_payload = maybe_build_stage1b_generator_payload_with_policy_guidance(
            task_metadata={
                "workflow_id": payload.get("workflow_id"),
                "env_key": args.env_key,
                "algorithm": args.train_config,
                "seed": args.seed,
            },
            sparse_baseline_metrics=_stage1_sparse_scalar_metrics(sparse_run_summary),
            pbrs_search_space=_stage1b_policy_guidance_search_space(args),
            integrated_guidance_card=payload.get("integrated_guidance_card") or None,
            existing_reward_only_payload={
                "goal": "early_budget_selection",
                "sparse_diagnosis": payload["critic_sparse_diagnosis"],
                "dense_reference_config": dense_reference_config,
                "policy_guidance_used": False,
                "policy_guidance_fallback_to_reward_only": True,
            },
        )
        call = _llm_json_call(
            workflow_dir=workflow_dir,
            workflow_id=manifest["workflow_id"],
            profile=str(getattr(args, "workflow_profile", "")),
            env_key=str(args.env_key),
            algorithm=str(args.train_config),
            cache_name="stage1b_generator_round1",
            agent_role="generator",
            agent_name="stage1b_round1_generator",
            call_type="generator",
            stage="stage1b",
            reason="stage1b generator round1",
            prompt_path=(
                POLICY_GUIDED_GENERATOR_R1_PROMPT
                if payload.get("policy_guidance_used")
                else GENERATOR_R1_PROMPT
            ),
            input_payload=generator_input_payload,
            system_prompt="Propose diverse early dense PBRS candidates. Return strict JSON only.",
            use_real_llm=bool(getattr(args, "use_real_llm", False)),
            api_key_env=str(args.api_key_env),
            base_url=str(args.base_url),
            model=str(args.model),
            temperature=float(args.temperature),
            llm_timeout=float(args.llm_timeout),
            llm_max_retries=int(args.llm_max_retries),
            llm_retry_backoff=float(args.llm_retry_backoff),
            fallback_response=fallback,
            llm_routing=getattr(args, "llm_routing", None),
            round_id=1,
            artifact_path=str(stage1b_artifact_path),
        )
        payload["round1_candidates"] = _sanitize_candidates(
            list((call["parsed_response"] or {}).get("candidates") or []),
            prefix="round1",
        )[: initial_dense_candidates_per_round(args)]
        _enforce_formal_v2_candidates(
            candidates=payload["round1_candidates"],
            args=args,
            stage_label="stage1b_round1",
        )
        payload["round1_generator_status"] = "completed"
        if payload.get("policy_guidance_used"):
            append_stage1b_policy_guided_candidate_generation(
                memory,
                {
                    "policy_guidance_used": True,
                    "candidate_generation_basis": (
                        fallback.get("candidate_generation_basis") or {}
                    ),
                    "candidates": payload["round1_candidates"],
                },
            )
        memory = update_after_initial_candidate_generation(memory, round_id=1, generator_candidates=payload["round1_candidates"])
        save_memory(workflow_dir, memory)
        stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    if not payload["round1_run_ids"]:
        for candidate in payload["round1_candidates"]:
            plan = build_stage1b_candidate_plan(
                workflow_id=manifest["workflow_id"],
                round_id=1,
                candidate=candidate,
                env_key=args.env_key,
                train_config=args.train_config,
                seed=args.seed,
                time_limit=args.time_limit,
                local_results_path=local_results_path,
                budget_steps=early_budget_steps,
                target_checkpoint_step=target_endpoint_step,
                use_cuda=bool(args.stage1_use_cuda),
                budget_profile=str(getattr(args, "budget_profile", "formal")),
                experiment_tag=manifest["workflow_id"],
                endpoint_tolerance_steps=endpoint_tolerance,
                endpoint_selection_policy=endpoint_policy,
            )
            run_id = str(plan["run_id"])
            payload["round1_run_ids"].append(run_id)
            run_refs = launcher.find_matching_run_references(plan["train_config"])
            if not run_refs:
                stdout_path = workflow_dir / "logs" / f"{run_id}.log"
                launcher.launch_prebuilt_train_config_background(
                    plan["train_config"],
                    stdout_path=stdout_path,
                )
        stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    round1_results: List[Dict[str, Any]] = []
    round1_endpoints: List[Dict[str, Any]] = []
    round1_terminal = True
    for candidate, run_id in zip(payload["round1_candidates"], payload["round1_run_ids"]):
        plan = build_stage1b_candidate_plan(
            workflow_id=manifest["workflow_id"],
            round_id=1,
            candidate=candidate,
            env_key=args.env_key,
            train_config=args.train_config,
            seed=args.seed,
            time_limit=args.time_limit,
            local_results_path=local_results_path,
            budget_steps=early_budget_steps,
            target_checkpoint_step=target_endpoint_step,
            use_cuda=bool(args.stage1_use_cuda),
            budget_profile=str(getattr(args, "budget_profile", "formal")),
            experiment_tag=manifest["workflow_id"],
            endpoint_tolerance_steps=endpoint_tolerance,
            endpoint_selection_policy=endpoint_policy,
        )
        run_refs = launcher.find_matching_run_references(plan["train_config"])
        if not run_refs:
            round1_terminal = False
            round1_results.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "run_id": run_id,
                    "run_status": "launched",
                    "invalid_reason": "run not materialized yet",
                }
            )
            continue
        run_ref = sorted(run_refs, key=lambda item: int(item.get("run_id") or -1))[-1]
        candidate_with_metadata = deepcopy(candidate)
        candidate_with_metadata["candidate_training_t_max"] = int(plan["candidate_training_t_max"])
        result = _candidate_result_from_run_reference(
            candidate=candidate_with_metadata,
            run_id=run_id,
            run_reference=run_ref,
            target_endpoint_step=int(target_endpoint_step),
            endpoint_tolerance_steps=int(endpoint_tolerance),
            endpoint_selection_policy=str(endpoint_policy),
        )
        round1_results.append(result)
        if result.get("endpoint_checkpoint_path"):
            round1_endpoints.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "endpoint_checkpoint_path": result["endpoint_checkpoint_path"],
                    "endpoint_checkpoint_step": result["endpoint_checkpoint_step"],
                }
            )
        if not _terminal_run_status(str(result.get("run_status") or "")):
            round1_terminal = False
    payload["round1_results"] = round1_results
    payload["round1_endpoint_checkpoints"] = round1_endpoints
    memory = update_after_initial_candidate_results(
        memory,
        round_id=1,
        candidate_run_ids=payload["round1_run_ids"],
        candidate_results=payload["round1_results"],
    )
    save_memory(workflow_dir, memory)
    stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    if not round1_terminal:
        payload["stage1b_overall_status"] = "running"
        return payload

    if payload["round1_critic_analysis_status"] != "completed":
        fallback = _deterministic_round1_analysis(payload["round1_results"])
        call = _llm_json_call(
            workflow_dir=workflow_dir,
            workflow_id=manifest["workflow_id"],
            profile=str(getattr(args, "workflow_profile", "")),
            env_key=str(args.env_key),
            algorithm=str(args.train_config),
            cache_name="stage1b_critic_round1_analysis",
            agent_role="critic",
            agent_name="stage1b_round1_critic",
            call_type="critic",
            stage="stage1b",
            reason="stage1b critic round1",
            prompt_path=CRITIC_R1_PROMPT,
            input_payload={
                "goal": "early_budget_selection",
                "round1_results": payload["round1_results"],
            },
            system_prompt="Analyze early dense candidate results and advise a revision round. Return strict JSON only.",
            use_real_llm=bool(getattr(args, "use_real_llm", False)),
            api_key_env=str(args.api_key_env),
            base_url=str(args.base_url),
            model=str(args.model),
            temperature=float(args.temperature),
            llm_timeout=float(args.llm_timeout),
            llm_max_retries=int(args.llm_max_retries),
            llm_retry_backoff=float(args.llm_retry_backoff),
            fallback_response=fallback,
            llm_routing=getattr(args, "llm_routing", None),
            round_id=1,
            artifact_path=str(stage1b_artifact_path),
        )
        payload["round1_critic_analysis_status"] = "completed"
        payload["round1_critic_analysis_cache_path"] = call["cache_path"]
        payload["round1_critic_analysis"] = deepcopy(call["parsed_response"])
        memory = update_after_initial_round_analysis(memory, round_id=1, critic_analysis=payload["round1_critic_analysis"])
        save_memory(workflow_dir, memory)
        stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    if payload["round2_generator_status"] != "completed":
        fallback = _deterministic_round2_candidates(
            payload["round1_results"],
            initial_dense_candidates_per_round(args),
            env_family=_stage1b_env_family(args),
        )
        round2_input_payload = {
            "goal": "early_budget_selection",
            "round1_results": payload["round1_results"],
            "round1_critic_analysis": payload["round1_critic_analysis"],
        }
        if str(getattr(args, "pbrs_version", "") or "") in {"lbf_pbrs_v2", "rware_pbrs_v2"}:
            round2_input_payload["pbrs_search_space"] = _stage1b_policy_guidance_search_space(args)
        call = _llm_json_call(
            workflow_dir=workflow_dir,
            workflow_id=manifest["workflow_id"],
            profile=str(getattr(args, "workflow_profile", "")),
            env_key=str(args.env_key),
            algorithm=str(args.train_config),
            cache_name="stage1b_generator_round2",
            agent_role="generator",
            agent_name="stage1b_round2_generator",
            call_type="generator",
            stage="stage1b",
            reason="stage1b generator round2",
            prompt_path=(
                POLICY_GUIDED_GENERATOR_R2_PROMPT
                if payload.get("policy_guidance_used")
                and str(getattr(args, "pbrs_version", "") or "") in {"lbf_pbrs_v2", "rware_pbrs_v2"}
                else GENERATOR_R2_PROMPT
            ),
            input_payload=round2_input_payload,
            system_prompt="Revise early dense PBRS candidates after round-1 evidence. Return strict JSON only.",
            use_real_llm=bool(getattr(args, "use_real_llm", False)),
            api_key_env=str(args.api_key_env),
            base_url=str(args.base_url),
            model=str(args.model),
            temperature=float(args.temperature),
            llm_timeout=float(args.llm_timeout),
            llm_max_retries=int(args.llm_max_retries),
            llm_retry_backoff=float(args.llm_retry_backoff),
            fallback_response=fallback,
            llm_routing=getattr(args, "llm_routing", None),
            round_id=2,
            artifact_path=str(stage1b_artifact_path),
        )
        payload["round2_candidates"] = _sanitize_candidates(
            list((call["parsed_response"] or {}).get("candidates") or []),
            prefix="round2",
        )[: initial_dense_candidates_per_round(args)]
        _enforce_formal_v2_candidates(
            candidates=payload["round2_candidates"],
            args=args,
            stage_label="stage1b_round2",
        )
        payload["round2_generator_status"] = "completed"
        memory = update_after_initial_candidate_generation(memory, round_id=2, generator_candidates=payload["round2_candidates"])
        save_memory(workflow_dir, memory)
        stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    if not payload["round2_run_ids"]:
        for candidate in payload["round2_candidates"]:
            plan = build_stage1b_candidate_plan(
                workflow_id=manifest["workflow_id"],
                round_id=2,
                candidate=candidate,
                env_key=args.env_key,
                train_config=args.train_config,
                seed=args.seed,
                time_limit=args.time_limit,
                local_results_path=local_results_path,
                budget_steps=early_budget_steps,
                target_checkpoint_step=target_endpoint_step,
                use_cuda=bool(args.stage1_use_cuda),
                budget_profile=str(getattr(args, "budget_profile", "formal")),
                experiment_tag=manifest["workflow_id"],
                endpoint_tolerance_steps=endpoint_tolerance,
                endpoint_selection_policy=endpoint_policy,
            )
            run_id = str(plan["run_id"])
            payload["round2_run_ids"].append(run_id)
            run_refs = launcher.find_matching_run_references(plan["train_config"])
            if not run_refs:
                stdout_path = workflow_dir / "logs" / f"{run_id}.log"
                launcher.launch_prebuilt_train_config_background(
                    plan["train_config"],
                    stdout_path=stdout_path,
                )
        stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    round2_results: List[Dict[str, Any]] = []
    round2_endpoints: List[Dict[str, Any]] = []
    round2_terminal = True
    for candidate, run_id in zip(payload["round2_candidates"], payload["round2_run_ids"]):
        plan = build_stage1b_candidate_plan(
            workflow_id=manifest["workflow_id"],
            round_id=2,
            candidate=candidate,
            env_key=args.env_key,
            train_config=args.train_config,
            seed=args.seed,
            time_limit=args.time_limit,
            local_results_path=local_results_path,
            budget_steps=early_budget_steps,
            target_checkpoint_step=target_endpoint_step,
            use_cuda=bool(args.stage1_use_cuda),
            budget_profile=str(getattr(args, "budget_profile", "formal")),
            experiment_tag=manifest["workflow_id"],
            endpoint_tolerance_steps=endpoint_tolerance,
            endpoint_selection_policy=endpoint_policy,
        )
        run_refs = launcher.find_matching_run_references(plan["train_config"])
        if not run_refs:
            round2_terminal = False
            round2_results.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "run_id": run_id,
                    "run_status": "launched",
                    "invalid_reason": "run not materialized yet",
                }
            )
            continue
        run_ref = sorted(run_refs, key=lambda item: int(item.get("run_id") or -1))[-1]
        candidate_with_metadata = deepcopy(candidate)
        candidate_with_metadata["candidate_training_t_max"] = int(plan["candidate_training_t_max"])
        result = _candidate_result_from_run_reference(
            candidate=candidate_with_metadata,
            run_id=run_id,
            run_reference=run_ref,
            target_endpoint_step=int(target_endpoint_step),
            endpoint_tolerance_steps=int(endpoint_tolerance),
            endpoint_selection_policy=str(endpoint_policy),
        )
        round2_results.append(result)
        if result.get("endpoint_checkpoint_path"):
            round2_endpoints.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "endpoint_checkpoint_path": result["endpoint_checkpoint_path"],
                    "endpoint_checkpoint_step": result["endpoint_checkpoint_step"],
                }
            )
        if not _terminal_run_status(str(result.get("run_status") or "")):
            round2_terminal = False
    payload["round2_results"] = round2_results
    payload["round2_endpoint_checkpoints"] = round2_endpoints
    memory = update_after_initial_candidate_results(
        memory,
        round_id=2,
        candidate_run_ids=payload["round2_run_ids"],
        candidate_results=payload["round2_results"],
    )
    save_memory(workflow_dir, memory)
    stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    if not round2_terminal:
        payload["stage1b_overall_status"] = "running"
        return payload

    if payload["final_critic_selection_status"] != "completed":
        all_results = list(payload["round1_results"]) + list(payload["round2_results"])
        fallback = _build_final_selection_fallback(all_results)
        call = _llm_json_call(
            workflow_dir=workflow_dir,
            workflow_id=manifest["workflow_id"],
            profile=str(getattr(args, "workflow_profile", "")),
            env_key=str(args.env_key),
            algorithm=str(args.train_config),
            cache_name="stage1b_critic_final_selection",
            agent_role="critic",
            agent_name="stage1b_final_critic",
            call_type="critic",
            stage="stage1b",
            reason="stage1b critic final selection",
            prompt_path=CRITIC_FINAL_PROMPT,
            input_payload={
                "goal": "early_budget_selection",
                "all_candidate_results": all_results,
            },
            system_prompt="Select the best early dense candidate for downstream continuation. Return strict JSON only.",
            use_real_llm=bool(getattr(args, "use_real_llm", False)),
            api_key_env=str(args.api_key_env),
            base_url=str(args.base_url),
            model=str(args.model),
            temperature=float(args.temperature),
            llm_timeout=float(args.llm_timeout),
            llm_max_retries=int(args.llm_max_retries),
            llm_retry_backoff=float(args.llm_retry_backoff),
            fallback_response=fallback,
            llm_routing=getattr(args, "llm_routing", None),
            round_id=2,
            artifact_path=str(stage1b_artifact_path),
        )
        final_selection = deepcopy(call["parsed_response"])
        valid_ids = {item["candidate_id"] for item in all_results if item.get("endpoint_checkpoint_path")}
        if str(final_selection.get("selected_candidate_id") or "") not in valid_ids:
            final_selection = fallback
        final_selection, normalization_warnings, validation_errors = _normalize_stage1b_final_selection(
            final_selection=final_selection,
            candidate_results=all_results,
            manifest_context={
                "workflow_id": manifest["workflow_id"],
                "target_endpoint_step": target_endpoint_step,
                "endpoint_selection_policy": endpoint_policy,
            },
        )
        payload["final_critic_selection_status"] = "completed"
        payload["final_critic_selection_cache_path"] = call["cache_path"]
        payload["final_critic_selection"] = deepcopy(final_selection)
        payload["final_critic_selection"]["normalization_warnings"] = list(normalization_warnings)
        payload["final_critic_selection"]["validation_errors"] = list(validation_errors)
        payload["selected_candidate_id"] = final_selection["selected_candidate_id"]
        payload["selected_initial_dense_config"] = deepcopy(final_selection["selected_initial_dense_config"])
        selected_result = next(
            (
                item
                for item in all_results
                if item.get("candidate_id") == final_selection["selected_candidate_id"]
            ),
            None,
        )
        payload["selected_endpoint_checkpoint_path"] = final_selection["selected_endpoint_checkpoint_path"]
        payload["selected_endpoint_checkpoint_step"] = int(final_selection["selected_endpoint_checkpoint_step"])
        payload["selected_checkpoint_root_dir"] = final_selection.get("selected_checkpoint_root_dir")
        payload["selected_endpoint_checkpoint_path_argument"] = final_selection.get(
            "selected_endpoint_checkpoint_path_argument"
        )
        payload["selected_load_step"] = final_selection.get("selected_load_step")
        payload["selected_source_run_id"] = final_selection.get("selected_source_run_id")
        payload["selected_source_workflow_role"] = final_selection.get(
            "selected_source_workflow_role"
        )
        payload["selected_endpoint_distance_from_target"] = (
            (selected_result or {}).get("distance_from_target")
        )
        selected_warning = (selected_result or {}).get("endpoint_warning")
        if normalization_warnings:
            joined = " | ".join(str(item) for item in normalization_warnings if item)
            selected_warning = f"{selected_warning} | {joined}" if selected_warning else joined
        payload["selected_endpoint_warning"] = selected_warning
        payload["selection_reason"] = final_selection.get("selection_reason")
        payload["supported_reward_hypotheses"] = list(final_selection.get("supported_reward_hypotheses") or [])
        payload["rejected_reward_hypotheses"] = list(final_selection.get("rejected_reward_hypotheses") or [])
        payload["search_priors_for_stage2_and_stage3"] = deepcopy(final_selection.get("search_priors_for_stage2_and_stage3") or {})
        payload["dense_reference_selection"] = {
            "selection_mode": "dual_llm_stage1b_final_selection",
            "selected_beta": payload["selected_initial_dense_config"]["beta"],
            "selected_wc": payload["selected_initial_dense_config"].get("wc"),
            "selected_wp": payload["selected_initial_dense_config"].get("wp"),
            "selected_pbrs_version": payload["selected_initial_dense_config"].get("pbrs_version"),
            "selected_mode": payload["selected_initial_dense_config"].get("mode"),
            "selected_active_terms": list(payload["selected_initial_dense_config"].get("active_terms") or []),
            "selected_weights": deepcopy(payload["selected_initial_dense_config"].get("weights") or {}),
            "reason": payload["selection_reason"],
            "candidate_training_t_max": int(early_budget_steps),
            "target_endpoint_step": int(target_endpoint_step),
            "selected_endpoint_checkpoint_step": int(payload["selected_endpoint_checkpoint_step"]),
            "selected_checkpoint_root_dir": payload.get("selected_checkpoint_root_dir"),
            "selected_endpoint_checkpoint_path_argument": payload.get(
                "selected_endpoint_checkpoint_path_argument"
            ),
            "selected_load_step": payload.get("selected_load_step"),
            "selected_source_run_id": payload.get("selected_source_run_id"),
            "selected_source_workflow_role": payload.get("selected_source_workflow_role"),
            "distance_from_target": payload.get("selected_endpoint_distance_from_target"),
            "endpoint_selection_policy": str(endpoint_policy),
            "search_priors_for_stage2_and_stage3": deepcopy(payload["search_priors_for_stage2_and_stage3"]),
        }
        memory = update_after_initial_final_selection(memory, final_selection=final_selection)
        save_memory(workflow_dir, memory)
        stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    selected_candidate = {
        "candidate_id": payload["selected_candidate_id"],
        **deepcopy(payload["selected_initial_dense_config"]),
    }
    if payload["dense_reference_continuation_run_id"] is None:
        try:
            dense_plan = build_dense_reference_continuation_plan(
                workflow_id=manifest["stage_ids"]["stage_1b_dense_reference"],
                selected_candidate=selected_candidate,
                selected_checkpoint_path=str(payload["selected_endpoint_checkpoint_path"]),
                selected_checkpoint_step=int(payload["selected_endpoint_checkpoint_step"]),
                env_key=args.env_key,
                train_config=args.train_config,
                seed=args.seed,
                time_limit=args.time_limit,
                local_results_path=local_results_path,
                target_t_max=dense_reference_target_t_max(args),
                use_cuda=bool(args.stage1_use_cuda),
                budget_profile=str(getattr(args, "budget_profile", "formal")),
                experiment_tag=manifest["workflow_id"],
                run_id=manifest["stage_ids"]["stage_1b_dense_reference"],
            )
        except Exception as exc:
            payload["dense_reference_status"] = "failed_planning"
            payload["dense_reference_valid"] = False
            payload["dense_reference_metrics_available"] = False
            payload["dense_reference_failure_reason"] = f"dense_reference_planning_failed:{exc}"
            payload["dense_reference_checkpoint_root_valid"] = False
            payload["dense_reference_load_step_resolved"] = False
            payload["dense_reference_source_checkpoint_exists"] = False
            stage1b_artifact_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            raise Stage1bImplementationError(str(exc)) from exc
        payload["dense_reference_source_candidate_id"] = selected_candidate["candidate_id"]
        payload["dense_reference_source_checkpoint_path"] = str(
            dense_plan["dense_reference_source_checkpoint_path"]
        )
        payload["dense_reference_source_checkpoint_root_dir"] = str(
            dense_plan["dense_reference_source_checkpoint_root_dir"]
        )
        payload["dense_reference_source_checkpoint_step"] = int(
            dense_plan["dense_reference_source_checkpoint_step"]
        )
        payload["dense_reference_checkpoint_path_argument"] = str(
            dense_plan["dense_reference_checkpoint_path_argument"]
        )
        payload["dense_reference_load_step_argument"] = int(
            dense_plan["dense_reference_load_step_argument"]
        )
        payload["dense_reference_source_checkpoint_exists"] = bool(
            dense_plan.get("dense_reference_source_checkpoint_exists", False)
        )
        payload["dense_reference_checkpoint_root_valid"] = bool(
            dense_plan.get("dense_reference_checkpoint_root_valid", False)
        )
        payload["dense_reference_load_step_resolved"] = bool(
            dense_plan.get("dense_reference_load_step_resolved", False)
        )
        payload["dense_reference_pbrs_config"] = deepcopy(selected_candidate)
        payload["dense_reference_target_t_max"] = int(dense_plan["dense_reference_target_t_max"])
        payload["dense_reference_continuation_run_id"] = str(dense_plan["run_id"])
        payload["dense_reference_run"] = deepcopy(dense_plan)
        payload["adaptive_mainline_source"] = build_adaptive_mainline_source(
            selected_candidate=selected_candidate,
            selected_checkpoint_path=str(payload["selected_endpoint_checkpoint_path"]),
            selected_checkpoint_step=int(payload["selected_endpoint_checkpoint_step"]),
        )
        payload["adaptive_mainline_source_candidate_id"] = payload["adaptive_mainline_source"]["adaptive_mainline_source_candidate_id"]
        payload["adaptive_mainline_source_checkpoint_path"] = payload["adaptive_mainline_source"]["adaptive_mainline_source_checkpoint_path"]
        payload["adaptive_mainline_source_checkpoint_step"] = payload["adaptive_mainline_source"]["adaptive_mainline_source_checkpoint_step"]
        payload["adaptive_mainline_source_checkpoint_root_dir"] = payload["adaptive_mainline_source"].get(
            "adaptive_mainline_source_checkpoint_root_dir"
        )
        payload["adaptive_mainline_checkpoint_path_argument"] = payload["adaptive_mainline_source"].get(
            "adaptive_mainline_checkpoint_path_argument"
        )
        payload["adaptive_mainline_load_step_argument"] = payload["adaptive_mainline_source"].get(
            "adaptive_mainline_load_step_argument"
        )
        payload["adaptive_mainline_load_step_resolved"] = bool(
            payload["adaptive_mainline_source"].get("adaptive_mainline_load_step_resolved", False)
        )
        payload["adaptive_mainline_initial_pbrs_config"] = deepcopy(
            payload["adaptive_mainline_source"]["adaptive_mainline_initial_pbrs_config"]
        )
        planning_error = str(dense_plan.get("dense_reference_planning_error") or "")
        if dense_plan.get("dense_reference_source_is_mock"):
            payload["dense_reference_status"] = "skipped_mock_source"
            payload["dense_reference_valid"] = False
            payload["dense_reference_metrics_available"] = False
            payload["dense_reference_failure_reason"] = "skipped_mock_source"
        elif planning_error:
            payload["dense_reference_status"] = "failed_planning"
            payload["dense_reference_valid"] = False
            payload["dense_reference_metrics_available"] = False
            payload["dense_reference_failure_reason"] = planning_error
        else:
            run_refs = launcher.find_matching_run_references(dense_plan["train_config"])
            if not run_refs:
                stdout_path = workflow_dir / "logs" / f"{dense_plan['run_id']}.log"
                launcher.launch_prebuilt_train_config_background(
                    dense_plan["train_config"],
                    stdout_path=stdout_path,
                )
        memory = update_after_dense_reference_launch(
            memory,
            {
                "dense_reference_source_candidate_id": payload["dense_reference_source_candidate_id"],
                "dense_reference_source_checkpoint_path": payload["dense_reference_source_checkpoint_path"],
                "dense_reference_source_checkpoint_step": payload["dense_reference_source_checkpoint_step"],
                "dense_reference_pbrs_config": payload["dense_reference_pbrs_config"],
                "dense_reference_run_id": payload["dense_reference_continuation_run_id"],
                "dense_reference_target_t_max": payload["dense_reference_target_t_max"],
            },
        )
        save_memory(workflow_dir, memory)
        stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    dense_plan = payload.get("dense_reference_run") or {}
    dense_train_config = dense_plan.get("train_config")
    if dense_train_config:
        run_refs = launcher.find_matching_run_references(dense_train_config)
        if run_refs:
            dense_run_ref = sorted(run_refs, key=lambda item: int(item.get("run_id") or -1))[-1]
            payload["dense_reference_run_dir"] = str(dense_run_ref.get("run_dir"))
            payload["dense_reference_run"].setdefault("result", {})
            payload["dense_reference_run"]["result"]["run_reference"] = {
                "run_id": dense_run_ref.get("run_id"),
                "run_dir": dense_run_ref.get("run_dir"),
                "workflow_id": dense_train_config.get("workflow_id"),
            }
            dense_state = _dense_reference_state_from_run_reference(
                run_reference=dense_run_ref,
                existing_state=payload,
            )
            payload.update(dense_state)
            payload["dense_reference_run_id"] = dense_run_ref.get("run_id")
        else:
            if str(payload.get("dense_reference_status") or "") not in {
                "failed_planning",
                "skipped_mock_source",
            }:
                payload["dense_reference_status"] = "running"
                payload["dense_reference_valid"] = False
                payload["dense_reference_metrics_available"] = False
                payload["dense_reference_failure_reason"] = None

    payload["stage1b_overall_status"] = (
        "completed"
        if (
            payload.get("dense_reference_run_dir")
            and str(payload.get("dense_reference_status") or "") in {"completed", "failed", "skipped"}
        )
        or str(payload.get("dense_reference_status") or "") in {"failed_planning", "skipped_mock_source"}
        else "running"
    )
    stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return payload


def _fallback_enabled(args: argparse.Namespace) -> bool:
    phase1_method = getattr(args, "phase1_method", None) or {}
    return bool(
        getattr(args, "fallback_to_single_llm_initial_config", None)
        if getattr(args, "fallback_to_single_llm_initial_config", None) is not None
        else phase1_method.get("fallback_to_single_llm_initial_config")
    )


def _deterministic_fallback_config() -> Dict[str, Any]:
    return {
        "candidate_id": "stage1b_fallback_default",
        "beta": 0.3,
        "wc": 0.5,
        "wp": 0.5,
        "candidate_type": "reference_like",
        "hypothesis": "Deterministic fallback uses a stable balanced dense initialization.",
        "expected_early_effect": "Provide a safe dense initialization checkpoint when the dual-LLM path fails.",
        "risk": "May be suboptimal because it bypasses dual-LLM search.",
    }


def _deterministic_rware_fallback_config() -> Dict[str, Any]:
    normalized = normalize_rware_pbrs_v2_config(
        {
            "pbrs_version": "rware_pbrs_v2",
            "mode": "balanced_delivery_progress",
            "beta": 0.30,
            "weights": {
                "shelf": 0.25,
                "pickup": 0.15,
                "goal": 0.35,
                "deliv": 0.15,
                "traffic": 0.10,
                "stab": 0.0,
            },
        }
    )
    return {
        "candidate_id": "stage1b_rware_fallback_default",
        "candidate_type": "reference_like",
        "pbrs_version": "rware_pbrs_v2",
        "mode": str(normalized.get("mode") or "balanced_delivery_progress"),
        "beta": float(normalized.get("beta", 0.30)),
        "active_terms": list(normalized.get("active_terms") or []),
        "weights": deepcopy(normalized.get("weights") or {}),
        "evidence_keys_used": [],
        "hypothesis": "Deterministic fallback uses a stable balanced RWARE PBRS-v2 dense initialization.",
        "expected_early_effect": "Provide a safe dense initialization checkpoint when the dual-LLM path fails.",
        "risk": "May be suboptimal because it bypasses dual-LLM search.",
    }


def _is_rware_formal_profile(args: argparse.Namespace) -> bool:
    return _stage1b_env_family(args) == "rware" and bool(
        getattr(args, "formal_execute_guard_required", False)
        or "formal" in str(getattr(args, "workflow_profile", "") or "").lower()
        or str(getattr(args, "budget_profile", "") or "").lower().startswith("rware_mappo_formal")
    )


def _resolve_stage1b_fallback_candidate(
    *,
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    sparse_run_summary: Dict[str, Any],
    fallback_reason: str,
) -> tuple[Dict[str, Any], str]:
    use_lbf_v2 = str(getattr(args, "pbrs_version", "") or "") == "lbf_pbrs_v2"
    use_rware_v2 = str(getattr(args, "pbrs_version", "") or "") == "rware_pbrs_v2"
    phase1_method = getattr(args, "phase1_method", None) or {}
    dense_reference_config = deepcopy((phase1_method.get("dense_reference") or {}))
    if use_rware_v2 and _is_rware_formal_profile(args):
        raise Stage1bImplementationError(
            "formal RWARE Stage1b fallback is disabled; fail-fast instead of using deterministic or legacy fallback"
        )
    if bool(getattr(args, "use_real_llm", False)):
        try:
            selection = select_dense_reference_with_llm(
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
            candidate = {
                    "candidate_id": "stage1b_fallback_single_llm",
                    "beta": float(selection["selected_beta"]),
                    "wc": float(selection["selected_wc"]),
                    "wp": float(selection["selected_wp"]),
                    "candidate_type": "reference_like",
                    "hypothesis": selection.get("reason") or fallback_reason,
                    "expected_early_effect": "Single-LLM fallback dense initialization.",
                    "risk": "Fallback path reduces search breadth.",
                }
            if use_lbf_v2:
                legacy_candidate = candidate
                candidate = legacy_lbf_candidate_to_v2(
                    beta=legacy_candidate["beta"],
                    wc=legacy_candidate["wc"],
                    wp=legacy_candidate["wp"],
                    candidate_id=legacy_candidate["candidate_id"],
                    candidate_type=legacy_candidate["candidate_type"],
                )
                candidate.update(
                    {
                        "hypothesis": legacy_candidate["hypothesis"],
                        "expected_early_effect": legacy_candidate["expected_early_effect"],
                        "risk": legacy_candidate["risk"],
                    }
                )
            return (candidate, "single_llm")
        except Exception:
            pass
    candidate = (
        _deterministic_rware_fallback_config()
        if use_rware_v2
        else _deterministic_fallback_config()
    )
    if use_lbf_v2:
        legacy_candidate = candidate
        candidate = legacy_lbf_candidate_to_v2(
            beta=legacy_candidate["beta"],
            wc=legacy_candidate["wc"],
            wp=legacy_candidate["wp"],
            candidate_id=legacy_candidate["candidate_id"],
            candidate_type=legacy_candidate["candidate_type"],
        )
        candidate.update(
            {
                "hypothesis": legacy_candidate["hypothesis"],
                "expected_early_effect": legacy_candidate["expected_early_effect"],
                "risk": legacy_candidate["risk"],
            }
        )
    return candidate, "deterministic_default"


def _record_fallback_in_memory(
    *,
    workflow_dir: Path,
    memory: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    stage1b = memory.setdefault("stage1b_early_dense_search", {})
    stage1b["fallback"] = {
        "used": bool(payload.get("stage1b_fallback_used")),
        "reason": payload.get("fallback_reason"),
        "config": deepcopy(payload.get("fallback_config") or {}),
        "source": payload.get("fallback_source"),
        "run_id": payload.get("fallback_run_id"),
    }
    save_memory(workflow_dir, memory)


def _run_stage1b_fallback(
    *,
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    workflow_dir: Path,
    stage1b_artifact_path: Path,
    sparse_run_summary: Dict[str, Any],
    fallback_reason: str,
) -> Dict[str, Any]:
    early_budget_steps = initial_dense_budget_steps(args)
    target_endpoint_step = initial_dense_target_checkpoint_step(args)
    endpoint_tolerance = stage1b_endpoint_tolerance_steps(args)
    endpoint_policy = stage1b_endpoint_selection_policy(args)
    if stage1b_artifact_path.exists():
        payload = json.loads(stage1b_artifact_path.read_text(encoding="utf-8"))
    else:
        payload = default_dual_llm_stage1b_payload(
            workflow_id=manifest["stage_ids"]["stage_1b_dense_reference"],
            workflow_dir=workflow_dir,
            early_budget_steps=early_budget_steps,
        )
    memory = load_or_init_memory(
        workflow_dir,
        _static_context(args, manifest["workflow_id"]),
    )
    launcher = EPyMARLTrainLauncher(repo_root=ROOT, python_executable=args.python_executable)
    local_results_path = "results"
    fallback_candidate, fallback_source = _resolve_stage1b_fallback_candidate(
        args=args,
        manifest=manifest,
        sparse_run_summary=sparse_run_summary,
        fallback_reason=fallback_reason,
    )
    payload["stage1b_fallback_used"] = True
    payload["fallback_reason"] = fallback_reason
    payload["fallback_config"] = deepcopy(fallback_candidate)
    payload["fallback_source"] = fallback_source
    payload["fallback_candidates_launched"] = True

    plan = build_stage1b_candidate_plan(
        workflow_id=manifest["workflow_id"],
        round_id=99,
        candidate=fallback_candidate,
        env_key=args.env_key,
        train_config=args.train_config,
        seed=args.seed,
        time_limit=args.time_limit,
        local_results_path=local_results_path,
        budget_steps=early_budget_steps,
        target_checkpoint_step=target_endpoint_step,
        use_cuda=bool(args.stage1_use_cuda),
        budget_profile=str(getattr(args, "budget_profile", "formal")),
        experiment_tag=manifest["workflow_id"],
        endpoint_tolerance_steps=endpoint_tolerance,
        endpoint_selection_policy=endpoint_policy,
    )
    payload["fallback_run_id"] = str(plan["run_id"])
    run_refs = launcher.find_matching_run_references(plan["train_config"])
    if not run_refs:
        stdout_path = workflow_dir / "logs" / f"{plan['run_id']}.log"
        launcher.launch_prebuilt_train_config_background(
            plan["train_config"],
            stdout_path=stdout_path,
        )
        _record_fallback_in_memory(workflow_dir=workflow_dir, memory=memory, payload=payload)
        payload["stage1b_overall_status"] = "running"
        stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return payload

    run_ref = sorted(run_refs, key=lambda item: int(item.get("run_id") or -1))[-1]
    result = _candidate_result_from_run_reference(
        candidate={**deepcopy(fallback_candidate), "candidate_training_t_max": int(plan["candidate_training_t_max"])},
        run_id=str(plan["run_id"]),
        run_reference=run_ref,
        target_endpoint_step=int(target_endpoint_step),
        endpoint_tolerance_steps=int(endpoint_tolerance),
        endpoint_selection_policy=str(endpoint_policy),
    )
    payload["round1_results"] = [result]
    payload["round1_endpoint_checkpoints"] = []
    if result.get("endpoint_checkpoint_path"):
        payload["round1_endpoint_checkpoints"].append(
            {
                "candidate_id": fallback_candidate["candidate_id"],
                "endpoint_checkpoint_path": result["endpoint_checkpoint_path"],
                "endpoint_checkpoint_step": result["endpoint_checkpoint_step"],
            }
        )
    if not _terminal_run_status(str(result.get("run_status") or "")):
        _record_fallback_in_memory(workflow_dir=workflow_dir, memory=memory, payload=payload)
        payload["stage1b_overall_status"] = "running"
        stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return payload
    if not result.get("endpoint_checkpoint_path"):
        _record_fallback_in_memory(workflow_dir=workflow_dir, memory=memory, payload=payload)
        payload["stage1b_overall_status"] = "failed"
        stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(
            f"fallback candidate failed to materialize endpoint checkpoint: {result.get('invalid_reason')}"
        )

    payload["selected_candidate_id"] = fallback_candidate["candidate_id"]
    payload["selected_initial_dense_config"] = deepcopy(
        result.get("pbrs_config")
        or (
            normalize_lbf_pbrs_v2_config(fallback_candidate)
            if str(fallback_candidate.get("pbrs_version") or "") == "lbf_pbrs_v2"
            else normalize_rware_pbrs_v2_config(fallback_candidate)
            if str(fallback_candidate.get("pbrs_version") or "") == "rware_pbrs_v2"
            else {
                "beta": float(fallback_candidate["beta"]),
                "wc": float(fallback_candidate["wc"]),
                "wp": float(fallback_candidate["wp"]),
            }
        )
    )
    payload["selected_endpoint_checkpoint_path"] = result["endpoint_checkpoint_path"]
    payload["selected_endpoint_checkpoint_step"] = int(result["endpoint_checkpoint_step"])
    payload["selected_checkpoint_root_dir"] = result.get("checkpoint_root_dir")
    payload["selected_endpoint_checkpoint_path_argument"] = result.get(
        "endpoint_checkpoint_path_argument"
    )
    payload["selected_load_step"] = result.get("load_step")
    payload["selected_source_run_id"] = result.get("source_run_id")
    payload["selected_source_workflow_role"] = result.get("source_workflow_role")
    payload["selected_endpoint_distance_from_target"] = result.get("distance_from_target")
    payload["selected_endpoint_warning"] = result.get("endpoint_warning")
    payload["selection_reason"] = f"Fallback path selected {fallback_source} initial configuration after Stage 1b error."
    payload["supported_reward_hypotheses"] = [payload["selection_reason"]]
    payload["rejected_reward_hypotheses"] = []
    payload["search_priors_for_stage2_and_stage3"] = {
        "fallback_used": True,
        "fallback_source": fallback_source,
    }
    payload["dense_reference_selection"] = {
        "selection_mode": f"fallback_{fallback_source}",
        "selected_beta": payload["selected_initial_dense_config"]["beta"],
        "selected_wc": payload["selected_initial_dense_config"].get("wc"),
        "selected_wp": payload["selected_initial_dense_config"].get("wp"),
        "selected_pbrs_version": payload["selected_initial_dense_config"].get("pbrs_version"),
        "selected_mode": payload["selected_initial_dense_config"].get("mode"),
        "selected_active_terms": list(payload["selected_initial_dense_config"].get("active_terms") or []),
        "selected_weights": deepcopy(payload["selected_initial_dense_config"].get("weights") or {}),
        "reason": payload["selection_reason"],
        "candidate_training_t_max": int(early_budget_steps),
        "target_endpoint_step": int(target_endpoint_step),
        "selected_endpoint_checkpoint_step": int(payload["selected_endpoint_checkpoint_step"]),
        "selected_checkpoint_root_dir": payload.get("selected_checkpoint_root_dir"),
        "selected_endpoint_checkpoint_path_argument": payload.get(
            "selected_endpoint_checkpoint_path_argument"
        ),
        "selected_load_step": payload.get("selected_load_step"),
        "selected_source_run_id": payload.get("selected_source_run_id"),
        "selected_source_workflow_role": payload.get("selected_source_workflow_role"),
        "distance_from_target": payload.get("selected_endpoint_distance_from_target"),
        "endpoint_selection_policy": str(endpoint_policy),
    }
    memory = update_after_initial_final_selection(
        memory,
        final_selection={
            "selected_candidate_id": payload["selected_candidate_id"],
            "selected_initial_dense_config": deepcopy(payload["selected_initial_dense_config"]),
            "selected_endpoint_checkpoint_path": payload["selected_endpoint_checkpoint_path"],
            "selected_endpoint_checkpoint_step": payload["selected_endpoint_checkpoint_step"],
            "selected_checkpoint_root_dir": payload.get("selected_checkpoint_root_dir"),
            "selected_endpoint_checkpoint_path_argument": payload.get(
                "selected_endpoint_checkpoint_path_argument"
            ),
            "selected_load_step": payload.get("selected_load_step"),
            "selected_source_run_id": payload.get("selected_source_run_id"),
            "selected_source_workflow_role": payload.get("selected_source_workflow_role"),
            "selection_reason": payload["selection_reason"],
            "supported_reward_hypotheses": deepcopy(payload["supported_reward_hypotheses"]),
            "rejected_reward_hypotheses": deepcopy(payload["rejected_reward_hypotheses"]),
            "search_priors_for_stage2_and_stage3": deepcopy(payload["search_priors_for_stage2_and_stage3"]),
        },
    )

    selected_candidate = {
        "candidate_id": payload["selected_candidate_id"],
        **deepcopy(payload["selected_initial_dense_config"]),
    }
    try:
        dense_plan = build_dense_reference_continuation_plan(
            workflow_id=manifest["stage_ids"]["stage_1b_dense_reference"],
            selected_candidate=selected_candidate,
            selected_checkpoint_path=str(payload["selected_endpoint_checkpoint_path"]),
            selected_checkpoint_step=int(payload["selected_endpoint_checkpoint_step"]),
            env_key=args.env_key,
            train_config=args.train_config,
            seed=args.seed,
            time_limit=args.time_limit,
            local_results_path=local_results_path,
            target_t_max=dense_reference_target_t_max(args),
            use_cuda=bool(args.stage1_use_cuda),
            budget_profile=str(getattr(args, "budget_profile", "formal")),
            experiment_tag=manifest["workflow_id"],
            run_id=manifest["stage_ids"]["stage_1b_dense_reference"],
        )
    except Exception as exc:
        payload["dense_reference_status"] = "failed_planning"
        payload["dense_reference_valid"] = False
        payload["dense_reference_metrics_available"] = False
        payload["dense_reference_failure_reason"] = f"dense_reference_planning_failed:{exc}"
        payload["dense_reference_checkpoint_root_valid"] = False
        payload["dense_reference_load_step_resolved"] = False
        stage1b_artifact_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        raise Stage1bImplementationError(str(exc)) from exc
    payload["dense_reference_source_candidate_id"] = selected_candidate["candidate_id"]
    payload["dense_reference_source_checkpoint_path"] = str(
        dense_plan["dense_reference_source_checkpoint_path"]
    )
    payload["dense_reference_source_checkpoint_root_dir"] = str(
        dense_plan["dense_reference_source_checkpoint_root_dir"]
    )
    payload["dense_reference_source_checkpoint_step"] = int(
        dense_plan["dense_reference_source_checkpoint_step"]
    )
    payload["dense_reference_checkpoint_path_argument"] = str(
        dense_plan["dense_reference_checkpoint_path_argument"]
    )
    payload["dense_reference_load_step_argument"] = int(
        dense_plan["dense_reference_load_step_argument"]
    )
    payload["dense_reference_source_checkpoint_exists"] = bool(
        dense_plan.get("dense_reference_source_checkpoint_exists", False)
    )
    payload["dense_reference_checkpoint_root_valid"] = bool(
        dense_plan.get("dense_reference_checkpoint_root_valid", False)
    )
    payload["dense_reference_load_step_resolved"] = bool(
        dense_plan.get("dense_reference_load_step_resolved", False)
    )
    payload["dense_reference_pbrs_config"] = deepcopy(selected_candidate)
    payload["dense_reference_target_t_max"] = int(dense_plan["dense_reference_target_t_max"])
    payload["dense_reference_continuation_run_id"] = str(dense_plan["run_id"])
    payload["dense_reference_run"] = deepcopy(dense_plan)
    payload["adaptive_mainline_source"] = build_adaptive_mainline_source(
        selected_candidate=selected_candidate,
        selected_checkpoint_path=str(payload["selected_endpoint_checkpoint_path"]),
        selected_checkpoint_step=int(payload["selected_endpoint_checkpoint_step"]),
    )
    payload["adaptive_mainline_source_candidate_id"] = payload["adaptive_mainline_source"]["adaptive_mainline_source_candidate_id"]
    payload["adaptive_mainline_source_checkpoint_path"] = payload["adaptive_mainline_source"]["adaptive_mainline_source_checkpoint_path"]
    payload["adaptive_mainline_source_checkpoint_step"] = payload["adaptive_mainline_source"]["adaptive_mainline_source_checkpoint_step"]
    payload["adaptive_mainline_source_checkpoint_root_dir"] = payload["adaptive_mainline_source"].get(
        "adaptive_mainline_source_checkpoint_root_dir"
    )
    payload["adaptive_mainline_checkpoint_path_argument"] = payload["adaptive_mainline_source"].get(
        "adaptive_mainline_checkpoint_path_argument"
    )
    payload["adaptive_mainline_load_step_argument"] = payload["adaptive_mainline_source"].get(
        "adaptive_mainline_load_step_argument"
    )
    payload["adaptive_mainline_load_step_resolved"] = bool(
        payload["adaptive_mainline_source"].get("adaptive_mainline_load_step_resolved", False)
    )
    payload["adaptive_mainline_initial_pbrs_config"] = deepcopy(
        payload["adaptive_mainline_source"]["adaptive_mainline_initial_pbrs_config"]
    )
    dense_refs = launcher.find_matching_run_references(dense_plan["train_config"])
    if not dense_refs:
        stdout_path = workflow_dir / "logs" / f"{dense_plan['run_id']}.log"
        launcher.launch_prebuilt_train_config_background(
            dense_plan["train_config"],
            stdout_path=stdout_path,
        )
        payload["dense_reference_status"] = "running"
        payload["dense_reference_valid"] = False
        payload["dense_reference_metrics_available"] = False
        payload["dense_reference_failure_reason"] = None
    else:
        dense_ref = sorted(dense_refs, key=lambda item: int(item.get("run_id") or -1))[-1]
        payload["dense_reference_run_dir"] = str(dense_ref.get("run_dir"))
        payload["dense_reference_run"].setdefault("result", {})
        payload["dense_reference_run"]["result"]["run_reference"] = {
            "run_id": dense_ref.get("run_id"),
            "run_dir": dense_ref.get("run_dir"),
            "workflow_id": dense_plan["train_config"].get("workflow_id"),
        }
        dense_state = _dense_reference_state_from_run_reference(
            run_reference=dense_ref,
            existing_state=payload,
        )
        payload.update(dense_state)
        payload["dense_reference_run_id"] = dense_ref.get("run_id")
    memory = update_after_dense_reference_launch(
        memory,
        {
            "dense_reference_source_candidate_id": payload["dense_reference_source_candidate_id"],
            "dense_reference_source_checkpoint_path": payload["dense_reference_source_checkpoint_path"],
            "dense_reference_source_checkpoint_step": payload["dense_reference_source_checkpoint_step"],
            "dense_reference_pbrs_config": deepcopy(payload["dense_reference_pbrs_config"]),
            "dense_reference_run_id": payload["dense_reference_continuation_run_id"],
            "dense_reference_target_t_max": payload["dense_reference_target_t_max"],
        },
    )
    _record_fallback_in_memory(workflow_dir=workflow_dir, memory=memory, payload=payload)
    payload["stage1b_overall_status"] = (
        "completed"
        if payload.get("dense_reference_run_dir")
        and str(payload.get("dense_reference_status") or "") in {"completed", "failed", "skipped"}
        else "running"
    )
    stage1b_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return payload


def run_or_reconcile_dual_llm_stage1b(
    *,
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    workflow_dir: Path,
    stage1b_artifact_path: Path,
    sparse_run_summary: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        if bool(getattr(args, "stage1b_force_fallback", False)):
            raise Stage1bRecoverableFallbackError("forced Stage 1b fallback for smoke validation")
        return _run_or_reconcile_dual_llm_stage1b_core(
            args=args,
            manifest=manifest,
            workflow_dir=workflow_dir,
            stage1b_artifact_path=stage1b_artifact_path,
            sparse_run_summary=sparse_run_summary,
        )
    except Exception as exc:
        if not _should_fallback_for_stage1b_exception(args=args, exc=exc):
            if stage1b_artifact_path.exists():
                payload = json.loads(stage1b_artifact_path.read_text(encoding="utf-8"))
            else:
                payload = default_dual_llm_stage1b_payload(
                    workflow_id=manifest["stage_ids"]["stage_1b_dense_reference"],
                    workflow_dir=workflow_dir,
                    early_budget_steps=initial_dense_budget_steps(args),
                )
            payload["stage1b_overall_status"] = "failed"
            payload["stage1b_fallback_used"] = False
            payload["fallback_source"] = None
            payload["fallback_reason"] = None
            payload.setdefault("errors", [])
            payload["errors"].append(
                {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "category": "implementation_error",
                }
            )
            stage1b_artifact_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            raise
        return _run_stage1b_fallback(
            args=args,
            manifest=manifest,
            workflow_dir=workflow_dir,
            stage1b_artifact_path=stage1b_artifact_path,
            sparse_run_summary=sparse_run_summary,
            fallback_reason=str(exc),
        )
