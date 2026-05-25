from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional


MEMORY_FILENAME = "llm_experiment_memory.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_path(workflow_dir: Path | str) -> Path:
    return Path(workflow_dir) / MEMORY_FILENAME


def _default_memory(static_context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "workflow_id": str(static_context.get("workflow_id") or ""),
        "env_key": str(static_context.get("env_key") or ""),
        "algorithm": str(static_context.get("algorithm") or ""),
        "seed": int(static_context.get("seed") or 1),
        "objective_metric": str(static_context.get("objective_metric") or "test_sparse_return_mean"),
        "llm_routing": deepcopy(static_context.get("llm_routing") or {}),
        "task_summary": deepcopy(static_context.get("task_summary") or {}),
        "pbrs_definition": deepcopy(static_context.get("pbrs_definition") or {}),
        "stage1b_early_dense_search": {
            "early_budget_steps": int(static_context.get("early_budget_steps") or 800000),
            "critic_sparse_diagnosis": {},
            "rounds": [
                {
                    "round_id": 1,
                    "generator_candidates": [],
                    "candidate_run_ids": [],
                    "candidate_results": [],
                    "critic_analysis": {},
                },
                {
                    "round_id": 2,
                    "generator_candidates": [],
                    "candidate_run_ids": [],
                    "candidate_results": [],
                    "critic_analysis": {},
                },
            ],
            "critic_final_selection": {},
            "selected_initial_dense_config": {},
            "selected_candidate_id": "",
            "selected_endpoint_checkpoint_path": "",
            "selected_endpoint_checkpoint_step": int(static_context.get("early_budget_steps") or 800000),
            "supported_reward_hypotheses": [],
            "rejected_reward_hypotheses": [],
            "search_priors_for_stage2_and_stage3": {},
        },
        "fixed_checkpoint_plan": [],
        "dense_reference_continuation": {},
        "adaptive_decision_history": [],
        "last_updated": _utc_now_iso(),
    }


def load_or_init_memory(
    workflow_dir: Path | str,
    static_context: Dict[str, Any],
) -> Dict[str, Any]:
    path = _memory_path(workflow_dir)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("workflow_id", str(static_context.get("workflow_id") or ""))
                payload.setdefault("env_key", str(static_context.get("env_key") or ""))
                payload.setdefault("algorithm", str(static_context.get("algorithm") or ""))
                payload.setdefault("seed", int(static_context.get("seed") or 1))
                payload.setdefault("objective_metric", str(static_context.get("objective_metric") or "test_sparse_return_mean"))
                payload.setdefault("llm_routing", deepcopy(static_context.get("llm_routing") or {}))
                payload.setdefault("task_summary", deepcopy(static_context.get("task_summary") or {}))
                payload.setdefault("pbrs_definition", deepcopy(static_context.get("pbrs_definition") or {}))
                payload.setdefault("stage1b_early_dense_search", deepcopy(_default_memory(static_context)["stage1b_early_dense_search"]))
                payload.setdefault("fixed_checkpoint_plan", [])
                payload.setdefault("dense_reference_continuation", {})
                payload.setdefault("adaptive_decision_history", [])
                payload["last_updated"] = _utc_now_iso()
                return payload
        except Exception:
            pass
    payload = _default_memory(static_context)
    save_memory(workflow_dir, payload)
    return payload


def save_memory(workflow_dir: Path | str, memory: Dict[str, Any]) -> Path:
    path = _memory_path(workflow_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    memory["last_updated"] = _utc_now_iso()
    path.write_text(
        json.dumps(memory, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def build_memory_summary_for_prompt(
    memory: Dict[str, Any],
    current_stage: str,
    current_decision_id: Optional[str] = None,
    max_chars: int = 6000,
) -> str:
    payload = {
        "workflow_id": memory.get("workflow_id"),
        "env_key": memory.get("env_key"),
        "algorithm": memory.get("algorithm"),
        "seed": memory.get("seed"),
        "objective_metric": memory.get("objective_metric"),
        "current_stage": current_stage,
        "current_decision_id": current_decision_id,
        "stage1b_early_dense_search": memory.get("stage1b_early_dense_search"),
        "fixed_checkpoint_plan": memory.get("fixed_checkpoint_plan"),
        "dense_reference_continuation": memory.get("dense_reference_continuation"),
        "adaptive_decision_history": memory.get("adaptive_decision_history"),
    }
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    truncated = deepcopy(payload)
    history = list(truncated.get("adaptive_decision_history") or [])
    truncated["adaptive_decision_history"] = history[-2:]
    rounds = list((((truncated.get("stage1b_early_dense_search") or {}).get("rounds")) or []))
    for item in rounds:
        item["candidate_results"] = list(item.get("candidate_results") or [])[-3:]
    text = json.dumps(truncated, indent=2, sort_keys=True, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def update_after_critic_sparse_diagnosis(memory: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
    memory.setdefault("stage1b_early_dense_search", {})["critic_sparse_diagnosis"] = deepcopy(diagnosis)
    return memory


def update_after_initial_candidate_generation(
    memory: Dict[str, Any],
    *,
    round_id: int,
    generator_candidates: list[Dict[str, Any]],
) -> Dict[str, Any]:
    rounds = memory.setdefault("stage1b_early_dense_search", {}).setdefault("rounds", [])
    if 1 <= round_id <= len(rounds):
        rounds[round_id - 1]["generator_candidates"] = deepcopy(generator_candidates)
    return memory


def update_after_initial_candidate_results(
    memory: Dict[str, Any],
    *,
    round_id: int,
    candidate_run_ids: list[Any],
    candidate_results: list[Dict[str, Any]],
) -> Dict[str, Any]:
    rounds = memory.setdefault("stage1b_early_dense_search", {}).setdefault("rounds", [])
    if 1 <= round_id <= len(rounds):
        rounds[round_id - 1]["candidate_run_ids"] = list(candidate_run_ids)
        rounds[round_id - 1]["candidate_results"] = deepcopy(candidate_results)
    return memory


def update_after_initial_round_analysis(
    memory: Dict[str, Any],
    *,
    round_id: int,
    critic_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    rounds = memory.setdefault("stage1b_early_dense_search", {}).setdefault("rounds", [])
    if 1 <= round_id <= len(rounds):
        rounds[round_id - 1]["critic_analysis"] = deepcopy(critic_analysis)
    return memory


def update_after_initial_final_selection(
    memory: Dict[str, Any],
    *,
    final_selection: Dict[str, Any],
) -> Dict[str, Any]:
    stage1b = memory.setdefault("stage1b_early_dense_search", {})
    stage1b["critic_final_selection"] = deepcopy(final_selection)
    stage1b["selected_initial_dense_config"] = deepcopy(final_selection.get("selected_initial_dense_config") or {})
    stage1b["selected_candidate_id"] = str(final_selection.get("selected_candidate_id") or "")
    stage1b["selected_endpoint_checkpoint_path"] = str(final_selection.get("selected_endpoint_checkpoint_path") or "")
    stage1b["selected_endpoint_checkpoint_step"] = int(final_selection.get("selected_endpoint_checkpoint_step") or 0)
    stage1b["supported_reward_hypotheses"] = list(final_selection.get("supported_reward_hypotheses") or [])
    stage1b["rejected_reward_hypotheses"] = list(final_selection.get("rejected_reward_hypotheses") or [])
    stage1b["search_priors_for_stage2_and_stage3"] = deepcopy(final_selection.get("search_priors_for_stage2_and_stage3") or {})
    return memory


def update_after_dense_reference_launch(memory: Dict[str, Any], dense_reference_payload: Dict[str, Any]) -> Dict[str, Any]:
    memory["dense_reference_continuation"] = deepcopy(dense_reference_payload)
    return memory


def update_after_fixed_checkpoint_plan(
    memory: Dict[str, Any],
    checkpoint_plan: list[Dict[str, Any]],
) -> Dict[str, Any]:
    memory["fixed_checkpoint_plan"] = deepcopy(checkpoint_plan)
    return memory


def update_after_stage3_decision(memory: Dict[str, Any], decision_payload: Dict[str, Any]) -> Dict[str, Any]:
    history = memory.setdefault("adaptive_decision_history", [])
    decision_id = str(decision_payload.get("decision_id") or "")
    existing_index = next(
        (index for index, item in enumerate(history) if str(item.get("decision_id") or "") == decision_id),
        None,
    )
    if existing_index is None:
        history.append(deepcopy(decision_payload))
    else:
        history[existing_index] = deepcopy(decision_payload)
    return memory
