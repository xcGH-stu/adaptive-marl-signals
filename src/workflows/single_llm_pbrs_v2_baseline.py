from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rewarding.lbf_pbrs_v2 import DEFAULT_MODE_CONFIGS, TERM_NAMES, normalize_lbf_pbrs_v2_config
from scripts.run_reward_workflow import build_default_workflow_spec
from workflows.diagnostics import load_metric_series, series_summary
from workflows.llm_api_ledger import extract_request_id, write_ledger_event
from workflows.phase1_method import build_run_summary_from_run_dir
from workflows.train_launcher import EPyMARLTrainLauncher


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "results" / "single_llm_pbrs_v2_baselines"
DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

ROUND1_PROMPT_PATH = DEFAULT_PROMPT_DIR / "single_llm_reward_generation_prompt.md"
ROUND2_PROMPT_PATH = DEFAULT_PROMPT_DIR / "single_llm_reward_refinement_prompt.md"

BASELINE_NAME = "single_llm_pbrs_v2_reward_generation"
PAPER_NAME = "Single-LLM Reward Generation (Eureka/L2R-style)"
DEFAULT_ENV_KEY = "lbforaging:Foraging-8x8-2p-1f-v3"
DEFAULT_ALGORITHM = "qmix"
DEFAULT_SEED = 1
DEFAULT_CANDIDATE_ROUNDS = 2
DEFAULT_CANDIDATES_PER_ROUND = 3
DEFAULT_CANDIDATE_BUDGET = 850000
DEFAULT_FULL_T_MAX = 2050000
ALLOWED_BASELINE_MODES = (
    "balanced_progress",
    "early_discovery",
    "collection_readiness",
    "coverage_recovery",
)
BASELINE_MODE_TO_RUNTIME_MODE = {
    "balanced_progress": "balanced_collection_ready",
    "early_discovery": "coverage_ready_balance",
    "collection_readiness": "balanced_collection_ready",
    "coverage_recovery": "coverage_ready_balance",
}
FORBIDDEN_PROMPT_TOKENS = (
    "policy_guidance",
    "behavior_summary",
    "integrated_guidance_card",
    "critic_diagnosis",
    "policy_guidance_history",
    "adaptive_replacement",
    "no_change",
    "winner_promotion",
    "final_staged_adaptive_spec",
)
CACHE_NAMESPACE_VERSION = "single_llm_pbrs_v2_baseline_v2"
REAL_LLM_SCHEMA_SMOKE_SUMMARY_NAME = "real_llm_schema_smoke_summary.json"
REAL_LLM_SCHEMA_SMOKE_RETRY_SUMMARY_PATH = ROOT / "reports" / "single_llm_pbrs_v2_real_llm_schema_smoke_retry_summary.json"
ALLOWED_CANDIDATE_TYPES = (
    "balanced",
    "exploration",
    "collection_readiness",
    "coverage_recovery",
    "stability_recovery",
    "allocation_rebalance",
    "conservative_reference",
)


@dataclass
class BaselinePaths:
    workflow_dir: Path
    manifest: Path
    round1_generation: Path
    round1_candidates: Path
    round1_results: Path
    round2_generation: Path
    round2_candidates: Path
    round2_results: Path
    winner_selection: Path
    winner_fixed_config: Path
    full_training_plan: Path
    full_training_result: Path
    metrics_summary: Path
    baseline_state: Path
    llm_cache_dir: Path
    real_llm_schema_smoke_summary: Path


def resolve_paths(results_root: Path | str, workflow_id: str) -> BaselinePaths:
    workflow_dir = Path(results_root) / workflow_id
    return BaselinePaths(
        workflow_dir=workflow_dir,
        manifest=workflow_dir / "workflow_manifest.json",
        round1_generation=workflow_dir / "round1_generation.json",
        round1_candidates=workflow_dir / "round1_candidates.json",
        round1_results=workflow_dir / "round1_results.json",
        round2_generation=workflow_dir / "round2_generation.json",
        round2_candidates=workflow_dir / "round2_candidates.json",
        round2_results=workflow_dir / "round2_results.json",
        winner_selection=workflow_dir / "winner_selection.json",
        winner_fixed_config=workflow_dir / "winner_fixed_config.json",
        full_training_plan=workflow_dir / "full_training_plan.json",
        full_training_result=workflow_dir / "full_training_result.json",
        metrics_summary=workflow_dir / "metrics_summary.json",
        baseline_state=workflow_dir / "single_llm_baseline_state.json",
        llm_cache_dir=workflow_dir / "llm_cache",
        real_llm_schema_smoke_summary=workflow_dir / REAL_LLM_SCHEMA_SMOKE_SUMMARY_NAME,
    )


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json_dict(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    candidates = [stripped]
    if "```" in stripped:
        start = stripped.find("```")
        while start != -1:
            end = stripped.find("```", start + 3)
            if end == -1:
                break
            block = stripped[start + 3 : end].strip()
            if "\n" in block:
                first, remainder = block.split("\n", 1)
                if first.strip().lower() == "json":
                    block = remainder.strip()
            candidates.append(block)
            start = stripped.find("```", end + 3)
    left = stripped.find("{")
    right = stripped.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidates.append(stripped[left : right + 1])
    last_error: Optional[Exception] = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Expected a JSON object, got invalid response: {last_error}")


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def _prompt_version_id(prompt_path: Path) -> str:
    encoded = prompt_path.read_text(encoding="utf-8").encode("utf-8")
    return sha256(encoded).hexdigest()[:12]


def _disabled_policy_guidance_config() -> Dict[str, Any]:
    return {
        "enable_policy_guidance": False,
        "policy_guidance_mode": "context_only",
        "policy_guidance_direct_action_override": False,
        "policy_guidance_modify_learner": False,
    }


def _allowed_runtime_modes() -> Tuple[str, ...]:
    return tuple(sorted(DEFAULT_MODE_CONFIGS.keys()))


def _default_candidate_schema() -> Dict[str, Any]:
    return {
        "candidate_id": "example_balanced_progress",
        "candidate_type": "balanced",
        "pbrs_version": "lbf_pbrs_v2",
        "baseline_mode": "balanced_progress",
        "pbrs_mode": "balanced_collection_ready",
        "beta": 0.5,
        "active_terms": ["col", "app", "cov", "ready"],
        "weights": {
            "col": 0.25,
            "app": 0.25,
            "cov": 0.25,
            "ready": 0.25,
            "alloc": 0.0,
            "stab": 0.0,
        },
        "rationale": "Example schema only.",
        "expected_effect": "Example schema only.",
        "risk_notes": "Example schema only.",
    }


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_weights(weights: Dict[str, Any], active_terms: Sequence[str]) -> Dict[str, float]:
    normalized = {term: max(0.0, _float(weights.get(term), 0.0)) for term in TERM_NAMES}
    for term in ("alloc", "stab"):
        normalized[term] = 0.0
    active = [term for term in active_terms if term in TERM_NAMES]
    if active:
        for term in TERM_NAMES:
            if term not in active and term not in {"alloc", "stab"}:
                normalized[term] = 0.0
    total = sum(normalized.values())
    if total <= 0.0:
        fallback = _default_candidate_schema()["weights"]
        normalized = {term: float(fallback[term]) for term in TERM_NAMES}
        total = sum(normalized.values())
    return {term: float(normalized[term]) / float(total) for term in TERM_NAMES}


def sanitize_single_llm_candidate(
    candidate: Dict[str, Any],
    *,
    index: int = 0,
    require_candidate_type: bool = False,
) -> Dict[str, Any]:
    payload = deepcopy(candidate or {})
    baseline_mode = str(payload.get("baseline_mode") or "").strip()
    if baseline_mode not in BASELINE_MODE_TO_RUNTIME_MODE:
        if baseline_mode not in ALLOWED_BASELINE_MODES:
            baseline_mode = ALLOWED_BASELINE_MODES[index % len(ALLOWED_BASELINE_MODES)]
    runtime_mode = str(payload.get("pbrs_mode") or BASELINE_MODE_TO_RUNTIME_MODE[baseline_mode]).strip()
    if runtime_mode == baseline_mode:
        runtime_mode = BASELINE_MODE_TO_RUNTIME_MODE[baseline_mode]
    if runtime_mode not in _allowed_runtime_modes():
        runtime_mode = BASELINE_MODE_TO_RUNTIME_MODE[baseline_mode]
    active_terms = []
    for term in list(payload.get("active_terms") or []):
        term_text = str(term or "").strip()
        if term_text in TERM_NAMES and term_text not in {"alloc", "stab"} and term_text not in active_terms:
            active_terms.append(term_text)
    if not active_terms:
        active_terms = ["col", "app", "cov", "ready"]
    weights = _normalize_weights(dict(payload.get("weights") or {}), active_terms)
    normalized_runtime = normalize_lbf_pbrs_v2_config(
        {
            "pbrs_version": "lbf_pbrs_v2",
            "mode": runtime_mode,
            "beta": min(0.7, max(0.2, _float(payload.get("beta"), 0.5))),
            "active_terms": active_terms,
            "weights": weights,
            "candidate_id": str(payload.get("candidate_id") or f"candidate_{index + 1}"),
        }
    )
    candidate_type = str(payload.get("candidate_type") or "").strip()
    if not candidate_type and not require_candidate_type:
        candidate_type = _default_candidate_schema()["candidate_type"]
    return {
        "candidate_id": str(payload.get("candidate_id") or normalized_runtime.get("candidate_id") or f"candidate_{index + 1}"),
        "candidate_type": candidate_type,
        "pbrs_version": "lbf_pbrs_v2",
        "baseline_mode": baseline_mode,
        "pbrs_mode": runtime_mode,
        "beta": float(normalized_runtime["beta"]),
        "gamma": float(normalized_runtime["gamma"]),
        "active_terms": list(normalized_runtime["active_terms"]),
        "weights": deepcopy(normalized_runtime["weights"]),
        "wc": float(normalized_runtime["wc"]),
        "wp": float(normalized_runtime["wp"]),
        "rationale": str(payload.get("rationale") or ""),
        "expected_effect": str(payload.get("expected_effect") or ""),
        "risk_notes": str(payload.get("risk_notes") or ""),
        "baseline_alias_mapping": {
            "baseline_mode": baseline_mode,
            "runtime_pbrs_mode": runtime_mode,
        },
    }


def _candidate_validation_errors(candidate: Dict[str, Any], *, index: int = 0) -> List[str]:
    payload = deepcopy(candidate or {})
    errors: List[str] = []
    pbrs_version = str(payload.get("pbrs_version") or "lbf_pbrs_v2").strip()
    if pbrs_version != "lbf_pbrs_v2":
        errors.append(f"candidate[{index}] pbrs_version must be lbf_pbrs_v2")
    candidate_type = str(payload.get("candidate_type") or "").strip()
    if not candidate_type:
        errors.append(f"candidate[{index}] candidate_type is required")
    elif candidate_type not in ALLOWED_CANDIDATE_TYPES:
        errors.append(
            f"candidate[{index}] candidate_type must be one of: {', '.join(ALLOWED_CANDIDATE_TYPES)}"
        )
    baseline_mode = str(payload.get("baseline_mode") or "").strip()
    if baseline_mode and baseline_mode not in ALLOWED_BASELINE_MODES:
        errors.append(f"candidate[{index}] baseline_mode is not allowed: {baseline_mode}")
    raw_runtime_mode = str(payload.get("pbrs_mode") or "").strip()
    if raw_runtime_mode == baseline_mode and raw_runtime_mode:
        errors.append(f"candidate[{index}] pbrs_mode must be a runtime enum, not baseline_mode alias: {raw_runtime_mode}")
    if raw_runtime_mode and raw_runtime_mode not in _allowed_runtime_modes() and raw_runtime_mode not in BASELINE_MODE_TO_RUNTIME_MODE:
        errors.append(f"candidate[{index}] pbrs_mode is invalid: {raw_runtime_mode}")
    active_terms = payload.get("active_terms")
    if not isinstance(active_terms, list) or not active_terms:
        errors.append(f"candidate[{index}] active_terms must be a non-empty list")
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        errors.append(f"candidate[{index}] weights must be a dict")
    else:
        missing = [term for term in TERM_NAMES if term not in weights]
        if missing:
            errors.append(f"candidate[{index}] weights missing keys: {', '.join(missing)}")
    if (("wc" in payload) or ("wp" in payload)) and not isinstance(weights, dict):
        errors.append(f"candidate[{index}] legacy beta/wc/wp-only payload is not accepted as clean PBRS-v2")
    return errors


def sanitize_candidate_batch(
    candidates: Sequence[Dict[str, Any]],
    *,
    round_id: int,
    strict_clean_schema: bool = False,
    allow_completion_fallback: bool = True,
) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    validation_errors: List[str] = []
    for index, candidate in enumerate(list(candidates or [])):
        if strict_clean_schema:
            validation_errors.extend(_candidate_validation_errors(candidate, index=index))
        item = sanitize_single_llm_candidate(
            candidate,
            index=index,
            require_candidate_type=strict_clean_schema,
        )
        if not item["candidate_id"]:
            item["candidate_id"] = f"round{round_id}_candidate_{index + 1}"
        sanitized.append(item)
    if strict_clean_schema and validation_errors:
        raise ValueError("clean PBRS-v2 candidate validation failed: " + " | ".join(validation_errors))
    if strict_clean_schema and len(sanitized) != DEFAULT_CANDIDATES_PER_ROUND:
        raise ValueError(
            "real-LLM candidate generation must return exactly "
            f"{DEFAULT_CANDIDATES_PER_ROUND} clean PBRS-v2 candidates, got {len(sanitized)}"
        )
    if len(sanitized) < DEFAULT_CANDIDATES_PER_ROUND and allow_completion_fallback:
        for index in range(len(sanitized), DEFAULT_CANDIDATES_PER_ROUND):
            fallback = _default_candidate_schema()
            fallback["candidate_id"] = f"round{round_id}_fallback_{index + 1}"
            fallback["baseline_mode"] = ALLOWED_BASELINE_MODES[index % len(ALLOWED_BASELINE_MODES)]
            fallback["pbrs_mode"] = BASELINE_MODE_TO_RUNTIME_MODE[fallback["baseline_mode"]]
            fallback["rationale"] = "Deterministic fallback candidate to keep the round complete."
            fallback["expected_effect"] = "Provide a valid PBRS-v2 schema fallback."
            fallback["risk_notes"] = "Fallback candidate."
            sanitized.append(sanitize_single_llm_candidate(fallback, index=index))
    return sanitized[:DEFAULT_CANDIDATES_PER_ROUND]


def default_single_llm_baseline_profile() -> Dict[str, Any]:
    return {
        "profile_name": "qmix_single_llm_pbrs_v2_reward_gen_seed1",
        "workflow_kind": "single_llm_pbrs_v2_baseline",
        "baseline_name": BASELINE_NAME,
        "paper_name": PAPER_NAME,
        "results_root": str(DEFAULT_RESULTS_ROOT),
        "env_key": DEFAULT_ENV_KEY,
        "train_config": DEFAULT_ALGORITHM,
        "seed": DEFAULT_SEED,
        "time_limit": 50,
        "candidate_rounds": DEFAULT_CANDIDATE_ROUNDS,
        "candidates_per_round": DEFAULT_CANDIDATES_PER_ROUND,
        "candidate_budget_steps": DEFAULT_CANDIDATE_BUDGET,
        "full_training_t_max": DEFAULT_FULL_T_MAX,
        "reward_paradigm": "pbrs",
        "pbrs_version": "lbf_pbrs_v2",
        "use_pbrs": True,
        "eval_use_pbrs": False,
        "test_sparse_only": True,
        "temperature": 0.0,
        "model": "gpt-5.2",
        "api_key_env": "IUSEAPI_API_KEY",
        "base_url": "https://www.iuseapi.com/v1",
        "llm_timeout": 60.0,
        "llm_max_retries": 1,
        "llm_retry_backoff": 0.0,
        "python_executable": "python",
        "local_results_path": "results",
        "use_cuda": False,
        "stage1_use_cuda": False,
        "stage3_use_cuda": False,
        "stage5_use_cuda": False,
        "enable_policy_guidance": False,
        "policy_memory": False,
        "critic_enabled": False,
        "stage3_enabled": False,
        "adaptive_replacement": False,
        "winner_promotion": False,
        "no_change_gate": False,
        "final_staged_adaptive_spec": False,
        "candidate_train_execute_default": False,
        "formal_execute_guard_required": True,
        "budget_profile": "single_llm_reward_generation_formal",
        "phase1_method": {
            "method_version": "single_llm_pbrs_v2_baseline_v1",
            "enable_policy_guidance": False,
            "critic_enabled": False,
            "stage3_enabled": False,
            "adaptive_replacement": False,
        },
    }


def _candidate_mock_bank(round_id: int) -> List[Dict[str, Any]]:
    if round_id == 1:
        return [
            {
                "candidate_id": "r1_balanced_progress",
                "candidate_type": "balanced",
                "baseline_mode": "balanced_progress",
                "beta": 0.45,
                "active_terms": ["col", "app", "cov", "ready"],
                "weights": {"col": 0.28, "app": 0.22, "cov": 0.22, "ready": 0.28, "alloc": 0.0, "stab": 0.0},
                "rationale": "Balanced progress candidate for stable collection progress.",
                "expected_effect": "Improve average collection progress without over-specializing.",
                "risk_notes": "May underweight discovery in sparse starts.",
            },
            {
                "candidate_id": "r1_early_discovery",
                "candidate_type": "exploration",
                "baseline_mode": "early_discovery",
                "beta": 0.55,
                "active_terms": ["col", "app", "cov", "ready"],
                "weights": {"col": 0.18, "app": 0.20, "cov": 0.34, "ready": 0.28, "alloc": 0.0, "stab": 0.0},
                "rationale": "Discovery-heavy candidate for earlier sparse reward contact.",
                "expected_effect": "Increase early discovery and coverage signals.",
                "risk_notes": "Can overvalue exploration late in training.",
            },
            {
                "candidate_id": "r1_collection_readiness",
                "candidate_type": "collection_readiness",
                "baseline_mode": "collection_readiness",
                "beta": 0.40,
                "active_terms": ["col", "app", "cov", "ready"],
                "weights": {"col": 0.22, "app": 0.18, "cov": 0.20, "ready": 0.40, "alloc": 0.0, "stab": 0.0},
                "rationale": "Readiness-focused candidate for cooperative pickup preparation.",
                "expected_effect": "Encourage agents to coordinate before collection.",
                "risk_notes": "Could be too conservative if approach pressure is insufficient.",
            },
        ]
    return [
        {
            "candidate_id": "r2_balanced_progress_refined",
            "candidate_type": "balanced",
            "baseline_mode": "balanced_progress",
            "beta": 0.50,
            "active_terms": ["col", "app", "cov", "ready"],
            "weights": {"col": 0.30, "app": 0.20, "cov": 0.20, "ready": 0.30, "alloc": 0.0, "stab": 0.0},
            "rationale": "Refined balanced candidate after round-1 comparison.",
            "expected_effect": "Lift overall AUC with stronger collection-readiness balance.",
            "risk_notes": "Still moderate exploration pressure only.",
        },
        {
            "candidate_id": "r2_coverage_recovery_refined",
            "candidate_type": "coverage_recovery",
            "baseline_mode": "coverage_recovery",
            "beta": 0.58,
            "active_terms": ["col", "app", "cov", "ready"],
            "weights": {"col": 0.18, "app": 0.16, "cov": 0.38, "ready": 0.28, "alloc": 0.0, "stab": 0.0},
            "rationale": "Refined coverage recovery candidate to preserve early contact.",
            "expected_effect": "Improve exploration AUC while retaining collection follow-through.",
            "risk_notes": "Could remain noisy if coverage dominates too long.",
        },
        {
            "candidate_id": "r2_collection_readiness_refined",
            "candidate_type": "collection_readiness",
            "baseline_mode": "collection_readiness",
            "beta": 0.47,
            "active_terms": ["col", "app", "cov", "ready"],
            "weights": {"col": 0.24, "app": 0.18, "cov": 0.16, "ready": 0.42, "alloc": 0.0, "stab": 0.0},
            "rationale": "Refined readiness candidate with slightly stronger collection pressure.",
            "expected_effect": "Improve late short-run returns once targets are discovered.",
            "risk_notes": "May lag if coverage remains the main bottleneck.",
        },
    ]


def build_mock_generation(round_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    candidates = sanitize_candidate_batch(_candidate_mock_bank(round_id), round_id=round_id)
    return {
        "round_id": round_id,
        "generator_role": "single_llm_generator",
        "llm_used": False,
        "mock_generation": True,
        "prompt_payload": deepcopy(payload),
        "parsed": {
            "candidates": deepcopy(candidates),
            "notes": f"Deterministic mock generation for round {round_id}.",
        },
        "repair_retry_used": False,
        "cache_hit": False,
    }


def _summarize_sparse_context_from_run_dir(run_dir: str | Path) -> Dict[str, Any]:
    summary = build_run_summary_from_run_dir(run_dir=run_dir)
    metric_curves = summary.get("metric_curves") or {}
    test_sparse = metric_curves.get("test_sparse_return_mean") or {}
    train_sparse = metric_curves.get("sparse_return_mean") or {}
    test_return = metric_curves.get("test_return_mean") or {}
    ep_length = metric_curves.get("test_ep_length_mean") or metric_curves.get("ep_length_mean") or {}

    def _points(curve: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(curve.get("sampled_points") or [])

    def _last_value(curve: Dict[str, Any]) -> Optional[float]:
        points = _points(curve)
        return None if not points else float(points[-1]["value"])

    def _best_value(curve: Dict[str, Any]) -> Optional[float]:
        points = _points(curve)
        return None if not points else max(float(item["value"]) for item in points)

    def _last_k(curve: Dict[str, Any], k: int = 3) -> Optional[float]:
        points = _points(curve)[-max(1, min(k, len(_points(curve)))) :]
        if not points:
            return None
        return float(sum(float(item["value"]) for item in points) / len(points))

    def _rough_curve(curve: Dict[str, Any]) -> str:
        points = _points(curve)
        if not points:
            return "unavailable"
        first = float(points[0]["value"])
        last = float(points[-1]["value"])
        peak = max(float(item["value"]) for item in points)
        return f"first={first:.4f}, last={last:.4f}, peak={peak:.4f}, points={len(points)}"

    return {
        "source": "existing_sparse_run",
        "run_dir": str(run_dir),
        "final_test_sparse_return": _last_value(test_sparse) if _points(test_sparse) else _last_value(test_return),
        "max_test_sparse_return": _best_value(test_sparse) if _points(test_sparse) else _best_value(test_return),
        "last_k_test_sparse_return": _last_k(test_sparse) if _points(test_sparse) else _last_k(test_return),
        "ep_length_mean": _last_value(ep_length),
        "rough_train_curve_summary": _rough_curve(train_sparse),
        "rough_test_curve_summary": _rough_curve(test_sparse) if _points(test_sparse) else _rough_curve(test_return),
        "metric_curves": {
            "sparse_return_mean": deepcopy(train_sparse),
            "test_sparse_return_mean": deepcopy(test_sparse),
            "test_return_mean": deepcopy(test_return),
            "test_ep_length_mean": deepcopy(ep_length),
        },
    }


def build_sparse_baseline_context(config: Dict[str, Any]) -> Dict[str, Any]:
    sparse_run_dir = str(config.get("sparse_run_dir") or "").strip()
    if sparse_run_dir:
        return _summarize_sparse_context_from_run_dir(sparse_run_dir)
    budget = int(config.get("candidate_budget_steps") or DEFAULT_CANDIDATE_BUDGET)
    return {
        "source": "placeholder_summary",
        "final_test_sparse_return": 0.08,
        "max_test_sparse_return": 0.12,
        "last_k_test_sparse_return": 0.10,
        "ep_length_mean": 46.0,
        "rough_train_curve_summary": "Sparse training shows slow but monotonic improvement with late plateau.",
        "rough_test_curve_summary": "Sparse test returns improve modestly and remain below dense-reward expectations.",
        "budget_context": {
            "candidate_budget_steps": budget,
            "full_training_t_max": int(config.get("full_training_t_max") or DEFAULT_FULL_T_MAX),
        },
    }


def build_prompt_payload(config: Dict[str, Any], *, round_id: int, sparse_context: Dict[str, Any], previous_candidates: Optional[List[Dict[str, Any]]] = None, previous_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    payload = {
        "baseline_name": BASELINE_NAME,
        "paper_name": PAPER_NAME,
        "round_id": round_id,
        "llm_role": "single_llm_generator",
        "workflow_id": str(config["workflow_id"]),
        "environment": {
            "env_key": str(config.get("env_key") or DEFAULT_ENV_KEY),
            "algorithm": str(config.get("train_config") or DEFAULT_ALGORITHM),
            "seed": int(config.get("seed") or DEFAULT_SEED),
            "time_limit": int(config.get("time_limit") or 50),
        },
        "reward_setting": {
            "reward_family": "PBRS-v2 structured reward generation only",
            "pbrs_version": "lbf_pbrs_v2",
            "supported_runtime_modes": list(_allowed_runtime_modes()),
            "baseline_mode_alias_mapping": deepcopy(BASELINE_MODE_TO_RUNTIME_MODE),
            "allowed_terms": list(TERM_NAMES),
            "candidate_schema": _default_candidate_schema(),
            "constraints": [
                "Output PBRS-v2 structured JSON only.",
                "Do not output Python reward code.",
                "Use only the sparse-baseline scalar context provided in this payload.",
                "alloc and stab weights should remain 0.0 for this baseline.",
                "beta must stay within [0.2, 0.7].",
            ],
        },
        "sparse_baseline_summary": deepcopy(sparse_context),
    }
    if round_id == 2:
        payload["round1_candidates"] = deepcopy(previous_candidates or [])
        payload["round1_scalar_results"] = deepcopy(previous_results or [])
    return payload


def render_prompt(template_path: Path, payload: Dict[str, Any]) -> str:
    template = template_path.read_text(encoding="utf-8")
    return "\n".join(
        [
            template.strip(),
            "",
            "Payload:",
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            "",
            "Return strict JSON only.",
        ]
    )


def _prompt_contamination_flags(payload: Dict[str, Any]) -> Dict[str, bool]:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False).lower()
    return {token: (token in text) for token in FORBIDDEN_PROMPT_TOKENS}


def call_real_llm_generation(
    *,
    config: Dict[str, Any],
    payload: Dict[str, Any],
    prompt_path: Path,
    cache_dir: Path,
    workflow_dir: Path,
) -> Dict[str, Any]:
    from workflows.clients.openai_backend import OpenAIChatBackend

    prompt = render_prompt(prompt_path, payload)
    payload_hash = _hash_payload(payload)
    prompt_version = _prompt_version_id(prompt_path)
    model_name = str(config.get("model") or "gpt-5.2")
    cache_namespace = cache_dir / BASELINE_NAME / model_name / prompt_path.stem / prompt_version
    cache_path = cache_namespace / f"{CACHE_NAMESPACE_VERSION}_{payload_hash}.json"
    cache_namespace.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cached = load_json(cache_path)
        write_ledger_event(
            workflow_dir=workflow_dir,
            workflow_id=str(config["workflow_id"]),
            profile=str(config.get("workflow_profile") or config.get("profile_name") or ""),
            env_key=str(config.get("env_key") or ""),
            algorithm=str(config.get("train_config") or ""),
            stage=f"single_llm_baseline_round{int(payload['round_id'])}",
            call_type="schema_smoke_candidate_generation" if bool(config.get("dry_run")) else "candidate_generation",
            reason=f"round{int(payload['round_id'])} generation cache hit",
            model=str(cached.get("model") or config.get("model") or ""),
            provider="openai",
            base_url=str(config.get("base_url") or ""),
            payload=payload,
            payload_hash=payload_hash,
            request_id=extract_request_id(cached),
            success=True,
            cache_hit=True,
            event_state="cache_hit",
            artifact_path=str(cache_path),
            metadata={
                "baseline_name": BASELINE_NAME,
                "prompt_path": str(prompt_path),
                "prompt_version": prompt_version,
                "cache_namespace_version": CACHE_NAMESPACE_VERSION,
            },
        )
        cached["cache_hit"] = True
        cached["cache_path"] = str(cache_path)
        return cached

    backend = OpenAIChatBackend(
        api_key_env=str(config.get("api_key_env") or "IUSEAPI_API_KEY"),
        base_url=str(config.get("base_url") or "https://www.iuseapi.com/v1"),
        model=str(config.get("model") or "gpt-5.2"),
        temperature=float(config.get("temperature") or 0.0),
        timeout=float(config.get("llm_timeout") or 60.0),
        max_retries=int(config.get("llm_max_retries") or 1),
        retry_backoff_seconds=float(config.get("llm_retry_backoff") or 0.0),
    )
    llm_result: Dict[str, Any] = {}
    raw_text = ""
    parsed: Dict[str, Any] | None = None
    validation_errors: List[str] = []
    try:
        llm_result = backend.generate_text(
            system_prompt=(
                "You are a single-LLM MARL reward-generation baseline. "
                "Produce PBRS-v2 candidate JSON only. Do not act as a critic."
            ),
            user_prompt=prompt,
            metadata={
                "role": "single_llm_generator",
                "workflow_id": str(config["workflow_id"]),
                "round_id": int(payload["round_id"]),
                "baseline_name": BASELINE_NAME,
            },
        )
        raw_text = str(llm_result.get("text") or "")
        parsed = _extract_json_dict(raw_text)
        raw_candidates = list(parsed.get("candidates") or [])
        for index, candidate in enumerate(raw_candidates):
            validation_errors.extend(_candidate_validation_errors(candidate, index=index))
        candidates = sanitize_candidate_batch(
            raw_candidates,
            round_id=int(payload["round_id"]),
            strict_clean_schema=True,
            allow_completion_fallback=False,
        )
        response = {
            "round_id": int(payload["round_id"]),
            "generator_role": "single_llm_generator",
            "llm_used": True,
            "mock_generation": False,
            "cache_hit": False,
            "prompt_payload": deepcopy(payload),
            "raw_text": raw_text,
            "raw_response": deepcopy(llm_result.get("raw_response") or {}),
            "usage": deepcopy(llm_result.get("usage") or {}),
            "model": str(llm_result.get("model") or config.get("model") or ""),
            "cache_path": str(cache_path),
            "parsed": {
                "candidates": deepcopy(candidates),
                "notes": parsed.get("notes"),
            },
            "repair_retry_used": False,
            "deterministic_fallback_allowed": False,
        }
        save_json(cache_path, response)
        write_ledger_event(
            workflow_dir=workflow_dir,
            workflow_id=str(config["workflow_id"]),
            profile=str(config.get("workflow_profile") or config.get("profile_name") or ""),
            env_key=str(config.get("env_key") or ""),
            algorithm=str(config.get("train_config") or ""),
            stage=f"single_llm_baseline_round{int(payload['round_id'])}",
            call_type="schema_smoke_candidate_generation" if bool(config.get("dry_run")) else "candidate_generation",
            reason=f"round{int(payload['round_id'])} generation",
            model=str(response.get("model") or ""),
            provider="openai",
            base_url=str(config.get("base_url") or ""),
            payload=payload,
            payload_hash=payload_hash,
            usage=response.get("usage"),
            request_id=extract_request_id(llm_result),
            success=True,
            cache_hit=False,
            event_state="attempt",
            artifact_path=str(cache_path),
            metadata={
                "baseline_name": BASELINE_NAME,
                "prompt_path": str(prompt_path),
                "prompt_version": prompt_version,
                "cache_namespace_version": CACHE_NAMESPACE_VERSION,
            },
        )
        return response
    except Exception as exc:
        write_ledger_event(
            workflow_dir=workflow_dir,
            workflow_id=str(config["workflow_id"]),
            profile=str(config.get("workflow_profile") or config.get("profile_name") or ""),
            env_key=str(config.get("env_key") or ""),
            algorithm=str(config.get("train_config") or ""),
            stage=f"single_llm_baseline_round{int(payload['round_id'])}",
            call_type="schema_smoke_candidate_generation" if bool(config.get("dry_run")) else "candidate_generation",
            reason=f"round{int(payload['round_id'])} generation failure",
            model=str(llm_result.get("model") or model_name),
            provider="openai",
            base_url=str(config.get("base_url") or ""),
            payload=payload,
            payload_hash=payload_hash,
            usage=deepcopy(llm_result.get("usage") or {}),
            request_id=extract_request_id(llm_result) if llm_result else None,
            success=False,
            cache_hit=False,
            event_state="error",
            error_type=type(exc).__name__,
            metadata={
                "baseline_name": BASELINE_NAME,
                "prompt_path": str(prompt_path),
                "prompt_version": prompt_version,
                "cache_namespace_version": CACHE_NAMESPACE_VERSION,
                "raw_text": raw_text,
                "parsed": parsed,
                "validation_errors": validation_errors,
                "deterministic_fallback_allowed": False,
            },
        )
        raise


def _recommended_interval(budget_steps: int) -> int:
    if budget_steps <= 5000:
        return max(200, budget_steps // 5)
    return min(50000, max(1000, budget_steps // 5))


def build_train_workflow_spec(config: Dict[str, Any], *, budget_steps: int, candidate: Dict[str, Any], phase_name: str, full_training: bool = False) -> Dict[str, Any]:
    workflow_spec = deepcopy(build_default_workflow_spec(1, reward_paradigm="pbrs"))
    workflow_spec.pop("llm_workflow", None)
    workflow_spec.pop("reward_spec_change_budget", None)
    workflow_spec.pop("pbrs_validation", None)
    workflow_spec["single_llm_baseline_spec"] = {
        "baseline_name": BASELINE_NAME,
        "paper_name": PAPER_NAME,
        "single_llm_only": True,
        "critic_enabled": False,
        "policy_guidance_enabled": False,
        "stage3_enabled": False,
        "adaptive_replacement_enabled": False,
        "winner_selection_mode": "rule_based_priority",
        "llm_workflow_scaffold_unused": True,
    }
    train_spec = workflow_spec["train"]
    train_spec["config"] = str(config.get("train_config") or DEFAULT_ALGORITHM)
    train_spec["seed"] = int(config.get("seed") or DEFAULT_SEED)
    train_spec["env_args"]["key"] = str(config.get("env_key") or DEFAULT_ENV_KEY)
    train_spec["env_args"]["time_limit"] = int(config.get("time_limit") or 50)
    train_spec["use_dense_reward"] = False
    train_spec["apply_dense_reward_in_eval"] = False
    train_spec["overrides"]["t_max"] = int(budget_steps)
    train_spec["overrides"]["test_interval"] = _recommended_interval(int(budget_steps))
    train_spec["overrides"]["runner_log_interval"] = _recommended_interval(int(budget_steps))
    train_spec["overrides"]["learner_log_interval"] = _recommended_interval(int(budget_steps))
    train_spec["overrides"]["use_cuda"] = bool(config.get("use_cuda", False))
    train_spec["checkpointing"] = {
        "enabled": False,
        "save_model_interval": 0,
        "save_final_model": False,
        "checkpoint_path": "",
        "load_step": 0,
        "local_results_path": str(config.get("local_results_path") or "results"),
    }
    train_spec["env_args"].update(
        {
            "use_pbrs": True,
            "eval_use_pbrs": False,
            "pbrs_beta": float(candidate["beta"]),
            "pbrs_wc": float(candidate["wc"]),
            "pbrs_wp": float(candidate["wp"]),
            "pbrs_version": "lbf_pbrs_v2",
            "pbrs_mode": str(candidate["pbrs_mode"]),
            "pbrs_active_terms": deepcopy(candidate["active_terms"]),
            "pbrs_weights": deepcopy(candidate["weights"]),
            "pbrs_gamma": float(candidate.get("gamma") or 0.99),
            "pbrs_variant": "original",
        }
    )
    workflow_spec["workflow_id"] = str(config["workflow_id"])
    workflow_spec["experiment_metadata"] = {
        "experiment_family": "single_llm_pbrs_v2_baseline",
        "budget_profile": "tiny_smoke" if bool(config.get("tiny_smoke")) else str(config.get("budget_profile") or "formal"),
        "seed": int(config.get("seed") or DEFAULT_SEED),
        "baseline_name": BASELINE_NAME,
        "paper_name": PAPER_NAME,
        "full_training": bool(full_training),
        "from_scratch": True,
    }
    workflow_spec["base_policy_guidance_spec"] = _disabled_policy_guidance_config()
    return workflow_spec


def build_training_plan(
    config: Dict[str, Any],
    *,
    round_id: int,
    candidate: Dict[str, Any],
    budget_steps: int,
    phase_name: str,
    label_suffix: str,
) -> Dict[str, Any]:
    workflow_spec = build_train_workflow_spec(
        config,
        budget_steps=budget_steps,
        candidate=candidate,
        phase_name=phase_name,
        full_training=(phase_name == "single_llm_full_training"),
    )
    launcher = EPyMARLTrainLauncher(
        repo_root=ROOT,
        python_executable=str(config.get("python_executable") or "python"),
    )
    plan = launcher.build_training_plan(
        round_id=round_id,
        workflow_spec=workflow_spec,
        reward_module_path="",
        alpha_policy=None,
        policy_guidance_spec=_disabled_policy_guidance_config(),
        workflow_id=str(config["workflow_id"]),
        candidate_id=str(candidate["candidate_id"]),
        reward_paradigm="pbrs",
        active_pbrs_field=None,
        active_pbrs_field_source=None,
        candidate_value=None,
        active_checkpoint_context=None,
        active_field_carryover_context=None,
        candidate_selection_context={
            "baseline_name": BASELINE_NAME,
            "round_id": round_id,
            "decision_source": "single_llm_baseline_rule_based" if phase_name == "single_llm_full_training" else "candidate_evaluation",
        },
        phase_name=phase_name,
        train_overrides_override=None,
        label_override=f"{config['workflow_id']}_{candidate['candidate_id']}{label_suffix}",
    )
    return {
        "workflow_spec": workflow_spec,
        "train_config": deepcopy(plan["train_config"]),
        "command": list(plan["command"]),
    }


def _load_metric_series_with_fallback(run_dir: str | Path) -> Tuple[Dict[str, Any], str]:
    primary = load_metric_series(run_dir, metric_name="test_sparse_return_mean")
    if primary["steps"] and primary["values"]:
        return primary, "test_sparse_return_mean"
    fallback = load_metric_series(run_dir, metric_name="test_return_mean")
    return fallback, "test_return_mean" if fallback["steps"] and fallback["values"] else "missing"


def summarize_candidate_run(result: Dict[str, Any], *, candidate_budget_steps: int) -> Dict[str, Any]:
    run_reference = deepcopy(result.get("run_reference") or {})
    run_dir = str(run_reference.get("run_dir") or "")
    series, metric_source = _load_metric_series_with_fallback(run_dir)
    summary = series_summary(series, last_k=5)
    auc_raw = summary.get("auc")
    auc_mean = None if auc_raw is None else float(auc_raw) / float(max(1, int(candidate_budget_steps)))
    return {
        "candidate_id": str((result.get("train_config") or {}).get("label") or ""),
        "run_reference": run_reference,
        "metric_source": metric_source,
        "series_summary": summary,
        "short_run_auc_mean": auc_mean,
        "short_run_auc_raw": auc_raw,
        "last_k_mean_test_sparse_return": summary.get("last_5_mean"),
        "max_test_sparse_return": summary.get("best"),
        "final_test_sparse_return": summary.get("final"),
        "metrics_summary": deepcopy(result.get("metrics_summary") or {}),
    }


def rule_based_winner_selection(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def _score(item: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return (
            float(item.get("short_run_auc_mean")) if item.get("short_run_auc_mean") is not None else float("-inf"),
            float(item.get("last_k_mean_test_sparse_return")) if item.get("last_k_mean_test_sparse_return") is not None else float("-inf"),
            float(item.get("max_test_sparse_return")) if item.get("max_test_sparse_return") is not None else float("-inf"),
            float(item.get("final_test_sparse_return")) if item.get("final_test_sparse_return") is not None else float("-inf"),
        )

    ranked = sorted(list(results or []), key=_score, reverse=True)
    winner = ranked[0] if ranked else {}
    return {
        "decision_source": "rule_based_priority",
        "metric_priority": [
            "short_run_auc_mean",
            "last_k_mean_test_sparse_return",
            "max_test_sparse_return",
            "final_test_sparse_return",
        ],
        "ranked_candidates": [
            {
                "candidate_id": str(item.get("candidate_id") or ""),
                "short_run_auc_mean": item.get("short_run_auc_mean"),
                "last_k_mean_test_sparse_return": item.get("last_k_mean_test_sparse_return"),
                "max_test_sparse_return": item.get("max_test_sparse_return"),
                "final_test_sparse_return": item.get("final_test_sparse_return"),
            }
            for item in ranked
        ],
        "winner_candidate_id": str(winner.get("candidate_id") or ""),
    }


def _candidate_result_entry(candidate: Dict[str, Any], result: Dict[str, Any], *, candidate_budget_steps: int) -> Dict[str, Any]:
    summary = summarize_candidate_run(result, candidate_budget_steps=candidate_budget_steps)
    summary["candidate_id"] = str(candidate["candidate_id"])
    summary["candidate"] = deepcopy(candidate)
    return summary


def _mock_tiny_training_result(plan: Dict[str, Any], candidate: Dict[str, Any], *, ordinal: int, round_id: int) -> Dict[str, Any]:
    pseudo_run_dir = (
        ROOT
        / "results"
        / "diagnostics"
        / "single_llm_pbrs_v2_mock_runs"
        / f"round{round_id}_{ordinal}_{candidate['candidate_id']}"
    )
    pseudo_run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "train_config": deepcopy(plan["train_config"]),
        "run_reference": {
            "run_dir": str(pseudo_run_dir),
            "run_id": 1000 + round_id * 10 + ordinal,
            "metrics_json": str(pseudo_run_dir / "metrics.json"),
            "info_json": str(pseudo_run_dir / "info.json"),
            "config_json": str(pseudo_run_dir / "config.json"),
        },
        "metrics_summary": {
            "metric_summary": {
                "test_sparse_return_mean": {
                    "last_value": 0.1 + 0.02 * ordinal + 0.03 * round_id,
                    "best_value": 0.12 + 0.02 * ordinal + 0.03 * round_id,
                }
            }
        },
    }


def _mock_tiny_summary(candidate: Dict[str, Any], *, ordinal: int, round_id: int, candidate_budget_steps: int, plan: Dict[str, Any]) -> Dict[str, Any]:
    base = 0.10 + 0.02 * ordinal + 0.03 * round_id
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate": deepcopy(candidate),
        "run_reference": deepcopy((_mock_tiny_training_result(plan, candidate, ordinal=ordinal, round_id=round_id)).get("run_reference") or {}),
        "metric_source": "test_sparse_return_mean",
        "series_summary": {
            "num_points": 5,
            "final": round(base, 6),
            "best": round(base + 0.03, 6),
            "last_5_mean": round(base - 0.005, 6),
            "auc": float(candidate_budget_steps) * float(base - 0.01),
        },
        "short_run_auc_mean": round(base - 0.01, 6),
        "short_run_auc_raw": float(candidate_budget_steps) * float(base - 0.01),
        "last_k_mean_test_sparse_return": round(base - 0.005, 6),
        "max_test_sparse_return": round(base + 0.03, 6),
        "final_test_sparse_return": round(base, 6),
        "metrics_summary": {"mock": True},
        "train_config": deepcopy(plan["train_config"]),
    }


def run_candidate_round(
    config: Dict[str, Any],
    *,
    round_id: int,
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    plans: List[Dict[str, Any]] = []
    candidate_budget_steps = int(config.get("candidate_budget_steps") or DEFAULT_CANDIDATE_BUDGET)
    tiny_smoke = bool(config.get("tiny_smoke"))
    execute = bool(config.get("execute"))
    launcher = EPyMARLTrainLauncher(
        repo_root=ROOT,
        python_executable=str(config.get("python_executable") or "python"),
    )

    for ordinal, candidate in enumerate(list(candidates or []), start=1):
        plan = build_training_plan(
            config,
            round_id=round_id,
            candidate=candidate,
            budget_steps=candidate_budget_steps,
            phase_name=f"single_llm_round{round_id}_candidate",
            label_suffix=f"_round{round_id}_candidate",
        )
        plans.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "train_config": deepcopy(plan["train_config"]),
                "command": list(plan["command"]),
            }
        )
        if not execute and not tiny_smoke:
            continue
        result = launcher.execute_training_plan(
            {
                "train_config": deepcopy(plan["train_config"]),
                "command": list(plan["command"]),
            }
        )
        results.append(_candidate_result_entry(candidate, result, candidate_budget_steps=candidate_budget_steps))
    return {
        "round_id": round_id,
        "candidate_budget_steps": candidate_budget_steps,
        "executed_training": bool(execute) or bool(tiny_smoke),
        "plans": plans,
        "results": results,
    }


def build_full_training_plan(config: Dict[str, Any], winner_config: Dict[str, Any]) -> Dict[str, Any]:
    plan = build_training_plan(
        config,
        round_id=99,
        candidate=winner_config,
        budget_steps=int(config.get("full_training_t_max") or DEFAULT_FULL_T_MAX),
        phase_name="single_llm_full_training",
        label_suffix="_full_training_from_scratch",
    )
    plan["from_scratch"] = True
    plan["load_step"] = 0
    plan["checkpoint_path"] = ""
    plan["eval_use_pbrs"] = False
    plan["test_sparse_only"] = True
    return plan


def _resolve_source_winner_workflow_dir(config: Dict[str, Any]) -> Path:
    raw_value = str(config.get("source_winner_workflow_dir") or "").strip()
    if not raw_value:
        raise ValueError("winner-only formal mode requires --source-winner-workflow-dir")
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(f"source winner workflow dir not found: {resolved}")
    return resolved


def _load_required_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")
    return load_json(path)


def _validate_clean_winner_config(candidate: Dict[str, Any]) -> Dict[str, Any]:
    errors = _candidate_validation_errors(candidate, index=0)
    if errors:
        raise ValueError("winner config is not clean lbf_pbrs_v2: " + " | ".join(errors))
    sanitized = sanitize_single_llm_candidate(candidate, index=0, require_candidate_type=True)
    if str(sanitized.get("pbrs_version") or "") != "lbf_pbrs_v2":
        raise ValueError("winner config must use pbrs_version=lbf_pbrs_v2")
    if str(sanitized.get("pbrs_mode") or "") not in _allowed_runtime_modes():
        raise ValueError(f"winner config pbrs_mode is invalid: {sanitized.get('pbrs_mode')}")
    return sanitized


def _candidate_has_required_clean_fields(candidate: Dict[str, Any]) -> bool:
    weights = candidate.get("weights")
    return (
        str(candidate.get("pbrs_version") or "").strip() == "lbf_pbrs_v2"
        and bool(str(candidate.get("pbrs_mode") or "").strip())
        and candidate.get("beta") is not None
        and isinstance(candidate.get("active_terms"), list)
        and isinstance(weights, dict)
        and all(term in weights for term in TERM_NAMES)
        and bool(str(candidate.get("candidate_type") or "").strip())
    )


def _build_real_llm_schema_smoke_retry_summary(
    *,
    config: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    generation: Dict[str, Any],
    code_modified: bool,
    files_modified: Sequence[str],
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    candidate_list = list(candidates or [])
    workflow_dir = resolve_paths(Path(str(config["results_root"])), str(config["workflow_id"])).workflow_dir
    workflow_ledger_path = workflow_dir / "llm_api_call_ledger.jsonl"
    all_candidates_have_candidate_type = all(bool(str(item.get("candidate_type") or "").strip()) for item in candidate_list)
    all_candidates_clean_lbf_pbrs_v2 = all(
        _candidate_has_required_clean_fields(item) and not _candidate_validation_errors(item, index=index)
        for index, item in enumerate(candidate_list)
    )
    return {
        "schema_smoke_retry_completed": True,
        "training_started": False,
        "real_llm_called": bool(generation.get("llm_used")),
        "baseline_is_single_llm_only": True,
        "critic_disabled": True,
        "policy_guidance_disabled": True,
        "stage3_disabled": True,
        "deterministic_fallback_used": False,
        "candidate_count": len(candidate_list),
        "candidate_type_required": True,
        "all_candidates_have_candidate_type": all_candidates_have_candidate_type,
        "all_candidates_clean_lbf_pbrs_v2": all_candidates_clean_lbf_pbrs_v2,
        "legacy_beta_wc_wp_contamination": False,
        "cache_isolated_from_main_method": str((generation.get("cache_path") or "")).startswith(str(workflow_dir)),
        "ledger_recorded": workflow_ledger_path.exists(),
        "safe_for_tiny_real_llm_execute_smoke": (
            bool(generation.get("llm_used"))
            and len(candidate_list) == DEFAULT_CANDIDATES_PER_ROUND
            and all_candidates_have_candidate_type
            and all_candidates_clean_lbf_pbrs_v2
        ),
        "code_modified": bool(code_modified),
        "files_modified": list(files_modified),
        "warnings": list(warnings or []),
    }


def _winner_only_formal_metrics_contract() -> List[str]:
    return [
        "pbrs_v2_runtime_used_mean",
        "pbrs_weights__col_mean",
        "pbrs_weights__app_mean",
        "pbrs_weights__cov_mean",
        "pbrs_weights__ready_mean",
        "pbrs_weights__alloc_mean",
        "pbrs_weights__stab_mean",
        "test_sparse_return_mean or equivalent sparse test metric",
    ]


def _run_winner_only_formal_mode(config: Dict[str, Any], paths: BaselinePaths, manifest: Dict[str, Any]) -> Dict[str, Any]:
    source_workflow_dir = _resolve_source_winner_workflow_dir(config)
    winner_selection = _load_required_json(source_workflow_dir / "winner_selection.json")
    winner_candidate_id = str(winner_selection.get("winner_candidate_id") or "").strip()
    if not winner_candidate_id:
        raise ValueError("winner_selection.json does not contain winner_candidate_id")
    winner_config = _validate_clean_winner_config(
        _load_required_json(source_workflow_dir / "winner_fixed_config.json")
    )
    if str(winner_config.get("candidate_id") or "").strip() != winner_candidate_id:
        raise ValueError(
            "winner_fixed_config.json candidate_id does not match winner_selection.json winner_candidate_id"
        )

    cfg = deepcopy(config)
    cfg["full_training_t_max"] = int(cfg.get("full_training_t_max") or DEFAULT_FULL_T_MAX)
    cfg["use_real_llm"] = False
    cfg["candidate_rounds"] = 0
    cfg["candidates_per_round"] = 0
    manifest["mode"]["use_real_llm"] = False
    manifest["mode"]["winner_only_formal"] = True
    manifest["config_summary"]["candidate_rounds"] = 0
    manifest["config_summary"]["candidates_per_round"] = 0
    manifest["config_summary"]["full_training_t_max"] = int(cfg["full_training_t_max"])
    manifest["status"] = "winner_only_formal_planned"
    manifest["source_winner_workflow_dir"] = str(source_workflow_dir)
    manifest["warnings"] = list(manifest.get("warnings") or [])

    save_json(paths.manifest, manifest)
    save_json(
        paths.baseline_state,
        {
            "workflow_id": cfg["workflow_id"],
            "status": "winner_only_formal_planned",
            "winner_candidate_id": winner_candidate_id,
            "source_winner_workflow_dir": str(source_workflow_dir),
            "single_llm_history": [],
        },
    )
    save_json(paths.winner_selection, winner_selection)
    save_json(paths.winner_fixed_config, winner_config)

    full_training_plan = build_full_training_plan(cfg, winner_config)
    save_json(paths.full_training_plan, full_training_plan)
    executed = False
    full_result_payload: Dict[str, Any] = {
        "executed": False,
        "from_scratch": True,
        "planned_only": True,
        "metrics_contract": _winner_only_formal_metrics_contract(),
    }
    if cfg.get("execute") and not cfg.get("dry_run"):
        launcher = EPyMARLTrainLauncher(
            repo_root=ROOT,
            python_executable=str(cfg.get("python_executable") or "python"),
        )
        result = launcher.execute_training_plan(
            {
                "train_config": deepcopy(full_training_plan["train_config"]),
                "command": list(full_training_plan["command"]),
            }
        )
        full_result_payload = {
            "executed": True,
            "from_scratch": True,
            "planned_only": False,
            "result": result,
            "metrics_contract": _winner_only_formal_metrics_contract(),
        }
        executed = True
        manifest["status"] = "completed"
    save_json(paths.full_training_result, full_result_payload)

    metrics_summary = {
        "workflow_id": str(cfg["workflow_id"]),
        "baseline_name": BASELINE_NAME,
        "paper_name": PAPER_NAME,
        "winner_candidate_id": winner_candidate_id,
        "rule_based_winner_selection": True,
        "winner_only_formal_mode": True,
        "candidate_generation_skipped": True,
        "candidate_training_skipped": True,
        "formal_full_training_t_max": int(cfg["full_training_t_max"]),
        "would_start_only_winner_full_training": True,
        "would_call_llm": False,
        "would_rerun_candidates": False,
        "metrics_contract": _winner_only_formal_metrics_contract(),
    }
    save_json(paths.metrics_summary, metrics_summary)
    save_json(paths.manifest, manifest)
    return {
        "workflow_id": str(cfg["workflow_id"]),
        "workflow_dir": str(paths.workflow_dir),
        "manifest": manifest,
        "winner_selection": winner_selection,
        "winner_fixed_config": winner_config,
        "full_training_plan": full_training_plan,
        "full_training_result": full_result_payload,
        "metrics_summary": metrics_summary,
        "winner_only_formal": {
            "candidate_generation_skipped": True,
            "candidate_training_skipped": True,
            "would_call_llm": False,
            "would_rerun_candidates": False,
            "would_start_only_winner_full_training": True,
            "would_append_no_new_llm_ledger_entries": True,
            "winner_selection_artifact_loaded": True,
            "winner_config_clean_lbf_pbrs_v2": True,
            "source_winner_workflow_dir": str(source_workflow_dir),
            "executed": executed,
        },
    }


def build_metrics_summary(
    *,
    config: Dict[str, Any],
    manifest: Dict[str, Any],
    round1_results: Sequence[Dict[str, Any]],
    round2_results: Sequence[Dict[str, Any]],
    winner_selection: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "workflow_id": str(config["workflow_id"]),
        "baseline_name": BASELINE_NAME,
        "paper_name": PAPER_NAME,
        "candidate_rounds": int(config.get("candidate_rounds") or DEFAULT_CANDIDATE_ROUNDS),
        "candidates_per_round": int(config.get("candidates_per_round") or DEFAULT_CANDIDATES_PER_ROUND),
        "round1_result_count": len(list(round1_results or [])),
        "round2_result_count": len(list(round2_results or [])),
        "winner_candidate_id": str(winner_selection.get("winner_candidate_id") or ""),
        "rule_based_winner_selection": True,
        "llm_roles_used": list(manifest.get("llm_roles_used") or []),
        "contamination_flags": deepcopy(manifest.get("contamination_flags") or {}),
    }


def build_manifest(config: Dict[str, Any], sparse_context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "workflow_id": str(config["workflow_id"]),
        "workflow_profile": str(config.get("workflow_profile") or config.get("profile_name") or ""),
        "baseline_name": BASELINE_NAME,
        "paper_name": PAPER_NAME,
        "status": "planned",
        "mode": {
            "dry_run": bool(config.get("dry_run")),
            "tiny_smoke": bool(config.get("tiny_smoke")),
            "use_real_llm": bool(config.get("use_real_llm")),
            "execute": bool(config.get("execute")),
            "winner_only_formal": bool(config.get("winner_only_formal")),
        },
        "config_summary": {
            "env_key": str(config.get("env_key") or DEFAULT_ENV_KEY),
            "train_config": str(config.get("train_config") or DEFAULT_ALGORITHM),
            "seed": int(config.get("seed") or DEFAULT_SEED),
            "candidate_rounds": int(config.get("candidate_rounds") or DEFAULT_CANDIDATE_ROUNDS),
            "candidates_per_round": int(config.get("candidates_per_round") or DEFAULT_CANDIDATES_PER_ROUND),
            "candidate_budget_steps": int(config.get("candidate_budget_steps") or DEFAULT_CANDIDATE_BUDGET),
            "full_training_t_max": int(config.get("full_training_t_max") or DEFAULT_FULL_T_MAX),
            "eval_use_pbrs": False,
            "test_sparse_only": True,
        },
        "isolation_guards": {
            "critic_enabled": False,
            "policy_guidance_enabled": False,
            "stage3_enabled": False,
            "adaptive_replacement_enabled": False,
            "no_change_gate_used": False,
            "winner_promotion_used": False,
            "final_staged_adaptive_spec_used": False,
        },
        "contamination_flags": {
            "policy_guidance_contamination_detected": False,
            "dual_llm_critic_contamination_detected": False,
            "stage3_contamination_detected": False,
        },
        "llm_roles_used": [],
        "sparse_baseline_context": deepcopy(sparse_context),
        "warnings": [],
    }


def run_single_llm_pbrs_v2_baseline(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(default_single_llm_baseline_profile())
    cfg.update(deepcopy(config or {}))
    cfg.setdefault("workflow_id", "qmix_single_llm_pbrs_v2_reward_gen_seed1_run1")
    cfg.setdefault("results_root", str(DEFAULT_RESULTS_ROOT))
    cfg.setdefault("candidate_rounds", DEFAULT_CANDIDATE_ROUNDS)
    cfg.setdefault("candidates_per_round", DEFAULT_CANDIDATES_PER_ROUND)
    cfg.setdefault("candidate_budget_steps", DEFAULT_CANDIDATE_BUDGET)
    cfg.setdefault("full_training_t_max", DEFAULT_FULL_T_MAX)
    cfg["execute"] = bool(cfg.get("execute", False))
    cfg["dry_run"] = bool(cfg.get("dry_run", False))
    cfg["tiny_smoke"] = bool(cfg.get("tiny_smoke", False))
    cfg["use_real_llm"] = bool(cfg.get("use_real_llm", False))
    cfg["winner_only_formal"] = bool(cfg.get("winner_only_formal", False))

    paths = resolve_paths(Path(str(cfg["results_root"])), str(cfg["workflow_id"]))
    paths.workflow_dir.mkdir(parents=True, exist_ok=True)

    sparse_context = build_sparse_baseline_context(cfg)
    manifest = build_manifest(cfg, sparse_context)
    round1_payload = build_prompt_payload(cfg, round_id=1, sparse_context=sparse_context)
    prompt_audit = _prompt_contamination_flags(round1_payload)
    if any(prompt_audit.values()):
        manifest["contamination_flags"]["policy_guidance_contamination_detected"] = bool(prompt_audit.get("policy_guidance"))
        manifest["contamination_flags"]["dual_llm_critic_contamination_detected"] = bool(prompt_audit.get("critic_diagnosis"))
        manifest["contamination_flags"]["stage3_contamination_detected"] = any(
            prompt_audit.get(key, False) for key in ("adaptive_replacement", "winner_promotion", "final_staged_adaptive_spec", "no_change")
        )

    save_json(paths.manifest, manifest)
    save_json(paths.baseline_state, {"workflow_id": cfg["workflow_id"], "status": "started", "single_llm_history": []})

    if cfg["winner_only_formal"]:
        return _run_winner_only_formal_mode(cfg, paths, manifest)

    if cfg["dry_run"] and not cfg["use_real_llm"]:
        round1_generation = {
            "round_id": 1,
            "generator_role": "single_llm_generator",
            "llm_used": False,
            "dry_run": True,
            "prompt_payload": deepcopy(round1_payload),
            "prompt_path": str(ROUND1_PROMPT_PATH),
            "planned_candidate_schema": _default_candidate_schema(),
        }
        round2_generation = {
            "round_id": 2,
            "generator_role": "single_llm_generator",
            "llm_used": False,
            "dry_run": True,
            "prompt_path": str(ROUND2_PROMPT_PATH),
            "planned_candidate_schema": _default_candidate_schema(),
        }
        save_json(paths.round1_generation, round1_generation)
        save_json(paths.round1_candidates, {"round_id": 1, "candidates": [], "planned_only": True})
        save_json(paths.round1_results, {"round_id": 1, "results": [], "planned_only": True})
        save_json(paths.round2_generation, round2_generation)
        save_json(paths.round2_candidates, {"round_id": 2, "candidates": [], "planned_only": True})
        save_json(paths.round2_results, {"round_id": 2, "results": [], "planned_only": True})
        save_json(paths.winner_selection, {"decision_source": "rule_based_priority", "planned_only": True})
        save_json(paths.winner_fixed_config, {"planned_only": True})
        save_json(paths.full_training_plan, {"planned_only": True, "from_scratch": True, "eval_use_pbrs": False, "test_sparse_only": True})
        metrics_summary = build_metrics_summary(
            config=cfg,
            manifest=manifest,
            round1_results=[],
            round2_results=[],
            winner_selection={"winner_candidate_id": ""},
        )
        save_json(paths.metrics_summary, metrics_summary)
        manifest["status"] = "dry_run_completed"
        save_json(paths.manifest, manifest)
        return {
            "workflow_id": cfg["workflow_id"],
            "workflow_dir": str(paths.workflow_dir),
            "manifest": manifest,
            "metrics_summary": metrics_summary,
        }

    round1_generation = (
        call_real_llm_generation(
            config=cfg,
            payload=round1_payload,
            prompt_path=ROUND1_PROMPT_PATH,
            cache_dir=paths.llm_cache_dir,
            workflow_dir=paths.workflow_dir,
        )
        if cfg["use_real_llm"]
        else build_mock_generation(1, round1_payload)
    )
    manifest["llm_roles_used"].append("single_llm_generator")
    round1_candidates = sanitize_candidate_batch(list((round1_generation.get("parsed") or {}).get("candidates") or []), round_id=1)
    save_json(paths.round1_generation, round1_generation)
    save_json(paths.round1_candidates, {"round_id": 1, "candidates": round1_candidates})

    if cfg["dry_run"] and cfg["use_real_llm"] and not cfg["tiny_smoke"] and not cfg["execute"]:
        retry_summary = _build_real_llm_schema_smoke_retry_summary(
            config=cfg,
            candidates=round1_candidates,
            generation=round1_generation,
            code_modified=True,
            files_modified=[
                "policy_method/src/workflows/prompts/single_llm_reward_generation_prompt.md",
                "policy_method/src/workflows/prompts/single_llm_reward_refinement_prompt.md",
                "policy_method/src/workflows/single_llm_pbrs_v2_baseline.py",
            ],
            warnings=[],
        )
        save_json(
            paths.real_llm_schema_smoke_summary,
            {
                "workflow_id": str(cfg["workflow_id"]),
                "baseline_name": BASELINE_NAME,
                "status": "ready_for_future_real_llm_schema_smoke",
                "artifact_path": str(paths.real_llm_schema_smoke_summary),
                "llm_cache_dir": str(paths.llm_cache_dir / BASELINE_NAME / str(cfg.get("model") or "gpt-5.2")),
                "ledger_enabled": True,
                "deterministic_fallback_allowed": False,
            },
        )
        save_json(REAL_LLM_SCHEMA_SMOKE_RETRY_SUMMARY_PATH, retry_summary)
        save_json(paths.round1_results, {"round_id": 1, "results": [], "planned_only": True})
        save_json(paths.round2_generation, {"round_id": 2, "skipped": True, "reason": "real_llm_dryrun_round1_only"})
        save_json(paths.round2_candidates, {"round_id": 2, "candidates": [], "planned_only": True})
        save_json(paths.round2_results, {"round_id": 2, "results": [], "planned_only": True})
        save_json(paths.winner_selection, {"decision_source": "rule_based_priority", "planned_only": True})
        save_json(paths.winner_fixed_config, {"planned_only": True})
        save_json(paths.full_training_plan, {"planned_only": True, "from_scratch": True, "eval_use_pbrs": False, "test_sparse_only": True})
        metrics_summary = build_metrics_summary(
            config=cfg,
            manifest=manifest,
            round1_results=[],
            round2_results=[],
            winner_selection={"winner_candidate_id": ""},
        )
        save_json(paths.metrics_summary, metrics_summary)
        manifest["status"] = "real_llm_dryrun_completed"
        save_json(paths.manifest, manifest)
        return {
            "workflow_id": cfg["workflow_id"],
            "workflow_dir": str(paths.workflow_dir),
            "manifest": manifest,
            "metrics_summary": metrics_summary,
        }

    round1_run = run_candidate_round(cfg, round_id=1, candidates=round1_candidates)
    save_json(paths.round1_results, round1_run)

    round2_payload = build_prompt_payload(
        cfg,
        round_id=2,
        sparse_context=sparse_context,
        previous_candidates=round1_candidates,
        previous_results=list(round1_run.get("results") or []),
    )
    round2_generation = (
        call_real_llm_generation(
            config=cfg,
            payload=round2_payload,
            prompt_path=ROUND2_PROMPT_PATH,
            cache_dir=paths.llm_cache_dir,
            workflow_dir=paths.workflow_dir,
        )
        if cfg["use_real_llm"] and cfg["execute"]
        else build_mock_generation(2, round2_payload)
    )
    round2_candidates = sanitize_candidate_batch(list((round2_generation.get("parsed") or {}).get("candidates") or []), round_id=2)
    save_json(paths.round2_generation, round2_generation)
    save_json(paths.round2_candidates, {"round_id": 2, "candidates": round2_candidates})

    round2_run = run_candidate_round(cfg, round_id=2, candidates=round2_candidates)
    save_json(paths.round2_results, round2_run)

    all_results = list(round1_run.get("results") or []) + list(round2_run.get("results") or [])
    winner_selection = rule_based_winner_selection(all_results)
    winner_id = str(winner_selection.get("winner_candidate_id") or "")
    winner_config = {}
    for item in all_results:
        if str(item.get("candidate_id") or "") == winner_id:
            winner_config = deepcopy(item.get("candidate") or {})
            break
    save_json(paths.winner_selection, winner_selection)
    save_json(paths.winner_fixed_config, winner_config or {"winner_candidate_id": winner_id})

    full_training_plan = build_full_training_plan(cfg, winner_config)
    save_json(paths.full_training_plan, full_training_plan)
    # Tiny execute smoke must stop after validating candidate generation, tiny training,
    # and winner planning. Formal winner training stays planned-only and from-scratch.
    if cfg["execute"] and not cfg["tiny_smoke"]:
        launcher = EPyMARLTrainLauncher(
            repo_root=ROOT,
            python_executable=str(cfg.get("python_executable") or "python"),
        )
        full_result = launcher.execute_training_plan(
            {
                "train_config": deepcopy(full_training_plan["train_config"]),
                "command": list(full_training_plan["command"]),
            }
        )
        save_json(paths.full_training_result, {"executed": True, "result": full_result, "from_scratch": True})
    else:
        save_json(paths.full_training_result, {"executed": False, "from_scratch": True})

    metrics_summary = build_metrics_summary(
        config=cfg,
        manifest=manifest,
        round1_results=list(round1_run.get("results") or []),
        round2_results=list(round2_run.get("results") or []),
        winner_selection=winner_selection,
    )
    save_json(paths.metrics_summary, metrics_summary)
    manifest["status"] = "completed"
    save_json(paths.manifest, manifest)
    save_json(
        paths.baseline_state,
        {
            "workflow_id": cfg["workflow_id"],
            "status": "completed",
            "single_llm_history": [
                {"round_id": 1, "role": "single_llm_generator"},
                {"round_id": 2, "role": "single_llm_generator"},
            ],
            "winner_candidate_id": winner_id,
        },
    )
    return {
        "workflow_id": cfg["workflow_id"],
        "workflow_dir": str(paths.workflow_dir),
        "manifest": manifest,
        "round1_results": round1_run,
        "round2_results": round2_run,
        "winner_selection": winner_selection,
        "metrics_summary": metrics_summary,
    }
