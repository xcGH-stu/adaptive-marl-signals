from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rewarding.lbf_pbrs_v2 import legacy_lbf_candidate_to_v2
from rewarding.rware_pbrs_v2 import normalize_rware_pbrs_v2_config
from workflows.clients.openai_backend import OpenAIChatBackend, probe_openai_backend

from .policy_critic_client import PolicyCriticClient
from .policy_behavior_collector import (
    build_synthetic_profile_traces,
    collect_policy_behavior_summary_from_traces,
)
from .policy_guidance_schema import (
    build_fallback_integrated_guidance_card,
    validate_integrated_guidance_card,
    validate_policy_guided_candidate_batch,
    validate_policy_behavior_summary,
)


POLICY_GUIDANCE_INTEGRATION_POINTS = [
    "stage1b_critic_sparse_diagnosis",
    "stage1b_generator_candidate_generation",
    "stage3_checkpoint_pre_diagnosis",
    "stage3_round1_result_analysis",
    "stage3_generator_round1_candidate_generation",
    "stage3_generator_round2_candidate_generation",
]

DEFAULT_REAL_LLM_DRYRUN_CACHE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "reports"
    / "llm_cache"
    / "policy_guided_real_llm_dryrun"
)
RWARE_STAGE1B_CANDIDATE_TYPES = [
    "requested_shelf_acquisition",
    "pickup_progress",
    "goal_progress",
    "traffic_conservative",
    "late_stability",
    "reference_like",
    "conservative",
]
RWARE_STAGE3_CANDIDATE_TYPES = [
    "goal_progress",
    "traffic_conservative",
    "late_stability",
    "reference_like",
    "conservative",
]


def _policy_guidance_env_family(env_key: Any = None, pbrs_version: Any = None) -> str:
    env_text = str(env_key or "").strip().lower()
    version_text = str(pbrs_version or "").strip().lower()
    if "rware" in env_text or version_text == "rware_pbrs_v2":
        return "rware"
    if "lbforaging" in env_text or env_text.startswith("lbf") or version_text == "lbf_pbrs_v2":
        return "lbf"
    return "unknown"


def _policy_guidance_search_space(*, env_family: str) -> Dict[str, Any]:
    if env_family == "rware":
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


def _build_rware_policy_guided_candidate(
    *,
    candidate_id: str,
    candidate_type: str,
    evidence_key: str,
    rationale: str,
    stage3_round2: bool = False,
) -> Dict[str, Any]:
    mode = "balanced_delivery_progress"
    beta = 0.30
    weights = {
        "shelf": 0.25,
        "pickup": 0.15,
        "goal": 0.35,
        "deliv": 0.15,
        "traffic": 0.10,
        "stab": 0.0,
    }
    normalized_type = str(candidate_type or "").strip().lower()
    if normalized_type in {"requested_shelf_acquisition", "shelf_acquisition"}:
        mode = "requested_shelf_acquisition"
        beta = 0.35
        weights = {"shelf": 0.45, "pickup": 0.25, "goal": 0.10, "deliv": 0.05, "traffic": 0.15, "stab": 0.0}
    elif normalized_type in {"pickup_progress", "pickup_recovery"}:
        mode = "requested_shelf_acquisition"
        beta = 0.40
        weights = {"shelf": 0.35, "pickup": 0.35, "goal": 0.10, "deliv": 0.05, "traffic": 0.15, "stab": 0.0}
    elif normalized_type in {"goal_progress", "carrying_to_goal", "progress_shift"}:
        mode = "carrying_to_goal"
        beta = 0.45 if not stage3_round2 else 0.40
        weights = {"shelf": 0.05, "pickup": 0.10, "goal": 0.50, "deliv": 0.20, "traffic": 0.15, "stab": 0.0}
    elif normalized_type in {"traffic_conservative", "coordination_recovery"}:
        mode = "traffic_conservative"
        beta = 0.20
        weights = {"shelf": 0.10, "pickup": 0.10, "goal": 0.20, "deliv": 0.10, "traffic": 0.40, "stab": 0.10}
    elif normalized_type in {"late_stability", "stability_recovery", "conservative"}:
        mode = "late_stability"
        beta = 0.20
        weights = {"shelf": 0.05, "pickup": 0.10, "goal": 0.25, "deliv": 0.25, "traffic": 0.15, "stab": 0.20}
    normalized = normalize_rware_pbrs_v2_config(
        {
            "pbrs_version": "rware_pbrs_v2",
            "mode": mode,
            "beta": beta,
            "weights": weights,
        }
    )
    return {
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "pbrs_version": "rware_pbrs_v2",
        "mode": str(normalized.get("mode") or mode),
        "beta": float(normalized.get("beta", beta)),
        "active_terms": list(normalized.get("active_terms") or []),
        "weights": deepcopy(normalized.get("weights") or {}),
        "evidence_keys_used": [evidence_key] if evidence_key else [],
        "evidence_key_used": evidence_key,
        "rationale": rationale,
    }


def _to_policy_guided_candidate(
    *,
    env_family: str,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    if env_family == "lbf":
        merged_candidate = legacy_lbf_candidate_to_v2(
            beta=candidate.get("beta", 0.5),
            wc=candidate.get("wc", 0.5),
            wp=candidate.get("wp"),
            candidate_id=candidate.get("candidate_id"),
            candidate_type=candidate.get("candidate_type"),
            evidence_keys_used=[candidate.get("evidence_key_used")]
            if candidate.get("evidence_key_used")
            else [],
        )
        merged_candidate.update(
            {
                "rationale": candidate.get("rationale"),
                "hypothesis": candidate.get("hypothesis"),
                "expected_early_effect": candidate.get("expected_early_effect"),
                "risk": candidate.get("risk"),
                "evidence_key_used": candidate.get("evidence_key_used", ""),
            }
        )
        return merged_candidate
    return candidate


def _build_policy_guidance_backend(
    *,
    use_real_llm: bool,
    api_key_env: str,
    base_url: str,
    model: str,
    temperature: float,
    llm_timeout: float,
    llm_max_retries: int,
    llm_retry_backoff: float,
) -> Tuple[OpenAIChatBackend | None, Dict[str, Any]]:
    probe = probe_openai_backend(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
    )
    if not use_real_llm:
        probe["backend_reason"] = "use_real_llm_disabled"
        return None, probe
    if not bool(probe.get("api_key_detected")):
        probe["backend_reason"] = "api_key_missing"
        return None, probe
    if not bool(probe.get("openai_import_available")):
        probe["backend_reason"] = "openai_import_failed"
        return None, probe
    try:
        backend = OpenAIChatBackend(
            api_key_env=api_key_env,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=llm_timeout,
            max_retries=max(1, int(llm_max_retries)),
            retry_backoff_seconds=llm_retry_backoff,
        )
    except Exception as exc:
        probe["backend_reason"] = "backend_init_failed"
        probe["backend_error"] = f"{exc.__class__.__name__}: {exc}"
        return None, probe
    probe["backend_reason"] = "backend_ready"
    return backend, probe


def build_critic_payload_with_integrated_guidance(
    base_payload: Dict[str, Any] | None,
    *,
    integrated_guidance_card: Dict[str, Any],
    behavior_summary: Dict[str, Any] | None = None,
    intervention_point: str,
) -> Dict[str, Any]:
    payload = deepcopy(base_payload or {})
    payload["policy_guidance_context"] = {
        "intervention_point": str(intervention_point),
        "integrated_guidance_card": validate_integrated_guidance_card(
            integrated_guidance_card
        ),
        "behavior_summary": deepcopy(behavior_summary or {}),
        "used_by_critic": True,
        "used_by_generator": False,
    }
    return payload


def build_generator_payload_with_integrated_guidance(
    base_payload: Dict[str, Any] | None,
    *,
    integrated_guidance_card: Dict[str, Any],
    intervention_point: str,
) -> Dict[str, Any]:
    payload = deepcopy(base_payload or {})
    payload["policy_guidance_context"] = {
        "intervention_point": str(intervention_point),
        "integrated_guidance_card": validate_integrated_guidance_card(
            integrated_guidance_card
        ),
        "used_by_critic": False,
        "used_by_generator": True,
    }
    return payload


def load_stage1_behavior_summary(summary_path: str | Path) -> Dict[str, Any]:
    path = Path(summary_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stage1 behavior summary artifact must be a dict")
    payload["_summary_path"] = str(path)
    return payload


def _milestone_label(target_step: int, milestones: List[int]) -> str:
    if not milestones:
        return f"step_{int(target_step)}"
    if int(target_step) == int(milestones[0]):
        return "early_milestone"
    if int(target_step) == int(milestones[-1]):
        return "late_milestone"
    return f"milestone_{int(target_step)}"


def _deterministic_stage1b_implications(milestones: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected = None
    for item in milestones:
        flags = ((item.get("behavior_summary") or {}).get("evidence_flags") or {})
        if any(flags.values()):
            selected = item
            break
    if selected is None:
        selected = milestones[-1]
    summary = selected.get("behavior_summary") or {}
    flags = summary.get("evidence_flags") or {}
    env_family = _policy_guidance_env_family(
        summary.get("env_key"),
        ((summary.get("metadata") or {}).get("pbrs_version")),
    )
    if env_family == "rware":
        candidate_types: List[str] = ["reference_like"]
        primary = "stable_baseline"
        if flags.get("requested_shelf_access_low"):
            primary = "requested_shelf_access_low"
            candidate_types.append("requested_shelf_acquisition")
        if flags.get("pickup_progress_weak"):
            primary = "pickup_progress_weak"
            candidate_types.append("pickup_progress")
        if flags.get("carrying_progress_weak") or flags.get("delivery_progress_weak"):
            primary = "delivery_progress_weak"
            candidate_types.append("goal_progress")
        if flags.get("traffic_blocking_high"):
            primary = "traffic_blocking_high"
            candidate_types.append("traffic_conservative")
        if flags.get("route_stability_low"):
            primary = "route_stability_low"
            candidate_types.append("late_stability")
        return {
            "early_primary_failure": primary,
            "recommended_candidate_types": list(dict.fromkeys(candidate_types)),
            "beta_search_prior": "unknown",
            "wc_search_prior": "unknown",
            "wp_search_prior": "unknown",
        }
    candidate_types: List[str] = ["reference_like"]
    beta = "same"
    wc = "same"
    wp = "same"
    primary = "stable_baseline"
    if flags.get("low_coverage") or flags.get("food_discovery_failure"):
        primary = "under_exploration"
        candidate_types.append("exploration_boost")
        beta = "up"
    if flags.get("near_food_collection_failure"):
        primary = "collection_failure"
        candidate_types.append("collection_recovery")
    if flags.get("over_concentration_possible"):
        primary = "coordination_congestion"
        candidate_types.append("wc_down_wp_up")
        wc = "down"
        wp = "up"
    return {
        "early_primary_failure": primary,
        "recommended_candidate_types": candidate_types,
        "beta_search_prior": beta,
        "wc_search_prior": wc,
        "wp_search_prior": wp,
    }


def generate_stage1_behavior_summary_artifact(
    *,
    workflow_dir: str | Path,
    workflow_id: str,
    env_key: str,
    algorithm: str,
    seed: int,
    milestone_steps: List[int],
    eval_episodes_per_milestone: int,
    source_run_ref: Dict[str, Any] | None = None,
    force_rewrite: bool = False,
) -> Dict[str, Any]:
    workflow_path = Path(workflow_dir).expanduser().resolve()
    summary_path = workflow_path / "stage1_sparse_policy_behavior_summary.json"
    if summary_path.exists() and not force_rewrite:
        existing = load_stage1_behavior_summary(summary_path)
        return {
            "artifact": existing,
            "summary_path": str(summary_path),
            "summary_exists": True,
            "workflow_local": True,
            "reused_prior_artifact": False,
            "generated": False,
            "workflow_id": str(
                (existing.get("metadata") or {}).get("workflow_id") or workflow_id
            ),
        }

    milestones: List[Dict[str, Any]] = []
    for target_step in [int(value) for value in milestone_steps]:
        label = _milestone_label(target_step, milestone_steps)
        traces = build_synthetic_profile_traces(
            profile="stage1_sparse",
            env_key=env_key,
            episode_count=max(1, int(eval_episodes_per_milestone)),
        )
        summary = collect_policy_behavior_summary_from_traces(
            env_key=env_key,
            intervention_point="stage1_after_sparse_baseline",
            episode_traces=traces,
            num_eval_episodes=max(1, int(eval_episodes_per_milestone)),
            stage_label=label,
            target_step=int(target_step),
            actual_step=int(target_step),
            source_run_ref=deepcopy(source_run_ref or {}),
        )
        summary.setdefault("metadata", {})
        summary["metadata"]["workflow_id"] = str(workflow_id)
        summary["metadata"]["stage1_behavior_summary_origin"] = "workflow_local_fresh_generate"
        milestones.append(
            {
                "target_step": int(target_step),
                "actual_step": int(target_step),
                "label": label,
                "num_eval_episodes": int(eval_episodes_per_milestone),
                "extraction_source": "synthetic_trace_collection",
                "behavior_summary": summary,
                "compact_episode_summaries": list(summary.get("episode_summaries") or []),
            }
        )

    artifact = {
        "stage": "stage1_sparse_baseline",
        "env_key": str(env_key),
        "algorithm": str(algorithm),
        "seed": int(seed),
        "workflow_id": str(workflow_id),
        "metadata": {
            "workflow_id": str(workflow_id),
            "stage1_behavior_summary_origin": "workflow_local_fresh_generate",
            "source_run_ref": deepcopy(source_run_ref or {}),
        },
        "milestones": milestones,
        "stage1b_implications": _deterministic_stage1b_implications(milestones),
    }
    summary_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "artifact": artifact,
        "summary_path": str(summary_path),
        "summary_exists": True,
        "workflow_local": True,
        "reused_prior_artifact": False,
        "generated": True,
        "workflow_id": str(workflow_id),
    }


def find_stage1_behavior_summary_path(
    *,
    explicit_path: str | Path | None = None,
    workflow_dir: str | Path | None = None,
    search_roots: List[str | Path] | None = None,
    allow_reuse_stage1_behavior_summary: bool = False,
    require_workflow_local: bool = False,
) -> Tuple[str | None, Dict[str, Any]]:
    candidates: List[Path] = []
    metadata = {
        "workflow_local": False,
        "reused_prior_artifact": False,
        "found_via": None,
    }
    workflow_path = (
        Path(workflow_dir).expanduser().resolve() if workflow_dir is not None else None
    )
    if explicit_path not in (None, ""):
        path = Path(str(explicit_path)).expanduser()
        candidates.append(path if path.is_absolute() else path.resolve())
    if workflow_path is not None:
        candidates.append(workflow_path / "stage1_sparse_policy_behavior_summary.json")
        candidates.extend(sorted(workflow_path.rglob("stage1_sparse_policy_behavior_summary.json")))
    if allow_reuse_stage1_behavior_summary and not require_workflow_local:
        for root in list(search_roots or []):
            root_path = Path(root).expanduser().resolve()
            candidates.append(root_path / "stage1_sparse_policy_behavior_summary.json")
            if root_path.exists():
                candidates.extend(sorted(root_path.rglob("stage1_sparse_policy_behavior_summary.json")))
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            workflow_local = bool(
                workflow_path is not None
                and (
                    resolved == (workflow_path / "stage1_sparse_policy_behavior_summary.json")
                    or workflow_path in resolved.parents
                )
            )
            if require_workflow_local and not workflow_local:
                continue
            metadata["workflow_local"] = workflow_local
            metadata["reused_prior_artifact"] = not workflow_local
            metadata["found_via"] = (
                "workflow_dir"
                if workflow_local
                else "search_roots"
            )
            return str(resolved), metadata
    return None, metadata


def select_stage1_behavior_milestone(
    artifact: Dict[str, Any],
    *,
    target_step: int = 800000,
) -> Dict[str, Any]:
    milestones = list(artifact.get("milestones") or [])
    if not milestones:
        raise ValueError("stage1 behavior summary has no milestones")
    def _distance(item: Dict[str, Any]) -> int:
        actual_step = int(item.get("actual_step") or item.get("target_step") or 0)
        return abs(actual_step - int(target_step))
    selected = min(milestones, key=_distance)
    selected["behavior_summary"] = validate_policy_behavior_summary(
        selected.get("behavior_summary")
    )
    return selected


def build_stage1b_critic_payload_with_policy_guidance(
    *,
    task_metadata: Dict[str, Any],
    sparse_baseline_metrics: Dict[str, Any],
    stage1_behavior_summary_path: str | Path,
    stage1_behavior_artifact: Dict[str, Any] | None = None,
    target_step: int = 800000,
) -> Dict[str, Any]:
    artifact = deepcopy(stage1_behavior_artifact) if stage1_behavior_artifact else load_stage1_behavior_summary(stage1_behavior_summary_path)
    selected = select_stage1_behavior_milestone(artifact, target_step=target_step)
    env_family = _policy_guidance_env_family(
        (task_metadata or {}).get("env_key"),
        (((selected.get("behavior_summary") or {}).get("metadata") or {}).get("pbrs_version")),
    )
    return {
        "task_metadata": deepcopy(task_metadata),
        "sparse_baseline_metrics": deepcopy(sparse_baseline_metrics),
        "stage1_behavior_summary_path": str(Path(stage1_behavior_summary_path).expanduser().resolve()),
        "stage1_behavior_summary_exists": True,
        "stage1_behavior_summary_stage": artifact.get("stage"),
        "stage1_behavior_summary_selected_milestone": {
            "target_step": int(selected.get("target_step") or 0),
            "actual_step": int(selected.get("actual_step") or 0),
            "label": str(selected.get("label") or ""),
            "extraction_source": str(selected.get("extraction_source") or ""),
        },
        "stage1_behavior_summary": deepcopy(selected.get("behavior_summary") or {}),
        "policy_guidance_target": {
            "stage1b_goal": "select early dense initialization near the 800k endpoint checkpoint",
            "pbrs_search_space": ["pbrs_version", "mode", "beta", "active_terms", "weights"],
            "env_family": env_family,
            "target_step": int(target_step),
        },
    }


def build_stage1b_generator_payload_with_policy_guidance(
    *,
    task_metadata: Dict[str, Any],
    sparse_baseline_metrics: Dict[str, Any],
    pbrs_search_space: Dict[str, Any],
    integrated_guidance_card: Dict[str, Any] | None,
    existing_reward_only_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = deepcopy(existing_reward_only_payload or {})
    env_family = _policy_guidance_env_family(
        (task_metadata or {}).get("env_key"),
        (((pbrs_search_space or {}).get("constraints") or {}).get("pbrs_version") or [""])[0],
    )
    validated_card = None
    if integrated_guidance_card:
        validated_card = validate_integrated_guidance_card(integrated_guidance_card)
    if validated_card is None or bool(validated_card.get("fallback_to_reward_only", False)):
        payload.setdefault("policy_guidance_used", False)
        payload.setdefault("policy_guidance_status", "reward_only_fallback")
        payload.setdefault("policy_guidance_source", "reward_only_fallback")
        payload.setdefault("policy_guidance_fallback_to_reward_only", True)
        payload.setdefault(
            "policy_guidance_failure_reason",
            str(((validated_card or {}).get("metadata") or {}).get("failure_reason") or "policy guidance unavailable"),
        )
        payload.setdefault("task_metadata", deepcopy(task_metadata))
        payload.setdefault("sparse_baseline_metrics", deepcopy(sparse_baseline_metrics))
        payload.setdefault("pbrs_search_space", deepcopy(pbrs_search_space))
        payload.setdefault("env_family", env_family)
        if env_family == "rware":
            payload.pop("integrated_guidance_card", None)
        return payload
    payload.update(
        {
            "task_metadata": deepcopy(task_metadata),
            "sparse_baseline_metrics": deepcopy(sparse_baseline_metrics),
            "pbrs_search_space": deepcopy(pbrs_search_space),
            "env_family": env_family,
            "integrated_guidance_card": validated_card,
            "policy_guidance_used": True,
            "policy_guidance_status": "integrated_guidance_attached",
            "policy_guidance_source": "stage1_behavior_summary_via_critic",
            "policy_guidance_fallback_to_reward_only": False,
            "policy_guidance_failure_reason": None,
            "policy_guidance_evidence_keys": list(
                ((validated_card.get("policy_diagnosis") or {}).get("behavior_evidence"))
                or []
            ),
        }
    )
    return payload


def derive_stage1b_reward_diagnosis(
    *,
    sparse_baseline_metrics: Dict[str, Any],
    stage1_behavior_artifact: Dict[str, Any],
    selected_milestone: Dict[str, Any],
) -> Dict[str, Any]:
    env_family = _policy_guidance_env_family(
        ((selected_milestone.get("behavior_summary") or {}).get("env_key")),
        (((selected_milestone.get("behavior_summary") or {}).get("metadata") or {}).get("pbrs_version")),
    )
    scalar_evidence: List[str] = []
    for key in (
        "best_test_sparse_return_mean",
        "final_test_sparse_return_mean",
        "train_return_mean",
        "sample_efficiency_status",
    ):
        if key in sparse_baseline_metrics:
            scalar_evidence.append(f"sparse_baseline_metrics.{key}={sparse_baseline_metrics.get(key)}")
    scalar_evidence.append(
        "stage1_sparse_policy_behavior_summary.milestones."
        f"{int(selected_milestone.get('target_step') or 0)}"
    )
    return {
        "learning_stage": "stage1_sparse_baseline_to_stage1b",
        "metric_evidence": scalar_evidence,
        "performance_issue": str(
            sparse_baseline_metrics.get("performance_issue")
            or "sparse baseline requires early dense initialization guidance"
        ),
        "search_objective": (
            "select an early dense checkpoint near step 800k and guide RWARE PBRS-v2 mode/active_terms/weights/beta search"
            if env_family == "rware"
            else "select an early dense checkpoint near step 800k and guide beta/wc/wp search"
        ),
        "stage1b_implications": deepcopy(stage1_behavior_artifact.get("stage1b_implications") or {}),
    }


def maybe_build_stage1b_integrated_guidance(
    *,
    task_metadata: Dict[str, Any],
    sparse_baseline_metrics: Dict[str, Any],
    policy_guidance_enabled: bool,
    stage1_behavior_summary_required: bool = False,
    stage1_behavior_summary_path: str | Path | None = None,
    workflow_dir: str | Path | None = None,
    search_roots: List[str | Path] | None = None,
    fallback_to_reward_only: bool = True,
    target_step: int = 800000,
    mock_llm_mode: bool = True,
    use_real_llm: bool = False,
    api_key_env: str = "IUSEAPI_API_KEY",
    base_url: str = "https://www.iuseapi.com/v1",
    model: str = "gpt-5.2",
    temperature: float = 0.2,
    llm_timeout: float = 60.0,
    llm_max_retries: int = 3,
    llm_retry_backoff: float = 5.0,
    reuse_policy_guidance_cache: bool = False,
    policy_guidance_cache_root: str | Path | None = None,
    policy_guidance_cache_lookup_mode: str = "exact_or_latest_by_intervention",
    allow_reuse_stage1_behavior_summary: bool = False,
    require_workflow_local_stage1_behavior_summary: bool = False,
) -> Dict[str, Any]:
    result = {
        "policy_guidance_enabled": bool(policy_guidance_enabled),
        "policy_guidance_used": False,
        "integrated_guidance_card": None,
        "policy_guidance_failure_reason": None,
        "stage1_behavior_summary_path": None,
        "stage1_behavior_summary_abs_path": None,
        "stage1_behavior_summary_exists": False,
        "stage1_behavior_summary_workflow_local": False,
        "stage1_behavior_summary_reused_from_prior_artifact": False,
        "stage1_behavior_summary_workflow_id": None,
        "fallback_to_reward_only": bool(fallback_to_reward_only),
        "integrated_guidance_card_path": None,
        "intervention_point": "stage1_after_sparse_baseline",
        "policy_guidance_source": "reward_only_fallback",
        "policy_guidance_evidence_keys": [],
        "selected_milestone": None,
        "stage1_behavior_artifact": None,
    }
    if not policy_guidance_enabled:
        result["policy_guidance_failure_reason"] = "policy guidance disabled"
        return result
    summary_path, summary_resolution = find_stage1_behavior_summary_path(
        explicit_path=stage1_behavior_summary_path,
        workflow_dir=workflow_dir,
        search_roots=search_roots,
        allow_reuse_stage1_behavior_summary=bool(allow_reuse_stage1_behavior_summary),
        require_workflow_local=bool(require_workflow_local_stage1_behavior_summary),
    )
    if summary_path is None:
        result["policy_guidance_failure_reason"] = "stage1 behavior summary not found"
        if stage1_behavior_summary_required:
            result["fallback_to_reward_only"] = True
        return result
    result["stage1_behavior_summary_path"] = str(stage1_behavior_summary_path or summary_path)
    result["stage1_behavior_summary_abs_path"] = str(Path(summary_path).resolve())
    result["stage1_behavior_summary_exists"] = True
    result["stage1_behavior_summary_workflow_local"] = bool(
        summary_resolution.get("workflow_local", False)
    )
    result["stage1_behavior_summary_reused_from_prior_artifact"] = bool(
        summary_resolution.get("reused_prior_artifact", False)
    )
    try:
        artifact = load_stage1_behavior_summary(summary_path)
        result["stage1_behavior_summary_workflow_id"] = str(
            artifact.get("workflow_id")
            or (artifact.get("metadata") or {}).get("workflow_id")
            or ""
        )
        selected_milestone = select_stage1_behavior_milestone(artifact, target_step=target_step)
        compact_episode_summaries = list(
            selected_milestone.get("compact_episode_summaries")
            or (selected_milestone.get("behavior_summary") or {}).get("episode_summaries")
            or []
        )
        if not compact_episode_summaries:
            raise ValueError("selected milestone has no compact episode summaries")
        reward_diagnosis = derive_stage1b_reward_diagnosis(
            sparse_baseline_metrics=sparse_baseline_metrics,
            stage1_behavior_artifact=artifact,
            selected_milestone=selected_milestone,
        )
        backend, backend_probe = _build_policy_guidance_backend(
            use_real_llm=use_real_llm and not bool(mock_llm_mode),
            api_key_env=api_key_env,
            base_url=base_url,
            model=model,
            temperature=temperature,
            llm_timeout=llm_timeout,
            llm_max_retries=llm_max_retries,
            llm_retry_backoff=llm_retry_backoff,
        )
        client = PolicyCriticClient(
            model=model,
            mock_llm_mode=bool(mock_llm_mode),
            fallback_to_reward_only=bool(fallback_to_reward_only),
            backend=backend,
            reuse_cache=bool(reuse_policy_guidance_cache),
            cache_root=Path(policy_guidance_cache_root).expanduser().resolve()
            if policy_guidance_cache_root not in (None, "")
            else DEFAULT_REAL_LLM_DRYRUN_CACHE_ROOT,
            cache_lookup_mode=str(policy_guidance_cache_lookup_mode or "exact_or_latest_by_intervention"),
            workflow_dir=Path(workflow_dir).resolve() if workflow_dir not in (None, "") else None,
            workflow_id=str(task_metadata.get("workflow_id") or ""),
            profile=str(task_metadata.get("profile") or ""),
            env_key=str(task_metadata.get("env_key") or ""),
            algorithm=str(task_metadata.get("algorithm") or ""),
        )
        critic_result = client.generate_stage1b_integrated_guidance(
            sparse_baseline_metrics=sparse_baseline_metrics,
            stage1_behavior_artifact=artifact,
            selected_behavior_summary=selected_milestone["behavior_summary"],
            selected_milestone=selected_milestone,
        )
        card = validate_integrated_guidance_card(critic_result["integrated_guidance_card"])
        evidence_keys = list((card.get("policy_diagnosis") or {}).get("behavior_evidence") or [])
        result.update(
            {
                "policy_guidance_used": not bool(card.get("fallback_to_reward_only", False)),
                "integrated_guidance_card": card,
                "policy_guidance_failure_reason": (
                    str((card.get("metadata") or {}).get("failure_reason") or "")
                    or None
                ),
                "policy_guidance_source": (
                    "stage1_behavior_summary_via_critic"
                    if not bool(card.get("fallback_to_reward_only", False))
                    else "reward_only_fallback"
                ),
                "policy_guidance_evidence_keys": evidence_keys,
                "selected_milestone": {
                    "target_step": int(selected_milestone.get("target_step") or 0),
                    "actual_step": int(selected_milestone.get("actual_step") or 0),
                    "label": str(selected_milestone.get("label") or ""),
                    "extraction_source": str(selected_milestone.get("extraction_source") or ""),
                },
                "stage1_behavior_artifact": artifact,
                "reward_diagnosis": reward_diagnosis,
                "call_record": deepcopy(critic_result.get("call_record") or {}),
                "backend_probe": backend_probe,
            }
        )
        return _normalize_policy_guidance_result(result)
    except Exception as exc:
        result["policy_guidance_failure_reason"] = str(exc)
        return _normalize_policy_guidance_result(result)


def _normalize_policy_guidance_result(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = deepcopy(result or {})
    card = payload.get("integrated_guidance_card")
    validated_card = None
    if isinstance(card, dict) and card:
        try:
            validated_card = validate_integrated_guidance_card(card)
        except Exception:
            validated_card = None
    if validated_card is not None:
        payload["integrated_guidance_card"] = validated_card
    card_fallback = bool((validated_card or {}).get("fallback_to_reward_only", False))
    guidance_used = bool(validated_card is not None and not card_fallback)
    failure_reason = payload.get("policy_guidance_failure_reason")
    if validated_card is not None:
        failure_reason = str(
            ((validated_card.get("metadata") or {}).get("failure_reason"))
            or failure_reason
            or ""
        ).strip()
    else:
        failure_reason = str(failure_reason or "").strip()

    payload["policy_guidance_used"] = guidance_used
    payload["fallback_to_reward_only"] = not guidance_used
    payload["policy_guidance_fallback_to_reward_only"] = not guidance_used
    if guidance_used:
        payload["policy_guidance_failure_reason"] = None
        if not str(payload.get("policy_guidance_source") or "").strip():
            payload["policy_guidance_source"] = "integrated_guidance_attached"
    else:
        payload["policy_guidance_failure_reason"] = failure_reason or "policy guidance unavailable"
        payload["policy_guidance_source"] = "reward_only_fallback"
    if validated_card is not None and guidance_used and not payload.get("policy_guidance_evidence_keys"):
        payload["policy_guidance_evidence_keys"] = list(
            ((validated_card.get("policy_diagnosis") or {}).get("behavior_evidence")) or []
        )
    payload.setdefault("policy_guidance_evidence_keys", [])
    return payload


def build_reward_only_stage1b_generator_payload(
    *,
    task_metadata: Dict[str, Any],
    sparse_baseline_metrics: Dict[str, Any],
    pbrs_search_space: Dict[str, Any],
) -> Dict[str, Any]:
    env_family = _policy_guidance_env_family(
        (task_metadata or {}).get("env_key"),
        (((pbrs_search_space or {}).get("constraints") or {}).get("pbrs_version") or [""])[0],
    )
    return {
        "task_metadata": deepcopy(task_metadata),
        "sparse_baseline_metrics": deepcopy(sparse_baseline_metrics),
        "pbrs_search_space": deepcopy(pbrs_search_space),
        "env_family": env_family,
        "policy_guidance_used": False,
        "policy_guidance_status": "reward_only_baseline",
        "policy_guidance_source": "reward_only_fallback",
        "policy_guidance_fallback_to_reward_only": True,
    }


def maybe_build_stage1b_generator_payload_with_policy_guidance(
    *,
    task_metadata: Dict[str, Any],
    sparse_baseline_metrics: Dict[str, Any],
    pbrs_search_space: Dict[str, Any],
    integrated_guidance_card: Dict[str, Any] | None,
    existing_reward_only_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if integrated_guidance_card is None:
        return build_reward_only_stage1b_generator_payload(
            task_metadata=task_metadata,
            sparse_baseline_metrics=sparse_baseline_metrics,
            pbrs_search_space=pbrs_search_space,
        )
    try:
        return build_stage1b_generator_payload_with_policy_guidance(
            task_metadata=task_metadata,
            sparse_baseline_metrics=sparse_baseline_metrics,
            pbrs_search_space=pbrs_search_space,
            integrated_guidance_card=integrated_guidance_card,
            existing_reward_only_payload=existing_reward_only_payload,
        )
    except Exception:
        fallback = build_fallback_integrated_guidance_card(
            intervention_point="stage1_after_sparse_baseline",
            failure_reason="generator payload builder fallback",
            reward_diagnosis={},
        )
        return build_stage1b_generator_payload_with_policy_guidance(
            task_metadata=task_metadata,
            sparse_baseline_metrics=sparse_baseline_metrics,
            pbrs_search_space=pbrs_search_space,
            integrated_guidance_card=fallback,
            existing_reward_only_payload=existing_reward_only_payload,
        )


def build_stage3_pre_checkpoint_policy_guidance(
    *,
    task_metadata: Dict[str, Any],
    workflow_dir: str | Path | None = None,
    checkpoint_id: str,
    checkpoint_step: int,
    current_pbrs_config: Dict[str, Any],
    reward_metrics: Dict[str, Any],
    behavior_summary: Dict[str, Any] | None,
    behavior_summary_path: str | Path | None = None,
    sparse_dense_context: Dict[str, Any] | None = None,
    policy_guidance_enabled: bool = True,
    fallback_to_reward_only: bool = True,
    mock_llm_mode: bool = True,
    use_real_llm: bool = False,
    api_key_env: str = "IUSEAPI_API_KEY",
    base_url: str = "https://www.iuseapi.com/v1",
    model: str = "gpt-5.2",
    temperature: float = 0.2,
    llm_timeout: float = 60.0,
    llm_max_retries: int = 3,
    llm_retry_backoff: float = 5.0,
    reuse_policy_guidance_cache: bool = False,
    policy_guidance_cache_root: str | Path | None = None,
    policy_guidance_cache_lookup_mode: str = "exact_or_latest_by_intervention",
) -> Dict[str, Any]:
    intervention_point = f"stage3_{str(checkpoint_id).lower()}_pre_checkpoint"
    result = {
        "policy_guidance_enabled": bool(policy_guidance_enabled),
        "policy_guidance_used": False,
        "integrated_guidance_card": None,
        "policy_guidance_failure_reason": None,
        "policy_guidance_behavior_summary_path": (
            None if behavior_summary_path in (None, "") else str(Path(str(behavior_summary_path)).expanduser().resolve())
        ),
        "integrated_guidance_card_path": None,
        "fallback_to_reward_only": bool(fallback_to_reward_only),
        "policy_guidance_source": "reward_only_fallback",
        "policy_guidance_evidence_keys": [],
        "intervention_point": intervention_point,
        "checkpoint_id": str(checkpoint_id),
        "round_id": 0,
    }
    if not policy_guidance_enabled:
        result["policy_guidance_failure_reason"] = "policy guidance disabled"
        return result
    try:
        validated_summary = validate_policy_behavior_summary(behavior_summary)
        backend, backend_probe = _build_policy_guidance_backend(
            use_real_llm=use_real_llm and not bool(mock_llm_mode),
            api_key_env=api_key_env,
            base_url=base_url,
            model=model,
            temperature=temperature,
            llm_timeout=llm_timeout,
            llm_max_retries=llm_max_retries,
            llm_retry_backoff=llm_retry_backoff,
        )
        critic = PolicyCriticClient(
            model=model,
            mock_llm_mode=bool(mock_llm_mode),
            fallback_to_reward_only=bool(fallback_to_reward_only),
            backend=backend,
            reuse_cache=bool(reuse_policy_guidance_cache),
            cache_root=Path(policy_guidance_cache_root).expanduser().resolve()
            if policy_guidance_cache_root not in (None, "")
            else DEFAULT_REAL_LLM_DRYRUN_CACHE_ROOT,
            cache_lookup_mode=str(policy_guidance_cache_lookup_mode or "exact_or_latest_by_intervention"),
            workflow_dir=Path(workflow_dir).resolve() if workflow_dir not in (None, "") else None,
            workflow_id=str(task_metadata.get("workflow_id") or ""),
            profile=str(task_metadata.get("profile") or ""),
            env_key=str(task_metadata.get("env_key") or ""),
            algorithm=str(task_metadata.get("algorithm") or ""),
        )
        critic_result = critic.generate_stage3_pre_checkpoint_integrated_guidance(
            intervention_point=intervention_point,
            checkpoint_id=str(checkpoint_id),
            checkpoint_step=int(checkpoint_step),
            current_pbrs_config=deepcopy(current_pbrs_config or {}),
            reward_metrics=deepcopy(reward_metrics or {}),
            behavior_summary=validated_summary,
            sparse_dense_context=deepcopy(sparse_dense_context or {}),
        )
        card = validate_integrated_guidance_card(critic_result["integrated_guidance_card"])
        result.update(
            {
                "policy_guidance_used": not bool(card.get("fallback_to_reward_only", False)),
                "integrated_guidance_card": card,
                "policy_guidance_failure_reason": (
                    str((card.get("metadata") or {}).get("failure_reason") or "") or None
                ),
                "policy_guidance_source": (
                    "stage3_checkpoint_behavior_summary_via_critic"
                    if not bool(card.get("fallback_to_reward_only", False))
                    else "reward_only_fallback"
                ),
                "policy_guidance_evidence_keys": list(
                    (card.get("policy_diagnosis") or {}).get("behavior_evidence") or []
                ),
                "call_record": deepcopy(critic_result.get("call_record") or {}),
                "backend_probe": backend_probe,
            }
        )
        return _normalize_policy_guidance_result(result)
    except Exception as exc:
        result["policy_guidance_failure_reason"] = str(exc)
        return _normalize_policy_guidance_result(result)


def build_stage3_after_round1_policy_guidance(
    *,
    task_metadata: Dict[str, Any],
    workflow_dir: str | Path | None = None,
    checkpoint_id: str,
    checkpoint_step: int,
    round_id: int,
    reward_metrics: Dict[str, Any],
    branch_results: list[Dict[str, Any]],
    branch_behavior_summaries: list[Dict[str, Any]],
    summary_for_policy_diagnosis: Dict[str, Any] | None,
    behavior_summary_path: str | Path | None = None,
    previous_integrated_guidance_card: Dict[str, Any] | None = None,
    policy_guidance_enabled: bool = True,
    fallback_to_reward_only: bool = True,
    mock_llm_mode: bool = True,
    use_real_llm: bool = False,
    api_key_env: str = "IUSEAPI_API_KEY",
    base_url: str = "https://www.iuseapi.com/v1",
    model: str = "gpt-5.2",
    temperature: float = 0.2,
    llm_timeout: float = 60.0,
    llm_max_retries: int = 3,
    llm_retry_backoff: float = 5.0,
    reuse_policy_guidance_cache: bool = False,
    policy_guidance_cache_root: str | Path | None = None,
    policy_guidance_cache_lookup_mode: str = "exact_or_latest_by_intervention",
) -> Dict[str, Any]:
    intervention_point = f"stage3_{str(checkpoint_id).lower()}_after_round1"
    result = {
        "policy_guidance_enabled": bool(policy_guidance_enabled),
        "policy_guidance_used": False,
        "integrated_guidance_card": None,
        "policy_guidance_failure_reason": None,
        "policy_guidance_behavior_summary_path": (
            None if behavior_summary_path in (None, "") else str(Path(str(behavior_summary_path)).expanduser().resolve())
        ),
        "integrated_guidance_card_path": None,
        "fallback_to_reward_only": bool(fallback_to_reward_only),
        "policy_guidance_source": "reward_only_fallback",
        "policy_guidance_evidence_keys": [],
        "intervention_point": intervention_point,
        "checkpoint_id": str(checkpoint_id),
        "round_id": int(round_id),
    }
    if not policy_guidance_enabled:
        result["policy_guidance_failure_reason"] = "policy guidance disabled"
        return result
    try:
        validated_summary = validate_policy_behavior_summary(summary_for_policy_diagnosis)
        backend, backend_probe = _build_policy_guidance_backend(
            use_real_llm=use_real_llm and not bool(mock_llm_mode),
            api_key_env=api_key_env,
            base_url=base_url,
            model=model,
            temperature=temperature,
            llm_timeout=llm_timeout,
            llm_max_retries=llm_max_retries,
            llm_retry_backoff=llm_retry_backoff,
        )
        critic = PolicyCriticClient(
            model=model,
            mock_llm_mode=bool(mock_llm_mode),
            fallback_to_reward_only=bool(fallback_to_reward_only),
            backend=backend,
            reuse_cache=bool(reuse_policy_guidance_cache),
            cache_root=Path(policy_guidance_cache_root).expanduser().resolve()
            if policy_guidance_cache_root not in (None, "")
            else DEFAULT_REAL_LLM_DRYRUN_CACHE_ROOT,
            cache_lookup_mode=str(policy_guidance_cache_lookup_mode or "exact_or_latest_by_intervention"),
            workflow_dir=Path(workflow_dir).resolve() if workflow_dir not in (None, "") else None,
            workflow_id=str(task_metadata.get("workflow_id") or ""),
            profile=str(task_metadata.get("profile") or ""),
            env_key=str(task_metadata.get("env_key") or ""),
            algorithm=str(task_metadata.get("algorithm") or ""),
        )
        critic_result = critic.generate_stage3_after_round1_integrated_guidance(
            intervention_point=intervention_point,
            checkpoint_id=str(checkpoint_id),
            checkpoint_step=int(checkpoint_step),
            round_id=int(round_id),
            reward_metrics=deepcopy(reward_metrics or {}),
            behavior_summary=validated_summary,
            branch_results=deepcopy(branch_results or []),
            branch_behavior_summaries=deepcopy(branch_behavior_summaries or []),
            previous_integrated_guidance_card=deepcopy(previous_integrated_guidance_card or {}),
        )
        card = validate_integrated_guidance_card(critic_result["integrated_guidance_card"])
        evidence_keys = list((card.get("policy_diagnosis") or {}).get("behavior_evidence") or [])
        for item in list(card.get("branch_evidence") or []):
            evidence_keys.extend(list(item.get("behavior_evidence") or [])[:2])
        result.update(
            {
                "policy_guidance_used": not bool(card.get("fallback_to_reward_only", False)),
                "integrated_guidance_card": card,
                "policy_guidance_failure_reason": (
                    str((card.get("metadata") or {}).get("failure_reason") or "") or None
                ),
                "policy_guidance_source": (
                    "stage3_round1_branch_behavior_summary_via_critic"
                    if not bool(card.get("fallback_to_reward_only", False))
                    else "reward_only_fallback"
                ),
                "policy_guidance_evidence_keys": evidence_keys,
                "call_record": deepcopy(critic_result.get("call_record") or {}),
                "backend_probe": backend_probe,
            }
        )
        return _normalize_policy_guidance_result(result)
    except Exception as exc:
        result["policy_guidance_failure_reason"] = str(exc)
        return _normalize_policy_guidance_result(result)


def build_stage3_round1_generator_payload_with_policy_guidance(
    *,
    task_metadata: Dict[str, Any],
    checkpoint_id: str,
    checkpoint_step: int,
    reward_metrics: Dict[str, Any],
    current_pbrs_config: Dict[str, Any],
    integrated_guidance_card: Dict[str, Any] | None,
    existing_reward_only_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = deepcopy(existing_reward_only_payload or {})
    payload.setdefault("task_metadata", deepcopy(task_metadata))
    payload.setdefault("checkpoint_id", str(checkpoint_id))
    payload.setdefault("checkpoint_step", int(checkpoint_step))
    payload.setdefault("reward_metrics", deepcopy(reward_metrics or {}))
    payload.setdefault("current_pbrs_config", deepcopy(current_pbrs_config or {}))
    validated_card = None
    if integrated_guidance_card:
        validated_card = validate_integrated_guidance_card(integrated_guidance_card)
    if validated_card is None or bool(validated_card.get("fallback_to_reward_only", False)):
        payload["policy_guidance_used"] = False
        payload["policy_guidance_source"] = "reward_only_fallback"
        payload["policy_guidance_fallback_to_reward_only"] = True
        return payload
    payload.update(
        {
            "policy_guidance_used": True,
            "policy_guidance_source": "stage3_checkpoint_behavior_summary_via_critic",
            "policy_guidance_fallback_to_reward_only": False,
            "policy_guidance_evidence_keys": list(
                ((validated_card.get("policy_diagnosis") or {}).get("behavior_evidence")) or []
            ),
            "integrated_guidance_card": validated_card,
        }
    )
    return payload


def build_stage3_round2_generator_payload_with_policy_guidance(
    *,
    task_metadata: Dict[str, Any],
    checkpoint_id: str,
    checkpoint_step: int,
    round_id: int,
    branch_results: list[Dict[str, Any]],
    integrated_guidance_card: Dict[str, Any] | None,
    existing_reward_only_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = deepcopy(existing_reward_only_payload or {})
    payload.setdefault("task_metadata", deepcopy(task_metadata))
    payload.setdefault("checkpoint_id", str(checkpoint_id))
    payload.setdefault("checkpoint_step", int(checkpoint_step))
    payload.setdefault("round_id", int(round_id))
    payload.setdefault("branch_results", deepcopy(branch_results or []))
    validated_card = None
    if integrated_guidance_card:
        validated_card = validate_integrated_guidance_card(integrated_guidance_card)
    if validated_card is None or bool(validated_card.get("fallback_to_reward_only", False)):
        payload["policy_guidance_used"] = False
        payload["policy_guidance_source"] = "reward_only_fallback"
        payload["policy_guidance_fallback_to_reward_only"] = True
        return payload
    evidence_keys = list(
        ((validated_card.get("policy_diagnosis") or {}).get("behavior_evidence")) or []
    )
    for item in list(validated_card.get("branch_evidence") or []):
        evidence_keys.extend(list(item.get("behavior_evidence") or [])[:2])
    payload.update(
        {
            "policy_guidance_used": True,
            "policy_guidance_source": "stage3_round1_branch_behavior_summary_via_critic",
            "policy_guidance_fallback_to_reward_only": False,
            "policy_guidance_evidence_keys": evidence_keys,
            "integrated_guidance_card": validated_card,
        }
    )
    return payload


def build_policy_guided_stage1b_candidate_batch(
    *,
    integrated_guidance_card: Dict[str, Any],
    max_candidates: int = 3,
    env_family: str = "lbf",
) -> Dict[str, Any]:
    guidance = deepcopy((integrated_guidance_card or {}).get("candidate_generation_guidance") or {})
    evidence_keys = list(
        ((integrated_guidance_card or {}).get("policy_diagnosis") or {}).get("behavior_evidence")
        or []
    )[:3]
    implications = list(guidance.get("reward_search_implications") or [])
    fallback = bool((integrated_guidance_card or {}).get("fallback_to_reward_only", False))
    if env_family == "rware":
        candidate_types = list(guidance.get("candidate_types") or [])
        evidence_blob = " ".join(evidence_keys).lower()
        if "near_requested_shelf_steps" in evidence_blob or "shelf_acquisition" in evidence_blob:
            candidate_types = ["requested_shelf_acquisition", "reference_like"] + candidate_types
        if "pickup_success_events" in evidence_blob or "pickup_progress" in evidence_blob:
            candidate_types = ["pickup_progress", "reference_like"] + candidate_types
        if "carrying_steps" in evidence_blob or "near_goal_while_carrying_steps" in evidence_blob:
            candidate_types = ["goal_progress", "reference_like"] + candidate_types
        if "blocked_move_steps" in evidence_blob or "congestion_proxy" in evidence_blob:
            candidate_types = ["traffic_conservative", "reference_like"] + candidate_types
        if "route_oscillation_score" in evidence_blob or "route_stability" in evidence_blob:
            candidate_types = ["late_stability", "reference_like"] + candidate_types
        if fallback or not integrated_guidance_card.get("use_in_reward_generation", False):
            candidate_types = ["conservative", "reference_like"]
        if not candidate_types:
            candidate_types = list(RWARE_STAGE1B_CANDIDATE_TYPES)
        candidates = []
        seen = set()
        for index, candidate_type in enumerate(candidate_types):
            if len(candidates) >= max(1, max_candidates):
                break
            evidence_key = evidence_keys[index % len(evidence_keys)] if evidence_keys else ""
            candidate = _build_rware_policy_guided_candidate(
                candidate_id=f"stage1b_pg_{candidate_type}_{index+1}",
                candidate_type=candidate_type,
                evidence_key=evidence_key,
                rationale=(
                    f"Uses evidence key {evidence_key or 'reward_only_fallback'}; "
                    f"implication {implications[0] if implications else 'conservative RWARE reward-only search'}."
                ),
            )
            signature = (
                candidate["mode"],
                round(float(candidate["beta"]), 3),
                tuple(round(float(candidate["weights"][term]), 3) for term in ("shelf", "pickup", "goal", "deliv", "traffic", "stab")),
            )
            if signature in seen:
                continue
            seen.add(signature)
            candidate.update(
                {
                    "hypothesis": implications[0] if implications else "RWARE-native policy-guided fallback candidate.",
                    "expected_early_effect": "guide Stage 1b early dense initialization using RWARE-native behavior evidence",
                    "risk": "policy-guidance mock candidate with native RWARE schema",
                }
            )
            candidates.append(candidate)
        return validate_policy_guided_candidate_batch(
            {
                "policy_guidance_used": not fallback,
                "policy_guidance_candidate_collapsed": len(seen) < len(candidates),
                "collapsed_reason": (
                    "stage1b_policy_guided_candidates_collapsed_after_rware_projection"
                    if len(seen) < len(candidates)
                    else ""
                ),
                "candidate_generation_basis": {
                    "reward_diagnosis_used": True,
                    "policy_diagnosis_used": not bool(
                        ((integrated_guidance_card or {}).get("policy_diagnosis") or {}).get(
                            "insufficient_evidence", False
                        )
                    ),
                    "evidence_keys": evidence_keys,
                },
                "candidates": candidates,
            }
        )
    base_beta = 0.30
    base_wc = 0.50
    candidate_types = list(guidance.get("candidate_types") or [])
    evidence_blob = " ".join(evidence_keys).lower()
    if "low coverage" in evidence_blob or "coverage" in evidence_blob:
        candidate_types = ["coverage_recovery", "reference_like"] + candidate_types
    if "failed collect" in evidence_blob or "final collection failure" in evidence_blob:
        candidate_types = ["collection_geometry", "progress_shift"] + candidate_types
    if "over-concentration" in evidence_blob or "allocation" in evidence_blob:
        candidate_types = ["allocation_rebalance", "reference_like"] + candidate_types
    if "stability" in evidence_blob or "oscillation" in evidence_blob:
        candidate_types = ["stability_recovery", "progress_shift"] + candidate_types
    if fallback or not integrated_guidance_card.get("use_in_reward_generation", False):
        candidate_types = ["conservative", "reference_like"]
    if not candidate_types:
        candidate_types = ["reference_like", "balanced", "conservative"]
    candidates: List[Dict[str, Any]] = []
    for index, candidate_type in enumerate(candidate_types[: max(1, max_candidates)]):
        evidence_key = evidence_keys[index % len(evidence_keys)] if evidence_keys else ""
        beta = _adjust_value(base_beta, str(guidance.get("beta_direction") or "same"), 0.06 * (index + 1))
        wc = _adjust_value(base_wc, str(guidance.get("wc_direction") or "same"), 0.08 * (index + 1))
        if candidate_type in {"exploration_boost", "progress_shift", "coverage_recovery"}:
            beta = min(1.0, beta + 0.08)
        if candidate_type in {"coordination_recovery", "wc_down_wp_up", "recovery", "allocation_rebalance", "collection_geometry"}:
            wc = max(0.0, wc - 0.10)
        if candidate_type in {"stability_recovery", "reference_like"}:
            beta = max(0.0, beta - 0.04)
        wp = 1.0 - wc
        candidates.append(
            _to_policy_guided_candidate(
                env_family=env_family,
                candidate={
                    "candidate_id": f"stage1b_pg_{candidate_type}_{index+1}",
                    "beta": round(beta, 3),
                    "wc": round(wc, 3),
                    "wp": round(wp, 3),
                    "candidate_type": candidate_type,
                    "evidence_key_used": evidence_key,
                    "rationale": (
                        f"Uses evidence key {evidence_key or 'reward_only_fallback'}; "
                        f"implication {implications[0] if implications else 'conservative reward-only search'}."
                    ),
                    "hypothesis": (
                        implications[0]
                        if implications
                        else "Fallback reward-only candidate because policy guidance is unavailable."
                    ),
                    "expected_early_effect": (
                        "guide Stage 1b early dense initialization using integrated reward and policy evidence"
                    ),
                    "risk": "tiny-smoke policy-guidance mock candidate",
                },
            )
        )
    return validate_policy_guided_candidate_batch(
        {
            "policy_guidance_used": not fallback,
            "policy_guidance_candidate_collapsed": len(
                {(item['beta'], item['wc']) for item in candidates}
            ) < len(candidates),
            "collapsed_reason": (
                "stage1b_policy_guided_candidates_collapsed_after_numeric_projection"
                if len({(item['beta'], item['wc']) for item in candidates}) < len(candidates)
                else ""
            ),
            "candidate_generation_basis": {
                "reward_diagnosis_used": True,
                "policy_diagnosis_used": not bool(
                    ((integrated_guidance_card or {}).get("policy_diagnosis") or {}).get(
                        "insufficient_evidence", False
                    )
                ),
                "evidence_keys": evidence_keys,
            },
            "candidates": candidates,
        }
    )


def build_policy_guided_stage3_candidate_batch(
    *,
    integrated_guidance_card: Dict[str, Any],
    checkpoint_id: str,
    round_id: int,
    max_candidates: int = 3,
    env_family: str = "lbf",
) -> Dict[str, Any]:
    guidance = deepcopy((integrated_guidance_card or {}).get("candidate_generation_guidance") or {})
    round2_guidance = deepcopy((integrated_guidance_card or {}).get("round2_search_guidance") or {})
    evidence_keys = list(
        ((integrated_guidance_card or {}).get("policy_diagnosis") or {}).get("behavior_evidence")
        or []
    )[:4]
    implications = list(guidance.get("reward_search_implications") or [])
    fallback = bool((integrated_guidance_card or {}).get("fallback_to_reward_only", False))
    if env_family == "rware":
        candidate_types = list(guidance.get("candidate_types") or [])
        evidence_blob = " ".join(evidence_keys).lower()
        if "near_requested_shelf_steps" in evidence_blob or "shelf_acquisition" in evidence_blob:
            candidate_types = ["requested_shelf_acquisition", "reference_like"] + candidate_types
        if "pickup_success_events" in evidence_blob or "pickup_progress" in evidence_blob:
            candidate_types = ["pickup_progress", "reference_like"] + candidate_types
        if "carrying_steps" in evidence_blob or "near_goal_while_carrying_steps" in evidence_blob:
            candidate_types = ["goal_progress", "reference_like"] + candidate_types
        if "delivery_success_events" in evidence_blob:
            candidate_types = ["goal_progress", "late_stability"] + candidate_types
        if "blocked_move_steps" in evidence_blob or "congestion_proxy" in evidence_blob:
            candidate_types = ["traffic_conservative", "reference_like"] + candidate_types
        if "route_oscillation_score" in evidence_blob or "route_stability" in evidence_blob:
            candidate_types = ["late_stability", "reference_like"] + candidate_types
        if int(round_id) == 2 and round2_guidance.get("around_candidate"):
            candidate_types = ["late_stability", "goal_progress"] + candidate_types
        if fallback or not integrated_guidance_card.get("use_in_reward_generation", False):
            candidate_types = ["conservative", "reference_like"]
        if not candidate_types:
            candidate_types = list(RWARE_STAGE3_CANDIDATE_TYPES)
        candidates = []
        seen = set()
        for index, candidate_type in enumerate(candidate_types):
            if len(candidates) >= max(1, max_candidates):
                break
            evidence_key = evidence_keys[index % len(evidence_keys)] if evidence_keys else ""
            candidate = _build_rware_policy_guided_candidate(
                candidate_id=f"stage3_{str(checkpoint_id).lower()}_r{int(round_id)}_{candidate_type}_{index+1}",
                candidate_type=candidate_type,
                evidence_key=evidence_key,
                rationale=(
                    f"Uses guidance evidence key {evidence_key or 'reward_only_fallback'}; "
                    f"implication {implications[0] if implications else 'conservative RWARE reward-only search'}."
                ),
                stage3_round2=int(round_id) == 2,
            )
            signature = (
                candidate["mode"],
                round(float(candidate["beta"]), 3),
                tuple(round(float(candidate["weights"][term]), 3) for term in ("shelf", "pickup", "goal", "deliv", "traffic", "stab")),
            )
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(candidate)
        return validate_policy_guided_candidate_batch(
            {
                "policy_guidance_used": not fallback,
                "policy_guidance_candidate_collapsed": len(seen) < len(candidates),
                "collapsed_reason": (
                    "stage3_policy_guided_candidates_collapsed_after_rware_projection"
                    if len(seen) < len(candidates)
                    else ""
                ),
                "candidate_generation_basis": {
                    "reward_diagnosis_used": True,
                    "policy_diagnosis_used": not bool(
                        ((integrated_guidance_card or {}).get("policy_diagnosis") or {}).get(
                            "insufficient_evidence", False
                        )
                    ),
                    "evidence_keys": evidence_keys,
                },
                "candidates": candidates,
            }
        )
    candidate_types = list(guidance.get("candidate_types") or [])
    evidence_blob = " ".join(evidence_keys).lower()
    if "low coverage" in evidence_blob or "coverage" in evidence_blob:
        candidate_types = ["coverage_recovery", "reference_like"] + candidate_types
    if "failed collect" in evidence_blob or "final collection failure" in evidence_blob:
        candidate_types = ["collection_geometry", "progress_shift"] + candidate_types
    if "over-concentration" in evidence_blob or "allocation" in evidence_blob:
        candidate_types = ["allocation_rebalance", "reference_like"] + candidate_types
    if "stability" in evidence_blob or "oscillation" in evidence_blob:
        candidate_types = ["stability_recovery", "reference_like"] + candidate_types
    if int(round_id) == 2 and round2_guidance.get("around_candidate"):
        candidate_types = ["stability_recovery", "reference_like"] + candidate_types
    if fallback or not integrated_guidance_card.get("use_in_reward_generation", False):
        candidate_types = ["conservative", "reference_like"]
    if not candidate_types:
        candidate_types = ["reference_like", "coordination_recovery", "progress_shift"]
    base_beta = 0.20 if int(round_id) == 2 else 0.25
    base_wc = 0.45 if int(round_id) == 2 else 0.50
    candidates: List[Dict[str, Any]] = []
    for index, candidate_type in enumerate(candidate_types[: max(1, max_candidates)]):
        evidence_key = evidence_keys[index % len(evidence_keys)] if evidence_keys else ""
        beta = _adjust_value(base_beta, str(guidance.get("beta_direction") or "same"), 0.05 * (index + 1))
        wc = _adjust_value(base_wc, str(guidance.get("wc_direction") or "same"), 0.08 * (index + 1))
        if candidate_type in {"coordination_recovery", "progress_shift", "allocation_rebalance", "collection_geometry"}:
            wc = max(0.0, wc - 0.08)
        if candidate_type in {"stability_recovery", "conservative", "coverage_recovery"}:
            beta = max(0.0, beta - 0.04)
        if candidate_type in {"coverage_recovery"}:
            beta = min(1.0, beta + 0.10)
        wp = 1.0 - wc
        candidates.append(
            _to_policy_guided_candidate(
                env_family=env_family,
                candidate={
                    "candidate_id": f"stage3_{str(checkpoint_id).lower()}_r{int(round_id)}_{candidate_type}_{index+1}",
                    "beta": round(beta, 3),
                    "wc": round(wc, 3),
                    "wp": round(wp, 3),
                    "candidate_type": candidate_type,
                    "evidence_key_used": evidence_key,
                    "rationale": (
                        f"Uses guidance evidence key {evidence_key or 'reward_only_fallback'}; "
                        f"implication {implications[0] if implications else 'conservative reward-only search'}."
                    ),
                },
            )
        )
    return validate_policy_guided_candidate_batch(
        {
            "policy_guidance_used": not fallback,
            "policy_guidance_candidate_collapsed": len(
                {(item['beta'], item['wc']) for item in candidates}
            ) < len(candidates),
            "collapsed_reason": (
                "stage3_policy_guided_candidates_collapsed_after_numeric_projection"
                if len({(item['beta'], item['wc']) for item in candidates}) < len(candidates)
                else ""
            ),
            "candidate_generation_basis": {
                "reward_diagnosis_used": True,
                "policy_diagnosis_used": not bool(
                    ((integrated_guidance_card or {}).get("policy_diagnosis") or {}).get(
                        "insufficient_evidence", False
                    )
                ),
                "evidence_keys": evidence_keys,
            },
            "candidates": candidates,
        }
    )


def _adjust_value(base: float, direction: str, step: float) -> float:
    if direction == "up":
        return min(1.0, base + step)
    if direction == "down":
        return max(0.0, base - step)
    return base
