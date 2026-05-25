from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


ARCHIVED_MAPPO_QMIX_RESULTS_ROOT = str(
    Path(__file__).resolve().parents[2]
    / "archived_results"
    / "policy_method_mappo_qmix"
    / "end_to_end_workflows"
)
PORTABLE_STAGE1_BUNDLE_ROOT = str(
    Path(__file__).resolve().parents[1]
    / "reuse_artifacts"
    / "stage1_sparse_baselines"
)


def _apply_stage3_effective_update_budget_defaults(profile: Dict[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(profile)
    update_candidates_per_round = int(
        resolved.get("stage3_update_candidates_per_round")
        or resolved.get("stage3_candidates_per_round")
        or 3
    )
    shared_fields = {
        "stage3_branch_rounds": int(resolved.get("stage3_branch_rounds") or 2),
        "stage3_candidates_per_round": update_candidates_per_round,
        "stage3_update_candidates_per_round": update_candidates_per_round,
        "stage3_include_no_change_control": bool(
            resolved.get("stage3_include_no_change_control", True)
        ),
        "stage3_no_change_counts_toward_round_budget": False,
        "stage3_reuse_no_change_across_rounds": True,
        "stage3_min_effective_update_candidates_per_checkpoint": int(
            resolved.get("stage3_min_effective_update_candidates_per_checkpoint") or 6
        ),
        "stage3_require_nontrivial_config_delta": True,
        "stage3_max_fallback_style_candidates_per_checkpoint": int(
            resolved.get("stage3_max_fallback_style_candidates_per_checkpoint") or 1
        ),
        "stage3_require_candidate_type_diversity": True,
        "policy_guided_stage3_require_behavior_evidence": True,
    }
    resolved.update(shared_fields)

    phase1_method = deepcopy(resolved.get("phase1_method") or {})
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement.update(
        {
            key: deepcopy(value)
            for key, value in shared_fields.items()
        }
    )
    phase1_method["adaptive_replacement"] = adaptive_replacement
    resolved["phase1_method"] = phase1_method
    return resolved


def _mark_lbf_policy_guided_sidecar(profile: Dict[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(profile)
    resolved["pbrs_version"] = "lbf_pbrs_v2"
    resolved["pbrs_mode"] = "balanced_collection_ready"
    resolved["pbrs_supported_modes"] = [
        "balanced_collection_ready",
        "coverage_ready_balance",
        "approach_collection_push",
        "allocation_stability_support",
    ]
    resolved["pbrs_supported_terms"] = ["col", "app", "cov", "ready", "alloc", "stab"]
    resolved["formal_lbf_pbrs_v2_guard"] = True
    resolved["dense_reference_sidecar_enabled"] = True
    resolved["sparse_eval_only"] = True
    resolved["eval_use_pbrs"] = False
    return resolved


def _apply_clean_lbf_profile_flags(
    profile: Dict[str, Any],
    *,
    use_real_llm: bool,
    use_real_llm_for_field_analysis: bool,
) -> Dict[str, Any]:
    resolved = deepcopy(profile)
    resolved["pbrs_version"] = "lbf_pbrs_v2"
    resolved["pbrs_mode"] = "balanced_collection_ready"
    resolved["use_real_llm"] = bool(use_real_llm)
    resolved["use_real_llm_for_field_analysis"] = bool(use_real_llm_for_field_analysis)
    resolved["mock_llm_mode"] = False
    resolved["sparse_eval_only"] = True
    resolved["eval_use_pbrs"] = False
    resolved["apply_dense_reward_in_eval"] = False
    resolved["stage3_branch_rounds"] = 2
    resolved["stage3_candidates_per_round"] = 3
    resolved["stage3_update_candidates_per_round"] = 3
    resolved["stage3_include_no_change_control"] = True
    resolved["stage3_no_change_counts_toward_round_budget"] = False
    resolved["stage3_reuse_no_change_across_rounds"] = True
    resolved["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    resolved["stage3_require_nontrivial_config_delta"] = True
    resolved["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    resolved["stage3_require_candidate_type_diversity"] = True
    resolved["policy_guided_stage3_require_behavior_evidence"] = True

    phase1_method = deepcopy(resolved.get("phase1_method") or {})
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["policy_guidance_stage1b_fallback_to_reward_only"] = False
    phase1_method["enable_stage5_final_full_run"] = False
    phase1_method["final_spec_is_summary_only"] = True
    phase1_method["rolling_adaptive_mainline_is_final_result"] = True
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = True
    adaptive_replacement["mock_llm_mode"] = False
    adaptive_replacement["fallback_to_deterministic_on_llm_error"] = False
    adaptive_replacement["add_deterministic_recovery_fallback"] = False
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 3
    adaptive_replacement["stage3_update_candidates_per_round"] = 3
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["stage3_no_change_counts_toward_round_budget"] = False
    adaptive_replacement["stage3_reuse_no_change_across_rounds"] = True
    adaptive_replacement["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    adaptive_replacement["stage3_require_nontrivial_config_delta"] = True
    adaptive_replacement["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    adaptive_replacement["stage3_require_candidate_type_diversity"] = True
    adaptive_replacement["policy_guided_stage3_require_behavior_evidence"] = True
    phase1_method["adaptive_replacement"] = adaptive_replacement
    resolved["phase1_method"] = phase1_method
    return resolved


def _base_profile() -> Dict[str, Any]:
    return {
        "reward_paradigm": "pbrs",
        "max_rounds": 3,
        "min_count": 3,
        "max_count": 3,
        "min_step_gap": 50000,
        "max_points_per_metric": 15,
        "field_rounds": ["beta", "wc"],
        "branch_budget_steps": 205000,
        "reuse_completed_candidates": True,
        "native_original_pbrs_branching": True,
        "use_real_llm": True,
        "use_real_llm_for_field_analysis": True,
        "model": "gpt-5.2",
        "temperature": 0.2,
        "llm_timeout": 60.0,
        "llm_max_retries": 3,
        "llm_retry_backoff": 5.0,
        "budget_profile": "formal",
        # Match the lighter baseline cadence unless a caller explicitly overrides it.
        "save_model_interval": 50000,
        "checkpointing_strategy": {
            "stage1_enable_checkpointing": False,
            "stage3_enable_checkpointing": False,
            "stage5_enable_checkpointing": False,
        },
        "recovery_policy": {
            "stage234_max_attempts": 3,
        },
        "phase1_method": {
            "method_version": "phase1_strong_baseline_v1",
            "stage_boundary_strategy": "sparse_then_post_dense_formal_selection",
            "checkpoint_selection_mode": "post_dense_dual_source",
            "dense_reference": {
                "selection_mode": "llm_select_fixed_beta_wc",
                "reward_family": "native_original_pbrs",
                "candidate_beta_values": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "candidate_wc_values": [0.1, 0.3, 0.5, 0.7, 0.9],
                "wp_policy": "derive_as_1_minus_wc",
                "gamma": 0.99,
                "variant": "original",
                "selection_goal": (
                    "Choose a stable dense reference native original PBRS configuration "
                    "(beta and wc, with wp derived as 1-wc) that exhibits a complete "
                    "cooperative learning trajectory; this is a boundary-calibration "
                    "reference, not necessarily the globally best fixed-PBRS run."
                ),
            },
            "stage3_role": "candidate_generator",
            "stage3_interpretation": (
                "Stage 3 branch validation starts from checkpoints selected on the completed "
                "dense-reference run. Each branch changes only one target coefficient while "
                "keeping the remaining coefficients fixed to the dense-reference configuration. "
                "Its role is to generate promising stage-conditioned candidate values and shrink "
                "the search space, not to serve as final proof of global optimality."
            ),
            "dense_validation": {
                "enabled": True,
                "top_k_schedules": 3,
                "selection_goal": (
                    "Run a small number of from-scratch dense-on-dense validations on "
                    "the most promising stage-conditioned schedules before the final "
                    "confirmatory full run."
                ),
            },
        },
        "resource_policy": {
            "stage1_use_cuda": True,
            "stage3_use_cuda": True,
            "stage3_max_parallel_candidates": 6,
            "stage5_use_cuda": True,
        },
    }


def _base_adaptive_profile() -> Dict[str, Any]:
    profile = _base_profile()
    profile.update(
        {
            "use_dual_llm_stage1b_initial_search": True,
            "initial_dense_search_rounds": 2,
            "initial_dense_candidates_per_round": 3,
            "initial_dense_candidate_budget_steps": 850000,
            "initial_dense_candidate_target_checkpoint_step": 800000,
            "stage1b_selected_endpoint_target_step": 800000,
            "stage1b_endpoint_tolerance_steps": 50000,
            "stage1b_endpoint_selection_policy": "prefer_ge_then_le_nearest",
            "initial_dense_candidate_parallel": True,
            "initial_dense_candidate_save_endpoint_checkpoint": True,
            "fallback_candidates_run_in_normal_path": False,
            "keep_dense_reference_continuation": True,
            "dense_reference_starts_from_stage1b_best_checkpoint": True,
            "adaptive_mainline_starts_from_stage1b_best_checkpoint": True,
            "use_fixed_intervention_checkpoints": True,
            "fixed_intervention_checkpoints": [1000000, 1500000],
            "checkpoint_count": 2,
            "branch_budget_steps": 300000,
            "branch_test_interval": 50000,
            "stage3_branch_rounds": 2,
            "stage3_candidates_per_round": 3,
            "stage3_include_no_change_control": True,
            "stage3_no_change_counts_toward_round_budget": True,
            "stage3_candidate_review_enabled": True,
            "stage3_candidate_repair_max_rounds": 1,
            "use_winner_branch_promotion": True,
            "stage3_selection_metric_mode": "stability_aware",
            "stage3_last_k_eval_points": 5,
            "stage3_best_metric_weight": 0.10,
            "stage3_last_k_mean_weight": 0.35,
            "stage3_auc_mean_weight": 0.35,
            "stage3_final_weight": 0.20,
            "stage3_use_stability_penalty": True,
            "stage3_stability_penalty_weight": 0.10,
            "stage3_update_margin_last_k": 0.03,
            "stage3_update_margin_auc": 0.02,
            "stage3_final_tolerance": 0.02,
            "stage3_spike_gap_threshold": 0.10,
            "stage3_std_tolerance": 0.05,
            "stage3_tie_tolerance": 0.01,
            "stage3_low_return_adaptive_margin": True,
            "stage3_min_margin_last_k": 0.01,
            "stage3_min_margin_auc": 0.005,
            "stage3_relative_margin_ratio": 0.10,
            "stage3_reference_aware_gate": True,
            "stage3_reference_lead_threshold": 0.05,
            "stage3_reference_tolerance": 0.03,
            "stage3_reference_extra_margin": 0.02,
            "stage3_first_replacement_extra_margin": 0.02,
            "stage3_detect_revert_to_stage1b_config": True,
            "stage3_oscillation_detection": True,
            "stage3_llm_result_diagnosis_report_only": True,
            "stage3_final_decision_source": "deterministic_gate",
            "llm_recommendation_used_for_final_decision": False,
            "fallback_to_single_llm_initial_config": True,
            "fallback_to_existing_stage3": True,
            "enable_first_change_fixed_initial_baseline": False,
            "enable_stage4_final_spec_summary": True,
            "enable_stage5_final_full_run": False,
            "final_spec_is_summary_only": True,
            "rolling_adaptive_mainline_is_final_result": True,
            "checkpoint_retention_mode": "new_method_minimal_v1",
            "stage1_sparse_save_checkpoints": False,
            "stage1b_candidate_save_endpoint_checkpoint": True,
            "stage1b_candidate_save_intermediate_checkpoints": False,
            "stage1b_keep_only_endpoint_checkpoint": True,
            "dense_reference_save_checkpoints": False,
            "adaptive_mainline_save_c1_c2_source_checkpoints": True,
            "adaptive_mainline_save_intermediate_checkpoints": False,
            "adaptive_mainline_save_final_checkpoint": False,
            "stage3_branch_save_endpoint_checkpoint": True,
            "stage3_branch_save_intermediate_checkpoints": False,
            "delete_failed_branch_checkpoints": False,
            "delete_loser_branch_checkpoints": False,
            "auto_checkpoint_cleanup": False,
            "checkpoint_cleanup_dry_run_only": True,
        }
    )
    profile["phase1_method"] = {
        "method_version": "phase1_adaptive_replacement_v1",
        "stage_boundary_strategy": "selected_stage1b_checkpoint_then_fixed_interventions",
        "checkpoint_selection_mode": "fixed_intervention_schedule",
        "use_dual_llm_stage1b_initial_search": True,
        "fallback_to_single_llm_initial_config": True,
        "keep_dense_reference_continuation": True,
        "dense_reference_starts_from_stage1b_best_checkpoint": True,
        "adaptive_mainline_starts_from_stage1b_best_checkpoint": True,
        "use_fixed_intervention_checkpoints": True,
        "fixed_intervention_checkpoints": [1000000, 1500000],
        "checkpoint_count": 2,
        "fixed_checkpoint_plan": [
            {
                "name": "C1",
                "step": 1000000,
                "stage_label": "post_initialization_transition",
                "intervention_question": "Does the early-selected dense reward configuration remain suitable after the initial 800k dense search phase?",
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
                "intervention_question": "Does the current reward configuration remain stable and task-aligned in the later training phase?",
                "recommended_candidate_focus": [
                    "beta_down",
                    "wc_down_wp_up",
                    "reference_like",
                    "recovery",
                ],
                "branch_budget_feasible": True,
            },
        ],
        "dense_reference": deepcopy((profile.get("phase1_method") or {}).get("dense_reference") or {}),
        "initial_dense_search": {
            "enabled": True,
            "rounds": 2,
            "candidates_per_round": 3,
            "candidate_budget_steps": 850000,
            "candidate_target_checkpoint_step": 800000,
            "selected_endpoint_target_step": 800000,
            "endpoint_tolerance_steps": 50000,
            "endpoint_selection_policy": "prefer_ge_then_le_nearest",
            "parallel": True,
            "save_endpoint_checkpoint": True,
            "save_intermediate_checkpoints": False,
            "keep_only_endpoint_checkpoint": True,
        },
        "adaptive_replacement": {
            "enabled": True,
            "selection_goal": (
                "At each formal checkpoint, compare a small neighborhood of candidate PBRS "
                "configurations against the current no-change configuration, then directly "
                "replace the mainline configuration only when a candidate shows a meaningful "
                "short-horizon improvement in the current training context."
            ),
            "fixed_intervention_checkpoints": [1000000, 1500000],
            "use_fixed_intervention_checkpoints": True,
            "branch_budget_steps": 300000,
            "stage3_branch_rounds": 2,
            "stage3_candidates_per_round": 3,
            "stage3_include_no_change_control": True,
            "stage3_no_change_counts_toward_round_budget": True,
            "stage3_candidate_review_enabled": True,
            "stage3_candidate_repair_max_rounds": 1,
            "use_winner_branch_promotion": True,
            "fallback_to_existing_stage3": True,
            "stage3_selection_metric_mode": "stability_aware",
            "stage3_last_k_eval_points": 5,
            "stage3_best_metric_weight": 0.10,
            "stage3_last_k_mean_weight": 0.35,
            "stage3_auc_mean_weight": 0.35,
            "stage3_final_weight": 0.20,
            "stage3_use_stability_penalty": True,
            "stage3_stability_penalty_weight": 0.10,
            "stage3_update_margin_last_k": 0.03,
            "stage3_update_margin_auc": 0.02,
            "stage3_final_tolerance": 0.02,
            "stage3_spike_gap_threshold": 0.10,
            "stage3_std_tolerance": 0.05,
            "stage3_tie_tolerance": 0.01,
            "stage3_low_return_adaptive_margin": True,
            "stage3_min_margin_last_k": 0.01,
            "stage3_min_margin_auc": 0.005,
            "stage3_relative_margin_ratio": 0.10,
            "stage3_reference_aware_gate": True,
            "stage3_reference_lead_threshold": 0.05,
            "stage3_reference_tolerance": 0.03,
            "stage3_reference_extra_margin": 0.02,
            "stage3_first_replacement_extra_margin": 0.02,
            "stage3_detect_revert_to_stage1b_config": True,
            "stage3_oscillation_detection": True,
            "stage3_llm_result_diagnosis_report_only": True,
            "stage3_final_decision_source": "deterministic_gate",
            "llm_recommendation_used_for_final_decision": False,
            "candidate_generation": {
                "beta_values": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "wc_values": [0.1, 0.3, 0.5, 0.7, 0.9],
                "beta_neighbor_radius": 1,
                "wc_neighbor_radius": 1,
                "include_diagonal_pairs": True,
                "include_no_change": True,
                "max_candidates_per_checkpoint": 9,
            },
            "use_llm_stage3_candidate_generation": True,
            "use_llm_stage3_result_diagnosis": True,
            "llm_stage3_max_candidates": 9,
            "llm_stage3_retry_count": 2,
            "llm_stage3_use_cache": True,
            "enforce_recovery_candidate_when_underperforming": True,
            "llm_candidate_repair_retry_count": 1,
            "add_deterministic_recovery_fallback": True,
            "fallback_to_deterministic_on_llm_error": True,
            "launch_first_change_fixed_initial_baseline": False,
            "first_change_baseline_non_blocking": True,
            "mock_llm_mode": False,
            "decision_rule": {
                "primary_metric": "best_test_sparse_return_mean",
                "secondary_metric": "last_test_sparse_return_mean",
                "min_improvement": 0.03,
                "fallback_to_no_change": True,
            },
            "continuation": {
                "mainline_use_cuda": True,
                "save_model_interval": 300000,
                "poll_interval_seconds": 30.0,
            },
        },
        "dense_validation": {
            "enabled": False,
            "top_k_schedules": 0,
            "selection_goal": (
                "Adaptive replacement uses online checkpoint-wise replacement decisions, "
                "so an additional template-level dense validation stage is skipped."
            ),
        },
        "enable_stage4_final_spec_summary": True,
        "enable_stage5_final_full_run": False,
        "final_spec_is_summary_only": True,
        "rolling_adaptive_mainline_is_final_result": True,
        "checkpoint_retention_mode": "new_method_minimal_v1",
    }
    return profile


def _profile_8x8_qmix() -> Dict[str, Any]:
    profile = _base_profile()
    profile.update(
        {
            "profile_name": "lbf_8x8_qmix",
            "env_key": "lbforaging:Foraging-8x8-2p-1f-v3",
            "train_config": "qmix",
            "seed": 1,
            "time_limit": 50,
            "t_max": 2050000,
            "test_interval": 50000,
            "runner_log_interval": 50000,
            "learner_log_interval": 50000,
            "use_cuda": True,
            "stage3_use_cuda": True,
            "max_parallel_candidates": 6,
        }
    )
    return profile


def _profile_8x8_2p2f_qmix() -> Dict[str, Any]:
    profile = _base_profile()
    profile.update(
        {
            "profile_name": "lbf_8x8_2p2f_qmix",
            "env_key": "lbforaging:Foraging-8x8-2p-2f-coop-v3",
            "train_config": "qmix",
            "seed": 1,
            "time_limit": 50,
            "t_max": 2050000,
            "test_interval": 50000,
            "runner_log_interval": 50000,
            "learner_log_interval": 50000,
            "use_cuda": True,
            "stage3_use_cuda": True,
            "max_parallel_candidates": 6,
        }
    )
    return profile


def _profile_2s_8x8_2p2f_qmix() -> Dict[str, Any]:
    profile = _base_profile()
    profile.update(
        {
            "profile_name": "lbf_2s_8x8_2p2f_qmix",
            "env_key": "lbforaging:Foraging-2s-8x8-2p-2f-coop-v3",
            "train_config": "qmix",
            "seed": 1,
            "time_limit": 50,
            "t_max": 2050000,
            "test_interval": 50000,
            "runner_log_interval": 50000,
            "learner_log_interval": 50000,
            "use_cuda": True,
            "stage3_use_cuda": True,
            "max_parallel_candidates": 6,
        }
    )
    return profile


def _profile_10x10_qmix() -> Dict[str, Any]:
    profile = _base_profile()
    profile.update(
        {
            "profile_name": "lbf_10x10_qmix",
            "env_key": "lbforaging:Foraging-10x10-2p-1f-v3",
            "train_config": "qmix",
            "seed": 1,
            "time_limit": 50,
            "t_max": 2050000,
            "test_interval": 50000,
            "runner_log_interval": 50000,
            "learner_log_interval": 50000,
            "use_cuda": True,
            # 10x10 native branch resumes have shown GPU OOM in practice.
            "stage3_use_cuda": False,
            "max_parallel_candidates": 6,
        }
    )
    return profile


def _profile_10x10_3p3f_qmix() -> Dict[str, Any]:
    profile = _base_profile()
    profile.update(
        {
            "profile_name": "lbf_10x10_3p3f_qmix",
            "env_key": "lbforaging:Foraging-10x10-3p-3f-v3",
            "train_config": "qmix",
            "seed": 1,
            "time_limit": 50,
            "t_max": 2050000,
            "test_interval": 50000,
            "runner_log_interval": 50000,
            "learner_log_interval": 50000,
            "use_cuda": True,
            # Keep Stage 3 conservative on 10x10 branch resumes unless we measure otherwise.
            "stage3_use_cuda": False,
            "max_parallel_candidates": 6,
        }
    )
    return profile


def _profile_15x15_qmix() -> Dict[str, Any]:
    profile = _base_profile()
    profile.update(
        {
            "profile_name": "lbf_15x15_qmix",
            "env_key": "lbforaging:Foraging-15x15-4p-3f-v3",
            "train_config": "qmix",
            "seed": 1,
            "time_limit": 50,
            "t_max": 2050000,
            "test_interval": 50000,
            "runner_log_interval": 50000,
            "learner_log_interval": 50000,
            "use_cuda": True,
            # Keep Stage 3 more conservative on the largest default profile.
            "stage3_use_cuda": False,
            "max_parallel_candidates": 3,
        }
    )
    return profile


def _profile_15x15_3p4f_qmix() -> Dict[str, Any]:
    profile = _base_profile()
    profile.update(
        {
            "profile_name": "lbf_15x15_3p4f_qmix",
            "env_key": "lbforaging:Foraging-15x15-3p-4f-v3",
            "train_config": "qmix",
            "seed": 1,
            "time_limit": 50,
            "t_max": 2050000,
            "test_interval": 50000,
            "runner_log_interval": 50000,
            "learner_log_interval": 50000,
            "use_cuda": True,
            "stage3_use_cuda": False,
            "max_parallel_candidates": 3,
        }
    )
    return profile


def _apply_new_adaptive_budget_defaults(profile: Dict[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(profile)
    env_key = str(resolved.get("env_key") or "")
    if env_key.startswith("lbforaging:"):
        resolved["t_max"] = int(resolved.get("t_max") or 2050000)
    if bool(resolved.get("mappo_force_full_budget_profile", False)):
        resolved["t_max"] = 20000000
    return resolved


def _profile_8x8_qmix_adaptive() -> Dict[str, Any]:
    profile = _base_adaptive_profile()
    profile.update(_profile_8x8_qmix())
    profile["profile_name"] = "lbf_8x8_qmix_adaptive"
    profile["phase1_method"] = deepcopy(_base_adaptive_profile()["phase1_method"])
    return _apply_new_adaptive_budget_defaults(profile)


def _profile_8x8_2p2f_qmix_adaptive() -> Dict[str, Any]:
    profile = _base_adaptive_profile()
    profile.update(_profile_8x8_2p2f_qmix())
    profile["profile_name"] = "lbf_8x8_2p2f_qmix_adaptive"
    profile["phase1_method"] = deepcopy(_base_adaptive_profile()["phase1_method"])
    return _apply_new_adaptive_budget_defaults(profile)


def _profile_2s_8x8_2p2f_qmix_adaptive() -> Dict[str, Any]:
    profile = _base_adaptive_profile()
    profile.update(_profile_2s_8x8_2p2f_qmix())
    profile["profile_name"] = "lbf_2s_8x8_2p2f_qmix_adaptive"
    profile["phase1_method"] = deepcopy(_base_adaptive_profile()["phase1_method"])
    return _apply_new_adaptive_budget_defaults(profile)


def _profile_10x10_qmix_adaptive() -> Dict[str, Any]:
    profile = _base_adaptive_profile()
    profile.update(_profile_10x10_qmix())
    profile["profile_name"] = "lbf_10x10_qmix_adaptive"
    profile["phase1_method"] = deepcopy(_base_adaptive_profile()["phase1_method"])
    return _apply_new_adaptive_budget_defaults(profile)


def _profile_10x10_3p3f_qmix_adaptive() -> Dict[str, Any]:
    profile = _base_adaptive_profile()
    profile.update(_profile_10x10_3p3f_qmix())
    profile["profile_name"] = "lbf_10x10_3p3f_qmix_adaptive"
    profile["phase1_method"] = deepcopy(_base_adaptive_profile()["phase1_method"])
    return _apply_new_adaptive_budget_defaults(profile)


def _profile_15x15_3p4f_qmix_adaptive() -> Dict[str, Any]:
    profile = _base_adaptive_profile()
    profile.update(_profile_15x15_3p4f_qmix())
    profile["profile_name"] = "lbf_15x15_3p4f_qmix_adaptive"
    profile["phase1_method"] = deepcopy(_base_adaptive_profile()["phase1_method"])
    return _apply_new_adaptive_budget_defaults(profile)


def _profile_qmix_dual_llm_single_seed_pilot() -> Dict[str, Any]:
    profile = _profile_10x10_3p3f_qmix_adaptive()
    profile["profile_name"] = "qmix_dual_llm_single_seed_pilot"
    profile["seed"] = 1
    profile["train_config"] = "qmix"
    profile["env_key"] = "lbforaging:Foraging-10x10-3p-3f-v3"
    profile["t_max"] = 2050000
    profile["use_cuda"] = True
    profile["stage1_use_cuda"] = True
    # Keep branch resumes conservative on the larger map until the real pilot measures GPU headroom.
    profile["stage3_use_cuda"] = False
    profile["stage5_use_cuda"] = True
    profile["max_parallel_candidates"] = 6
    profile["branch_budget_steps"] = 300000
    profile["save_model_interval"] = 50000
    profile["local_results_path"] = "results"
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["dense_reference_target_t_max"] = 2050000
    phase1_method["adaptive_mainline_target_t_max"] = 2050000
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["fallback_to_single_llm_initial_config"] = True
    phase1_method["keep_dense_reference_continuation"] = True
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["use_fixed_intervention_checkpoints"] = True
    phase1_method["fixed_intervention_checkpoints"] = [1000000, 1500000]
    phase1_method["checkpoint_count"] = 2
    phase1_method["initial_dense_search"]["rounds"] = 2
    phase1_method["initial_dense_search"]["candidates_per_round"] = 3
    phase1_method["initial_dense_search"]["candidate_budget_steps"] = 850000
    phase1_method["initial_dense_search"]["candidate_target_checkpoint_step"] = 800000
    phase1_method["initial_dense_search"]["selected_endpoint_target_step"] = 800000
    phase1_method["initial_dense_search"]["endpoint_tolerance_steps"] = 50000
    phase1_method["initial_dense_search"]["endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    phase1_method["initial_dense_search"]["parallel"] = True
    phase1_method["initial_dense_search"]["save_endpoint_checkpoint"] = True
    phase1_method["initial_dense_search"]["save_intermediate_checkpoints"] = False
    phase1_method["initial_dense_search"]["keep_only_endpoint_checkpoint"] = True
    phase1_method["adaptive_replacement"]["fixed_intervention_checkpoints"] = [1000000, 1500000]
    phase1_method["adaptive_replacement"]["use_fixed_intervention_checkpoints"] = True
    phase1_method["adaptive_replacement"]["branch_budget_steps"] = 300000
    phase1_method["adaptive_replacement"]["stage3_branch_rounds"] = 2
    phase1_method["adaptive_replacement"]["stage3_candidates_per_round"] = 3
    phase1_method["adaptive_replacement"]["stage3_include_no_change_control"] = True
    phase1_method["adaptive_replacement"]["stage3_no_change_counts_toward_round_budget"] = True
    phase1_method["adaptive_replacement"]["stage3_candidate_review_enabled"] = True
    phase1_method["adaptive_replacement"]["stage3_candidate_repair_max_rounds"] = 1
    phase1_method["adaptive_replacement"]["use_winner_branch_promotion"] = True
    phase1_method["adaptive_replacement"]["fallback_to_existing_stage3"] = True
    phase1_method["adaptive_replacement"]["launch_first_change_fixed_initial_baseline"] = False
    phase1_method["adaptive_replacement"]["mock_llm_mode"] = False
    phase1_method["enable_stage4_final_spec_summary"] = True
    phase1_method["enable_stage5_final_full_run"] = False
    phase1_method["final_spec_is_summary_only"] = True
    phase1_method["rolling_adaptive_mainline_is_final_result"] = True
    phase1_method["checkpoint_retention_mode"] = "new_method_minimal_v1"
    phase1_method["fixed_checkpoint_plan"] = [
        {
            "name": "C1",
            "step": 1000000,
            "stage_label": "post_initialization_transition",
            "intervention_question": "Does the early-selected dense reward configuration remain suitable after the initial 800k dense search phase?",
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
            "intervention_question": "Does the current reward configuration remain stable and task-aligned in the later training phase?",
            "recommended_candidate_focus": [
                "beta_down",
                "wc_down_wp_up",
                "reference_like",
                "recovery",
            ],
            "branch_budget_feasible": True,
        },
    ]
    profile["phase1_method"] = phase1_method
    profile["initial_dense_candidate_budget_steps"] = 850000
    profile["initial_dense_candidate_target_checkpoint_step"] = 800000
    profile["stage1b_selected_endpoint_target_step"] = 800000
    profile["stage1b_endpoint_tolerance_steps"] = 50000
    profile["stage1b_endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    profile["fallback_candidates_run_in_normal_path"] = False
    profile["enable_first_change_fixed_initial_baseline"] = False
    profile["enable_stage4_final_spec_summary"] = True
    profile["enable_stage5_final_full_run"] = False
    profile["final_spec_is_summary_only"] = True
    profile["rolling_adaptive_mainline_is_final_result"] = True
    profile["checkpoint_retention_mode"] = "new_method_minimal_v1"
    profile["stage1_sparse_save_checkpoints"] = False
    profile["stage1b_candidate_save_endpoint_checkpoint"] = True
    profile["stage1b_candidate_save_intermediate_checkpoints"] = False
    profile["stage1b_keep_only_endpoint_checkpoint"] = True
    profile["dense_reference_save_checkpoints"] = False
    profile["adaptive_mainline_save_c1_c2_source_checkpoints"] = True
    profile["adaptive_mainline_save_intermediate_checkpoints"] = False
    profile["adaptive_mainline_save_final_checkpoint"] = False
    profile["stage3_branch_save_endpoint_checkpoint"] = True
    profile["stage3_branch_save_intermediate_checkpoints"] = False
    profile["delete_failed_branch_checkpoints"] = False
    profile["delete_loser_branch_checkpoints"] = False
    profile["auto_checkpoint_cleanup"] = False
    profile["checkpoint_cleanup_dry_run_only"] = True
    return _apply_new_adaptive_budget_defaults(profile)


def _apply_formal_qmix_single_seed_overrides(profile: Dict[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(profile)
    resolved["seed"] = 1
    resolved["train_config"] = "qmix"
    resolved["t_max"] = 2050000
    resolved["use_cuda"] = True
    resolved["stage1_use_cuda"] = True
    resolved["branch_budget_steps"] = 300000
    resolved["stage3_branch_rounds"] = 2
    resolved["stage3_candidates_per_round"] = 3
    resolved["stage3_include_no_change_control"] = True
    resolved["stage3_candidate_review_enabled"] = True
    resolved["use_winner_branch_promotion"] = True
    resolved["save_model_interval"] = 50000
    resolved["local_results_path"] = "results"
    phase1_method = deepcopy(resolved["phase1_method"])
    phase1_method["dense_reference_target_t_max"] = 2050000
    phase1_method["adaptive_mainline_target_t_max"] = 2050000
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["fallback_to_single_llm_initial_config"] = True
    phase1_method["keep_dense_reference_continuation"] = True
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["use_fixed_intervention_checkpoints"] = True
    phase1_method["fixed_intervention_checkpoints"] = [1000000, 1500000]
    phase1_method["checkpoint_count"] = 2
    phase1_method["initial_dense_search"]["rounds"] = 2
    phase1_method["initial_dense_search"]["candidates_per_round"] = 3
    phase1_method["initial_dense_search"]["candidate_budget_steps"] = 850000
    phase1_method["initial_dense_search"]["candidate_target_checkpoint_step"] = 800000
    phase1_method["initial_dense_search"]["selected_endpoint_target_step"] = 800000
    phase1_method["initial_dense_search"]["endpoint_tolerance_steps"] = 50000
    phase1_method["initial_dense_search"]["endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    phase1_method["initial_dense_search"]["parallel"] = True
    phase1_method["initial_dense_search"]["save_endpoint_checkpoint"] = True
    phase1_method["initial_dense_search"]["save_intermediate_checkpoints"] = False
    phase1_method["initial_dense_search"]["keep_only_endpoint_checkpoint"] = True
    phase1_method["adaptive_replacement"]["fixed_intervention_checkpoints"] = [1000000, 1500000]
    phase1_method["adaptive_replacement"]["use_fixed_intervention_checkpoints"] = True
    phase1_method["adaptive_replacement"]["branch_budget_steps"] = 300000
    phase1_method["adaptive_replacement"]["stage3_branch_rounds"] = 2
    phase1_method["adaptive_replacement"]["stage3_candidates_per_round"] = 3
    phase1_method["adaptive_replacement"]["stage3_include_no_change_control"] = True
    phase1_method["adaptive_replacement"]["stage3_no_change_counts_toward_round_budget"] = True
    phase1_method["adaptive_replacement"]["stage3_candidate_review_enabled"] = True
    phase1_method["adaptive_replacement"]["stage3_candidate_repair_max_rounds"] = 1
    phase1_method["adaptive_replacement"]["use_winner_branch_promotion"] = True
    phase1_method["adaptive_replacement"]["fallback_to_existing_stage3"] = True
    phase1_method["adaptive_replacement"]["launch_first_change_fixed_initial_baseline"] = False
    phase1_method["adaptive_replacement"]["mock_llm_mode"] = False
    phase1_method["enable_stage4_final_spec_summary"] = True
    phase1_method["enable_stage5_final_full_run"] = False
    phase1_method["final_spec_is_summary_only"] = True
    phase1_method["rolling_adaptive_mainline_is_final_result"] = True
    phase1_method["checkpoint_retention_mode"] = "new_method_minimal_v1"
    phase1_method["fixed_checkpoint_plan"] = [
        {
            "name": "C1",
            "step": 1000000,
            "stage_label": "post_initialization_transition",
            "intervention_question": "Does the early-selected dense reward configuration remain suitable after the initial 800k dense search phase?",
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
            "intervention_question": "Does the current reward configuration remain stable and task-aligned in the later training phase?",
            "recommended_candidate_focus": [
                "beta_down",
                "wc_down_wp_up",
                "reference_like",
                "recovery",
            ],
            "branch_budget_feasible": True,
        },
    ]
    resolved["phase1_method"] = phase1_method
    resolved["initial_dense_candidate_budget_steps"] = 850000
    resolved["initial_dense_candidate_target_checkpoint_step"] = 800000
    resolved["stage1b_selected_endpoint_target_step"] = 800000
    resolved["stage1b_endpoint_tolerance_steps"] = 50000
    resolved["stage1b_endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    resolved["fallback_candidates_run_in_normal_path"] = False
    resolved["enable_first_change_fixed_initial_baseline"] = False
    resolved["enable_stage4_final_spec_summary"] = True
    resolved["enable_stage5_final_full_run"] = False
    resolved["final_spec_is_summary_only"] = True
    resolved["rolling_adaptive_mainline_is_final_result"] = True
    resolved["checkpoint_retention_mode"] = "new_method_minimal_v1"
    resolved["stage1_sparse_save_checkpoints"] = False
    resolved["stage1b_candidate_save_endpoint_checkpoint"] = True
    resolved["stage1b_candidate_save_intermediate_checkpoints"] = False
    resolved["stage1b_keep_only_endpoint_checkpoint"] = True
    resolved["dense_reference_save_checkpoints"] = False
    resolved["adaptive_mainline_save_c1_c2_source_checkpoints"] = True
    resolved["adaptive_mainline_save_intermediate_checkpoints"] = False
    resolved["adaptive_mainline_save_final_checkpoint"] = False
    resolved["stage3_branch_save_endpoint_checkpoint"] = True
    resolved["stage3_branch_save_intermediate_checkpoints"] = False
    resolved["delete_failed_branch_checkpoints"] = False
    resolved["delete_loser_branch_checkpoints"] = False
    resolved["auto_checkpoint_cleanup"] = False
    resolved["checkpoint_cleanup_dry_run_only"] = True
    return _apply_new_adaptive_budget_defaults(resolved)


def _apply_policy_guidance_profile_overrides(profile: Dict[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(profile)
    resolved["enable_policy_guidance"] = True
    resolved["policy_guidance_mode"] = "context_only"
    resolved["policy_guidance_direct_action_override"] = False
    resolved["policy_guidance_modify_learner"] = False
    resolved["policy_guidance_fallback_to_reward_only"] = True
    resolved["policy_guidance_collect_stage1_milestones"] = True
    resolved["policy_guidance_points"] = [
        "stage1_after_sparse_baseline",
        "stage3_c1_pre_checkpoint",
        "stage3_c1_after_round1",
        "stage3_c2_pre_checkpoint",
        "stage3_c2_after_round1",
    ]
    resolved["policy_guidance_stage1_milestones"] = [200000, 500000, 800000, 2050000]
    resolved["policy_guidance_eval_episodes_stage1_per_milestone"] = 10
    resolved["policy_guidance_stage1_milestone_match_tolerance"] = 25000
    resolved["policy_guidance_c1_pre_eval_episodes"] = 10
    resolved["policy_guidance_c1_after_round1_eval_episodes_per_branch"] = 5
    resolved["policy_guidance_c2_pre_eval_episodes"] = 10
    resolved["policy_guidance_c2_after_round1_eval_episodes_per_branch"] = 10
    resolved["policy_guidance_collect_stage1_intervals"] = False
    resolved["policy_guidance_collect_stage1_final"] = True
    resolved["policy_guidance_save_raw_trajectories"] = False
    resolved["policy_guidance_save_compact_episode_summaries"] = True
    resolved["policy_guidance_stage1_behavior_summary_required"] = False
    resolved["policy_guidance_stage1_behavior_summary_path"] = None
    resolved["policy_guidance_stage1_behavior_summary_reuse_allowed"] = False
    resolved["policy_guidance_stage1_behavior_summary_must_be_fresh"] = True
    resolved["allow_reuse_stage1_behavior_summary"] = False
    resolved["policy_guidance_stage1b_use_integrated_guidance"] = True
    resolved["policy_guidance_stage1b_fallback_to_reward_only"] = True
    resolved["policy_guidance_stage3_use_integrated_guidance"] = True
    resolved["policy_guidance_stage3_fallback_to_reward_only"] = True
    resolved["policy_guidance_stage3_behavior_summary_required"] = False
    resolved["policy_guidance_stage3_record_manifest"] = True
    resolved["policy_guidance_stage3_record_memory"] = True
    resolved["policy_guidance_first_supported_envs"] = ["lbforaging"]
    resolved["policy_guidance_extractor_preference"] = "env_state_then_obs_action"

    phase1_method = deepcopy(resolved["phase1_method"])
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = True
    adaptive_replacement["policy_guidance_mode"] = "context_only"
    adaptive_replacement["policy_guidance_direct_action_override"] = False
    adaptive_replacement["policy_guidance_modify_learner"] = False
    adaptive_replacement["policy_guidance_fallback_to_reward_only"] = True
    adaptive_replacement["policy_guidance_collect_stage1_milestones"] = True
    adaptive_replacement["policy_guidance_points"] = list(
        resolved["policy_guidance_points"]
    )
    adaptive_replacement["policy_guidance_stage1_milestones"] = list(
        resolved["policy_guidance_stage1_milestones"]
    )
    adaptive_replacement["policy_guidance_eval_episodes_stage1_per_milestone"] = int(
        resolved["policy_guidance_eval_episodes_stage1_per_milestone"]
    )
    adaptive_replacement["policy_guidance_stage1_milestone_match_tolerance"] = int(
        resolved["policy_guidance_stage1_milestone_match_tolerance"]
    )
    adaptive_replacement["policy_guidance_c1_pre_eval_episodes"] = int(
        resolved["policy_guidance_c1_pre_eval_episodes"]
    )
    adaptive_replacement[
        "policy_guidance_c1_after_round1_eval_episodes_per_branch"
    ] = int(resolved["policy_guidance_c1_after_round1_eval_episodes_per_branch"])
    adaptive_replacement["policy_guidance_c2_pre_eval_episodes"] = int(
        resolved["policy_guidance_c2_pre_eval_episodes"]
    )
    adaptive_replacement[
        "policy_guidance_c2_after_round1_eval_episodes_per_branch"
    ] = int(resolved["policy_guidance_c2_after_round1_eval_episodes_per_branch"])
    adaptive_replacement["policy_guidance_collect_stage1_intervals"] = False
    adaptive_replacement["policy_guidance_collect_stage1_final"] = True
    adaptive_replacement["policy_guidance_save_raw_trajectories"] = False
    adaptive_replacement["policy_guidance_save_compact_episode_summaries"] = True
    adaptive_replacement["policy_guidance_stage1_behavior_summary_required"] = False
    adaptive_replacement["policy_guidance_stage1_behavior_summary_path"] = None
    adaptive_replacement["policy_guidance_stage1_behavior_summary_reuse_allowed"] = False
    adaptive_replacement["policy_guidance_stage1_behavior_summary_must_be_fresh"] = True
    adaptive_replacement["allow_reuse_stage1_behavior_summary"] = False
    adaptive_replacement["policy_guidance_stage1b_use_integrated_guidance"] = True
    adaptive_replacement["policy_guidance_stage1b_fallback_to_reward_only"] = True
    adaptive_replacement["policy_guidance_stage3_use_integrated_guidance"] = True
    adaptive_replacement["policy_guidance_stage3_fallback_to_reward_only"] = True
    adaptive_replacement["policy_guidance_stage3_behavior_summary_required"] = False
    adaptive_replacement["policy_guidance_stage3_record_manifest"] = True
    adaptive_replacement["policy_guidance_stage3_record_memory"] = True
    adaptive_replacement["policy_guidance_first_supported_envs"] = ["lbforaging"]
    adaptive_replacement["policy_guidance_extractor_preference"] = (
        "env_state_then_obs_action"
    )
    phase1_method["adaptive_replacement"] = adaptive_replacement
    resolved["phase1_method"] = phase1_method
    return _apply_new_adaptive_budget_defaults(resolved)


def _profile_qmix_lbf_8x8_2p_2f_dual_llm_single_seed_pilot() -> Dict[str, Any]:
    profile = _profile_8x8_2p2f_qmix_adaptive()
    profile["profile_name"] = "qmix_lbf_8x8_2p_2f_dual_llm_single_seed_pilot"
    return _apply_formal_qmix_single_seed_overrides(profile)


def _profile_qmix_lbf_2s_8x8_2p_2f_dual_llm_single_seed_pilot() -> Dict[str, Any]:
    profile = _profile_2s_8x8_2p2f_qmix_adaptive()
    profile["profile_name"] = "qmix_lbf_2s_8x8_2p_2f_dual_llm_single_seed_pilot"
    return _apply_formal_qmix_single_seed_overrides(profile)


def _profile_qmix_dual_llm_policy_guided_single_seed_pilot() -> Dict[str, Any]:
    profile = _profile_qmix_dual_llm_single_seed_pilot()
    profile["profile_name"] = "qmix_dual_llm_policy_guided_single_seed_pilot"
    return _apply_policy_guidance_profile_overrides(profile)


def _profile_qmix_policy_guidance_stage1_tiny_smoke() -> Dict[str, Any]:
    profile = _profile_8x8_qmix()
    profile.update(
        {
            "profile_name": "qmix_policy_guidance_stage1_tiny_smoke",
            "seed": 1,
            "train_config": "qmix",
            "env_key": "lbforaging:Foraging-8x8-2p-1f-v3",
            "use_cuda": False,
            "stage1_use_cuda": False,
            "stage3_use_cuda": False,
            "stage5_use_cuda": False,
            "t_max": 10000,
            "test_interval": 2000,
            "runner_log_interval": 2000,
            "learner_log_interval": 2000,
            "save_model_interval": 0,
            "save_model": False,
            "budget_profile": "policy_stage1_tiny_smoke",
            "local_results_path": "results",
        }
    )
    profile = _apply_policy_guidance_profile_overrides(profile)
    profile["policy_guidance_stage1_milestones"] = [2000, 5000, 8000, 10000]
    profile["policy_guidance_eval_episodes_stage1_per_milestone"] = 2
    profile["policy_guidance_collect_stage1_milestones"] = True
    profile["policy_guidance_stage1_milestone_match_tolerance"] = 250
    profile["policy_guidance_save_raw_trajectories"] = False
    profile["policy_guidance_save_compact_episode_summaries"] = True
    return profile


def _profile_qmix_policy_guidance_stage1b_tiny_smoke() -> Dict[str, Any]:
    profile = _profile_dual_llm_stage1b_tiny_smoke()
    profile["profile_name"] = "qmix_policy_guidance_stage1b_tiny_smoke"
    profile = _apply_policy_guidance_profile_overrides(profile)
    profile["mock_llm_mode"] = True
    profile["use_real_llm"] = False
    profile["policy_guidance_stage1_milestones"] = [2000, 5000, 8000, 10000]
    profile["policy_guidance_eval_episodes_stage1_per_milestone"] = 2
    profile["policy_guidance_stage1_behavior_summary_required"] = False
    profile["policy_guidance_stage1b_use_integrated_guidance"] = True
    profile["policy_guidance_stage1b_fallback_to_reward_only"] = True
    profile["initial_dense_candidates_per_round"] = 2
    profile["initial_dense_candidate_budget_steps"] = 5000
    profile["initial_dense_candidate_launch_enabled"] = False
    return profile


def _profile_qmix_policy_guidance_stage3_hook_tiny_smoke() -> Dict[str, Any]:
    profile = _profile_dual_llm_stage3_tiny_execute_smoke()
    profile["profile_name"] = "qmix_policy_guidance_stage3_hook_tiny_smoke"
    profile = _apply_policy_guidance_profile_overrides(profile)
    profile["mock_llm_mode"] = True
    profile["use_real_llm"] = False
    profile["branch_budget_steps"] = 3000
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 2
    profile["stage3_include_no_change_control"] = True
    return profile


def _profile_qmix_policy_guided_tiny_end_to_end_smoke() -> Dict[str, Any]:
    profile = _profile_dual_llm_stage1b_tiny_smoke()
    profile["profile_name"] = "qmix_policy_guided_tiny_end_to_end_smoke"
    profile = _apply_policy_guidance_profile_overrides(profile)
    profile["mock_llm_mode"] = True
    profile["use_real_llm"] = False
    profile["policy_guidance_mode"] = "context_only"
    profile["policy_guidance_direct_action_override"] = False
    profile["policy_guidance_modify_learner"] = False
    profile["policy_guidance_fallback_to_reward_only"] = True
    profile["policy_guidance_stage1_milestones"] = [2000, 5000, 8000, 10000]
    profile["policy_guidance_eval_episodes_stage1_per_milestone"] = 2
    profile["use_dual_llm_stage1b_initial_search"] = True
    profile["initial_dense_search_rounds"] = 1
    profile["initial_dense_candidates_per_round"] = 2
    profile["initial_dense_candidate_budget_steps"] = 5000
    profile["initial_dense_candidate_launch_enabled"] = False
    profile["use_fixed_intervention_checkpoints"] = True
    profile["fixed_intervention_checkpoints"] = [12000, 16000]
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 2
    profile["stage3_include_no_change_control"] = True
    profile["branch_budget_steps"] = 3000
    profile["stage3_branch_launch_enabled"] = False
    profile["long_training_enabled"] = False
    phase1_method = deepcopy(profile["phase1_method"])
    initial_dense_search = deepcopy(phase1_method.get("initial_dense_search") or {})
    initial_dense_search["rounds"] = 1
    initial_dense_search["candidates_per_round"] = 2
    initial_dense_search["candidate_budget_steps"] = 5000
    phase1_method["initial_dense_search"] = initial_dense_search
    phase1_method["fixed_intervention_checkpoints"] = [12000, 16000]
    phase1_method["checkpoint_count"] = 2
    fixed_checkpoint_plan = list(phase1_method.get("fixed_checkpoint_plan") or [])
    if len(fixed_checkpoint_plan) >= 2:
        fixed_checkpoint_plan[0]["step"] = 12000
        fixed_checkpoint_plan[1]["step"] = 16000
    phase1_method["fixed_checkpoint_plan"] = fixed_checkpoint_plan
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["fixed_intervention_checkpoints"] = [12000, 16000]
    adaptive_replacement["branch_budget_steps"] = 3000
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 2
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["mock_llm_mode"] = True
    adaptive_replacement["stage3_branch_launch_enabled"] = False
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _profile_qmix_policy_guided_tiny_execute_smoke() -> Dict[str, Any]:
    profile = _profile_qmix_policy_guided_tiny_end_to_end_smoke()
    profile["profile_name"] = "qmix_policy_guided_tiny_execute_smoke"
    profile["budget_profile"] = "policy_guided_tiny_execute_smoke"
    profile["reuse_policy_guidance_cache"] = True
    profile["mock_llm_mode"] = False
    profile["use_real_llm"] = False
    profile["long_training_enabled"] = False
    profile["initial_dense_candidate_launch_enabled"] = True
    profile["stage3_branch_launch_enabled"] = True
    profile["initial_dense_candidate_target_checkpoint_step"] = 5000
    phase1_method = deepcopy(profile["phase1_method"])
    initial_dense_search = deepcopy(phase1_method.get("initial_dense_search") or {})
    initial_dense_search["candidate_target_checkpoint_step"] = 5000
    initial_dense_search["selected_endpoint_target_step"] = 5000
    initial_dense_search["parallel"] = False
    phase1_method["initial_dense_search"] = initial_dense_search
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["stage3_branch_launch_enabled"] = True
    adaptive_replacement["mock_llm_mode"] = False
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _profile_qmix_policy_guided_sidecar_stage3_budget_smoke_seed1() -> Dict[str, Any]:
    profile = _profile_qmix_policy_guided_tiny_execute_smoke()
    profile["profile_name"] = "qmix_policy_guided_sidecar_stage3_budget_smoke_seed1"
    profile["budget_profile"] = "policy_guided_sidecar_stage3_budget_smoke"
    profile["seed"] = 1
    profile["train_config"] = "qmix"
    profile["env_key"] = "lbforaging:Foraging-8x8-2p-1f-v3"
    profile["use_cuda"] = False
    profile["stage1_use_cuda"] = False
    profile["stage3_use_cuda"] = False
    profile["stage5_use_cuda"] = False
    profile["t_max"] = 19000
    profile["branch_budget_steps"] = 3000
    profile["max_parallel_candidates"] = 3
    profile["use_real_llm"] = False
    profile["use_real_llm_for_field_analysis"] = False
    profile["mock_llm_mode"] = False
    profile["reuse_policy_guidance_cache"] = True
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 3
    profile["stage3_update_candidates_per_round"] = 3
    profile["stage3_include_no_change_control"] = True
    profile["stage3_no_change_counts_toward_round_budget"] = False
    profile["stage3_reuse_no_change_across_rounds"] = True
    profile["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    profile["stage3_require_nontrivial_config_delta"] = True
    profile["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    profile["stage3_require_candidate_type_diversity"] = True
    profile["policy_guided_stage3_require_behavior_evidence"] = True
    phase1_method = deepcopy(profile["phase1_method"])
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["branch_budget_steps"] = 3000
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 3
    adaptive_replacement["stage3_update_candidates_per_round"] = 3
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["stage3_no_change_counts_toward_round_budget"] = False
    adaptive_replacement["stage3_reuse_no_change_across_rounds"] = True
    adaptive_replacement["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    adaptive_replacement["stage3_require_nontrivial_config_delta"] = True
    adaptive_replacement["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    adaptive_replacement["stage3_require_candidate_type_diversity"] = True
    adaptive_replacement["policy_guided_stage3_require_behavior_evidence"] = True
    adaptive_replacement["stage3_branch_launch_enabled"] = True
    adaptive_replacement["mock_llm_mode"] = False
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _profile_qmix_policy_guided_small_pilot_seed1() -> Dict[str, Any]:
    profile = _profile_8x8_qmix_adaptive()
    profile["profile_name"] = "qmix_policy_guided_small_pilot_seed1"
    profile = _apply_policy_guidance_profile_overrides(profile)
    profile["seed"] = 1
    profile["train_config"] = "qmix"
    profile["env_key"] = "lbforaging:Foraging-8x8-2p-1f-v3"
    profile["use_cuda"] = False
    profile["stage1_use_cuda"] = False
    profile["stage3_use_cuda"] = False
    profile["stage5_use_cuda"] = False
    profile["t_max"] = 50000
    profile["test_interval"] = 5000
    profile["runner_log_interval"] = 5000
    profile["learner_log_interval"] = 5000
    profile["save_model_interval"] = 5000
    profile["budget_profile"] = "policy_guided_small_pilot"
    profile["local_results_path"] = "results"
    profile["long_training_enabled"] = False
    profile["use_real_llm"] = True
    profile["use_real_llm_for_field_analysis"] = True
    profile["mock_llm_mode"] = False
    profile["llm_model_routing"] = {
        "enabled": True,
        "validation_provider": "deepseek",
        "validation_model": "deepseek-chat",
        "formal_provider": "openai",
        "formal_model": "gpt-5.2",
        "formal_only_when_allow_formal_execute": True,
    }
    profile["use_cuda"] = True
    profile["stage1_use_cuda"] = True
    profile["stage3_use_cuda"] = True
    profile["stage5_use_cuda"] = True
    profile["max_parallel_candidates"] = 6
    profile["reuse_policy_guidance_cache"] = True
    profile["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    profile["policy_guidance_stage1_milestones"] = [5000, 10000, 20000, 30000]
    profile["policy_guidance_eval_episodes_stage1_per_milestone"] = 2
    profile["policy_guidance_stage1_milestone_match_tolerance"] = 500
    profile["policy_guidance_collect_stage1_milestones"] = True
    profile["policy_guidance_stage1_behavior_summary_reuse_allowed"] = False
    profile["policy_guidance_stage1_behavior_summary_must_be_fresh"] = True
    profile["allow_reuse_stage1_behavior_summary"] = False
    profile["stage1_sparse_save_checkpoints"] = False
    profile["use_dual_llm_stage1b_initial_search"] = True
    profile["initial_dense_search_rounds"] = 1
    profile["initial_dense_candidates_per_round"] = 2
    profile["initial_dense_candidate_budget_steps"] = 10000
    profile["initial_dense_candidate_target_checkpoint_step"] = 10000
    profile["stage1b_selected_endpoint_target_step"] = 10000
    profile["initial_dense_candidate_parallel"] = False
    profile["stage1b_candidate_save_endpoint_checkpoint"] = True
    profile["stage1b_candidate_save_intermediate_checkpoints"] = False
    profile["stage1b_keep_only_endpoint_checkpoint"] = True
    profile["dense_reference_save_checkpoints"] = False
    profile["enable_dense_reference"] = True
    profile["dense_reference_t_max"] = 60000
    profile["adaptive_mainline_target_t_max"] = 60000
    profile["dense_reference_target_t_max"] = 60000
    profile["use_fixed_intervention_checkpoints"] = True
    profile["fixed_intervention_checkpoints"] = [35000, 50000]
    profile["branch_budget_steps"] = 10000
    profile["max_parallel_candidates"] = 2
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 2
    profile["stage3_include_no_change_control"] = True
    profile["stage3_branch_save_endpoint_checkpoint"] = True
    profile["stage3_branch_save_intermediate_checkpoints"] = False
    profile["checkpoint_retention_mode"] = "new_method_minimal_v1"
    profile["adaptive_mainline_save_c1_c2_source_checkpoints"] = True
    profile["adaptive_mainline_save_intermediate_checkpoints"] = False
    profile["adaptive_mainline_save_final_checkpoint"] = False
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["dense_reference_target_t_max"] = 60000
    phase1_method["adaptive_mainline_target_t_max"] = 60000
    phase1_method["reuse_policy_guidance_cache"] = True
    phase1_method["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    phase1_method["checkpoint_retention_mode"] = "new_method_minimal_v1"
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["keep_dense_reference_continuation"] = True
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["use_fixed_intervention_checkpoints"] = True
    phase1_method["fixed_intervention_checkpoints"] = [35000, 50000]
    phase1_method["checkpoint_count"] = 2
    initial_dense_search = deepcopy(phase1_method.get("initial_dense_search") or {})
    initial_dense_search["enabled"] = True
    initial_dense_search["rounds"] = 1
    initial_dense_search["candidates_per_round"] = 2
    initial_dense_search["candidate_budget_steps"] = 10000
    initial_dense_search["candidate_target_checkpoint_step"] = 10000
    initial_dense_search["selected_endpoint_target_step"] = 10000
    initial_dense_search["endpoint_tolerance_steps"] = 2000
    initial_dense_search["parallel"] = False
    initial_dense_search["save_endpoint_checkpoint"] = True
    initial_dense_search["save_intermediate_checkpoints"] = False
    initial_dense_search["keep_only_endpoint_checkpoint"] = True
    phase1_method["initial_dense_search"] = initial_dense_search
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["fixed_intervention_checkpoints"] = [35000, 50000]
    adaptive_replacement["branch_budget_steps"] = 10000
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 2
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["use_winner_branch_promotion"] = True
    adaptive_replacement["mock_llm_mode"] = True
    adaptive_replacement["reuse_policy_guidance_cache"] = True
    adaptive_replacement["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    adaptive_replacement["policy_guidance_stage1_milestones"] = [5000, 10000, 20000, 30000]
    adaptive_replacement["policy_guidance_eval_episodes_stage1_per_milestone"] = 2
    adaptive_replacement["policy_guidance_stage1_milestone_match_tolerance"] = 500
    adaptive_replacement["policy_guidance_stage1_behavior_summary_reuse_allowed"] = False
    adaptive_replacement["policy_guidance_stage1_behavior_summary_must_be_fresh"] = True
    adaptive_replacement["allow_reuse_stage1_behavior_summary"] = False
    adaptive_replacement["continuation"] = {
        "mainline_use_cuda": False,
        "save_model_interval": 5000,
        "poll_interval_seconds": 30.0,
    }
    phase1_method["adaptive_replacement"] = adaptive_replacement
    phase1_method["fixed_checkpoint_plan"] = [
        {
            "name": "C1",
            "step": 35000,
            "stage_label": "post_initialization_transition",
            "intervention_question": "Small-pilot checkpoint after the selected Stage 1b initialization source.",
            "recommended_candidate_focus": ["reference_like", "recovery", "beta_down"],
            "branch_budget_feasible": True,
        },
        {
            "name": "C2",
            "step": 50000,
            "stage_label": "late_stability",
            "intervention_question": "Small-pilot late checkpoint for adaptive stability validation.",
            "recommended_candidate_focus": ["reference_like", "beta_down", "wc_down_wp_up"],
            "branch_budget_feasible": True,
        },
    ]
    profile["phase1_method"] = phase1_method
    return profile


def _build_single_llm_pbrs_v2_reward_gen_profile(
    *,
    profile_name: str,
    algorithm: str,
    env_key: str,
    formal_t_max: int,
    candidate_budget_steps: int | None = None,
) -> Dict[str, Any]:
    from workflows.single_llm_pbrs_v2_baseline import default_single_llm_baseline_profile

    profile = deepcopy(default_single_llm_baseline_profile())
    resolved_candidate_budget_steps = int(
        candidate_budget_steps
        if candidate_budget_steps is not None
        else int(profile.get("candidate_budget_steps") or 850000)
    )
    profile["profile_name"] = profile_name
    profile["workflow_profile"] = profile_name
    profile["default_workflow_id_suffix"] = "run1"
    profile["algorithm"] = str(algorithm)
    profile["train_config"] = str(algorithm)
    profile["env_key"] = str(env_key)
    profile["seed"] = 1
    profile["pbrs_version"] = "lbf_pbrs_v2"
    profile["candidate_rounds"] = 2
    profile["candidates_per_round"] = 3
    profile["candidate_budget_steps"] = resolved_candidate_budget_steps
    profile["full_training_t_max"] = int(formal_t_max)
    profile["formal_t_max"] = int(formal_t_max)
    profile["eval_use_pbrs"] = False
    profile["apply_dense_reward_in_eval"] = False
    profile["test_sparse_only"] = True
    profile["enable_policy_guidance"] = False
    profile["critic_enabled"] = False
    profile["stage3_enabled"] = False
    profile["adaptive_replacement"] = False
    profile["winner_promotion"] = False
    profile["no_change_gate"] = False
    profile["final_staged_adaptive_spec"] = False
    profile["critic_disabled"] = True
    profile["policy_guidance_disabled"] = True
    profile["stage3_disabled"] = True
    profile["no_change_disabled"] = True
    profile["adaptive_replacement_disabled"] = True
    profile["winner_selection_rule_based"] = True
    profile["deterministic_fallback_disabled"] = True
    profile["single_llm_baseline"] = True
    profile["fallback_to_deterministic_on_llm_error"] = False
    profile["add_deterministic_recovery_fallback"] = False
    phase1_method = deepcopy(profile.get("phase1_method") or {})
    phase1_method["enable_policy_guidance"] = False
    phase1_method["critic_enabled"] = False
    phase1_method["stage3_enabled"] = False
    phase1_method["adaptive_replacement"] = False
    profile["phase1_method"] = phase1_method
    return profile


def _profile_qmix_single_llm_pbrs_v2_reward_gen_seed1() -> Dict[str, Any]:
    profile = _build_single_llm_pbrs_v2_reward_gen_profile(
        profile_name="qmix_single_llm_pbrs_v2_reward_gen_seed1",
        algorithm="qmix",
        env_key="lbforaging:Foraging-8x8-2p-1f-v3",
        formal_t_max=2050000,
    )
    return profile


def _profile_qmix_single_llm_pbrs_v2_reward_gen_8x8_2p_1f_seed1() -> Dict[str, Any]:
    return _build_single_llm_pbrs_v2_reward_gen_profile(
        profile_name="qmix_single_llm_pbrs_v2_reward_gen_8x8_2p_1f_seed1",
        algorithm="qmix",
        env_key="lbforaging:Foraging-8x8-2p-1f-v3",
        formal_t_max=2050000,
    )


def _profile_qmix_single_llm_pbrs_v2_reward_gen_8x8_2p_2f_seed1() -> Dict[str, Any]:
    return _build_single_llm_pbrs_v2_reward_gen_profile(
        profile_name="qmix_single_llm_pbrs_v2_reward_gen_8x8_2p_2f_seed1",
        algorithm="qmix",
        env_key="lbforaging:Foraging-8x8-2p-2f-coop-v3",
        formal_t_max=2050000,
    )


def _profile_qmix_single_llm_pbrs_v2_reward_gen_2s_8x8_2p_2f_seed1() -> Dict[str, Any]:
    return _build_single_llm_pbrs_v2_reward_gen_profile(
        profile_name="qmix_single_llm_pbrs_v2_reward_gen_2s_8x8_2p_2f_seed1",
        algorithm="qmix",
        env_key="lbforaging:Foraging-2s-8x8-2p-2f-coop-v3",
        formal_t_max=2050000,
    )


def _profile_qmix_single_llm_pbrs_v2_reward_gen_10x10_2p_1f_seed1() -> Dict[str, Any]:
    return _build_single_llm_pbrs_v2_reward_gen_profile(
        profile_name="qmix_single_llm_pbrs_v2_reward_gen_10x10_2p_1f_seed1",
        algorithm="qmix",
        env_key="lbforaging:Foraging-10x10-2p-1f-v3",
        formal_t_max=2050000,
    )


def _profile_qmix_single_llm_pbrs_v2_reward_gen_15x15_3p_4f_seed1() -> Dict[str, Any]:
    return _build_single_llm_pbrs_v2_reward_gen_profile(
        profile_name="qmix_single_llm_pbrs_v2_reward_gen_15x15_3p_4f_seed1",
        algorithm="qmix",
        env_key="lbforaging:Foraging-15x15-3p-4f-v3",
        formal_t_max=2050000,
    )


def _profile_mappo_single_llm_pbrs_v2_reward_gen_8x8_2p_1f_seed1() -> Dict[str, Any]:
    return _build_single_llm_pbrs_v2_reward_gen_profile(
        profile_name="mappo_single_llm_pbrs_v2_reward_gen_8x8_2p_1f_seed1",
        algorithm="mappo",
        env_key="lbforaging:Foraging-8x8-2p-1f-v3",
        formal_t_max=10000000,
    )


def _profile_mappo_single_llm_pbrs_v2_reward_gen_8x8_2p_2f_seed1() -> Dict[str, Any]:
    return _build_single_llm_pbrs_v2_reward_gen_profile(
        profile_name="mappo_single_llm_pbrs_v2_reward_gen_8x8_2p_2f_seed1",
        algorithm="mappo",
        env_key="lbforaging:Foraging-8x8-2p-2f-coop-v3",
        formal_t_max=10000000,
    )


def _profile_mappo_single_llm_pbrs_v2_reward_gen_2s_8x8_2p_2f_seed1() -> Dict[str, Any]:
    return _build_single_llm_pbrs_v2_reward_gen_profile(
        profile_name="mappo_single_llm_pbrs_v2_reward_gen_2s_8x8_2p_2f_seed1",
        algorithm="mappo",
        env_key="lbforaging:Foraging-2s-8x8-2p-2f-coop-v3",
        formal_t_max=10000000,
    )


def _profile_mappo_single_llm_pbrs_v2_reward_gen_10x10_2p_1f_seed1() -> Dict[str, Any]:
    return _build_single_llm_pbrs_v2_reward_gen_profile(
        profile_name="mappo_single_llm_pbrs_v2_reward_gen_10x10_2p_1f_seed1",
        algorithm="mappo",
        env_key="lbforaging:Foraging-10x10-2p-1f-v3",
        formal_t_max=10000000,
    )


def _profile_mappo_single_llm_pbrs_v2_reward_gen_15x15_3p_4f_seed1() -> Dict[str, Any]:
    return _build_single_llm_pbrs_v2_reward_gen_profile(
        profile_name="mappo_single_llm_pbrs_v2_reward_gen_15x15_3p_4f_seed1",
        algorithm="mappo",
        env_key="lbforaging:Foraging-15x15-3p-4f-v3",
        formal_t_max=10000000,
    )


def _profile_qmix_reward_only_small_pilot_seed1() -> Dict[str, Any]:
    profile = _profile_qmix_policy_guided_small_pilot_seed1()
    profile["profile_name"] = "qmix_reward_only_small_pilot_seed1"
    profile["enable_policy_guidance"] = False
    profile["reuse_policy_guidance_cache"] = False
    profile["policy_guidance_fallback_to_reward_only"] = True
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["enable_policy_guidance"] = False
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = False
    adaptive_replacement["reuse_policy_guidance_cache"] = False
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _profile_qmix_lbf_pbrs_v2_single_clean_workflow_pilot_seed1() -> Dict[str, Any]:
    profile = _profile_qmix_policy_guided_small_pilot_seed1()
    profile["profile_name"] = "qmix_lbf_pbrs_v2_single_clean_workflow_pilot_seed1"
    profile = _mark_lbf_policy_guided_sidecar(profile)
    profile["seed"] = 1
    profile["train_config"] = "qmix"
    profile["env_key"] = "lbforaging:Foraging-8x8-2p-1f-v3"
    profile["budget_profile"] = "lbf_pbrs_v2_single_clean_workflow_pilot"
    profile["t_max"] = 150000
    profile["stage1_sparse_t_max"] = 50000
    profile["dense_reference_t_max"] = 150000
    profile["dense_reference_target_t_max"] = 150000
    profile["adaptive_mainline_target_t_max"] = 150000
    profile["fixed_intervention_checkpoints"] = [80000, 120000]
    profile["branch_budget_steps"] = 30000
    profile["max_parallel_candidates"] = 1
    profile["use_real_llm"] = True
    profile["use_real_llm_for_field_analysis"] = False
    profile["mock_llm_mode"] = False
    profile["llm_model_routing"] = {"enabled": False}
    profile["model"] = "gpt-5.2"
    profile["api_key_env"] = "IUSEAPI_API_KEY"
    profile["base_url"] = "https://www.iuseapi.com/v1"
    profile["temperature"] = 0.0
    profile["llm_timeout"] = 60.0
    profile["llm_max_retries"] = 1
    profile["llm_retry_backoff"] = 0.0
    profile["reuse_policy_guidance_cache"] = False
    profile["policy_guidance_cache_lookup_mode"] = "exact_only"
    profile["policy_guidance_stage1_milestones"] = [10000, 25000, 40000, 50000]
    profile["policy_guidance_eval_episodes_stage1_per_milestone"] = 2
    profile["policy_guidance_stage1_behavior_summary_reuse_allowed"] = False
    profile["policy_guidance_stage1_behavior_summary_must_be_fresh"] = True
    profile["allow_reuse_stage1_behavior_summary"] = False
    profile["initial_dense_search_rounds"] = 2
    profile["initial_dense_candidates_per_round"] = 3
    profile["initial_dense_candidate_budget_steps"] = 50000
    profile["initial_dense_candidate_target_checkpoint_step"] = 50000
    profile["stage1b_selected_endpoint_target_step"] = 50000
    profile["initial_dense_candidate_parallel"] = False
    profile["dense_reference_sidecar_enabled"] = True
    profile["dense_reference_from_selected_stage1b_endpoint"] = True
    profile["dense_reference_fixed_pbrs"] = True
    profile["dense_reference_disable_adaptive_update"] = True
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 3
    profile["stage3_update_candidates_per_round"] = 3
    profile["stage3_include_no_change_control"] = True
    profile["stage3_no_change_counts_toward_round_budget"] = False
    profile["stage3_reuse_no_change_across_rounds"] = True
    profile["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    profile["stage3_require_nontrivial_config_delta"] = True
    profile["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    profile["stage3_require_candidate_type_diversity"] = True
    profile["policy_guided_stage3_require_behavior_evidence"] = True
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["dense_reference_target_t_max"] = 150000
    phase1_method["adaptive_mainline_target_t_max"] = 150000
    phase1_method["reuse_policy_guidance_cache"] = False
    phase1_method["policy_guidance_cache_lookup_mode"] = "exact_only"
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["fallback_to_single_llm_initial_config"] = False
    phase1_method["policy_guidance_stage1b_fallback_to_reward_only"] = False
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["keep_dense_reference_continuation"] = True
    phase1_method["use_fixed_intervention_checkpoints"] = True
    phase1_method["fixed_intervention_checkpoints"] = [80000, 120000]
    phase1_method["checkpoint_count"] = 2
    initial_dense_search = deepcopy(phase1_method.get("initial_dense_search") or {})
    initial_dense_search["enabled"] = True
    initial_dense_search["rounds"] = 2
    initial_dense_search["candidates_per_round"] = 3
    initial_dense_search["candidate_budget_steps"] = 50000
    initial_dense_search["candidate_target_checkpoint_step"] = 50000
    initial_dense_search["selected_endpoint_target_step"] = 50000
    initial_dense_search["endpoint_tolerance_steps"] = 5000
    initial_dense_search["endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    initial_dense_search["parallel"] = False
    initial_dense_search["save_endpoint_checkpoint"] = True
    initial_dense_search["save_intermediate_checkpoints"] = False
    initial_dense_search["keep_only_endpoint_checkpoint"] = True
    phase1_method["initial_dense_search"] = initial_dense_search
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = True
    adaptive_replacement["mock_llm_mode"] = False
    adaptive_replacement["reuse_policy_guidance_cache"] = False
    adaptive_replacement["policy_guidance_cache_lookup_mode"] = "exact_only"
    adaptive_replacement["policy_guidance_stage1_behavior_summary_reuse_allowed"] = False
    adaptive_replacement["policy_guidance_stage1_behavior_summary_must_be_fresh"] = True
    adaptive_replacement["allow_reuse_stage1_behavior_summary"] = False
    adaptive_replacement["fallback_to_deterministic_on_llm_error"] = False
    adaptive_replacement["use_winner_branch_promotion"] = True
    adaptive_replacement["fixed_intervention_checkpoints"] = [80000, 120000]
    adaptive_replacement["branch_budget_steps"] = 30000
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 3
    adaptive_replacement["stage3_update_candidates_per_round"] = 3
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["stage3_no_change_counts_toward_round_budget"] = False
    adaptive_replacement["stage3_reuse_no_change_across_rounds"] = True
    adaptive_replacement["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    adaptive_replacement["stage3_require_nontrivial_config_delta"] = True
    adaptive_replacement["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    adaptive_replacement["stage3_require_candidate_type_diversity"] = True
    adaptive_replacement["policy_guided_stage3_require_behavior_evidence"] = True
    phase1_method["adaptive_replacement"] = adaptive_replacement
    phase1_method["fixed_checkpoint_plan"] = [
        {
            "name": "C1",
            "step": 80000,
            "stage_label": "post_initialization_transition",
            "intervention_question": "Does the selected Stage 1b LBF PBRS-v2 initialization remain suitable at the first pilot checkpoint?",
            "recommended_candidate_focus": ["reference_like", "recovery", "beta_down"],
            "branch_budget_feasible": True,
        },
        {
            "name": "C2",
            "step": 120000,
            "stage_label": "late_stability",
            "intervention_question": "Does the current LBF PBRS-v2 reward configuration remain stable and aligned at the second pilot checkpoint?",
            "recommended_candidate_focus": ["reference_like", "recovery", "beta_down"],
            "branch_budget_feasible": True,
        },
    ]
    profile["phase1_method"] = phase1_method
    return profile


def _profile_qmix_policy_guided_formal_pilot_seed1() -> Dict[str, Any]:
    profile = _profile_10x10_qmix_adaptive()
    profile["profile_name"] = "qmix_policy_guided_formal_pilot_seed1"
    profile = _apply_formal_qmix_single_seed_overrides(profile)
    profile = _apply_policy_guidance_profile_overrides(profile)
    profile["seed"] = 1
    profile["train_config"] = "qmix"
    profile["env_key"] = "lbforaging:Foraging-10x10-2p-1f-v3"
    profile["budget_profile"] = "policy_guided_formal_pilot"
    profile["use_real_llm"] = True
    profile["use_real_llm_for_field_analysis"] = True
    profile["mock_llm_mode"] = False
    profile["reuse_policy_guidance_cache"] = True
    profile["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    profile["stage1_sparse_save_checkpoints"] = False
    profile["stage1_sparse_save_checkpoints"] = False
    profile["policy_guidance_stage1_milestones"] = [200000, 500000, 800000, 2050000]
    profile["policy_guidance_eval_episodes_stage1_per_milestone"] = 10
    profile["policy_guidance_stage1_behavior_summary_must_be_fresh"] = True
    profile["policy_guidance_stage1_behavior_summary_reuse_allowed"] = False
    profile["allow_reuse_stage1_behavior_summary"] = False
    profile["enable_dense_reference"] = True
    profile["dense_reference_from_selected_stage1b_endpoint"] = True
    profile["dense_reference_fixed_pbrs"] = True
    profile["dense_reference_disable_adaptive_update"] = True
    profile["dense_reference_save_checkpoints"] = False
    profile["dense_reference_t_max"] = 2050000
    profile["adaptive_mainline_from_selected_stage1b_endpoint"] = True
    profile["adaptive_mainline_target_t_max"] = 2050000
    profile["dense_reference_target_t_max"] = 2050000
    profile["use_fixed_intervention_checkpoints"] = True
    profile["fixed_intervention_checkpoints"] = [1000000, 1500000]
    profile["branch_budget_steps"] = 300000
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 3
    profile["stage3_include_no_change_control"] = True
    profile["stage3_branch_save_endpoint_checkpoint"] = True
    profile["stage3_branch_save_intermediate_checkpoints"] = False
    profile["checkpoint_retention_mode"] = "new_method_minimal_v1"
    profile["adaptive_mainline_save_c1_c2_source_checkpoints"] = True
    profile["adaptive_mainline_save_intermediate_checkpoints"] = False
    profile["adaptive_mainline_save_final_checkpoint"] = False
    profile["adaptive_mainline_save_c1_c2_source_only"] = True
    profile["save_raw_trajectories"] = False
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["dense_reference_target_t_max"] = 2050000
    phase1_method["adaptive_mainline_target_t_max"] = 2050000
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["keep_dense_reference_continuation"] = True
    phase1_method["reuse_policy_guidance_cache"] = True
    phase1_method["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    phase1_method["checkpoint_retention_mode"] = "new_method_minimal_v1"
    phase1_method["use_fixed_intervention_checkpoints"] = True
    phase1_method["fixed_intervention_checkpoints"] = [1000000, 1500000]
    initial_dense_search = deepcopy(phase1_method.get("initial_dense_search") or {})
    initial_dense_search["enabled"] = True
    initial_dense_search["rounds"] = 2
    initial_dense_search["candidates_per_round"] = 3
    initial_dense_search["candidate_budget_steps"] = 850000
    initial_dense_search["candidate_target_checkpoint_step"] = 800000
    initial_dense_search["selected_endpoint_target_step"] = 800000
    initial_dense_search["endpoint_tolerance_steps"] = 50000
    initial_dense_search["endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    initial_dense_search["save_endpoint_checkpoint"] = True
    initial_dense_search["save_intermediate_checkpoints"] = False
    initial_dense_search["keep_only_endpoint_checkpoint"] = True
    phase1_method["initial_dense_search"] = initial_dense_search
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = True
    adaptive_replacement["reuse_policy_guidance_cache"] = True
    adaptive_replacement["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    adaptive_replacement["policy_guidance_stage1_milestones"] = [200000, 500000, 800000, 2050000]
    adaptive_replacement["policy_guidance_eval_episodes_stage1_per_milestone"] = 10
    adaptive_replacement["policy_guidance_stage1_behavior_summary_must_be_fresh"] = True
    adaptive_replacement["policy_guidance_stage1_behavior_summary_reuse_allowed"] = False
    adaptive_replacement["allow_reuse_stage1_behavior_summary"] = False
    adaptive_replacement["fixed_intervention_checkpoints"] = [1000000, 1500000]
    adaptive_replacement["branch_budget_steps"] = 300000
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 3
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["use_winner_branch_promotion"] = True
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _profile_qmix_reward_only_formal_pilot_seed1() -> Dict[str, Any]:
    profile = _profile_qmix_policy_guided_formal_pilot_seed1()
    profile["profile_name"] = "qmix_reward_only_formal_pilot_seed1"
    profile["enable_policy_guidance"] = False
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["enable_policy_guidance"] = False
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = False
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _build_qmix_policy_guided_sidecar_profile(
    *,
    profile_name: str,
    env_key: str,
    stage3_use_cuda: bool,
    max_parallel_candidates: int,
) -> Dict[str, Any]:
    profile = _base_adaptive_profile()
    profile["phase1_method"] = deepcopy(_base_adaptive_profile()["phase1_method"])
    profile["profile_name"] = profile_name
    profile["seed"] = 1
    profile["train_config"] = "qmix"
    profile["env_key"] = env_key
    profile["time_limit"] = 50
    profile["t_max"] = 2050000
    profile["stage1_sparse_t_max"] = 2050000
    profile["test_interval"] = 50000
    profile["runner_log_interval"] = 50000
    profile["learner_log_interval"] = 50000
    profile["use_cuda"] = True
    profile["stage1_use_cuda"] = True
    profile["stage3_use_cuda"] = bool(stage3_use_cuda)
    profile["stage5_use_cuda"] = True
    profile["max_parallel_candidates"] = int(max_parallel_candidates)
    profile["budget_profile"] = "policy_guided_sidecar"
    profile = _apply_formal_qmix_single_seed_overrides(profile)
    profile = _apply_policy_guidance_profile_overrides(profile)
    profile = _mark_lbf_policy_guided_sidecar(profile)
    profile["enable_dense_reference"] = True
    profile["dense_reference_sidecar_enabled"] = True
    profile["dense_reference_from_selected_stage1b_endpoint"] = True
    profile["dense_reference_fixed_pbrs"] = True
    profile["dense_reference_disable_adaptive_update"] = True
    profile["dense_reference_save_checkpoints"] = False
    profile["dense_reference_t_max"] = 2050000
    profile["adaptive_mainline_from_selected_stage1b_endpoint"] = True
    profile["adaptive_mainline_target_t_max"] = 2050000
    profile["dense_reference_target_t_max"] = 2050000
    profile["fixed_intervention_checkpoints"] = [1000000, 1500000]
    profile["branch_budget_steps"] = 300000
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 3
    profile["stage3_update_candidates_per_round"] = 3
    profile["stage3_include_no_change_control"] = True
    profile["stage3_no_change_counts_toward_round_budget"] = False
    profile["stage3_reuse_no_change_across_rounds"] = True
    profile["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    profile["stage3_require_nontrivial_config_delta"] = True
    profile["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    profile["stage3_require_candidate_type_diversity"] = True
    profile["policy_guided_stage3_require_behavior_evidence"] = True
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["dense_reference_target_t_max"] = 2050000
    phase1_method["adaptive_mainline_target_t_max"] = 2050000
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["keep_dense_reference_continuation"] = True
    phase1_method["reuse_policy_guidance_cache"] = True
    phase1_method["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    phase1_method["use_fixed_intervention_checkpoints"] = True
    phase1_method["fixed_intervention_checkpoints"] = [1000000, 1500000]
    phase1_method["checkpoint_count"] = 2
    initial_dense_search = deepcopy(phase1_method.get("initial_dense_search") or {})
    initial_dense_search["enabled"] = True
    initial_dense_search["rounds"] = 2
    initial_dense_search["candidates_per_round"] = 3
    initial_dense_search["candidate_budget_steps"] = 850000
    initial_dense_search["candidate_target_checkpoint_step"] = 800000
    initial_dense_search["selected_endpoint_target_step"] = 800000
    initial_dense_search["endpoint_tolerance_steps"] = 50000
    initial_dense_search["endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    initial_dense_search["parallel"] = True
    initial_dense_search["save_endpoint_checkpoint"] = True
    initial_dense_search["save_intermediate_checkpoints"] = False
    initial_dense_search["keep_only_endpoint_checkpoint"] = True
    phase1_method["initial_dense_search"] = initial_dense_search
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = True
    adaptive_replacement["reuse_policy_guidance_cache"] = True
    adaptive_replacement["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    adaptive_replacement["fixed_intervention_checkpoints"] = [1000000, 1500000]
    adaptive_replacement["branch_budget_steps"] = 300000
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 3
    adaptive_replacement["stage3_update_candidates_per_round"] = 3
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["stage3_no_change_counts_toward_round_budget"] = False
    adaptive_replacement["stage3_reuse_no_change_across_rounds"] = True
    adaptive_replacement["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    adaptive_replacement["stage3_require_nontrivial_config_delta"] = True
    adaptive_replacement["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    adaptive_replacement["stage3_require_candidate_type_diversity"] = True
    adaptive_replacement["policy_guided_stage3_require_behavior_evidence"] = True
    adaptive_replacement["use_winner_branch_promotion"] = True
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _profile_qmix_pbrs_v2_policy_guided_8x8_2p_1f_sidecar() -> Dict[str, Any]:
    return _apply_clean_lbf_profile_flags(
        _build_qmix_policy_guided_sidecar_profile(
            profile_name="qmix_pbrs_v2_policy_guided_8x8_2p_1f_sidecar",
            env_key="lbforaging:Foraging-8x8-2p-1f-v3",
            stage3_use_cuda=True,
            max_parallel_candidates=6,
        ),
        use_real_llm=True,
        use_real_llm_for_field_analysis=True,
    )


def _profile_qmix_lbf_pbrs_v2_8x8_2p_1f_full_clean_seed1() -> Dict[str, Any]:
    profile = _profile_qmix_pbrs_v2_policy_guided_8x8_2p_1f_sidecar()
    profile["profile_name"] = "qmix_lbf_pbrs_v2_8x8_2p_1f_full_clean_seed1"
    profile["budget_profile"] = "lbf_pbrs_v2_full_clean"
    profile["full_clean_runnable"] = True
    profile["formal_execute_guard_required"] = True
    profile["summary_only_sidecar"] = False
    profile["seed"] = 1
    profile["fallback_to_single_llm_initial_config"] = False
    profile["eval_use_pbrs"] = False
    profile["apply_dense_reward_in_eval"] = False
    profile["rolling_adaptive_mainline_is_final_result"] = True

    phase1_method = deepcopy(profile.get("phase1_method") or {})
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["enable_stage5_final_full_run"] = False
    phase1_method["final_spec_is_summary_only"] = True
    phase1_method["rolling_adaptive_mainline_is_final_result"] = True
    phase1_method["fallback_to_single_llm_initial_config"] = False
    phase1_method["policy_guidance_stage1b_fallback_to_reward_only"] = False
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = True
    adaptive_replacement["stage3_branch_launch_enabled"] = True
    adaptive_replacement["fallback_to_deterministic_on_llm_error"] = False
    adaptive_replacement["add_deterministic_recovery_fallback"] = False
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _profile_qmix_lbf_pbrs_v2_10x10_2p_1f_full_clean_seed1() -> Dict[str, Any]:
    return _configure_stage1reuse_adaptive_full_clean_profile(
        _profile_qmix_pbrs_v2_policy_guided_10x10_2p_1f_sidecar(),
        profile_name="qmix_lbf_pbrs_v2_10x10_2p_1f_full_clean_seed1",
        bundle_id="qmix_10x10_2p_1f_seed1",
        budget_profile="lbf_pbrs_v2_full_clean",
    )


def _profile_qmix_lbf_pbrs_v2_8x8_2p_2f_full_clean_seed1() -> Dict[str, Any]:
    return _configure_stage1reuse_adaptive_full_clean_profile(
        _profile_qmix_pbrs_v2_policy_guided_8x8_2p_2f_sidecar(),
        profile_name="qmix_lbf_pbrs_v2_8x8_2p_2f_full_clean_seed1",
        bundle_id="qmix_8x8_2p_2f_seed1",
        budget_profile="lbf_pbrs_v2_full_clean",
    )


def _profile_qmix_lbf_pbrs_v2_2s_8x8_2p_2f_full_clean_seed1() -> Dict[str, Any]:
    return _configure_stage1reuse_adaptive_full_clean_profile(
        _profile_qmix_pbrs_v2_policy_guided_2s_8x8_2p_2f_sidecar(),
        profile_name="qmix_lbf_pbrs_v2_2s_8x8_2p_2f_full_clean_seed1",
        bundle_id="qmix_2s_8x8_2p_2f_seed1",
        budget_profile="lbf_pbrs_v2_full_clean",
    )


def _profile_qmix_lbf_pbrs_v2_15x15_3p_4f_full_clean_seed1() -> Dict[str, Any]:
    return _configure_stage1reuse_adaptive_full_clean_profile(
        _profile_qmix_pbrs_v2_policy_guided_15x15_3p_4f_sidecar(),
        profile_name="qmix_lbf_pbrs_v2_15x15_3p_4f_full_clean_seed1",
        bundle_id="qmix_15x15_3p_4f_seed1",
        budget_profile="lbf_pbrs_v2_full_clean",
    )


def _profile_qmix_lbf_pbrs_v2_10x10_2p_1f_stage1reuse_stage1b_50k_deepseek_smoke() -> Dict[str, Any]:
    profile = _profile_qmix_lbf_pbrs_v2_10x10_2p_1f_full_clean_seed1()
    profile["profile_name"] = "qmix_lbf_pbrs_v2_10x10_2p_1f_stage1reuse_stage1b_50k_deepseek_smoke"
    profile["budget_profile"] = "policy_guided_small_pilot"
    profile["formal_execute_guard_required"] = False
    profile["summary_only_sidecar"] = False
    profile["use_real_llm"] = True
    profile["use_real_llm_for_field_analysis"] = True
    profile["mock_llm_mode"] = False
    profile["llm_model_routing"] = {
        "enabled": True,
        "validation_provider": "deepseek",
        "validation_model": "deepseek-chat",
        "formal_provider": "openai",
        "formal_model": "gpt-5.2",
        "formal_only_when_allow_formal_execute": True,
    }
    profile["t_max"] = 2050000
    profile["stage1_sparse_t_max"] = 2050000
    profile["test_interval"] = 5000
    profile["runner_log_interval"] = 5000
    profile["learner_log_interval"] = 5000
    profile["save_model_interval"] = 5000
    profile["reuse_policy_guidance_cache"] = False
    profile["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    profile["initial_dense_search_rounds"] = 1
    profile["initial_dense_candidates_per_round"] = 3
    profile["initial_dense_candidate_budget_steps"] = 50000
    profile["initial_dense_candidate_target_checkpoint_step"] = 50000
    profile["stage1b_selected_endpoint_target_step"] = 50000
    profile["stage1b_endpoint_tolerance_steps"] = 5000
    profile["stage1b_candidate_save_endpoint_checkpoint"] = False
    profile["stage1b_candidate_save_intermediate_checkpoints"] = False
    profile["stage1b_keep_only_endpoint_checkpoint"] = True
    profile["dense_reference_t_max"] = 60000
    profile["dense_reference_target_t_max"] = 60000
    profile["adaptive_mainline_target_t_max"] = 60000
    profile["fixed_intervention_checkpoints"] = [50000]
    profile["branch_budget_steps"] = 50000
    profile["stage3_use_cuda"] = False
    profile["stage5_use_cuda"] = False
    phase1_method = deepcopy(profile.get("phase1_method") or {})
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["enable_stage5_final_full_run"] = False
    phase1_method["final_spec_is_summary_only"] = True
    phase1_method["rolling_adaptive_mainline_is_final_result"] = False
    phase1_method["checkpoint_selection_mode"] = "post_dense_dual_source"
    phase1_method["dense_reference_target_t_max"] = 60000
    phase1_method["adaptive_mainline_target_t_max"] = 60000
    initial_dense_search = deepcopy(phase1_method.get("initial_dense_search") or {})
    initial_dense_search["rounds"] = 1
    initial_dense_search["candidates_per_round"] = 3
    initial_dense_search["candidate_budget_steps"] = 50000
    initial_dense_search["candidate_target_checkpoint_step"] = 50000
    initial_dense_search["selected_endpoint_target_step"] = 50000
    initial_dense_search["endpoint_tolerance_steps"] = 5000
    phase1_method["initial_dense_search"] = initial_dense_search
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["branch_budget_steps"] = 50000
    adaptive_replacement["stage3_branch_launch_enabled"] = False
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _profile_qmix_pbrs_v2_policy_guided_8x8_2p_2f_sidecar() -> Dict[str, Any]:
    return _apply_clean_lbf_profile_flags(
        _build_qmix_policy_guided_sidecar_profile(
            profile_name="qmix_pbrs_v2_policy_guided_8x8_2p_2f_sidecar",
            env_key="lbforaging:Foraging-8x8-2p-2f-coop-v3",
            stage3_use_cuda=True,
            max_parallel_candidates=6,
        ),
        use_real_llm=True,
        use_real_llm_for_field_analysis=True,
    )


def _profile_qmix_pbrs_v2_policy_guided_2s_8x8_2p_2f_sidecar() -> Dict[str, Any]:
    return _apply_clean_lbf_profile_flags(
        _build_qmix_policy_guided_sidecar_profile(
            profile_name="qmix_pbrs_v2_policy_guided_2s_8x8_2p_2f_sidecar",
            env_key="lbforaging:Foraging-2s-8x8-2p-2f-coop-v3",
            stage3_use_cuda=True,
            max_parallel_candidates=6,
        ),
        use_real_llm=True,
        use_real_llm_for_field_analysis=True,
    )


def _profile_qmix_pbrs_v2_policy_guided_10x10_2p_1f_sidecar() -> Dict[str, Any]:
    return _apply_clean_lbf_profile_flags(
        _build_qmix_policy_guided_sidecar_profile(
            profile_name="qmix_pbrs_v2_policy_guided_10x10_2p_1f_sidecar",
            env_key="lbforaging:Foraging-10x10-2p-1f-v3",
            stage3_use_cuda=False,
            max_parallel_candidates=6,
        ),
        use_real_llm=True,
        use_real_llm_for_field_analysis=True,
    )


def _profile_qmix_pbrs_v2_policy_guided_15x15_3p_4f_sidecar() -> Dict[str, Any]:
    return _apply_clean_lbf_profile_flags(
        _build_qmix_policy_guided_sidecar_profile(
            profile_name="qmix_pbrs_v2_policy_guided_15x15_3p_4f_sidecar",
            env_key="lbforaging:Foraging-15x15-3p-4f-v3",
            stage3_use_cuda=False,
            max_parallel_candidates=3,
        ),
        use_real_llm=True,
        use_real_llm_for_field_analysis=True,
    )


def _build_qmix_model52_reuse_stage1_rerun_profile(
    *,
    profile_name: str,
    env_key: str,
    source_workflow_id: str,
    stage3_use_cuda: bool,
    max_parallel_candidates: int,
) -> Dict[str, Any]:
    profile = _build_qmix_policy_guided_sidecar_profile(
        profile_name=profile_name,
        env_key=env_key,
        stage3_use_cuda=stage3_use_cuda,
        max_parallel_candidates=max_parallel_candidates,
    )
    profile["model"] = "gpt-5.2"
    profile["use_real_llm"] = True
    profile["use_real_llm_for_field_analysis"] = True
    profile["mock_llm_mode"] = False
    profile["reuse_policy_guidance_cache"] = True
    profile["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    profile["stage1_reuse"] = _build_stage1_completed_artifact_reuse_config(
        source_workflow_id=source_workflow_id,
        source_seed=1,
        source_results_root=ARCHIVED_MAPPO_QMIX_RESULTS_ROOT,
        reuse_behavior_summary=True,
        reuse_sparse_metrics_for_context=True,
    )
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["reuse_policy_guidance_cache"] = True
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["reuse_policy_guidance_cache"] = True
    adaptive_replacement["llm_stage3_use_cache"] = True
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _profile_qmix_pbrs_v2_policy_guided_8x8_2p_2f_sidecar_run2_model52_reuse_stage1() -> Dict[str, Any]:
    return _build_qmix_model52_reuse_stage1_rerun_profile(
        profile_name="qmix_pbrs_v2_policy_guided_8x8_2p_2f_sidecar_run2_model52_reuse_stage1",
        env_key="lbforaging:Foraging-8x8-2p-2f-coop-v3",
        source_workflow_id="qmix_pbrs_v2_policy_guided_8x8_2p_2f_sidecar_run",
        stage3_use_cuda=True,
        max_parallel_candidates=6,
    )


def _profile_qmix_pbrs_v2_policy_guided_2s_8x8_2p_2f_sidecar_run2_model52_reuse_stage1() -> Dict[str, Any]:
    return _build_qmix_model52_reuse_stage1_rerun_profile(
        profile_name="qmix_pbrs_v2_policy_guided_2s_8x8_2p_2f_sidecar_run2_model52_reuse_stage1",
        env_key="lbforaging:Foraging-2s-8x8-2p-2f-coop-v3",
        source_workflow_id="qmix_pbrs_v2_policy_guided_2s_8x8_2p_2f_sidecar_run",
        stage3_use_cuda=True,
        max_parallel_candidates=6,
    )


def _profile_qmix_pbrs_v2_policy_guided_10x10_2p_1f_sidecar_run2_model52_reuse_stage1() -> Dict[str, Any]:
    return _build_qmix_model52_reuse_stage1_rerun_profile(
        profile_name="qmix_pbrs_v2_policy_guided_10x10_2p_1f_sidecar_run2_model52_reuse_stage1",
        env_key="lbforaging:Foraging-10x10-2p-1f-v3",
        source_workflow_id="qmix_pbrs_v2_policy_guided_10x10_2p_1f_sidecar_run",
        stage3_use_cuda=False,
        max_parallel_candidates=6,
    )


def _profile_qmix_pbrs_v2_policy_guided_15x15_3p_4f_sidecar_run2_model52_reuse_stage1() -> Dict[str, Any]:
    return _build_qmix_model52_reuse_stage1_rerun_profile(
        profile_name="qmix_pbrs_v2_policy_guided_15x15_3p_4f_sidecar_run2_model52_reuse_stage1",
        env_key="lbforaging:Foraging-15x15-3p-4f-v3",
        source_workflow_id="qmix_pbrs_v2_policy_guided_15x15_3p_4f_sidecar_run",
        stage3_use_cuda=False,
        max_parallel_candidates=3,
    )


def _build_shared_sparse_reuse_config(
    *,
    source_workflow_id: str,
    source_seed: int,
    reuse_behavior_summary: bool,
    reuse_sparse_metrics_for_context: bool,
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "source_workflow_id": str(source_workflow_id),
        "source_seed": int(source_seed),
        "reuse_behavior_summary": bool(reuse_behavior_summary),
        "reuse_sparse_metrics_for_context": bool(reuse_sparse_metrics_for_context),
        "skip_sparse_training": True,
        "count_as_evaluation_baseline": False,
        "reuse_mode": "shared_sparse_diagnosis",
    }


def _build_stage1_completed_artifact_reuse_config(
    *,
    source_workflow_id: str,
    source_seed: int,
    source_results_root: str,
    reuse_behavior_summary: bool,
    reuse_sparse_metrics_for_context: bool,
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "source_workflow_id": str(source_workflow_id),
        "source_seed": int(source_seed),
        "source_stage": "stage1",
        "source_results_root": str(source_results_root),
        "allow_partial_source_workflow": True,
        "require_source_stage1_completed": True,
        "reuse_behavior_summary": bool(reuse_behavior_summary),
        "reuse_sparse_metrics_for_context": bool(reuse_sparse_metrics_for_context),
        "skip_sparse_training": True,
        "count_as_evaluation_baseline": False,
        "reuse_mode": "stage1_completed_artifact_reuse",
    }


def _build_portable_stage1_bundle_reuse_config(
    *,
    bundle_id: str,
    reuse_behavior_summary: bool,
    reuse_sparse_metrics_for_context: bool,
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "bundle_id": str(bundle_id),
        "reuse_behavior_summary": bool(reuse_behavior_summary),
        "reuse_sparse_metrics_for_context": bool(reuse_sparse_metrics_for_context),
        "skip_sparse_training": True,
        "count_as_evaluation_baseline": False,
        "reuse_mode": "portable_stage1_bundle",
        "bundle_root": str(PORTABLE_STAGE1_BUNDLE_ROOT),
    }


def _configure_stage1reuse_adaptive_full_clean_profile(
    base_profile: Dict[str, Any],
    *,
    profile_name: str,
    bundle_id: str,
    budget_profile: str,
) -> Dict[str, Any]:
    profile = deepcopy(base_profile)
    profile["profile_name"] = profile_name
    profile["budget_profile"] = budget_profile
    profile["full_clean_runnable"] = True
    profile["formal_execute_guard_required"] = True
    profile["summary_only_sidecar"] = False
    profile["seed"] = 1
    profile["pbrs_version"] = "lbf_pbrs_v2"
    profile["fallback_to_single_llm_initial_config"] = False
    profile["eval_use_pbrs"] = False
    profile["apply_dense_reward_in_eval"] = False
    profile["rolling_adaptive_mainline_is_final_result"] = True
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 3
    profile["stage3_update_candidates_per_round"] = 3
    profile["stage3_include_no_change_control"] = True
    profile["stage3_no_change_counts_toward_round_budget"] = False
    profile["stage3_reuse_no_change_across_rounds"] = True
    profile["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    profile["stage3_require_nontrivial_config_delta"] = True
    profile["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    profile["stage3_require_candidate_type_diversity"] = True
    profile["policy_guided_stage3_require_behavior_evidence"] = True
    profile["stage3_gate_mode"] = "soft_risk_adjusted"
    profile["stage3_soft_promotion_margin"] = 0.03
    profile["stage3_reference_soft_penalty"] = 0.005
    profile["stage3_first_replacement_soft_penalty"] = 0.0
    profile["stage1_reuse"] = _build_portable_stage1_bundle_reuse_config(
        bundle_id=bundle_id,
        reuse_behavior_summary=True,
        reuse_sparse_metrics_for_context=True,
    )

    phase1_method = deepcopy(profile.get("phase1_method") or {})
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["enable_stage5_final_full_run"] = False
    phase1_method["final_spec_is_summary_only"] = True
    phase1_method["rolling_adaptive_mainline_is_final_result"] = True
    phase1_method["fallback_to_single_llm_initial_config"] = False
    phase1_method["policy_guidance_stage1b_fallback_to_reward_only"] = False
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["use_fixed_intervention_checkpoints"] = True
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = True
    adaptive_replacement["mock_llm_mode"] = False
    adaptive_replacement["stage3_branch_launch_enabled"] = True
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 3
    adaptive_replacement["stage3_update_candidates_per_round"] = 3
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["stage3_no_change_counts_toward_round_budget"] = False
    adaptive_replacement["stage3_reuse_no_change_across_rounds"] = True
    adaptive_replacement["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    adaptive_replacement["stage3_require_nontrivial_config_delta"] = True
    adaptive_replacement["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    adaptive_replacement["stage3_require_candidate_type_diversity"] = True
    adaptive_replacement["policy_guided_stage3_require_behavior_evidence"] = True
    adaptive_replacement["stage3_gate_mode"] = "soft_risk_adjusted"
    adaptive_replacement["stage3_soft_promotion_margin"] = 0.03
    adaptive_replacement["stage3_reference_soft_penalty"] = 0.005
    adaptive_replacement["stage3_first_replacement_soft_penalty"] = 0.0
    adaptive_replacement["fallback_to_deterministic_on_llm_error"] = False
    adaptive_replacement["add_deterministic_recovery_fallback"] = False
    adaptive_replacement["use_winner_branch_promotion"] = True
    adaptive_replacement["fixed_intervention_checkpoints"] = deepcopy(
        profile.get("fixed_intervention_checkpoints")
        or phase1_method.get("fixed_intervention_checkpoints")
        or adaptive_replacement.get("fixed_intervention_checkpoints")
        or []
    )
    adaptive_replacement["branch_budget_steps"] = int(
        profile.get("branch_budget_steps")
        or adaptive_replacement.get("branch_budget_steps")
        or 0
    )
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _profile_mappo_pbrs_v2_medium_seed1(*, policy_guided: bool) -> Dict[str, Any]:
    profile = _profile_8x8_qmix_adaptive()
    profile["profile_name"] = (
        "mappo_pbrs_v2_policy_guided_medium_seed1_sidecar"
        if policy_guided
        else "mappo_pbrs_v2_reward_only_medium_seed1_sidecar"
    )
    profile["seed"] = 1
    profile["train_config"] = "mappo"
    profile["env_key"] = "lbforaging:Foraging-8x8-2p-1f-v3"
    profile["budget_profile"] = "mappo_medium_sidecar"
    profile["t_max"] = 10000000
    profile["stage1_sparse_t_max"] = 10000000
    profile["dense_reference_t_max"] = 10000000
    profile["dense_reference_target_t_max"] = 10000000
    profile["adaptive_mainline_target_t_max"] = 10000000
    profile["use_real_llm"] = True
    profile["use_real_llm_for_field_analysis"] = True
    profile["mock_llm_mode"] = False
    profile["llm_model_routing"] = {
        "enabled": True,
        "validation_provider": "deepseek",
        "validation_model": "deepseek-chat",
        "formal_provider": "openai",
        "formal_model": "gpt-5.2",
        "formal_only_when_allow_formal_execute": True,
    }
    profile["reuse_policy_guidance_cache"] = True
    profile["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    profile["save_raw_trajectories"] = False
    profile["fixed_intervention_checkpoints"] = [1500000, 2500000]
    profile["branch_budget_steps"] = 500000
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 3
    profile["stage3_update_candidates_per_round"] = 3
    profile["stage3_include_no_change_control"] = True
    profile["stage3_no_change_counts_toward_round_budget"] = False
    profile["stage3_reuse_no_change_across_rounds"] = True
    profile["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    profile["stage3_require_nontrivial_config_delta"] = True
    profile["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    profile["stage3_require_candidate_type_diversity"] = True
    profile["policy_guided_stage3_require_behavior_evidence"] = True
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["dense_reference_target_t_max"] = 10000000
    phase1_method["adaptive_mainline_target_t_max"] = 10000000
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["keep_dense_reference_continuation"] = True
    phase1_method["use_fixed_intervention_checkpoints"] = True
    phase1_method["fixed_intervention_checkpoints"] = [1500000, 2500000]
    phase1_method["checkpoint_count"] = 2
    initial_dense_search = deepcopy(phase1_method.get("initial_dense_search") or {})
    initial_dense_search["enabled"] = True
    initial_dense_search["rounds"] = 2
    initial_dense_search["candidates_per_round"] = 3
    initial_dense_search["candidate_budget_steps"] = 850000
    initial_dense_search["candidate_target_checkpoint_step"] = 800000
    initial_dense_search["selected_endpoint_target_step"] = 800000
    initial_dense_search["endpoint_tolerance_steps"] = 50000
    initial_dense_search["endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    initial_dense_search["parallel"] = True
    initial_dense_search["save_endpoint_checkpoint"] = True
    initial_dense_search["save_intermediate_checkpoints"] = False
    initial_dense_search["keep_only_endpoint_checkpoint"] = True
    phase1_method["initial_dense_search"] = initial_dense_search
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = bool(policy_guided)
    adaptive_replacement["reuse_policy_guidance_cache"] = True
    adaptive_replacement["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    adaptive_replacement["fixed_intervention_checkpoints"] = [1500000, 2500000]
    adaptive_replacement["branch_budget_steps"] = 500000
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 3
    adaptive_replacement["stage3_update_candidates_per_round"] = 3
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["stage3_no_change_counts_toward_round_budget"] = False
    adaptive_replacement["stage3_reuse_no_change_across_rounds"] = True
    adaptive_replacement["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    adaptive_replacement["stage3_require_nontrivial_config_delta"] = True
    adaptive_replacement["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    adaptive_replacement["stage3_require_candidate_type_diversity"] = True
    adaptive_replacement["policy_guided_stage3_require_behavior_evidence"] = True
    adaptive_replacement["use_winner_branch_promotion"] = True
    phase1_method["adaptive_replacement"] = adaptive_replacement
    phase1_method["fixed_checkpoint_plan"] = [
        {
            "name": "C1",
            "step": 1500000,
            "stage_label": "post_initialization_transition",
            "intervention_question": "Does the selected Stage 1b MAPPO dense initialization remain suitable at the first medium-budget checkpoint?",
            "recommended_candidate_focus": [
                "beta_down",
                "beta_up",
                "reference_like",
                "recovery",
            ],
            "branch_budget_feasible": True,
        },
        {
            "name": "C2",
            "step": 2500000,
            "stage_label": "late_stability",
            "intervention_question": "Does the current MAPPO reward configuration remain stable and aligned at the second medium-budget checkpoint?",
            "recommended_candidate_focus": [
                "beta_down",
                "wc_down_wp_up",
                "reference_like",
                "recovery",
            ],
            "branch_budget_feasible": True,
        },
    ]
    profile["phase1_method"] = phase1_method
    if policy_guided:
        profile = _apply_policy_guidance_profile_overrides(profile)
        profile["enable_policy_guidance"] = True
        profile["policy_guidance_stage1_behavior_summary_required"] = False
        profile["policy_guidance_stage1_milestones"] = [15000000, 20000000, 30000000, 40000000]
        profile["policy_guidance_first_supported_envs"] = ["rware"]
        phase1_method = deepcopy(profile["phase1_method"])
        adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
        adaptive_replacement["policy_guidance_stage1_milestones"] = [15000000, 20000000, 30000000, 40000000]
        adaptive_replacement["policy_guidance_first_supported_envs"] = ["rware"]
        phase1_method["adaptive_replacement"] = adaptive_replacement
        profile["phase1_method"] = phase1_method
    else:
        profile["enable_policy_guidance"] = False
        phase1_method = deepcopy(profile["phase1_method"])
        phase1_method["enable_policy_guidance"] = False
        adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
        adaptive_replacement["enable_policy_guidance"] = False
        phase1_method["adaptive_replacement"] = adaptive_replacement
        profile["phase1_method"] = phase1_method
    return profile


def _profile_mappo_pbrs_v2_policy_guided_medium_seed1_sidecar() -> Dict[str, Any]:
    return _profile_mappo_pbrs_v2_medium_seed1(policy_guided=True)


def _profile_mappo_pbrs_v2_reward_only_medium_seed1_sidecar() -> Dict[str, Any]:
    return _profile_mappo_pbrs_v2_medium_seed1(policy_guided=False)


def _profile_mappo_pbrs_v2_policy_guided_medium_seed2_sidecar_reuse_sparse() -> Dict[str, Any]:
    profile = _profile_mappo_pbrs_v2_policy_guided_medium_seed1_sidecar()
    profile["profile_name"] = "mappo_pbrs_v2_policy_guided_medium_seed2_sidecar_reuse_sparse"
    profile["seed"] = 2
    profile["stage1_reuse"] = _build_shared_sparse_reuse_config(
        source_workflow_id="mappo_pbrs_v2_policy_guided_medium_seed1_sidecar_workflow",
        source_seed=1,
        reuse_behavior_summary=True,
        reuse_sparse_metrics_for_context=True,
    )
    return profile


def _profile_mappo_pbrs_v2_policy_guided_medium_seed3_sidecar_reuse_sparse() -> Dict[str, Any]:
    profile = _profile_mappo_pbrs_v2_policy_guided_medium_seed1_sidecar()
    profile["profile_name"] = "mappo_pbrs_v2_policy_guided_medium_seed3_sidecar_reuse_sparse"
    profile["seed"] = 3
    profile["stage1_reuse"] = _build_shared_sparse_reuse_config(
        source_workflow_id="mappo_pbrs_v2_policy_guided_medium_seed1_sidecar_workflow",
        source_seed=1,
        reuse_behavior_summary=True,
        reuse_sparse_metrics_for_context=True,
    )
    return profile


def _profile_mappo_pbrs_v2_reward_only_medium_seed2_sidecar_reuse_sparse_context_disabled() -> Dict[str, Any]:
    profile = _profile_mappo_pbrs_v2_reward_only_medium_seed1_sidecar()
    profile["profile_name"] = "mappo_pbrs_v2_reward_only_medium_seed2_sidecar_reuse_sparse_context_disabled"
    profile["seed"] = 2
    profile["stage1_reuse"] = _build_shared_sparse_reuse_config(
        source_workflow_id="mappo_pbrs_v2_reward_only_medium_seed1_sidecar_workflow",
        source_seed=1,
        reuse_behavior_summary=False,
        reuse_sparse_metrics_for_context=True,
    )
    return profile


def _profile_rware_mappo_pbrs_v2_formal_seed1(*, policy_guided: bool) -> Dict[str, Any]:
    profile = _profile_mappo_pbrs_v2_medium_seed1(policy_guided=policy_guided)
    profile["profile_name"] = (
        "mappo_rware_pbrs_v2_policy_guided_formal_seed1"
        if policy_guided
        else "mappo_rware_pbrs_v2_reward_only_formal_seed1"
    )
    profile["train_config"] = "mappo"
    profile["env_key"] = "rware-tiny-2ag-easy-v2"
    profile["time_limit"] = 50
    profile["pbrs_version"] = "rware_pbrs_v2"
    profile["budget_profile"] = "rware_mappo_formal"
    profile["formal_execute_guard_required"] = True
    profile["workflow_result_isolation_required"] = True
    profile["default_workflow_id_suffix"] = "formal_seed1"
    profile["policy_guidance_history_expected_count"] = 1 if policy_guided else 0
    profile["reward_only_expected_policy_guidance_history_count"] = 0
    profile["use_real_llm"] = False
    profile["use_real_llm_for_field_analysis"] = False
    profile["mock_llm_mode"] = True
    profile["eval_use_pbrs"] = False
    profile["sparse_eval_only"] = True
    profile["test_sparse_reward_only"] = True
    profile["t_max"] = 40000000
    profile["stage1_sparse_t_max"] = 40000000
    profile["dense_reference_t_max"] = 40000000
    profile["dense_reference_target_t_max"] = 40000000
    profile["adaptive_mainline_target_t_max"] = 40000000
    profile["fixed_intervention_checkpoints"] = [20000000, 30000000]
    profile["branch_budget_steps"] = 1000000
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 2
    profile["stage3_update_candidates_per_round"] = 2
    profile["stage3_include_no_change_control"] = True
    profile["stage3_no_change_counts_toward_round_budget"] = False
    profile["stage3_reuse_no_change_across_rounds"] = True
    profile["stage3_min_effective_update_candidates_per_checkpoint"] = 4
    profile["stage3_require_nontrivial_config_delta"] = True
    profile["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    profile["stage3_require_candidate_type_diversity"] = True
    profile["policy_guided_stage3_require_behavior_evidence"] = bool(policy_guided)
    profile["fallback_to_deterministic_on_llm_error"] = False
    profile["add_deterministic_recovery_fallback"] = False

    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["dense_reference_target_t_max"] = 40000000
    phase1_method["adaptive_mainline_target_t_max"] = 40000000
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["keep_dense_reference_continuation"] = True
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["use_fixed_intervention_checkpoints"] = True
    phase1_method["fixed_intervention_checkpoints"] = [20000000, 30000000]
    phase1_method["checkpoint_count"] = 2

    initial_dense_search = deepcopy(phase1_method.get("initial_dense_search") or {})
    initial_dense_search["enabled"] = True
    initial_dense_search["rounds"] = 2
    initial_dense_search["candidates_per_round"] = 2
    initial_dense_search["candidate_budget_steps"] = 15000000
    initial_dense_search["candidate_target_checkpoint_step"] = 15000000
    initial_dense_search["selected_endpoint_target_step"] = 15000000
    initial_dense_search["endpoint_tolerance_steps"] = 500000
    initial_dense_search["endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    initial_dense_search["parallel"] = True
    initial_dense_search["save_endpoint_checkpoint"] = True
    initial_dense_search["save_intermediate_checkpoints"] = False
    initial_dense_search["keep_only_endpoint_checkpoint"] = True
    phase1_method["initial_dense_search"] = initial_dense_search

    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = bool(policy_guided)
    adaptive_replacement["fixed_intervention_checkpoints"] = [20000000, 30000000]
    adaptive_replacement["branch_budget_steps"] = 1000000
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 2
    adaptive_replacement["stage3_update_candidates_per_round"] = 2
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["stage3_no_change_counts_toward_round_budget"] = False
    adaptive_replacement["stage3_reuse_no_change_across_rounds"] = True
    adaptive_replacement["stage3_min_effective_update_candidates_per_checkpoint"] = 4
    adaptive_replacement["stage3_require_nontrivial_config_delta"] = True
    adaptive_replacement["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    adaptive_replacement["stage3_require_candidate_type_diversity"] = True
    adaptive_replacement["policy_guided_stage3_require_behavior_evidence"] = bool(policy_guided)
    adaptive_replacement["fallback_to_deterministic_on_llm_error"] = False
    adaptive_replacement["add_deterministic_recovery_fallback"] = False
    adaptive_replacement["use_winner_branch_promotion"] = True
    phase1_method["adaptive_replacement"] = adaptive_replacement
    phase1_method["fixed_checkpoint_plan"] = [
        {
            "name": "C1",
            "step": 20000000,
            "stage_label": "post_initialization_transition",
            "intervention_question": "Does the selected Stage 1b RWARE MAPPO dense initialization remain suitable at the first formal checkpoint?",
            "recommended_candidate_focus": [
                "reference_like",
                "recovery",
                "goal_progress",
            ],
            "branch_budget_feasible": True,
        },
        {
            "name": "C2",
            "step": 30000000,
            "stage_label": "late_stability",
            "intervention_question": "Does the current RWARE MAPPO reward configuration remain stable and aligned at the second formal checkpoint?",
            "recommended_candidate_focus": [
                "reference_like",
                "recovery",
                "late_stability",
            ],
            "branch_budget_feasible": True,
        },
    ]
    profile["phase1_method"] = phase1_method

    if policy_guided:
        profile = _apply_policy_guidance_profile_overrides(profile)
        profile["enable_policy_guidance"] = True
        profile["policy_guidance_stage1_behavior_summary_required"] = False
        profile["policy_guidance_stage1_milestones"] = [15000000, 20000000, 30000000, 40000000]
        profile["policy_guidance_first_supported_envs"] = ["rware"]
        phase1_method = deepcopy(profile["phase1_method"])
        adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
        adaptive_replacement["policy_guidance_stage1_milestones"] = [15000000, 20000000, 30000000, 40000000]
        adaptive_replacement["policy_guidance_first_supported_envs"] = ["rware"]
        phase1_method["adaptive_replacement"] = adaptive_replacement
        profile["phase1_method"] = phase1_method
    else:
        profile["enable_policy_guidance"] = False
        phase1_method = deepcopy(profile["phase1_method"])
        phase1_method["enable_policy_guidance"] = False
        adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
        adaptive_replacement["enable_policy_guidance"] = False
        phase1_method["adaptive_replacement"] = adaptive_replacement
        profile["phase1_method"] = phase1_method
    return profile


def _profile_mappo_rware_pbrs_v2_policy_guided_formal_seed1() -> Dict[str, Any]:
    return _profile_rware_mappo_pbrs_v2_formal_seed1(policy_guided=True)


def _profile_mappo_rware_pbrs_v2_reward_only_formal_seed1() -> Dict[str, Any]:
    return _profile_rware_mappo_pbrs_v2_formal_seed1(policy_guided=False)


def _profile_rware_mappo_guarded_formal_execute_smoke_seed1(
    *,
    policy_guided: bool,
) -> Dict[str, Any]:
    profile = _profile_rware_mappo_pbrs_v2_formal_seed1(policy_guided=policy_guided)
    profile["profile_name"] = (
        "mappo_rware_pbrs_v2_policy_guided_guarded_formal_execute_smoke_seed1"
        if policy_guided
        else "mappo_rware_pbrs_v2_reward_only_guarded_formal_execute_smoke_seed1"
    )
    profile["budget_profile"] = "rware_guarded_formal_execute_smoke"
    profile["default_workflow_id_suffix"] = "guarded_formal_execute_smoke_seed1"
    profile["use_cuda"] = False
    profile["stage1_use_cuda"] = False
    profile["stage3_use_cuda"] = False
    profile["stage5_use_cuda"] = False
    profile["max_parallel_candidates"] = 2
    profile["use_real_llm"] = False
    profile["use_real_llm_for_field_analysis"] = False
    profile["mock_llm_mode"] = True
    profile["eval_use_pbrs"] = False
    profile["sparse_eval_only"] = True
    profile["test_sparse_reward_only"] = True
    profile["t_max"] = 10000
    profile["stage1_sparse_t_max"] = 5000
    profile["dense_reference_t_max"] = 5000
    profile["dense_reference_target_t_max"] = 5000
    profile["adaptive_mainline_target_t_max"] = 8000
    profile["fixed_intervention_checkpoints"] = [5000, 7000]
    profile["branch_budget_steps"] = 2000
    profile["initial_dense_search_rounds"] = 2
    profile["initial_dense_candidates_per_round"] = 2
    profile["initial_dense_candidate_budget_steps"] = 3000
    profile["initial_dense_candidate_target_checkpoint_step"] = 3000
    profile["stage1b_selected_endpoint_target_step"] = 3000
    profile["stage1b_endpoint_tolerance_steps"] = 500
    profile["stage1b_endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 2
    profile["stage3_update_candidates_per_round"] = 2
    profile["stage3_include_no_change_control"] = True
    profile["stage3_no_change_counts_toward_round_budget"] = False
    profile["stage3_reuse_no_change_across_rounds"] = True
    profile["stage3_min_effective_update_candidates_per_checkpoint"] = 4
    profile["stage3_require_nontrivial_config_delta"] = True
    profile["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    profile["stage3_require_candidate_type_diversity"] = True
    profile["policy_guided_stage3_require_behavior_evidence"] = bool(policy_guided)
    profile["fallback_to_deterministic_on_llm_error"] = False
    profile["add_deterministic_recovery_fallback"] = False

    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["dense_reference_target_t_max"] = 5000
    phase1_method["adaptive_mainline_target_t_max"] = 8000
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["keep_dense_reference_continuation"] = True
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["use_fixed_intervention_checkpoints"] = True
    phase1_method["fixed_intervention_checkpoints"] = [5000, 7000]
    phase1_method["checkpoint_count"] = 2
    phase1_method["enable_stage5_final_full_run"] = False
    phase1_method["final_spec_is_summary_only"] = True
    phase1_method["rolling_adaptive_mainline_is_final_result"] = True

    initial_dense_search = deepcopy(phase1_method.get("initial_dense_search") or {})
    initial_dense_search["enabled"] = True
    initial_dense_search["rounds"] = 2
    initial_dense_search["candidates_per_round"] = 2
    initial_dense_search["candidate_budget_steps"] = 3000
    initial_dense_search["candidate_target_checkpoint_step"] = 3000
    initial_dense_search["selected_endpoint_target_step"] = 3000
    initial_dense_search["endpoint_tolerance_steps"] = 500
    initial_dense_search["endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    initial_dense_search["parallel"] = True
    initial_dense_search["save_endpoint_checkpoint"] = True
    initial_dense_search["save_intermediate_checkpoints"] = False
    initial_dense_search["keep_only_endpoint_checkpoint"] = True
    phase1_method["initial_dense_search"] = initial_dense_search

    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = bool(policy_guided)
    adaptive_replacement["fixed_intervention_checkpoints"] = [5000, 7000]
    adaptive_replacement["branch_budget_steps"] = 2000
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 2
    adaptive_replacement["stage3_update_candidates_per_round"] = 2
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["stage3_no_change_counts_toward_round_budget"] = False
    adaptive_replacement["stage3_reuse_no_change_across_rounds"] = True
    adaptive_replacement["stage3_min_effective_update_candidates_per_checkpoint"] = 4
    adaptive_replacement["stage3_require_nontrivial_config_delta"] = True
    adaptive_replacement["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    adaptive_replacement["stage3_require_candidate_type_diversity"] = True
    adaptive_replacement["policy_guided_stage3_require_behavior_evidence"] = bool(policy_guided)
    adaptive_replacement["fallback_to_deterministic_on_llm_error"] = False
    adaptive_replacement["add_deterministic_recovery_fallback"] = False
    adaptive_replacement["use_winner_branch_promotion"] = True
    adaptive_replacement["mock_llm_mode"] = True
    phase1_method["adaptive_replacement"] = adaptive_replacement
    phase1_method["fixed_checkpoint_plan"] = [
        {
            "name": "C1",
            "step": 5000,
            "stage_label": "post_initialization_transition",
            "intervention_question": "Guarded formal execute smoke checkpoint after the tiny dense reference continuation.",
            "recommended_candidate_focus": [
                "reference_like",
                "recovery",
                "goal_progress",
            ],
            "branch_budget_feasible": True,
        },
        {
            "name": "C2",
            "step": 7000,
            "stage_label": "late_stability",
            "intervention_question": "Guarded formal execute smoke late checkpoint under tiny budget.",
            "recommended_candidate_focus": [
                "reference_like",
                "recovery",
                "late_stability",
            ],
            "branch_budget_feasible": True,
        },
    ]
    profile["phase1_method"] = phase1_method

    if policy_guided:
        profile["enable_policy_guidance"] = True
        profile["policy_guidance_stage1_behavior_summary_required"] = False
        profile["policy_guidance_stage1_milestones"] = [2000, 3000, 5000, 7000]
        profile["policy_guidance_first_supported_envs"] = ["rware"]
        phase1_method = deepcopy(profile["phase1_method"])
        adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
        adaptive_replacement["policy_guidance_stage1_milestones"] = [2000, 3000, 5000, 7000]
        adaptive_replacement["policy_guidance_first_supported_envs"] = ["rware"]
        phase1_method["adaptive_replacement"] = adaptive_replacement
        profile["phase1_method"] = phase1_method
    else:
        profile["enable_policy_guidance"] = False
        phase1_method = deepcopy(profile["phase1_method"])
        phase1_method["enable_policy_guidance"] = False
        adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
        adaptive_replacement["enable_policy_guidance"] = False
        phase1_method["adaptive_replacement"] = adaptive_replacement
        profile["phase1_method"] = phase1_method
    return profile


def _profile_mappo_rware_pbrs_v2_policy_guided_guarded_formal_execute_smoke_seed1() -> Dict[str, Any]:
    return _profile_rware_mappo_guarded_formal_execute_smoke_seed1(policy_guided=True)


def _profile_mappo_rware_pbrs_v2_reward_only_guarded_formal_execute_smoke_seed1() -> Dict[str, Any]:
    return _profile_rware_mappo_guarded_formal_execute_smoke_seed1(policy_guided=False)


def _build_mappo_policy_guided_medium_sidecar_profile(
    *,
    profile_name: str,
    env_key: str,
    stage3_use_cuda: bool,
    max_parallel_candidates: int,
) -> Dict[str, Any]:
    profile = _base_adaptive_profile()
    profile["phase1_method"] = deepcopy(_base_adaptive_profile()["phase1_method"])
    profile["profile_name"] = profile_name
    profile["seed"] = 1
    profile["train_config"] = "mappo"
    profile["env_key"] = env_key
    profile["time_limit"] = 50
    profile["t_max"] = 10000000
    profile["stage1_sparse_t_max"] = 10000000
    profile["dense_reference_t_max"] = 10000000
    profile["dense_reference_target_t_max"] = 10000000
    profile["adaptive_mainline_target_t_max"] = 10000000
    profile["use_cuda"] = True
    profile["stage1_use_cuda"] = True
    profile["stage3_use_cuda"] = bool(stage3_use_cuda)
    profile["stage5_use_cuda"] = True
    profile["max_parallel_candidates"] = int(max_parallel_candidates)
    profile["budget_profile"] = "mappo_medium_sidecar"
    profile["use_real_llm"] = False
    profile["use_real_llm_for_field_analysis"] = False
    profile["mock_llm_mode"] = True
    profile["reuse_policy_guidance_cache"] = True
    profile["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    profile["save_raw_trajectories"] = False
    profile["enable_dense_reference"] = True
    profile["dense_reference_sidecar_enabled"] = True
    profile["dense_reference_from_selected_stage1b_endpoint"] = True
    profile["dense_reference_fixed_pbrs"] = True
    profile["dense_reference_disable_adaptive_update"] = True
    profile["dense_reference_save_checkpoints"] = False
    profile["adaptive_mainline_from_selected_stage1b_endpoint"] = True
    profile["fixed_intervention_checkpoints"] = [1500000, 2500000]
    profile["branch_budget_steps"] = 500000
    profile["stage3_branch_rounds"] = 2
    profile["stage3_candidates_per_round"] = 3
    profile["stage3_update_candidates_per_round"] = 3
    profile["stage3_include_no_change_control"] = True
    profile["stage3_no_change_counts_toward_round_budget"] = False
    profile["stage3_reuse_no_change_across_rounds"] = True
    profile["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    profile["stage3_require_nontrivial_config_delta"] = True
    profile["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    profile["stage3_require_candidate_type_diversity"] = True
    profile["policy_guided_stage3_require_behavior_evidence"] = True
    profile = _apply_policy_guidance_profile_overrides(profile)
    profile = _mark_lbf_policy_guided_sidecar(profile)
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["dense_reference_target_t_max"] = 10000000
    phase1_method["adaptive_mainline_target_t_max"] = 10000000
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["keep_dense_reference_continuation"] = True
    phase1_method["reuse_policy_guidance_cache"] = True
    phase1_method["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    phase1_method["use_fixed_intervention_checkpoints"] = True
    phase1_method["fixed_intervention_checkpoints"] = [1500000, 2500000]
    phase1_method["checkpoint_count"] = 2
    initial_dense_search = deepcopy(phase1_method.get("initial_dense_search") or {})
    initial_dense_search["enabled"] = True
    initial_dense_search["rounds"] = 2
    initial_dense_search["candidates_per_round"] = 3
    initial_dense_search["candidate_budget_steps"] = 850000
    initial_dense_search["candidate_target_checkpoint_step"] = 800000
    initial_dense_search["selected_endpoint_target_step"] = 800000
    initial_dense_search["endpoint_tolerance_steps"] = 50000
    initial_dense_search["endpoint_selection_policy"] = "prefer_ge_then_le_nearest"
    initial_dense_search["parallel"] = True
    initial_dense_search["save_endpoint_checkpoint"] = True
    initial_dense_search["save_intermediate_checkpoints"] = False
    initial_dense_search["keep_only_endpoint_checkpoint"] = True
    phase1_method["initial_dense_search"] = initial_dense_search
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["enable_policy_guidance"] = True
    adaptive_replacement["reuse_policy_guidance_cache"] = True
    adaptive_replacement["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    adaptive_replacement["fixed_intervention_checkpoints"] = [1500000, 2500000]
    adaptive_replacement["branch_budget_steps"] = 500000
    adaptive_replacement["stage3_branch_rounds"] = 2
    adaptive_replacement["stage3_candidates_per_round"] = 3
    adaptive_replacement["stage3_update_candidates_per_round"] = 3
    adaptive_replacement["stage3_include_no_change_control"] = True
    adaptive_replacement["stage3_no_change_counts_toward_round_budget"] = False
    adaptive_replacement["stage3_reuse_no_change_across_rounds"] = True
    adaptive_replacement["stage3_min_effective_update_candidates_per_checkpoint"] = 6
    adaptive_replacement["stage3_require_nontrivial_config_delta"] = True
    adaptive_replacement["stage3_max_fallback_style_candidates_per_checkpoint"] = 1
    adaptive_replacement["stage3_require_candidate_type_diversity"] = True
    adaptive_replacement["policy_guided_stage3_require_behavior_evidence"] = True
    adaptive_replacement["use_winner_branch_promotion"] = True
    phase1_method["adaptive_replacement"] = adaptive_replacement
    checkpoint_name = "MAPPO"
    phase1_method["fixed_checkpoint_plan"] = [
        {
            "name": "C1",
            "step": 1500000,
            "stage_label": "post_initialization_transition",
            "intervention_question": f"Does the selected Stage 1b {checkpoint_name} dense initialization remain suitable at the first medium-budget checkpoint?",
            "recommended_candidate_focus": ["beta_down", "beta_up", "reference_like", "recovery"],
            "branch_budget_feasible": True,
        },
        {
            "name": "C2",
            "step": 2500000,
            "stage_label": "late_stability",
            "intervention_question": f"Does the current {checkpoint_name} reward configuration remain stable and aligned at the second medium-budget checkpoint?",
            "recommended_candidate_focus": ["beta_down", "wc_down_wp_up", "reference_like", "recovery"],
            "branch_budget_feasible": True,
        },
    ]
    profile["phase1_method"] = phase1_method
    return profile


def _profile_mappo_pbrs_v2_policy_guided_medium_8x8_2p_2f_sidecar() -> Dict[str, Any]:
    return _apply_clean_lbf_profile_flags(
        _build_mappo_policy_guided_medium_sidecar_profile(
            profile_name="mappo_pbrs_v2_policy_guided_medium_8x8_2p_2f_sidecar",
            env_key="lbforaging:Foraging-8x8-2p-2f-coop-v3",
            stage3_use_cuda=True,
            max_parallel_candidates=6,
        ),
        use_real_llm=True,
        use_real_llm_for_field_analysis=True,
    )


def _profile_mappo_pbrs_v2_policy_guided_medium_2s_8x8_2p_2f_sidecar() -> Dict[str, Any]:
    return _apply_clean_lbf_profile_flags(
        _build_mappo_policy_guided_medium_sidecar_profile(
            profile_name="mappo_pbrs_v2_policy_guided_medium_2s_8x8_2p_2f_sidecar",
            env_key="lbforaging:Foraging-2s-8x8-2p-2f-coop-v3",
            stage3_use_cuda=True,
            max_parallel_candidates=6,
        ),
        use_real_llm=True,
        use_real_llm_for_field_analysis=True,
    )


def _profile_mappo_pbrs_v2_policy_guided_medium_10x10_2p_1f_sidecar() -> Dict[str, Any]:
    return _apply_clean_lbf_profile_flags(
        _build_mappo_policy_guided_medium_sidecar_profile(
            profile_name="mappo_pbrs_v2_policy_guided_medium_10x10_2p_1f_sidecar",
            env_key="lbforaging:Foraging-10x10-2p-1f-v3",
            stage3_use_cuda=False,
            max_parallel_candidates=6,
        ),
        use_real_llm=True,
        use_real_llm_for_field_analysis=True,
    )


def _profile_mappo_pbrs_v2_policy_guided_medium_15x15_3p_4f_sidecar() -> Dict[str, Any]:
    return _apply_clean_lbf_profile_flags(
        _build_mappo_policy_guided_medium_sidecar_profile(
            profile_name="mappo_pbrs_v2_policy_guided_medium_15x15_3p_4f_sidecar",
            env_key="lbforaging:Foraging-15x15-3p-4f-v3",
            stage3_use_cuda=False,
            max_parallel_candidates=3,
        ),
        use_real_llm=True,
        use_real_llm_for_field_analysis=True,
    )


def _profile_mappo_lbf_pbrs_v2_15x15_3p_4f_full_clean_seed1() -> Dict[str, Any]:
    return _configure_stage1reuse_adaptive_full_clean_profile(
        _profile_mappo_pbrs_v2_policy_guided_medium_15x15_3p_4f_sidecar(),
        profile_name="mappo_lbf_pbrs_v2_15x15_3p_4f_full_clean_seed1",
        bundle_id="mappo_15x15_3p_4f_seed1",
        budget_profile="mappo_lbf_pbrs_v2_full_clean",
    )
 

def _profile_mappo_pbrs_v2_policy_guided_medium_8x8_2p_1f_sidecar() -> Dict[str, Any]:
    return _apply_clean_lbf_profile_flags(
        _build_mappo_policy_guided_medium_sidecar_profile(
            profile_name="mappo_pbrs_v2_policy_guided_medium_8x8_2p_1f_sidecar",
            env_key="lbforaging:Foraging-8x8-2p-1f-v3",
            stage3_use_cuda=True,
            max_parallel_candidates=6,
        ),
        use_real_llm=True,
        use_real_llm_for_field_analysis=True,
    )


def _build_mappo_model52_reuse_stage1_rerun_profile(
    *,
    profile_name: str,
    env_key: str,
    source_workflow_id: str,
    stage3_use_cuda: bool,
    max_parallel_candidates: int,
) -> Dict[str, Any]:
    profile = _build_mappo_policy_guided_medium_sidecar_profile(
        profile_name=profile_name,
        env_key=env_key,
        stage3_use_cuda=stage3_use_cuda,
        max_parallel_candidates=max_parallel_candidates,
    )
    profile["model"] = "gpt-5.2"
    profile["use_real_llm"] = True
    profile["use_real_llm_for_field_analysis"] = True
    profile["mock_llm_mode"] = False
    profile["reuse_policy_guidance_cache"] = True
    profile["policy_guidance_cache_lookup_mode"] = "exact_or_latest_by_intervention"
    profile["stage1_reuse"] = _build_stage1_completed_artifact_reuse_config(
        source_workflow_id=source_workflow_id,
        source_seed=1,
        source_results_root=ARCHIVED_MAPPO_QMIX_RESULTS_ROOT,
        reuse_behavior_summary=True,
        reuse_sparse_metrics_for_context=True,
    )
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["reuse_policy_guidance_cache"] = True
    adaptive_replacement = deepcopy(phase1_method.get("adaptive_replacement") or {})
    adaptive_replacement["reuse_policy_guidance_cache"] = True
    adaptive_replacement["llm_stage3_use_cache"] = True
    phase1_method["adaptive_replacement"] = adaptive_replacement
    profile["phase1_method"] = phase1_method
    return profile


def _profile_mappo_pbrs_v2_policy_guided_medium_8x8_2p_2f_sidecar_run2_model52_reuse_stage1() -> Dict[str, Any]:
    return _build_mappo_model52_reuse_stage1_rerun_profile(
        profile_name="mappo_pbrs_v2_policy_guided_medium_8x8_2p_2f_sidecar_run2_model52_reuse_stage1",
        env_key="lbforaging:Foraging-8x8-2p-2f-coop-v3",
        source_workflow_id="mappo_pbrs_v2_policy_guided_medium_8x8_2p_2f_sidecar_run",
        stage3_use_cuda=True,
        max_parallel_candidates=6,
    )


def _profile_mappo_pbrs_v2_policy_guided_medium_2s_8x8_2p_2f_sidecar_run2_model52_reuse_stage1() -> Dict[str, Any]:
    return _build_mappo_model52_reuse_stage1_rerun_profile(
        profile_name="mappo_pbrs_v2_policy_guided_medium_2s_8x8_2p_2f_sidecar_run2_model52_reuse_stage1",
        env_key="lbforaging:Foraging-2s-8x8-2p-2f-coop-v3",
        source_workflow_id="mappo_pbrs_v2_policy_guided_medium_2s_8x8_2p_2f_sidecar_run",
        stage3_use_cuda=True,
        max_parallel_candidates=6,
    )


def _profile_mappo_pbrs_v2_policy_guided_medium_10x10_2p_1f_sidecar_run2_model52_reuse_stage1() -> Dict[str, Any]:
    return _build_mappo_model52_reuse_stage1_rerun_profile(
        profile_name="mappo_pbrs_v2_policy_guided_medium_10x10_2p_1f_sidecar_run2_model52_reuse_stage1",
        env_key="lbforaging:Foraging-10x10-2p-1f-v3",
        source_workflow_id="mappo_pbrs_v2_policy_guided_medium_10x10_2p_1f_sidecar_run",
        stage3_use_cuda=False,
        max_parallel_candidates=6,
    )


def _profile_mappo_pbrs_v2_policy_guided_medium_15x15_3p_4f_sidecar_run2_model52_reuse_stage1() -> Dict[str, Any]:
    return _build_mappo_model52_reuse_stage1_rerun_profile(
        profile_name="mappo_pbrs_v2_policy_guided_medium_15x15_3p_4f_sidecar_run2_model52_reuse_stage1",
        env_key="lbforaging:Foraging-15x15-3p-4f-v3",
        source_workflow_id="mappo_pbrs_v2_policy_guided_medium_15x15_3p_4f_sidecar_run",
        stage3_use_cuda=False,
        max_parallel_candidates=3,
    )


def _profile_15x15_qmix_adaptive() -> Dict[str, Any]:
    profile = _base_adaptive_profile()
    profile.update(_profile_15x15_qmix())
    profile["profile_name"] = "lbf_15x15_qmix_adaptive"
    profile["phase1_method"] = deepcopy(_base_adaptive_profile()["phase1_method"])
    return _apply_new_adaptive_budget_defaults(profile)


def _profile_dual_llm_stage1b_tiny_smoke() -> Dict[str, Any]:
    profile = _base_adaptive_profile()
    profile.update(_profile_8x8_qmix())
    profile.update(
        {
            "profile_name": "dual_llm_stage1b_tiny_smoke",
            "seed": 1,
            "use_cuda": False,
            "stage1_use_cuda": False,
            "stage3_use_cuda": False,
            "stage5_use_cuda": False,
            "t_max": 20000,
            "test_interval": 2500,
            "runner_log_interval": 2500,
            "learner_log_interval": 2500,
            "save_model_interval": 2500,
            "branch_budget_steps": 3000,
            "budget_profile": "tiny_smoke",
            "use_real_llm": False,
            "use_real_llm_for_field_analysis": False,
            "initial_dense_candidates_per_round": 2,
            "initial_dense_candidate_budget_steps": 5000,
            "fixed_intervention_checkpoints": [10000, 15000],
            "checkpoint_count": 2,
        }
    )
    phase1_method = deepcopy(_base_adaptive_profile()["phase1_method"])
    phase1_method["fixed_intervention_checkpoints"] = [10000, 15000]
    phase1_method["checkpoint_count"] = 2
    phase1_method["dense_reference_target_t_max"] = 15000
    phase1_method["initial_dense_search"]["candidates_per_round"] = 2
    phase1_method["initial_dense_search"]["candidate_budget_steps"] = 5000
    phase1_method["adaptive_replacement"]["fixed_intervention_checkpoints"] = [10000, 15000]
    phase1_method["adaptive_replacement"]["branch_budget_steps"] = 3000
    phase1_method["fixed_checkpoint_plan"] = [
        {
            "name": "C1",
            "step": 10000,
            "stage_label": "post_initialization_transition",
            "intervention_question": "Tiny smoke checkpoint after the early initialization run.",
            "recommended_candidate_focus": ["reference_like", "recovery"],
            "branch_budget_feasible": True,
        },
        {
            "name": "C2",
            "step": 15000,
            "stage_label": "late_stability",
            "intervention_question": "Tiny smoke late checkpoint for wiring validation.",
            "recommended_candidate_focus": ["reference_like", "beta_down"],
            "branch_budget_feasible": True,
        },
    ]
    profile["phase1_method"] = phase1_method
    return _apply_new_adaptive_budget_defaults(profile)


def _profile_dual_llm_stage1b_tiny_execute_smoke() -> Dict[str, Any]:
    profile = _profile_dual_llm_stage1b_tiny_smoke()
    profile.update(
        {
            "profile_name": "dual_llm_stage1b_tiny_execute_smoke",
            "budget_profile": "tiny_execute_smoke",
            "initial_dense_candidate_parallel": True,
            "initial_dense_candidate_save_endpoint_checkpoint": True,
        }
    )
    profile["phase1_method"] = deepcopy(profile["phase1_method"])
    profile["phase1_method"]["dense_reference_target_t_max"] = 15000
    profile["phase1_method"]["adaptive_mainline_target_t_max"] = 15000
    profile["phase1_method"]["fixed_intervention_checkpoints"] = [12000, 16000]
    profile["phase1_method"]["fixed_checkpoint_plan"] = [
        {
            "name": "C1",
            "step": 12000,
            "stage_label": "post_initialization_transition",
            "intervention_question": "Tiny execute smoke checkpoint after selected initialization.",
            "recommended_candidate_focus": ["reference_like", "recovery"],
            "branch_budget_feasible": True,
        },
        {
            "name": "C2",
            "step": 16000,
            "stage_label": "late_stability",
            "intervention_question": "Tiny execute smoke late checkpoint.",
            "recommended_candidate_focus": ["reference_like", "beta_down"],
            "branch_budget_feasible": True,
        },
    ]
    return profile


def _profile_dual_llm_stage3_tiny_execute_smoke() -> Dict[str, Any]:
    profile = _profile_dual_llm_stage1b_tiny_execute_smoke()
    profile.update(
        {
            "profile_name": "dual_llm_stage3_tiny_execute_smoke",
            "budget_profile": "tiny_stage3_execute_smoke",
            "branch_budget_steps": 3000,
            "max_parallel_candidates": 3,
            "use_real_llm": False,
            "t_max": 19000,
        }
    )
    phase1_method = deepcopy(profile["phase1_method"])
    phase1_method["fixed_intervention_checkpoints"] = [12000, 16000]
    phase1_method["checkpoint_count"] = 2
    phase1_method["use_dual_llm_stage1b_initial_search"] = True
    phase1_method["dense_reference_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_mainline_starts_from_stage1b_best_checkpoint"] = True
    phase1_method["adaptive_replacement"]["fixed_intervention_checkpoints"] = [12000, 16000]
    phase1_method["adaptive_replacement"]["branch_budget_steps"] = 3000
    phase1_method["adaptive_replacement"]["stage3_branch_rounds"] = 2
    phase1_method["adaptive_replacement"]["stage3_candidates_per_round"] = 3
    phase1_method["adaptive_replacement"]["stage3_include_no_change_control"] = True
    phase1_method["adaptive_replacement"]["stage3_no_change_counts_toward_round_budget"] = True
    phase1_method["adaptive_replacement"]["stage3_candidate_review_enabled"] = True
    phase1_method["adaptive_replacement"]["stage3_candidate_repair_max_rounds"] = 1
    phase1_method["adaptive_replacement"]["use_winner_branch_promotion"] = True
    phase1_method["adaptive_replacement"]["fallback_to_existing_stage3"] = True
    phase1_method["adaptive_replacement"]["mock_llm_mode"] = True
    phase1_method["fixed_checkpoint_plan"] = [
        {
            "name": "C1",
            "step": 12000,
            "stage_label": "post_initialization_transition",
            "intervention_question": "Tiny Stage 3 smoke checkpoint after the Stage 1b selected source continuation.",
            "recommended_candidate_focus": ["reference_like", "recovery"],
            "branch_budget_feasible": True,
        },
        {
            "name": "C2",
            "step": 16000,
            "stage_label": "late_stability",
            "intervention_question": "Tiny Stage 3 smoke late checkpoint for winner-promotion validation.",
            "recommended_candidate_focus": ["reference_like", "beta_down"],
            "branch_budget_feasible": True,
        },
    ]
    profile["phase1_method"] = phase1_method
    return profile


def _sync_legacy_resource_keys(profile: Dict[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(profile)
    resource_policy = deepcopy(resolved.get("resource_policy") or {})
    resource_policy["stage1_use_cuda"] = bool(resolved.get("use_cuda"))
    resource_policy["stage3_use_cuda"] = bool(resolved.get("stage3_use_cuda"))
    resource_policy["stage3_max_parallel_candidates"] = int(resolved.get("max_parallel_candidates") or 0)
    resource_policy["stage5_use_cuda"] = bool(resolved.get("use_cuda"))
    resolved["resource_policy"] = resource_policy
    return resolved


WORKFLOW_PROFILES: Dict[str, Dict[str, Any]] = {
    "lbf_8x8_qmix": _profile_8x8_qmix(),
    "lbf_8x8_2p2f_qmix": _profile_8x8_2p2f_qmix(),
    "lbf_2s_8x8_2p2f_qmix": _profile_2s_8x8_2p2f_qmix(),
    "lbf_10x10_qmix": _profile_10x10_qmix(),
    "lbf_10x10_3p3f_qmix": _profile_10x10_3p3f_qmix(),
    "lbf_15x15_qmix": _profile_15x15_qmix(),
    "lbf_8x8_qmix_adaptive": _profile_8x8_qmix_adaptive(),
    "lbf_8x8_2p2f_qmix_adaptive": _profile_8x8_2p2f_qmix_adaptive(),
    "lbf_2s_8x8_2p2f_qmix_adaptive": _profile_2s_8x8_2p2f_qmix_adaptive(),
    "lbf_10x10_qmix_adaptive": _profile_10x10_qmix_adaptive(),
    "lbf_10x10_3p3f_qmix_adaptive": _profile_10x10_3p3f_qmix_adaptive(),
    "qmix_dual_llm_single_seed_pilot": _profile_qmix_dual_llm_single_seed_pilot(),
    "qmix_dual_llm_policy_guided_single_seed_pilot": _profile_qmix_dual_llm_policy_guided_single_seed_pilot(),
    "qmix_policy_guidance_stage1_tiny_smoke": _profile_qmix_policy_guidance_stage1_tiny_smoke(),
    "qmix_policy_guidance_stage1b_tiny_smoke": _profile_qmix_policy_guidance_stage1b_tiny_smoke(),
    "qmix_policy_guidance_stage3_hook_tiny_smoke": _profile_qmix_policy_guidance_stage3_hook_tiny_smoke(),
    "qmix_policy_guided_tiny_end_to_end_smoke": _profile_qmix_policy_guided_tiny_end_to_end_smoke(),
    "qmix_policy_guided_tiny_execute_smoke": _profile_qmix_policy_guided_tiny_execute_smoke(),
    "qmix_policy_guided_sidecar_stage3_budget_smoke_seed1": _profile_qmix_policy_guided_sidecar_stage3_budget_smoke_seed1(),
    "qmix_policy_guided_small_pilot_seed1": _profile_qmix_policy_guided_small_pilot_seed1(),
    "qmix_single_llm_pbrs_v2_reward_gen_seed1": _profile_qmix_single_llm_pbrs_v2_reward_gen_seed1(),
    "qmix_single_llm_pbrs_v2_reward_gen_8x8_2p_1f_seed1": _profile_qmix_single_llm_pbrs_v2_reward_gen_8x8_2p_1f_seed1(),
    "qmix_single_llm_pbrs_v2_reward_gen_8x8_2p_2f_seed1": _profile_qmix_single_llm_pbrs_v2_reward_gen_8x8_2p_2f_seed1(),
    "qmix_single_llm_pbrs_v2_reward_gen_2s_8x8_2p_2f_seed1": _profile_qmix_single_llm_pbrs_v2_reward_gen_2s_8x8_2p_2f_seed1(),
    "qmix_single_llm_pbrs_v2_reward_gen_10x10_2p_1f_seed1": _profile_qmix_single_llm_pbrs_v2_reward_gen_10x10_2p_1f_seed1(),
    "qmix_single_llm_pbrs_v2_reward_gen_15x15_3p_4f_seed1": _profile_qmix_single_llm_pbrs_v2_reward_gen_15x15_3p_4f_seed1(),
    "qmix_reward_only_small_pilot_seed1": _profile_qmix_reward_only_small_pilot_seed1(),
    "qmix_lbf_pbrs_v2_single_clean_workflow_pilot_seed1": _profile_qmix_lbf_pbrs_v2_single_clean_workflow_pilot_seed1(),
    "qmix_policy_guided_formal_pilot_seed1": _profile_qmix_policy_guided_formal_pilot_seed1(),
    "qmix_reward_only_formal_pilot_seed1": _profile_qmix_reward_only_formal_pilot_seed1(),
    "qmix_lbf_pbrs_v2_8x8_2p_1f_full_clean_seed1": _profile_qmix_lbf_pbrs_v2_8x8_2p_1f_full_clean_seed1(),
    "qmix_lbf_pbrs_v2_10x10_2p_1f_full_clean_seed1": _profile_qmix_lbf_pbrs_v2_10x10_2p_1f_full_clean_seed1(),
    "qmix_lbf_pbrs_v2_8x8_2p_2f_full_clean_seed1": _profile_qmix_lbf_pbrs_v2_8x8_2p_2f_full_clean_seed1(),
    "qmix_lbf_pbrs_v2_2s_8x8_2p_2f_full_clean_seed1": _profile_qmix_lbf_pbrs_v2_2s_8x8_2p_2f_full_clean_seed1(),
    "qmix_lbf_pbrs_v2_15x15_3p_4f_full_clean_seed1": _profile_qmix_lbf_pbrs_v2_15x15_3p_4f_full_clean_seed1(),
    "qmix_lbf_pbrs_v2_10x10_2p_1f_stage1reuse_stage1b_50k_deepseek_smoke": _profile_qmix_lbf_pbrs_v2_10x10_2p_1f_stage1reuse_stage1b_50k_deepseek_smoke(),
    "qmix_pbrs_v2_policy_guided_8x8_2p_1f_sidecar": _profile_qmix_pbrs_v2_policy_guided_8x8_2p_1f_sidecar(),
    "qmix_pbrs_v2_policy_guided_8x8_2p_2f_sidecar": _profile_qmix_pbrs_v2_policy_guided_8x8_2p_2f_sidecar(),
    "qmix_pbrs_v2_policy_guided_2s_8x8_2p_2f_sidecar": _profile_qmix_pbrs_v2_policy_guided_2s_8x8_2p_2f_sidecar(),
    "qmix_pbrs_v2_policy_guided_10x10_2p_1f_sidecar": _profile_qmix_pbrs_v2_policy_guided_10x10_2p_1f_sidecar(),
    "qmix_pbrs_v2_policy_guided_15x15_3p_4f_sidecar": _profile_qmix_pbrs_v2_policy_guided_15x15_3p_4f_sidecar(),
    "qmix_pbrs_v2_policy_guided_8x8_2p_2f_sidecar_run2_model52_reuse_stage1": _profile_qmix_pbrs_v2_policy_guided_8x8_2p_2f_sidecar_run2_model52_reuse_stage1(),
    "qmix_pbrs_v2_policy_guided_2s_8x8_2p_2f_sidecar_run2_model52_reuse_stage1": _profile_qmix_pbrs_v2_policy_guided_2s_8x8_2p_2f_sidecar_run2_model52_reuse_stage1(),
    "qmix_pbrs_v2_policy_guided_10x10_2p_1f_sidecar_run2_model52_reuse_stage1": _profile_qmix_pbrs_v2_policy_guided_10x10_2p_1f_sidecar_run2_model52_reuse_stage1(),
    "qmix_pbrs_v2_policy_guided_15x15_3p_4f_sidecar_run2_model52_reuse_stage1": _profile_qmix_pbrs_v2_policy_guided_15x15_3p_4f_sidecar_run2_model52_reuse_stage1(),
    "mappo_pbrs_v2_policy_guided_medium_seed1_sidecar": _profile_mappo_pbrs_v2_policy_guided_medium_seed1_sidecar(),
    "mappo_pbrs_v2_reward_only_medium_seed1_sidecar": _profile_mappo_pbrs_v2_reward_only_medium_seed1_sidecar(),
    "mappo_pbrs_v2_policy_guided_medium_seed2_sidecar_reuse_sparse": _profile_mappo_pbrs_v2_policy_guided_medium_seed2_sidecar_reuse_sparse(),
    "mappo_pbrs_v2_policy_guided_medium_seed3_sidecar_reuse_sparse": _profile_mappo_pbrs_v2_policy_guided_medium_seed3_sidecar_reuse_sparse(),
    "mappo_pbrs_v2_reward_only_medium_seed2_sidecar_reuse_sparse_context_disabled": _profile_mappo_pbrs_v2_reward_only_medium_seed2_sidecar_reuse_sparse_context_disabled(),
    "mappo_single_llm_pbrs_v2_reward_gen_8x8_2p_1f_seed1": _profile_mappo_single_llm_pbrs_v2_reward_gen_8x8_2p_1f_seed1(),
    "mappo_single_llm_pbrs_v2_reward_gen_8x8_2p_2f_seed1": _profile_mappo_single_llm_pbrs_v2_reward_gen_8x8_2p_2f_seed1(),
    "mappo_single_llm_pbrs_v2_reward_gen_2s_8x8_2p_2f_seed1": _profile_mappo_single_llm_pbrs_v2_reward_gen_2s_8x8_2p_2f_seed1(),
    "mappo_single_llm_pbrs_v2_reward_gen_10x10_2p_1f_seed1": _profile_mappo_single_llm_pbrs_v2_reward_gen_10x10_2p_1f_seed1(),
    "mappo_single_llm_pbrs_v2_reward_gen_15x15_3p_4f_seed1": _profile_mappo_single_llm_pbrs_v2_reward_gen_15x15_3p_4f_seed1(),
    "mappo_lbf_pbrs_v2_15x15_3p_4f_full_clean_seed1": _profile_mappo_lbf_pbrs_v2_15x15_3p_4f_full_clean_seed1(),
    "mappo_pbrs_v2_policy_guided_medium_8x8_2p_1f_sidecar": _profile_mappo_pbrs_v2_policy_guided_medium_8x8_2p_1f_sidecar(),
    "mappo_pbrs_v2_policy_guided_medium_8x8_2p_2f_sidecar": _profile_mappo_pbrs_v2_policy_guided_medium_8x8_2p_2f_sidecar(),
    "mappo_pbrs_v2_policy_guided_medium_2s_8x8_2p_2f_sidecar": _profile_mappo_pbrs_v2_policy_guided_medium_2s_8x8_2p_2f_sidecar(),
    "mappo_pbrs_v2_policy_guided_medium_10x10_2p_1f_sidecar": _profile_mappo_pbrs_v2_policy_guided_medium_10x10_2p_1f_sidecar(),
    "mappo_pbrs_v2_policy_guided_medium_15x15_3p_4f_sidecar": _profile_mappo_pbrs_v2_policy_guided_medium_15x15_3p_4f_sidecar(),
    "mappo_pbrs_v2_policy_guided_medium_8x8_2p_2f_sidecar_run2_model52_reuse_stage1": _profile_mappo_pbrs_v2_policy_guided_medium_8x8_2p_2f_sidecar_run2_model52_reuse_stage1(),
    "mappo_pbrs_v2_policy_guided_medium_2s_8x8_2p_2f_sidecar_run2_model52_reuse_stage1": _profile_mappo_pbrs_v2_policy_guided_medium_2s_8x8_2p_2f_sidecar_run2_model52_reuse_stage1(),
    "mappo_pbrs_v2_policy_guided_medium_10x10_2p_1f_sidecar_run2_model52_reuse_stage1": _profile_mappo_pbrs_v2_policy_guided_medium_10x10_2p_1f_sidecar_run2_model52_reuse_stage1(),
    "mappo_pbrs_v2_policy_guided_medium_15x15_3p_4f_sidecar_run2_model52_reuse_stage1": _profile_mappo_pbrs_v2_policy_guided_medium_15x15_3p_4f_sidecar_run2_model52_reuse_stage1(),
    "mappo_rware_pbrs_v2_policy_guided_formal_seed1": _profile_mappo_rware_pbrs_v2_policy_guided_formal_seed1(),
    "mappo_rware_pbrs_v2_reward_only_formal_seed1": _profile_mappo_rware_pbrs_v2_reward_only_formal_seed1(),
    "mappo_rware_pbrs_v2_policy_guided_guarded_formal_execute_smoke_seed1": _profile_mappo_rware_pbrs_v2_policy_guided_guarded_formal_execute_smoke_seed1(),
    "mappo_rware_pbrs_v2_reward_only_guarded_formal_execute_smoke_seed1": _profile_mappo_rware_pbrs_v2_reward_only_guarded_formal_execute_smoke_seed1(),
    "qmix_lbf_8x8_2p_2f_dual_llm_single_seed_pilot": _profile_qmix_lbf_8x8_2p_2f_dual_llm_single_seed_pilot(),
    "qmix_lbf_2s_8x8_2p_2f_dual_llm_single_seed_pilot": _profile_qmix_lbf_2s_8x8_2p_2f_dual_llm_single_seed_pilot(),
    "lbf_15x15_qmix_adaptive": _profile_15x15_qmix_adaptive(),
    "dual_llm_stage1b_tiny_smoke": _profile_dual_llm_stage1b_tiny_smoke(),
    "dual_llm_stage1b_tiny_execute_smoke": _profile_dual_llm_stage1b_tiny_execute_smoke(),
    "dual_llm_stage3_tiny_execute_smoke": _profile_dual_llm_stage3_tiny_execute_smoke(),
}

DEPRECATED_WORKFLOW_PROFILES = {
    name
    for name, profile in WORKFLOW_PROFILES.items()
    if (
        str(profile.get("workflow_kind") or "") != "single_llm_pbrs_v2_baseline"
        and str((profile.get("phase1_method") or {}).get("method_version") or "") != "phase1_adaptive_replacement_v1"
    )
}


def list_workflow_profiles() -> list[str]:
    return sorted(WORKFLOW_PROFILES.keys())


def list_launchable_workflow_profiles() -> list[str]:
    """Return only the adaptive end-to-end profiles that are safe for new launches."""

    return sorted(name for name in WORKFLOW_PROFILES.keys() if name not in DEPRECATED_WORKFLOW_PROFILES)


def is_summary_only_sidecar_profile(name: str) -> bool:
    profile = WORKFLOW_PROFILES[name]
    phase1_method = profile.get("phase1_method") or {}
    return bool(
        profile.get("summary_only_sidecar")
        or "sidecar" in str(name).lower()
        or (
            phase1_method.get("final_spec_is_summary_only", False)
            and not bool(profile.get("full_clean_runnable"))
        )
    )


def _portable_stage1_reuse_config_is_valid(stage1_reuse: Dict[str, Any]) -> bool:
    if not stage1_reuse or not bool(stage1_reuse.get("enabled", False)):
        return True
    return bool(
        str(stage1_reuse.get("reuse_mode") or "") == "portable_stage1_bundle"
        and str(stage1_reuse.get("bundle_id") or "").strip()
        and bool(stage1_reuse.get("skip_sparse_training", False))
        and str(stage1_reuse.get("bundle_root") or "").strip()
    )


def is_full_clean_runnable_workflow_profile(name: str) -> bool:
    profile = WORKFLOW_PROFILES[name]
    phase1_method = profile.get("phase1_method") or {}
    adaptive_replacement = phase1_method.get("adaptive_replacement") or {}
    stage1_reuse = profile.get("stage1_reuse") or {}
    fixed_checkpoint_plan = list(phase1_method.get("fixed_checkpoint_plan") or [])
    checkpoint_names = [str(item.get("name") or "") for item in fixed_checkpoint_plan]
    adaptive_mainline_is_final_result = bool(
        profile.get("rolling_adaptive_mainline_is_final_result")
        or phase1_method.get("rolling_adaptive_mainline_is_final_result")
    )
    return bool(
        profile.get("full_clean_runnable")
        and str(phase1_method.get("method_version") or "") == "phase1_adaptive_replacement_v1"
        and str(profile.get("pbrs_version") or "") == "lbf_pbrs_v2"
        and bool(phase1_method.get("use_dual_llm_stage1b_initial_search", False))
        and bool(adaptive_replacement.get("enable_policy_guidance", False))
        and bool(adaptive_replacement.get("stage3_branch_launch_enabled", True))
        and checkpoint_names[:2] == ["C1", "C2"]
        and int(adaptive_replacement.get("stage3_branch_rounds") or 0) == 2
        and int(
            adaptive_replacement.get("stage3_update_candidates_per_round")
            or adaptive_replacement.get("stage3_candidates_per_round")
            or 0
        )
        == 3
        and bool(adaptive_replacement.get("stage3_include_no_change_control", False))
        and not bool(adaptive_replacement.get("stage3_no_change_counts_toward_round_budget", True))
        and bool(adaptive_replacement.get("stage3_reuse_no_change_across_rounds", False))
        and int(adaptive_replacement.get("stage3_min_effective_update_candidates_per_checkpoint") or 0)
        == 6
        and not bool(
            profile.get("fallback_to_single_llm_initial_config")
            if profile.get("fallback_to_single_llm_initial_config") is not None
            else phase1_method.get("fallback_to_single_llm_initial_config", True)
        )
        and not bool(adaptive_replacement.get("fallback_to_deterministic_on_llm_error", True))
        and not bool(adaptive_replacement.get("add_deterministic_recovery_fallback", True))
        and bool(profile.get("sparse_eval_only", False))
        and not bool(profile.get("eval_use_pbrs", True))
        and not bool(profile.get("apply_dense_reward_in_eval", True))
        and _portable_stage1_reuse_config_is_valid(stage1_reuse)
        and adaptive_mainline_is_final_result
    )


def get_workflow_profile(name: str) -> Dict[str, Any]:
    try:
        return _sync_legacy_resource_keys(
            _apply_stage3_effective_update_budget_defaults(WORKFLOW_PROFILES[name])
        )
    except KeyError as exc:
        available = ", ".join(list_workflow_profiles())
        raise ValueError(f"unknown workflow profile: {name}. available: {available}") from exc


def is_deprecated_workflow_profile(name: str) -> bool:
    return name in DEPRECATED_WORKFLOW_PROFILES
