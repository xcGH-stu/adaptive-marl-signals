from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workflows.clients.critic_client import StubCriticBackend, TemplateBasedCriticClient
from workflows.clients.generator_client import (
    StubGeneratorBackend,
    TemplateBasedGeneratorClient,
)
from workflows.clients.openai_backend import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    FallbackLLMBackend,
    OpenAIChatBackend,
)
from workflows.branch_critic_patch import (
    apply_branch_critic_patch_to_workflow_spec,
    build_branch_plan_summary_from_manifest,
    load_branch_plan_manifest,
    load_workflow_patch_from_branch_plan,
    load_attached_branch_critic_patch,
)
from workflows.policy_guidance import get_default_policy_guidance_spec
from workflows.reward_workflow import RewardWorkflow
from workflows.train_launcher import EPyMARLTrainLauncher


def _build_default_train_spec() -> dict:
    return {
        "config": "qmix",
        "seed": 1,
        "env_config": "gymma",
        "env_args": {
            "key": "lbforaging:Foraging-8x8-2p-1f-v3",
            "time_limit": 50,
        },
        "use_dense_reward": True,
        "overrides": {
            "t_max": 2000,
            "test_interval": 500,
            "runner_log_interval": 500,
            "learner_log_interval": 500,
            "use_cuda": False,
        },
        "checkpointing": {
            "enabled": False,
            "save_model_interval": 50000,
            "checkpoint_path": "",
            "load_step": 0,
            "local_results_path": "results",
        },
        "apply_dense_reward_in_eval": False,
        "log_dense_reward_details": True,
    }


def _build_default_llm_workflow_config() -> dict:
    return {
        "enable_llm_workflow": True,
        "enable_short_branch_validation": True,
        "num_candidates": 3,
        "branch_train_budget": {
            "t_max": 120000,
            "test_interval": 20000,
            "runner_log_interval": 20000,
            "learner_log_interval": 20000,
        },
        "anchor_save_interval": 50000,
        "promote_policy": "critic_diagnosis_first",
        "generator_model_name": DEFAULT_MODEL,
        "critic_model_name": DEFAULT_MODEL,
        "llm_backend": "mock",
        "temperature": 0.2,
        "max_tokens": 4000,
        "mock_llm_mode": True,
        "reward_spec_path": None,
        "active_spec_update_mode": "minimal_patch",
        "alpha_policy_config": {"type": "constant", "value": 1.0},
        "workflow_root_dir": str(ROOT / "results" / "llm_reward_workflows"),
        "save_prompts": True,
        "save_branch_metrics": True,
        "save_spec_history": True,
    }


def build_default_pbrs_workflow_spec(max_rounds: int) -> dict:
    return {
        "reward_paradigm": "pbrs",
        "task_description": (
            "Optimize potential-based reward shaping for multi-agent reinforcement learning "
            "in Level-Based Foraging while preserving the original sparse task objective."
        ),
        "environment_description": (
            "EPyMARL training on LBF through the gymma wrapper with dynamically loaded "
            "reward modules. This workflow is dedicated to PBRS coefficient selection."
        ),
        "reward_constraints": [
            "Do not modify the original sparse reward semantics.",
            "Do not implement alpha mixing inside the reward specification.",
            "Keep heuristic reward terms fixed unless the workflow explicitly relaxes that rule.",
            "Return a structured reward_spec that fits the PBRS-oriented scaffold.",
        ],
        "alpha_policy_settings": {
            "allow_llm_updates": False,
            "fixed_policy": {
                "type": "constant",
                "value": 1.0,
            },
        },
        "llm_workflow": _build_default_llm_workflow_config(),
        "base_policy_guidance_spec": get_default_policy_guidance_spec(),
        "max_rounds": max_rounds,
        "reward_spec_change_budget": {
            "max_weight_updates": 2,
            "max_trigger_variant_updates": 1,
            "max_enable_changes": 1,
            "max_pbrs_updates": 1,
        },
        "pbrs_tuning": {
            "field_schedule": ["beta", "wc", "wp"][:max_rounds],
            "design_update_policy": {
                "allow_variant_updates": False,
                "allow_gate_updates": False,
                "allow_closeness_updates": False,
                "allowed_structural_pbrs_fields": [],
                "per_round_policy": {},
                "per_stage_policy": {
                    "early_exploration": {
                        "allow_variant_updates": False,
                        "allow_gate_updates": False,
                        "allow_closeness_updates": False,
                        "allowed_structural_pbrs_fields": [],
                    },
                    "mid_progress": {
                        "allow_variant_updates": True,
                        "allow_gate_updates": True,
                        "allow_closeness_updates": False,
                        "allowed_structural_pbrs_fields": ["gate_radius"],
                    },
                    "late_plateau": {
                        "allow_variant_updates": True,
                        "allow_gate_updates": True,
                        "allow_closeness_updates": True,
                        "allowed_structural_pbrs_fields": [
                            "gate_radius",
                            "gate_mode",
                            "closeness_mode",
                        ],
                    },
                },
                "per_checkpoint_policy": {},
            },
            "per_stage_active_fields": {
                "early_exploration": "beta",
                "mid_progress": "wc",
                "late_plateau": "wp",
            },
            "per_checkpoint_active_fields": {},
            "followup_field_priority_mode": "diagnosis_first",
            "freeze_reward_terms": True,
            "max_parallel_candidates": 6,
            "candidate_values": {
                "beta": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "wc": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "wp": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            },
            "structural_candidate_values": {
                "variant": ["original", "semi_strict_gate"],
                "gate_radius": [1, 2, 3],
                "gate_mode": ["topk_level_sum", "feasible_assignment"],
                "closeness_mode": ["topk_mean", "topk_nearest"],
            },
            "per_stage_candidate_values": {
                "early_exploration": {
                    "beta": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                },
                "mid_progress": {
                    "wc": [0.3, 0.5, 0.7, 0.9],
                },
                "late_plateau": {
                    "wp": [0.1, 0.3, 0.5],
                },
            },
            "per_checkpoint_candidate_values": {},
            "base_pbrs": {
                "enabled": True,
                "variant": "original",
                "beta": 0.3,
                "gamma": 0.99,
                "wc": 0.6,
                "wp": 0.4,
            },
        },
        "pbrs_validation": {
            "enabled": True,
            "shortlist_size": 2,
            "metric_priority": [
                "best_test_sparse_return_mean",
                "last_test_sparse_return_mean",
                "best_test_return_mean",
                "last_test_return_mean",
                "last_sparse_return_mean",
                "last_return_mean",
            ],
            "short_run_overrides": {
                "t_max": 120000,
                "test_interval": 20000,
                "runner_log_interval": 20000,
                "learner_log_interval": 20000,
            },
            "per_stage_overrides": {
                "early_exploration": {
                    "shortlist_size": 3,
                },
                "late_plateau": {
                    "shortlist_size": 2,
                },
            },
        },
        "pbrs_checkpoint_plan": {
            "selected_checkpoints": [],
            "round_to_checkpoint": {},
        },
        "train": _build_default_train_spec(),
    }


def build_default_heuristic_workflow_spec(max_rounds: int) -> dict:
    return {
        "reward_paradigm": "heuristic",
        "task_description": (
            "Optimize heuristic dense reward terms for multi-agent reinforcement learning "
            "in Level-Based Foraging while preserving the original sparse task objective."
        ),
        "environment_description": (
            "EPyMARL training on LBF through the gymma wrapper with dynamically loaded "
            "reward modules. This workflow is dedicated to term-based heuristic shaping."
        ),
        "reward_constraints": [
            "Do not modify the original sparse reward semantics.",
            "Do not implement alpha mixing inside the reward specification.",
            "Focus on supported reward terms, trigger variants, and term weights.",
            "Do not treat PBRS coefficient sweep as part of this workflow unless explicitly added later.",
        ],
        "alpha_policy_settings": {
            "allow_llm_updates": True,
        },
        "llm_workflow": _build_default_llm_workflow_config(),
        "base_policy_guidance_spec": get_default_policy_guidance_spec(),
        "max_rounds": max_rounds,
        "reward_spec_change_budget": {
            "max_weight_updates": 2,
            "max_trigger_variant_updates": 1,
            "max_enable_changes": 1,
            "max_pbrs_updates": 0,
        },
        "train": _build_default_train_spec(),
    }


def build_default_workflow_spec(max_rounds: int, reward_paradigm: str = "pbrs") -> dict:
    if reward_paradigm == "heuristic":
        return build_default_heuristic_workflow_spec(max_rounds)
    return build_default_pbrs_workflow_spec(max_rounds)


def infer_reward_paradigm_from_workflow_spec(workflow_spec: dict) -> str:
    reward_paradigm = workflow_spec.get("reward_paradigm")
    if isinstance(reward_paradigm, str):
        return reward_paradigm
    if isinstance(workflow_spec.get("pbrs_tuning"), dict):
        return "pbrs"
    return "heuristic"


def load_existing_workflow_spec(
    workflow_id: str,
    *,
    storage_root: str | None = None,
) -> dict:
    results_root = Path(storage_root) if storage_root is not None else ROOT / "results" / "llm_reward_workflows"
    manifest_path = results_root / workflow_id / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    workflow_spec = manifest.get("workflow_spec")
    if not isinstance(workflow_spec, dict):
        raise ValueError(
            f"manifest for workflow {workflow_id} does not contain a valid workflow_spec"
        )
    loaded = deepcopy(workflow_spec)
    loaded["reward_paradigm"] = infer_reward_paradigm_from_workflow_spec(loaded)
    loaded["workflow_id"] = workflow_id
    return loaded


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument(
        "--start-round",
        type=int,
        default=1,
        help="Start or resume the workflow from this round index.",
    )
    parser.add_argument(
        "--reward-paradigm",
        choices=["pbrs", "heuristic"],
        default=None,
        help="Select PBRS-specific or heuristic-rule workflow mode.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch EPyMARL training.",
    )
    parser.add_argument(
        "--storage-root",
        default=None,
        help="Optional override for workflow artifact root directory.",
    )
    parser.add_argument(
        "--use-real-llm",
        action="store_true",
        help="Use the real OpenAI-compatible LLM backend instead of stubs.",
    )
    parser.add_argument(
        "--api-key-env",
        default="IUSEAPI_API_KEY",
        help="Environment variable containing the OpenAI-compatible API key.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the OpenAI-compatible API service.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name used for both Generator and Critic.",
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help="Optional model override used only for the Generator.",
    )
    parser.add_argument(
        "--critic-model",
        default=None,
        help="Optional model override used only for the Critic.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for real LLM calls.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for each real LLM request.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for retryable real LLM failures.",
    )
    parser.add_argument(
        "--llm-retry-backoff",
        type=float,
        default=5.0,
        help="Base backoff seconds for retryable real LLM failures.",
    )
    parser.add_argument(
        "--critic-fallback-mode",
        choices=["none", "stub"],
        default="none",
        help="Optional fallback used when the real Critic backend times out or disconnects.",
    )
    parser.add_argument(
        "--generator-fallback-mode",
        choices=["none", "stub"],
        default="none",
        help="Optional fallback used when the real Generator backend times out or disconnects.",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=None,
        help="Optional override for training t_max.",
    )
    parser.add_argument(
        "--test-interval",
        type=int,
        default=None,
        help="Optional override for training test_interval.",
    )
    parser.add_argument(
        "--runner-log-interval",
        type=int,
        default=None,
        help="Optional override for runner_log_interval.",
    )
    parser.add_argument(
        "--learner-log-interval",
        type=int,
        default=None,
        help="Optional override for learner_log_interval.",
    )
    parser.add_argument(
        "--env-key",
        default=None,
        help="Optional override for train.env_args.key.",
    )
    parser.add_argument(
        "--train-config",
        default=None,
        help="Optional override for train.config, for example qmix or maa2c.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for train.seed.",
    )
    parser.add_argument(
        "--experiment-family",
        default=None,
        help="Optional experiment family label such as sparse_baseline, pbrs_fixed, single_llm, dual_llm.",
    )
    parser.add_argument(
        "--budget-profile",
        default=None,
        help="Optional budget profile label such as sanity or formal.",
    )
    parser.add_argument(
        "--experiment-tag",
        default=None,
        help="Optional free-form experiment tag stored in workflow metadata.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Optional override for train.env_args.time_limit.",
    )
    parser.add_argument(
        "--use-existing-workflow-spec",
        action="store_true",
        help="Load workflow_spec from the existing workflow manifest instead of rebuilding defaults.",
    )
    parser.add_argument(
        "--apply-attached-branch-critic-patch",
        action="store_true",
        help="Apply branch_critic_patch.json or manifest.branch_critic_patch onto the loaded workflow_spec.",
    )
    parser.add_argument(
        "--branch-plan-id",
        default=None,
        help="Optional PBRS branch plan id whose workflow_patch.json should be applied before running.",
    )
    parser.add_argument(
        "--followup-field-priority-mode",
        choices=["diagnosis_first", "stage_locked"],
        default=None,
        help="Optional override for how follow-up rounds choose the active PBRS field when stage-aware branch guidance and prior diagnosis disagree.",
    )
    parser.add_argument(
        "--branch-results-root",
        default=str(ROOT / "results" / "pbrs_branch_plans"),
        help="Root directory containing PBRS branch plan artifacts.",
    )
    return parser


def apply_workflow_arg_overrides(
    workflow_spec: dict,
    args: argparse.Namespace,
) -> dict:
    workflow_spec = deepcopy(workflow_spec)
    workflow_spec["workflow_id"] = args.workflow_id
    if args.reward_paradigm is not None:
        workflow_spec["reward_paradigm"] = args.reward_paradigm
    if args.max_rounds is not None:
        workflow_spec["max_rounds"] = args.max_rounds

    train_overrides = workflow_spec["train"]["overrides"]
    env_args = workflow_spec["train"]["env_args"]
    if args.train_config is not None:
        workflow_spec["train"]["config"] = args.train_config
    if args.seed is not None:
        workflow_spec["train"]["seed"] = args.seed
    if args.t_max is not None:
        train_overrides["t_max"] = args.t_max
    if args.test_interval is not None:
        train_overrides["test_interval"] = args.test_interval
    if args.runner_log_interval is not None:
        train_overrides["runner_log_interval"] = args.runner_log_interval
    if args.learner_log_interval is not None:
        train_overrides["learner_log_interval"] = args.learner_log_interval
    if args.env_key is not None:
        env_args["key"] = args.env_key
    if args.time_limit is not None:
        env_args["time_limit"] = args.time_limit
    experiment_metadata = deepcopy(workflow_spec.get("experiment_metadata") or {})
    if args.experiment_family is not None:
        experiment_metadata["experiment_family"] = args.experiment_family
    if args.budget_profile is not None:
        experiment_metadata["budget_profile"] = args.budget_profile
    if args.seed is not None:
        experiment_metadata["seed"] = args.seed
    if args.experiment_tag is not None:
        experiment_metadata["experiment_tag"] = args.experiment_tag
    if experiment_metadata:
        workflow_spec["experiment_metadata"] = experiment_metadata
    if args.followup_field_priority_mode is not None:
        pbrs_tuning = workflow_spec.setdefault("pbrs_tuning", {})
        pbrs_tuning["followup_field_priority_mode"] = args.followup_field_priority_mode
    return workflow_spec


def build_workflow_spec_from_args(args: argparse.Namespace) -> dict:
    if getattr(args, "use_existing_workflow_spec", False):
        try:
            workflow_spec = load_existing_workflow_spec(
                args.workflow_id,
                storage_root=args.storage_root,
            )
        except FileNotFoundError:
            selected_reward_paradigm = args.reward_paradigm or "pbrs"
            workflow_spec = build_default_workflow_spec(
                args.max_rounds or 1,
                reward_paradigm=selected_reward_paradigm,
            )
    else:
        selected_reward_paradigm = args.reward_paradigm or "pbrs"
        workflow_spec = build_default_workflow_spec(
            args.max_rounds or 1,
            reward_paradigm=selected_reward_paradigm,
        )
    workflow_spec = apply_workflow_arg_overrides(workflow_spec, args)
    if getattr(args, "apply_attached_branch_critic_patch", False):
        storage_root = (
            args.storage_root
            if args.storage_root is not None
            else ROOT / "results" / "llm_reward_workflows"
        )
        attached_patch = load_attached_branch_critic_patch(
            workflow_id=args.workflow_id,
            storage_root=storage_root,
        )
        if isinstance(attached_patch, dict):
            updated = apply_branch_critic_patch_to_workflow_spec(
                workflow_spec=workflow_spec,
                workflow_patch=attached_patch,
            )
            if isinstance(updated, dict):
                workflow_spec = updated
                workflow_spec["branch_critic_patch"] = deepcopy(attached_patch)
    if getattr(args, "branch_plan_id", None):
        branch_plan_manifest = load_branch_plan_manifest(
            branch_plan_id=args.branch_plan_id,
            branch_results_root=args.branch_results_root,
        )
        if isinstance(branch_plan_manifest, dict):
            workflow_spec["branch_plan_summary"] = build_branch_plan_summary_from_manifest(
                branch_plan_manifest,
                branch_plan_dir=str(Path(args.branch_results_root) / args.branch_plan_id),
            )
            branch_plan_workflow_spec = branch_plan_manifest.get("workflow_spec") or {}
            branch_checkpoint_plan = (
                branch_plan_workflow_spec.get("pbrs_checkpoint_plan")
                if isinstance(branch_plan_workflow_spec, dict)
                else None
            )
            if isinstance(branch_checkpoint_plan, dict):
                workflow_spec["pbrs_checkpoint_plan"] = deepcopy(branch_checkpoint_plan)
        branch_plan_patch = load_workflow_patch_from_branch_plan(
            branch_plan_id=args.branch_plan_id,
            branch_results_root=args.branch_results_root,
        )
        if isinstance(branch_plan_patch, dict):
            updated = apply_branch_critic_patch_to_workflow_spec(
                workflow_spec=workflow_spec,
                workflow_patch=branch_plan_patch,
            )
            if isinstance(updated, dict):
                workflow_spec = updated
                workflow_spec["branch_critic_patch"] = deepcopy(branch_plan_patch)
                workflow_spec["branch_critic_patch_source"] = {
                    "type": "branch_plan",
                    "branch_plan_id": args.branch_plan_id,
                }
    return workflow_spec


def build_workflow_from_args(args: argparse.Namespace) -> RewardWorkflow:
    workflow_spec = build_workflow_spec_from_args(args)
    if args.use_real_llm:
        generator_model = args.generator_model or args.model
        critic_model = args.critic_model or args.model
        llm_backend = OpenAIChatBackend(
            api_key=os.environ.get(args.api_key_env),
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            model=generator_model,
            temperature=args.temperature,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            retry_backoff_seconds=args.llm_retry_backoff,
        )
        generator_backend = llm_backend
        critic_backend = OpenAIChatBackend(
            api_key=os.environ.get(args.api_key_env),
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            model=critic_model,
            temperature=args.temperature,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            retry_backoff_seconds=args.llm_retry_backoff,
        )
        if getattr(args, "generator_fallback_mode", "none") == "stub":
            generator_backend = FallbackLLMBackend(
                primary_backend=generator_backend,
                fallback_backend=StubGeneratorBackend(),
                fallback_label="stub_generator_on_error",
            )
        if getattr(args, "critic_fallback_mode", "none") == "stub":
            critic_backend = FallbackLLMBackend(
                primary_backend=critic_backend,
                fallback_backend=StubCriticBackend(),
                fallback_label="stub_critic_on_error",
            )
    else:
        generator_backend = StubGeneratorBackend()
        critic_backend = StubCriticBackend()

    generator_client = TemplateBasedGeneratorClient(
        llm_backend=generator_backend,
    )
    critic_client = TemplateBasedCriticClient(
        llm_backend=critic_backend,
    )
    train_launcher = EPyMARLTrainLauncher(
        repo_root=ROOT,
        python_executable=args.python_executable,
    )

    workflow = RewardWorkflow(
        workflow_id=args.workflow_id,
        workflow_spec=workflow_spec,
        generator_client=generator_client,
        critic_client=critic_client,
        train_launcher=train_launcher,
        storage_root=args.storage_root,
    )
    return workflow


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    workflow = build_workflow_from_args(args)
    result = workflow.run(start_round=args.start_round)

    print("reward workflow run completed")
    print(f"workflow_id={result.workflow_id}")
    print(f"start_round={args.start_round}")
    print(f"completed_rounds={result.completed_rounds}")
    print(f"manifest_status={result.manifest.get('status')}")
    print(f"last_round_status={result.last_round_status.get('status')}")


if __name__ == "__main__":
    main()
