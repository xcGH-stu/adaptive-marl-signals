from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]


def _budget_defaults(budget_profile: str) -> Dict[str, Any]:
    if budget_profile == "formal":
        return {
            "env_key": "lbforaging:Foraging-15x15-4p-3f-v3",
            "train_config": "qmix",
            "time_limit": 50,
            "t_max": 2050000,
            "test_interval": 25000,
            "runner_log_interval": 25000,
            "learner_log_interval": 25000,
            "save_model_interval": 25000,
        }
    if budget_profile == "pilot":
        return {
            "env_key": "lbforaging:Foraging-8x8-2p-1f-v3",
            "train_config": "qmix",
            "time_limit": 50,
            "t_max": 120000,
            "test_interval": 20000,
            "runner_log_interval": 20000,
            "learner_log_interval": 20000,
            "save_model_interval": 20000,
        }
    return {
        "env_key": "lbforaging:Foraging-8x8-2p-1f-v3",
        "train_config": "qmix",
        "time_limit": 50,
        "t_max": 60000,
        "test_interval": 10000,
        "runner_log_interval": 10000,
        "learner_log_interval": 10000,
        "save_model_interval": 10000,
    }


def build_experiment_profile(
    *,
    experiment_type: str,
    budget_profile: str,
    seed: int,
) -> Dict[str, Any]:
    budget = _budget_defaults(budget_profile)
    common = {
        "experiment_type": experiment_type,
        "budget_profile": budget_profile,
        "seed": seed,
        "algorithm": budget["train_config"],
        "env_key": budget["env_key"],
        "time_limit": budget["time_limit"],
        "t_max": budget["t_max"],
        "test_interval": budget["test_interval"],
        "runner_log_interval": budget["runner_log_interval"],
        "learner_log_interval": budget["learner_log_interval"],
        "save_model_interval": budget["save_model_interval"],
        "python_executable": "/home/epymarl/miniconda3/envs/epymarl_new/bin/python",
        "results_root": str(ROOT / "results"),
    }

    if experiment_type == "sparse_baseline":
        return {
            **common,
            "script": "src/scripts/run_sparse_baseline_with_checkpoints.py",
            "cli_template": [
                "PYTHONPATH=src:/home/epymarl/code/lbforaging_local:$PYTHONPATH",
                "python",
                "src/scripts/run_sparse_baseline_with_checkpoints.py",
                "--baseline-id",
                f"{budget_profile}_sparse_seed{seed}",
                "--reward-paradigm",
                "pbrs",
                "--train-config",
                budget["train_config"],
                "--env-key",
                budget["env_key"],
                "--seed",
                str(seed),
                "--t-max",
                str(budget["t_max"]),
                "--test-interval",
                str(budget["test_interval"]),
                "--runner-log-interval",
                str(budget["runner_log_interval"]),
                "--learner-log-interval",
                str(budget["learner_log_interval"]),
                "--save-model-interval",
                str(budget["save_model_interval"]),
                "--enable-checkpointing",
                "--experiment-family",
                "sparse_baseline",
                "--budget-profile",
                budget_profile,
                "--format",
                "text",
            ],
        }

    if experiment_type == "pbrs_fixed":
        return {
            **common,
            "script": "src/scripts/run_fixed_reward_baseline.py",
            "cli_template": [
                "PYTHONPATH=src:/home/epymarl/code/lbforaging_local:$PYTHONPATH",
                "python",
                "src/scripts/run_fixed_reward_baseline.py",
                "--run-id",
                f"{budget_profile}_pbrs_fixed_seed{seed}",
                "--reward-paradigm",
                "pbrs",
                "--train-config",
                budget["train_config"],
                "--env-key",
                budget["env_key"],
                "--seed",
                str(seed),
                "--t-max",
                str(budget["t_max"]),
                "--test-interval",
                str(budget["test_interval"]),
                "--runner-log-interval",
                str(budget["runner_log_interval"]),
                "--learner-log-interval",
                str(budget["learner_log_interval"]),
                "--experiment-family",
                "pbrs_fixed",
                "--budget-profile",
                budget_profile,
                "--format",
                "text",
            ],
        }

    if experiment_type == "single_llm":
        return {
            **common,
            "script": "src/scripts/run_reward_workflow.py",
            "notes": "Practical single-LLM approximation: same model for Generator/Critic, no short-branch validation, one candidate per round.",
            "workflow_overrides": {
                "llm_workflow": {
                    "enable_short_branch_validation": False,
                    "num_candidates": 1,
                }
            },
            "cli_template": [
                "PYTHONPATH=src:/home/epymarl/code/lbforaging_local:$PYTHONPATH",
                "python",
                "src/scripts/run_reward_workflow.py",
                "--workflow-id",
                f"{budget_profile}_single_llm_seed{seed}",
                "--reward-paradigm",
                "pbrs",
                "--max-rounds",
                "3",
                "--train-config",
                budget["train_config"],
                "--env-key",
                budget["env_key"],
                "--seed",
                str(seed),
                "--t-max",
                str(budget["t_max"]),
                "--test-interval",
                str(budget["test_interval"]),
                "--runner-log-interval",
                str(budget["runner_log_interval"]),
                "--learner-log-interval",
                str(budget["learner_log_interval"]),
                "--use-real-llm",
                "--experiment-family",
                "single_llm",
                "--budget-profile",
                budget_profile,
            ],
        }

    if experiment_type == "dual_llm":
        return {
            **common,
            "script": "src/scripts/run_reward_workflow.py",
            "workflow_overrides": {
                "llm_workflow": {
                    "enable_short_branch_validation": True,
                    "num_candidates": 3,
                }
            },
            "cli_template": [
                "PYTHONPATH=src:/home/epymarl/code/lbforaging_local:$PYTHONPATH",
                "python",
                "src/scripts/run_reward_workflow.py",
                "--workflow-id",
                f"{budget_profile}_dual_llm_seed{seed}",
                "--reward-paradigm",
                "pbrs",
                "--max-rounds",
                "3",
                "--train-config",
                budget["train_config"],
                "--env-key",
                budget["env_key"],
                "--seed",
                str(seed),
                "--t-max",
                str(budget["t_max"]),
                "--test-interval",
                str(budget["test_interval"]),
                "--runner-log-interval",
                str(budget["runner_log_interval"]),
                "--learner-log-interval",
                str(budget["learner_log_interval"]),
                "--use-real-llm",
                "--generator-fallback-mode",
                "stub",
                "--critic-fallback-mode",
                "stub",
                "--experiment-family",
                "dual_llm",
                "--budget-profile",
                budget_profile,
            ],
        }

    raise ValueError(f"unsupported experiment_type: {experiment_type}")


def export_experiment_templates(*, output_dir: str | Path, seed: int = 1) -> Dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    for budget_profile in ["sanity", "pilot", "formal"]:
        budget_dir = output_root / budget_profile
        budget_dir.mkdir(parents=True, exist_ok=True)
        for experiment_type in [
            "sparse_baseline",
            "pbrs_fixed",
            "single_llm",
            "dual_llm",
        ]:
            payload = build_experiment_profile(
                experiment_type=experiment_type,
                budget_profile=budget_profile,
                seed=seed,
            )
            output_path = budget_dir / f"{experiment_type}.json"
            output_path.write_text(
                __import__("json").dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            written[f"{budget_profile}/{experiment_type}"] = str(output_path)
    return written
