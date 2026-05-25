from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainingLogAnalyzer:
    """
    Build a compact, structured summary of training artifacts for Critic input.

    This analyzer intentionally keeps the output close to raw experimental data:
    no heuristic observations, no recommendations, and no inferred conclusions.
    """

    DEFAULT_METRICS_OF_INTEREST = [
        "sparse_return_mean",
        "test_sparse_return_mean",
        "dense_return_mean",
        "test_dense_return_mean",
        "mixed_return_mean",
        "test_mixed_return_mean",
        "return_mean",
        "test_return_mean",
        "reward_alpha_mean",
        "test_reward_alpha_mean",
        "dense_reward_applied_mean",
        "test_dense_reward_applied_mean",
        "ep_length_mean",
        "test_ep_length_mean",
        "loss",
        "grad_norm",
        "q_taken_mean",
        "target_mean",
        "td_error_abs",
    ]

    def summarize(
        self,
        *,
        train_config: Dict[str, Any],
        run_reference: Dict[str, Any],
    ) -> Dict[str, Any]:
        metrics_path = Path(run_reference["metrics_json"])
        metrics = self._load_metrics(metrics_path)

        summary = {
            "run_metadata": self._build_run_metadata(train_config, run_reference),
            "alpha_policy_config": train_config.get("alpha_policy_config"),
            "policy_guidance_config": train_config.get("policy_guidance_config"),
            "metric_summary": {},
            "reward_breakdown_summary": {},
        }

        metrics_of_interest = self._resolve_metrics_of_interest(metrics)
        for metric_name in metrics_of_interest:
            metric_payload = metrics.get(metric_name)
            if not metric_payload:
                continue
            metric_summary = self._summarize_metric(metric_payload)
            if metric_summary is not None:
                summary["metric_summary"][metric_name] = metric_summary
                reward_term_name = self._extract_reward_term_name(metric_name)
                if reward_term_name is not None:
                    summary["reward_breakdown_summary"][
                        reward_term_name
                    ] = metric_summary

        return summary

    def _load_metrics(self, metrics_path: Path) -> Dict[str, Any]:
        if not metrics_path.exists():
            return {}
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_run_metadata(
        self,
        train_config: Dict[str, Any],
        run_reference: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "workflow_id": train_config.get("workflow_id"),
            "workflow_round": train_config.get("workflow_round"),
            "reward_paradigm": train_config.get("reward_paradigm"),
            "seed": train_config.get("seed"),
            "experiment_metadata": train_config.get("experiment_metadata"),
            "active_pbrs_field": train_config.get("active_pbrs_field"),
            "active_pbrs_field_source": train_config.get("active_pbrs_field_source"),
            "candidate_value": train_config.get("candidate_value"),
            "active_checkpoint_context": train_config.get("active_checkpoint_context"),
            "active_field_carryover_context": train_config.get(
                "active_field_carryover_context"
            ),
            "candidate_selection_context": train_config.get("candidate_selection_context"),
            "policy_guidance_config": train_config.get("policy_guidance_config"),
            "label": train_config.get("label"),
            "algorithm": train_config.get("config"),
            "env_config": train_config.get("env_config"),
            "env_args": train_config.get("env_args", {}),
            "overrides": train_config.get("overrides", {}),
            "use_dense_reward": train_config.get("use_dense_reward"),
            "save_model": train_config.get("save_model"),
            "save_model_interval": train_config.get("save_model_interval"),
            "checkpoint_path": train_config.get("checkpoint_path"),
            "load_step": train_config.get("load_step"),
            "local_results_path": train_config.get("local_results_path"),
            "reward_module_path": train_config.get("reward_module_path"),
            "apply_dense_reward_in_eval": train_config.get(
                "apply_dense_reward_in_eval"
            ),
            "log_dense_reward_details": train_config.get(
                "log_dense_reward_details"
            ),
            "run_dir": run_reference.get("run_dir"),
            "run_id": run_reference.get("run_id"),
            "checkpoint_root_dir": run_reference.get("checkpoint_root_dir"),
            "available_checkpoint_steps": run_reference.get("available_checkpoint_steps"),
            "latest_checkpoint_step": run_reference.get("latest_checkpoint_step"),
        }

    def _resolve_metrics_of_interest(self, metrics: Dict[str, Any]) -> List[str]:
        ordered = []
        seen = set()
        for metric_name in self.DEFAULT_METRICS_OF_INTEREST:
            if metric_name in metrics and metric_name not in seen:
                ordered.append(metric_name)
                seen.add(metric_name)
        for metric_name in sorted(metrics):
            if metric_name.endswith("_mean") and metric_name not in seen:
                ordered.append(metric_name)
                seen.add(metric_name)
        return ordered

    def _summarize_metric(self, metric_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        values = metric_payload.get("values", [])
        steps = metric_payload.get("steps", [])
        if not values:
            return None

        numeric_values = [float(value) for value in values]
        numeric_steps = [int(step) for step in steps] if steps else []

        point_count = len(numeric_values)
        indices = self._sample_indices(point_count)
        sampled_points = []
        for idx in indices:
            sampled_points.append(
                {
                    "index": idx,
                    "step": numeric_steps[idx] if idx < len(numeric_steps) else None,
                    "value": numeric_values[idx],
                }
            )

        summary = {
            "num_points": point_count,
            "first_value": numeric_values[0],
            "last_value": numeric_values[-1],
            "best_value": max(numeric_values),
            "worst_value": min(numeric_values),
            "first_step": numeric_steps[0] if numeric_steps else None,
            "last_step": numeric_steps[-1] if numeric_steps else None,
            "sampled_points": sampled_points,
        }

        return summary

    def _sample_indices(self, point_count: int) -> List[int]:
        if point_count <= 3:
            return list(range(point_count))
        candidate_indices = [0, point_count // 2, point_count - 1]
        indices: List[int] = []
        for idx in candidate_indices:
            if idx not in indices:
                indices.append(idx)
        return indices

    def _extract_reward_term_name(self, metric_name: str) -> Optional[str]:
        prefix = "reward_breakdown__"
        suffix = "_mean"
        if not metric_name.startswith(prefix):
            return None
        if not metric_name.endswith(suffix):
            return None
        return metric_name[len(prefix) : -len(suffix)]
