from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from workflows.policy_guidance import build_fallback_integrated_guidance_card
from workflows.production_stage1b import (
    _build_final_selection_fallback,
    _deterministic_round1_analysis,
)
from workflows.train_launcher import EPyMARLTrainLauncher, TrainLaunchResult


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _points_to_series(sampled_points: List[Dict[str, Any]], *, fallback_step: int = 0) -> Dict[str, Any]:
    steps: List[int] = []
    values: List[float] = []
    for item in sampled_points:
        try:
            steps.append(int(item["step"]))
            values.append(float(item["value"]))
        except Exception:
            continue
    if not steps:
        steps = [int(fallback_step), int(fallback_step) + 50000, int(fallback_step) + 100000]
        values = [0.0, 0.0, 0.0]
    dedup: Dict[int, float] = {}
    for step, value in zip(steps, values):
        dedup[int(step)] = float(value)
    ordered_steps = sorted(dedup.keys())
    return {"steps": ordered_steps, "values": [dedup[step] for step in ordered_steps]}


def _curve_payload(summary: Dict[str, Any], *, fallback_step: int = 0) -> Dict[str, Any]:
    sampled = list(summary.get("sampled_points") or [])
    if not sampled and summary.get("first_step") is not None:
        sampled.append({"step": summary.get("first_step"), "value": summary.get("first_value", 0.0)})
    if not sampled and summary.get("last_step") is not None:
        sampled.append({"step": summary.get("last_step"), "value": summary.get("last_value", 0.0)})
    if (
        summary.get("last_step") is not None
        and summary.get("last_value") is not None
        and not any(int(item.get("step", -1)) == int(summary["last_step"]) for item in sampled)
    ):
        sampled.append({"step": summary["last_step"], "value": summary["last_value"]})
    return _points_to_series(sampled, fallback_step=fallback_step)


def _metric_curves_from_summary(metrics_summary: Dict[str, Any], *, fallback_step: int = 0) -> Dict[str, Any]:
    curves = deepcopy(metrics_summary.get("metric_curves") or {})
    if curves:
        payload: Dict[str, Any] = {}
        for metric_name, metric_payload in curves.items():
            payload[str(metric_name)] = _curve_payload(metric_payload or {}, fallback_step=fallback_step)
        return payload
    metric_summary = deepcopy(metrics_summary.get("metric_summary") or {})
    payload = {}
    for metric_name, metric_payload in metric_summary.items():
        payload[str(metric_name)] = _curve_payload(metric_payload or {}, fallback_step=fallback_step)
    return payload


def _write_run_fixture(
    *,
    run_dir: Path,
    run_id: int,
    workflow_id: str,
    label: str,
    train_config: Dict[str, Any],
    metrics: Dict[str, Any],
    checkpoint_steps: List[int],
    status: str = "COMPLETED",
    run_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    available_steps = sorted({int(step) for step in checkpoint_steps if step is not None})
    for step in available_steps:
        (checkpoint_root / str(step)).mkdir(parents=True, exist_ok=True)
    latest_step = max(available_steps) if available_steps else None
    config_payload = deepcopy(train_config)
    config_payload.setdefault("workflow_id", workflow_id)
    config_payload.setdefault("label", label)
    config_payload.setdefault("workflow_round", 0)
    config_payload.setdefault("reward_module_path", "")
    config_payload.setdefault("local_results_path", "results")
    metrics_payload = deepcopy(metrics)
    unique_token = f"fixture_{workflow_id}_{run_id}"
    info_payload = {
        "model_root_path": str(checkpoint_root),
        "saved_model_steps": available_steps,
        "latest_model_path": str(checkpoint_root / str(latest_step)) if latest_step is not None else None,
        "result": {
            "workflow_id": workflow_id,
            "config_name": config_payload.get("config"),
            "env_key": ((config_payload.get("env_args") or {}).get("key")),
            "args": {"seed": config_payload.get("seed", 1)},
            "unique_token": unique_token,
            "model_root_path": str(checkpoint_root),
            "saved_model_steps": available_steps,
            "latest_model_path": str(checkpoint_root / str(latest_step)) if latest_step is not None else None,
            "status": status,
        }
    }
    if isinstance(run_metadata, dict):
        info_payload["result"].update(deepcopy(run_metadata))
        info_payload["result"]["model_root_path"] = str(checkpoint_root)
        info_payload["result"]["saved_model_steps"] = available_steps
        info_payload["result"]["latest_model_path"] = (
            str(checkpoint_root / str(latest_step)) if latest_step is not None else None
        )
    run_payload = {
        "_id": int(run_id),
        "status": status,
        "experiment": {"name": workflow_id},
    }
    (run_dir / "cout.txt").write_text("fixture replay run\n", encoding="utf-8")
    _save_json(run_dir / "config.json", config_payload)
    _save_json(run_dir / "metrics.json", metrics_payload)
    _save_json(run_dir / "info.json", info_payload)
    _save_json(run_dir / "run.json", run_payload)
    return {
        "sacred_base_dir": str(run_dir.parent),
        "run_dir": str(run_dir),
        "run_id": int(run_id),
        "config_json": str(run_dir / "config.json"),
        "run_json": str(run_dir / "run.json"),
        "info_json": str(run_dir / "info.json"),
        "metrics_json": str(run_dir / "metrics.json"),
        "cout_txt": str(run_dir / "cout.txt"),
        "status": status,
        "checkpoint_root_dir": str(checkpoint_root),
        "available_checkpoint_steps": available_steps,
        "latest_checkpoint_step": latest_step,
        "latest_model_path": str(checkpoint_root / str(latest_step)) if latest_step is not None else None,
    }


class ReplayResultProvider:
    def __init__(
        self,
        *,
        source_root: str | Path,
        source_workflow_id: str,
        replay_root: str | Path,
        replay_workflow_id: str,
    ) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self.source_workflow_id = str(source_workflow_id)
        self.replay_root = Path(replay_root).expanduser().resolve()
        self.replay_workflow_id = str(replay_workflow_id)
        self.end_to_end_dir = self.source_root / "end_to_end_workflows" / self.source_workflow_id
        self.adaptive_dir = (
            self.source_root
            / "adaptive_checkpoint_replacement_workflows"
            / f"{self.source_workflow_id}_stage234"
        )
        self.fixture_run_root = self.replay_root / self.replay_workflow_id / "_fixture_runs"
        self._run_counter = 9000
        self._materialized_runs: Dict[str, Dict[str, Any]] = {}
        self.missing_fixtures: List[str] = []
        self.stage1b_artifact = _load_json(self.end_to_end_dir / "stage_1b_dense_reference.json")
        self.stage2_artifact = _load_json(self.end_to_end_dir / "stage_2_selection.json")
        self.workflow_manifest = _load_json(self.end_to_end_dir / "workflow_manifest.json")
        self.stage1_behavior_summary_path = self.end_to_end_dir / "stage1_sparse_policy_behavior_summary.json"
        self.stage1_behavior_summary = (
            _load_json(self.stage1_behavior_summary_path)
            if self.stage1_behavior_summary_path.exists()
            else {}
        )
        self.adaptive_manifest = _load_json(self.adaptive_dir / "adaptive_checkpoint_replacement_manifest.json")
        self.stage3_decision = deepcopy((self.adaptive_manifest.get("decisions") or [])[0] or {})
        self.stage3_branch_results: Dict[str, Dict[str, Any]] = {}
        for result_path in self.adaptive_dir.glob("decision_01_post_initialization_transition/branches/*/branch_result.json"):
            payload = _load_json(result_path)
            candidate_id = str(payload.get("candidate_id") or "")
            if candidate_id:
                self.stage3_branch_results[candidate_id] = payload
        stage3_diagnosis_path = (
            self.adaptive_dir
            / "decision_01_post_initialization_transition"
            / "llm_result_diagnosis.json"
        )
        self.stage3_diagnosis = self._load_optional_json(stage3_diagnosis_path)
        self.guidance_cards = {
            "stage1_after_sparse_baseline": self._load_optional_json(
                self.end_to_end_dir
                / "policy_guidance"
                / "stage1_after_sparse_baseline_integrated_guidance_card.json"
            ),
            "stage3_c1_pre_checkpoint": self._load_optional_json(
                self.adaptive_dir
                / "decision_01_post_initialization_transition"
                / "policy_guidance"
                / "stage3_c1_pre_checkpoint_integrated_guidance_card.json"
            ),
            "stage3_c1_after_round1": self._load_optional_json(
                self.adaptive_dir
                / "decision_01_post_initialization_transition"
                / "policy_guidance"
                / "stage3_c1_after_round1_integrated_guidance_card.json"
            ),
        }
        if not self.stage1_behavior_summary:
            self.missing_fixtures.append("stage1_behavior_summary")
        if not self.stage3_branch_results:
            self.missing_fixtures.append("stage3_c1_branch_results")
        if not self.stage3_decision:
            self.missing_fixtures.append("stage3_c1_decision_manifest")
        self.round1_source_candidate_ids = [
            str(item.get("candidate_id") or "")
            for item in list(self.stage1b_artifact.get("round1_candidates") or [])
            if item.get("candidate_id")
        ]
        self.round2_source_candidates = {
            str(item.get("candidate_id") or ""): deepcopy(item)
            for item in list(self.stage1b_artifact.get("round2_candidates") or [])
            if item.get("candidate_id")
        }
        self.round2_source_results = {
            str(item.get("candidate_id") or ""): deepcopy(item)
            for item in list(self.stage1b_artifact.get("round2_results") or [])
            if item.get("candidate_id")
        }
        if len(self.round1_source_candidate_ids) < 3:
            self.missing_fixtures.append("stage1b_round1_candidate_context")
        if len(self.round2_source_results) < 3:
            self.missing_fixtures.append("stage1b_round2_candidate_results")

    def _load_optional_json(self, path: Path) -> Dict[str, Any]:
        return _load_json(path) if path.exists() else {}

    def _next_run_id(self) -> int:
        self._run_counter += 1
        return self._run_counter

    def _run_key(self, category: str, identifier: str) -> str:
        return f"{category}:{identifier}"

    def _materialize_sparse_run(self) -> Dict[str, Any]:
        key = self._run_key("stage1", "sparse_baseline")
        if key in self._materialized_runs:
            return self._materialized_runs[key]
        run_dir = self.fixture_run_root / "stage1_sparse_baseline"
        env_key = str((self.workflow_manifest.get("config_summary") or {}).get("env_key") or "")
        train_config = {
            "config": str((self.workflow_manifest.get("config_summary") or {}).get("train_config") or "qmix"),
            "seed": int((self.workflow_manifest.get("config_summary") or {}).get("seed") or 1),
            "env_args": {"key": env_key, "use_pbrs": False, "eval_use_pbrs": False},
            "workflow_id": f"{self.replay_workflow_id}_stage1_fixture",
            "workflow_round": 0,
            "label": f"fixture_sparse_{self.replay_workflow_id}",
            "reward_module_path": "",
            "local_results_path": "results",
        }
        metrics = {
            "test_sparse_return_mean": {"steps": [50000, 150000, 300000, 500000], "values": [0.0, 0.02, 0.04, 0.05]},
            "sparse_return_mean": {"steps": [50000, 150000, 300000, 500000], "values": [0.0, 0.03, 0.05, 0.06]},
            "return_mean": {"steps": [50000, 150000, 300000, 500000], "values": [-0.1, -0.05, -0.02, 0.0]},
        }
        run_ref = _write_run_fixture(
            run_dir=run_dir,
            run_id=self._next_run_id(),
            workflow_id=train_config["workflow_id"],
            label=train_config["label"],
            train_config=train_config,
            metrics=metrics,
            checkpoint_steps=[500000],
            run_metadata={
                "env_key": env_key,
                "config_name": train_config["config"],
                "args": {"seed": train_config["seed"]},
            },
        )
        self._materialized_runs[key] = run_ref
        self.missing_fixtures.append("stage1_sparse_run_exact_metrics_missing_used_synthetic")
        return run_ref

    def copy_stage1_behavior_summary(self, workflow_dir: Path) -> Path:
        target = workflow_dir / "stage1_sparse_policy_behavior_summary.json"
        payload = deepcopy(self.stage1_behavior_summary)
        payload.setdefault("metadata", {})
        payload["metadata"]["workflow_id"] = self.replay_workflow_id
        payload["workflow_id"] = self.replay_workflow_id
        _save_json(target, payload)
        return target

    def build_stage1_fixture_payload(self, workflow_dir: Path) -> Dict[str, Any]:
        sparse_ref = self._materialize_sparse_run()
        summary_path = self.copy_stage1_behavior_summary(workflow_dir)
        return {
            "status": "skipped_reused",
            "reuse_mode": "fixture_replay",
            "source_workflow_id": self.source_workflow_id,
            "source_seed": int((self.workflow_manifest.get("config_summary") or {}).get("seed") or 1),
            "source_stage": "stage1_sparse_baseline",
            "source_results_root": str(self.source_root),
            "allow_partial_source_workflow": True,
            "require_source_stage1_completed": False,
            "shared_calibration": {
                "source_seed": int((self.workflow_manifest.get("config_summary") or {}).get("seed") or 1),
                "target_seed": int((self.workflow_manifest.get("config_summary") or {}).get("seed") or 1),
                "seed_differs": False,
            },
            "count_as_evaluation_baseline": False,
            "result": {
                "run_reference": {
                    "run_id": sparse_ref["run_id"],
                    "run_dir": sparse_ref["run_dir"],
                    "workflow_id": f"{self.replay_workflow_id}_stage1_fixture",
                }
            },
            "reuse_artifacts": {
                "stage1_behavior_summary_path": str(summary_path),
                "stage1_sparse_metrics_path": str(workflow_dir / "stage_1_sparse_baseline.json"),
            },
        }

    def build_stage2_c1_payload(self) -> Dict[str, Any]:
        payload = deepcopy(self.stage2_artifact)
        selection = payload.setdefault("selection", {})
        selected = list(selection.get("selected_checkpoints") or [])
        selection["selected_checkpoints"] = selected[:1]
        payload["selection_source"] = str(payload.get("selection_source") or "fixture_replay_source")
        return payload

    def stage3_round_candidates(self, round_id: int) -> List[Dict[str, Any]]:
        for round_payload in list(self.stage3_decision.get("rounds") or []):
            if int(round_payload.get("round_id") or 0) == int(round_id):
                return deepcopy(list(round_payload.get("generator_candidates") or []))
        self.missing_fixtures.append(f"stage3_round_{int(round_id)}_generator_candidates")
        return []

    def stage3_result_diagnosis_payload(self) -> Dict[str, Any]:
        parsed = deepcopy(self.stage3_diagnosis.get("parsed") or {})
        if parsed:
            return parsed
        return {
            "decision": "retain_no_change",
            "recommended_candidate_id": "no_change",
            "ranking": [{"candidate_id": "no_change", "rank": 1, "reason": "fixture fallback"}],
            "confidence": "medium",
            "metric_basis": "fixture_fallback",
            "risk_notes": ["source result diagnosis missing; deterministic replay fallback used"],
            "short_reason": "No-change remains the safest winner in the fixture replay.",
        }

    def policy_guidance_card(self, intervention_point: str, reward_diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        card = deepcopy(self.guidance_cards.get(intervention_point) or {})
        if card:
            return card
        self.missing_fixtures.append(f"policy_guidance_card_missing:{intervention_point}")
        return build_fallback_integrated_guidance_card(
            intervention_point=intervention_point,
            failure_reason=f"missing source guidance card for {intervention_point}",
            reward_diagnosis=reward_diagnosis,
        )

    def stage1b_round1_candidates(self) -> Dict[str, Any]:
        candidates = []
        for item in list(self.stage1b_artifact.get("round1_candidates") or [])[:3]:
            if item.get("candidate_id"):
                candidates.append(deepcopy(item))
        if len(candidates) < 3:
            self.missing_fixtures.append("stage1b_round1_candidates_used_synthetic_topup")
            topups = [
                {
                    "candidate_id": "fixture_r1_balanced",
                    "beta": 0.3,
                    "wc": 0.5,
                    "wp": 0.5,
                    "candidate_type": "balanced",
                    "hypothesis": "Synthetic fixture balanced baseline.",
                    "expected_early_effect": "steady early support",
                    "risk": "synthetic topup",
                },
                {
                    "candidate_id": "fixture_r1_reference_like",
                    "beta": 0.3,
                    "wc": 0.6,
                    "wp": 0.4,
                    "candidate_type": "reference_like",
                    "hypothesis": "Synthetic fixture reference-like control.",
                    "expected_early_effect": "stable continuation",
                    "risk": "synthetic topup",
                },
                {
                    "candidate_id": "fixture_r1_progress_shift",
                    "beta": 0.5,
                    "wc": 0.2,
                    "wp": 0.8,
                    "candidate_type": "progress_shift",
                    "hypothesis": "Synthetic fixture more exploratory option.",
                    "expected_early_effect": "broader discovery",
                    "risk": "synthetic topup",
                },
            ]
            while len(candidates) < 3 and topups:
                candidates.append(topups.pop(0))
        return {"goal": "early_budget_selection", "candidates": candidates[:3]}

    def stage1b_round2_candidates(self) -> Dict[str, Any]:
        candidates = [
            deepcopy(self.round2_source_candidates[candidate_id])
            for candidate_id in list(self.round2_source_candidates.keys())[:3]
        ]
        if len(candidates) < 3:
            self.missing_fixtures.append("stage1b_round2_candidates_missing")
        return {"goal": "early_budget_selection", "candidates": candidates[:3]}

    def _materialize_stage1b_round_run(
        self,
        *,
        candidate_id: str,
        round_id: int,
        train_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        key = self._run_key(f"stage1b_round{int(round_id)}", candidate_id)
        if key in self._materialized_runs:
            return self._materialized_runs[key]
        result_record: Optional[Dict[str, Any]] = None
        if int(round_id) == 2:
            result_record = deepcopy(self.round2_source_results.get(candidate_id) or {})
        if not result_record:
            result_record = {
                "candidate_id": candidate_id,
                "run_id": f"{self.replay_workflow_id}_stage1b_round{int(round_id)}_{candidate_id}",
                "run_status": "COMPLETED",
                "endpoint_checkpoint_step": 800000 + (int(round_id) * 100) + (len(candidate_id) % 97),
                "metrics_summary": {
                    "metric_curves": {
                        "test_sparse_return_mean": {
                            "sampled_points": [
                                {"step": 50, "value": 0.0},
                                {"step": 250000, "value": 0.03 + 0.005 * int(round_id)},
                                {"step": 500000, "value": 0.05 + 0.005 * int(round_id)},
                                {"step": 800000, "value": 0.06 + 0.005 * int(round_id)},
                            ]
                        },
                        "sparse_return_mean": {
                            "sampled_points": [
                                {"step": 50, "value": 0.0},
                                {"step": 250000, "value": 0.04 + 0.005 * int(round_id)},
                                {"step": 500000, "value": 0.06 + 0.005 * int(round_id)},
                                {"step": 800000, "value": 0.07 + 0.005 * int(round_id)},
                            ]
                        },
                    },
                    "run_metadata": {
                        "workflow_id": train_config.get("workflow_id"),
                        "algorithm": train_config.get("config"),
                        "env_args": deepcopy(train_config.get("env_args") or {}),
                    },
                },
            }
            self.missing_fixtures.append(f"stage1b_round{int(round_id)}_result_missing_used_synthetic:{candidate_id}")
        curves = _metric_curves_from_summary(
            deepcopy(result_record.get("metrics_summary") or {}),
            fallback_step=50,
        )
        endpoint_step = int(result_record.get("endpoint_checkpoint_step") or 800000)
        run_dir = self.fixture_run_root / f"stage1b_round{int(round_id)}" / candidate_id
        run_ref = _write_run_fixture(
            run_dir=run_dir,
            run_id=self._next_run_id(),
            workflow_id=str(train_config.get("workflow_id") or f"{self.replay_workflow_id}_{candidate_id}"),
            label=str(train_config.get("label") or candidate_id),
            train_config=train_config,
            metrics=curves,
            checkpoint_steps=[endpoint_step],
            run_metadata=deepcopy(((result_record.get("metrics_summary") or {}).get("run_metadata")) or {}),
        )
        self._materialized_runs[key] = run_ref
        return run_ref

    def _materialize_dense_reference_run(self, train_config: Dict[str, Any]) -> Dict[str, Any]:
        key = self._run_key("stage1b_dense_reference", str(train_config.get("workflow_id") or "dense_reference"))
        if key in self._materialized_runs:
            return self._materialized_runs[key]
        endpoint_step = int(train_config.get("load_step") or 800000)
        target_t_max = int((train_config.get("overrides") or {}).get("t_max") or endpoint_step + 700000)
        metrics = {
            "test_sparse_return_mean": {
                "steps": [endpoint_step, 1000000, 1250000, target_t_max],
                "values": [0.05, 0.08, 0.11, 0.12],
            },
            "sparse_return_mean": {
                "steps": [endpoint_step, 1000000, 1250000, target_t_max],
                "values": [0.06, 0.09, 0.12, 0.13],
            },
        }
        run_dir = self.fixture_run_root / "stage1b_dense_reference"
        run_ref = _write_run_fixture(
            run_dir=run_dir,
            run_id=self._next_run_id(),
            workflow_id=str(train_config.get("workflow_id") or f"{self.replay_workflow_id}_stage1b_dense_reference"),
            label=str(train_config.get("label") or "fixture_dense_reference"),
            train_config=train_config,
            metrics=metrics,
            checkpoint_steps=[endpoint_step, 1000000, 1500000, target_t_max],
            run_metadata={
                "env_key": ((train_config.get("env_args") or {}).get("key")),
                "config_name": train_config.get("config"),
                "args": {"seed": train_config.get("seed", 1)},
            },
        )
        self._materialized_runs[key] = run_ref
        return run_ref

    def _materialize_initial_mainline_seed(self, train_config: Dict[str, Any]) -> Dict[str, Any]:
        key = self._run_key("stage3_seed", str(train_config.get("workflow_id") or "initial_mainline_seed"))
        if key in self._materialized_runs:
            return self._materialized_runs[key]
        start_step = int(train_config.get("load_step") or 800000)
        target_step = int((train_config.get("overrides") or {}).get("t_max") or 1000000)
        metrics = {
            "test_sparse_return_mean": {
                "steps": [start_step, start_step + 50000, target_step],
                "values": [0.07, 0.08, 0.09],
            },
            "sparse_return_mean": {
                "steps": [start_step, start_step + 50000, target_step],
                "values": [0.08, 0.09, 0.10],
            },
        }
        run_dir = self.fixture_run_root / "stage3_initial_mainline_seed"
        run_ref = _write_run_fixture(
            run_dir=run_dir,
            run_id=self._next_run_id(),
            workflow_id=str(train_config.get("workflow_id") or f"{self.replay_workflow_id}_stage234_initial_mainline_seed"),
            label=str(train_config.get("label") or "fixture_initial_mainline_seed"),
            train_config=train_config,
            metrics=metrics,
            checkpoint_steps=[start_step, target_step],
            run_metadata={
                "env_key": ((train_config.get("env_args") or {}).get("key")),
                "config_name": train_config.get("config"),
                "args": {"seed": train_config.get("seed", 1)},
            },
        )
        self._materialized_runs[key] = run_ref
        return run_ref

    def _materialize_stage3_branch_run(self, train_config: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
        key = self._run_key("stage3_branch", candidate_id)
        if key in self._materialized_runs:
            return self._materialized_runs[key]
        branch_result = deepcopy(self.stage3_branch_results.get(candidate_id) or {})
        if not branch_result:
            env_args = deepcopy(train_config.get("env_args") or {})
            target_beta = env_args.get("pbrs_beta")
            target_wc = env_args.get("pbrs_wc")
            try:
                target_beta_value = float(target_beta)
                target_wc_value = float(target_wc)
            except Exception:
                target_beta_value = None
                target_wc_value = None
            if target_beta_value is not None and target_wc_value is not None:
                for payload in self.stage3_branch_results.values():
                    config = payload.get("pbrs_config") or {}
                    try:
                        beta_value = float(config.get("beta"))
                        wc_value = float(config.get("wc"))
                    except Exception:
                        continue
                    if abs(beta_value - target_beta_value) < 1e-9 and abs(wc_value - target_wc_value) < 1e-9:
                        branch_result = deepcopy(payload)
                        break
        if not branch_result:
            env_args = deepcopy(train_config.get("env_args") or {})
            try:
                beta_value = float(env_args.get("pbrs_beta"))
                wc_value = float(env_args.get("pbrs_wc"))
            except Exception:
                self.missing_fixtures.append(f"stage3_branch_result_missing:{candidate_id}")
                return None
            branch_result = {
                "candidate_id": candidate_id,
                "pbrs_config": {
                    "beta": beta_value,
                    "wc": wc_value,
                    "wp": float(env_args.get("pbrs_wp", round(1.0 - wc_value, 10))),
                },
                "actual_checkpoint_step": int((train_config.get("overrides") or {}).get("t_max") or 1300000),
                "metrics_summary": {
                    "metric_summary": {
                        "test_sparse_return_mean": {
                            "sampled_points": [
                                {"step": 1000000, "value": 0.04},
                                {"step": 1150000, "value": 0.06},
                                {"step": 1300000, "value": 0.05},
                            ]
                        }
                    },
                    "run_metadata": {
                        "workflow_id": train_config.get("workflow_id"),
                        "algorithm": train_config.get("config"),
                        "env_args": deepcopy(env_args),
                    },
                },
            }
            self.missing_fixtures.append(f"stage3_branch_result_missing_used_synthetic:{candidate_id}")
        summary = deepcopy(branch_result.get("metrics_summary") or {})
        metric_payload = deepcopy(summary.get("metric_summary") or {})
        curves = {}
        for metric_name, metric_summary in metric_payload.items():
            curves[str(metric_name)] = _curve_payload(
                metric_summary or {},
                fallback_step=int(metric_summary.get("first_step") or 1000000),
            )
        if not curves:
            curves = _metric_curves_from_summary(summary, fallback_step=1000000)
        actual_step = int(branch_result.get("actual_checkpoint_step") or 1300000)
        run_dir = self.fixture_run_root / "stage3_c1_branches" / candidate_id
        run_ref = _write_run_fixture(
            run_dir=run_dir,
            run_id=self._next_run_id(),
            workflow_id=str(train_config.get("workflow_id") or f"{self.replay_workflow_id}_stage234"),
            label=str(train_config.get("label") or candidate_id),
            train_config=train_config,
            metrics=curves,
            checkpoint_steps=[actual_step],
            run_metadata=deepcopy(summary.get("run_metadata") or {}),
        )
        self._materialized_runs[key] = run_ref
        return run_ref

    def resolve_run_reference(self, train_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        phase_name = str(train_config.get("phase_name") or "")
        workflow_id = str(train_config.get("workflow_id") or "")
        candidate_context = deepcopy(train_config.get("candidate_selection_context") or {})
        if workflow_id.endswith("_stage1_fixture"):
            return self._materialize_sparse_run()
        if "_stage1b_round" in workflow_id:
            match = re.search(r"_stage1b_round(\d+)_", workflow_id)
            round_id = int(match.group(1)) if match else 1
            candidate_id = workflow_id.split(f"_stage1b_round{round_id}_", 1)[-1]
            return self._materialize_stage1b_round_run(
                candidate_id=candidate_id,
                round_id=round_id,
                train_config=train_config,
            )
        if workflow_id.endswith("_stage1b_dense_reference"):
            return self._materialize_dense_reference_run(train_config)
        if phase_name == "adaptive_initial_mainline_seed" or workflow_id.endswith("_initial_mainline_seed"):
            return self._materialize_initial_mainline_seed(train_config)
        if phase_name == "adaptive_branch":
            candidate_id = str(candidate_context.get("candidate_id") or "")
            if candidate_id:
                return self._materialize_stage3_branch_run(train_config, candidate_id)
        if phase_name in {"adaptive_mainline", "adaptive_first_change_fixed_initial_baseline"}:
            return None
        return None

    def recover_existing_result(self, train_config: Dict[str, Any]) -> Dict[str, Any]:
        run_reference = self.resolve_run_reference(train_config)
        if run_reference is None:
            raise FileNotFoundError(f"missing fixture run reference for {train_config.get('workflow_id')}")
        launcher = EPyMARLTrainLauncher(repo_root=Path(__file__).resolve().parents[2])
        metrics_summary = launcher.log_analyzer.summarize(
            train_config=train_config,
            run_reference=run_reference,
        )
        return TrainLaunchResult(
            train_config=deepcopy(train_config),
            run_reference=run_reference,
            metrics_summary=metrics_summary,
        ).__dict__


class FixtureTrainLauncher(EPyMARLTrainLauncher):
    def __init__(self, *, provider: ReplayResultProvider, repo_root: Optional[Path | str] = None, python_executable: Optional[str] = None) -> None:
        super().__init__(repo_root=repo_root, python_executable=python_executable)
        self.provider = provider

    def find_matching_run_references(self, train_config: Dict[str, Any]) -> list[Dict[str, Any]]:
        run_reference = self.provider.resolve_run_reference(train_config)
        return [run_reference] if run_reference is not None else []

    def recover_existing_result(self, train_config: Dict[str, Any]) -> Dict[str, Any]:
        return self.provider.recover_existing_result(train_config)

    def launch_prebuilt_train_config_background(
        self,
        train_config: Dict[str, Any],
        *,
        stdout_path: Optional[Path | str] = None,
        stderr_to_stdout: bool = True,
    ) -> Dict[str, Any]:
        raise RuntimeError(
            f"fixture replay attempted to launch real training for {train_config.get('workflow_id')}"
        )


class RecordingLLMBackend:
    def __init__(self, *, provider: ReplayResultProvider, **_: Any) -> None:
        self.provider = provider

    def _extract_payload(self, user_prompt: str, markers: List[str]) -> Dict[str, Any]:
        decoder = json.JSONDecoder()
        for marker in markers:
            index = user_prompt.find(marker)
            if index == -1:
                continue
            search_start = user_prompt.find("{", index)
            if search_start == -1:
                continue
            try:
                payload, _ = decoder.raw_decode(user_prompt[search_start:])
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        meta = deepcopy(metadata or {})
        role = str(meta.get("role") or "")
        if role == "policy_guidance_integrated_critic":
            intervention_point = str(meta.get("intervention_point") or "stage1_after_sparse_baseline")
            payload = self._extract_payload(user_prompt, ["Input payload:"])
            response = self.provider.policy_guidance_card(
                intervention_point,
                deepcopy(payload.get("reward_diagnosis") or {}),
            )
        elif role == "adaptive_stage3_candidate_generation":
            payload = self._extract_payload(user_prompt, ["Adaptive Stage 3 candidate-generation payload:"])
            round_id = int(payload.get("round_id") or 1)
            response = {"candidates": self.provider.stage3_round_candidates(round_id)}
        elif role == "adaptive_stage3_result_diagnosis":
            response = self.provider.stage3_result_diagnosis_payload()
        else:
            agent_name = str(meta.get("agent_name") or "")
            if agent_name == "stage1b_sparse_diagnosis":
                response = {
                    "goal": "early_budget_selection",
                    "observed_sparse_failures": ["fixture replay reused Stage1 sparse evidence"],
                    "reward_design_needs": ["verify Stage1b call chain without launching training"],
                    "initial_search_priors": {"prefer_diverse_round1_candidates": True},
                    "stage_hypotheses": ["reference-like and stability-aware candidates should remain safe"],
                }
            elif agent_name == "stage1b_round1_generator":
                response = self.provider.stage1b_round1_candidates()
            elif agent_name == "stage1b_round1_critic":
                payload = self._extract_payload(user_prompt, ["Payload:"])
                response = _deterministic_round1_analysis(list(payload.get("round1_results") or []))
            elif agent_name == "stage1b_round2_generator":
                response = self.provider.stage1b_round2_candidates()
            elif agent_name == "stage1b_final_critic":
                payload = self._extract_payload(user_prompt, ["Payload:"])
                response = _build_final_selection_fallback(list(payload.get("all_candidate_results") or []))
            else:
                response = {"status": "recording_backend_unhandled", "role": role, "metadata": meta}
        return {
            "text": json.dumps(response, ensure_ascii=False),
            "raw_response": {"recording_backend": True, "metadata": meta, "system_prompt": system_prompt[:80]},
            "backend_metadata": {
                "provider": "recording",
                "model": "recording-llm",
                "base_url": "recording://fixture",
            },
            "model": "recording-llm",
            "usage": None,
            "request_id": None,
        }
