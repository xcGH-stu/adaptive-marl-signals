from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from rewarding.spec_renderer import render_reward_code_from_spec
from rewarding.spec_schema import (
    SUPPORTED_PBRS_CLOSENESS_MODES,
    SUPPORTED_PBRS_GATE_MODES,
    SUPPORTED_PBRS_VARIANTS,
    get_default_reward_spec,
    validate_reward_spec,
)
from workflows.reward_spec_budget import (
    DEFAULT_REWARD_SPEC_CHANGE_BUDGET,
    enforce_reward_spec_change_budget,
)
from workflows.candidate_manager import (
    attach_candidate_proposals_to_runs,
    build_candidate_comparison_table,
    build_candidate_diff_table,
    build_candidate_execution_plan,
)
from workflows.reward_spec_updater import derive_suggested_reward_spec
from workflows.storage import WorkflowStorage
from workflows.branch_critic_patch import select_branch_evidence_for_checkpoint_context
from workflows.policy_guidance import (
    get_default_policy_guidance_spec,
    summarize_policy_guidance_spec,
    validate_policy_guidance_spec,
)


class GeneratorClient(Protocol):
    def generate(
        self,
        *,
        round_id: int,
        workflow_spec: Dict[str, Any],
        previous_round: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Return a dict containing at least:
            prompt: str
            response: dict
            reward_code: str
        Optionally:
            reward_spec: dict
        """


class CriticClient(Protocol):
    def critique(
        self,
        *,
        round_id: int,
        workflow_spec: Dict[str, Any],
        current_round: Dict[str, Any],
        previous_round: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Return a dict containing at least:
            prompt: str
            response: dict
        Optionally:
            alpha_policy: dict
        """


class TrainLauncher(Protocol):
    def run_training(
        self,
        *,
        round_id: int,
        workflow_spec: Dict[str, Any],
        reward_module_path: str,
        alpha_policy: Optional[Dict[str, Any]],
        policy_guidance_spec: Optional[Dict[str, Any]],
        workflow_id: str,
        candidate_id: Optional[str] = None,
        reward_paradigm: Optional[str] = None,
        active_pbrs_field: Optional[str] = None,
        active_pbrs_field_source: Optional[str] = None,
        candidate_value: Optional[Any] = None,
        active_checkpoint_context: Optional[Dict[str, Any]] = None,
        candidate_selection_context: Optional[Dict[str, Any]] = None,
        phase_name: str = "main",
        train_overrides_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return a dict containing at least:
            train_config: dict
            run_reference: dict
        Optionally:
            metrics_summary: dict
        """


@dataclass
class WorkflowRunResult:
    workflow_id: str
    completed_rounds: int
    manifest: Dict[str, Any]
    last_round_status: Dict[str, Any]


@dataclass
class RewardWorkflow:
    workflow_id: str
    workflow_spec: Dict[str, Any]
    generator_client: GeneratorClient
    critic_client: CriticClient
    train_launcher: TrainLauncher
    storage_root: Optional[Path | str] = None
    storage: WorkflowStorage = field(init=False)

    def __post_init__(self):
        reward_paradigm = self._get_reward_paradigm()
        if reward_paradigm not in {"pbrs", "heuristic"}:
            raise ValueError(
                "workflow_spec.reward_paradigm must be 'pbrs' or 'heuristic', "
                f"got {reward_paradigm!r}"
            )
        schedule = self._get_pbrs_field_schedule()
        if reward_paradigm == "pbrs" and schedule:
            max_rounds = int(self.workflow_spec.get("max_rounds", 1))
            stage_schedule_present = self._has_stage_aware_active_field_config()
            if len(schedule) < max_rounds and not stage_schedule_present:
                raise ValueError(
                    "pbrs_tuning.field_schedule length must be at least max_rounds unless "
                    "stage-aware active-field configuration is provided "
                    f"(got {len(schedule)} < {max_rounds})"
                )
            candidate_values = self._get_pbrs_candidate_values()
            required_fields = set(schedule)
            if stage_schedule_present:
                required_fields.update(self._get_stage_aware_active_fields())
            missing_candidate_fields = [
                field_name for field_name in sorted(required_fields) if field_name not in candidate_values
            ]
            if missing_candidate_fields:
                raise ValueError(
                    "pbrs_tuning.candidate_values must provide candidate lists for every "
                    "scheduled field; missing: "
                    + ", ".join(sorted(missing_candidate_fields))
                )
        manifest_data = {
            "status": "initialized",
            "workflow_spec": self.workflow_spec,
            "max_rounds": int(self.workflow_spec.get("max_rounds", 1)),
            "reward_paradigm": reward_paradigm,
        }
        self.storage = WorkflowStorage.create(
            self.workflow_id,
            root_dir=self.storage_root,
            manifest_data=manifest_data,
        )

    def run(self, *, start_round: int = 1) -> WorkflowRunResult:
        max_rounds = int(self.workflow_spec.get("max_rounds", 1))
        start_round = int(start_round)
        if start_round < 1 or start_round > max_rounds:
            raise ValueError(
                f"start_round must be in [1, {max_rounds}], got {start_round}"
            )
        self.storage.update_manifest({"status": "running", "failure_reason": None})
        self.storage.append_timeline_event(
            "workflow_started" if start_round == 1 else "workflow_resumed",
            {
                "workflow_id": self.workflow_id,
                "max_rounds": max_rounds,
                "start_round": start_round,
            },
        )

        last_round_status = {"status": "not_started"}
        completed_rounds = max(0, start_round - 1)

        try:
            for round_id in range(start_round, max_rounds + 1):
                last_round_status = self.run_round(round_id)
                completed_rounds = round_id
        except Exception as exc:
            self.storage.update_manifest(
                {
                    "status": "failed",
                    "failure_reason": str(exc),
                }
            )
            self.storage.append_timeline_event(
                "workflow_failed",
                {"workflow_id": self.workflow_id, "error": str(exc)},
            )
            raise

        manifest = self.storage.mark_workflow_completed()
        return WorkflowRunResult(
            workflow_id=self.workflow_id,
            completed_rounds=completed_rounds,
            manifest=manifest,
            last_round_status=last_round_status,
        )

    def run_round(self, round_id: int) -> Dict[str, Any]:
        previous_round = self.storage.get_previous_round_artifacts(round_id)
        active_checkpoint_context = self._get_active_pbrs_checkpoint_context(round_id)
        active_pbrs_field, active_pbrs_field_source = self._resolve_active_pbrs_field(
            round_id=round_id,
            active_checkpoint_context=active_checkpoint_context,
            previous_round=previous_round,
        )
        active_field_carryover_context = self._build_active_field_carryover_context(
            active_pbrs_field=active_pbrs_field,
            active_pbrs_field_source=active_pbrs_field_source,
            previous_round=previous_round,
        )
        candidate_selection_context = self._resolve_candidate_selection_context(
            round_id=round_id,
            active_pbrs_field=active_pbrs_field,
            active_checkpoint_context=active_checkpoint_context,
            previous_round=previous_round,
        )
        allow_term_updates = self._allow_term_updates_for_round(round_id)
        if previous_round is not None:
            suggested_reward_spec = derive_suggested_reward_spec(
                previous_round,
                allowed_pbrs_fields=self._get_allowed_pbrs_update_fields_for_previous_round(
                    round_id
                ),
                allow_term_updates=allow_term_updates,
            )
            if suggested_reward_spec is not None:
                previous_round["suggested_reward_spec"] = suggested_reward_spec
        critic_resume_payload = self._load_critic_resume_payload(
            round_id=round_id,
            previous_round=previous_round,
            active_pbrs_field=active_pbrs_field,
            active_pbrs_field_source=active_pbrs_field_source,
            active_field_carryover_context=active_field_carryover_context,
            candidate_selection_context=candidate_selection_context,
            active_checkpoint_context=active_checkpoint_context,
        )
        if critic_resume_payload is not None:
            return self._resume_round_from_critic_stage(
                round_id=round_id,
                previous_round=previous_round,
                current_round=critic_resume_payload["current_round"],
                active_pbrs_field=active_pbrs_field,
                active_pbrs_field_source=active_pbrs_field_source,
                active_field_carryover_context=active_field_carryover_context,
                candidate_selection_context=candidate_selection_context,
                active_checkpoint_context=active_checkpoint_context,
                validation_skipped=bool(critic_resume_payload.get("validation_skipped", False)),
                validation_skip_reason=critic_resume_payload.get("validation_skip_reason"),
                reused_candidate_count=int(critic_resume_payload.get("reused_candidate_count", 0)),
                rerun_candidate_count=int(critic_resume_payload.get("rerun_candidate_count", 0)),
            )
        self.storage.ensure_round(round_id)
        self.storage.save_round_status(
            round_id,
            "round_started",
            {
                "reward_paradigm": self._get_reward_paradigm(),
                "active_pbrs_field": active_pbrs_field,
                "active_pbrs_field_source": active_pbrs_field_source,
                "active_field_carryover_context": active_field_carryover_context,
                "candidate_selection_context": candidate_selection_context,
                "active_checkpoint_context": active_checkpoint_context,
            },
        )
        self.storage.append_timeline_event(
            "round_started",
            {
                "round_id": round_id,
                "reward_paradigm": self._get_reward_paradigm(),
                "active_pbrs_field": active_pbrs_field,
                "active_pbrs_field_source": active_pbrs_field_source,
                "active_field_carryover_context": active_field_carryover_context,
                "candidate_selection_context": candidate_selection_context,
                "active_checkpoint_context": active_checkpoint_context,
            },
        )

        try:
            generator_output = self.generator_client.generate(
                round_id=round_id,
                workflow_spec=self.workflow_spec,
                previous_round=previous_round,
            )
            self._validate_generator_output(generator_output)
            reward_spec = generator_output.get("reward_spec")
            if reward_spec is None:
                reward_spec = (
                    previous_round.get("suggested_reward_spec")
                    if previous_round is not None
                    else None
                )
            if reward_spec is None:
                reward_spec = self._build_initial_reward_spec()
            validated_reward_spec = validate_reward_spec(reward_spec)
            policy_guidance_spec = generator_output.get("policy_guidance_spec")
            if policy_guidance_spec is None:
                policy_guidance_spec = self._get_base_policy_guidance_spec_for_round(
                    round_id=round_id,
                    previous_round=previous_round,
                )
            validated_policy_guidance_spec = validate_policy_guidance_spec(
                policy_guidance_spec
            )
            base_reward_spec = self._get_base_reward_spec_for_round(
                round_id=round_id,
                previous_round=previous_round,
            )
            validated_reward_spec = self._enforce_round_tuning_constraints(
                round_id=round_id,
                base_reward_spec=base_reward_spec,
                candidate_reward_spec=validated_reward_spec,
            )
            if previous_round is not None and previous_round.get("suggested_reward_spec") is not None:
                change_summary = enforce_reward_spec_change_budget(
                    base_spec=base_reward_spec,
                    candidate_spec=validated_reward_spec,
                    budget=self.workflow_spec.get(
                        "reward_spec_change_budget",
                        DEFAULT_REWARD_SPEC_CHANGE_BUDGET,
                    ),
                )
            else:
                change_summary = None
            rendered_reward_code = render_reward_code_from_spec(validated_reward_spec)
            candidate_runs = self._build_round_candidates(
                round_id=round_id,
                reward_spec=validated_reward_spec,
                reward_code=rendered_reward_code,
                active_pbrs_field=active_pbrs_field,
                active_checkpoint_context=active_checkpoint_context,
                candidate_selection_context=candidate_selection_context,
            )
            candidate_proposals = (
                ((generator_output.get("response") or {}).get("parsed") or {}).get(
                    "candidate_proposals"
                )
                or []
            )
            candidate_runs = attach_candidate_proposals_to_runs(
                candidate_runs=candidate_runs,
                candidate_proposals=candidate_proposals,
                base_reward_spec=base_reward_spec,
            )
            candidate_execution_plan = build_candidate_execution_plan(candidate_runs)

            round_paths = self.storage.save_generator_artifacts(
                round_id,
                prompt=generator_output["prompt"],
                response=generator_output["response"],
                candidate_proposals=candidate_proposals,
                reward_spec=validated_reward_spec,
                policy_guidance_spec=validated_policy_guidance_spec,
                reward_code=rendered_reward_code,
            )
            self.storage.save_candidate_manifest(
                round_id,
                [
                    {
                        "candidate_id": candidate_run["candidate_id"],
                        "active_pbrs_field": candidate_run["active_pbrs_field"],
                        "active_pbrs_field_source": candidate_run["active_pbrs_field_source"],
                        "candidate_value": candidate_run["candidate_value"],
                        "candidate_selection_context": candidate_run.get(
                            "candidate_selection_context"
                        ),
                        "active_checkpoint_context": candidate_run["active_checkpoint_context"],
                        "candidate_proposal": candidate_run.get("candidate_proposal"),
                        "candidate_diff": candidate_run.get("candidate_diff"),
                        "reward_spec_path": str(candidate_run["reward_spec_path"]),
                        "reward_function_path": str(candidate_run["reward_module_path"]),
                    }
                    for candidate_run in candidate_runs
                ],
                metadata={
                    "reward_paradigm": self._get_reward_paradigm(),
                    "active_pbrs_field": active_pbrs_field,
                    "active_pbrs_field_source": active_pbrs_field_source,
                    "active_field_carryover_context": active_field_carryover_context,
                    "candidate_selection_context": candidate_selection_context,
                    "active_checkpoint_context": active_checkpoint_context,
                },
            )
            self.storage.save_candidate_execution_plan(round_id, candidate_execution_plan)
        except Exception as exc:
            self._record_round_failure(round_id, stage="generator", error=exc)
            raise

        self.storage.save_round_status(round_id, "training_started")
        self.storage.append_timeline_event(
            "training_started",
            {
                "round_id": round_id,
                "reward_paradigm": self._get_reward_paradigm(),
                "active_pbrs_field": active_pbrs_field,
                "active_pbrs_field_source": active_pbrs_field_source,
                "active_field_carryover_context": active_field_carryover_context,
                "candidate_selection_context": candidate_selection_context,
                "active_checkpoint_context": active_checkpoint_context,
            },
        )

        alpha_policy = generator_output.get("alpha_policy")
        if self._alpha_policy_updates_enabled():
            if alpha_policy is None:
                alpha_policy = validated_reward_spec.get("alpha_policy")
        else:
            alpha_policy = self._get_frozen_alpha_policy(
                reward_spec=validated_reward_spec,
                base_reward_spec=base_reward_spec,
            )
        validation_candidate_results = None
        validation_summary = None
        validation_skipped = False
        validation_skip_reason = None
        if self._use_pbrs_validation_stage(round_id):
            if self._should_skip_pbrs_validation_stage(
                round_id=round_id,
                candidate_runs=candidate_runs,
            ):
                validation_skipped = True
                validation_skip_reason = "single_candidate"
                validation_summary = {
                    "skipped": True,
                    "skip_reason": validation_skip_reason,
                    "shortlist_size": len(candidate_runs),
                    "active_checkpoint_context": deepcopy(active_checkpoint_context),
                }
                self.storage.save_json(
                    round_id,
                    "validation_candidate_results.json",
                    {
                        "candidate_results": [],
                        "validation_summary": validation_summary,
                    },
                )
                self.storage.append_timeline_event(
                    "validation_skipped",
                    {
                        "round_id": round_id,
                        "reward_paradigm": self._get_reward_paradigm(),
                        "active_pbrs_field": active_pbrs_field,
                        "active_pbrs_field_source": active_pbrs_field_source,
                        "active_field_carryover_context": active_field_carryover_context,
                        "candidate_selection_context": candidate_selection_context,
                        "active_checkpoint_context": active_checkpoint_context,
                        "skip_reason": validation_skip_reason,
                        "candidate_count": len(candidate_runs),
                    },
                )
            else:
                try:
                    validation_candidate_results, validation_summary, candidate_runs = (
                        self._run_pbrs_validation_stage(
                            round_id=round_id,
                            candidate_runs=candidate_runs,
                            alpha_policy=alpha_policy,
                            policy_guidance_spec=validated_policy_guidance_spec,
                            active_checkpoint_context=active_checkpoint_context,
                            active_field_carryover_context=active_field_carryover_context,
                        )
                    )
                    self.storage.save_json(
                        round_id,
                        "validation_candidate_results.json",
                        {
                            "candidate_results": validation_candidate_results,
                            "validation_summary": validation_summary,
                        },
                    )
                    self.storage.append_timeline_event(
                        "validation_completed",
                        {
                            "round_id": round_id,
                            "reward_paradigm": self._get_reward_paradigm(),
                            "active_pbrs_field": active_pbrs_field,
                            "active_pbrs_field_source": active_pbrs_field_source,
                            "active_field_carryover_context": active_field_carryover_context,
                            "candidate_selection_context": candidate_selection_context,
                            "active_checkpoint_context": active_checkpoint_context,
                            "shortlist_size": len(candidate_runs),
                        },
                    )
                except Exception as exc:
                    self._record_round_failure(round_id, stage="validation", error=exc)
                    raise
        try:
            candidate_results = self._run_candidate_trainings(
                round_id=round_id,
                candidate_runs=candidate_runs,
                alpha_policy=alpha_policy,
                policy_guidance_spec=validated_policy_guidance_spec,
                active_checkpoint_context=active_checkpoint_context,
                active_field_carryover_context=active_field_carryover_context,
                candidate_selection_context=candidate_selection_context,
            )
            reused_candidate_count = sum(
                1 for candidate_result in candidate_results if candidate_result.get("reused_existing")
            )
            rerun_candidate_count = len(candidate_results) - reused_candidate_count
            single_train_output = (
                candidate_results[0]["_train_output"] if len(candidate_results) == 1 else None
            )

            if len(candidate_results) == 1:
                self.storage.record_training_run(
                    round_id,
                    train_config=single_train_output["train_config"],
                    run_reference=single_train_output["run_reference"],
                    metrics_summary=single_train_output.get("metrics_summary"),
                )
            else:
                self.storage.save_json(
                    round_id,
                    "candidate_results.json",
                    {
                        "candidate_results": [
                            {
                                key: value
                                for key, value in candidate_result.items()
                                if key != "_train_output"
                            }
                            for candidate_result in candidate_results
                        ]
                    },
                )
        except Exception as exc:
            self._record_round_failure(round_id, stage="training", error=exc)
            raise

        current_round = {
            "round_id": round_id,
            "reward_spec": validated_reward_spec,
            "policy_guidance_spec": validated_policy_guidance_spec,
            "policy_guidance_summary": summarize_policy_guidance_spec(
                validated_policy_guidance_spec
            ),
            "reward_code": round_paths.reward_function.read_text(encoding="utf-8"),
            "train_output": single_train_output if len(candidate_results) == 1 else None,
            "train_metrics_summary": (
                candidate_results[0]["train_metrics_summary"] if len(candidate_results) == 1 else None
            ),
            "reward_module_path": str(round_paths.reward_function),
            "alpha_policy": alpha_policy,
            "reward_spec_change_summary": change_summary,
            "generator_candidate_proposals": deepcopy(candidate_proposals),
            "candidate_diffs": build_candidate_diff_table(candidate_runs),
            "candidate_execution_plan": deepcopy(candidate_execution_plan),
            "active_pbrs_field": active_pbrs_field,
            "active_pbrs_field_source": active_pbrs_field_source,
            "active_field_carryover_context": active_field_carryover_context,
            "candidate_selection_context": candidate_selection_context,
            "active_checkpoint_context": active_checkpoint_context,
            "effective_pbrs_design_update_policy": self._get_effective_pbrs_design_update_policy(
                round_id=round_id,
                active_checkpoint_context=active_checkpoint_context,
            ),
            "branch_critic_recommendation": deepcopy(
                self.workflow_spec.get("branch_critic_recommendation")
            ),
            "branch_critic_patch": deepcopy(self.workflow_spec.get("branch_critic_patch")),
            "branch_plan_summary": deepcopy(self.workflow_spec.get("branch_plan_summary")),
            "branch_stage_evidence": self._get_branch_stage_evidence(
                active_checkpoint_context=active_checkpoint_context
            ),
            "branch_design_transition": deepcopy(
                (
                    (self.workflow_spec.get("branch_critic_recommendation") or {}).get(
                        "recommended_design_transition"
                    )
                    or (self.workflow_spec.get("branch_critic_patch") or {}).get(
                        "recommended_design_transition"
                    )
                )
            ),
            "branch_design_transition_application": deepcopy(
                self.workflow_spec.get("branch_design_transition_application")
            ),
            "search_strategy_context_summary": self._build_search_strategy_context_summary(
                round_id=round_id,
                active_pbrs_field=active_pbrs_field,
                active_pbrs_field_source=active_pbrs_field_source,
                active_field_carryover_context=active_field_carryover_context,
                candidate_selection_context=candidate_selection_context,
                active_checkpoint_context=active_checkpoint_context,
                previous_round=previous_round,
            ),
            "critic_iteration_context": self._build_critic_iteration_context(
                round_id=round_id,
                active_pbrs_field=active_pbrs_field,
                active_checkpoint_context=active_checkpoint_context,
                candidate_selection_context=candidate_selection_context,
                validation_candidate_results=validation_candidate_results,
                validation_summary=validation_summary,
                candidate_results=[
                    {
                        key: value
                        for key, value in candidate_result.items()
                        if key != "_train_output"
                    }
                    for candidate_result in candidate_results
                ],
                previous_round=previous_round,
            ),
            "validation_candidate_results": validation_candidate_results,
            "validation_summary": validation_summary,
            "validation_skipped": validation_skipped,
            "validation_skip_reason": validation_skip_reason,
            "reused_candidate_count": reused_candidate_count,
            "rerun_candidate_count": rerun_candidate_count,
            "candidate_results": [
                {
                    key: value
                    for key, value in candidate_result.items()
                    if key != "_train_output"
                }
                for candidate_result in candidate_results
            ],
            "candidate_comparison_table": build_candidate_comparison_table(
                validation_candidate_results=validation_candidate_results,
                candidate_results=[
                    {
                        key: value
                        for key, value in candidate_result.items()
                        if key != "_train_output"
                    }
                    for candidate_result in candidate_results
                ],
            ),
        }
        search_strategy_recommendation = None
        structured_diagnosis = None

        try:
            critic_output = self.critic_client.critique(
                round_id=round_id,
                workflow_spec=self.workflow_spec,
                current_round=current_round,
                previous_round=previous_round,
            )
            self._validate_critic_output(critic_output)
            self.storage.save_critic_artifacts(
                round_id,
                prompt=critic_output["prompt"],
                response=critic_output["response"],
                alpha_policy=critic_output.get("alpha_policy"),
            )
            critic_parsed = (critic_output.get("response") or {}).get("parsed")
            search_strategy_recommendation = (
                critic_parsed.get("search_strategy_recommendation")
                if isinstance(critic_parsed, dict)
                else None
            )
            structured_diagnosis = (
                critic_parsed.get("structured_diagnosis")
                if isinstance(critic_parsed, dict)
                else None
            )
        except Exception as exc:
            self._record_round_failure(round_id, stage="critic", error=exc)
            raise

        round_status = self.storage.save_round_status(
            round_id,
            "completed",
            {
                "reward_paradigm": self._get_reward_paradigm(),
                "active_pbrs_field": active_pbrs_field,
                "active_pbrs_field_source": active_pbrs_field_source,
                "active_field_carryover_context": active_field_carryover_context,
                "candidate_selection_context": candidate_selection_context,
                "active_checkpoint_context": active_checkpoint_context,
                "critic_structured_diagnosis": structured_diagnosis,
                "critic_verdict": (
                    structured_diagnosis.get("verdict")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "critic_best_candidate": (
                    structured_diagnosis.get("best_candidate")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "critic_whether_to_full_run": (
                    structured_diagnosis.get("whether_to_full_run")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "search_strategy_recommendation": search_strategy_recommendation,
                "validation_skipped": validation_skipped,
                "validation_skip_reason": validation_skip_reason,
                "reused_candidate_count": reused_candidate_count,
                "rerun_candidate_count": rerun_candidate_count,
            },
        )
        self.storage.append_timeline_event(
            "round_completed",
            {
                "round_id": round_id,
                "reward_paradigm": self._get_reward_paradigm(),
                "active_pbrs_field": active_pbrs_field,
                "active_pbrs_field_source": active_pbrs_field_source,
                "active_field_carryover_context": active_field_carryover_context,
                "candidate_selection_context": candidate_selection_context,
                "active_checkpoint_context": active_checkpoint_context,
                "critic_structured_diagnosis": structured_diagnosis,
                "critic_verdict": (
                    structured_diagnosis.get("verdict")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "critic_best_candidate": (
                    structured_diagnosis.get("best_candidate")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "critic_whether_to_full_run": (
                    structured_diagnosis.get("whether_to_full_run")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "search_strategy_recommendation": search_strategy_recommendation,
                "validation_skipped": validation_skipped,
                "validation_skip_reason": validation_skip_reason,
                "reused_candidate_count": reused_candidate_count,
                "rerun_candidate_count": rerun_candidate_count,
            },
        )
        return round_status

    def _load_critic_resume_payload(
        self,
        *,
        round_id: int,
        previous_round: Optional[Dict[str, Any]],
        active_pbrs_field: Optional[str],
        active_pbrs_field_source: Optional[str],
        active_field_carryover_context: Optional[Dict[str, Any]],
        candidate_selection_context: Optional[Dict[str, Any]],
        active_checkpoint_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        round_artifacts = self.storage.get_round_artifacts(round_id)
        if not isinstance(round_artifacts, dict):
            return None
        round_status = round_artifacts.get("status")
        if not isinstance(round_status, dict):
            return None
        if round_status.get("status") != "failed" or round_status.get("stage") != "critic":
            return None
        if not isinstance(round_artifacts.get("generator_response"), dict):
            return None
        if not isinstance(round_artifacts.get("reward_spec"), dict):
            return None
        if not isinstance(round_artifacts.get("reward_code"), str):
            return None

        candidate_manifest = round_artifacts.get("candidate_manifest") or {}
        candidate_entries = candidate_manifest.get("candidates")
        if not isinstance(candidate_entries, list) or not candidate_entries:
            return None

        candidate_results_payload = round_artifacts.get("candidate_results")
        candidate_results = None
        single_train_output = None
        if isinstance(candidate_results_payload, dict):
            candidate_results = candidate_results_payload.get("candidate_results")
        if not isinstance(candidate_results, list):
            train_config = round_artifacts.get("train_config")
            train_run_ref = round_artifacts.get("train_run_ref")
            if not (isinstance(train_config, dict) and isinstance(train_run_ref, dict)):
                return None
            single_candidate = candidate_entries[0]
            candidate_results = [
                {
                    "candidate_id": single_candidate.get("candidate_id"),
                    "active_pbrs_field": single_candidate.get("active_pbrs_field"),
                    "active_pbrs_field_source": single_candidate.get("active_pbrs_field_source"),
                    "candidate_value": single_candidate.get("candidate_value"),
                    "candidate_selection_context": single_candidate.get(
                        "candidate_selection_context"
                    ),
                    "active_checkpoint_context": single_candidate.get(
                        "active_checkpoint_context"
                    ),
                    "reward_spec": round_artifacts.get("reward_spec"),
                    "reward_module_path": str(self.storage.get_round_paths(round_id).reward_function),
                    "train_config": train_config,
                    "run_reference": {
                        "run_dir": train_run_ref.get("run_dir"),
                        "run_id": train_run_ref.get("run_id"),
                        "metrics_json": train_run_ref.get("metrics_json"),
                    },
                    "train_metrics_summary": round_artifacts.get("train_metrics_summary"),
                    "phase_name": "main",
                    "reused_existing": True,
                }
            ]
            single_train_output = {
                "train_config": train_config,
                "run_reference": train_run_ref,
                "metrics_summary": round_artifacts.get("train_metrics_summary"),
            }
        else:
            for candidate in candidate_results:
                if isinstance(candidate, dict):
                    candidate.setdefault("reward_module_path", None)
                    candidate.setdefault("reused_existing", True)

        validation_payload = round_artifacts.get("validation_candidate_results_payload") or {}
        validation_candidate_results = validation_payload.get("candidate_results")
        validation_summary = validation_payload.get("validation_summary")
        validation_skipped = bool((validation_summary or {}).get("skipped", False))
        validation_skip_reason = (validation_summary or {}).get("skip_reason")

        current_round = {
            "round_id": round_id,
            "reward_spec": round_artifacts["reward_spec"],
            "policy_guidance_spec": round_artifacts.get("policy_guidance_spec"),
            "policy_guidance_summary": summarize_policy_guidance_spec(
                round_artifacts.get("policy_guidance_spec")
            ),
            "reward_code": round_artifacts["reward_code"],
            "train_output": single_train_output,
            "train_metrics_summary": round_artifacts.get("train_metrics_summary"),
            "reward_module_path": str(self.storage.get_round_paths(round_id).reward_function),
            "alpha_policy": round_artifacts.get("alpha_policy"),
            "reward_spec_change_summary": None,
            "generator_candidate_proposals": (
                ((round_artifacts.get("candidate_proposals") or {}).get("candidate_proposals"))
                if isinstance(round_artifacts.get("candidate_proposals"), dict)
                else (
                    (((round_artifacts.get("generator_response") or {}).get("parsed") or {}).get("candidate_proposals"))
                    if isinstance(round_artifacts.get("generator_response"), dict)
                    else None
                )
            ),
            "candidate_execution_plan": round_artifacts.get("candidate_execution_plan"),
            "active_pbrs_field": active_pbrs_field,
            "active_pbrs_field_source": active_pbrs_field_source,
            "active_field_carryover_context": active_field_carryover_context,
            "candidate_selection_context": candidate_selection_context,
            "active_checkpoint_context": active_checkpoint_context,
            "effective_pbrs_design_update_policy": self._get_effective_pbrs_design_update_policy(
                round_id=round_id,
                active_checkpoint_context=active_checkpoint_context,
            ),
            "branch_critic_recommendation": deepcopy(
                self.workflow_spec.get("branch_critic_recommendation")
            ),
            "branch_critic_patch": deepcopy(self.workflow_spec.get("branch_critic_patch")),
            "branch_plan_summary": deepcopy(self.workflow_spec.get("branch_plan_summary")),
            "branch_stage_evidence": self._get_branch_stage_evidence(
                active_checkpoint_context=active_checkpoint_context
            ),
            "branch_design_transition": deepcopy(
                (
                    (self.workflow_spec.get("branch_critic_recommendation") or {}).get(
                        "recommended_design_transition"
                    )
                    or (self.workflow_spec.get("branch_critic_patch") or {}).get(
                        "recommended_design_transition"
                    )
                )
            ),
            "branch_design_transition_application": deepcopy(
                self.workflow_spec.get("branch_design_transition_application")
            ),
            "search_strategy_context_summary": self._build_search_strategy_context_summary(
                round_id=round_id,
                active_pbrs_field=active_pbrs_field,
                active_pbrs_field_source=active_pbrs_field_source,
                active_field_carryover_context=active_field_carryover_context,
                candidate_selection_context=candidate_selection_context,
                active_checkpoint_context=active_checkpoint_context,
                previous_round=previous_round,
            ),
            "critic_iteration_context": self._build_critic_iteration_context(
                round_id=round_id,
                active_pbrs_field=active_pbrs_field,
                active_checkpoint_context=active_checkpoint_context,
                candidate_selection_context=candidate_selection_context,
                validation_candidate_results=validation_candidate_results,
                validation_summary=validation_summary,
                candidate_results=candidate_results,
                previous_round=previous_round,
            ),
            "validation_candidate_results": validation_candidate_results,
            "validation_summary": validation_summary,
            "validation_skipped": validation_skipped,
            "validation_skip_reason": validation_skip_reason,
            "reused_candidate_count": len(candidate_results),
            "rerun_candidate_count": 0,
            "candidate_results": candidate_results,
            "candidate_diffs": [
                {
                    "candidate_id": item.get("candidate_id"),
                    "candidate_value": item.get("candidate_value"),
                    "candidate_proposal": item.get("candidate_proposal"),
                    "candidate_diff": item.get("candidate_diff"),
                }
                for item in (candidate_entries or [])
                if isinstance(item, dict)
            ],
            "candidate_comparison_table": build_candidate_comparison_table(
                validation_candidate_results=validation_candidate_results,
                candidate_results=candidate_results,
            ),
        }
        return {
            "current_round": current_round,
            "validation_skipped": validation_skipped,
            "validation_skip_reason": validation_skip_reason,
            "reused_candidate_count": len(candidate_results),
            "rerun_candidate_count": 0,
        }

    def _resume_round_from_critic_stage(
        self,
        *,
        round_id: int,
        previous_round: Optional[Dict[str, Any]],
        current_round: Dict[str, Any],
        active_pbrs_field: Optional[str],
        active_pbrs_field_source: Optional[str],
        active_field_carryover_context: Optional[Dict[str, Any]],
        candidate_selection_context: Optional[Dict[str, Any]],
        active_checkpoint_context: Optional[Dict[str, Any]],
        validation_skipped: bool,
        validation_skip_reason: Optional[str],
        reused_candidate_count: int,
        rerun_candidate_count: int,
    ) -> Dict[str, Any]:
        self.storage.append_timeline_event(
            "round_resumed_from_critic",
            {
                "round_id": round_id,
                "reward_paradigm": self._get_reward_paradigm(),
                "active_pbrs_field": active_pbrs_field,
                "active_pbrs_field_source": active_pbrs_field_source,
                "active_checkpoint_context": active_checkpoint_context,
            },
        )
        search_strategy_recommendation = None
        structured_diagnosis = None
        try:
            critic_output = self.critic_client.critique(
                round_id=round_id,
                workflow_spec=self.workflow_spec,
                current_round=current_round,
                previous_round=previous_round,
            )
            self._validate_critic_output(critic_output)
            self.storage.save_critic_artifacts(
                round_id,
                prompt=critic_output["prompt"],
                response=critic_output["response"],
                alpha_policy=critic_output.get("alpha_policy"),
            )
            critic_parsed = (critic_output.get("response") or {}).get("parsed")
            search_strategy_recommendation = (
                critic_parsed.get("search_strategy_recommendation")
                if isinstance(critic_parsed, dict)
                else None
            )
            structured_diagnosis = (
                critic_parsed.get("structured_diagnosis")
                if isinstance(critic_parsed, dict)
                else None
            )
        except Exception as exc:
            self._record_round_failure(round_id, stage="critic", error=exc)
            raise

        round_status = self.storage.save_round_status(
            round_id,
            "completed",
            {
                "reward_paradigm": self._get_reward_paradigm(),
                "active_pbrs_field": active_pbrs_field,
                "active_pbrs_field_source": active_pbrs_field_source,
                "active_field_carryover_context": active_field_carryover_context,
                "candidate_selection_context": candidate_selection_context,
                "active_checkpoint_context": active_checkpoint_context,
                "critic_structured_diagnosis": structured_diagnosis,
                "critic_verdict": (
                    structured_diagnosis.get("verdict")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "critic_best_candidate": (
                    structured_diagnosis.get("best_candidate")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "critic_whether_to_full_run": (
                    structured_diagnosis.get("whether_to_full_run")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "search_strategy_recommendation": search_strategy_recommendation,
                "validation_skipped": validation_skipped,
                "validation_skip_reason": validation_skip_reason,
                "reused_candidate_count": reused_candidate_count,
                "rerun_candidate_count": rerun_candidate_count,
            },
        )
        self.storage.append_timeline_event(
            "round_completed",
            {
                "round_id": round_id,
                "reward_paradigm": self._get_reward_paradigm(),
                "active_pbrs_field": active_pbrs_field,
                "active_pbrs_field_source": active_pbrs_field_source,
                "active_field_carryover_context": active_field_carryover_context,
                "candidate_selection_context": candidate_selection_context,
                "active_checkpoint_context": active_checkpoint_context,
                "critic_structured_diagnosis": structured_diagnosis,
                "critic_verdict": (
                    structured_diagnosis.get("verdict")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "critic_best_candidate": (
                    structured_diagnosis.get("best_candidate")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "critic_whether_to_full_run": (
                    structured_diagnosis.get("whether_to_full_run")
                    if isinstance(structured_diagnosis, dict)
                    else None
                ),
                "search_strategy_recommendation": search_strategy_recommendation,
                "validation_skipped": validation_skipped,
                "validation_skip_reason": validation_skip_reason,
                "reused_candidate_count": reused_candidate_count,
                "rerun_candidate_count": rerun_candidate_count,
                "resumed_from_stage": "critic",
            },
        )
        return round_status

    def _get_pbrs_tuning_config(self) -> Dict[str, Any]:
        if self._get_reward_paradigm() != "pbrs":
            return {}
        config = self.workflow_spec.get("pbrs_tuning")
        if not isinstance(config, dict):
            return {}
        return config

    def _get_pbrs_validation_config(self) -> Dict[str, Any]:
        if self._get_reward_paradigm() != "pbrs":
            return {}
        config = self.workflow_spec.get("pbrs_validation")
        if not isinstance(config, dict):
            return {}
        return config

    def _get_pbrs_checkpoint_plan(self) -> Dict[str, Any]:
        if self._get_reward_paradigm() != "pbrs":
            return {}
        plan = self.workflow_spec.get("pbrs_checkpoint_plan")
        if not isinstance(plan, dict):
            return {}
        return plan

    def _get_branch_plan_summary(self) -> Optional[Dict[str, Any]]:
        summary = self.workflow_spec.get("branch_plan_summary")
        return summary if isinstance(summary, dict) else None

    def _get_branch_stage_evidence(
        self,
        *,
        active_checkpoint_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return select_branch_evidence_for_checkpoint_context(
            self._get_branch_plan_summary(),
            active_checkpoint_context,
        )

    def _get_selected_pbrs_checkpoints(self) -> list[Dict[str, Any]]:
        plan = self._get_pbrs_checkpoint_plan()
        checkpoints = plan.get("selected_checkpoints")
        if not isinstance(checkpoints, list):
            return []
        return [item for item in checkpoints if isinstance(item, dict)]

    def _get_active_pbrs_checkpoint_context(
        self,
        round_id: int,
    ) -> Optional[Dict[str, Any]]:
        checkpoints = self._get_selected_pbrs_checkpoints()
        if not checkpoints:
            return None

        plan = self._get_pbrs_checkpoint_plan()
        round_to_checkpoint = plan.get("round_to_checkpoint")
        selected_checkpoint = None
        if isinstance(round_to_checkpoint, dict):
            checkpoint_name = round_to_checkpoint.get(str(round_id))
            if checkpoint_name is None:
                checkpoint_name = round_to_checkpoint.get(int(round_id))
            if checkpoint_name is not None:
                selected_checkpoint = next(
                    (
                        item
                        for item in checkpoints
                        if str(item.get("name")) == str(checkpoint_name)
                    ),
                    None,
                )

        if selected_checkpoint is None:
            checkpoint_index = int(round_id) - 1
            if 0 <= checkpoint_index < len(checkpoints):
                selected_checkpoint = checkpoints[checkpoint_index]

        if selected_checkpoint is None:
            return None

        context = deepcopy(selected_checkpoint)
        context["checkpoint_index"] = checkpoints.index(selected_checkpoint)
        context["round_id"] = int(round_id)
        return context

    def _get_effective_pbrs_validation_config(self, round_id: int) -> Dict[str, Any]:
        validation_config = deepcopy(self._get_pbrs_validation_config())
        active_checkpoint_context = self._get_active_pbrs_checkpoint_context(round_id)
        if active_checkpoint_context is None:
            return validation_config

        per_checkpoint_overrides = validation_config.pop("per_checkpoint_overrides", None)
        if isinstance(per_checkpoint_overrides, dict):
            checkpoint_override = per_checkpoint_overrides.get(
                str(active_checkpoint_context.get("name"))
            )
            if isinstance(checkpoint_override, dict):
                validation_config = self._merge_config_patch(
                    validation_config, checkpoint_override
                )

        per_stage_overrides = validation_config.pop("per_stage_overrides", None)
        if isinstance(per_stage_overrides, dict):
            stage_label = active_checkpoint_context.get("stage_label")
            if stage_label is not None:
                stage_override = per_stage_overrides.get(str(stage_label))
                if isinstance(stage_override, dict):
                    validation_config = self._merge_config_patch(
                        validation_config, stage_override
                    )

        validation_config["active_checkpoint_context"] = deepcopy(active_checkpoint_context)
        return validation_config

    def _merge_config_patch(
        self,
        base: Dict[str, Any],
        patch: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = deepcopy(base)
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = dict(merged[key])
                nested.update(deepcopy(value))
                merged[key] = nested
            else:
                merged[key] = deepcopy(value)
        return merged

    def _use_pbrs_validation_stage(self, round_id: int) -> bool:
        if self._get_reward_paradigm() != "pbrs":
            return False
        if self._get_active_pbrs_field(round_id) is None:
            return False
        return bool(self._get_effective_pbrs_validation_config(round_id).get("enabled", False))

    def _should_skip_pbrs_validation_stage(
        self,
        *,
        round_id: int,
        candidate_runs: list[Dict[str, Any]],
    ) -> bool:
        validation_config = self._get_effective_pbrs_validation_config(round_id)
        if validation_config.get("skip_if_single_candidate", True) and len(candidate_runs) <= 1:
            return True
        return False

    def _get_reward_paradigm(self) -> str:
        reward_paradigm = self.workflow_spec.get("reward_paradigm", "heuristic")
        if not isinstance(reward_paradigm, str):
            return "heuristic"
        return reward_paradigm

    def _alpha_policy_updates_enabled(self) -> bool:
        if self._get_reward_paradigm() == "pbrs":
            return False
        alpha_settings = self.workflow_spec.get("alpha_policy_settings")
        if not isinstance(alpha_settings, dict):
            return True
        return bool(alpha_settings.get("allow_llm_updates", True))

    def _get_frozen_alpha_policy(
        self,
        *,
        reward_spec: Dict[str, Any],
        base_reward_spec: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        alpha_settings = self.workflow_spec.get("alpha_policy_settings")
        if isinstance(alpha_settings, dict):
            fixed_policy = alpha_settings.get("fixed_policy")
            if isinstance(fixed_policy, dict):
                return deepcopy(fixed_policy)
        if isinstance(base_reward_spec, dict) and base_reward_spec.get("alpha_policy") is not None:
            return deepcopy(base_reward_spec["alpha_policy"])
        if reward_spec.get("alpha_policy") is not None:
            return deepcopy(reward_spec["alpha_policy"])
        return None

    def _get_pbrs_field_schedule(self) -> list[str]:
        config = self._get_pbrs_tuning_config()
        schedule = config.get("field_schedule")
        if isinstance(schedule, list) and schedule:
            return [str(item) for item in schedule]
        return []

    def _get_followup_field_priority_mode(self) -> str:
        config = self._get_pbrs_tuning_config()
        value = config.get("followup_field_priority_mode")
        if value in {"diagnosis_first", "stage_locked"}:
            return str(value)
        return "diagnosis_first"

    def _has_stage_aware_active_field_config(self) -> bool:
        tuning_config = self._get_pbrs_tuning_config()
        return any(
            isinstance(tuning_config.get(field_name), dict) and tuning_config.get(field_name)
            for field_name in ("per_stage_active_fields", "per_checkpoint_active_fields")
        )

    def _get_stage_aware_active_fields(self) -> set[str]:
        tuning_config = self._get_pbrs_tuning_config()
        supported_fields = self._get_supported_pbrs_candidate_fields()
        resolved_fields: set[str] = set()

        for field_name in ("per_stage_active_fields", "per_checkpoint_active_fields"):
            raw_mapping = tuning_config.get(field_name)
            if not isinstance(raw_mapping, dict):
                continue
            for value in raw_mapping.values():
                if isinstance(value, str) and value in supported_fields:
                    resolved_fields.add(value)

        checkpoints = self._get_selected_pbrs_checkpoints()
        for checkpoint in checkpoints:
            recommended_focus = checkpoint.get("recommended_focus")
            if not isinstance(recommended_focus, str):
                continue
            inferred_field = self._infer_pbrs_field_from_text(recommended_focus)
            if inferred_field is not None and inferred_field in supported_fields:
                resolved_fields.add(inferred_field)
        return resolved_fields

    def _resolve_active_pbrs_field(
        self,
        *,
        round_id: int,
        active_checkpoint_context: Optional[Dict[str, Any]] = None,
        previous_round: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        if active_checkpoint_context is None:
            active_checkpoint_context = self._get_active_pbrs_checkpoint_context(round_id)
        if previous_round is None:
            previous_round = self.storage.get_previous_round_artifacts(round_id)
        tuning_config = self._get_pbrs_tuning_config()
        candidate_fields = self._get_supported_pbrs_candidate_fields()
        followup_field_priority_mode = self._get_followup_field_priority_mode()

        def resolve_stage_or_checkpoint_field() -> tuple[Optional[str], Optional[str]]:
            if active_checkpoint_context is None:
                return None, None

            per_checkpoint_active_fields = tuning_config.get("per_checkpoint_active_fields")
            if isinstance(per_checkpoint_active_fields, dict):
                checkpoint_field = per_checkpoint_active_fields.get(
                    str(active_checkpoint_context.get("name"))
                )
                if isinstance(checkpoint_field, str) and checkpoint_field in candidate_fields:
                    return checkpoint_field, "per_checkpoint_active_fields"

            stage_label = active_checkpoint_context.get("stage_label")
            per_stage_active_fields = tuning_config.get("per_stage_active_fields")
            if stage_label is not None and isinstance(per_stage_active_fields, dict):
                stage_field = per_stage_active_fields.get(str(stage_label))
                if isinstance(stage_field, str) and stage_field in candidate_fields:
                    return stage_field, "per_stage_active_fields"

            recommended_focus = active_checkpoint_context.get("recommended_focus")
            inferred_field = self._infer_pbrs_field_from_text(recommended_focus)
            if inferred_field is not None and inferred_field in candidate_fields:
                return inferred_field, "checkpoint_recommended_focus"

            return None, None

        previous_structured_diagnosis = self._get_previous_structured_diagnosis(previous_round)
        previous_search_strategy = None
        if isinstance(previous_round, dict):
            critic_response = previous_round.get("critic_response")
            if isinstance(critic_response, dict):
                critic_parsed = critic_response.get("parsed")
                if isinstance(critic_parsed, dict):
                    previous_search_strategy = critic_parsed.get(
                        "search_strategy_recommendation"
                    )
        if followup_field_priority_mode == "stage_locked":
            stage_field, stage_field_source = resolve_stage_or_checkpoint_field()
            if stage_field is not None:
                return stage_field, stage_field_source
        diagnosis_patch_field = self._extract_field_from_structured_minimal_patch(
            previous_structured_diagnosis
        )
        if (
            isinstance(diagnosis_patch_field, str)
            and diagnosis_patch_field in candidate_fields
            and self._structured_diagnosis_allows_carryover(previous_structured_diagnosis)
        ):
            return (
                diagnosis_patch_field,
                "critic_structured_diagnosis_minimal_patch",
            )
        if isinstance(previous_search_strategy, dict):
            next_active_field_action = previous_search_strategy.get("next_active_field_action")
            suggested_next_active_field = previous_search_strategy.get(
                "suggested_next_active_field"
            )
            if (
                next_active_field_action == "switch_field"
                and isinstance(suggested_next_active_field, str)
                and suggested_next_active_field in candidate_fields
            ):
                return (
                    suggested_next_active_field,
                    "critic_search_strategy_next_active_field",
                )

        stage_field, stage_field_source = resolve_stage_or_checkpoint_field()
        if stage_field is not None:
            return stage_field, stage_field_source

        schedule = self._get_pbrs_field_schedule()
        if not schedule:
            return None, None
        index = int(round_id) - 1
        if index < 0 or index >= len(schedule):
            return None, None
        return schedule[index], "field_schedule"

    def _get_active_pbrs_field(self, round_id: int) -> Optional[str]:
        return self._resolve_active_pbrs_field(round_id=round_id)[0]

    def _infer_pbrs_field_from_text(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        lowered = value.lower()
        for field_name in ("beta", "wc", "wp", "gamma"):
            if field_name in lowered:
                return field_name
        return None

    def _freeze_reward_terms_during_pbrs_tuning(self) -> bool:
        config = self._get_pbrs_tuning_config()
        return bool(config.get("freeze_reward_terms", False))

    def _get_pbrs_design_update_policy(self) -> Dict[str, Any]:
        config = self._get_pbrs_tuning_config()
        policy = config.get("design_update_policy", {})
        return policy if isinstance(policy, dict) else {}

    def _merge_design_policy_layer(
        self,
        base_policy: Dict[str, Any],
        layer: Any,
    ) -> Dict[str, Any]:
        if not isinstance(layer, dict):
            return base_policy
        merged = dict(base_policy)
        for key in (
            "allow_variant_updates",
            "allow_gate_updates",
            "allow_closeness_updates",
        ):
            if key in layer:
                merged[key] = bool(layer.get(key))
        if "allowed_structural_pbrs_fields" in layer:
            fields = layer.get("allowed_structural_pbrs_fields")
            merged["allowed_structural_pbrs_fields"] = (
                [str(field_name) for field_name in fields]
                if isinstance(fields, list)
                else []
            )
        return merged

    def _get_effective_pbrs_design_update_policy(
        self,
        *,
        round_id: int,
        active_checkpoint_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        policy = self._get_pbrs_design_update_policy()
        effective = self._merge_design_policy_layer({}, policy)
        per_round_policy = policy.get("per_round_policy", {})
        if isinstance(per_round_policy, dict):
            effective = self._merge_design_policy_layer(
                effective,
                per_round_policy.get(str(round_id)),
            )
        checkpoint_context = (
            active_checkpoint_context
            if active_checkpoint_context is not None
            else self._get_active_pbrs_checkpoint_context(round_id)
        )
        stage_label = None
        checkpoint_name = None
        if isinstance(checkpoint_context, dict):
            stage_label = checkpoint_context.get("stage_label")
            checkpoint_name = checkpoint_context.get("name")
        per_stage_policy = policy.get("per_stage_policy", {})
        if isinstance(per_stage_policy, dict) and isinstance(stage_label, str):
            effective = self._merge_design_policy_layer(
                effective,
                per_stage_policy.get(stage_label),
            )
        per_checkpoint_policy = policy.get("per_checkpoint_policy", {})
        if isinstance(per_checkpoint_policy, dict) and isinstance(checkpoint_name, str):
            effective = self._merge_design_policy_layer(
                effective,
                per_checkpoint_policy.get(checkpoint_name),
            )
        effective["round_id"] = int(round_id)
        effective["active_checkpoint_context"] = deepcopy(checkpoint_context)
        return effective

    def _get_allowed_structural_pbrs_fields(
        self,
        *,
        round_id: int,
        active_checkpoint_context: Optional[Dict[str, Any]] = None,
    ) -> set[str]:
        policy = self._get_effective_pbrs_design_update_policy(
            round_id=round_id,
            active_checkpoint_context=active_checkpoint_context,
        )
        allowed_fields: set[str] = set()
        if bool(policy.get("allow_variant_updates", False)):
            allowed_fields.add("variant")
        if bool(policy.get("allow_gate_updates", False)):
            allowed_fields.update({"gate_mode", "gate_radius"})
        if bool(policy.get("allow_closeness_updates", False)):
            allowed_fields.add("closeness_mode")
        explicit_fields = policy.get("allowed_structural_pbrs_fields", [])
        if isinstance(explicit_fields, list):
            allowed_fields.update(str(field_name) for field_name in explicit_fields)
        return allowed_fields

    def _get_top_level_pbrs_keys_for_field(self, field_name: Optional[str]) -> set[str]:
        if field_name is None:
            return set()
        mapping = {
            "beta": {"beta"},
            "gamma": {"gamma"},
            "wc": {"wc"},
            "wp": {"wp"},
            "variant": {"variant"},
            "gate_mode": {"gate"},
            "gate_radius": {"gate", "semi_strict_gate_radius"},
            "closeness_mode": {"closeness"},
        }
        return set(mapping.get(str(field_name), {str(field_name)}))

    def _allow_term_updates_for_round(self, round_id: int) -> bool:
        if self._get_reward_paradigm() != "pbrs":
            return True
        if self._get_active_pbrs_field(round_id) is None:
            return True
        return not self._freeze_reward_terms_during_pbrs_tuning()

    def _get_allowed_pbrs_update_fields_for_previous_round(
        self,
        round_id: int,
    ) -> set[str]:
        if self._get_reward_paradigm() != "pbrs":
            return set()

        previous_active_field = self._get_active_pbrs_field(round_id - 1)
        allowed_fields: set[str] = set()
        if previous_active_field is not None:
            allowed_fields.add(previous_active_field)

        tuning_config = self._get_pbrs_tuning_config()
        static_fields = tuning_config.get("allowed_static_pbrs_fields", [])
        if isinstance(static_fields, list):
            allowed_fields.update(str(field_name) for field_name in static_fields)
        allowed_fields.update(self._get_allowed_structural_pbrs_fields(round_id=round_id - 1))
        return allowed_fields

    def _get_parallel_candidate_limit(self) -> int:
        config = self._get_pbrs_tuning_config()
        raw_value = config.get("max_parallel_candidates")
        if raw_value is None:
            return 1
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            return 1
        return max(1, parsed)

    def _get_supported_pbrs_candidate_fields(self) -> set[str]:
        return {
            "beta",
            "gamma",
            "wc",
            "wp",
            "variant",
            "gate_mode",
            "gate_radius",
            "closeness_mode",
        }

    def _normalize_candidate_option_value(
        self,
        *,
        field_name: str,
        value: Any,
    ) -> Optional[Any]:
        if field_name in {"beta", "gamma", "wc", "wp"}:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        if field_name == "gate_radius":
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed >= 0 else None
        if field_name == "variant":
            value = str(value)
            return value if value in SUPPORTED_PBRS_VARIANTS else None
        if field_name == "gate_mode":
            value = str(value)
            return value if value in SUPPORTED_PBRS_GATE_MODES else None
        if field_name == "closeness_mode":
            value = str(value)
            return value if value in SUPPORTED_PBRS_CLOSENESS_MODES else None
        return None

    def _get_pbrs_candidate_values(self) -> Dict[str, list[Any]]:
        config = self._get_pbrs_tuning_config()
        merged: Dict[str, list[Any]] = {}
        for key in ("candidate_values", "structural_candidate_values"):
            normalized = self._normalize_candidate_value_mapping(config.get(key))
            for field_name, values in normalized.items():
                merged[field_name] = list(values)
        return merged

    def _normalize_candidate_value_mapping(
        self,
        raw_candidate_values: Any,
    ) -> Dict[str, list[Any]]:
        if not isinstance(raw_candidate_values, dict):
            return {}
        candidate_values: Dict[str, list[Any]] = {}
        supported_fields = self._get_supported_pbrs_candidate_fields()
        for field_name, values in raw_candidate_values.items():
            normalized_field = str(field_name)
            if normalized_field not in supported_fields or not isinstance(values, list):
                continue
            normalized_values: list[Any] = []
            for value in values:
                normalized_value = self._normalize_candidate_option_value(
                    field_name=normalized_field,
                    value=value,
                )
                if normalized_value is not None and normalized_value not in normalized_values:
                    normalized_values.append(normalized_value)
            if normalized_values:
                candidate_values[normalized_field] = normalized_values
        return candidate_values

    def _get_effective_pbrs_candidate_values(
        self,
        *,
        round_id: int,
        active_pbrs_field: Optional[str],
    ) -> list[Any]:
        return self._resolve_candidate_selection_context(
            round_id=round_id,
            active_pbrs_field=active_pbrs_field,
            active_checkpoint_context=self._get_active_pbrs_checkpoint_context(round_id),
            previous_round=self.storage.get_previous_round_artifacts(round_id),
        ).get("effective_candidate_values", [])

    def _get_previous_structured_diagnosis(
        self,
        previous_round: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(previous_round, dict):
            return None
        critic_response = previous_round.get("critic_response")
        if not isinstance(critic_response, dict):
            return None
        parsed = critic_response.get("parsed")
        if not isinstance(parsed, dict):
            return None
        diagnosis = parsed.get("structured_diagnosis")
        return diagnosis if isinstance(diagnosis, dict) else None

    def _structured_diagnosis_allows_carryover(
        self,
        diagnosis: Optional[Dict[str, Any]],
    ) -> bool:
        if not isinstance(diagnosis, dict):
            return False
        verdict = diagnosis.get("verdict")
        if verdict not in {"accept", "revise"}:
            return False
        whether_to_full_run = diagnosis.get("whether_to_full_run")
        return bool(whether_to_full_run) if whether_to_full_run is not None else True

    def _extract_best_candidate_value(
        self,
        diagnosis: Optional[Dict[str, Any]],
    ) -> Optional[Any]:
        if not isinstance(diagnosis, dict):
            return None
        best_candidate = diagnosis.get("best_candidate")
        if not isinstance(best_candidate, dict):
            return None
        return best_candidate.get("candidate_value")

    def _extract_field_from_structured_minimal_patch(
        self,
        diagnosis: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if not isinstance(diagnosis, dict):
            return None
        minimal_patch = diagnosis.get("minimal_patch")
        if not isinstance(minimal_patch, dict):
            return None
        pbrs_updates = minimal_patch.get("pbrs_updates")
        if not isinstance(pbrs_updates, list) or not pbrs_updates:
            return None
        first_update = pbrs_updates[0]
        if not isinstance(first_update, dict):
            return None
        field_name = first_update.get("name")
        return str(field_name) if isinstance(field_name, str) else None

    def _resolve_candidate_selection_context(
        self,
        *,
        round_id: int,
        active_pbrs_field: Optional[str],
        active_checkpoint_context: Optional[Dict[str, Any]],
        previous_round: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if active_pbrs_field is None:
            return {
                "active_pbrs_field": None,
                "candidate_value_source": None,
                "effective_candidate_values": [],
                "base_candidate_values": [],
                "checkpoint_candidate_values": None,
                "stage_candidate_values": None,
                "critic_recommended_candidate_values": None,
                "previous_search_strategy_applied": False,
                "previous_search_strategy_source_round": None,
            }

        base_candidate_values = self._get_pbrs_candidate_values()
        candidate_values = list(base_candidate_values.get(active_pbrs_field, []))
        tuning_config = self._get_pbrs_tuning_config()
        candidate_value_source = "candidate_values"
        checkpoint_candidate_values = None
        stage_candidate_values = None
        critic_recommended_candidate_values = None
        previous_search_strategy_applied = False
        previous_search_strategy_source_round = None

        if active_checkpoint_context is not None:
            per_checkpoint_candidate_values = self._normalize_candidate_value_mapping(
                (
                    (
                        tuning_config.get("per_checkpoint_candidate_values") or {}
                    ).get(str(active_checkpoint_context.get("name")), {})
                    if isinstance(tuning_config.get("per_checkpoint_candidate_values"), dict)
                    else {}
                )
            )
            if per_checkpoint_candidate_values.get(active_pbrs_field):
                checkpoint_candidate_values = list(
                    per_checkpoint_candidate_values[active_pbrs_field]
                )
                candidate_values = list(checkpoint_candidate_values)
                candidate_value_source = "per_checkpoint_candidate_values"

            stage_label = active_checkpoint_context.get("stage_label")
            if stage_label is not None:
                per_stage_root = tuning_config.get("per_stage_candidate_values")
                if isinstance(per_stage_root, dict):
                    per_stage_candidate_values = self._normalize_candidate_value_mapping(
                        per_stage_root.get(str(stage_label), {})
                    )
                    if per_stage_candidate_values.get(active_pbrs_field):
                        stage_candidate_values = list(
                            per_stage_candidate_values[active_pbrs_field]
                        )
                        candidate_values = list(stage_candidate_values)
                        candidate_value_source = "per_stage_candidate_values"

        previous_active_field = None
        previous_structured_diagnosis = self._get_previous_structured_diagnosis(previous_round)
        diagnosis_recommended_candidate_values = None
        previous_search_strategy = None
        if isinstance(previous_round, dict):
            previous_search_strategy_source_round = previous_round.get("round_id")
            previous_status = previous_round.get("status")
            if isinstance(previous_status, dict):
                previous_active_field = previous_status.get("active_pbrs_field")
            critic_response = previous_round.get("critic_response")
            if isinstance(critic_response, dict):
                critic_parsed = critic_response.get("parsed")
                if isinstance(critic_parsed, dict):
                    previous_search_strategy = critic_parsed.get(
                        "search_strategy_recommendation"
                    )

        diagnosis_best_candidate_value = self._extract_best_candidate_value(
            previous_structured_diagnosis
        )
        if (
            previous_active_field == active_pbrs_field
            and diagnosis_best_candidate_value is not None
            and self._structured_diagnosis_allows_carryover(previous_structured_diagnosis)
        ):
            normalized_best_value = self._normalize_candidate_option_value(
                field_name=active_pbrs_field,
                value=diagnosis_best_candidate_value,
            )
            if normalized_best_value is not None:
                diagnosis_recommended_candidate_values = [normalized_best_value]
                candidate_values = list(diagnosis_recommended_candidate_values)
                candidate_value_source = "critic_structured_diagnosis"

        if (
            active_pbrs_field in {"beta", "gamma", "wc", "wp"}
            and previous_active_field == active_pbrs_field
            and diagnosis_recommended_candidate_values is None
            and isinstance(previous_search_strategy, dict)
            and previous_search_strategy.get("candidate_value_action") in {"narrow", "widen", "shift"}
        ):
            suggested_candidate_values = previous_search_strategy.get("suggested_candidate_values")
            if isinstance(suggested_candidate_values, list):
                normalized_values = []
                for value in suggested_candidate_values:
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if numeric_value not in normalized_values:
                        normalized_values.append(numeric_value)
                if normalized_values:
                    critic_recommended_candidate_values = normalized_values
                    candidate_values = list(normalized_values)
                    candidate_value_source = "critic_search_strategy_recommendation"
                    previous_search_strategy_applied = True

        return {
            "active_pbrs_field": active_pbrs_field,
            "candidate_value_source": candidate_value_source,
            "effective_candidate_values": candidate_values,
            "base_candidate_values": list(base_candidate_values.get(active_pbrs_field, [])),
            "checkpoint_candidate_values": checkpoint_candidate_values,
            "stage_candidate_values": stage_candidate_values,
            "diagnosis_recommended_candidate_values": diagnosis_recommended_candidate_values,
            "critic_recommended_candidate_values": critic_recommended_candidate_values,
            "previous_search_strategy_applied": previous_search_strategy_applied,
            "previous_search_strategy_source_round": previous_search_strategy_source_round,
        }

    def _build_search_strategy_context_summary(
        self,
        *,
        round_id: int,
        active_pbrs_field: Optional[str],
        active_pbrs_field_source: Optional[str],
        active_field_carryover_context: Optional[Dict[str, Any]],
        candidate_selection_context: Optional[Dict[str, Any]],
        active_checkpoint_context: Optional[Dict[str, Any]],
        previous_round: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        previous_search_strategy_recommendation = None
        previous_structured_diagnosis = self._get_previous_structured_diagnosis(previous_round)
        previous_active_pbrs_field = None
        if isinstance(previous_round, dict):
            previous_status = previous_round.get("status")
            if isinstance(previous_status, dict):
                previous_active_pbrs_field = previous_status.get("active_pbrs_field")
            critic_response = previous_round.get("critic_response")
            if isinstance(critic_response, dict):
                parsed = critic_response.get("parsed")
                if isinstance(parsed, dict):
                    recommendation = parsed.get("search_strategy_recommendation")
                    if isinstance(recommendation, dict):
                        previous_search_strategy_recommendation = recommendation

        checkpoint_name = None
        checkpoint_stage = None
        if isinstance(active_checkpoint_context, dict):
            checkpoint_name = active_checkpoint_context.get("name")
            checkpoint_stage = active_checkpoint_context.get("stage_label")

        candidate_value_source = None
        effective_candidate_values = []
        if isinstance(candidate_selection_context, dict):
            candidate_value_source = candidate_selection_context.get("candidate_value_source")
            effective_candidate_values = list(
                candidate_selection_context.get("effective_candidate_values", [])
            )

        return {
            "round_id": round_id,
            "active_pbrs_field": active_pbrs_field,
            "active_pbrs_field_source": active_pbrs_field_source,
            "active_field_carryover_context": active_field_carryover_context,
            "checkpoint_name": checkpoint_name,
            "checkpoint_stage_label": checkpoint_stage,
            "candidate_value_source": candidate_value_source,
            "effective_candidate_values": effective_candidate_values,
            "previous_active_pbrs_field": previous_active_pbrs_field,
            "previous_structured_diagnosis": previous_structured_diagnosis,
            "decision_priority": [
                "structured_diagnosis",
                "search_strategy_recommendation",
            ],
            "previous_search_strategy_recommendation": previous_search_strategy_recommendation,
            "previous_search_strategy_applied": bool(
                (candidate_selection_context or {}).get("previous_search_strategy_applied")
            ),
            "previous_search_strategy_source_round": (
                (candidate_selection_context or {}).get("previous_search_strategy_source_round")
            ),
        }

    def _build_critic_iteration_context(
        self,
        *,
        round_id: int,
        active_pbrs_field: Optional[str],
        active_checkpoint_context: Optional[Dict[str, Any]],
        candidate_selection_context: Optional[Dict[str, Any]],
        validation_candidate_results: Any,
        validation_summary: Any,
        candidate_results: Any,
        previous_round: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        branch_stage_evidence = self._get_branch_stage_evidence(
            active_checkpoint_context=active_checkpoint_context
        )
        effective_candidate_values = []
        if isinstance(candidate_selection_context, dict):
            effective_candidate_values = list(
                candidate_selection_context.get("effective_candidate_values", [])
            )
        checkpoint_name = None
        checkpoint_stage_label = None
        if isinstance(active_checkpoint_context, dict):
            checkpoint_name = active_checkpoint_context.get("name")
            checkpoint_stage_label = active_checkpoint_context.get("stage_label")
        return {
            "round_id": round_id,
            "active_pbrs_field": active_pbrs_field,
            "checkpoint_name": checkpoint_name,
            "checkpoint_stage_label": checkpoint_stage_label,
            "effective_candidate_values": effective_candidate_values,
            "branch_stage_evidence": deepcopy(branch_stage_evidence),
            "previous_structured_diagnosis": self._get_previous_structured_diagnosis(
                previous_round
            ),
            "validation_candidate_results": deepcopy(validation_candidate_results),
            "validation_summary": deepcopy(validation_summary),
            "candidate_results": deepcopy(candidate_results),
        }

    def _build_active_field_carryover_context(
        self,
        *,
        active_pbrs_field: Optional[str],
        active_pbrs_field_source: Optional[str],
        previous_round: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        previous_search_strategy_recommendation = None
        previous_structured_diagnosis = self._get_previous_structured_diagnosis(previous_round)
        previous_round_id = None
        if isinstance(previous_round, dict):
            previous_round_id = previous_round.get("round_id")
            critic_response = previous_round.get("critic_response")
            if isinstance(critic_response, dict):
                parsed = critic_response.get("parsed")
                if isinstance(parsed, dict):
                    recommendation = parsed.get("search_strategy_recommendation")
                    if isinstance(recommendation, dict):
                        previous_search_strategy_recommendation = recommendation

        suggested_next_active_field = None
        previous_switch_field_action = False
        if isinstance(previous_search_strategy_recommendation, dict):
            suggested_next_active_field = previous_search_strategy_recommendation.get(
                "suggested_next_active_field"
            )
            previous_switch_field_action = (
                previous_search_strategy_recommendation.get("next_active_field_action")
                == "switch_field"
            )

        return {
            "active_pbrs_field": active_pbrs_field,
            "active_pbrs_field_source": active_pbrs_field_source,
            "previous_structured_diagnosis_field": self._extract_field_from_structured_minimal_patch(
                previous_structured_diagnosis
            ),
            "previous_switch_field_action": previous_switch_field_action,
            "previous_switch_field_applied": (
                active_pbrs_field_source == "critic_search_strategy_next_active_field"
            ),
            "previous_diagnosis_field_applied": (
                active_pbrs_field_source == "critic_structured_diagnosis_minimal_patch"
            ),
            "suggested_next_active_field": suggested_next_active_field,
            "source_round_id": previous_round_id,
        }

    def _build_initial_reward_spec(self) -> Dict[str, Any]:
        reward_spec = get_default_reward_spec()
        tuning_config = self._get_pbrs_tuning_config()
        base_pbrs = tuning_config.get("base_pbrs")
        if isinstance(base_pbrs, dict):
            merged = deepcopy(reward_spec)
            merged_pbrs = dict(merged.get("pbrs", {}))
            merged_pbrs.update(base_pbrs)
            merged["pbrs"] = merged_pbrs
            return validate_reward_spec(merged)
        return reward_spec

    def _build_initial_policy_guidance_spec(self) -> Dict[str, Any]:
        guidance = get_default_policy_guidance_spec()
        base_guidance = self.workflow_spec.get("base_policy_guidance_spec")
        if isinstance(base_guidance, dict):
            merged = deepcopy(guidance)
            merged.update(base_guidance)
            return validate_policy_guidance_spec(merged)
        return guidance

    def _get_base_policy_guidance_spec_for_round(
        self,
        *,
        round_id: int,
        previous_round: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if previous_round is not None and isinstance(
            previous_round.get("policy_guidance_spec"), dict
        ):
            return validate_policy_guidance_spec(previous_round["policy_guidance_spec"])
        return validate_policy_guidance_spec(self._build_initial_policy_guidance_spec())

    def _get_base_reward_spec_for_round(
        self,
        *,
        round_id: int,
        previous_round: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if previous_round is not None and isinstance(previous_round.get("suggested_reward_spec"), dict):
            return validate_reward_spec(previous_round["suggested_reward_spec"])
        if previous_round is not None and isinstance(previous_round.get("reward_spec"), dict):
            return validate_reward_spec(previous_round["reward_spec"])
        return validate_reward_spec(self._build_initial_reward_spec())

    def _enforce_round_tuning_constraints(
        self,
        *,
        round_id: int,
        base_reward_spec: Dict[str, Any],
        candidate_reward_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        active_field = self._get_active_pbrs_field(round_id)
        if active_field is None:
            return candidate_reward_spec

        validated_base = validate_reward_spec(base_reward_spec)
        validated_candidate = validate_reward_spec(candidate_reward_spec)
        tuning_config = self._get_pbrs_tuning_config()
        base_pbrs = dict(validated_base.get("pbrs", {}))
        candidate_pbrs = dict(validated_candidate.get("pbrs", {}))
        allowed_static_fields = set(tuning_config.get("allowed_static_pbrs_fields", []))
        allowed_static_fields.update(
            self._get_allowed_structural_pbrs_fields(round_id=round_id)
        )
        allowed_top_level_keys = set()
        for field_name in allowed_static_fields:
            allowed_top_level_keys.update(self._get_top_level_pbrs_keys_for_field(field_name))
        allowed_top_level_keys.update(self._get_top_level_pbrs_keys_for_field(active_field))

        for field_name, base_value in base_pbrs.items():
            if field_name in allowed_top_level_keys:
                continue
            candidate_pbrs[field_name] = base_value
        validated_candidate["pbrs"] = candidate_pbrs

        if not self._alpha_policy_updates_enabled():
            validated_candidate["alpha_policy"] = self._get_frozen_alpha_policy(
                reward_spec=validated_candidate,
                base_reward_spec=validated_base,
            )

        if self._freeze_reward_terms_during_pbrs_tuning():
            if validated_base.get("terms") != validated_candidate.get("terms"):
                validated_candidate["terms"] = deepcopy(validated_base.get("terms", []))

        changed_fields = {
            key
            for key in set(base_pbrs) | set(candidate_pbrs)
            if base_pbrs.get(key) != candidate_pbrs.get(key)
        }
        disallowed_fields = {
            field_name
            for field_name in changed_fields
            if field_name not in allowed_top_level_keys
        }
        if disallowed_fields:
            raise ValueError(
                "Sequential PBRS tuning mode allows only the active field "
                f"{active_field!r} and explicitly allowed structural/static PBRS fields to change in round {round_id}; got disallowed changes: "
                + ", ".join(sorted(disallowed_fields))
            )
        return validated_candidate

    def _build_round_candidates(
        self,
        *,
        round_id: int,
        reward_spec: Dict[str, Any],
        reward_code: str,
        active_pbrs_field: Optional[str],
        active_checkpoint_context: Optional[Dict[str, Any]],
        candidate_selection_context: Optional[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        if active_pbrs_field is None:
            candidate_paths = self.storage.save_candidate_reward_artifacts(
                round_id,
                candidate_id="candidate_01",
                reward_spec=reward_spec,
                reward_code=reward_code,
            )
            return [
                {
                    "candidate_id": "candidate_01",
                    "active_pbrs_field": None,
                    "active_pbrs_field_source": None,
                    "candidate_value": None,
                    "candidate_selection_context": deepcopy(candidate_selection_context),
                    "active_checkpoint_context": deepcopy(active_checkpoint_context),
                    "reward_spec": reward_spec,
                    "reward_spec_path": candidate_paths.reward_spec,
                    "reward_module_path": candidate_paths.reward_function,
                }
            ]

        candidate_values = list(
            (candidate_selection_context or {}).get("effective_candidate_values", [])
        )
        if not candidate_values:
            candidate_values = [self._extract_pbrs_field_value(reward_spec, active_pbrs_field)]

        candidate_runs = []
        for index, candidate_value in enumerate(candidate_values, start=1):
            candidate_id = self._format_candidate_id(
                index=index,
                field_name=active_pbrs_field,
                candidate_value=candidate_value,
            )
            candidate_reward_spec = deepcopy(reward_spec)
            candidate_reward_spec.setdefault("pbrs", {})
            self._apply_pbrs_candidate_value(
                candidate_reward_spec,
                field_name=active_pbrs_field,
                candidate_value=candidate_value,
            )
            candidate_reward_spec = validate_reward_spec(candidate_reward_spec)
            candidate_reward_code = render_reward_code_from_spec(candidate_reward_spec)
            candidate_paths = self.storage.save_candidate_reward_artifacts(
                round_id,
                candidate_id=candidate_id,
                reward_spec=candidate_reward_spec,
                reward_code=candidate_reward_code,
            )
            candidate_runs.append(
                {
                    "candidate_id": candidate_id,
                    "active_pbrs_field": active_pbrs_field,
                    "active_pbrs_field_source": self._resolve_active_pbrs_field(
                        round_id=round_id,
                        active_checkpoint_context=active_checkpoint_context,
                    )[1],
                    "candidate_value": candidate_value,
                    "candidate_selection_context": deepcopy(candidate_selection_context),
                    "active_checkpoint_context": deepcopy(active_checkpoint_context),
                    "reward_spec": candidate_reward_spec,
                    "reward_spec_path": candidate_paths.reward_spec,
                    "reward_module_path": candidate_paths.reward_function,
                }
            )
        return candidate_runs

    def _attach_candidate_proposals_to_runs(
        self,
        *,
        candidate_runs: list[Dict[str, Any]],
        candidate_proposals: list[Dict[str, Any]],
        base_reward_spec: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        return attach_candidate_proposals_to_runs(
            candidate_runs=candidate_runs,
            candidate_proposals=candidate_proposals,
            base_reward_spec=base_reward_spec,
        )

    def _build_candidate_diff_table(
        self,
        candidate_runs: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        return build_candidate_diff_table(candidate_runs)

    def _build_candidate_comparison_table(
        self,
        *,
        validation_candidate_results: Any,
        candidate_results: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return build_candidate_comparison_table(
            validation_candidate_results=validation_candidate_results,
            candidate_results=candidate_results,
        )

    def _run_pbrs_validation_stage(
        self,
        *,
        round_id: int,
        candidate_runs: list[Dict[str, Any]],
        alpha_policy: Optional[Dict[str, Any]],
        policy_guidance_spec: Optional[Dict[str, Any]],
        active_checkpoint_context: Optional[Dict[str, Any]],
        active_field_carryover_context: Optional[Dict[str, Any]] = None,
        candidate_selection_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[list[Dict[str, Any]], Dict[str, Any], list[Dict[str, Any]]]:
        validation_config = self._get_effective_pbrs_validation_config(round_id)
        validation_results = self._run_candidate_trainings(
            round_id=round_id,
            candidate_runs=candidate_runs,
            alpha_policy=alpha_policy,
            policy_guidance_spec=policy_guidance_spec,
            active_checkpoint_context=active_checkpoint_context,
            active_field_carryover_context=active_field_carryover_context,
            candidate_selection_context=candidate_selection_context,
            phase_name="validation",
            train_overrides_override=validation_config.get("short_run_overrides"),
            persist_candidate_artifacts=False,
            allow_reuse=False,
        )
        shortlisted_ids, validation_summary = self._select_validation_shortlist(
            validation_results,
            validation_config=validation_config,
        )
        shortlisted_runs = [
            candidate_run
            for candidate_run in candidate_runs
            if str(candidate_run["candidate_id"]) in shortlisted_ids
        ]
        if not shortlisted_runs:
            shortlisted_runs = candidate_runs[:1]
        return validation_results, validation_summary, shortlisted_runs

    def _select_validation_shortlist(
        self,
        candidate_results: list[Dict[str, Any]],
        *,
        validation_config: Optional[Dict[str, Any]] = None,
    ) -> tuple[set[str], Dict[str, Any]]:
        validation_config = dict(validation_config or {})
        shortlist_size = int(validation_config.get("shortlist_size", 2))
        shortlist_size = max(1, min(shortlist_size, len(candidate_results)))
        metric_priority = validation_config.get(
            "metric_priority",
            [
                "best_test_sparse_return_mean",
                "last_test_sparse_return_mean",
                "best_test_return_mean",
                "last_test_return_mean",
                "last_sparse_return_mean",
                "last_return_mean",
            ],
        )
        ranking_metric = None
        for metric_name in metric_priority:
            if any(
                self._extract_candidate_metric_value(candidate, metric_name) is not None
                for candidate in candidate_results
            ):
                ranking_metric = metric_name
                break
        if ranking_metric is None:
            ranked_results = list(candidate_results)
        else:
            ranked_results = sorted(
                candidate_results,
                key=lambda candidate: (
                    self._extract_candidate_metric_value(candidate, ranking_metric)
                    is not None,
                    self._extract_candidate_metric_value(candidate, ranking_metric)
                    or float("-inf"),
                ),
                reverse=True,
            )
        shortlisted = ranked_results[:shortlist_size]
        shortlist_ids = {str(candidate["candidate_id"]) for candidate in shortlisted}
        summary = {
            "ranking_metric": ranking_metric,
            "shortlist_size": shortlist_size,
            "active_checkpoint_context": validation_config.get("active_checkpoint_context"),
            "shortlisted_candidate_ids": sorted(shortlist_ids),
            "ranked_candidates": [
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_value": candidate.get("candidate_value"),
                    "ranking_value": (
                        self._extract_candidate_metric_value(candidate, ranking_metric)
                        if ranking_metric is not None
                        else None
                    ),
                }
                for candidate in ranked_results
            ],
        }
        return shortlist_ids, summary

    def _extract_candidate_metric_value(
        self,
        candidate_result: Dict[str, Any],
        metric_name: Optional[str],
    ) -> Optional[float]:
        if metric_name is None:
            return None
        metrics_summary = candidate_result.get("train_metrics_summary")
        if not isinstance(metrics_summary, dict):
            return None
        metric_summary = metrics_summary.get("metric_summary", {})
        if not isinstance(metric_summary, dict):
            return None
        if metric_name.startswith("best_"):
            payload = metric_summary.get(metric_name[len("best_"):], {})
            value = payload.get("best_value") if isinstance(payload, dict) else None
        elif metric_name.startswith("last_"):
            payload = metric_summary.get(metric_name[len("last_"):], {})
            value = payload.get("last_value") if isinstance(payload, dict) else None
        else:
            payload = metric_summary.get(metric_name, {})
            value = payload.get("last_value") if isinstance(payload, dict) else None
        return float(value) if value is not None else None

    def _run_candidate_trainings(
        self,
        *,
        round_id: int,
        candidate_runs: list[Dict[str, Any]],
        alpha_policy: Optional[Dict[str, Any]],
        policy_guidance_spec: Optional[Dict[str, Any]],
        active_checkpoint_context: Optional[Dict[str, Any]] = None,
        active_field_carryover_context: Optional[Dict[str, Any]] = None,
        candidate_selection_context: Optional[Dict[str, Any]] = None,
        phase_name: str = "main",
        train_overrides_override: Optional[Dict[str, Any]] = None,
        persist_candidate_artifacts: bool = True,
        allow_reuse: bool = True,
    ) -> list[Dict[str, Any]]:
        max_workers = min(len(candidate_runs), self._get_parallel_candidate_limit())
        if max_workers <= 1:
            return [
                self._run_single_candidate_training(
                    round_id=round_id,
                    candidate_run=candidate_run,
                    alpha_policy=alpha_policy,
                    policy_guidance_spec=policy_guidance_spec,
                    active_checkpoint_context=active_checkpoint_context,
                    active_field_carryover_context=active_field_carryover_context,
                    candidate_selection_context=candidate_selection_context,
                    phase_name=phase_name,
                    train_overrides_override=train_overrides_override,
                    persist_candidate_artifacts=persist_candidate_artifacts,
                    allow_reuse=allow_reuse,
                )
                for candidate_run in candidate_runs
            ]

        results_by_candidate_id: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_candidate_id = {
                executor.submit(
                    self._run_single_candidate_training,
                    round_id=round_id,
                    candidate_run=candidate_run,
                    alpha_policy=alpha_policy,
                    policy_guidance_spec=policy_guidance_spec,
                    active_checkpoint_context=active_checkpoint_context,
                    active_field_carryover_context=active_field_carryover_context,
                    candidate_selection_context=candidate_selection_context,
                    phase_name=phase_name,
                    train_overrides_override=train_overrides_override,
                    persist_candidate_artifacts=persist_candidate_artifacts,
                    allow_reuse=allow_reuse,
                ): str(candidate_run["candidate_id"])
                for candidate_run in candidate_runs
            }
            for future in as_completed(future_to_candidate_id):
                candidate_result = future.result()
                results_by_candidate_id[str(candidate_result["candidate_id"])] = candidate_result

        return [
            results_by_candidate_id[str(candidate_run["candidate_id"])]
            for candidate_run in candidate_runs
        ]

    def _run_single_candidate_training(
        self,
        *,
        round_id: int,
        candidate_run: Dict[str, Any],
        alpha_policy: Optional[Dict[str, Any]],
        policy_guidance_spec: Optional[Dict[str, Any]],
        active_checkpoint_context: Optional[Dict[str, Any]] = None,
        active_field_carryover_context: Optional[Dict[str, Any]] = None,
        candidate_selection_context: Optional[Dict[str, Any]] = None,
        phase_name: str = "main",
        train_overrides_override: Optional[Dict[str, Any]] = None,
        persist_candidate_artifacts: bool = True,
        allow_reuse: bool = True,
    ) -> Dict[str, Any]:
        existing_result = None
        if allow_reuse and persist_candidate_artifacts and phase_name == "main":
            existing_result = self._load_existing_candidate_training_result(
                round_id=round_id,
                candidate_run=candidate_run,
            )
        if existing_result is not None:
            return existing_result

        train_output = self.train_launcher.run_training(
            round_id=round_id,
            workflow_spec=self.workflow_spec,
            reward_module_path=str(candidate_run["reward_module_path"]),
            alpha_policy=alpha_policy,
            policy_guidance_spec=policy_guidance_spec,
            workflow_id=self.workflow_id,
            candidate_id=str(candidate_run["candidate_id"]),
            reward_paradigm=self._get_reward_paradigm(),
            active_pbrs_field=candidate_run["active_pbrs_field"],
            active_pbrs_field_source=candidate_run.get("active_pbrs_field_source"),
            candidate_value=candidate_run["candidate_value"],
            active_checkpoint_context=active_checkpoint_context,
            active_field_carryover_context=active_field_carryover_context,
            candidate_selection_context=candidate_selection_context,
            phase_name=phase_name,
            train_overrides_override=train_overrides_override,
        )
        self._validate_train_output(train_output)
        if persist_candidate_artifacts and phase_name == "main":
            self.storage.record_candidate_training_run(
                round_id,
                candidate_id=str(candidate_run["candidate_id"]),
                train_config=train_output["train_config"],
                run_reference=train_output["run_reference"],
                metrics_summary=train_output.get("metrics_summary"),
            )
        return {
            "candidate_id": candidate_run["candidate_id"],
            "active_pbrs_field": candidate_run["active_pbrs_field"],
            "active_pbrs_field_source": candidate_run.get("active_pbrs_field_source"),
            "candidate_value": candidate_run["candidate_value"],
            "candidate_selection_context": deepcopy(candidate_selection_context),
            "active_checkpoint_context": deepcopy(active_checkpoint_context),
            "reward_spec": candidate_run["reward_spec"],
            "reward_module_path": str(candidate_run["reward_module_path"]),
            "train_config": train_output["train_config"],
            "run_reference": {
                "run_dir": train_output["run_reference"].get("run_dir"),
                "run_id": train_output["run_reference"].get("run_id"),
                "metrics_json": train_output["run_reference"].get("metrics_json"),
            },
            "train_metrics_summary": train_output.get("metrics_summary"),
            "phase_name": phase_name,
            "reused_existing": False,
            "_train_output": train_output,
        }

    def _load_existing_candidate_training_result(
        self,
        *,
        round_id: int,
        candidate_run: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        candidate_id = str(candidate_run["candidate_id"])
        candidate_paths = self.storage.get_candidate_paths(round_id, candidate_id)
        if not (
            candidate_paths.train_config.exists()
            and candidate_paths.train_run_ref.exists()
            and candidate_paths.train_metrics_summary.exists()
        ):
            return None

        train_config = json.loads(candidate_paths.train_config.read_text(encoding="utf-8"))
        run_reference = json.loads(candidate_paths.train_run_ref.read_text(encoding="utf-8"))
        metrics_summary = json.loads(
            candidate_paths.train_metrics_summary.read_text(encoding="utf-8")
        )
        return {
            "candidate_id": candidate_run["candidate_id"],
            "active_pbrs_field": candidate_run["active_pbrs_field"],
            "active_pbrs_field_source": candidate_run.get("active_pbrs_field_source"),
            "candidate_value": candidate_run["candidate_value"],
            "candidate_selection_context": deepcopy(
                candidate_run.get("candidate_selection_context")
            ),
            "active_checkpoint_context": deepcopy(
                candidate_run.get("active_checkpoint_context")
            ),
            "reward_spec": candidate_run["reward_spec"],
            "reward_module_path": str(candidate_run["reward_module_path"]),
            "train_config": train_config,
            "run_reference": {
                "run_dir": run_reference.get("run_dir"),
                "run_id": run_reference.get("run_id"),
                "metrics_json": run_reference.get("metrics_json"),
            },
            "train_metrics_summary": metrics_summary,
            "reused_existing": True,
            "_train_output": {
                "train_config": train_config,
                "run_reference": run_reference,
                "metrics_summary": metrics_summary,
            },
        }

    def _format_candidate_id(
        self,
        *,
        index: int,
        field_name: str,
        candidate_value: Any,
    ) -> str:
        normalized_value = str(candidate_value).replace("-", "m").replace(".", "p")
        normalized_value = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in normalized_value
        )
        return f"candidate_{index:02d}_{field_name}_{normalized_value}"

    def _extract_pbrs_field_value(
        self,
        reward_spec: Dict[str, Any],
        field_name: str,
    ) -> Any:
        pbrs = reward_spec.get("pbrs", {})
        if field_name == "gate_mode":
            gate = pbrs.get("gate") or {}
            return gate.get("mode") if isinstance(gate, dict) else None
        if field_name == "gate_radius":
            gate = pbrs.get("gate") or {}
            if isinstance(gate, dict) and gate.get("radius") is not None:
                return gate.get("radius")
            return pbrs.get("semi_strict_gate_radius")
        if field_name == "closeness_mode":
            closeness = pbrs.get("closeness") or {}
            return closeness.get("mode") if isinstance(closeness, dict) else None
        return pbrs.get(field_name)

    def _apply_pbrs_candidate_value(
        self,
        reward_spec: Dict[str, Any],
        *,
        field_name: str,
        candidate_value: Any,
    ) -> None:
        pbrs = reward_spec.setdefault("pbrs", {})
        if field_name in {"beta", "gamma", "wc", "wp"}:
            pbrs[field_name] = float(candidate_value)
            return
        if field_name == "variant":
            pbrs["variant"] = str(candidate_value)
            return
        if field_name == "gate_mode":
            gate = dict(pbrs.get("gate", {}))
            gate["mode"] = str(candidate_value)
            pbrs["gate"] = gate
            return
        if field_name == "gate_radius":
            parsed = int(candidate_value)
            gate = dict(pbrs.get("gate", {}))
            gate["radius"] = parsed
            pbrs["gate"] = gate
            pbrs["semi_strict_gate_radius"] = parsed
            return
        if field_name == "closeness_mode":
            closeness = dict(pbrs.get("closeness", {}))
            closeness["mode"] = str(candidate_value)
            pbrs["closeness"] = closeness
            return
        pbrs[field_name] = candidate_value

    def _validate_generator_output(self, output: Dict[str, Any]) -> None:
        required_keys = ["prompt", "response", "reward_spec", "candidate_proposals"]
        missing = [key for key in required_keys if key not in output]
        if missing:
            raise ValueError(
                f"Generator output is missing required keys: {', '.join(missing)}"
            )
        if not isinstance(output.get("candidate_proposals"), list) or not output["candidate_proposals"]:
            raise ValueError('generator_output["candidate_proposals"] must be a non-empty list')
        if (
            "alpha_policy" in output
            and output["alpha_policy"] is not None
            and not isinstance(output["alpha_policy"], dict)
        ):
            raise ValueError('generator_output["alpha_policy"] must be a dict when provided')
        if (
            "policy_guidance_spec" in output
            and output["policy_guidance_spec"] is not None
            and not isinstance(output["policy_guidance_spec"], dict)
        ):
            raise ValueError(
                'generator_output["policy_guidance_spec"] must be a dict when provided'
            )
        if not self._alpha_policy_updates_enabled():
            output["alpha_policy"] = None

    def _validate_train_output(self, output: Dict[str, Any]) -> None:
        required_keys = ["train_config", "run_reference"]
        missing = [key for key in required_keys if key not in output]
        if missing:
            raise ValueError(
                f"Train launcher output is missing required keys: {', '.join(missing)}"
            )

    def _validate_critic_output(self, output: Dict[str, Any]) -> None:
        required_keys = ["prompt", "response"]
        missing = [key for key in required_keys if key not in output]
        if missing:
            raise ValueError(
                f"Critic output is missing required keys: {', '.join(missing)}"
            )
        if (
            "alpha_policy" in output
            and output["alpha_policy"] is not None
            and not isinstance(output["alpha_policy"], dict)
        ):
            raise ValueError('critic_output["alpha_policy"] must be a dict when provided')
        if not self._alpha_policy_updates_enabled():
            output["alpha_policy"] = None

    def _record_round_failure(self, round_id: int, *, stage: str, error: Exception) -> None:
        self.storage.mark_round_failed(
            round_id,
            reason=str(error),
            extra={"stage": stage},
        )
        self.storage.append_timeline_event(
            "round_failed",
            {"round_id": round_id, "stage": stage, "error": str(error)},
        )
