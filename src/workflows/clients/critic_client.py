from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Protocol

from rewarding.spec_schema import get_supported_pbrs_design_space
from workflows.alpha_policy import validate_alpha_policy
from workflows.branch_critic_patch import select_branch_evidence_for_checkpoint_context
from workflows.critic_diagnosis import validate_structured_diagnosis
from workflows.policy_guidance import summarize_policy_guidance_spec
from workflows.reward_spec_budget import DEFAULT_REWARD_SPEC_CHANGE_BUDGET
from workflows.reward_update_schema import sanitize_reward_update_plan

DEFAULT_PROMPT_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "critic_prompt.md"
)
DEFAULT_PBRS_PROMPT_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "critic_prompt_pbrs.md"
)
DEFAULT_HEURISTIC_PROMPT_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "critic_prompt_heuristic.md"
)
DEFAULT_ENV_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "lbf_env_reference.md"
)


class LLMBackend(Protocol):
    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return a dict containing at least:
            text: str
        Optional fields can include:
            raw_response: dict
            model: str
            usage: dict
        """


@dataclass
class CritiqueResult:
    prompt: str
    response: Dict[str, Any]
    alpha_policy: Optional[Dict[str, Any]] = None


class TemplateBasedCriticClient:
    def __init__(
        self,
        llm_backend: LLMBackend,
        prompt_spec_path: Optional[Path | str] = None,
        env_reference_path: Optional[Path | str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm_backend = llm_backend
        self.prompt_spec_path = (
            Path(prompt_spec_path)
            if prompt_spec_path is not None
            else DEFAULT_PROMPT_SPEC_PATH
        )
        self.prompt_spec_paths = {
            "default": self.prompt_spec_path,
            "pbrs": DEFAULT_PBRS_PROMPT_SPEC_PATH,
            "heuristic": DEFAULT_HEURISTIC_PROMPT_SPEC_PATH,
        }
        self.env_reference_path = (
            Path(env_reference_path)
            if env_reference_path is not None
            else DEFAULT_ENV_REFERENCE_PATH
        )
        self.system_prompt = system_prompt or self._default_system_prompt()

    def critique(
        self,
        *,
        round_id: int,
        workflow_spec: Dict[str, Any],
        current_round: Dict[str, Any],
        previous_round: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = self.build_prompt(
            round_id=round_id,
            workflow_spec=workflow_spec,
            current_round=current_round,
            previous_round=previous_round,
        )
        llm_result = self.llm_backend.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            metadata={
                "round_id": round_id,
                "workflow_id": workflow_spec.get("workflow_id"),
                "role": "critic",
                "workflow_spec": workflow_spec,
                "current_round": current_round,
                "previous_round": previous_round,
            },
        )
        parsed = self.parse_response(llm_result["text"])
        response = {
            "raw_text": llm_result["text"],
            "raw_response": llm_result.get("raw_response"),
            "model": llm_result.get("model"),
            "usage": llm_result.get("usage"),
            "parsed": parsed,
        }
        return CritiqueResult(
            prompt=prompt,
            response=response,
            alpha_policy=parsed.get("alpha_policy"),
        ).__dict__

    def build_prompt(
        self,
        *,
        round_id: int,
        workflow_spec: Dict[str, Any],
        current_round: Dict[str, Any],
        previous_round: Optional[Dict[str, Any]],
    ) -> str:
        prompt_spec = self._resolve_prompt_spec_path(workflow_spec).read_text(encoding="utf-8")
        env_reference = self.env_reference_path.read_text(encoding="utf-8")
        reward_paradigm = self._get_reward_paradigm(workflow_spec)
        alpha_updates_enabled = self._alpha_policy_updates_enabled(workflow_spec)
        pbrs_tuning = (
            workflow_spec.get("pbrs_tuning", {}) if reward_paradigm == "pbrs" else {}
        )
        pbrs_design_space = get_supported_pbrs_design_space()
        candidate_values = pbrs_tuning.get("candidate_values", {})
        active_checkpoint_context = self._get_active_checkpoint_context(
            workflow_spec=workflow_spec,
            round_id=round_id,
        )
        pbrs_design_update_policy = self._get_effective_design_update_policy(
            workflow_spec=workflow_spec,
            round_id=round_id,
            active_checkpoint_context=active_checkpoint_context,
        )
        active_pbrs_field = self._get_active_pbrs_field(
            workflow_spec=workflow_spec,
            round_id=round_id,
            active_checkpoint_context=active_checkpoint_context,
            previous_round=previous_round,
        )
        effective_candidate_values = self._get_effective_candidate_values(
            workflow_spec=workflow_spec,
            round_id=round_id,
            active_pbrs_field=active_pbrs_field,
            previous_round=previous_round,
        )
        branch_critic_recommendation = self._get_branch_critic_recommendation(
            workflow_spec=workflow_spec
        )
        branch_critic_patch = self._get_branch_critic_patch(workflow_spec=workflow_spec)
        branch_plan_summary = self._get_branch_plan_summary(workflow_spec=workflow_spec)
        branch_stage_evidence = self._get_branch_stage_evidence(
            workflow_spec=workflow_spec,
            active_checkpoint_context=active_checkpoint_context,
        )
        branch_design_transition = self._get_branch_design_transition(
            branch_critic_recommendation=branch_critic_recommendation,
            branch_critic_patch=branch_critic_patch,
        )
        active_field_carryover_context = self._build_active_field_carryover_context(
            active_pbrs_field=active_pbrs_field,
            previous_round=previous_round,
        )

        prompt_sections = [
            "Prompt specification:",
            prompt_spec,
            "",
            f"Current round: {round_id}",
            "",
            "Task description:",
            workflow_spec.get("task_description", "No task description provided."),
            "",
            "Environment description:",
            workflow_spec.get("environment_description", "No environment description provided."),
            "",
            f"Reward paradigm for this workflow: {reward_paradigm}",
            "",
            "LBF environment reference:",
            env_reference,
            "",
            "Current round artifacts:",
            json.dumps(current_round, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Current policy guidance summary:",
            json.dumps(
                summarize_policy_guidance_spec(current_round.get("policy_guidance_spec")),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Diagnosis-first reminder:",
            "Base your main decision on candidate diffs, candidate comparison evidence, and the required structured_diagnosis fields.",
            "Use search_strategy_recommendation only as a secondary helper for follow-up narrowing after you form the diagnosis.",
            "",
            "Candidate diff table for this round:",
            json.dumps(
                current_round.get("candidate_diffs"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Candidate comparison table for this round:",
            json.dumps(
                current_round.get("candidate_comparison_table"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Previous round structured diagnosis:",
            json.dumps(
                (
                    ((previous_round or {}).get("critic_response") or {}).get("parsed", {})
                    if isinstance(previous_round, dict)
                    else {}
                ).get("structured_diagnosis"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Search strategy context summary for this round:",
            json.dumps(
                current_round.get("search_strategy_context_summary"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Critic iteration context for this round:",
            json.dumps(
                current_round.get("critic_iteration_context"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "PBRS tuning configuration:",
            json.dumps(pbrs_tuning, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Supported PBRS design space:",
            json.dumps(pbrs_design_space, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "PBRS design update policy:",
            json.dumps(
                pbrs_design_update_policy,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Active PBRS checkpoint context for this round:",
            json.dumps(active_checkpoint_context, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Active field carryover context for this round:",
            json.dumps(active_field_carryover_context, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Branch-level PBRS recommendation carried into this workflow:",
            json.dumps(
                branch_critic_recommendation,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Branch-level design transition recommendation carried into this workflow:",
            json.dumps(
                branch_design_transition,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Attached branch-critic workflow patch metadata:",
            json.dumps(branch_critic_patch, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Attached branch-plan summary:",
            json.dumps(branch_plan_summary, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Branch-stage evidence for this round:",
            json.dumps(branch_stage_evidence, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Important note about reward structure:",
            "The reward_spec field is the source of truth for the current dense reward design.",
            "Treat reward_code as a rendered execution artifact derived from that structure.",
            (
                "When proposing changes, focus on PBRS coefficient choices rather than free-form code rewrites."
                if reward_paradigm == "pbrs"
                else "When proposing changes, focus on reward terms, trigger variants, weights, and alpha policy rather than free-form code rewrites."
            ),
            "",
            "Important note about train_metrics_summary:",
            "The train_metrics_summary field is a compressed view of raw experimental data.",
            "It contains metric values and sampled points only, not interpreted conclusions.",
            "You must analyze this raw evidence to identify issues, patterns, and next-step changes.",
        ]

        if previous_round:
            prompt_sections.extend(
                [
                    "",
                    "Previous round artifacts:",
                    json.dumps(previous_round, ensure_ascii=False, indent=2, sort_keys=True),
                ]
            )

        if active_pbrs_field is not None:
            prompt_sections.extend(
                [
                    "",
                    f"Active PBRS coefficient in this round: {active_pbrs_field}",
                    "This round may contain multiple candidate experiments for that coefficient.",
                ]
            )
        if active_pbrs_field is not None and isinstance(candidate_values, dict):
            field_candidates = (
                effective_candidate_values
                if effective_candidate_values
                else candidate_values.get(active_pbrs_field)
            )
            if isinstance(field_candidates, list) and field_candidates:
                prompt_sections.extend(
                    [
                        f"Candidate values evaluated in this round: {json.dumps(field_candidates, ensure_ascii=False)}",
                        "These values may reflect stage-aware candidate overrides from the active checkpoint context.",
                        "Use the candidate_results evidence to choose the best value for the active coefficient.",
                        "When you emit pbrs_updates, target the current round's active PBRS coefficient only.",
                        "Prefer direction='set' with the chosen candidate value.",
                    ]
                )
        elif active_pbrs_field is not None:
            prompt_sections.extend(
                [
                    "When you emit pbrs_updates, target the current round's active PBRS coefficient only.",
                ]
            )

        if reward_paradigm == "pbrs":
            prompt_sections.extend(
                [
                    "",
                    "PBRS workflow mode notes:",
                    "Choose the best PBRS coefficient value from the candidate experiments for the active field.",
                    "Do not emit alpha_policy in this mode; alpha updates are disabled.",
                    "Do not propose heuristic term redesign unless the workflow explicitly allows term changes.",
                    "If active checkpoint context is present, use it to judge whether a candidate is appropriate for this training stage rather than only globally.",
                    "If a branch-level PBRS recommendation is present, treat it as prior checkpoint-branch evidence that should be weighed against the current round's direct results.",
                ]
            )
        elif not alpha_updates_enabled:
            prompt_sections.extend(
                [
                    "",
                    "Alpha-policy updates are disabled for this workflow. Do not emit alpha_policy.",
                ]
            )

        prompt_sections.extend(
            [
                "",
                "Analyze the current round and produce the required structured Critic output now.",
            ]
        )
        return "\n".join(prompt_sections)

    def parse_response(self, text: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Critic output is not valid JSON: {exc}") from exc

        required_keys = [
            "analysis_summary",
            "reward_issues",
            "preserve_elements",
            "improvement_suggestions",
            "reward_update_plan",
            "structured_diagnosis",
        ]
        missing = [key for key in required_keys if key not in parsed]
        if missing:
            raise ValueError(
                f"Critic output is missing required keys: {', '.join(missing)}"
            )

        if not isinstance(parsed["analysis_summary"], str):
            raise ValueError('Critic "analysis_summary" must be a string')
        if not isinstance(parsed["reward_issues"], list):
            raise ValueError('Critic "reward_issues" must be a list')
        if not isinstance(parsed["preserve_elements"], list):
            raise ValueError('Critic "preserve_elements" must be a list')
        if not isinstance(parsed["improvement_suggestions"], list):
            raise ValueError('Critic "improvement_suggestions" must be a list')
        parsed["structured_diagnosis"] = validate_structured_diagnosis(
            parsed["structured_diagnosis"]
        )
        parsed["reward_update_plan"], warnings = sanitize_reward_update_plan(
            parsed["reward_update_plan"],
            budget=DEFAULT_REWARD_SPEC_CHANGE_BUDGET,
        )
        parsed["reward_update_plan_warnings"] = warnings
        parsed["search_strategy_recommendation"] = self._validate_search_strategy_recommendation(
            parsed.get("search_strategy_recommendation")
        )

        alpha_policy = parsed.get("alpha_policy")
        if alpha_policy is not None:
            validate_alpha_policy(alpha_policy)

        return parsed

    def _resolve_prompt_spec_path(self, workflow_spec: Dict[str, Any]) -> Path:
        reward_paradigm = self._get_reward_paradigm(workflow_spec)
        return self.prompt_spec_paths.get(reward_paradigm, self.prompt_spec_paths["default"])

    def _get_reward_paradigm(self, workflow_spec: Dict[str, Any]) -> str:
        reward_paradigm = workflow_spec.get("reward_paradigm", "heuristic")
        if reward_paradigm not in {"pbrs", "heuristic"}:
            return "heuristic"
        return reward_paradigm

    def _alpha_policy_updates_enabled(self, workflow_spec: Dict[str, Any]) -> bool:
        settings = workflow_spec.get("alpha_policy_settings")
        if not isinstance(settings, dict):
            return True
        return bool(settings.get("allow_llm_updates", True))

    def _get_branch_critic_recommendation(
        self,
        *,
        workflow_spec: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        value = workflow_spec.get("branch_critic_recommendation")
        return value if isinstance(value, dict) else None

    def _get_branch_critic_patch(
        self,
        *,
        workflow_spec: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        value = workflow_spec.get("branch_critic_patch")
        return value if isinstance(value, dict) else None

    def _get_branch_plan_summary(
        self,
        *,
        workflow_spec: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        value = workflow_spec.get("branch_plan_summary")
        return value if isinstance(value, dict) else None

    def _get_branch_stage_evidence(
        self,
        *,
        workflow_spec: Dict[str, Any],
        active_checkpoint_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return select_branch_evidence_for_checkpoint_context(
            self._get_branch_plan_summary(workflow_spec=workflow_spec),
            active_checkpoint_context,
        )

    def _get_branch_design_transition(
        self,
        *,
        branch_critic_recommendation: Optional[Dict[str, Any]],
        branch_critic_patch: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if isinstance(branch_critic_recommendation, dict):
            value = branch_critic_recommendation.get("recommended_design_transition")
            if isinstance(value, dict):
                return value
        if isinstance(branch_critic_patch, dict):
            value = branch_critic_patch.get("recommended_design_transition")
            if isinstance(value, dict):
                return value
        return {}

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

    def _get_effective_design_update_policy(
        self,
        *,
        workflow_spec: Dict[str, Any],
        round_id: int,
        active_checkpoint_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self._get_reward_paradigm(workflow_spec) != "pbrs":
            return {}
        pbrs_tuning = workflow_spec.get("pbrs_tuning", {})
        policy = (
            pbrs_tuning.get("design_update_policy", {})
            if isinstance(pbrs_tuning, dict)
            else {}
        )
        effective = self._merge_design_policy_layer({}, policy)
        per_round_policy = policy.get("per_round_policy", {})
        if isinstance(per_round_policy, dict):
            effective = self._merge_design_policy_layer(
                effective,
                per_round_policy.get(str(round_id)),
            )
        stage_label = None
        checkpoint_name = None
        if isinstance(active_checkpoint_context, dict):
            stage_label = active_checkpoint_context.get("stage_label")
            checkpoint_name = active_checkpoint_context.get("name")
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
        return effective

    def _get_active_checkpoint_context(
        self,
        *,
        workflow_spec: Dict[str, Any],
        round_id: int,
    ) -> Optional[Dict[str, Any]]:
        if self._get_reward_paradigm(workflow_spec) != "pbrs":
            return None
        checkpoint_plan = workflow_spec.get("pbrs_checkpoint_plan")
        if not isinstance(checkpoint_plan, dict):
            return None
        selected = checkpoint_plan.get("selected_checkpoints")
        if not isinstance(selected, list):
            return None
        checkpoints = [item for item in selected if isinstance(item, dict)]
        if not checkpoints:
            return None

        round_to_checkpoint = checkpoint_plan.get("round_to_checkpoint")
        checkpoint = None
        if isinstance(round_to_checkpoint, dict):
            checkpoint_name = round_to_checkpoint.get(str(round_id))
            if checkpoint_name is None:
                checkpoint_name = round_to_checkpoint.get(int(round_id))
            if checkpoint_name is not None:
                checkpoint = next(
                    (
                        item
                        for item in checkpoints
                        if str(item.get("name")) == str(checkpoint_name)
                    ),
                    None,
                )
        if checkpoint is None:
            checkpoint_index = int(round_id) - 1
            if 0 <= checkpoint_index < len(checkpoints):
                checkpoint = checkpoints[checkpoint_index]
        if checkpoint is None:
            return None

        result = dict(checkpoint)
        result["checkpoint_index"] = checkpoints.index(checkpoint)
        result["round_id"] = int(round_id)
        return result

    def _get_active_pbrs_field(
        self,
        *,
        workflow_spec: Dict[str, Any],
        round_id: int,
        active_checkpoint_context: Optional[Dict[str, Any]],
        previous_round: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        pbrs_tuning = workflow_spec.get("pbrs_tuning", {})
        candidate_fields = set(
            self._normalize_candidate_value_mapping(
                pbrs_tuning.get("candidate_values", {})
            ).keys()
        )

        previous_structured_diagnosis = self._get_previous_structured_diagnosis(previous_round)
        previous_search_strategy_recommendation = None
        if isinstance(previous_round, dict):
            critic_response = previous_round.get("critic_response")
            if isinstance(critic_response, dict):
                parsed = critic_response.get("parsed")
                if isinstance(parsed, dict):
                    previous_search_strategy_recommendation = parsed.get(
                        "search_strategy_recommendation"
                    )
        diagnosis_patch_field = self._extract_field_from_structured_minimal_patch(
            previous_structured_diagnosis
        )
        if (
            isinstance(diagnosis_patch_field, str)
            and diagnosis_patch_field in candidate_fields
            and self._structured_diagnosis_allows_carryover(previous_structured_diagnosis)
        ):
            return diagnosis_patch_field
        if isinstance(previous_search_strategy_recommendation, dict):
            if (
                previous_search_strategy_recommendation.get("next_active_field_action")
                == "switch_field"
            ):
                suggested_next_active_field = previous_search_strategy_recommendation.get(
                    "suggested_next_active_field"
                )
                if (
                    isinstance(suggested_next_active_field, str)
                    and suggested_next_active_field in candidate_fields
                ):
                    return suggested_next_active_field

        if active_checkpoint_context is not None:
            per_checkpoint_active_fields = pbrs_tuning.get("per_checkpoint_active_fields")
            if isinstance(per_checkpoint_active_fields, dict):
                checkpoint_field = per_checkpoint_active_fields.get(
                    str(active_checkpoint_context.get("name"))
                )
                if isinstance(checkpoint_field, str) and checkpoint_field in candidate_fields:
                    return checkpoint_field

            stage_label = active_checkpoint_context.get("stage_label")
            per_stage_active_fields = pbrs_tuning.get("per_stage_active_fields")
            if stage_label is not None and isinstance(per_stage_active_fields, dict):
                stage_field = per_stage_active_fields.get(str(stage_label))
                if isinstance(stage_field, str) and stage_field in candidate_fields:
                    return stage_field

            recommended_focus = active_checkpoint_context.get("recommended_focus")
            inferred_field = self._infer_pbrs_field_from_text(recommended_focus)
            if inferred_field is not None and inferred_field in candidate_fields:
                return inferred_field

        field_schedule = pbrs_tuning.get("field_schedule")
        if isinstance(field_schedule, list) and 0 < round_id <= len(field_schedule):
            return str(field_schedule[round_id - 1])
        return None

    def _normalize_candidate_value_mapping(
        self,
        raw_candidate_values: Any,
    ) -> Dict[str, list[float]]:
        if not isinstance(raw_candidate_values, dict):
            return {}
        candidate_values: Dict[str, list[float]] = {}
        for field_name, values in raw_candidate_values.items():
            if not isinstance(values, list):
                continue
            normalized_values = []
            for value in values:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric_value not in normalized_values:
                    normalized_values.append(numeric_value)
            if normalized_values:
                candidate_values[str(field_name)] = normalized_values
        return candidate_values

    def _infer_pbrs_field_from_text(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        lowered = value.lower()
        for field_name in ("beta", "wc", "wp", "gamma"):
            if field_name in lowered:
                return field_name
        return None

    def _get_effective_candidate_values(
        self,
        *,
        workflow_spec: Dict[str, Any],
        round_id: int,
        active_pbrs_field: Optional[str],
        previous_round: Optional[Dict[str, Any]] = None,
    ) -> list[float]:
        if active_pbrs_field is None:
            return []

        pbrs_tuning = workflow_spec.get("pbrs_tuning", {})
        candidate_values = self._normalize_candidate_value_mapping(
            pbrs_tuning.get("candidate_values", {})
        ).get(active_pbrs_field, [])
        active_checkpoint_context = self._get_active_checkpoint_context(
            workflow_spec=workflow_spec,
            round_id=round_id,
        )
        if active_checkpoint_context is None:
            return list(candidate_values)

        per_checkpoint_root = pbrs_tuning.get("per_checkpoint_candidate_values")
        if isinstance(per_checkpoint_root, dict):
            checkpoint_values = self._normalize_candidate_value_mapping(
                per_checkpoint_root.get(str(active_checkpoint_context.get("name")), {})
            ).get(active_pbrs_field)
            if checkpoint_values:
                candidate_values = checkpoint_values

        stage_label = active_checkpoint_context.get("stage_label")
        per_stage_root = pbrs_tuning.get("per_stage_candidate_values")
        if stage_label is not None and isinstance(per_stage_root, dict):
            stage_values = self._normalize_candidate_value_mapping(
                per_stage_root.get(str(stage_label), {})
            ).get(active_pbrs_field)
            if stage_values:
                candidate_values = stage_values

        previous_structured_diagnosis = self._get_previous_structured_diagnosis(previous_round)
        previous_active_field = None
        if isinstance(previous_round, dict):
            previous_status = previous_round.get("status")
            if isinstance(previous_status, dict):
                previous_active_field = previous_status.get("active_pbrs_field")
        diagnosis_best_candidate_value = self._extract_best_candidate_value(
            previous_structured_diagnosis
        )
        if (
            previous_active_field == active_pbrs_field
            and diagnosis_best_candidate_value is not None
            and self._structured_diagnosis_allows_carryover(previous_structured_diagnosis)
        ):
            try:
                return [float(diagnosis_best_candidate_value)]
            except (TypeError, ValueError):
                pass

        return list(candidate_values)

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

    def _default_system_prompt(self) -> str:
        return (
            "You are the Critic LLM in a multi-round reward optimization workflow. "
            "You analyze structured reward specifications and training outcomes, then return "
            "strictly valid JSON with concrete issues, preserve signals, improvement directions, "
            "and updated alpha policy when needed."
        )

    def _validate_search_strategy_recommendation(
        self,
        value: Any,
    ) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("Critic search_strategy_recommendation must be a dict when provided")

        valid_candidate_actions = {"keep", "narrow", "widen", "shift"}
        valid_next_field_actions = {"keep_current", "switch_field", "no_opinion"}
        validated: Dict[str, Any] = {}

        candidate_value_action = value.get("candidate_value_action")
        if candidate_value_action is not None:
            if candidate_value_action not in valid_candidate_actions:
                raise ValueError(
                    "Critic search_strategy_recommendation.candidate_value_action must be one of: "
                    + ", ".join(sorted(valid_candidate_actions))
                )
            validated["candidate_value_action"] = candidate_value_action

        suggested_candidate_values = value.get("suggested_candidate_values")
        if suggested_candidate_values is not None:
            if not isinstance(suggested_candidate_values, list):
                raise ValueError(
                    "Critic search_strategy_recommendation.suggested_candidate_values must be a list"
                )
            normalized_values = []
            for index, item in enumerate(suggested_candidate_values):
                if not isinstance(item, (int, float)):
                    raise ValueError(
                        "Critic search_strategy_recommendation.suggested_candidate_values"
                        f"[{index}] must be numeric"
                    )
                value_float = float(item)
                if value_float not in normalized_values:
                    normalized_values.append(value_float)
            validated["suggested_candidate_values"] = normalized_values

        next_active_field_action = value.get("next_active_field_action")
        if next_active_field_action is not None:
            if next_active_field_action not in valid_next_field_actions:
                raise ValueError(
                    "Critic search_strategy_recommendation.next_active_field_action must be one of: "
                    + ", ".join(sorted(valid_next_field_actions))
                )
            validated["next_active_field_action"] = next_active_field_action

        suggested_next_active_field = value.get("suggested_next_active_field")
        if suggested_next_active_field is not None:
            if suggested_next_active_field not in {"beta", "wc", "wp", "gamma"}:
                raise ValueError(
                    "Critic search_strategy_recommendation.suggested_next_active_field must be one of: "
                    "beta, gamma, wc, wp"
                )
            validated["suggested_next_active_field"] = suggested_next_active_field

        rationale = value.get("rationale")
        if rationale is not None:
            if not isinstance(rationale, str):
                raise ValueError(
                    "Critic search_strategy_recommendation.rationale must be a string when provided"
                )
            validated["rationale"] = rationale

        return validated or None

    def _build_active_field_carryover_context(
        self,
        *,
        active_pbrs_field: Optional[str],
        previous_round: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        previous_search_strategy_recommendation = None
        previous_round_id = previous_round.get("round_id") if isinstance(previous_round, dict) else None
        if isinstance(previous_round, dict):
            critic_response = previous_round.get("critic_response")
            if isinstance(critic_response, dict):
                parsed = critic_response.get("parsed")
                if isinstance(parsed, dict):
                    previous_search_strategy_recommendation = parsed.get(
                        "search_strategy_recommendation"
                    )
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
            "previous_switch_field_action": previous_switch_field_action,
            "previous_switch_field_applied": (
                previous_switch_field_action and suggested_next_active_field == active_pbrs_field
            ),
            "suggested_next_active_field": suggested_next_active_field,
            "source_round_id": previous_round_id,
        }


class StubCriticBackend:
    """
    Deterministic backend for workflow development before wiring a real Critic LLM.
    """

    def _ranking_fields(self) -> list[str]:
        return [
            "best_test_sparse_return_mean",
            "last_test_sparse_return_mean",
            "best_test_return_mean",
            "last_test_return_mean",
            "last_mixed_return_mean",
            "last_sparse_return_mean",
            "last_return_mean",
        ]

    def _choose_best_record(self, records: list[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not records:
            return None, None
        for field_name in self._ranking_fields():
            values = [record.get(field_name) for record in records if record.get(field_name) is not None]
            if len(values) < 2:
                continue
            distinct = {round(float(value), 12) for value in values}
            if len(distinct) > 1:
                ranked = sorted(
                    records,
                    key=lambda record: (
                        record.get(field_name) is not None,
                        record.get(field_name) or float("-inf"),
                    ),
                    reverse=True,
                )
                return ranked[0], field_name
        for field_name in self._ranking_fields():
            ranked = [record for record in records if record.get(field_name) is not None]
            if ranked:
                ranked = sorted(
                    ranked,
                    key=lambda record: record.get(field_name) or float("-inf"),
                    reverse=True,
                )
                return ranked[0], field_name
        return records[0], None

    def _narrow_numeric_candidates(
        self,
        records: list[Dict[str, Any]],
        best_value: Any,
    ) -> list[float]:
        numeric_values = []
        for record in records:
            try:
                value = float(record.get("candidate_value"))
            except (TypeError, ValueError):
                continue
            if value not in numeric_values:
                numeric_values.append(value)
        numeric_values = sorted(numeric_values)
        try:
            best_numeric = float(best_value)
        except (TypeError, ValueError):
            return numeric_values[:3]
        if best_numeric not in numeric_values:
            numeric_values.append(best_numeric)
            numeric_values.sort()
        index = numeric_values.index(best_numeric)
        start = max(0, index - 1)
        end = min(len(numeric_values), index + 2)
        window = numeric_values[start:end]
        if len(window) == 2:
            if start > 0:
                window = numeric_values[start - 1 : end]
            elif end < len(numeric_values):
                window = numeric_values[start : end + 1]
        return window or [best_numeric]

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = metadata or {}
        current_round = metadata.get("current_round") if isinstance(metadata, dict) else None
        critic_iteration_context = (
            current_round.get("critic_iteration_context")
            if isinstance(current_round, dict)
            else None
        )
        branch_stage_evidence = (
            current_round.get("branch_stage_evidence")
            if isinstance(current_round, dict)
            else None
        )
        candidate_results = (
            critic_iteration_context.get("candidate_results")
            if isinstance(critic_iteration_context, dict)
            else None
        )
        if not isinstance(candidate_results, list):
            candidate_results = (
                current_round.get("candidate_results")
                if isinstance(current_round, dict)
                else None
            )
        if not isinstance(branch_stage_evidence, dict) and isinstance(critic_iteration_context, dict):
            branch_stage_evidence = critic_iteration_context.get("branch_stage_evidence")
        active_field_match = re.search(
            r"Active PBRS coefficient in this round:\s*([A-Za-z_]+)",
            user_prompt,
        )
        candidate_values_match = re.search(
            r"Candidate values evaluated in this round:\s*(\[[^\n]+\])",
            user_prompt,
        )
        active_field = active_field_match.group(1) if active_field_match else "beta"
        chosen_candidate = None
        search_action = "keep"
        analysis_summary = "Stub critic response for workflow integration testing."
        reward_issues = [
            "The current dense reward has not been evaluated by a real LLM yet."
        ]
        improvement_suggestions = [
            "Replace the stub critic backend with a real Critic model.",
            "Prefer small structured adjustments to terms, trigger variants, or weights."
        ]
        rationale = "Stub critic leaves the workflow search strategy unchanged."

        if isinstance(candidate_results, list) and candidate_results:
            best_record, ranking_metric = self._choose_best_record(candidate_results)
            if isinstance(best_record, dict):
                chosen_candidate = best_record.get("candidate_value")
                analysis_summary = (
                    "Fallback stub critic selected the strongest current-round candidate "
                    f"using {ranking_metric or 'available metrics'}."
                )
                reward_issues = [
                    "Real Critic output was unavailable, so the fallback critic used current-round metrics."
                ]
                improvement_suggestions = [
                    "Use the best observed candidate as the next default.",
                    "If evidence is still weak, keep future changes local to the same PBRS field.",
                ]
                if len(candidate_results) > 1:
                    narrowed = self._narrow_numeric_candidates(candidate_results, chosen_candidate)
                    if len(narrowed) >= 2:
                        search_action = "narrow"
                        rationale = (
                            "Fallback critic narrowed the next sweep around the best current-round candidate."
                        )
                    else:
                        narrowed = None
                else:
                    narrowed = None
            else:
                narrowed = None
        else:
            narrowed = None

        if chosen_candidate is None and isinstance(branch_stage_evidence, dict):
            best_branch_candidate = branch_stage_evidence.get("best_candidate") or {}
            chosen_candidate = best_branch_candidate.get("candidate_value")
            analysis_summary = (
                "Fallback stub critic used attached branch-stage evidence because "
                "current-round comparison evidence was unavailable or inconclusive."
            )
            reward_issues = [
                "Real Critic output was unavailable, so the fallback critic relied on prior branch-stage evidence."
            ]
            improvement_suggestions = [
                "Preserve the stage-specific branching preference unless new direct evidence overturns it.",
                "Use the selected stage-specific value as the default for the next local update.",
            ]
            rationale = "Fallback critic carried forward the strongest attached branch-stage preference."

        if candidate_values_match:
            try:
                parsed_candidates = json.loads(candidate_values_match.group(1))
                if chosen_candidate is None and isinstance(parsed_candidates, list) and parsed_candidates:
                    chosen_candidate = parsed_candidates[len(parsed_candidates) // 2]
            except json.JSONDecodeError:
                chosen_candidate = None
        payload = {
            "analysis_summary": analysis_summary,
            "reward_issues": reward_issues,
            "preserve_elements": [
                "Keep the current runtime and workflow integration structure."
            ],
            "improvement_suggestions": improvement_suggestions,
            "structured_diagnosis": {
                "best_candidate": (
                    {
                        "candidate_id": (
                            best_record.get("candidate_id")
                            if isinstance(locals().get("best_record"), dict)
                            else None
                        ),
                        "candidate_value": chosen_candidate,
                    }
                    if chosen_candidate is not None
                    else None
                ),
                "verdict": "accept" if chosen_candidate is not None else "uncertain",
                "evidence_summary": analysis_summary,
                "failure_attribution": reward_issues,
                "over_shaping_risk": (
                    "moderate"
                    if isinstance(chosen_candidate, (int, float)) and float(chosen_candidate) > 0.5
                    else "low"
                ),
                "minimal_patch": {
                    "pbrs_updates": (
                        [
                            {
                                "name": active_field,
                                "direction": "set",
                                "value": chosen_candidate,
                            }
                        ]
                        if chosen_candidate is not None
                        else []
                    ),
                },
                "whether_to_full_run": bool(chosen_candidate is not None),
            },
            "reward_update_plan": {
                "preserve_terms": ["approach_target_food", "coordination_ready"],
                "disable_terms": [],
                "weight_updates": [
                    {"name": "approach_target_food", "direction": "decrease"},
                    {"name": "coordination_ready", "direction": "keep"},
                ],
                "trigger_variant_updates": [],
                "pbrs_updates": (
                    [
                        {
                            "name": active_field,
                            "direction": "set",
                            "value": chosen_candidate,
                        }
                    ]
                    if chosen_candidate is not None
                    else []
                ),
            },
            "search_strategy_recommendation": {
                "candidate_value_action": search_action,
                "suggested_candidate_values": narrowed if search_action == "narrow" else None,
                "next_active_field_action": "keep_current",
                "rationale": rationale,
            },
        }
        return {
            "text": json.dumps(payload, ensure_ascii=False),
            "raw_response": {"stub": True, "metadata": metadata},
            "model": "stub-critic",
            "usage": None,
        }
