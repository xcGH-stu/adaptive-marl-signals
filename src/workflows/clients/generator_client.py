from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from rewarding.spec_schema import (
    get_default_reward_spec,
    get_supported_pbrs_design_space,
    get_supported_reward_terms,
    validate_reward_spec,
)
from workflows.alpha_policy import validate_alpha_policy
from workflows.branch_critic_patch import select_branch_evidence_for_checkpoint_context
from workflows.policy_guidance import (
    get_default_policy_guidance_spec,
    summarize_policy_guidance_spec,
    validate_policy_guidance_spec,
)

DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "rewarding"
    / "templates"
    / "reward_function_template.py"
)
DEFAULT_PROMPT_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "generator_prompt.md"
)
DEFAULT_PBRS_PROMPT_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "generator_prompt_pbrs.md"
)
DEFAULT_HEURISTIC_PROMPT_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "generator_prompt_heuristic.md"
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
class RewardGenerationResult:
    prompt: str
    response: Dict[str, Any]
    reward_code: str = ""
    reward_spec: Optional[Dict[str, Any]] = None
    alpha_policy: Optional[Dict[str, Any]] = None
    policy_guidance_spec: Optional[Dict[str, Any]] = None
    candidate_proposals: Optional[list[Dict[str, Any]]] = None


class TemplateBasedGeneratorClient:
    """
    Build a structured generator prompt and delegate the actual LLM call to an
    injected backend.

    The backend is intentionally provider-agnostic so the workflow layer does
    not depend on a single model vendor.
    """

    def __init__(
        self,
        llm_backend: LLMBackend,
        template_path: Optional[Path | str] = None,
        prompt_spec_path: Optional[Path | str] = None,
        env_reference_path: Optional[Path | str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm_backend = llm_backend
        self.template_path = (
            Path(template_path) if template_path is not None else DEFAULT_TEMPLATE_PATH
        )
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

    def generate(
        self,
        *,
        round_id: int,
        workflow_spec: Dict[str, Any],
        previous_round: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = self.build_prompt(
            round_id=round_id,
            workflow_spec=workflow_spec,
            previous_round=previous_round,
        )
        llm_result = self.llm_backend.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            metadata={
                "round_id": round_id,
                "workflow_id": workflow_spec.get("workflow_id"),
                "role": "generator",
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
        return RewardGenerationResult(
            prompt=prompt,
            response=response,
            reward_code=parsed.get("reward_code", ""),
            reward_spec=parsed.get("reward_spec"),
            alpha_policy=parsed.get("alpha_policy"),
            policy_guidance_spec=parsed.get("policy_guidance_spec"),
            candidate_proposals=parsed.get("candidate_proposals"),
        ).__dict__

    def build_prompt(
        self,
        *,
        round_id: int,
        workflow_spec: Dict[str, Any],
        previous_round: Optional[Dict[str, Any]],
    ) -> str:
        template_code = self.template_path.read_text(encoding="utf-8")
        prompt_spec = self._resolve_prompt_spec_path(workflow_spec).read_text(encoding="utf-8")
        env_reference = self.env_reference_path.read_text(encoding="utf-8")
        task_description = workflow_spec.get("task_description", "")
        env_description = workflow_spec.get("environment_description", "")
        reward_constraints = workflow_spec.get("reward_constraints", [])
        train_spec = workflow_spec.get("train", {})
        reward_paradigm = self._get_reward_paradigm(workflow_spec)
        alpha_updates_enabled = self._alpha_policy_updates_enabled(workflow_spec)
        pbrs_tuning = (
            workflow_spec.get("pbrs_tuning", {}) if reward_paradigm == "pbrs" else {}
        )
        pbrs_checkpoint_plan = (
            workflow_spec.get("pbrs_checkpoint_plan", {})
            if reward_paradigm == "pbrs"
            else {}
        )
        candidate_values = pbrs_tuning.get("candidate_values", {})
        pbrs_design_space = get_supported_pbrs_design_space()
        default_reward_spec = get_default_reward_spec()
        default_policy_guidance_spec = get_default_policy_guidance_spec()
        supported_reward_terms = get_supported_reward_terms()
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
        candidate_selection_context = self._build_candidate_selection_context(
            workflow_spec=workflow_spec,
            round_id=round_id,
            active_pbrs_field=active_pbrs_field,
            active_checkpoint_context=active_checkpoint_context,
            previous_round=previous_round,
        )
        branch_critic_recommendation = self._get_branch_critic_recommendation(
            workflow_spec=workflow_spec
        )
        branch_critic_patch = self._get_branch_critic_patch(workflow_spec=workflow_spec)
        branch_design_transition = self._get_branch_design_transition(
            branch_critic_recommendation=branch_critic_recommendation,
            branch_critic_patch=branch_critic_patch,
        )
        effective_candidate_values = self._get_effective_candidate_values(
            workflow_spec=workflow_spec,
            round_id=round_id,
            active_pbrs_field=active_pbrs_field,
            previous_round=previous_round,
        )
        previous_search_strategy_recommendation = self._get_previous_search_strategy_recommendation(
            previous_round
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
            task_description or "No task description provided.",
            "",
            "Environment description:",
            env_description or "No environment description provided.",
            "",
            "LBF environment reference:",
            env_reference,
            "",
            "Training configuration summary:",
            json.dumps(train_spec, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            f"Reward paradigm for this workflow: {reward_paradigm}",
            "",
            "Reward design constraints:",
            self._format_constraints(reward_constraints),
            "",
            "Default structured reward specification:",
            json.dumps(default_reward_spec, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Default structured policy guidance specification:",
            json.dumps(
                default_policy_guidance_spec,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Supported reward terms and trigger variants:",
            json.dumps(supported_reward_terms, ensure_ascii=False, indent=2, sort_keys=True),
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
            "PBRS checkpoint plan:",
            json.dumps(pbrs_checkpoint_plan, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Active PBRS checkpoint context for this round:",
            json.dumps(active_checkpoint_context, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Active field carryover context for this round:",
            json.dumps(active_field_carryover_context, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Candidate selection context for this round:",
            json.dumps(candidate_selection_context, ensure_ascii=False, indent=2, sort_keys=True),
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
            "Important note about the template:",
            "The reference template defines the required interface and module shape.",
            "It is an execution reference only. The actual design should be expressed through reward_spec.",
            "",
            "Reference template:",
            template_code,
        ]

        if reward_paradigm == "pbrs":
            prompt_sections.extend(
                [
                    "",
                    "PBRS workflow mode notes:",
                    "Treat reward_spec.pbrs as the primary design surface for this workflow.",
                    "Treat heuristic terms as fixed scaffold context unless the workflow explicitly allows term changes.",
                    "Do not generate or modify alpha_policy in this mode; the workflow will ignore it.",
                    "If PBRS checkpoint plan information is present, use it as stage-aware context rather than as a hard-coded reward term change request.",
                    "If active checkpoint context is present for this round, let it guide which PBRS regime is promising at this training stage.",
                    "If candidate selection context shows a narrowed or shifted sweep, treat that as an intentional workflow decision rather than a mistake.",
                    "If a branch-level PBRS recommendation is present, treat it as compressed structured Critic feedback rather than as raw branch evidence.",
                ]
            )
        elif reward_paradigm == "heuristic":
            prompt_sections.extend(
                [
                    "",
                    "Heuristic reward workflow mode notes:",
                    "Treat reward_spec.terms as the primary design surface for this workflow.",
                    "Do not redesign PBRS unless the workflow explicitly asks for PBRS changes.",
                ]
            )
            if not alpha_updates_enabled:
                prompt_sections.append(
                    "Do not generate or modify alpha_policy in this run because alpha updates are disabled."
                )

        if previous_round:
            previous_status = previous_round.get("status") if isinstance(previous_round, dict) else None
            previous_critic_feedback = None
            previous_critic_response = previous_round.get("critic_response") if isinstance(previous_round, dict) else None
            if isinstance(previous_critic_response, dict):
                previous_critic_feedback = previous_critic_response.get("parsed")
            prompt_sections.extend(
                [
                    "",
                    "Previous round compact summary:",
                    json.dumps(
                        {
                            "round_id": previous_round.get("round_id"),
                            "status": previous_status,
                            "reward_spec_change_summary": previous_round.get(
                                "reward_spec_change_summary"
                            ),
                            "suggested_reward_spec": previous_round.get(
                                "suggested_reward_spec"
                            ),
                            "policy_guidance_spec": previous_round.get(
                                "policy_guidance_spec"
                            ),
                            "policy_guidance_summary": summarize_policy_guidance_spec(
                                previous_round.get("policy_guidance_spec")
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    "",
                    "Previous Critic structured feedback:",
                    json.dumps(
                        previous_critic_feedback,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    "",
                    "Previous policy guidance summary:",
                    json.dumps(
                        summarize_policy_guidance_spec(
                            previous_round.get("policy_guidance_spec")
                        ),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    "",
                    "Improve on the previous round instead of rewriting blindly.",
                    "Pay special attention to any preserve_elements returned by the previous Critic response.",
                    "If the previous Critic response includes reward_update_plan, treat it as the default structured update budget.",
                    "If previous_round includes suggested_reward_spec, treat it as the system-generated default candidate for this round.",
                    "Prefer changing only a small number of terms, trigger variants, or weights unless the evidence strongly justifies a broader change.",
                    "Use the previous round evidence to decide what to keep, what to remove, and what new shaping directions to explore.",
                    "If the previous round remained too close to the template or failed to improve sparse performance, do not repeat the same reward structure without a clear reason.",
                ]
            )
            if previous_search_strategy_recommendation is not None:
                prompt_sections.extend(
                    [
                        "",
                        "Previous Critic search strategy recommendation:",
                        json.dumps(
                            previous_search_strategy_recommendation,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        ),
                        "Use this as context for why the current round may be sweeping a narrowed or shifted candidate set.",
                    ]
                )
        else:
            prompt_sections.extend(
                [
                    "",
                    "This is the first round. Produce a safe initial dense reward design.",
                    "Even in round 1, do not simply restate the template; use it only as an interface reference.",
                ]
            )

        if active_pbrs_field is not None:
            active_candidates = (
                effective_candidate_values
                if effective_candidate_values
                else (
                    candidate_values.get(active_pbrs_field)
                    if isinstance(candidate_values, dict)
                    else None
                )
            )
            prompt_sections.extend(
                [
                    "",
                    "Sequential PBRS tuning mode is active for this round.",
                    f"Active PBRS coefficient for round {round_id}: {active_pbrs_field}",
                    "Only change the active PBRS coefficient in reward_spec.pbrs.",
                    "Do not change other PBRS coefficients in this round.",
                    "If freeze_reward_terms is enabled in the PBRS tuning config, keep reward terms unchanged as well.",
                ]
            )
            if isinstance(active_candidates, list) and active_candidates:
                prompt_sections.extend(
                    [
                        f"The system will run multiple complete experiments for this coefficient using candidate values: {json.dumps(active_candidates, ensure_ascii=False)}",
                        "These values may already reflect stage-aware overrides from the active baseline checkpoint context.",
                        "Design the round-level reward_spec scaffold for this active coefficient, but expect the workflow to overwrite the active coefficient with each candidate value during training.",
                    ]
                )

        prompt_sections.extend(
            [
                "",
                "Generate the response now according to the prompt specification above.",
            ]
        )
        return "\n".join(prompt_sections)

    def parse_response(self, text: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Generator output is not valid JSON: {exc}") from exc

        reward_spec = parsed.get("reward_spec")
        if reward_spec is None:
            raise ValueError('Generator output must contain key "reward_spec"')
        validate_reward_spec(reward_spec)

        candidate_proposals = parsed.get("candidate_proposals")
        if not isinstance(candidate_proposals, list) or not candidate_proposals:
            raise ValueError('Generator output must contain non-empty key "candidate_proposals"')
        normalized_proposals = []
        for index, proposal in enumerate(candidate_proposals, start=1):
            if not isinstance(proposal, dict):
                raise ValueError("Each generator candidate_proposal must be a dict")
            normalized = self._normalize_candidate_proposal(proposal, index=index)
            candidate_id = normalized.get("candidate_id")
            changed_terms = normalized.get("changed_terms")
            expected_effect = normalized.get("expected_effect")
            risk_hypothesis = normalized.get("risk_hypothesis")
            rationale = normalized.get("rationale")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("Each generator candidate_proposal must include candidate_id")
            if not isinstance(changed_terms, list):
                raise ValueError(
                    f'Generator candidate_proposal "{candidate_id}" must include changed_terms list'
                )
            if not isinstance(expected_effect, str) or not expected_effect.strip():
                raise ValueError(
                    f'Generator candidate_proposal "{candidate_id}" must include expected_effect'
                )
            if not isinstance(risk_hypothesis, str) or not risk_hypothesis.strip():
                raise ValueError(
                    f'Generator candidate_proposal "{candidate_id}" must include risk_hypothesis'
                )
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError(
                    f'Generator candidate_proposal "{candidate_id}" must include rationale'
                )
            normalized.setdefault("proposal_index", index)
            normalized_proposals.append(normalized)
        parsed["candidate_proposals"] = normalized_proposals

        reward_code = parsed.get("reward_code")
        if reward_code is not None:
            if not isinstance(reward_code, str):
                raise ValueError('Generator "reward_code" must be a string when provided')
            if reward_code and "compute_dense_reward" not in reward_code:
                parsed["reward_code"] = ""

        alpha_policy = parsed.get("alpha_policy")
        if alpha_policy is None and isinstance(reward_spec, dict):
            alpha_policy = reward_spec.get("alpha_policy")
        if alpha_policy is not None:
            validate_alpha_policy(alpha_policy)
            parsed["alpha_policy"] = alpha_policy

        policy_guidance_spec = parsed.get("policy_guidance_spec")
        if policy_guidance_spec is None:
            policy_guidance_spec = get_default_policy_guidance_spec()
        parsed["policy_guidance_spec"] = validate_policy_guidance_spec(
            policy_guidance_spec
        )

        return parsed

    def _normalize_candidate_proposal(
        self,
        proposal: Dict[str, Any],
        *,
        index: int,
    ) -> Dict[str, Any]:
        normalized = dict(proposal)

        candidate_id = normalized.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            candidate_id = f"proposal_{index:02d}"
        candidate_id = candidate_id.strip()
        normalized["candidate_id"] = candidate_id

        changed_terms = normalized.get("changed_terms")
        if not isinstance(changed_terms, list) or not changed_terms:
            inferred_terms = self._infer_changed_terms_from_proposal(normalized)
            if inferred_terms:
                changed_terms = inferred_terms
            else:
                changed_terms = ["active_reward_spec"]
        normalized["changed_terms"] = [str(term) for term in changed_terms if str(term).strip()]
        if not normalized["changed_terms"]:
            normalized["changed_terms"] = ["active_reward_spec"]

        expected_effect = normalized.get("expected_effect")
        if not isinstance(expected_effect, str) or not expected_effect.strip():
            primary_term = normalized["changed_terms"][0]
            expected_effect = (
                f"Evaluate whether adjusting {primary_term} improves sparse learning "
                "under the current constrained search budget."
            )
        normalized["expected_effect"] = expected_effect.strip()

        risk_hypothesis = normalized.get("risk_hypothesis")
        if not isinstance(risk_hypothesis, str) or not risk_hypothesis.strip():
            primary_term = normalized["changed_terms"][0]
            risk_hypothesis = (
                f"Changing {primary_term} may over-shape training dynamics or fail to "
                "translate into sparse test improvement."
            )
        normalized["risk_hypothesis"] = risk_hypothesis.strip()

        rationale = normalized.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            rationale = (
                "Normalized from a compact real-LLM proposal so the workflow can keep "
                "the Generator in a candidate-proposer role without rejecting the run."
            )
        normalized["rationale"] = rationale.strip()
        return normalized

    def _infer_changed_terms_from_proposal(self, proposal: Dict[str, Any]) -> list[str]:
        explicit_keys = (
            proposal.get("changed_term"),
            proposal.get("changed_terms"),
            proposal.get("field"),
            proposal.get("fields"),
            proposal.get("term"),
            proposal.get("terms"),
            proposal.get("active_pbrs_field"),
        )
        inferred: list[str] = []
        for value in explicit_keys:
            if isinstance(value, str) and value.strip():
                inferred.append(value.strip())
            elif isinstance(value, list):
                inferred.extend(str(item).strip() for item in value if str(item).strip())
        if inferred:
            return inferred

        candidate_id = proposal.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id.strip():
            head = candidate_id.strip().split("_", 1)[0].strip()
            if head:
                return [head]
        return []

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
        previous_search_strategy_recommendation = self._get_previous_search_strategy_recommendation(
            previous_round
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

        previous_search_strategy_recommendation = self._get_previous_search_strategy_recommendation(
            previous_round
        )
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
        if (
            previous_active_field == active_pbrs_field
            and isinstance(previous_search_strategy_recommendation, dict)
            and previous_search_strategy_recommendation.get("candidate_value_action")
            in {"narrow", "widen", "shift"}
        ):
            suggested_values = previous_search_strategy_recommendation.get(
                "suggested_candidate_values"
            )
            if isinstance(suggested_values, list):
                normalized_values = []
                for value in suggested_values:
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if numeric_value not in normalized_values:
                        normalized_values.append(numeric_value)
                if normalized_values:
                    candidate_values = normalized_values

        return list(candidate_values)

    def _get_previous_search_strategy_recommendation(
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
        recommendation = parsed.get("search_strategy_recommendation")
        if not isinstance(recommendation, dict):
            return None
        return recommendation

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

    def _build_candidate_selection_context(
        self,
        *,
        workflow_spec: Dict[str, Any],
        round_id: int,
        active_pbrs_field: Optional[str],
        active_checkpoint_context: Optional[Dict[str, Any]],
        previous_round: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if active_pbrs_field is None:
            return {
                "active_pbrs_field": None,
                "candidate_value_source": None,
                "effective_candidate_values": [],
            }

        pbrs_tuning = workflow_spec.get("pbrs_tuning", {})
        base_candidate_values = self._normalize_candidate_value_mapping(
            pbrs_tuning.get("candidate_values", {})
        ).get(active_pbrs_field, [])
        candidate_value_source = "candidate_values"
        checkpoint_candidate_values = None
        stage_candidate_values = None
        critic_recommended_candidate_values = None
        effective_candidate_values = list(base_candidate_values)

        if active_checkpoint_context is not None:
            per_checkpoint_root = pbrs_tuning.get("per_checkpoint_candidate_values")
            if isinstance(per_checkpoint_root, dict):
                checkpoint_values = self._normalize_candidate_value_mapping(
                    per_checkpoint_root.get(str(active_checkpoint_context.get("name")), {})
                ).get(active_pbrs_field)
                if checkpoint_values:
                    checkpoint_candidate_values = list(checkpoint_values)
                    effective_candidate_values = list(checkpoint_candidate_values)
                    candidate_value_source = "per_checkpoint_candidate_values"

            stage_label = active_checkpoint_context.get("stage_label")
            per_stage_root = pbrs_tuning.get("per_stage_candidate_values")
            if stage_label is not None and isinstance(per_stage_root, dict):
                stage_values = self._normalize_candidate_value_mapping(
                    per_stage_root.get(str(stage_label), {})
                ).get(active_pbrs_field)
                if stage_values:
                    stage_candidate_values = list(stage_values)
                    effective_candidate_values = list(stage_candidate_values)
                    candidate_value_source = "per_stage_candidate_values"

        previous_search_strategy_recommendation = self._get_previous_search_strategy_recommendation(
            previous_round
        )
        previous_active_field = None
        if isinstance(previous_round, dict):
            previous_status = previous_round.get("status")
            if isinstance(previous_status, dict):
                previous_active_field = previous_status.get("active_pbrs_field")
        if (
            previous_active_field == active_pbrs_field
            and isinstance(previous_search_strategy_recommendation, dict)
            and previous_search_strategy_recommendation.get("candidate_value_action")
            in {"narrow", "widen", "shift"}
        ):
            suggested_values = previous_search_strategy_recommendation.get(
                "suggested_candidate_values"
            )
            if isinstance(suggested_values, list):
                normalized_values = []
                for value in suggested_values:
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if numeric_value not in normalized_values:
                        normalized_values.append(numeric_value)
                if normalized_values:
                    critic_recommended_candidate_values = normalized_values
                    effective_candidate_values = normalized_values
                    candidate_value_source = "critic_search_strategy_recommendation"

        return {
            "active_pbrs_field": active_pbrs_field,
            "candidate_value_source": candidate_value_source,
            "effective_candidate_values": list(effective_candidate_values),
            "base_candidate_values": list(base_candidate_values),
            "checkpoint_candidate_values": checkpoint_candidate_values,
            "stage_candidate_values": stage_candidate_values,
            "critic_recommended_candidate_values": critic_recommended_candidate_values,
        }

    def _build_active_field_carryover_context(
        self,
        *,
        active_pbrs_field: Optional[str],
        previous_round: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        previous_search_strategy_recommendation = self._get_previous_search_strategy_recommendation(
            previous_round
        )
        previous_round_id = previous_round.get("round_id") if isinstance(previous_round, dict) else None
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

    def _format_constraints(self, constraints: Any) -> str:
        if not constraints:
            return "- No additional constraints provided."
        if isinstance(constraints, list):
            return "\n".join(f"- {item}" for item in constraints)
        return str(constraints)

    def _default_system_prompt(self) -> str:
        return (
            "You generate structured dense reward specifications for multi-agent reinforcement learning. "
            "You must follow the provided scaffold exactly, preserve task semantics, "
            "and return strictly valid JSON."
        )


class StubGeneratorBackend:
    """
    Deterministic backend useful for development before wiring a real LLM API.
    """

    def __init__(self, template_path: Optional[Path | str] = None):
        self.template_path = (
            Path(template_path) if template_path is not None else DEFAULT_TEMPLATE_PATH
        )

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reward_code = self.template_path.read_text(encoding="utf-8")
        reward_spec = get_default_reward_spec()
        policy_guidance_spec = get_default_policy_guidance_spec()
        payload = {
            "reward_spec": reward_spec,
            "policy_guidance_spec": policy_guidance_spec,
            "reward_code": reward_code,
            "design_rationale": "Stub backend returning the reference reward template.",
            "candidate_proposals": [
                {
                    "candidate_id": "proposal_01",
                    "changed_terms": ["active_pbrs_field"],
                    "expected_effect": "Preserve the current constrained scaffold and let the workflow evaluate the scheduled candidate sweep.",
                    "risk_hypothesis": "The scheduled sweep may still be too weak to move sparse learning.",
                    "rationale": "Stub generator keeps the Generator in a candidate-proposer role without rewriting the reward freely.",
                },
                {
                    "candidate_id": "proposal_02",
                    "changed_terms": ["active_pbrs_field"],
                    "expected_effect": "Provide a nearby conservative explanation for the candidate sweep.",
                    "risk_hypothesis": "Conservative shaping may under-correct sparse learning plateaus.",
                    "rationale": "This placeholder proposal marks the low-risk edge of the constrained search space.",
                },
                {
                    "candidate_id": "proposal_03",
                    "changed_terms": ["active_pbrs_field"],
                    "expected_effect": "Provide a nearby aggressive explanation for the candidate sweep.",
                    "risk_hypothesis": "Aggressive shaping may distort mixed returns without improving sparse success.",
                    "rationale": "This placeholder proposal marks the higher-risk edge of the constrained search space.",
                },
            ],
        }
        return {
            "text": json.dumps(payload, ensure_ascii=False),
            "raw_response": {"stub": True, "metadata": metadata},
            "model": "stub-generator",
            "usage": None,
        }
