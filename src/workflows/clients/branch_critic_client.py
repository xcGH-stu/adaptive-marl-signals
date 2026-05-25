from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from workflows.critic_diagnosis import validate_structured_diagnosis


DEFAULT_PROMPT_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "branch_critic_prompt_pbrs.md"
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
        """


@dataclass
class BranchCritiqueResult:
    prompt: str
    response: Dict[str, Any]


class TemplateBasedPBRSBranchCriticClient:
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
        self.env_reference_path = (
            Path(env_reference_path)
            if env_reference_path is not None
            else DEFAULT_ENV_REFERENCE_PATH
        )
        self.system_prompt = system_prompt or self._default_system_prompt()

    def critique_branch_plan(
        self,
        *,
        branch_plan_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = self.build_prompt(branch_plan_summary=branch_plan_summary)
        llm_result = self.llm_backend.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            metadata={
                "branch_plan_id": branch_plan_summary.get("branch_plan_id"),
                "role": "branch_critic",
                "branch_plan_summary": branch_plan_summary,
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
        return BranchCritiqueResult(prompt=prompt, response=response).__dict__

    def build_prompt(
        self,
        *,
        branch_plan_summary: Dict[str, Any],
    ) -> str:
        prompt_spec = self.prompt_spec_path.read_text(encoding="utf-8")
        env_reference = self.env_reference_path.read_text(encoding="utf-8")
        critic_payload = branch_plan_summary.get("critic_payload") or {}
        sections = [
            "Prompt specification:",
            prompt_spec,
            "",
            "LBF environment reference:",
            env_reference,
            "",
            "Branch plan summary:",
            json.dumps(branch_plan_summary, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Critic payload:",
            json.dumps(critic_payload, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Analyze the checkpoint-branching evidence and return the required JSON now.",
        ]
        return "\n".join(sections)

    def parse_response(self, text: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Branch Critic output is not valid JSON: {exc}") from exc

        required_keys = [
            "analysis_summary",
            "checkpoint_findings",
            "cross_checkpoint_patterns",
            "recommended_followup",
            "final_recommendation",
            "structured_diagnosis",
        ]
        missing = [key for key in required_keys if key not in parsed]
        if missing:
            raise ValueError(
                "Branch Critic output is missing required keys: " + ", ".join(missing)
            )
        if not isinstance(parsed["checkpoint_findings"], list):
            raise ValueError("checkpoint_findings must be a list")
        if not isinstance(parsed["cross_checkpoint_patterns"], list):
            raise ValueError("cross_checkpoint_patterns must be a list")
        if not isinstance(parsed["recommended_followup"], list):
            raise ValueError("recommended_followup must be a list")
        if not isinstance(parsed["final_recommendation"], dict):
            raise ValueError("final_recommendation must be a dict")
        parsed["structured_diagnosis"] = validate_structured_diagnosis(
            parsed["structured_diagnosis"]
        )
        design_transition = parsed["final_recommendation"].get("recommended_design_transition")
        if design_transition is not None and not isinstance(design_transition, dict):
            raise ValueError(
                "final_recommendation.recommended_design_transition must be a dict when provided"
            )
        return parsed

    def _default_system_prompt(self) -> str:
        return (
            "You analyze PBRS checkpoint-branching results and recommend the most "
            "promising PBRS strategy across training stages. Return strictly valid JSON."
        )


class StubPBRSBranchCriticBackend:
    def _executed_checkpoints(self, branch_plan_summary: Dict[str, Any]) -> list[Dict[str, Any]]:
        critic_payload = branch_plan_summary.get("critic_payload") or {}
        comparisons = critic_payload.get("checkpoint_comparisons") or []
        executed = []
        for comparison in comparisons:
            if comparison.get("planned_only"):
                continue
            best_candidate = comparison.get("best_candidate") or {}
            records = comparison.get("candidate_records") or []
            if best_candidate or records:
                executed.append(comparison)
        return executed

    def _recommended_candidate_values(self, records: list[Dict[str, Any]], best_value: Any) -> list[float]:
        numeric_values = []
        for record in records:
            try:
                numeric_values.append(float(record.get("candidate_value")))
            except (TypeError, ValueError):
                continue
        numeric_values = sorted(set(numeric_values))
        try:
            best_value = float(best_value)
        except (TypeError, ValueError):
            return numeric_values[:3]
        if best_value not in numeric_values:
            numeric_values.append(best_value)
            numeric_values.sort()
        idx = numeric_values.index(best_value)
        start = max(0, idx - 1)
        end = min(len(numeric_values), idx + 2)
        window = numeric_values[start:end]
        if len(window) == 2:
            if start > 0:
                window = numeric_values[start - 1 : end]
            elif end < len(numeric_values):
                window = numeric_values[start : end + 1]
        return window or [best_value]

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary = ((metadata or {}).get("branch_plan_summary") or {}) if isinstance(metadata, dict) else {}
        executed = self._executed_checkpoints(summary)
        checkpoint_findings = []
        cross_checkpoint_patterns = []
        recommended_followup = []
        recommended_values_by_stage: Dict[str, float] = {}
        recommended_fields_by_stage: Dict[str, str] = {}
        recommended_strategy = "need_more_evidence"
        recommended_global_value = None
        recommended_next_field = "beta"
        recommended_candidate_values = [0.1, 0.3, 0.5]
        recommended_design_transition = {
            "transition_action": "keep_coefficients_only",
            "target_scope": "global",
            "target_stage": None,
            "allowed_structural_pbrs_fields": [],
            "rationale": "Structural PBRS updates should remain locked until stronger multi-stage evidence appears.",
        }
        rationale = "No executed branch metrics are present yet."
        analysis_summary = (
            "Branch plan structure is available, but no executed branch metrics are present yet."
        )

        if executed:
            active_fields = [
                item.get("active_pbrs_field")
                for item in executed
                if isinstance(item.get("active_pbrs_field"), str)
            ]
            if active_fields:
                recommended_next_field = active_fields[0]

            stage_values = []
            for comparison in executed:
                best_candidate = comparison.get("best_candidate") or {}
                records = comparison.get("candidate_records") or []
                ranking_metric = comparison.get("ranking_metric")
                checkpoint_name = comparison.get("checkpoint_name")
                stage_label = comparison.get("checkpoint_stage_label")
                best_value = best_candidate.get("candidate_value")
                score = (
                    best_candidate.get(ranking_metric)
                    if isinstance(ranking_metric, str)
                    else None
                )
                checkpoint_findings.append(
                    {
                        "checkpoint_name": checkpoint_name,
                        "stage_label": stage_label,
                        "finding": (
                            f"Best observed candidate is {best_candidate.get('candidate_id')} "
                            f"(value={best_value}) using ranking metric "
                            f"{ranking_metric or 'unavailable'} with score={score}."
                        ),
                    }
                )
                try:
                    stage_values.append((str(stage_label or checkpoint_name), float(best_value)))
                except (TypeError, ValueError):
                    continue
                if isinstance(stage_label or checkpoint_name, str) and isinstance(
                    comparison.get("active_pbrs_field"), str
                ):
                    recommended_fields_by_stage[str(stage_label or checkpoint_name)] = str(
                        comparison.get("active_pbrs_field")
                    )
                if recommended_global_value is None:
                    recommended_global_value = best_value
                    recommended_candidate_values = self._recommended_candidate_values(
                        records,
                        best_value,
                    )

            if len(stage_values) == 1:
                recommended_strategy = "global_value"
                analysis_summary = (
                    "Executed branch metrics are available for one checkpoint, so the recommendation "
                    "is a provisional global value centered on the best observed branch candidate."
                )
                cross_checkpoint_patterns.append(
                    "Only one checkpoint has executed evidence so far, so the recommendation is provisional."
                )
                recommended_followup.extend(
                    [
                        "Run additional checkpoint branches to test whether the same coefficient remains best later in training.",
                        "Reuse the narrowed candidate window around the current best value for the next branch sweep.",
                    ]
                )
                recommended_design_transition = {
                    "transition_action": "keep_coefficients_only",
                    "target_scope": "next_stage",
                    "target_stage": "mid_progress",
                    "allowed_structural_pbrs_fields": [],
                    "rationale": "Single-checkpoint evidence is enough to narrow coefficient search, but not enough to unlock PBRS structural changes yet.",
                }
                rationale = (
                    "A single checkpoint with executed branch metrics supports narrowing around the current best value, "
                    "but stage dependence cannot be ruled out yet."
                )
            else:
                distinct_values = {round(value, 12) for _, value in stage_values}
                if len(distinct_values) == 1:
                    recommended_strategy = "global_value"
                    recommended_global_value = stage_values[0][1]
                    cross_checkpoint_patterns.append(
                        "Executed checkpoints currently agree on the same best candidate value."
                    )
                    recommended_followup.extend(
                        [
                            "Promote the shared best value as the next default PBRS setting.",
                            "Run one confirmation sweep with a tight window around the shared value.",
                        ]
                    )
                    recommended_design_transition = {
                        "transition_action": "unlock_variant",
                        "target_scope": "next_stage",
                        "target_stage": "mid_progress",
                        "allowed_structural_pbrs_fields": ["variant"],
                        "rationale": "Consistent coefficient behavior across executed checkpoints suggests the next source of gain may be PBRS variant choice rather than wider coefficient search.",
                    }
                    rationale = (
                        "Multiple executed checkpoints currently point to the same best value, "
                        "which supports a provisional global setting."
                    )
                    analysis_summary = (
                        "Executed branch metrics across checkpoints currently support a shared PBRS value."
                    )
                else:
                    recommended_strategy = "stage_specific"
                    recommended_values_by_stage = {
                        stage_name: value for stage_name, value in stage_values
                    }
                    cross_checkpoint_patterns.append(
                        "Executed checkpoints prefer different PBRS values across stages."
                    )
                    recommended_followup.extend(
                        [
                            "Preserve stage-specific PBRS values in the next workflow iteration.",
                            "Expand checkpoint branching to confirm whether stage differences persist across seeds.",
                        ]
                    )
                    recommended_design_transition = {
                        "transition_action": "unlock_gate",
                        "target_scope": "next_stage",
                        "target_stage": "late_plateau",
                        "allowed_structural_pbrs_fields": ["variant", "gate_radius", "gate_mode"],
                        "rationale": "Divergent best values across checkpoints suggest stage-sensitive coordination geometry, so gate-related PBRS structure should be opened next.",
                    }
                    rationale = (
                        "Different executed checkpoints currently prefer different values, "
                        "which is stronger evidence for stage-specific PBRS scheduling."
                    )
                    analysis_summary = (
                        "Executed branch metrics indicate stage-dependent PBRS preferences across checkpoints."
                    )
        else:
            checkpoint_findings = [
                {
                    "checkpoint_name": "planned_checkpoints",
                    "stage_label": "unknown",
                    "finding": "No executed branch metrics are available yet.",
                }
            ]
            cross_checkpoint_patterns = [
                "The current branch plan is still waiting on executed evidence.",
            ]
            recommended_followup = [
                "Execute the planned branch runs for each checkpoint.",
                "Compare candidate rankings after branch metrics are available.",
            ]

        payload = {
            "analysis_summary": analysis_summary,
            "checkpoint_findings": checkpoint_findings,
            "cross_checkpoint_patterns": cross_checkpoint_patterns,
            "recommended_followup": recommended_followup,
            "structured_diagnosis": {
                "best_candidate": {
                    "scope": recommended_strategy,
                    "candidate_value": recommended_global_value,
                    "values_by_stage": recommended_values_by_stage,
                    "fields_by_stage": recommended_fields_by_stage,
                },
                "verdict": (
                    "accept" if recommended_strategy in {"global_value", "stage_specific"} else "uncertain"
                ),
                "evidence_summary": analysis_summary,
                "failure_attribution": cross_checkpoint_patterns,
                "over_shaping_risk": (
                    "moderate" if recommended_strategy == "stage_specific" else "low"
                ),
                "minimal_patch": {
                    "recommended_strategy": recommended_strategy,
                    "recommended_next_field": recommended_next_field,
                    "recommended_candidate_values": recommended_candidate_values,
                    "recommended_values_by_stage": recommended_values_by_stage,
                    "recommended_fields_by_stage": recommended_fields_by_stage,
                    "recommended_design_transition": recommended_design_transition,
                },
                "whether_to_full_run": bool(executed),
            },
            "final_recommendation": {
                "recommended_strategy": recommended_strategy,
                "recommended_values_by_stage": recommended_values_by_stage,
                "recommended_fields_by_stage": recommended_fields_by_stage,
                "recommended_global_value": recommended_global_value,
                "recommended_next_field": recommended_next_field,
                "recommended_candidate_values": recommended_candidate_values,
                "recommended_design_transition": recommended_design_transition,
                "rationale": rationale,
            },
        }
        return {
            "text": json.dumps(payload, ensure_ascii=False),
            "raw_response": {"stub": True, "metadata": metadata},
            "model": "stub-branch-critic",
            "usage": None,
        }
