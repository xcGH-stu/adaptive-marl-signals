from __future__ import annotations

from typing import Any, Dict, List


SUPPORTED_STAGE_LABELS = {
    "early_exploration",
    "early_transition",
    "mid_progress",
    "mid_plateau",
    "late_refinement",
    "late_plateau",
}
SUPPORTED_ADAPTATION_WINDOW_QUALITY = {"high", "medium", "low"}


def validate_checkpoint_selection(
    payload: Dict[str, Any],
    *,
    min_count: int = 3,
    max_count: int = 4,
    min_step_gap: int = 1,
    allowed_steps: List[int] | None = None,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint selection payload must be a dict")

    selected = payload.get("selected_checkpoints")
    if not isinstance(selected, list):
        raise ValueError('checkpoint selection must include a "selected_checkpoints" list')
    if len(selected) < min_count or len(selected) > max_count:
        raise ValueError(
            f"selected_checkpoints must contain between {min_count} and {max_count} items"
        )

    validated_checkpoints: List[Dict[str, Any]] = []
    previous_step = None
    seen_steps = set()
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            raise ValueError(f"selected_checkpoints[{index}] must be a dict")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"selected_checkpoints[{index}].name must be a non-empty string")
        step = item.get("step")
        if not isinstance(step, int) or step < 0:
            raise ValueError(f"selected_checkpoints[{index}].step must be a non-negative int")
        if step in seen_steps:
            raise ValueError(f"selected_checkpoints[{index}].step is duplicated: {step}")
        if allowed_steps is not None and step not in allowed_steps:
            raise ValueError(
                f"selected_checkpoints[{index}].step={step} is not in allowed checkpoint candidates"
            )
        if previous_step is not None and step - previous_step < min_step_gap:
            raise ValueError(
                f"selected_checkpoints[{index}].step={step} is too close to previous checkpoint "
                f"step={previous_step}; min_step_gap={min_step_gap}"
            )
        stage_label = item.get("stage_label")
        if stage_label is not None:
            if not isinstance(stage_label, str):
                raise ValueError(
                    f"selected_checkpoints[{index}].stage_label must be a string when provided"
                )
            if stage_label not in SUPPORTED_STAGE_LABELS:
                raise ValueError(
                    f"selected_checkpoints[{index}].stage_label must be one of: "
                    + ", ".join(sorted(SUPPORTED_STAGE_LABELS))
                )
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"selected_checkpoints[{index}].reason must be a non-empty string"
            )
        recommended_focus = item.get("recommended_focus")
        if recommended_focus is not None and (
            not isinstance(recommended_focus, str) or not recommended_focus.strip()
        ):
            raise ValueError(
                f"selected_checkpoints[{index}].recommended_focus must be a non-empty string when provided"
            )
        adaptation_window_quality = item.get("adaptation_window_quality")
        if adaptation_window_quality is not None:
            if not isinstance(adaptation_window_quality, str):
                raise ValueError(
                    f"selected_checkpoints[{index}].adaptation_window_quality must be a string when provided"
                )
            if adaptation_window_quality not in SUPPORTED_ADAPTATION_WINDOW_QUALITY:
                raise ValueError(
                    f"selected_checkpoints[{index}].adaptation_window_quality must be one of: "
                    + ", ".join(sorted(SUPPORTED_ADAPTATION_WINDOW_QUALITY))
                )

        validated_checkpoints.append(
            {
                "name": name.strip(),
                "step": step,
                "stage_label": stage_label,
                "reason": reason.strip(),
                "recommended_focus": (
                    recommended_focus.strip() if isinstance(recommended_focus, str) else None
                ),
                "adaptation_window_quality": (
                    adaptation_window_quality.strip()
                    if isinstance(adaptation_window_quality, str)
                    else None
                ),
            }
        )
        previous_step = step
        seen_steps.add(step)

    analysis_summary = payload.get("analysis_summary")
    if analysis_summary is not None and not isinstance(analysis_summary, str):
        raise ValueError("analysis_summary must be a string when provided")
    evidence_summary = payload.get("evidence_summary")
    if evidence_summary is not None and not isinstance(evidence_summary, str):
        raise ValueError("evidence_summary must be a string when provided")
    meets_stage_requirements = payload.get("meets_stage_requirements")
    if meets_stage_requirements is not None and not isinstance(meets_stage_requirements, bool):
        raise ValueError("meets_stage_requirements must be a boolean when provided")
    requirement_notes = payload.get("requirement_notes")
    if requirement_notes is not None and not isinstance(requirement_notes, str):
        raise ValueError("requirement_notes must be a string when provided")
    dynamic_checkpoint_notes = payload.get("dynamic_checkpoint_notes")
    if dynamic_checkpoint_notes is not None and not isinstance(dynamic_checkpoint_notes, str):
        raise ValueError("dynamic_checkpoint_notes must be a string when provided")
    adaptive_intervention_notes = payload.get("adaptive_intervention_notes")
    if adaptive_intervention_notes is not None and not isinstance(adaptive_intervention_notes, str):
        raise ValueError("adaptive_intervention_notes must be a string when provided")

    return {
        "analysis_summary": analysis_summary or "",
        "evidence_summary": evidence_summary or "",
        "meets_stage_requirements": (
            True if meets_stage_requirements is None else bool(meets_stage_requirements)
        ),
        "requirement_notes": requirement_notes or "",
        "dynamic_checkpoint_notes": dynamic_checkpoint_notes or "",
        "adaptive_intervention_notes": adaptive_intervention_notes or "",
        "selected_checkpoints": validated_checkpoints,
    }
