from __future__ import annotations

from typing import Any, Dict


REQUIRED_STRUCTURED_DIAGNOSIS_KEYS = [
    "best_candidate",
    "verdict",
    "evidence_summary",
    "failure_attribution",
    "over_shaping_risk",
    "minimal_patch",
    "whether_to_full_run",
]


def validate_structured_diagnosis(diagnosis: Any) -> Dict[str, Any]:
    if not isinstance(diagnosis, dict):
        raise ValueError('Critic "structured_diagnosis" must be a dict')
    missing = [key for key in REQUIRED_STRUCTURED_DIAGNOSIS_KEYS if key not in diagnosis]
    if missing:
        raise ValueError(
            "Critic structured_diagnosis is missing required keys: " + ", ".join(missing)
        )
    if diagnosis["verdict"] not in {"accept", "reject", "revise", "uncertain"}:
        raise ValueError(
            'Critic structured_diagnosis.verdict must be one of "accept", "reject", "revise", "uncertain"'
        )
    if not isinstance(diagnosis["evidence_summary"], str):
        raise ValueError("Critic structured_diagnosis.evidence_summary must be a string")
    if not isinstance(diagnosis["failure_attribution"], list):
        raise ValueError("Critic structured_diagnosis.failure_attribution must be a list")
    if not isinstance(diagnosis["over_shaping_risk"], str):
        raise ValueError("Critic structured_diagnosis.over_shaping_risk must be a string")
    if not isinstance(diagnosis["minimal_patch"], dict):
        raise ValueError("Critic structured_diagnosis.minimal_patch must be a dict")
    if not isinstance(diagnosis["whether_to_full_run"], bool):
        raise ValueError("Critic structured_diagnosis.whether_to_full_run must be a bool")
    return diagnosis
