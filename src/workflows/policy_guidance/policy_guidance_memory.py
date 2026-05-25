from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .policy_guidance_schema import validate_policy_guidance_call_record


def ensure_policy_guidance_history(memory: Dict[str, Any]) -> Dict[str, Any]:
    memory.setdefault("policy_guidance_history", [])
    history = memory.get("policy_guidance_history")
    if not isinstance(history, list):
        memory["policy_guidance_history"] = []
    return memory


def append_policy_guidance_call_record(
    memory: Dict[str, Any],
    record: Dict[str, Any],
) -> Dict[str, Any]:
    ensure_policy_guidance_history(memory)
    validated = validate_policy_guidance_call_record(record)
    history = memory["policy_guidance_history"]
    guidance_id = validated["guidance_id"]
    existing_index = next(
        (
            index
            for index, item in enumerate(history)
            if str(item.get("guidance_id") or "") == guidance_id
        ),
        None,
    )
    if existing_index is None:
        history.append(deepcopy(validated))
    else:
        history[existing_index] = deepcopy(validated)
    return memory


def append_stage1b_policy_guided_candidate_generation(
    memory: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    records = memory.setdefault("stage1b_policy_guided_candidate_generation", [])
    if not isinstance(records, list):
        records = []
        memory["stage1b_policy_guided_candidate_generation"] = records
    records.append(
        {
            "policy_guidance_used": bool(payload.get("policy_guidance_used", False)),
            "candidate_ids": [
                str(item.get("candidate_id") or "")
                for item in list(payload.get("candidates") or [])
            ],
            "candidate_types": [
                str(item.get("candidate_type") or "")
                for item in list(payload.get("candidates") or [])
            ],
            "evidence_keys_used": list(
                ((payload.get("candidate_generation_basis") or {}).get("evidence_keys"))
                or []
            ),
        }
    )
    return memory


def append_stage3_policy_guided_candidate_generation(
    memory: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    records = memory.setdefault("stage3_policy_guided_candidate_generation", [])
    if not isinstance(records, list):
        records = []
        memory["stage3_policy_guided_candidate_generation"] = records
    records.append(
        {
            "checkpoint_id": str(payload.get("checkpoint_id") or ""),
            "round_id": int(payload.get("round_id") or 0),
            "policy_guidance_used": bool(payload.get("policy_guidance_used", False)),
            "candidate_ids": [
                str(item.get("candidate_id") or "")
                for item in list(payload.get("candidates") or [])
            ],
            "candidate_types": [
                str(item.get("candidate_type") or "")
                for item in list(payload.get("candidates") or [])
            ],
            "evidence_keys_used": list(
                ((payload.get("candidate_generation_basis") or {}).get("evidence_keys"))
                or []
            ),
        }
    )
    return memory
