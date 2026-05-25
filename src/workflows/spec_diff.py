from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


def summarize_reward_spec_diff(
    base_spec: Dict[str, Any] | None,
    candidate_spec: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(base_spec, dict) or not isinstance(candidate_spec, dict):
        return {
            "changed_terms": [],
            "pbrs_changes": {},
            "alpha_policy_changed": False,
        }

    changed_terms: List[Dict[str, Any]] = []
    base_terms = {
        str(term.get("name")): term
        for term in base_spec.get("terms", [])
        if isinstance(term, dict) and term.get("name") is not None
    }
    candidate_terms = {
        str(term.get("name")): term
        for term in candidate_spec.get("terms", [])
        if isinstance(term, dict) and term.get("name") is not None
    }

    for term_name in sorted(set(base_terms.keys()) | set(candidate_terms.keys())):
        base_term = deepcopy(base_terms.get(term_name) or {})
        candidate_term = deepcopy(candidate_terms.get(term_name) or {})
        term_changes: Dict[str, Any] = {"name": term_name}
        changed = False
        for field_name in ("enabled", "weight", "scope", "trigger_variant", "constraints"):
            base_value = base_term.get(field_name)
            candidate_value = candidate_term.get(field_name)
            if base_value != candidate_value:
                term_changes[field_name] = {
                    "before": base_value,
                    "after": candidate_value,
                }
                changed = True
        if changed:
            changed_terms.append(term_changes)

    pbrs_changes = _diff_nested_dict(
        base_spec.get("pbrs") if isinstance(base_spec.get("pbrs"), dict) else {},
        candidate_spec.get("pbrs") if isinstance(candidate_spec.get("pbrs"), dict) else {},
    )

    return {
        "changed_terms": changed_terms,
        "pbrs_changes": pbrs_changes,
        "alpha_policy_changed": base_spec.get("alpha_policy") != candidate_spec.get("alpha_policy"),
    }


def _diff_nested_dict(
    base_value: Dict[str, Any],
    candidate_value: Dict[str, Any],
    *,
    prefix: str = "",
) -> Dict[str, Any]:
    changes: Dict[str, Any] = {}
    for key in sorted(set(base_value.keys()) | set(candidate_value.keys())):
        base_item = base_value.get(key)
        candidate_item = candidate_value.get(key)
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(base_item, dict) and isinstance(candidate_item, dict):
            nested = _diff_nested_dict(base_item, candidate_item, prefix=path)
            changes.update(nested)
            continue
        if base_item != candidate_item:
            changes[path] = {
                "before": base_item,
                "after": candidate_item,
            }
    return changes
