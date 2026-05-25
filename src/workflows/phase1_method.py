from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.select_pbrs_checkpoints import build_baseline_summary
from workflows.clients.openai_backend import OpenAIChatBackend

DEFAULT_INITIAL_DENSE_CONFIG_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "initial_dense_config_prompt.md"
)


def build_dense_reference_selection_payload(
    *,
    workflow_id: str,
    sparse_run_summary: Dict[str, Any],
    dense_reference_config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "goal": (
            "Select one fixed native original PBRS dense-reference configuration "
            "(beta and wc, with wp derived as 1-wc) for boundary calibration. "
            "The chosen configuration should be stable and likely to exhibit a "
            "complete learning progression, rather than being chosen purely for "
            "peak final return."
        ),
        "sparse_run_summary": deepcopy(sparse_run_summary),
        "dense_reference_config": deepcopy(dense_reference_config),
    }


def select_dense_reference_with_llm(
    *,
    workflow_id: str,
    sparse_run_summary: Dict[str, Any],
    dense_reference_config: Dict[str, Any],
    api_key_env: str,
    base_url: str,
    model: str,
    temperature: float,
    llm_timeout: float,
    llm_max_retries: int,
    llm_retry_backoff: float,
) -> Dict[str, Any]:
    payload = build_dense_reference_selection_payload(
        workflow_id=workflow_id,
        sparse_run_summary=sparse_run_summary,
        dense_reference_config=dense_reference_config,
    )
    prompt_spec = _load_initial_dense_config_prompt_spec()
    prompt = "\n".join(
        [
            "Prompt specification:",
            prompt_spec,
            "",
            "Payload:",
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            "",
            "Return strict JSON only.",
        ]
    )
    backend = OpenAIChatBackend(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=llm_timeout,
        max_retries=llm_max_retries,
        retry_backoff_seconds=llm_retry_backoff,
    )
    llm_result = backend.generate_text(
        system_prompt=(
            "Select a dense-reference native original PBRS configuration for a Phase 1 MARL workflow. "
            "Return strictly valid JSON only."
        ),
        user_prompt=prompt,
        metadata={
            "role": "phase1_dense_reference_selector",
            "workflow_id": workflow_id,
        },
    )
    parsed = _extract_json_dict(llm_result.get("text") or "")
    selected_beta, selected_wc, selected_wp = _extract_initial_dense_config(parsed)
    valid_beta_values = list(dense_reference_config.get("candidate_beta_values") or [])
    valid_wc_values = list(dense_reference_config.get("candidate_wc_values") or [])
    if selected_beta not in valid_beta_values:
        raise ValueError(
            "dense reference selector chose a beta value outside candidate_beta_values: "
            f"{selected_beta!r}"
        )
    if selected_wc not in valid_wc_values:
        raise ValueError(
            "dense reference selector chose a wc value outside candidate_wc_values: "
            f"{selected_wc!r}"
        )
    return {
        "workflow_id": workflow_id,
        "selection_mode": "llm_select_fixed_beta_wc",
        "dense_reference_config": deepcopy(dense_reference_config),
        "selected_beta": selected_beta,
        "selected_wc": selected_wc,
        "selected_wp": selected_wp,
        "expected_role": parsed.get("expected_role"),
        "risk": parsed.get("risk"),
        "candidate_search_prior": deepcopy(parsed.get("candidate_search_prior") or {}),
        "stage_hypotheses": deepcopy(parsed.get("stage_hypotheses") or {}),
        "reason": parsed.get("reason") or parsed.get("expected_role"),
        "selection_goal": parsed.get("selection_goal") or parsed.get("expected_role"),
        "prompt_path": str(DEFAULT_INITIAL_DENSE_CONFIG_PROMPT_PATH),
        "response": {
            "raw_text": llm_result.get("text"),
            "raw_response": llm_result.get("raw_response"),
            "model": llm_result.get("model"),
            "usage": llm_result.get("usage"),
            "parsed": parsed,
        },
    }


def build_dense_validation_candidates(
    *,
    final_spec: Dict[str, Any],
    field_round_summaries: List[Dict[str, Any]],
    top_k_schedules: int = 3,
) -> List[Dict[str, Any]]:
    if top_k_schedules < 1:
        return []
    base_schedule = deepcopy(final_spec.get("native_original_pbrs_stage_schedule") or {})
    if not isinstance(base_schedule, dict) or not base_schedule:
        return []

    candidates: List[Tuple[str, Dict[str, Any]]] = [("recommended_schedule", deepcopy(base_schedule))]
    perturbations = _collect_stagewise_alternatives(field_round_summaries=field_round_summaries)
    used_signatures = {_schedule_signature(base_schedule)}

    for field_name, stage_label, alt_value in perturbations:
        mutated = deepcopy(base_schedule)
        stage_key = _stage_key_from_label(stage_label)
        if stage_key is None:
            continue
        try:
            mutated[field_name][stage_key] = alt_value
        except Exception:
            continue
        if field_name == "wc" and "wp" in mutated:
            mutated["wp"][stage_key] = round(1.0 - float(alt_value), 10)
        signature = _schedule_signature(mutated)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        candidates.append((f"alt_{field_name}_{stage_label}_{alt_value}", mutated))
        if len(candidates) >= top_k_schedules:
            break

    payloads: List[Dict[str, Any]] = []
    for rank, (name, schedule) in enumerate(candidates[:top_k_schedules], start=1):
        payloads.append(
            {
                "candidate_rank": rank,
                "candidate_name": name,
                "native_original_pbrs_stage_schedule": schedule,
            }
        )
    return payloads


def build_dense_reference_artifact(
    *,
    workflow_id: str,
    sparse_run_summary: Dict[str, Any],
    selection_result: Dict[str, Any],
    run_plan: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "source_sparse_run_summary": deepcopy(sparse_run_summary),
        "dense_reference_selection": deepcopy(selection_result),
        "dense_reference_run": deepcopy(run_plan),
    }


def build_run_summary_from_run_dir(
    *,
    run_dir: str | Path,
    preferred_metrics: Optional[List[str]] = None,
    max_points_per_metric: int = 15,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    metrics = run_path / "metrics.json"
    config = run_path / "config.json"
    summary = build_baseline_summary(
        metrics_json_path=metrics,
        run_config_path=config,
        preferred_metrics=preferred_metrics
        or [
            "test_sparse_return_mean",
            "sparse_return_mean",
            "test_return_mean",
            "return_mean",
        ],
        max_points_per_metric=max_points_per_metric,
    )
    summary["run_dir"] = str(run_path)
    return summary


def materialize_dense_validation_final_specs(
    *,
    base_final_spec: Dict[str, Any],
    dense_validation_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for candidate in dense_validation_candidates:
        candidate_name = str(candidate.get("candidate_name") or f"candidate_{len(payloads) + 1}")
        schedule = deepcopy(candidate.get("native_original_pbrs_stage_schedule") or {})
        if not schedule:
            continue
        spec_payload = deepcopy(base_final_spec)
        spec_payload["native_original_pbrs_stage_schedule"] = schedule
        payloads.append(
            {
                "candidate_rank": int(candidate.get("candidate_rank") or (len(payloads) + 1)),
                "candidate_name": candidate_name,
                "final_spec": spec_payload,
            }
        )
    return payloads


def _collect_stagewise_alternatives(
    *,
    field_round_summaries: List[Dict[str, Any]],
) -> List[Tuple[str, str, Any]]:
    alternatives: List[Tuple[str, str, Any]] = []
    for round_summary in field_round_summaries:
        if not isinstance(round_summary, dict):
            continue
        field_name = str(round_summary.get("field_name") or "")
        recommendation = (
            (round_summary.get("field_recommendation") or {}).get("recommended_values_by_stage")
            or {}
        )
        for stage_comparison in round_summary.get("stage_comparisons") or []:
            if not isinstance(stage_comparison, dict):
                continue
            stage_label = str(stage_comparison.get("stage_label") or "")
            recommended_value = recommendation.get(stage_label)
            comparison_table = list(stage_comparison.get("comparison_table") or [])
            ranked = sorted(
                comparison_table,
                key=lambda row: (
                    _score_value(row.get("best_test_sparse_return_mean")),
                    _score_value(row.get("last_test_sparse_return_mean")),
                ),
                reverse=True,
            )
            for row in ranked:
                candidate_value = row.get("candidate_value")
                if candidate_value == recommended_value:
                    continue
                alternatives.append((field_name, stage_label, candidate_value))
                break
    return alternatives


def _score_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _stage_key_from_label(stage_label: str) -> Optional[str]:
    mapping = {
        "early_exploration": "early",
        "mid_progress": "mid",
        "late_plateau": "late",
    }
    return mapping.get(stage_label)


def _schedule_signature(schedule: Dict[str, Any]) -> str:
    return json.dumps(schedule, sort_keys=True, ensure_ascii=False)


def _extract_json_dict(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        start = stripped.find("```")
        while start != -1:
            end = stripped.find("```", start + 3)
            if end == -1:
                break
            block = stripped[start + 3 : end].strip()
            if "\n" in block:
                first, remainder = block.split("\n", 1)
                if first.strip().lower() == "json":
                    block = remainder.strip()
            candidates.append(block)
            start = stripped.find("```", end + 3)
    left = stripped.find("{")
    right = stripped.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidates.append(stripped[left : right + 1])

    last_error: Optional[Exception] = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ValueError(f"Expected a JSON object, got invalid response: {last_error}")


def _load_initial_dense_config_prompt_spec() -> str:
    if DEFAULT_INITIAL_DENSE_CONFIG_PROMPT_PATH.exists():
        return DEFAULT_INITIAL_DENSE_CONFIG_PROMPT_PATH.read_text(encoding="utf-8")
    return "\n".join(
        [
            "You are selecting a fixed dense-reference native original PBRS configuration.",
            "Choose one candidate pair (beta, wc); wp is derived automatically as 1-wc.",
            "The selected pair should be most suitable as a stable dense reference trajectory",
            "for recalibrating training-stage boundaries.",
            "Return strict JSON only.",
        ]
    )


def _extract_initial_dense_config(parsed: Dict[str, Any]) -> Tuple[Any, Any, float]:
    initial_config = parsed.get("initial_config")
    if isinstance(initial_config, dict):
        selected_beta = initial_config.get("beta")
        selected_wc = initial_config.get("wc")
        selected_wp = initial_config.get("wp")
    else:
        selected_beta = parsed.get("selected_beta")
        selected_wc = parsed.get("selected_wc")
        selected_wp = None
    if selected_wp is None and selected_wc is not None:
        selected_wp = round(1.0 - float(selected_wc), 10)
    elif selected_wp is not None:
        selected_wp = round(float(selected_wp), 10)
    return selected_beta, selected_wc, selected_wp
