from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional

from rewarding.spec_renderer import render_reward_code_from_spec
from rewarding.spec_schema import get_default_reward_spec, validate_reward_spec
from scripts.run_reward_workflow import build_default_workflow_spec
from workflows.clients.openai_backend import OpenAIChatBackend
from workflows.storage import WorkflowStorage
from workflows.train_launcher import EPyMARLTrainLauncher


DEFAULT_FIELD_ROUNDS = ["beta", "wc"]
DEFAULT_RESULTS_ROOT = (
    Path(__file__).resolve().parents[2] / "results" / "stage_conditioned_workflows"
)
ROUND_RANKING_FIELDS = [
    "best_test_sparse_return_mean",
    "last_test_sparse_return_mean",
    "best_test_return_mean",
    "last_test_return_mean",
    "last_mixed_return_mean",
    "last_sparse_return_mean",
    "last_return_mean",
]
REQUIRED_STAGE_LABELS = ("early_exploration", "mid_progress", "late_plateau")
DEFAULT_FIELD_CRITIC_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "stage3_field_critic_prompt.md"
)
_FIELD_TO_PBRS_ENV_ARG = {
    "beta": "pbrs_beta",
    "wc": "pbrs_wc",
    "wp": "pbrs_wp",
}


def _build_stage_comparison_table(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    table: List[Dict[str, Any]] = []
    for record in records:
        table.append(
            {
                "candidate_id": record.get("candidate_id"),
                "candidate_value": record.get("candidate_value"),
                "best_test_sparse_return_mean": record.get("best_test_sparse_return_mean"),
                "last_test_sparse_return_mean": record.get("last_test_sparse_return_mean"),
                "best_test_return_mean": record.get("best_test_return_mean"),
                "last_test_return_mean": record.get("last_test_return_mean"),
                "last_sparse_return_mean": record.get("last_sparse_return_mean"),
                "last_mixed_return_mean": record.get("last_mixed_return_mean"),
                "last_return_mean": record.get("last_return_mean"),
                "run_id": record.get("run_id"),
            }
        )
    return table


def _extract_json_object(text: str) -> Dict[str, Any]:
    candidate_texts = [text.strip()]
    stripped = text.strip()
    if "```" in stripped:
        fence_start = stripped.find("```")
        while fence_start != -1:
            fence_end = stripped.find("```", fence_start + 3)
            if fence_end == -1:
                break
            block = stripped[fence_start + 3 : fence_end].strip()
            if "\n" in block:
                first_line, remainder = block.split("\n", 1)
                if first_line.strip().lower() == "json":
                    block = remainder.strip()
            candidate_texts.append(block)
            fence_start = stripped.find("```", fence_end + 3)
    left = stripped.find("{")
    right = stripped.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidate_texts.append(stripped[left : right + 1])

    last_error: Optional[Exception] = None
    for candidate in candidate_texts:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ValueError(f"Field critic output is not valid JSON: {last_error}")


def _build_field_critic_payload(
    *,
    round_id: int,
    field_name: str,
    shared_candidate_values: List[Any],
    stage_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "round_id": round_id,
        "field_name": field_name,
        "shared_candidate_values": list(shared_candidate_values),
        "stages": [
            {
                "stage_label": str((stage_run.get("checkpoint") or {}).get("stage_label")),
                "checkpoint_name": (stage_run.get("checkpoint") or {}).get("name"),
                "checkpoint_step": (stage_run.get("checkpoint") or {}).get("step"),
                "comparison_table": _build_stage_comparison_table(stage_run.get("records") or []),
            }
            for stage_run in stage_runs
        ],
    }


def _analyze_field_round_with_llm(
    *,
    round_id: int,
    field_name: str,
    shared_candidate_values: List[Any],
    stage_runs: List[Dict[str, Any]],
    api_key_env: str,
    base_url: str,
    model: str,
    temperature: float,
    llm_timeout: float,
    llm_max_retries: int,
    llm_retry_backoff: float,
) -> Dict[str, Any]:
    payload = _build_field_critic_payload(
        round_id=round_id,
        field_name=field_name,
        shared_candidate_values=shared_candidate_values,
        stage_runs=stage_runs,
    )
    prompt_spec = DEFAULT_FIELD_CRITIC_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = "\n".join(
        [
            "Prompt specification:",
            prompt_spec,
            "",
            "Field round payload:",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Return the required JSON now.",
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
            "You analyze one Stage 3 field-validation round and choose the best "
            "candidate value for each training stage. Return strictly valid JSON."
        ),
        user_prompt=prompt,
        metadata={
            "role": "stage3_field_critic",
            "round_id": round_id,
            "field_name": field_name,
        },
    )
    parsed = _extract_json_object(llm_result["text"])
    recommended = parsed.get("recommended_values_by_stage")
    if not isinstance(recommended, dict):
        raise ValueError("Field critic output must include recommended_values_by_stage as an object")
    missing = [label for label in REQUIRED_STAGE_LABELS if label not in recommended]
    if missing:
        raise ValueError(
            "Field critic output is missing stage recommendations for: " + ", ".join(missing)
        )
    return {
        "prompt": prompt,
        "response": {
            "raw_text": llm_result["text"],
            "raw_response": llm_result.get("raw_response"),
            "model": llm_result.get("model"),
            "usage": llm_result.get("usage"),
            "parsed": parsed,
        },
    }


def _build_field_round_summary(
    *,
    round_id: int,
    field_name: str,
    selected_checkpoints: List[Dict[str, Any]],
    shared_candidate_values: List[Any],
    stage_runs: List[Dict[str, Any]],
    recommended_values_by_stage: Dict[str, Any],
) -> Dict[str, Any]:
    expected_stage_labels = [str(checkpoint.get("stage_label")) for checkpoint in selected_checkpoints]
    observed_stage_labels = [
        str((stage_run.get("checkpoint") or {}).get("stage_label"))
        for stage_run in stage_runs
    ]
    missing_stage_labels = [
        stage_label for stage_label in expected_stage_labels if stage_label not in observed_stage_labels
    ]
    stage_comparisons = []
    for stage_run in stage_runs:
        checkpoint = stage_run.get("checkpoint") or {}
        stage_label = str(checkpoint.get("stage_label"))
        best_candidate = stage_run.get("best_candidate") or {}
        stage_comparisons.append(
            {
                "stage_label": stage_label,
                "checkpoint_name": checkpoint.get("name"),
                "checkpoint_step": checkpoint.get("step"),
                "candidate_count": stage_run.get("candidate_count"),
                "ranking_metric": stage_run.get("ranking_metric"),
                "best_candidate": best_candidate if best_candidate else None,
                "comparison_table": stage_run.get("comparison_table") or [],
                "recommended_value": recommended_values_by_stage.get(stage_label),
            }
        )
    readiness = {
        "has_exactly_three_stages": observed_stage_labels == list(REQUIRED_STAGE_LABELS),
        "shared_candidate_values": list(shared_candidate_values),
        "all_stages_have_multiple_candidates": all(
            int(stage_run.get("candidate_count") or 0) >= 2 for stage_run in stage_runs
        ),
        "all_stages_complete": all(
            len(stage_run.get("records") or []) == int(stage_run.get("candidate_count") or 0)
            for stage_run in stage_runs
        ),
        "all_stages_have_recommendations": all(
            stage_label in recommended_values_by_stage for stage_label in expected_stage_labels
        ),
        "missing_stage_labels": missing_stage_labels,
    }
    readiness["ready_for_final_synthesis"] = all(
        [
            readiness["has_exactly_three_stages"],
            readiness["all_stages_have_multiple_candidates"],
            readiness["all_stages_complete"],
            readiness["all_stages_have_recommendations"],
        ]
    )
    return {
        "round_id": round_id,
        "field_name": field_name,
        "expected_stage_labels": expected_stage_labels,
        "observed_stage_labels": observed_stage_labels,
        "shared_candidate_values": list(shared_candidate_values),
        "stage_comparisons": stage_comparisons,
        "field_recommendation": {
            "field_name": field_name,
            "recommended_values_by_stage": dict(recommended_values_by_stage),
        },
        "readiness": readiness,
    }


def _extract_metric_last(summary: Optional[Dict[str, Any]], metric_name: str) -> Any:
    if not isinstance(summary, dict):
        return None
    metric_summary = summary.get("metric_summary", {})
    payload = metric_summary.get(metric_name)
    if not isinstance(payload, dict):
        return None
    return payload.get("last_value")


def _extract_metric_best(summary: Optional[Dict[str, Any]], metric_name: str) -> Any:
    if not isinstance(summary, dict):
        return None
    metric_summary = summary.get("metric_summary", {})
    payload = metric_summary.get(metric_name)
    if not isinstance(payload, dict):
        return None
    return payload.get("best_value")


def _choose_ranking_metric(records: List[Dict[str, Any]]) -> Optional[str]:
    for field_name in ROUND_RANKING_FIELDS:
        values = [record.get(field_name) for record in records if record.get(field_name) is not None]
        if len(values) < 2:
            continue
        distinct = {round(float(value), 12) for value in values}
        if len(distinct) > 1:
            return field_name
    for field_name in ROUND_RANKING_FIELDS:
        if any(record.get(field_name) is not None for record in records):
            return field_name
    return None


def _candidate_id(stage_label: str, field_name: str, value: Any) -> str:
    if isinstance(value, float):
        suffix = str(value).replace(".", "p")
    else:
        suffix = str(value)
    return f"{stage_label}_{field_name}_{suffix}"


def _branch_result_has_usable_metrics(branch_result: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(branch_result, dict):
        return False
    metrics_summary = branch_result.get("metrics_summary")
    if not isinstance(metrics_summary, dict):
        return False
    metric_summary = metrics_summary.get("metric_summary")
    if not isinstance(metric_summary, dict):
        return False
    for metric_name in (
        "test_sparse_return_mean",
        "sparse_return_mean",
        "test_return_mean",
        "return_mean",
        "mixed_return_mean",
    ):
        payload = metric_summary.get(metric_name)
        if isinstance(payload, dict) and (
            payload.get("last_value") is not None or payload.get("best_value") is not None
        ):
            return True
    return False


def _recover_or_load_existing_branch_result(
    *,
    launcher: EPyMARLTrainLauncher,
    branch_plan: Dict[str, Any],
    branch_result_path: Path,
) -> Optional[Dict[str, Any]]:
    if branch_result_path.exists():
        try:
            branch_result = json.loads(branch_result_path.read_text(encoding="utf-8"))
        except Exception:
            branch_result = None
        if _branch_result_has_usable_metrics(branch_result):
            return branch_result

    try:
        recovered = launcher.recover_existing_result(branch_plan["resume_train_config"])
    except Exception:
        return None

    branch_result = {
        "train_config": recovered["train_config"],
        "run_reference": recovered["run_reference"],
        "metrics_summary": recovered.get("metrics_summary"),
    }
    if _branch_result_has_usable_metrics(branch_result):
        branch_result_path.parent.mkdir(parents=True, exist_ok=True)
        branch_result_path.write_text(
            json.dumps(branch_result, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return branch_result
    return None


def _load_run_status_from_reference(run_reference: Dict[str, Any]) -> Dict[str, Any]:
    run_json_path = Path(str(run_reference.get("run_json") or ""))
    if not run_json_path.exists():
        return {}
    try:
        return json.loads(run_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _candidate_process_is_alive(*, label: str) -> bool:
    if not label:
        return False
    try:
        process = subprocess.run(
            ["ps", "-efww"],
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return False
    if process.returncode != 0:
        return False
    needle = f"label={label}"
    return any(needle in line and "src/main.py" in line for line in process.stdout.splitlines())


def _run_reference_has_recent_heartbeat(
    run_reference: Dict[str, Any],
    *,
    max_age_seconds: float = 180.0,
) -> bool:
    heartbeat = run_reference.get("heartbeat")
    if not isinstance(heartbeat, str) or not heartbeat:
        return False
    try:
        heartbeat_dt = datetime.fromisoformat(heartbeat)
    except ValueError:
        return False
    if heartbeat_dt.tzinfo is None:
        heartbeat_dt = heartbeat_dt.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - heartbeat_dt.astimezone(timezone.utc)).total_seconds()
    return age_seconds <= max_age_seconds


def _select_preferred_candidate_run_reference(
    *,
    launcher: EPyMARLTrainLauncher,
    branch_plan: Dict[str, Any],
) -> Dict[str, Any]:
    train_config = branch_plan["resume_train_config"]
    try:
        run_references = launcher.find_matching_run_references(train_config)
    except Exception:
        run_references = []
    if not run_references:
        return {}

    def _sort_key(run_reference: Dict[str, Any]) -> tuple[int, int]:
        run_status = str(run_reference.get("status") or "")
        run_id = int(run_reference.get("run_id") or -1)
        if run_status == "COMPLETED":
            return (3, run_id)
        if run_status == "RUNNING":
            process_alive = _candidate_process_is_alive(label=str(train_config.get("label") or ""))
            heartbeat_alive = _run_reference_has_recent_heartbeat(run_reference)
            if heartbeat_alive or process_alive:
                return (2, run_id)
            return (1, run_id)
        return (0, run_id)

    return max(run_references, key=_sort_key)


def _classify_existing_candidate_state(
    *,
    launcher: EPyMARLTrainLauncher,
    branch_plan: Dict[str, Any],
    branch_result_path: Path,
) -> Dict[str, Any]:
    if branch_result_path.exists():
        try:
            branch_result = json.loads(branch_result_path.read_text(encoding="utf-8"))
        except Exception:
            branch_result = None
        if _branch_result_has_usable_metrics(branch_result):
            return {"status": "completed", "branch_result": branch_result}

    preferred_run_reference = _select_preferred_candidate_run_reference(
        launcher=launcher,
        branch_plan=branch_plan,
    )
    if not preferred_run_reference:
        return {"status": "planned", "branch_result": None}
    try:
        metrics_summary = launcher.log_analyzer.summarize(
            train_config=branch_plan["resume_train_config"],
            run_reference=preferred_run_reference,
        )
    except Exception:
        metrics_summary = None

    run_reference = preferred_run_reference
    run_status_payload = _load_run_status_from_reference(run_reference)
    run_status = str(run_status_payload.get("status") or "")
    label = str(branch_plan["resume_train_config"].get("label") or "")
    process_alive = _candidate_process_is_alive(label=label) if run_status == "RUNNING" else False
    heartbeat_alive = _run_reference_has_recent_heartbeat(run_reference) if run_status == "RUNNING" else False

    branch_result = {
        "train_config": branch_plan["resume_train_config"],
        "run_reference": run_reference,
        "metrics_summary": metrics_summary,
    }
    if run_status == "COMPLETED" and _branch_result_has_usable_metrics(branch_result):
        branch_result_path.parent.mkdir(parents=True, exist_ok=True)
        branch_result_path.write_text(
            json.dumps(branch_result, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "branch_result": branch_result,
            "run_reference": run_reference,
            "last_seen_sacred_status": run_status,
            "last_seen_process_alive": process_alive or heartbeat_alive,
        }

    if run_status == "RUNNING":
        if process_alive or heartbeat_alive:
            return {
                "status": "running",
                "branch_result": None,
                "run_reference": run_reference,
                "last_seen_sacred_status": run_status,
                "last_seen_process_alive": process_alive or heartbeat_alive,
            }
        return {
            "status": "planned",
            "branch_result": None,
            "run_reference": run_reference,
            "last_seen_sacred_status": run_status,
            "last_seen_process_alive": False,
        }

    return {
        "status": "planned",
        "branch_result": None,
        "run_reference": run_reference,
        "last_seen_sacred_status": run_status,
        "last_seen_process_alive": process_alive,
    }


def _build_candidate_reward_spec(
    *,
    base_pbrs: Dict[str, Any],
    field_name: str,
    candidate_value: Any,
) -> Dict[str, Any]:
    reward_spec = get_default_reward_spec()
    reward_spec = deepcopy(reward_spec)
    reward_spec.setdefault("pbrs", {})
    reward_spec["pbrs"].update(deepcopy(base_pbrs))
    reward_spec["alpha_policy"] = {"type": "constant", "value": 1.0}
    reward_spec["pbrs"][field_name] = candidate_value
    return validate_reward_spec(reward_spec)


def _resolve_field_candidate_values(
    workflow_spec: Dict[str, Any],
    *,
    field_name: str,
) -> List[Any]:
    tuning = workflow_spec.get("pbrs_tuning") or {}
    candidate_values = tuning.get("candidate_values") or {}
    values = candidate_values.get(field_name) if isinstance(candidate_values, dict) else None
    if isinstance(values, list) and values:
        return list(values)
    raise ValueError(f"no candidate values configured for field={field_name}")


def _build_result_record(
    *,
    field_name: str,
    stage_label: str,
    checkpoint: Dict[str, Any],
    candidate_id: str,
    candidate_value: Any,
    branch_result: Dict[str, Any],
) -> Dict[str, Any]:
    metrics_summary = branch_result.get("metrics_summary") or {}
    run_reference = branch_result.get("run_reference") or {}
    return {
        "field_name": field_name,
        "stage_label": stage_label,
        "checkpoint_name": checkpoint.get("name"),
        "checkpoint_step": checkpoint.get("step"),
        "candidate_id": candidate_id,
        "candidate_value": candidate_value,
        "run_id": run_reference.get("run_id"),
        "best_test_sparse_return_mean": _extract_metric_best(metrics_summary, "test_sparse_return_mean"),
        "last_test_sparse_return_mean": _extract_metric_last(metrics_summary, "test_sparse_return_mean"),
        "best_test_return_mean": _extract_metric_best(metrics_summary, "test_return_mean"),
        "last_test_return_mean": _extract_metric_last(metrics_summary, "test_return_mean"),
        "last_sparse_return_mean": _extract_metric_last(metrics_summary, "sparse_return_mean"),
        "last_mixed_return_mean": _extract_metric_last(metrics_summary, "mixed_return_mean"),
        "last_return_mean": _extract_metric_last(metrics_summary, "return_mean"),
    }


def _round_manifest_path(workflow_dir: Path, *, round_index: int, field_name: str) -> Path:
    return workflow_dir / f"round_{round_index:02d}_{field_name}_manifest.json"


def _build_round_manifest(
    *,
    workflow_id: str,
    round_index: int,
    field_name: str,
    selected_checkpoints: List[Dict[str, Any]],
    shared_candidate_values: List[Any],
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for checkpoint in selected_checkpoints:
        stage_label = str(checkpoint.get("stage_label"))
        checkpoint_name = str(checkpoint.get("name"))
        for candidate_value in shared_candidate_values:
            candidates.append(
                {
                    "candidate_id": _candidate_id(stage_label, field_name, candidate_value),
                    "candidate_value": candidate_value,
                    "stage_label": stage_label,
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_step": checkpoint.get("step"),
                    "status": "pending",
                    "run_id": None,
                    "launch_attempts": 0,
                    "last_seen_sacred_status": None,
                    "last_seen_process_alive": None,
                    "branch_result_path": None,
                    "failure_path": None,
                    "last_updated_at": None,
                }
            )
    return {
        "workflow_id": workflow_id,
        "round_id": round_index,
        "field_name": field_name,
        "shared_candidate_values": list(shared_candidate_values),
        "expected_stage_labels": [str(checkpoint.get("stage_label")) for checkpoint in selected_checkpoints],
        "candidates": candidates,
        "gates": {
            "dispatch_complete": False,
            "collection_complete": False,
            "summary_complete": False,
            "ready_to_advance": False,
        },
        "summary_paths": {
            "field_round_json": None,
            "field_round_summary_json": None,
            "field_round_comparison_json": None,
            "field_round_critic_response_json": None,
        },
    }


def _save_round_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _update_round_manifest_candidate(
    round_manifest: Dict[str, Any],
    *,
    candidate_id: str,
    status: str,
    branch_result_path: Optional[Path] = None,
    failure_path: Optional[Path] = None,
    run_id: Optional[Any] = None,
    last_seen_sacred_status: Optional[Any] = None,
    last_seen_process_alive: Optional[Any] = None,
    increment_launch_attempts: bool = False,
) -> None:
    for candidate in round_manifest.get("candidates") or []:
        if str(candidate.get("candidate_id")) != candidate_id:
            continue
        candidate["status"] = status
        candidate["branch_result_path"] = None if branch_result_path is None else str(branch_result_path)
        candidate["failure_path"] = None if failure_path is None else str(failure_path)
        candidate["run_id"] = run_id
        if last_seen_sacred_status is not None:
            candidate["last_seen_sacred_status"] = last_seen_sacred_status
        if last_seen_process_alive is not None:
            candidate["last_seen_process_alive"] = last_seen_process_alive
        if increment_launch_attempts:
            candidate["launch_attempts"] = int(candidate.get("launch_attempts") or 0) + 1
        candidate["last_updated_at"] = "now"
        return


def _refresh_round_manifest_gates(
    round_manifest: Dict[str, Any],
    *,
    summary_complete: bool,
) -> None:
    candidates = list(round_manifest.get("candidates") or [])
    terminal_statuses = {"completed", "failed"}
    gates = {
        "dispatch_complete": bool(candidates),
        "collection_complete": bool(candidates)
        and all(str(candidate.get("status")) in terminal_statuses for candidate in candidates),
        "summary_complete": bool(summary_complete),
        "ready_to_advance": bool(summary_complete),
    }
    round_manifest["gates"] = gates
    round_manifest["dispatch_complete"] = gates["dispatch_complete"]
    round_manifest["collection_complete"] = gates["collection_complete"]
    round_manifest["summary_complete"] = gates["summary_complete"]
    round_manifest["ready_to_advance"] = gates["ready_to_advance"]
    status_counts: Dict[str, int] = {}
    for candidate in candidates:
        status = str(candidate.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    round_manifest["status_counts"] = status_counts


def _load_round_manifest_gates(round_manifest: Dict[str, Any]) -> Dict[str, Any]:
    gates = round_manifest.get("gates")
    if isinstance(gates, dict) and gates:
        return gates
    return {
        "dispatch_complete": bool(round_manifest.get("dispatch_complete")),
        "collection_complete": bool(round_manifest.get("collection_complete")),
        "summary_complete": bool(round_manifest.get("summary_complete")),
        "ready_to_advance": bool(round_manifest.get("ready_to_advance")),
    }


def _launch_candidate_background_and_bind_run(
    *,
    launcher: EPyMARLTrainLauncher,
    branch_plan: Dict[str, Any],
    branch_result_path: Path,
) -> Dict[str, Any]:
    stdout_path = branch_result_path.with_name("launch_stdout.log")
    train_config = branch_plan["resume_train_config"]
    try:
        preexisting_run_references = launcher.find_matching_run_references(train_config)
    except Exception:
        preexisting_run_references = []
    preexisting_run_ids = {
        int(run_reference.get("run_id"))
        for run_reference in preexisting_run_references
        if run_reference.get("run_id") is not None
    }
    launch_payload = launcher.launch_prebuilt_train_config_background(
        train_config,
        stdout_path=stdout_path,
    )
    run_reference: Dict[str, Any] = {}
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            matching_run_references = launcher.find_matching_run_references(train_config)
        except Exception:
            matching_run_references = []
        new_run_references = [
            run_reference
            for run_reference in matching_run_references
            if run_reference.get("run_id") is not None
            and int(run_reference.get("run_id")) not in preexisting_run_ids
        ]
        if new_run_references:
            run_reference = max(
                new_run_references,
                key=lambda reference: (
                    int(reference.get("run_id") or -1),
                    1 if _run_reference_has_recent_heartbeat(reference) else 0,
                ),
            )
            heartbeat_alive = _run_reference_has_recent_heartbeat(run_reference)
            process_alive = _candidate_process_is_alive(label=str(train_config.get("label") or ""))
            if heartbeat_alive or process_alive:
                break
        pid = launch_payload.get("pid")
        if pid is not None:
            try:
                os.kill(int(pid), 0)
            except OSError:
                break
        time.sleep(0.5)
    if run_reference.get("run_id") is None:
        fallback_run_reference = _select_preferred_candidate_run_reference(
            launcher=launcher,
            branch_plan=branch_plan,
        )
        fallback_run_id = fallback_run_reference.get("run_id")
        if fallback_run_id is not None and int(fallback_run_id) not in preexisting_run_ids:
            run_reference = fallback_run_reference
    return {
        "launch_payload": launch_payload,
        "run_reference": run_reference,
    }


def _sanitize_round_manifest_candidates(
    round_manifest: Dict[str, Any],
) -> None:
    for candidate in round_manifest.get("candidates") or []:
        status = str(candidate.get("status") or "")
        run_id = candidate.get("run_id")
        branch_result_path = candidate.get("branch_result_path")
        if status != "running":
            continue
        if run_id is not None:
            continue
        if branch_result_path and Path(str(branch_result_path)).exists():
            continue
        candidate["status"] = "pending"


def _recover_existing_round_payload(
    *,
    workflow_dir: Path,
    round_index: int,
    field_name: str,
) -> Optional[Dict[str, Any]]:
    round_json_path = workflow_dir / f"field_round_{round_index:02d}_{field_name}.json"
    summary_json_path = workflow_dir / f"field_round_{round_index:02d}_{field_name}_summary.json"
    critic_json_path = workflow_dir / f"field_round_{round_index:02d}_{field_name}_critic_response.json"
    if not round_json_path.exists():
        return None
    try:
        round_payload = json.loads(round_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if summary_json_path.exists():
        try:
            round_payload["field_round_summary"] = json.loads(summary_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if critic_json_path.exists():
        try:
            round_payload["field_critic_result"] = json.loads(critic_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return round_payload


def build_stage_conditioned_branching_manifest(
    *,
    workflow_id: str,
    stage_selection_result: Dict[str, Any],
    python_executable: str,
    execute: bool,
    reuse_completed_candidates: bool,
    storage_root: Optional[str] = None,
    field_rounds: Optional[List[str]] = None,
    branch_budget_steps: Optional[int] = None,
    max_parallel_candidates: Optional[int] = None,
    branch_use_cuda: Optional[bool] = None,
    native_original_pbrs_branching: bool = False,
    eval_use_pbrs: bool = False,
    use_real_llm_for_field_analysis: bool = False,
    api_key_env: str = "IUSEAPI_API_KEY",
    base_url: str = "https://www.iuseapi.com/v1",
    model: str = "gpt-5.5",
    temperature: float = 0.2,
    llm_timeout: float = 60.0,
    llm_max_retries: int = 3,
    llm_retry_backoff: float = 5.0,
) -> Dict[str, Any]:
    storage = WorkflowStorage.create(
        workflow_id,
        root_dir=storage_root or DEFAULT_RESULTS_ROOT,
        manifest_data={"status": "running", "workflow_kind": "stage_conditioned_branching"},
    )
    workflow_spec = build_default_workflow_spec(3, reward_paradigm="pbrs")
    workflow_spec["workflow_id"] = workflow_id
    baseline_run_config_path = Path(stage_selection_result["baseline_run_config_json"])
    baseline_run_config = json.loads(baseline_run_config_path.read_text(encoding="utf-8"))
    checkpoint_root_dir = stage_selection_result["baseline_checkpoint_root_dir"]
    available_steps = stage_selection_result["baseline_available_checkpoint_steps"]
    selected_checkpoints = list((stage_selection_result.get("selection") or {}).get("selected_checkpoints", []))
    if len(selected_checkpoints) != 3:
        raise ValueError(
            "stage-conditioned branching requires exactly three selected checkpoints "
            "(early_exploration, mid_progress, late_plateau)"
        )
    selected_stage_labels = [str(checkpoint.get("stage_label")) for checkpoint in selected_checkpoints]
    if tuple(selected_stage_labels) != REQUIRED_STAGE_LABELS:
        raise ValueError(
            "stage-conditioned branching requires stage labels in exact order: "
            + ", ".join(REQUIRED_STAGE_LABELS)
        )
    launcher = EPyMARLTrainLauncher(
        repo_root=Path(__file__).resolve().parents[2],
        python_executable=python_executable,
    )
    field_rounds = list(field_rounds or DEFAULT_FIELD_ROUNDS)
    base_pbrs = deepcopy(((workflow_spec.get("pbrs_tuning") or {}).get("base_pbrs") or {}))
    baseline_t_max = int((baseline_run_config.get("t_max") or 0) or 0)
    fixed_branch_budget_steps = int(
        branch_budget_steps
        if branch_budget_steps is not None
        else max(1, baseline_t_max // 10)
    )

    manifest_path = storage.workflow_dir / "stage_conditioned_branching_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "workflow_id": workflow_id,
            "workflow_kind": "stage_conditioned_branching",
            "rounds": [],
        }
    manifest.update(
        {
            "workflow_id": workflow_id,
            "workflow_kind": "stage_conditioned_branching",
            "workflow_spec": workflow_spec,
            "stage_selection_result": stage_selection_result,
            "field_rounds": field_rounds,
            "baseline_run_config": baseline_run_config,
            "baseline_checkpoint_root_dir": checkpoint_root_dir,
            "baseline_available_checkpoint_steps": available_steps,
            "branch_budget": {
                "mode": "fixed_delta_t",
                "branch_budget_steps": fixed_branch_budget_steps,
                "baseline_t_max": baseline_t_max,
            },
            "execution": {
                "mode": "parallel" if execute else "planned_only",
                "max_parallel_candidates": int(max_parallel_candidates or 0) or None,
                "branch_use_cuda": branch_use_cuda,
            },
            "branching_mode": {
                "native_original_pbrs": bool(native_original_pbrs_branching),
                "eval_use_pbrs": bool(eval_use_pbrs),
            },
            "field_analysis": {
                "mode": "real_llm" if use_real_llm_for_field_analysis else "best_metric_only",
                "prompt_path": str(DEFAULT_FIELD_CRITIC_PROMPT_PATH),
            },
        }
    )
    storage.save_workflow_json("stage_conditioned_branching_manifest.json", manifest)
    existing_rounds_by_field = {
        str((round_payload or {}).get("field_name")): round_payload
        for round_payload in (manifest.get("rounds") or [])
        if isinstance((round_payload or {}).get("field_name"), str)
    }
    refreshed_rounds = []

    for round_index, field_name in enumerate(field_rounds, start=1):
        existing_round = existing_rounds_by_field.get(field_name)
        if existing_round is None:
            existing_round = _recover_existing_round_payload(
                workflow_dir=storage.workflow_dir,
                round_index=round_index,
                field_name=field_name,
            )
        existing_summary = (existing_round or {}).get("field_round_summary") or {}
        existing_ready = bool((existing_summary.get("readiness") or {}).get("ready_for_final_synthesis"))
        if existing_round is not None and existing_ready:
            existing_round = deepcopy(existing_round)
            existing_round["round_id"] = round_index
            refreshed_rounds.append(existing_round)
            manifest["rounds"] = list(refreshed_rounds)
            storage.save_workflow_json("stage_conditioned_branching_manifest.json", manifest)
            storage.update_manifest(
                {
                    "status": "running",
                    "field_round_count": len(refreshed_rounds),
                    "last_completed_field_round": field_name,
                }
            )
            continue

        stage_runs = []
        recommended_values_by_stage: Dict[str, Any] = {}
        shared_candidate_values = _resolve_field_candidate_values(
            workflow_spec,
            field_name=field_name,
        )
        if len(shared_candidate_values) < 2:
            raise ValueError(
                f"field round {field_name} must evaluate multiple candidate values across stages"
            )
        stage_runs_by_label: Dict[str, Dict[str, Any]] = {}
        execution_jobs: List[Dict[str, Any]] = []
        round_manifest_path = _round_manifest_path(storage.workflow_dir, round_index=round_index, field_name=field_name)
        if round_manifest_path.exists():
            round_manifest = json.loads(round_manifest_path.read_text(encoding="utf-8"))
        else:
            round_manifest = _build_round_manifest(
                workflow_id=workflow_id,
                round_index=round_index,
                field_name=field_name,
                selected_checkpoints=selected_checkpoints,
                shared_candidate_values=shared_candidate_values,
            )
        _sanitize_round_manifest_candidates(round_manifest)
        _refresh_round_manifest_gates(round_manifest, summary_complete=False)
        _save_round_manifest(round_manifest_path, round_manifest)

        for checkpoint in selected_checkpoints:
            stage_label = str(checkpoint.get("stage_label"))
            checkpoint_name = str(checkpoint.get("name"))
            candidate_values = list(shared_candidate_values)
            stage_dir = storage.workflow_dir / f"round_{round_index:02d}_{field_name}" / checkpoint_name
            stage_dir.mkdir(parents=True, exist_ok=True)
            candidate_payloads = []
            records = []
            stage_run_payload = {
                "checkpoint": checkpoint,
                "field_name": field_name,
                "candidate_values": candidate_values,
                "candidate_count": len(candidate_values),
                "ranking_metric": None,
                "best_candidate": None,
                "comparison_table": [],
                "records": records,
                "candidates": candidate_payloads,
            }
            stage_runs_by_label[stage_label] = stage_run_payload
            for candidate_value in candidate_values:
                candidate_id = _candidate_id(stage_label, field_name, candidate_value)
                candidate_dir = stage_dir / candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                reward_spec = None
                reward_spec_path = None
                reward_function_path = None
                override_env_args = None
                candidate_native_pbrs_config = None
                if native_original_pbrs_branching:
                    candidate_native_pbrs_config = {
                        "use_pbrs": True,
                        "eval_use_pbrs": bool(eval_use_pbrs),
                        "pbrs_variant": str(base_pbrs.get("variant", "original")),
                        "pbrs_gamma": float(base_pbrs.get("gamma", 0.99)),
                        "pbrs_beta": float(base_pbrs.get("beta", 0.3)),
                        "pbrs_wc": float(base_pbrs.get("wc", 0.6)),
                        "pbrs_wp": float(base_pbrs.get("wp", 0.4)),
                    }
                    candidate_native_pbrs_config[_FIELD_TO_PBRS_ENV_ARG[field_name]] = float(candidate_value)
                    override_env_args = dict(candidate_native_pbrs_config)
                    native_config_path = candidate_dir / "native_pbrs_config.json"
                    native_config_path.write_text(
                        json.dumps(candidate_native_pbrs_config, indent=2, sort_keys=True, ensure_ascii=False),
                        encoding="utf-8",
                    )
                else:
                    reward_spec = _build_candidate_reward_spec(
                        base_pbrs=base_pbrs,
                        field_name=field_name,
                        candidate_value=candidate_value,
                    )
                    reward_spec_path = candidate_dir / "reward_spec.json"
                    reward_function_path = candidate_dir / "reward_function.py"
                    reward_spec_path.write_text(
                        json.dumps(reward_spec, indent=2, sort_keys=True, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    reward_function_path.write_text(
                        render_reward_code_from_spec(reward_spec),
                        encoding="utf-8",
                    )
                branch_plan = launcher.build_checkpoint_resume_plan(
                    base_train_config=baseline_run_config,
                    run_reference={
                        "checkpoint_root_dir": checkpoint_root_dir,
                        "available_checkpoint_steps": available_steps,
                    },
                    requested_step=int(checkpoint["step"]),
                    label_suffix=f"_{workflow_id}_{field_name}_{checkpoint_name}_{candidate_id}",
                    use_dense_reward=False if native_original_pbrs_branching else True,
                    reward_module_path="" if native_original_pbrs_branching else str(reward_function_path),
                    alpha_policy_config={"type": "constant", "value": 1.0},
                    workflow_id=workflow_id,
                    workflow_round=round_index,
                    reward_paradigm="pbrs",
                    active_pbrs_field=field_name,
                    active_pbrs_field_source="field_round",
                    candidate_value=candidate_value,
                    active_checkpoint_context=checkpoint,
                    active_field_carryover_context=None,
                    candidate_selection_context={
                        "field_round": field_name,
                        "stage_label": stage_label,
                        "candidate_values": candidate_values,
                    },
                    phase_name="stage_conditioned_branch",
                    save_model=False,
                    save_model_interval=baseline_run_config.get("save_model_interval"),
                    local_results_path=baseline_run_config.get("local_results_path"),
                    override_env_args=override_env_args,
                override_overrides={
                    "t_max": int(checkpoint["step"]) + fixed_branch_budget_steps,
                    **({"use_cuda": bool(branch_use_cuda)} if branch_use_cuda is not None else {}),
                },
            )
                payload = {
                    "candidate_id": candidate_id,
                    "candidate_value": candidate_value,
                    "reward_spec_path": None if reward_spec_path is None else str(reward_spec_path),
                    "reward_function_path": None if reward_function_path is None else str(reward_function_path),
                    "native_pbrs_config": candidate_native_pbrs_config,
                    "branch_plan": branch_plan,
                }
                branch_result_path = candidate_dir / "branch_result.json"
                if execute:
                    if reuse_completed_candidates:
                        existing_state = _classify_existing_candidate_state(
                            launcher=launcher,
                            branch_plan=branch_plan,
                            branch_result_path=branch_result_path,
                        )
                        branch_result = existing_state.get("branch_result")
                        existing_status = str(existing_state.get("status") or "planned")
                        if branch_result is not None:
                            payload["branch_result"] = branch_result
                            payload["branch_result_status"] = (
                                "reused"
                                if branch_result_path.exists()
                                else "recovered_existing_run"
                            )
                            _update_round_manifest_candidate(
                                round_manifest,
                                candidate_id=candidate_id,
                                status="completed",
                                branch_result_path=branch_result_path,
                                run_id=((branch_result.get("run_reference") or {}).get("run_id")),
                                last_seen_sacred_status=existing_state.get("last_seen_sacred_status"),
                                last_seen_process_alive=existing_state.get("last_seen_process_alive"),
                            )
                            records.append(
                                _build_result_record(
                                    field_name=field_name,
                                    stage_label=stage_label,
                                    checkpoint=checkpoint,
                                    candidate_id=candidate_id,
                                    candidate_value=candidate_value,
                                    branch_result=payload["branch_result"],
                                )
                            )
                        elif existing_status == "running":
                            payload["branch_result_status"] = "running"
                            _update_round_manifest_candidate(
                                round_manifest,
                                candidate_id=candidate_id,
                                status="running",
                                branch_result_path=branch_result_path,
                                run_id=(((existing_state.get("run_reference") or {}).get("run_id"))),
                                last_seen_sacred_status=existing_state.get("last_seen_sacred_status"),
                                last_seen_process_alive=existing_state.get("last_seen_process_alive"),
                            )
                        else:
                            execution_jobs.append(
                                {
                                    "field_name": field_name,
                                    "stage_label": stage_label,
                                    "checkpoint": checkpoint,
                                    "candidate_id": candidate_id,
                                    "candidate_value": candidate_value,
                                    "branch_plan": branch_plan,
                                    "branch_result_path": branch_result_path,
                                    "payload": payload,
                                    "records": records,
                                }
                            )
                            _update_round_manifest_candidate(
                                round_manifest,
                                candidate_id=candidate_id,
                                status="planned",
                                branch_result_path=branch_result_path,
                                run_id=(((existing_state.get("run_reference") or {}).get("run_id"))),
                                last_seen_sacred_status=existing_state.get("last_seen_sacred_status"),
                                last_seen_process_alive=existing_state.get("last_seen_process_alive"),
                            )
                    else:
                        # If the structured branch_result artifact is absent, prefer rerunning the
                        # candidate directly instead of paying an expensive global Sacred scan first.
                        # This keeps resume/retry behavior responsive for partially completed rounds.
                        execution_jobs.append(
                            {
                                "field_name": field_name,
                                "stage_label": stage_label,
                                "checkpoint": checkpoint,
                                "candidate_id": candidate_id,
                                "candidate_value": candidate_value,
                                "branch_plan": branch_plan,
                                "branch_result_path": branch_result_path,
                                "payload": payload,
                                "records": records,
                            }
                        )
                        _update_round_manifest_candidate(
                            round_manifest,
                            candidate_id=candidate_id,
                            status="planned",
                            branch_result_path=branch_result_path,
                        )
                else:
                    payload["branch_result_status"] = "planned_only"
                    _update_round_manifest_candidate(
                        round_manifest,
                        candidate_id=candidate_id,
                        status="planned_only",
                        branch_result_path=branch_result_path,
                    )
                candidate_payloads.append(payload)
        _save_round_manifest(round_manifest_path, round_manifest)

        if execute and execution_jobs:
            max_workers = int(max_parallel_candidates or 0)
            if max_workers <= 0:
                max_workers = min(len(execution_jobs), len(shared_candidate_values))
            if native_original_pbrs_branching:
                max_workers = min(max_workers, 2)
            current_running = sum(
                1 for candidate in (round_manifest.get("candidates") or [])
                if str(candidate.get("status")) == "running"
            )
            dispatch_slots = max(0, max_workers - current_running)
            for job in execution_jobs[:dispatch_slots]:
                try:
                    launch_result = _launch_candidate_background_and_bind_run(
                        launcher=launcher,
                        branch_plan=job["branch_plan"],
                        branch_result_path=job["branch_result_path"],
                    )
                except Exception as exc:
                    failure_payload = {
                        "field_name": job["field_name"],
                        "stage_label": job["stage_label"],
                        "candidate_id": job["candidate_id"],
                        "candidate_value": job["candidate_value"],
                        "checkpoint": job["checkpoint"],
                        "error": str(exc),
                    }
                    failure_path = job["branch_result_path"].with_name("branch_failure.json")
                    failure_path.write_text(
                        json.dumps(failure_payload, indent=2, sort_keys=True, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    job["payload"]["branch_result_status"] = "failed_to_launch"
                    job["payload"]["execution_error"] = str(exc)
                    _update_round_manifest_candidate(
                        round_manifest,
                        candidate_id=job["candidate_id"],
                        status="failed",
                        branch_result_path=job["branch_result_path"],
                        failure_path=failure_path,
                        last_seen_sacred_status="LAUNCH_FAILED",
                        last_seen_process_alive=False,
                    )
                    continue

                run_reference = launch_result.get("run_reference") or {}
                job["payload"]["branch_result_status"] = "launched"
                job["payload"]["launch_payload"] = launch_result.get("launch_payload")
                _update_round_manifest_candidate(
                    round_manifest,
                    candidate_id=job["candidate_id"],
                    status="running",
                    branch_result_path=job["branch_result_path"],
                    run_id=run_reference.get("run_id"),
                    last_seen_sacred_status="RUNNING" if run_reference.get("run_id") is not None else None,
                    last_seen_process_alive=True if run_reference.get("run_id") is not None else None,
                    increment_launch_attempts=True,
                )
            _refresh_round_manifest_gates(round_manifest, summary_complete=False)
            _save_round_manifest(round_manifest_path, round_manifest)

        field_critic_result = None
        for checkpoint in selected_checkpoints:
            stage_label = str(checkpoint.get("stage_label"))
            stage_run = stage_runs_by_label[stage_label]
            records = stage_run["records"]
            ranking_metric = _choose_ranking_metric(records)
            best_candidate = None
            if records:
                if ranking_metric is None:
                    best_candidate = records[0]
                else:
                    best_candidate = max(
                        records,
                        key=lambda item: (
                            item.get(ranking_metric) is not None,
                            item.get(ranking_metric) or float("-inf"),
                        ),
                    )
            stage_run["ranking_metric"] = ranking_metric
            stage_run["best_candidate"] = best_candidate
            stage_run["comparison_table"] = _build_stage_comparison_table(records)
            if best_candidate is not None:
                recommended_values_by_stage[stage_label] = best_candidate.get("candidate_value")
            stage_runs.append(stage_run)

        all_stage_records_complete = all(
            len(stage_run.get("records") or []) == int(stage_run.get("candidate_count") or 0)
            for stage_run in stage_runs
        )

        if not all_stage_records_complete:
            partial_round_payload = {
                "round_id": round_index,
                "field_name": field_name,
                "shared_candidate_values": shared_candidate_values,
                "stage_coverage": [str(checkpoint.get("stage_label")) for checkpoint in selected_checkpoints],
                "stage_runs": stage_runs,
                "candidate_count_by_stage": {
                    str(stage_run.get("checkpoint", {}).get("stage_label")): stage_run.get("candidate_count")
                    for stage_run in stage_runs
                },
                "field_recommendation": None,
                "field_round_summary": None,
                "field_critic_result": None,
            }
            refreshed_rounds.append(partial_round_payload)
            manifest["rounds"] = list(refreshed_rounds)
            _refresh_round_manifest_gates(round_manifest, summary_complete=False)
            _save_round_manifest(round_manifest_path, round_manifest)
            storage.save_workflow_json("stage_conditioned_branching_manifest.json", manifest)
            storage.update_manifest(
                {
                    "status": "running",
                    "field_round_count": len(refreshed_rounds),
                    "active_field_round": field_name,
                    "active_gate": "collection",
                }
            )
            return manifest

        if use_real_llm_for_field_analysis and all_stage_records_complete:
            field_critic_result = _analyze_field_round_with_llm(
                round_id=round_index,
                field_name=field_name,
                shared_candidate_values=shared_candidate_values,
                stage_runs=stage_runs,
                api_key_env=api_key_env,
                base_url=base_url,
                model=model,
                temperature=temperature,
                llm_timeout=llm_timeout,
                llm_max_retries=llm_max_retries,
                llm_retry_backoff=llm_retry_backoff,
            )
            parsed = ((field_critic_result.get("response") or {}).get("parsed") or {})
            llm_recommended_values = parsed.get("recommended_values_by_stage") or {}
            recommended_values_by_stage = {
                stage_label: llm_recommended_values[stage_label]
                for stage_label in REQUIRED_STAGE_LABELS
            }

        field_round_summary = _build_field_round_summary(
            round_id=round_index,
            field_name=field_name,
            selected_checkpoints=selected_checkpoints,
            shared_candidate_values=shared_candidate_values,
            stage_runs=stage_runs,
            recommended_values_by_stage=recommended_values_by_stage,
        )
        round_payload = {
            "round_id": round_index,
            "field_name": field_name,
            "shared_candidate_values": shared_candidate_values,
            "stage_coverage": [str(checkpoint.get("stage_label")) for checkpoint in selected_checkpoints],
            "stage_runs": stage_runs,
            "candidate_count_by_stage": {
                str(stage_run.get("checkpoint", {}).get("stage_label")): stage_run.get("candidate_count")
                for stage_run in stage_runs
            },
            "field_recommendation": {
                "field_name": field_name,
                "recommended_values_by_stage": recommended_values_by_stage,
                "source": "real_llm" if field_critic_result is not None else "best_metric",
            },
            "field_round_summary": field_round_summary,
            "field_critic_result": field_critic_result,
        }
        refreshed_rounds.append(round_payload)
        manifest["rounds"] = list(refreshed_rounds)
        storage.save_workflow_json(
            f"field_round_{round_index:02d}_{field_name}.json",
            round_payload,
        )
        storage.save_workflow_json(
            f"field_round_{round_index:02d}_{field_name}_comparison.json",
            {
                "round_id": round_index,
                "field_name": field_name,
                "stage_comparisons": [
                    {
                        "stage_label": stage_run.get("checkpoint", {}).get("stage_label"),
                        "checkpoint_name": stage_run.get("checkpoint", {}).get("name"),
                        "checkpoint_step": stage_run.get("checkpoint", {}).get("step"),
                        "ranking_metric": stage_run.get("ranking_metric"),
                        "best_candidate": stage_run.get("best_candidate"),
                        "comparison_table": stage_run.get("comparison_table"),
                    }
                    for stage_run in stage_runs
                ],
                "field_recommendation": round_payload.get("field_recommendation"),
            },
        )
        storage.save_workflow_json(
            f"field_round_{round_index:02d}_{field_name}_summary.json",
            field_round_summary,
        )
        if field_critic_result is not None:
            storage.save_workflow_json(
                f"field_round_{round_index:02d}_{field_name}_critic_response.json",
                field_critic_result,
            )
            storage.save_text(
                round_index,
                f"field_round_{round_index:02d}_{field_name}_critic_prompt.txt",
                field_critic_result["prompt"],
            )
        round_manifest["summary_paths"] = {
            "field_round_json": str(storage.workflow_dir / f"field_round_{round_index:02d}_{field_name}.json"),
            "field_round_summary_json": str(storage.workflow_dir / f"field_round_{round_index:02d}_{field_name}_summary.json"),
            "field_round_comparison_json": str(storage.workflow_dir / f"field_round_{round_index:02d}_{field_name}_comparison.json"),
            "field_round_critic_response_json": (
                str(storage.workflow_dir / f"field_round_{round_index:02d}_{field_name}_critic_response.json")
                if field_critic_result is not None
                else None
            ),
        }
        _refresh_round_manifest_gates(
            round_manifest,
            summary_complete=bool((field_round_summary.get("readiness") or {}).get("ready_for_final_synthesis")),
        )
        _save_round_manifest(round_manifest_path, round_manifest)
        storage.save_workflow_json("stage_conditioned_branching_manifest.json", manifest)
        storage.update_manifest(
            {
                "status": "running",
                "field_round_count": len(refreshed_rounds),
                "last_completed_field_round": field_name,
            }
        )

    manifest["field_round_summaries"] = [
        round_payload.get("field_round_summary") for round_payload in (manifest.get("rounds") or [])
    ]
    manifest["readiness"] = {
        "field_round_count": len(field_rounds),
        "ready_for_final_synthesis": all(
            bool((summary or {}).get("readiness", {}).get("ready_for_final_synthesis"))
            for summary in manifest["field_round_summaries"]
        ),
    }
    storage.save_workflow_json("stage_conditioned_branching_manifest.json", manifest)
    storage.update_manifest({"status": "completed", "field_round_count": len(field_rounds)})
    return manifest
