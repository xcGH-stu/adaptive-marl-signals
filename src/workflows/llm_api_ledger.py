from __future__ import annotations

import inspect
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = ROOT / "results" / "llm_api_call_ledger.jsonl"
DEFAULT_SOFT_LIMITS = {
    "per_workflow": 25,
    "stage1b": 7,
    "stage3_checkpoint": 7,
}
API_KEY_FIELD_TOKENS = ("api_key", "api-key", "authorization", "bearer", "token")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_payload_hash(payload: Any) -> str | None:
    if payload is None:
        return None
    try:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        encoded = repr(payload)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            lower = key_text.lower()
            if any(token in lower for token in API_KEY_FIELD_TOKENS):
                continue
            sanitized[key_text] = _json_safe(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_jsonl_line(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _warning(message: str) -> None:
    print(f"[llm_api_ledger] warning: {message}")


def _safe_load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _workflow_manifest_candidates(workflow_dir: Path) -> Iterable[Path]:
    yield workflow_dir / "workflow_manifest.json"
    yield workflow_dir / "adaptive_checkpoint_replacement_manifest.json"


def _append_manifest_warning(workflow_dir: Path | None, warning_payload: Dict[str, Any]) -> None:
    if workflow_dir is None:
        return
    for manifest_path in _workflow_manifest_candidates(workflow_dir):
        if not manifest_path.exists():
            continue
        payload = _safe_load_json(manifest_path)
        if not isinstance(payload, dict):
            continue
        warnings = payload.setdefault("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
            payload["warnings"] = warnings
        if warning_payload not in warnings:
            warnings.append(_json_safe(warning_payload))
        workflow_summary = payload.setdefault("workflow_summary", {})
        if isinstance(workflow_summary, dict):
            workflow_summary.setdefault("llm_budget_warnings", [])
            if warning_payload not in workflow_summary["llm_budget_warnings"]:
                workflow_summary["llm_budget_warnings"].append(_json_safe(warning_payload))
        try:
            _safe_save_json(manifest_path, payload)
        except Exception as exc:
            _warning(f"failed to persist manifest warning at {manifest_path}: {exc}")


def _load_ledger_records(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    records: list[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    except Exception:
        return []
    return records


def _soft_budget_scope(stage: str, checkpoint: str | None) -> str | None:
    if stage == "stage1b":
        return "stage1b"
    if stage in {"stage3_c1", "stage3_c2"} or (checkpoint and checkpoint.upper() in {"C1", "C2"}):
        return "stage3_checkpoint"
    return None


def _budget_counts(records: list[Dict[str, Any]], workflow_id: str, stage: str, checkpoint: str | None) -> Dict[str, int]:
    relevant = [item for item in records if str(item.get("workflow_id") or "") == workflow_id]
    call_events = [item for item in relevant if str(item.get("event_state") or "") in {"attempt", "cache_hit", "fallback", "repair"}]
    stage1b_count = sum(1 for item in call_events if str(item.get("stage") or "") == "stage1b")
    checkpoint_upper = str(checkpoint or "").upper()
    stage3_checkpoint_count = sum(
        1
        for item in call_events
        if str(item.get("stage") or "") in {"stage3_c1", "stage3_c2"}
        and str(item.get("checkpoint") or "").upper() == checkpoint_upper
    )
    return {
        "per_workflow": len(call_events),
        "stage1b": stage1b_count,
        "stage3_checkpoint": stage3_checkpoint_count,
    }


def _infer_provider(provider: Any, model: Any, base_url: Any) -> str | None:
    if provider not in (None, ""):
        return str(provider)
    model_text = str(model or "").lower()
    url_text = str(base_url or "").lower()
    if "deepseek" in model_text or "deepseek" in url_text:
        return "deepseek"
    if "openai" in model_text or "openai" in url_text or "gpt-" in model_text:
        return "openai"
    return None


def _usage_value(usage: Dict[str, Any] | None, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def build_ledger_event(
    *,
    workflow_dir: Path | None = None,
    workflow_id: str | None = None,
    profile: str | None = None,
    env_key: str | None = None,
    algorithm: str | None = None,
    stage: str = "unknown",
    call_type: str,
    reason: str | None = None,
    checkpoint: str | None = None,
    round_id: int | None = None,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    cache_key: str | None = None,
    cache_hit: bool = False,
    payload: Any = None,
    payload_hash: str | None = None,
    usage: Dict[str, Any] | None = None,
    request_id: str | None = None,
    success: bool | None = None,
    repair_used: bool = False,
    repair_retry_count: int = 0,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    artifact_path: str | None = None,
    caller_file: str | None = None,
    caller_function: str | None = None,
    error_type: str | None = None,
    event_state: str = "attempt",
    record_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
    soft_limits: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    if caller_file is None or caller_function is None:
        frame = inspect.stack()[1]
        caller_file = caller_file or frame.filename
        caller_function = caller_function or frame.function
    provider_value = _infer_provider(provider, model, base_url)
    workflow_path = Path(workflow_dir).resolve() if workflow_dir is not None else None
    workflow_name = workflow_id or (workflow_path.name if workflow_path is not None else "")
    record = {
        "record_id": record_id or f"llm_api_{uuid4().hex}",
        "timestamp_utc": utc_now_iso(),
        "workflow_id": str(workflow_name or ""),
        "profile": None if profile in (None, "") else str(profile),
        "env_key": None if env_key in (None, "") else str(env_key),
        "algorithm": None if algorithm in (None, "") else str(algorithm),
        "stage": str(stage or "unknown"),
        "call_type": str(call_type or "unknown"),
        "reason": None if reason in (None, "") else str(reason),
        "checkpoint": None if checkpoint in (None, "") else str(checkpoint),
        "round": None if round_id is None else int(round_id),
        "model": None if model in (None, "") else str(model),
        "provider": provider_value,
        "cache_key": None if cache_key in (None, "") else str(cache_key),
        "cache_hit": bool(cache_hit),
        "payload_hash": payload_hash or stable_payload_hash(payload),
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
        "request_id": None if request_id in (None, "") else str(request_id),
        "success": success,
        "repair_used": bool(repair_used),
        "repair_retry_count": int(repair_retry_count or 0),
        "fallback_used": bool(fallback_used),
        "fallback_reason": None if fallback_reason in (None, "") else str(fallback_reason),
        "artifact_path": None if artifact_path in (None, "") else str(artifact_path),
        "caller_file": str(caller_file or ""),
        "caller_function": str(caller_function or ""),
        "error_type": None if error_type in (None, "") else str(error_type),
        "event_state": str(event_state or "attempt"),
        "budget_warning": False,
        "warning_scope": None,
        "metadata": _json_safe(metadata or {}),
    }
    limits = dict(DEFAULT_SOFT_LIMITS)
    limits.update(soft_limits or {})
    root_records = _load_ledger_records(DEFAULT_LEDGER_PATH)
    counts = _budget_counts(root_records, record["workflow_id"], record["stage"], record["checkpoint"])
    counts["per_workflow"] += 1
    budget_scope = _soft_budget_scope(record["stage"], record["checkpoint"])
    if counts["per_workflow"] > int(limits["per_workflow"]):
        record["budget_warning"] = True
        record["warning_scope"] = "per_workflow"
    elif budget_scope == "stage1b" and counts["stage1b"] + 1 > int(limits["stage1b"]):
        record["budget_warning"] = True
        record["warning_scope"] = "stage1b"
    elif budget_scope == "stage3_checkpoint" and counts["stage3_checkpoint"] + 1 > int(limits["stage3_checkpoint"]):
        record["budget_warning"] = True
        record["warning_scope"] = "stage3_checkpoint"
    return record


def write_ledger_event(**kwargs: Any) -> Dict[str, Any]:
    workflow_dir = kwargs.get("workflow_dir")
    workflow_path = Path(workflow_dir).resolve() if workflow_dir is not None else None
    record = build_ledger_event(**kwargs)
    targets = [DEFAULT_LEDGER_PATH]
    if workflow_path is not None:
        targets.append(workflow_path / "llm_api_call_ledger.jsonl")
    for target in targets:
        try:
            _write_jsonl_line(target, record)
        except Exception as exc:
            _warning(f"failed to append ledger event to {target}: {exc}")
    if record.get("budget_warning"):
        warning_payload = {
            "type": "llm_api_budget_warning",
            "scope": record.get("warning_scope"),
            "workflow_id": record.get("workflow_id"),
            "stage": record.get("stage"),
            "checkpoint": record.get("checkpoint"),
            "record_id": record.get("record_id"),
        }
        _append_manifest_warning(workflow_path, warning_payload)
    return record


def extract_request_id(result: Dict[str, Any] | None) -> str | None:
    if not isinstance(result, dict):
        return None
    direct = result.get("request_id")
    if direct not in (None, ""):
        return str(direct)
    raw = result.get("raw_response")
    if isinstance(raw, dict):
        value = raw.get("id") or raw.get("request_id")
        if value not in (None, ""):
            return str(value)
    return None

