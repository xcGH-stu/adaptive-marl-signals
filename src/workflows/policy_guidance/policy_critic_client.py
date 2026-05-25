from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
from workflows.llm_api_ledger import extract_request_id
from workflows.llm_api_ledger import stable_payload_hash
from workflows.llm_api_ledger import write_ledger_event

from workflows.clients.openai_backend import OpenAIChatBackend

from .policy_guidance_schema import (
    build_fallback_integrated_guidance_card,
    validate_integrated_guidance_card,
    validate_policy_behavior_summary,
)


PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts" / "policy_guidance"
PROMPT_PATHS = {
    "stage1_after_sparse_baseline": PROMPTS_ROOT / "policy_stage1_sparse_diagnosis_prompt.md",
    "stage3_c1_pre_checkpoint": PROMPTS_ROOT / "policy_stage3_checkpoint_diagnosis_prompt.md",
    "stage3_c2_pre_checkpoint": PROMPTS_ROOT / "policy_stage3_checkpoint_diagnosis_prompt.md",
    "stage3_c1_after_round1": PROMPTS_ROOT / "policy_stage3_round1_analysis_prompt.md",
    "stage3_c2_after_round1": PROMPTS_ROOT / "policy_stage3_round1_analysis_prompt.md",
    "integrated_guidance_card": PROMPTS_ROOT / "integrated_guidance_card_prompt.md",
}


class PolicyCriticClient:
    def __init__(
        self,
        *,
        model: str = "gpt-5.2",
        mock_llm_mode: bool = True,
        fallback_to_reward_only: bool = True,
        backend: Optional[OpenAIChatBackend] = None,
        repair_retry_count: int = 0,
        reuse_cache: bool = False,
        cache_root: Optional[Path] = None,
        cache_lookup_mode: str = "exact_or_latest_by_intervention",
        workflow_dir: Optional[Path] = None,
        workflow_id: str = "",
        profile: str = "",
        env_key: str = "",
        algorithm: str = "",
    ) -> None:
        self.model = str(model or "gpt-5.2")
        self.mock_llm_mode = bool(mock_llm_mode)
        self.fallback_to_reward_only = bool(fallback_to_reward_only)
        self.backend = backend
        self.repair_retry_count = max(0, int(repair_retry_count))
        self.reuse_cache = bool(reuse_cache)
        self.cache_root = Path(cache_root).resolve() if cache_root else None
        self.cache_lookup_mode = str(cache_lookup_mode or "exact_or_latest_by_intervention")
        self.workflow_dir = Path(workflow_dir).resolve() if workflow_dir is not None else None
        self.workflow_id = str(workflow_id or "")
        self.profile = str(profile or "")
        self.env_key = str(env_key or "")
        self.algorithm = str(algorithm or "")

    def _stage_for_intervention_point(self, intervention_point: str) -> str:
        point = str(intervention_point or "").lower()
        if point == "stage1_after_sparse_baseline":
            return "policy_guidance"
        if "stage3_c1" in point:
            return "stage3_c1"
        if "stage3_c2" in point:
            return "stage3_c2"
        return "policy_guidance"

    def _checkpoint_for_intervention_point(self, intervention_point: str) -> str | None:
        point = str(intervention_point or "").lower()
        if "stage3_c1" in point:
            return "C1"
        if "stage3_c2" in point:
            return "C2"
        return None

    def _round_from_branch_results(self, branch_results: list[Dict[str, Any]]) -> int | None:
        if not branch_results:
            return None
        return 1

    def _log_event(
        self,
        *,
        intervention_point: str,
        call_type: str,
        reason: str,
        event_state: str,
        cache_hit: bool = False,
        payload: Any = None,
        usage: Dict[str, Any] | None = None,
        request_id: str | None = None,
        success: bool | None = None,
        repair_used: bool = False,
        repair_retry_count: int = 0,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        artifact_path: str | None = None,
        error_type: str | None = None,
        record_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
        round_id: int | None = None,
    ) -> Dict[str, Any]:
        return write_ledger_event(
            workflow_dir=self.workflow_dir,
            workflow_id=self.workflow_id,
            profile=self.profile,
            env_key=self.env_key,
            algorithm=self.algorithm,
            stage=self._stage_for_intervention_point(intervention_point),
            call_type=call_type,
            reason=reason,
            checkpoint=self._checkpoint_for_intervention_point(intervention_point),
            round_id=round_id,
            model=self.model,
            provider=(metadata or {}).get("provider"),
            base_url=((metadata or {}).get("backend_metadata") or {}).get("base_url"),
            cache_key=(metadata or {}).get("cache_key"),
            cache_hit=cache_hit,
            payload=payload,
            payload_hash=stable_payload_hash(payload),
            usage=usage,
            request_id=request_id,
            success=success,
            repair_used=repair_used,
            repair_retry_count=repair_retry_count,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            artifact_path=artifact_path,
            caller_file=__file__,
            caller_function="PolicyCriticClient.generate_integrated_guidance_card",
            error_type=error_type,
            event_state=event_state,
            record_id=record_id,
            metadata=metadata,
        )

    def generate_integrated_guidance_card(
        self,
        *,
        intervention_point: str,
        reward_diagnosis: Dict[str, Any],
        behavior_summary: Dict[str, Any],
        branch_results: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        validated_summary = validate_policy_behavior_summary(behavior_summary)
        call_record: Dict[str, Any] = {
            "intervention_point": str(intervention_point),
            "mock_llm_mode": self.mock_llm_mode,
            "fallback_used": False,
            "failure_reason": "",
            "prompt_path": str(
                PROMPT_PATHS.get(intervention_point)
                or PROMPT_PATHS["integrated_guidance_card"]
            ),
            "integrated_guidance_card": {},
        }
        ledger_payload = {
            "intervention_point": str(intervention_point),
            "reward_diagnosis": deepcopy(reward_diagnosis or {}),
            "behavior_summary": deepcopy(validated_summary or {}),
            "branch_results": deepcopy(branch_results or []),
        }
        round_id = self._round_from_branch_results(branch_results or [])
        cache_hit = self._load_cached_card(
            intervention_point=intervention_point,
            reward_diagnosis=reward_diagnosis,
            behavior_summary=validated_summary,
            branch_results=branch_results or [],
        )
        if cache_hit is not None:
            cached_card, cached_record = cache_hit
            call_record.update(cached_record)
            call_record["integrated_guidance_card"] = cached_card
            call_record["ledger_event"] = self._log_event(
                intervention_point=intervention_point,
                call_type="cache_hit",
                reason=f"policy guidance cache hit for {intervention_point}",
                event_state="cache_hit",
                cache_hit=True,
                payload=ledger_payload,
                usage=cached_record.get("usage"),
                request_id=extract_request_id(cached_record),
                success=True,
                fallback_used=bool(cached_record.get("fallback_used", False)),
                fallback_reason=cached_record.get("failure_reason"),
                metadata={"cache_key": cached_record.get("cache_dir"), "backend_metadata": cached_record.get("backend_metadata") or {}},
                round_id=round_id,
            )
            return {
                "integrated_guidance_card": cached_card,
                "call_record": call_record,
            }
        if not self.mock_llm_mode:
            return self._generate_real_integrated_guidance_card(
                intervention_point=intervention_point,
                reward_diagnosis=reward_diagnosis,
                behavior_summary=validated_summary,
                branch_results=branch_results or [],
                call_record=call_record,
            )

        if validated_summary.get("insufficient_evidence", False):
            fallback = build_fallback_integrated_guidance_card(
                intervention_point=intervention_point,
                failure_reason=validated_summary.get("insufficient_evidence_reason")
                or "insufficient policy behavior evidence",
                reward_diagnosis=reward_diagnosis,
            )
            call_record["fallback_used"] = self.fallback_to_reward_only
            call_record["failure_reason"] = fallback["metadata"]["failure_reason"]
            call_record["integrated_guidance_card"] = fallback
            call_record["ledger_event"] = self._log_event(
                intervention_point=intervention_point,
                call_type="fallback",
                reason=f"insufficient evidence fallback for {intervention_point}",
                event_state="fallback",
                payload=ledger_payload,
                success=None,
                fallback_used=True,
                fallback_reason=fallback["metadata"]["failure_reason"],
                round_id=round_id,
            )
            self._persist_cached_card(
                intervention_point=intervention_point,
                prompt_payload=ledger_payload,
                call_record=call_record,
                integrated_guidance_card=fallback,
            )
            return {
                "integrated_guidance_card": fallback,
                "call_record": call_record,
            }

        card = self._build_mock_integrated_guidance_card(
            intervention_point=intervention_point,
            reward_diagnosis=reward_diagnosis,
            behavior_summary=validated_summary,
            branch_results=branch_results or [],
        )
        validated_card = validate_integrated_guidance_card(card)
        call_record["integrated_guidance_card"] = validated_card
        call_record["ledger_event"] = self._log_event(
            intervention_point=intervention_point,
            call_type="guidance",
            reason=f"mock policy guidance for {intervention_point}",
            event_state="success",
            payload=ledger_payload,
            success=None,
            fallback_used=False,
            round_id=round_id,
        )
        self._persist_cached_card(
            intervention_point=intervention_point,
            prompt_payload=ledger_payload,
            call_record=call_record,
            integrated_guidance_card=validated_card,
        )
        return {
            "integrated_guidance_card": validated_card,
            "call_record": call_record,
        }

    def _generate_real_integrated_guidance_card(
        self,
        *,
        intervention_point: str,
        reward_diagnosis: Dict[str, Any],
        behavior_summary: Dict[str, Any],
        branch_results: list[Dict[str, Any]],
        call_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        fallback_prompt_payload = {
            "intervention_point": str(intervention_point),
            "reward_diagnosis": deepcopy(reward_diagnosis or {}),
            "behavior_summary": deepcopy(behavior_summary or {}),
            "branch_results": deepcopy(branch_results or []),
        }
        if self.backend is None:
            fallback = build_fallback_integrated_guidance_card(
                intervention_point=intervention_point,
                failure_reason="real LLM backend is not configured",
                reward_diagnosis=reward_diagnosis,
            )
            call_record["fallback_used"] = self.fallback_to_reward_only
            call_record["failure_reason"] = fallback["metadata"]["failure_reason"]
            call_record["integrated_guidance_card"] = fallback
            call_record["validation_errors"] = ["missing_backend"]
            call_record["ledger_event"] = self._log_event(
                intervention_point=intervention_point,
                call_type="fallback",
                reason=f"missing backend for {intervention_point}",
                event_state="fallback",
                payload=fallback_prompt_payload,
                success=False,
                fallback_used=True,
                fallback_reason=fallback["metadata"]["failure_reason"],
                error_type="missing_backend",
                round_id=self._round_from_branch_results(branch_results),
            )
            self._persist_cached_card(
                intervention_point=intervention_point,
                prompt_payload=fallback_prompt_payload,
                call_record=call_record,
                integrated_guidance_card=fallback,
            )
            return {"integrated_guidance_card": fallback, "call_record": call_record}

        prompt_payload = {
            "intervention_point": str(intervention_point),
            "model": self.model,
            "reward_diagnosis": deepcopy(reward_diagnosis or {}),
            "behavior_summary": deepcopy(behavior_summary or {}),
            "branch_results": deepcopy(branch_results or []),
        }
        prompt_text = self.render_prompt(
            intervention_point=intervention_point,
            prompt_payload=prompt_payload,
        )
        call_record["prompt_text"] = prompt_text
        call_record["prompt_payload"] = prompt_payload
        validation_errors: list[str] = []
        repair_attempts: list[Dict[str, Any]] = []
        raw_text = ""
        parsed: Dict[str, Any] = {}
        backend_metadata: Dict[str, Any] = {}
        round_id = self._round_from_branch_results(branch_results)
        attempt_event = self._log_event(
            intervention_point=intervention_point,
            call_type="guidance",
            reason=f"policy guidance request for {intervention_point}",
            event_state="attempt",
            payload=prompt_payload,
            success=None,
            round_id=round_id,
        )

        try:
            result = self.backend.generate_text(
                system_prompt=(
                    "You are the policy-side Critic for a MARL reward-search workflow. "
                    "Return a strict JSON object only."
                ),
                user_prompt=prompt_text,
                metadata={
                    "role": "policy_guidance_integrated_critic",
                    "intervention_point": intervention_point,
                },
            )
            raw_text = str(result.get("text") or "")
            backend_metadata = dict(result.get("backend_metadata") or {})
            parsed = self._extract_json_dict(raw_text)
            validated = validate_integrated_guidance_card(parsed)
            call_record["raw_text"] = raw_text
            call_record["parsed_response"] = deepcopy(parsed)
            call_record["backend_metadata"] = backend_metadata
            call_record["validation_errors"] = []
            call_record["repair_attempts"] = []
            call_record["integrated_guidance_card"] = validated
            call_record["ledger_event"] = self._log_event(
                intervention_point=intervention_point,
                call_type="guidance",
                reason=f"policy guidance request for {intervention_point}",
                event_state="success",
                payload=prompt_payload,
                usage=result.get("usage"),
                request_id=extract_request_id(result),
                success=True,
                fallback_used=False,
                metadata={"backend_metadata": backend_metadata},
                record_id=str(attempt_event.get("record_id") or ""),
                round_id=round_id,
            )
            self._persist_cached_card(
                intervention_point=intervention_point,
                prompt_payload=prompt_payload,
                call_record=call_record,
                integrated_guidance_card=validated,
            )
            return {"integrated_guidance_card": validated, "call_record": call_record}
        except Exception as exc:
            validation_errors.append(str(exc))

        for repair_index in range(1, self.repair_retry_count + 1):
            repair_prompt = self.render_repair_prompt(
                intervention_point=intervention_point,
                prompt_payload=prompt_payload,
                previous_raw_text=raw_text,
                validation_errors=validation_errors,
            )
            try:
                repair_result = self.backend.generate_text(
                    system_prompt=(
                        "You are repairing a JSON schema violation for a policy-guidance Critic. "
                        "Return a strict JSON object only."
                    ),
                    user_prompt=repair_prompt,
                    metadata={
                        "role": "policy_guidance_integrated_critic_repair",
                        "intervention_point": intervention_point,
                        "repair_index": repair_index,
                    },
                )
                repair_raw_text = str(repair_result.get("text") or "")
                repair_parsed = self._extract_json_dict(repair_raw_text)
                validated = validate_integrated_guidance_card(repair_parsed)
                repair_attempts.append(
                    {
                        "repair_index": repair_index,
                        "status": "success",
                        "raw_text": repair_raw_text,
                        "validation_errors": [],
                    }
                )
                call_record["raw_text"] = raw_text
                call_record["parsed_response"] = deepcopy(repair_parsed)
                call_record["backend_metadata"] = dict(repair_result.get("backend_metadata") or {})
                call_record["validation_errors"] = list(validation_errors)
                call_record["repair_attempts"] = repair_attempts
                call_record["repair_retry_used"] = True
                call_record["integrated_guidance_card"] = validated
                call_record["ledger_event"] = self._log_event(
                    intervention_point=intervention_point,
                    call_type="repair",
                    reason=f"policy guidance repair for {intervention_point}",
                    event_state="repair",
                    payload=prompt_payload,
                    usage=repair_result.get("usage"),
                    request_id=extract_request_id(repair_result),
                    success=True,
                    repair_used=True,
                    repair_retry_count=repair_index,
                    fallback_used=False,
                    metadata={"backend_metadata": dict(repair_result.get("backend_metadata") or {})},
                    record_id=str(attempt_event.get("record_id") or ""),
                    round_id=round_id,
                )
                self._persist_cached_card(
                    intervention_point=intervention_point,
                    prompt_payload=prompt_payload,
                    call_record=call_record,
                    integrated_guidance_card=validated,
                )
                return {"integrated_guidance_card": validated, "call_record": call_record}
            except Exception as repair_exc:
                validation_errors.append(str(repair_exc))
                repair_attempts.append(
                    {
                        "repair_index": repair_index,
                        "status": "failed",
                        "validation_errors": [str(repair_exc)],
                    }
                )

        fallback = build_fallback_integrated_guidance_card(
            intervention_point=intervention_point,
            failure_reason=validation_errors[-1] if validation_errors else "real llm validation failed",
            reward_diagnosis=reward_diagnosis,
        )
        call_record["fallback_used"] = self.fallback_to_reward_only
        call_record["failure_reason"] = fallback["metadata"]["failure_reason"]
        call_record["raw_text"] = raw_text
        call_record["parsed_response"] = deepcopy(parsed)
        call_record["backend_metadata"] = backend_metadata
        call_record["validation_errors"] = list(validation_errors)
        call_record["repair_attempts"] = repair_attempts
        call_record["repair_retry_used"] = bool(repair_attempts)
        call_record["integrated_guidance_card"] = fallback
        call_record["ledger_event"] = self._log_event(
            intervention_point=intervention_point,
            call_type="fallback",
            reason=f"policy guidance fallback for {intervention_point}",
            event_state="fallback",
            payload=prompt_payload,
            usage=None,
            request_id=None,
            success=False,
            repair_used=bool(repair_attempts),
            repair_retry_count=len(repair_attempts),
            fallback_used=True,
            fallback_reason=fallback["metadata"]["failure_reason"],
            metadata={"backend_metadata": backend_metadata},
            record_id=str(attempt_event.get("record_id") or ""),
            round_id=round_id,
            error_type="policy_guidance_validation_failed",
        )
        self._persist_cached_card(
            intervention_point=intervention_point,
            prompt_payload=prompt_payload,
            call_record=call_record,
            integrated_guidance_card=fallback,
        )
        return {"integrated_guidance_card": fallback, "call_record": call_record}

    def render_prompt(
        self,
        *,
        intervention_point: str,
        prompt_payload: Dict[str, Any],
    ) -> str:
        prompt_spec = (
            PROMPT_PATHS.get(intervention_point)
            or PROMPT_PATHS["integrated_guidance_card"]
        ).read_text(encoding="utf-8")
        schema_spec = PROMPT_PATHS["integrated_guidance_card"].read_text(encoding="utf-8")
        return "\n".join(
            [
                "Prompt specification:",
                prompt_spec,
                "",
                "Integrated card schema specification:",
                schema_spec,
                "",
                "Input payload:",
                json.dumps(prompt_payload, ensure_ascii=False, indent=2, sort_keys=True),
                "",
                "Return strict JSON only.",
            ]
        )

    def render_repair_prompt(
        self,
        *,
        intervention_point: str,
        prompt_payload: Dict[str, Any],
        previous_raw_text: str,
        validation_errors: list[str],
    ) -> str:
        return "\n".join(
            [
                self.render_prompt(
                    intervention_point=intervention_point,
                    prompt_payload=prompt_payload,
                ),
                "",
                "Previous invalid response:",
                previous_raw_text,
                "",
                "Validation errors:",
                json.dumps(validation_errors, ensure_ascii=False, indent=2),
                "",
                "Repair instruction:",
                "Return one strict JSON object that satisfies the required Integrated Guidance Card schema.",
            ]
        )

    @staticmethod
    def _extract_json_dict(text: str) -> Dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("response does not contain a JSON object")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("expected top-level JSON object")
        return payload

    @staticmethod
    def _payload_hash(payload: Dict[str, Any]) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _cache_dir_for_payload(
        self,
        *,
        intervention_point: str,
        prompt_payload: Dict[str, Any],
    ) -> Path | None:
        if self.cache_root is None:
            return None
        return self.cache_root / str(intervention_point) / self._payload_hash(prompt_payload)

    def _persist_cached_card(
        self,
        *,
        intervention_point: str,
        prompt_payload: Dict[str, Any],
        call_record: Dict[str, Any],
        integrated_guidance_card: Dict[str, Any],
    ) -> None:
        cache_dir = self._cache_dir_for_payload(
            intervention_point=intervention_point,
            prompt_payload=prompt_payload,
        )
        if cache_dir is None:
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "final_integrated_guidance_card.json").write_text(
            json.dumps(integrated_guidance_card, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (cache_dir / "prompt_rendered.txt").write_text(
            str(call_record.get("prompt_text") or ""),
            encoding="utf-8",
        )
        (cache_dir / "raw_response.txt").write_text(
            str(call_record.get("raw_text") or ""),
            encoding="utf-8",
        )
        (cache_dir / "parsed_response.json").write_text(
            json.dumps(call_record.get("parsed_response") or {}, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (cache_dir / "validation_errors.json").write_text(
            json.dumps({"errors": list(call_record.get("validation_errors") or [])}, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (cache_dir / "repair_attempts.json").write_text(
            json.dumps({"attempts": list(call_record.get("repair_attempts") or [])}, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (cache_dir / "cache_metadata.json").write_text(
            json.dumps(
                {
                    "backend_metadata": deepcopy(call_record.get("backend_metadata") or {}),
                    "failure_reason": call_record.get("failure_reason") or "",
                    "fallback_used": bool(call_record.get("fallback_used", False)),
                    "intervention_point": str(intervention_point),
                    "model": self.model,
                    "payload_hash": self._payload_hash(prompt_payload),
                    "success": not bool(call_record.get("fallback_used", False)),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _load_cached_card(
        self,
        *,
        intervention_point: str,
        reward_diagnosis: Dict[str, Any],
        behavior_summary: Dict[str, Any],
        branch_results: list[Dict[str, Any]],
    ) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
        if not self.reuse_cache or self.cache_root is None:
            return None
        intervention_root = self.cache_root / str(intervention_point)
        if not intervention_root.exists():
            return None
        prompt_payload = {
            "intervention_point": str(intervention_point),
            "reward_diagnosis": deepcopy(reward_diagnosis or {}),
            "behavior_summary": deepcopy(behavior_summary or {}),
            "branch_results": deepcopy(branch_results or []),
        }
        candidates = []
        exact_dir = intervention_root / self._payload_hash(prompt_payload)
        if exact_dir.exists():
            candidates.append(exact_dir)
        if "latest" in self.cache_lookup_mode:
            latest_dirs = sorted(
                [path.parent for path in intervention_root.rglob("final_integrated_guidance_card.json")],
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            candidates.extend(latest_dirs)
        seen: set[str] = set()
        for cache_dir in candidates:
            cache_dir = cache_dir.resolve()
            key = str(cache_dir)
            if key in seen:
                continue
            seen.add(key)
            card_path = cache_dir / "final_integrated_guidance_card.json"
            if not card_path.exists():
                continue
            try:
                card = validate_integrated_guidance_card(
                    json.loads(card_path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
            prompt_path = cache_dir / "prompt_rendered.txt"
            raw_path = cache_dir / "raw_response.txt"
            parsed_path = cache_dir / "parsed_response.json"
            validation_path = cache_dir / "validation_errors.json"
            repair_path = cache_dir / "repair_attempts.json"
            meta_path = cache_dir / "cache_metadata.json"
            parsed_response: Dict[str, Any] = {}
            validation_errors: list[str] = []
            repair_attempts: list[Dict[str, Any]] = []
            backend_metadata: Dict[str, Any] = {}
            if parsed_path.exists():
                try:
                    parsed_response = json.loads(parsed_path.read_text(encoding="utf-8"))
                except Exception:
                    parsed_response = {}
            if validation_path.exists():
                try:
                    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
                    validation_errors = list(validation_payload.get("errors") or [])
                except Exception:
                    validation_errors = []
            if repair_path.exists():
                try:
                    repair_payload = json.loads(repair_path.read_text(encoding="utf-8"))
                    repair_attempts = list(repair_payload.get("attempts") or [])
                except Exception:
                    repair_attempts = []
            if meta_path.exists():
                try:
                    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
                    backend_metadata = dict(meta_payload.get("backend_metadata") or {})
                except Exception:
                    backend_metadata = {}
            return (
                card,
                {
                    "cache_dir": str(cache_dir),
                    "prompt_text": prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "",
                    "raw_text": raw_path.read_text(encoding="utf-8") if raw_path.exists() else "",
                    "parsed_response": parsed_response,
                    "validation_errors": validation_errors,
                    "repair_attempts": repair_attempts,
                    "repair_retry_used": bool(repair_attempts),
                    "backend_metadata": {
                        **backend_metadata,
                        "model": self.model,
                        "cache_reused": True,
                        "cache_dir": str(cache_dir),
                    },
                    "llm_call_cached": True,
                },
            )
        return None

    def generate_stage1b_integrated_guidance(
        self,
        *,
        sparse_baseline_metrics: Dict[str, Any],
        stage1_behavior_artifact: Dict[str, Any],
        selected_behavior_summary: Dict[str, Any],
        selected_milestone: Dict[str, Any],
    ) -> Dict[str, Any]:
        reward_diagnosis = {
            "learning_stage": "stage1_sparse_baseline_to_stage1b",
            "metric_evidence": [
                f"sparse_baseline_metrics.best_test_sparse_return_mean={sparse_baseline_metrics.get('best_test_sparse_return_mean')}",
                f"sparse_baseline_metrics.final_test_sparse_return_mean={sparse_baseline_metrics.get('final_test_sparse_return_mean')}",
                "stage1_sparse_policy_behavior_summary_path="
                + str(stage1_behavior_artifact.get("_summary_path") or ""),
                "stage1_sparse_policy_behavior_summary.selected_target_step="
                + str(selected_milestone.get("target_step")),
            ],
            "performance_issue": str(
                sparse_baseline_metrics.get("performance_issue")
                or "sparse baseline needs an early dense initialization search"
            ),
        }
        result = self.generate_integrated_guidance_card(
            intervention_point="stage1_after_sparse_baseline",
            reward_diagnosis=reward_diagnosis,
            behavior_summary=selected_behavior_summary,
        )
        card = deepcopy(result["integrated_guidance_card"])
        guidance = deepcopy(card.get("candidate_generation_guidance") or {})
        stage1b_implications = deepcopy(stage1_behavior_artifact.get("stage1b_implications") or {})
        if stage1b_implications:
            guidance["reward_search_implications"] = list(
                guidance.get("reward_search_implications") or []
            ) + [
                "stage1b_implications.early_primary_failure="
                + str(stage1b_implications.get("early_primary_failure")),
                "stage1b_implications.recommended_candidate_types="
                + ",".join(list(stage1b_implications.get("recommended_candidate_types") or [])),
            ]
            if stage1b_implications.get("beta_search_prior") in {"up", "down", "same", "unknown"}:
                guidance["beta_direction"] = str(stage1b_implications.get("beta_search_prior"))
            if stage1b_implications.get("wc_search_prior") in {"up", "down", "same", "unknown"}:
                guidance["wc_direction"] = str(stage1b_implications.get("wc_search_prior"))
            if stage1b_implications.get("wp_search_prior") in {"up", "down", "same", "unknown"}:
                guidance["wp_direction"] = str(stage1b_implications.get("wp_search_prior"))
            guidance["candidate_types"] = list(dict.fromkeys(
                list(guidance.get("candidate_types") or [])
                + list(stage1b_implications.get("recommended_candidate_types") or [])
            ))
        card["candidate_generation_guidance"] = guidance
        metadata = deepcopy(card.get("metadata") or {})
        metadata["stage1_behavior_summary_path"] = str(stage1_behavior_artifact.get("_summary_path") or "")
        metadata["selected_milestone_target_step"] = int(selected_milestone.get("target_step") or 0)
        metadata["selected_milestone_actual_step"] = int(selected_milestone.get("actual_step") or 0)
        metadata["stage1b_target_checkpoint_step"] = 800000
        card["metadata"] = metadata
        validated_card = validate_integrated_guidance_card(card)
        result["integrated_guidance_card"] = validated_card
        result["call_record"]["integrated_guidance_card"] = validated_card
        return result

    def generate_stage3_pre_checkpoint_integrated_guidance(
        self,
        *,
        intervention_point: str,
        checkpoint_id: str,
        checkpoint_step: int,
        current_pbrs_config: Dict[str, Any],
        reward_metrics: Dict[str, Any],
        behavior_summary: Dict[str, Any],
        sparse_dense_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        reward_diagnosis = {
            "learning_stage": f"stage3_{str(checkpoint_id).lower()}_pre_checkpoint",
            "metric_evidence": [
                f"checkpoint_id={checkpoint_id}",
                f"checkpoint_step={checkpoint_step}",
                f"reward_metrics.current_test_sparse_return_mean={reward_metrics.get('current_test_sparse_return_mean')}",
                f"current_pbrs_config.beta={current_pbrs_config.get('beta')}",
                f"current_pbrs_config.wc={current_pbrs_config.get('wc')}",
            ],
            "performance_issue": str(
                reward_metrics.get("performance_issue")
                or "checkpoint pre-diagnosis before stage3 round-1 candidate generation"
            ),
        }
        result = self.generate_integrated_guidance_card(
            intervention_point=intervention_point,
            reward_diagnosis=reward_diagnosis,
            behavior_summary=behavior_summary,
        )
        card = deepcopy(result["integrated_guidance_card"])
        card["checkpoint_id"] = str(checkpoint_id)
        card["checkpoint_step"] = int(checkpoint_step)
        metadata = deepcopy(card.get("metadata") or {})
        metadata["current_pbrs_config"] = deepcopy(current_pbrs_config or {})
        metadata["sparse_dense_context"] = deepcopy(sparse_dense_context or {})
        card["metadata"] = metadata
        validated_card = validate_integrated_guidance_card(card)
        result["integrated_guidance_card"] = validated_card
        result["call_record"]["integrated_guidance_card"] = validated_card
        return result

    def generate_stage3_after_round1_integrated_guidance(
        self,
        *,
        intervention_point: str,
        checkpoint_id: str,
        checkpoint_step: int,
        round_id: int,
        reward_metrics: Dict[str, Any],
        behavior_summary: Dict[str, Any],
        branch_results: list[Dict[str, Any]],
        branch_behavior_summaries: list[Dict[str, Any]],
        previous_integrated_guidance_card: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        reward_diagnosis = {
            "learning_stage": f"stage3_{str(checkpoint_id).lower()}_after_round1",
            "metric_evidence": [
                f"checkpoint_id={checkpoint_id}",
                f"checkpoint_step={checkpoint_step}",
                f"round_id={round_id}",
                f"branch_result_count={len(branch_results)}",
                f"reward_metrics.current_test_sparse_return_mean={reward_metrics.get('current_test_sparse_return_mean')}",
            ],
            "performance_issue": str(
                reward_metrics.get("performance_issue")
                or "round-1 result analysis before stage3 round-2 candidate generation"
            ),
        }
        result = self.generate_integrated_guidance_card(
            intervention_point=intervention_point,
            reward_diagnosis=reward_diagnosis,
            behavior_summary=behavior_summary,
            branch_results=branch_results,
        )
        card = deepcopy(result["integrated_guidance_card"])
        card["checkpoint_id"] = str(checkpoint_id)
        card["checkpoint_step"] = int(checkpoint_step)
        card["round_id"] = int(round_id)
        card["branch_evidence"] = self._build_branch_evidence(
            branch_results=branch_results,
            branch_behavior_summaries=branch_behavior_summaries,
        )
        card["round2_search_guidance"] = self._build_round2_search_guidance(
            branch_results=branch_results,
            previous_integrated_guidance_card=previous_integrated_guidance_card,
        )
        metadata = deepcopy(card.get("metadata") or {})
        metadata["previous_integrated_guidance_card_present"] = bool(
            previous_integrated_guidance_card
        )
        card["metadata"] = metadata
        validated_card = validate_integrated_guidance_card(card)
        result["integrated_guidance_card"] = validated_card
        result["call_record"]["integrated_guidance_card"] = validated_card
        return result

    def _build_mock_integrated_guidance_card(
        self,
        *,
        intervention_point: str,
        reward_diagnosis: Dict[str, Any],
        behavior_summary: Dict[str, Any],
        branch_results: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        coverage_block = behavior_summary.get("coverage") or {}
        failed_collect_block = behavior_summary.get("failed_collect") or {}
        target_block = behavior_summary.get("target_concentration") or {}
        food_block = behavior_summary.get("food_discovery") or {}
        coverage = coverage_block.get("score")
        failed_collect = failed_collect_block.get("score")
        target_concentration = target_block.get("score")
        evidence_flags = deepcopy(behavior_summary.get("evidence_flags") or {})
        candidate_types = ["reference_like", "recovery"]
        beta_direction = "same"
        wc_direction = "same"
        wp_direction = "same"
        implications = []
        if evidence_flags.get("low_coverage") or (coverage is not None and coverage < 0.4):
            candidate_types.append("exploration_boost")
            beta_direction = "up"
            implications.append("Sparse exploration suggests more aggressive early shaping.")
        if evidence_flags.get("over_concentration_possible") or (
            target_concentration is not None and target_concentration > 0.65
        ):
            candidate_types.append("wc_down_wp_up")
            wc_direction = "down"
            wp_direction = "up"
            implications.append("Congestion suggests reducing coordination overweight.")
        if evidence_flags.get("near_food_collection_failure") or (
            failed_collect is not None and failed_collect > 0.5
        ):
            candidate_types.append("collection_recovery")
            implications.append(
                "Frequent failed collect patterns suggest recovery-style candidates."
            )
        if evidence_flags.get("food_discovery_failure") and "exploration_boost" not in candidate_types:
            candidate_types.append("exploration_boost")
        behavior_evidence = self._build_behavior_evidence(
            behavior_summary=behavior_summary,
            branch_results=branch_results,
        )
        for item in branch_results[:2]:
            branch_id = str(item.get("branch_id") or item.get("candidate_id") or "branch")
            metric = item.get("best_test_sparse_return_mean")
            behavior_evidence.append(
                f"Branch result {branch_id} observed best_test_sparse_return_mean={metric}."
            )
        return {
            "intervention_point": str(intervention_point),
            "reward_diagnosis": deepcopy(reward_diagnosis),
            "policy_diagnosis": {
                "policy_failure_mode": "coordination_congestion"
                if evidence_flags.get("over_concentration_possible")
                or (target_concentration or 0.0) > 0.65
                else "under_exploration",
                "behavior_evidence": behavior_evidence,
                "confidence": "medium",
                "insufficient_evidence": False,
                "failure_reason": "",
            },
            "candidate_generation_guidance": {
                "candidate_types": candidate_types,
                "beta_direction": beta_direction,
                "wc_direction": wc_direction,
                "wp_direction": wp_direction,
                "constraints": [
                    "context_only",
                    "do_not_override_actions",
                    "do_not_modify_learner",
                ],
                "reward_search_implications": implications
                or [
                    "Behavior evidence is mild; keep reward search close to the stable baseline."
                ],
            },
            "use_in_reward_generation": True,
            "fallback_to_reward_only": False,
            "metadata": {
                "mock_llm_mode": True,
                "branch_result_count": len(branch_results),
                "behavior_summary_stage": behavior_summary.get("milestone", {}).get("stage_label"),
            },
        }

    def _build_behavior_evidence(
        self,
        *,
        behavior_summary: Dict[str, Any],
        branch_results: list[Dict[str, Any]],
    ) -> list[str]:
        evidence = []
        coverage_metrics = (behavior_summary.get("coverage") or {}).get("metrics") or {}
        food_metrics = (behavior_summary.get("food_discovery") or {}).get("metrics") or {}
        failed_metrics = (behavior_summary.get("failed_collect") or {}).get("metrics") or {}
        target_metrics = (behavior_summary.get("target_concentration") or {}).get("metrics") or {}
        stability_metrics = (behavior_summary.get("stability") or {}).get("metrics") or {}
        flags = behavior_summary.get("evidence_flags") or {}
        evidence.append(
            "coverage.metrics.joint_visited_cell_ratio="
            + str(coverage_metrics.get("joint_visited_cell_ratio"))
        )
        evidence.append(
            "food_discovery.metrics.episodes_never_close_to_food="
            + str(food_metrics.get("episodes_never_close_to_food"))
        )
        evidence.append(
            "failed_collect.metrics.successful_collection_events="
            + str(failed_metrics.get("successful_collection_events"))
        )
        evidence.append(
            "target_concentration.metrics.same_target_ratio="
            + str(target_metrics.get("same_target_ratio"))
        )
        evidence.append(
            "stability.metrics.repeated_position_ratio="
            + str(stability_metrics.get("repeated_position_ratio"))
        )
        evidence.append(
            "milestone.target_step="
            + str((behavior_summary.get("milestone") or {}).get("target_step"))
        )
        for key, value in flags.items():
            if value:
                evidence.append(f"evidence_flags.{key}=true")
        evidence.extend(list(behavior_summary.get("behavior_signals") or [])[:3])
        return evidence

    def _build_branch_evidence(
        self,
        *,
        branch_results: list[Dict[str, Any]],
        branch_behavior_summaries: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        by_candidate: Dict[str, Dict[str, Any]] = {}
        for item in branch_results:
            candidate_id = str(item.get("candidate_id") or item.get("branch_id") or "")
            by_candidate[candidate_id] = {
                "candidate_id": candidate_id,
                "metric_evidence": [
                    f"best_test_sparse_return_mean={item.get('best_test_sparse_return_mean')}",
                    f"last_test_sparse_return_mean={item.get('last_test_sparse_return_mean')}",
                ],
                "behavior_evidence": [],
            }
        for summary in branch_behavior_summaries:
            metadata = summary.get("metadata") or {}
            candidate_id = str(
                metadata.get("branch_id")
                or summary.get("candidate_id")
                or summary.get("branch_id")
                or ""
            )
            entry = by_candidate.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "metric_evidence": [],
                    "behavior_evidence": [],
                },
            )
            entry["behavior_evidence"].extend(
                self._build_behavior_evidence(
                    behavior_summary=summary,
                    branch_results=[],
                )[:4]
            )
        return list(by_candidate.values())

    def _build_round2_search_guidance(
        self,
        *,
        branch_results: list[Dict[str, Any]],
        previous_integrated_guidance_card: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        sorted_results = sorted(
            branch_results,
            key=lambda item: float(item.get("best_test_sparse_return_mean") or -1e9),
            reverse=True,
        )
        best = sorted_results[0] if sorted_results else {}
        avoid: list[str] = []
        if best and float(best.get("last_test_sparse_return_mean") or -1e9) < float(
            best.get("best_test_sparse_return_mean") or -1e9
        ):
            avoid.append("unstable_spike")
        previous_types = list(
            (((previous_integrated_guidance_card or {}).get("candidate_generation_guidance") or {}).get("candidate_types"))
            or []
        )
        return {
            "around_candidate": str(best.get("candidate_id") or ""),
            "avoid_candidate_types": list(dict.fromkeys(avoid + previous_types[:1])),
            "stability_warning": (
                "Prefer candidates with stable last-window behavior, not only peak spikes."
            ),
            "prefer_stable_last_window": True,
        }
