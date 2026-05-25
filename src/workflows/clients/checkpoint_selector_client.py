from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from workflows.checkpoint_selector_schema import validate_checkpoint_selection


DEFAULT_PROMPT_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "checkpoint_selector_prompt.md"
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
class CheckpointSelectionResult:
    prompt: str
    response: Dict[str, Any]
    selection: Dict[str, Any]


class TemplateBasedCheckpointSelectorClient:
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

    def select_checkpoints(
        self,
        *,
        workflow_spec: Dict[str, Any],
        baseline_summary: Dict[str, Any],
        selection_constraints: Dict[str, Any],
        checkpoint_candidates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self.build_prompt(
            workflow_spec=workflow_spec,
            baseline_summary=baseline_summary,
            selection_constraints=selection_constraints,
            checkpoint_candidates=checkpoint_candidates,
        )
        llm_result = self.llm_backend.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            metadata={
                "workflow_id": workflow_spec.get("workflow_id"),
                "role": "checkpoint_selector",
            },
        )
        parsed = self.parse_response(
            llm_result["text"],
            selection_constraints=selection_constraints,
            checkpoint_candidates=checkpoint_candidates,
        )
        response = {
            "raw_text": llm_result["text"],
            "raw_response": llm_result.get("raw_response"),
            "model": llm_result.get("model"),
            "usage": llm_result.get("usage"),
            "parsed": parsed,
        }
        return CheckpointSelectionResult(
            prompt=prompt,
            response=response,
            selection=parsed,
        ).__dict__

    def build_prompt(
        self,
        *,
        workflow_spec: Dict[str, Any],
        baseline_summary: Dict[str, Any],
        selection_constraints: Dict[str, Any],
        checkpoint_candidates: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt_spec = self.prompt_spec_path.read_text(encoding="utf-8")
        env_reference = self.env_reference_path.read_text(encoding="utf-8")
        sections = [
            "Prompt specification:",
            prompt_spec,
            "",
            "Task description:",
            workflow_spec.get("task_description", "No task description provided."),
            "",
            "Environment description:",
            workflow_spec.get("environment_description", "No environment description provided."),
            "",
            "LBF environment reference:",
            env_reference,
            "",
            "Checkpoint selection constraints:",
            json.dumps(selection_constraints, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Sparse baseline trajectory summary:",
            json.dumps(baseline_summary, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Checkpoint candidates:",
            json.dumps(
                checkpoint_candidates or {"checkpoint_candidates": []},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Choose the checkpoints now according to the prompt specification above.",
        ]
        return "\n".join(sections)

    def parse_response(
        self,
        text: str,
        *,
        selection_constraints: Dict[str, Any],
        checkpoint_candidates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            normalized_text = _extract_json_object_text(text)
            if normalized_text is None:
                raise ValueError(f"Checkpoint selector output is not valid JSON: {exc}") from exc
            try:
                parsed = json.loads(normalized_text)
            except json.JSONDecodeError as normalized_exc:
                raise ValueError(
                    f"Checkpoint selector output is not valid JSON: {normalized_exc}"
                ) from normalized_exc

        min_count = int(selection_constraints.get("min_count", 3))
        max_count = int(selection_constraints.get("max_count", 4))
        min_step_gap = int(selection_constraints.get("min_step_gap", 1))
        allowed_steps = None
        if checkpoint_candidates is not None:
            allowed_steps = []
            for item in checkpoint_candidates.get("checkpoint_candidates") or []:
                try:
                    allowed_steps.append(int(item.get("checkpoint_step")))
                except (TypeError, ValueError, AttributeError):
                    continue
            allowed_steps = sorted(set(allowed_steps))
        return validate_checkpoint_selection(
            parsed,
            min_count=min_count,
            max_count=max_count,
            min_step_gap=min_step_gap,
            allowed_steps=allowed_steps,
        )

    def _default_system_prompt(self) -> str:
        return (
            "You select informative training checkpoints from sparse baseline trajectories "
            "for later PBRS branching experiments. Return strictly valid JSON."
        )


def _extract_json_object_text(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1].strip()
    return None


class StubCheckpointSelectorBackend:
    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "analysis_summary": "Select early, middle, and late checkpoints for PBRS branching.",
            "selected_checkpoints": [
                {
                    "name": "early_branch",
                    "step": 100000,
                    "stage_label": "early_exploration",
                    "reason": "Early phase after initial departure from flat behaviour.",
                    "recommended_focus": "Test whether beta should be stronger for exploration.",
                },
                {
                    "name": "mid_branch",
                    "step": 600000,
                    "stage_label": "mid_progress",
                    "reason": "Middle phase where learning may transition or slow down.",
                    "recommended_focus": "Test balance between wc and wp during coordination.",
                },
                {
                    "name": "late_branch",
                    "step": 1400000,
                    "stage_label": "late_plateau",
                    "reason": "Late phase where the trajectory may plateau or stabilize.",
                    "recommended_focus": "Test whether shaping should weaken in late training.",
                },
            ],
        }
        return {
            "text": json.dumps(payload, ensure_ascii=False),
            "raw_response": {"stub": True, "metadata": metadata},
            "model": "stub-checkpoint-selector",
            "usage": None,
        }
