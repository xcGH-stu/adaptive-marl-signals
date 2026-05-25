from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_WORKFLOW_ROOT = (
    Path(__file__).resolve().parents[2] / "results" / "llm_reward_workflows"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RoundPaths:
    round_id: int
    round_dir: Path
    generator_prompt: Path
    generator_response: Path
    candidate_proposals: Path
    reward_spec: Path
    policy_guidance: Path
    reward_function: Path
    alpha_policy: Path
    train_config: Path
    train_run_ref: Path
    train_metrics_summary: Path
    critic_prompt: Path
    critic_response: Path
    candidate_manifest: Path
    candidate_execution_plan: Path
    status: Path


@dataclass(frozen=True)
class CandidatePaths:
    round_id: int
    candidate_id: str
    candidate_dir: Path
    reward_spec: Path
    reward_function: Path
    train_config: Path
    train_run_ref: Path
    train_metrics_summary: Path


class WorkflowStorage:
    def __init__(
        self,
        workflow_id: str,
        root_dir: Optional[Path | str] = None,
    ):
        self.workflow_id = workflow_id
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_WORKFLOW_ROOT
        self.workflow_dir = self.root_dir / workflow_id
        self.manifest_path = self.workflow_dir / "manifest.json"
        self.timeline_path = self.workflow_dir / "timeline.jsonl"

    @classmethod
    def create(
        cls,
        workflow_id: str,
        *,
        root_dir: Optional[Path | str] = None,
        manifest_data: Optional[Dict[str, Any]] = None,
    ) -> "WorkflowStorage":
        storage = cls(workflow_id=workflow_id, root_dir=root_dir)
        storage.workflow_dir.mkdir(parents=True, exist_ok=True)

        if not storage.manifest_path.exists():
            manifest = {
                "workflow_id": workflow_id,
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
                "status": "initialized",
                "current_round": None,
                "round_count": 0,
            }
            if manifest_data:
                manifest.update(manifest_data)
            storage.save_manifest(manifest)

        if not storage.timeline_path.exists():
            storage.timeline_path.touch()

        storage.append_timeline_event(
            "workflow_initialized",
            {"workflow_id": workflow_id},
        )
        return storage

    @classmethod
    def load(
        cls,
        workflow_id: str,
        *,
        root_dir: Optional[Path | str] = None,
    ) -> "WorkflowStorage":
        storage = cls(workflow_id=workflow_id, root_dir=root_dir)
        if not storage.workflow_dir.exists():
            raise FileNotFoundError(f"workflow directory does not exist: {storage.workflow_dir}")
        if not storage.manifest_path.exists():
            raise FileNotFoundError(f"workflow manifest does not exist: {storage.manifest_path}")
        return storage

    def get_round_dir(self, round_id: int) -> Path:
        return self.workflow_dir / f"round_{int(round_id):02d}"

    def get_round_paths(self, round_id: int) -> RoundPaths:
        round_dir = self.get_round_dir(round_id)
        return RoundPaths(
            round_id=int(round_id),
            round_dir=round_dir,
            generator_prompt=round_dir / "generator_prompt.txt",
            generator_response=round_dir / "generator_response.json",
            candidate_proposals=round_dir / "candidate_proposals.json",
            reward_spec=round_dir / "reward_spec.json",
            policy_guidance=round_dir / "policy_guidance.json",
            reward_function=round_dir / "reward_function.py",
            alpha_policy=round_dir / "alpha_policy.json",
            train_config=round_dir / "train_config.json",
            train_run_ref=round_dir / "train_run_ref.json",
            train_metrics_summary=round_dir / "train_metrics_summary.json",
            critic_prompt=round_dir / "critic_prompt.txt",
            critic_response=round_dir / "critic_response.json",
            candidate_manifest=round_dir / "candidate_manifest.json",
            candidate_execution_plan=round_dir / "candidate_execution_plan.json",
            status=round_dir / "status.json",
        )

    def get_candidate_dir(self, round_id: int, candidate_id: str) -> Path:
        return self.get_round_dir(round_id) / "candidates" / str(candidate_id)

    def get_candidate_paths(self, round_id: int, candidate_id: str) -> CandidatePaths:
        candidate_dir = self.get_candidate_dir(round_id, candidate_id)
        return CandidatePaths(
            round_id=int(round_id),
            candidate_id=str(candidate_id),
            candidate_dir=candidate_dir,
            reward_spec=candidate_dir / "reward_spec.json",
            reward_function=candidate_dir / "reward_function.py",
            train_config=candidate_dir / "train_config.json",
            train_run_ref=candidate_dir / "train_run_ref.json",
            train_metrics_summary=candidate_dir / "train_metrics_summary.json",
        )

    def ensure_round(self, round_id: int) -> RoundPaths:
        paths = self.get_round_paths(round_id)
        paths.round_dir.mkdir(parents=True, exist_ok=True)

        if not paths.status.exists():
            self.save_round_status(round_id, "initialized")

        manifest = self.load_manifest()
        manifest["current_round"] = int(round_id)
        manifest["round_count"] = max(int(manifest.get("round_count", 0)), int(round_id))
        manifest["updated_at"] = _utc_now_iso()
        self.save_manifest(manifest)
        return paths

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        manifest = dict(manifest)
        manifest["updated_at"] = _utc_now_iso()
        _ensure_parent(self.manifest_path)
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    def load_manifest(self) -> Dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_workflow_json(self, filename: str, data: Dict[str, Any]) -> Path:
        path = self.workflow_dir / filename
        _ensure_parent(path)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        return path

    def load_workflow_json(self, filename: str) -> Dict[str, Any]:
        path = self.workflow_dir / filename
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def update_manifest(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        manifest = self.load_manifest()
        manifest.update(patch)
        self.save_manifest(manifest)
        return manifest

    def append_timeline_event(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        _ensure_parent(self.timeline_path)
        event = {
            "timestamp": _utc_now_iso(),
            "event_type": event_type,
            "payload": payload or {},
        }
        with self.timeline_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def save_round_status(
        self,
        round_id: int,
        status: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        paths = self.get_round_paths(round_id)
        paths.round_dir.mkdir(parents=True, exist_ok=True)
        status_data = {
            "round_id": int(round_id),
            "status": status,
            "updated_at": _utc_now_iso(),
        }
        if extra:
            status_data.update(extra)
        with paths.status.open("w", encoding="utf-8") as handle:
            json.dump(status_data, handle, indent=2, sort_keys=True)
        self.append_timeline_event(
            "round_status_updated",
            {"round_id": int(round_id), "status": status},
        )
        return status_data

    def load_round_status(self, round_id: int) -> Dict[str, Any]:
        with self.get_round_paths(round_id).status.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_text(self, round_id: int, filename: str, content: str) -> Path:
        path = self.get_round_dir(round_id) / filename
        _ensure_parent(path)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def load_text(self, round_id: int, filename: str) -> str:
        path = self.get_round_dir(round_id) / filename
        with path.open("r", encoding="utf-8") as handle:
            return handle.read()

    def save_json(self, round_id: int, filename: str, data: Dict[str, Any]) -> Path:
        path = self.get_round_dir(round_id) / filename
        _ensure_parent(path)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        return path

    def load_json(self, round_id: int, filename: str) -> Dict[str, Any]:
        path = self.get_round_dir(round_id) / filename
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_generator_artifacts(
        self,
        round_id: int,
        *,
        prompt: str,
        response: Dict[str, Any],
        candidate_proposals: Optional[list[Dict[str, Any]]] = None,
        reward_spec: Optional[Dict[str, Any]] = None,
        policy_guidance_spec: Optional[Dict[str, Any]] = None,
        reward_code: str,
    ) -> RoundPaths:
        paths = self.ensure_round(round_id)
        self.save_text(round_id, paths.generator_prompt.name, prompt)
        self.save_json(round_id, paths.generator_response.name, response)
        if candidate_proposals is not None:
            self.save_json(
                round_id,
                paths.candidate_proposals.name,
                {"candidate_proposals": candidate_proposals},
            )
        if reward_spec is not None:
            self.save_json(round_id, paths.reward_spec.name, reward_spec)
        if policy_guidance_spec is not None:
            self.save_json(round_id, paths.policy_guidance.name, policy_guidance_spec)
        self.save_text(round_id, paths.reward_function.name, reward_code)
        self.save_round_status(round_id, "generator_done")
        return paths

    def save_critic_artifacts(
        self,
        round_id: int,
        *,
        prompt: str,
        response: Dict[str, Any],
        alpha_policy: Optional[Dict[str, Any]] = None,
    ) -> RoundPaths:
        paths = self.ensure_round(round_id)
        self.save_text(round_id, paths.critic_prompt.name, prompt)
        self.save_json(round_id, paths.critic_response.name, response)
        if alpha_policy is not None:
            self.save_json(round_id, paths.alpha_policy.name, alpha_policy)
        self.save_round_status(round_id, "critic_done")
        return paths

    def save_candidate_manifest(
        self,
        round_id: int,
        candidates: list[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        paths = self.ensure_round(round_id)
        payload: Dict[str, Any] = {"candidates": candidates}
        if metadata:
            payload["metadata"] = metadata
        return self.save_json(round_id, paths.candidate_manifest.name, payload)

    def save_candidate_execution_plan(
        self,
        round_id: int,
        execution_plan: Dict[str, Any],
    ) -> Path:
        paths = self.ensure_round(round_id)
        return self.save_json(round_id, paths.candidate_execution_plan.name, execution_plan)

    def save_candidate_reward_artifacts(
        self,
        round_id: int,
        *,
        candidate_id: str,
        reward_spec: Dict[str, Any],
        reward_code: str,
    ) -> CandidatePaths:
        paths = self.get_candidate_paths(round_id, candidate_id)
        paths.candidate_dir.mkdir(parents=True, exist_ok=True)
        with paths.reward_spec.open("w", encoding="utf-8") as handle:
            json.dump(reward_spec, handle, indent=2, sort_keys=True)
        with paths.reward_function.open("w", encoding="utf-8") as handle:
            handle.write(reward_code)
        return paths

    def record_training_run(
        self,
        round_id: int,
        *,
        train_config: Dict[str, Any],
        run_reference: Dict[str, Any],
        metrics_summary: Optional[Dict[str, Any]] = None,
    ) -> RoundPaths:
        paths = self.ensure_round(round_id)
        self.save_json(round_id, paths.train_config.name, train_config)
        self.save_json(round_id, paths.train_run_ref.name, run_reference)
        if metrics_summary is not None:
            self.save_json(round_id, paths.train_metrics_summary.name, metrics_summary)
        self.save_round_status(round_id, "training_finished")
        return paths

    def record_candidate_training_run(
        self,
        round_id: int,
        *,
        candidate_id: str,
        train_config: Dict[str, Any],
        run_reference: Dict[str, Any],
        metrics_summary: Optional[Dict[str, Any]] = None,
    ) -> CandidatePaths:
        paths = self.get_candidate_paths(round_id, candidate_id)
        paths.candidate_dir.mkdir(parents=True, exist_ok=True)
        with paths.train_config.open("w", encoding="utf-8") as handle:
            json.dump(train_config, handle, indent=2, sort_keys=True)
        with paths.train_run_ref.open("w", encoding="utf-8") as handle:
            json.dump(run_reference, handle, indent=2, sort_keys=True)
        if metrics_summary is not None:
            with paths.train_metrics_summary.open("w", encoding="utf-8") as handle:
                json.dump(metrics_summary, handle, indent=2, sort_keys=True)
        self.save_round_status(round_id, "training_finished")
        return paths

    def get_previous_round_artifacts(self, round_id: int) -> Optional[Dict[str, Any]]:
        previous_round = int(round_id) - 1
        if previous_round <= 0:
            return None
        previous_dir = self.get_round_dir(previous_round)
        if not previous_dir.exists():
            return None

        paths = self.get_round_paths(previous_round)
        data: Dict[str, Any] = {"round_id": previous_round}
        if paths.reward_spec.exists():
            data["reward_spec"] = self.load_json(previous_round, paths.reward_spec.name)
        if paths.policy_guidance.exists():
            data["policy_guidance_spec"] = self.load_json(
                previous_round, paths.policy_guidance.name
            )
        if paths.reward_function.exists():
            data["reward_code"] = paths.reward_function.read_text(encoding="utf-8")
        if paths.alpha_policy.exists():
            data["alpha_policy"] = self.load_json(previous_round, paths.alpha_policy.name)
        if paths.critic_response.exists():
            data["critic_response"] = self.load_json(previous_round, paths.critic_response.name)
        if paths.candidate_proposals.exists():
            data["candidate_proposals"] = self.load_json(
                previous_round, paths.candidate_proposals.name
            )
        if paths.train_metrics_summary.exists():
            data["train_metrics_summary"] = self.load_json(
                previous_round, paths.train_metrics_summary.name
            )
        if paths.candidate_manifest.exists():
            data["candidate_manifest"] = self.load_json(
                previous_round, paths.candidate_manifest.name
            )
        if paths.candidate_execution_plan.exists():
            data["candidate_execution_plan"] = self.load_json(
                previous_round, paths.candidate_execution_plan.name
            )
        candidate_results_path = self.get_round_dir(previous_round) / "candidate_results.json"
        if candidate_results_path.exists():
            with candidate_results_path.open("r", encoding="utf-8") as handle:
                data["candidate_results"] = json.load(handle)
        if paths.status.exists():
            data["status"] = self.load_round_status(previous_round)
        return data

    def get_round_artifacts(self, round_id: int) -> Optional[Dict[str, Any]]:
        round_id = int(round_id)
        round_dir = self.get_round_dir(round_id)
        if not round_dir.exists():
            return None

        paths = self.get_round_paths(round_id)
        data: Dict[str, Any] = {"round_id": round_id}
        if paths.reward_spec.exists():
            data["reward_spec"] = self.load_json(round_id, paths.reward_spec.name)
        if paths.policy_guidance.exists():
            data["policy_guidance_spec"] = self.load_json(
                round_id, paths.policy_guidance.name
            )
        if paths.reward_function.exists():
            data["reward_code"] = paths.reward_function.read_text(encoding="utf-8")
        if paths.alpha_policy.exists():
            data["alpha_policy"] = self.load_json(round_id, paths.alpha_policy.name)
        if paths.generator_response.exists():
            data["generator_response"] = self.load_json(round_id, paths.generator_response.name)
        if paths.candidate_proposals.exists():
            data["candidate_proposals"] = self.load_json(round_id, paths.candidate_proposals.name)
        if paths.critic_response.exists():
            data["critic_response"] = self.load_json(round_id, paths.critic_response.name)
        if paths.train_config.exists():
            data["train_config"] = self.load_json(round_id, paths.train_config.name)
        if paths.train_run_ref.exists():
            data["train_run_ref"] = self.load_json(round_id, paths.train_run_ref.name)
        if paths.train_metrics_summary.exists():
            data["train_metrics_summary"] = self.load_json(round_id, paths.train_metrics_summary.name)
        if paths.candidate_manifest.exists():
            data["candidate_manifest"] = self.load_json(round_id, paths.candidate_manifest.name)
        if paths.candidate_execution_plan.exists():
            data["candidate_execution_plan"] = self.load_json(
                round_id, paths.candidate_execution_plan.name
            )
        validation_path = round_dir / "validation_candidate_results.json"
        if validation_path.exists():
            with validation_path.open("r", encoding="utf-8") as handle:
                data["validation_candidate_results_payload"] = json.load(handle)
        candidate_results_path = round_dir / "candidate_results.json"
        if candidate_results_path.exists():
            with candidate_results_path.open("r", encoding="utf-8") as handle:
                data["candidate_results"] = json.load(handle)
        if paths.status.exists():
            data["status"] = self.load_round_status(round_id)
        return data

    def mark_round_failed(
        self,
        round_id: int,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {"reason": reason}
        if extra:
            payload.update(extra)
        return self.save_round_status(round_id, "failed", payload)

    def mark_workflow_completed(self) -> Dict[str, Any]:
        manifest = self.update_manifest({"status": "completed"})
        self.append_timeline_event(
            "workflow_completed",
            {"workflow_id": self.workflow_id},
        )
        return manifest
