from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional

from experiments.qmix_hparam_presets import resolve_qmix_hparam_preset
from workflows.log_analyzer import TrainingLogAnalyzer
from workflows.train_checkpointing import (
    build_resume_train_config,
    normalize_checkpoint_steps,
    resolve_checkpoint_step,
)


@dataclass
class TrainLaunchResult:
    train_config: Dict[str, Any]
    run_reference: Dict[str, Any]
    metrics_summary: Dict[str, Any]


class EPyMARLTrainLauncher:
    """
    Launch an EPyMARL training run for a workflow round and return references to
    the resulting Sacred artifacts.
    """

    def __init__(
        self,
        repo_root: Optional[Path | str] = None,
        python_executable: Optional[str] = None,
    ):
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.python_executable = python_executable or sys.executable
        self.log_analyzer = TrainingLogAnalyzer()

    def run_training(
        self,
        *,
        round_id: int,
        workflow_spec: Dict[str, Any],
        reward_module_path: str,
        alpha_policy: Optional[Dict[str, Any]],
        policy_guidance_spec: Optional[Dict[str, Any]],
        workflow_id: str,
        candidate_id: Optional[str] = None,
        reward_paradigm: Optional[str] = None,
        active_pbrs_field: Optional[str] = None,
        active_pbrs_field_source: Optional[str] = None,
        candidate_value: Optional[float] = None,
        active_checkpoint_context: Optional[Dict[str, Any]] = None,
        active_field_carryover_context: Optional[Dict[str, Any]] = None,
        candidate_selection_context: Optional[Dict[str, Any]] = None,
        phase_name: str = "main",
        train_overrides_override: Optional[Dict[str, Any]] = None,
        label_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        training_plan = self.build_training_plan(
            round_id=round_id,
            workflow_spec=workflow_spec,
            reward_module_path=reward_module_path,
            alpha_policy=alpha_policy,
            policy_guidance_spec=policy_guidance_spec,
            workflow_id=workflow_id,
            candidate_id=candidate_id,
            reward_paradigm=reward_paradigm,
            active_pbrs_field=active_pbrs_field,
            active_pbrs_field_source=active_pbrs_field_source,
            candidate_value=candidate_value,
            active_checkpoint_context=active_checkpoint_context,
            active_field_carryover_context=active_field_carryover_context,
            candidate_selection_context=candidate_selection_context,
            phase_name=phase_name,
            train_overrides_override=train_overrides_override,
            label_override=label_override,
        )
        return self.execute_training_plan(training_plan)

    def execute_training_plan(
        self,
        training_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        train_config = training_plan["train_config"]
        command = training_plan["command"]
        return self._execute_train_config(train_config=train_config, command=command)

    def run_prebuilt_train_config(
        self,
        train_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        command = self._build_command(train_config)
        return self._execute_train_config(train_config=train_config, command=command)

    def launch_prebuilt_train_config_background(
        self,
        train_config: Dict[str, Any],
        *,
        stdout_path: Optional[Path | str] = None,
        stderr_to_stdout: bool = True,
    ) -> Dict[str, Any]:
        command = self._build_command(train_config)
        stdout_handle = None
        if stdout_path is not None:
            stdout_file = Path(stdout_path)
            stdout_file.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = stdout_file.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                text=True,
                stdout=stdout_handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT if stderr_to_stdout else subprocess.DEVNULL,
                start_new_session=True,
            )
        finally:
            if stdout_handle is not None:
                stdout_handle.close()
        return {
            "train_config": train_config,
            "command": command,
            "pid": process.pid,
            "stdout_path": str(stdout_path) if stdout_path is not None else None,
        }

    def recover_existing_result(
        self,
        train_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        run_reference = self._find_sacred_run_reference(train_config)
        metrics_summary = self.log_analyzer.summarize(
            train_config=train_config,
            run_reference=run_reference,
        )
        return TrainLaunchResult(
            train_config=train_config,
            run_reference=run_reference,
            metrics_summary=metrics_summary,
        ).__dict__

    def find_matching_run_references(
        self,
        train_config: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        matching_runs = self._find_matching_sacred_run_dirs(train_config)
        references: list[Dict[str, Any]] = []
        for run_dir in matching_runs:
            run_json_path = run_dir / "run.json"
            run_payload: Dict[str, Any] = {}
            if run_json_path.exists():
                try:
                    with run_json_path.open("r", encoding="utf-8") as handle:
                        run_payload = json.load(handle)
                except Exception:
                    run_payload = {}
            run_reference = {
                "sacred_base_dir": str(run_dir.parent),
                "run_dir": str(run_dir),
                "run_id": int(run_dir.name),
                "config_json": str(run_dir / "config.json"),
                "run_json": str(run_json_path),
                "info_json": str(run_dir / "info.json"),
                "metrics_json": str(run_dir / "metrics.json"),
                "cout_txt": str(run_dir / "cout.txt"),
                "status": run_payload.get("status"),
                "heartbeat": run_payload.get("heartbeat"),
                "stop_time": run_payload.get("stop_time"),
            }
            references.append(
                self._augment_run_reference_with_checkpoint_info(
                    train_config=train_config,
                    run_reference=run_reference,
                )
            )
        references.sort(key=lambda item: int(item.get("run_id") or -1))
        return references

    def _execute_train_config(
        self,
        *,
        train_config: Dict[str, Any],
        command: list[str],
    ) -> Dict[str, Any]:
        process = subprocess.run(
            command,
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

        if process.returncode != 0:
            raise RuntimeError(
                "training command failed with exit code "
                f"{process.returncode}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            )

        run_reference = self._find_sacred_run_reference(
            train_config,
        )
        run_reference["stdout"] = process.stdout
        run_reference["stderr"] = process.stderr
        run_reference["command"] = command

        metrics_summary = self.log_analyzer.summarize(
            train_config=train_config,
            run_reference=run_reference,
        )
        return TrainLaunchResult(
            train_config=train_config,
            run_reference=run_reference,
            metrics_summary=metrics_summary,
        ).__dict__

    def build_training_plan(
        self,
        *,
        round_id: int,
        workflow_spec: Dict[str, Any],
        reward_module_path: str,
        alpha_policy: Optional[Dict[str, Any]],
        policy_guidance_spec: Optional[Dict[str, Any]],
        workflow_id: str,
        candidate_id: Optional[str] = None,
        reward_paradigm: Optional[str] = None,
        active_pbrs_field: Optional[str] = None,
        active_pbrs_field_source: Optional[str] = None,
        candidate_value: Optional[float] = None,
        active_checkpoint_context: Optional[Dict[str, Any]] = None,
        active_field_carryover_context: Optional[Dict[str, Any]] = None,
        candidate_selection_context: Optional[Dict[str, Any]] = None,
        phase_name: str = "main",
        train_overrides_override: Optional[Dict[str, Any]] = None,
        label_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        train_config = self._build_train_config(
            round_id=round_id,
            workflow_spec=workflow_spec,
            reward_module_path=reward_module_path,
            alpha_policy=alpha_policy,
            policy_guidance_spec=policy_guidance_spec,
            workflow_id=workflow_id,
            candidate_id=candidate_id,
            reward_paradigm=reward_paradigm,
            active_pbrs_field=active_pbrs_field,
            active_pbrs_field_source=active_pbrs_field_source,
            candidate_value=candidate_value,
            active_checkpoint_context=active_checkpoint_context,
            active_field_carryover_context=active_field_carryover_context,
            candidate_selection_context=candidate_selection_context,
            phase_name=phase_name,
            train_overrides_override=train_overrides_override,
            label_override=label_override,
        )
        return {
            "train_config": train_config,
            "command": self._build_command(train_config),
        }

    def _build_train_config(
        self,
        *,
        round_id: int,
        workflow_spec: Dict[str, Any],
        reward_module_path: str,
        alpha_policy: Optional[Dict[str, Any]],
        policy_guidance_spec: Optional[Dict[str, Any]],
        workflow_id: str,
        candidate_id: Optional[str],
        reward_paradigm: Optional[str],
        active_pbrs_field: Optional[str],
        active_pbrs_field_source: Optional[str],
        candidate_value: Optional[float],
        active_checkpoint_context: Optional[Dict[str, Any]],
        active_field_carryover_context: Optional[Dict[str, Any]],
        candidate_selection_context: Optional[Dict[str, Any]],
        phase_name: str,
        train_overrides_override: Optional[Dict[str, Any]],
        label_override: Optional[str],
    ) -> Dict[str, Any]:
        train_spec = dict(workflow_spec.get("train", {}))
        overrides = dict(train_spec.get("overrides", {}))
        if isinstance(train_overrides_override, dict):
            overrides.update(train_overrides_override)
        env_args = dict(train_spec.get("env_args", {}))
        checkpointing = dict(train_spec.get("checkpointing", {}))
        use_dense_reward = bool(train_spec.get("use_dense_reward", True))
        seed = int(train_spec.get("seed", 1))
        save_model = bool(checkpointing.get("enabled", False))
        save_model_interval = int(checkpointing.get("save_model_interval", 50000))
        save_final_model = bool(checkpointing.get("save_final_model", False))
        checkpoint_path = str(checkpointing.get("checkpoint_path", "") or "")
        load_step = int(checkpointing.get("load_step", 0) or 0)
        local_results_path = str(
            checkpointing.get("local_results_path", train_spec.get("local_results_path", "results"))
            or "results"
        )

        label_suffix = f"_{candidate_id}" if candidate_id else ""
        if phase_name and phase_name != "main":
            label_suffix += f"_{phase_name}"
        label = (
            str(label_override)
            if label_override is not None
            else f"workflow_{workflow_id}_round_{round_id}{label_suffix}"
        )
        experiment_metadata = dict(workflow_spec.get("experiment_metadata", {}))
        experiment_metadata.setdefault("seed", seed)
        resolved_config, qmix_hparam_preset = resolve_qmix_hparam_preset(
            train_spec.get("config", "qmix"),
            train_spec.get("qmix_hparam_preset"),
        )
        experiment_metadata.setdefault("alg_config", resolved_config)
        experiment_metadata.setdefault("qmix_hparam_preset", qmix_hparam_preset)
        train_config = {
            "config": resolved_config,
            "seed": seed,
            "env_config": train_spec.get("env_config", "gymma"),
            "env_args": env_args,
            "overrides": overrides,
            "label": label,
            "use_dense_reward": use_dense_reward,
            "reward_module_path": reward_module_path,
            "alpha_policy_config": alpha_policy,
            "policy_guidance_config": policy_guidance_spec,
            "workflow_id": workflow_id,
            "workflow_round": round_id,
            "reward_paradigm": reward_paradigm,
            "active_pbrs_field": active_pbrs_field,
            "active_pbrs_field_source": active_pbrs_field_source,
            "candidate_value": candidate_value,
            "active_checkpoint_context": active_checkpoint_context,
            "active_field_carryover_context": active_field_carryover_context,
            "candidate_selection_context": candidate_selection_context,
            "phase_name": phase_name,
            "experiment_metadata": experiment_metadata,
            "alg_config": resolved_config,
            "qmix_hparam_preset": qmix_hparam_preset,
            "save_model": save_model,
            "save_model_interval": save_model_interval,
            "save_final_model": save_final_model,
            "checkpoint_path": checkpoint_path,
            "load_step": load_step,
            "local_results_path": local_results_path,
            "apply_dense_reward_in_eval": bool(
                train_spec.get("apply_dense_reward_in_eval", False)
            ),
            "log_dense_reward_details": bool(
                train_spec.get("log_dense_reward_details", True)
            ),
        }
        return train_config

    def _build_command(self, train_config: Dict[str, Any]) -> list[str]:
        command = [
            self.python_executable,
            "src/main.py",
            f"--config={train_config['config']}",
            f"--env-config={train_config['env_config']}",
            "with",
        ]

        env_args = train_config.get("env_args", {})
        for key, value in env_args.items():
            command.append(f"env_args.{key}={self._format_sacred_value(value)}")

        overrides = train_config.get("overrides", {})
        for key, value in overrides.items():
            command.append(f"{key}={self._format_sacred_value(value)}")

        command.extend(
            [
                f"seed={self._format_sacred_value(train_config['seed'])}",
                f"label={self._format_sacred_value(train_config['label'])}",
                f"use_dense_reward={self._format_sacred_value(train_config['use_dense_reward'])}",
                f"reward_module_path={self._format_sacred_value(train_config['reward_module_path'])}",
                f"alpha_policy_config={self._format_sacred_value(train_config['alpha_policy_config'])}",
                f"policy_guidance_config={self._format_sacred_value(train_config['policy_guidance_config'])}",
                f"workflow_id={self._format_sacred_value(train_config['workflow_id'])}",
                f"workflow_round={self._format_sacred_value(train_config['workflow_round'])}",
                f"alg_config={self._format_sacred_value(train_config['alg_config'])}",
                "qmix_hparam_preset="
                f"{self._format_sacred_value(train_config['qmix_hparam_preset'])}",
                f"save_model={self._format_sacred_value(train_config['save_model'])}",
                "save_model_interval="
                f"{self._format_sacred_value(train_config['save_model_interval'])}",
                f"checkpoint_path={self._format_sacred_value(train_config['checkpoint_path'])}",
                f"load_step={self._format_sacred_value(train_config['load_step'])}",
                f"save_final_model={self._format_sacred_value(train_config['save_final_model'])}",
                "local_results_path="
                f"{self._format_sacred_value(train_config['local_results_path'])}",
                "apply_dense_reward_in_eval="
                f"{self._format_sacred_value(train_config['apply_dense_reward_in_eval'])}",
                "log_dense_reward_details="
                f"{self._format_sacred_value(train_config['log_dense_reward_details'])}",
                "experiment_metadata="
                f"{self._format_sacred_value(train_config['experiment_metadata'])}",
            ]
        )
        return command

    def build_checkpoint_resume_plan(
        self,
        *,
        base_train_config: Dict[str, Any],
        run_reference: Dict[str, Any],
        requested_step: Optional[int] = None,
        label_suffix: str = "_branch",
        use_dense_reward: Optional[bool] = None,
        reward_module_path: Optional[str] = None,
        alpha_policy_config: Optional[Dict[str, Any]] = None,
        policy_guidance_config: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        workflow_round: Optional[int] = None,
        reward_paradigm: Optional[str] = None,
        active_pbrs_field: Optional[str] = None,
        active_pbrs_field_source: Optional[str] = None,
        candidate_value: Optional[float] = None,
        active_checkpoint_context: Optional[Dict[str, Any]] = None,
        active_field_carryover_context: Optional[Dict[str, Any]] = None,
        candidate_selection_context: Optional[Dict[str, Any]] = None,
        phase_name: Optional[str] = None,
        save_model: Optional[bool] = None,
        save_model_interval: Optional[int] = None,
        save_final_model: Optional[bool] = None,
        local_results_path: Optional[str] = None,
        override_env_args: Optional[Dict[str, Any]] = None,
        override_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        checkpoint_root_dir = run_reference.get("checkpoint_root_dir")
        if not isinstance(checkpoint_root_dir, str) or not checkpoint_root_dir:
            raise ValueError("run_reference does not contain checkpoint_root_dir")

        available_steps = normalize_checkpoint_steps(
            run_reference.get("available_checkpoint_steps")
        )
        checkpoint_step = resolve_checkpoint_step(available_steps, requested_step)
        resume_train_config = build_resume_train_config(
            base_train_config=base_train_config,
            checkpoint_root_dir=checkpoint_root_dir,
            checkpoint_step=checkpoint_step,
            label_suffix=label_suffix,
            use_dense_reward=use_dense_reward,
            reward_module_path=reward_module_path,
            alpha_policy_config=alpha_policy_config,
            policy_guidance_config=policy_guidance_config,
            workflow_id=workflow_id,
            workflow_round=workflow_round,
            reward_paradigm=reward_paradigm,
            active_pbrs_field=active_pbrs_field,
            active_pbrs_field_source=active_pbrs_field_source,
            candidate_value=candidate_value,
            active_checkpoint_context=active_checkpoint_context,
            active_field_carryover_context=active_field_carryover_context,
            candidate_selection_context=candidate_selection_context,
            phase_name=phase_name,
            save_model=save_model,
            save_model_interval=save_model_interval,
            save_final_model=save_final_model,
            local_results_path=local_results_path,
            override_env_args=override_env_args,
            override_overrides=override_overrides,
        )
        return {
            "checkpoint_root_dir": checkpoint_root_dir,
            "available_checkpoint_steps": available_steps,
            "selected_checkpoint_step": checkpoint_step,
            "resume_train_config": resume_train_config,
        }

    def _format_sacred_value(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (dict, list)):
            escaped = json.dumps(value, separators=(",", ":")).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _find_sacred_run_reference(
        self,
        train_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        import time

        deadline = time.time() + 15.0
        last_error: Optional[Exception] = None
        target_run: Optional[Path] = None
        sacred_config_root = self.repo_root / "results" / "sacred" / train_config["config"]

        while time.time() < deadline:
            if not sacred_config_root.exists():
                last_error = FileNotFoundError(
                    f"Sacred config directory not found: {sacred_config_root}"
                )
                time.sleep(0.5)
                continue

            matching_runs = self._find_matching_sacred_run_dirs(train_config)

            if matching_runs:
                candidate_run = max(matching_runs, key=lambda path: path.stat().st_mtime)
                metrics_json = candidate_run / "metrics.json"
                info_json = candidate_run / "info.json"
                if metrics_json.exists() and info_json.exists():
                    target_run = candidate_run
                    break
            time.sleep(0.5)

        if target_run is None:
            if last_error is None:
                last_error = FileNotFoundError(
                    "Unable to find Sacred run matching this workflow round. "
                    f"workflow_id={train_config['workflow_id']} "
                    f"workflow_round={train_config['workflow_round']}"
                )
            raise last_error

        sacred_base = target_run.parent
        run_reference = {
            "sacred_base_dir": str(sacred_base),
            "run_dir": str(target_run),
            "run_id": int(target_run.name),
            "config_json": str(target_run / "config.json"),
            "run_json": str(target_run / "run.json"),
            "info_json": str(target_run / "info.json"),
            "metrics_json": str(target_run / "metrics.json"),
            "cout_txt": str(target_run / "cout.txt"),
        }
        return self._augment_run_reference_with_checkpoint_info(
            train_config=train_config,
            run_reference=run_reference,
        )

    def _find_matching_sacred_run_dirs(
        self,
        train_config: Dict[str, Any],
    ) -> list[Path]:
        sacred_config_root = self.repo_root / "results" / "sacred" / train_config["config"]
        if not sacred_config_root.exists():
            return []
        matching_runs: list[Path] = []
        for config_path in sacred_config_root.glob("*/[0-9]*/config.json"):
            run_dir = config_path.parent
            try:
                with config_path.open("r", encoding="utf-8") as handle:
                    config_data = json.load(handle)
            except Exception:
                continue

            if config_data.get("label") != train_config["label"]:
                continue
            if config_data.get("workflow_id") != train_config["workflow_id"]:
                continue
            if config_data.get("workflow_round") != train_config["workflow_round"]:
                continue
            if config_data.get("reward_module_path") != train_config["reward_module_path"]:
                continue
            matching_runs.append(run_dir)
        return matching_runs

    def _augment_run_reference_with_checkpoint_info(
        self,
        *,
        train_config: Dict[str, Any],
        run_reference: Dict[str, Any],
    ) -> Dict[str, Any]:
        config_json_path = Path(run_reference["config_json"])
        info_json_path = Path(run_reference["info_json"])
        sacred_config: Dict[str, Any] = {}
        sacred_info: Dict[str, Any] = {}
        if config_json_path.exists():
            try:
                with config_json_path.open("r", encoding="utf-8") as handle:
                    sacred_config = json.load(handle)
            except Exception:
                sacred_config = {}
        if info_json_path.exists():
            try:
                with info_json_path.open("r", encoding="utf-8") as handle:
                    sacred_info = json.load(handle)
            except Exception:
                sacred_info = {}

        if not sacred_config and not sacred_info:
            run_reference["checkpoint_root_dir"] = None
            run_reference["available_checkpoint_steps"] = []
            run_reference["latest_checkpoint_step"] = None
            return run_reference

        checkpoint_root_dir = sacred_info.get("model_root_path")
        if not isinstance(checkpoint_root_dir, str) or not checkpoint_root_dir:
            unique_token = sacred_info.get("unique_token", sacred_config.get("unique_token"))
            local_results_path = sacred_config.get(
                "local_results_path",
                train_config.get("local_results_path", "results"),
            )
            if isinstance(unique_token, str) and unique_token:
                checkpoint_root_dir = str(
                    (self.repo_root / str(local_results_path) / "models" / unique_token).resolve()
                )

        available_checkpoint_steps = []
        saved_model_steps = sacred_info.get("saved_model_steps")
        if isinstance(saved_model_steps, list):
            for value in saved_model_steps:
                try:
                    available_checkpoint_steps.append(int(value))
                except (TypeError, ValueError):
                    continue
            available_checkpoint_steps = sorted(set(available_checkpoint_steps))

        latest_model_path = sacred_info.get("latest_model_path")
        if (
            (not isinstance(checkpoint_root_dir, str) or not checkpoint_root_dir)
            and isinstance(latest_model_path, str)
            and latest_model_path
        ):
            checkpoint_root_dir = str(Path(latest_model_path).parent)
        if checkpoint_root_dir:
            checkpoint_root_candidate = Path(checkpoint_root_dir)
            if not checkpoint_root_candidate.is_absolute():
                checkpoint_root_candidate = (self.repo_root / checkpoint_root_candidate).resolve()
            checkpoint_root = checkpoint_root_candidate
        else:
            checkpoint_root = None
        if checkpoint_root is not None and checkpoint_root.exists():
            discovered_steps = []
            for child in checkpoint_root.iterdir():
                if child.is_dir() and child.name.isdigit():
                    discovered_steps.append(int(child.name))
            available_checkpoint_steps = sorted(set(available_checkpoint_steps) | set(discovered_steps))
        if not available_checkpoint_steps and isinstance(latest_model_path, str) and latest_model_path:
            latest_model_candidate = Path(latest_model_path)
            if not latest_model_candidate.is_absolute():
                latest_model_candidate = (self.repo_root / latest_model_candidate).resolve()
            if latest_model_candidate.name.isdigit():
                available_checkpoint_steps = [int(latest_model_candidate.name)]
            if checkpoint_root is None:
                checkpoint_root = latest_model_candidate.parent

        run_reference["checkpoint_root_dir"] = str(checkpoint_root) if checkpoint_root is not None else None
        run_reference["available_checkpoint_steps"] = available_checkpoint_steps
        run_reference["latest_checkpoint_step"] = (
            available_checkpoint_steps[-1] if available_checkpoint_steps else None
        )
        run_reference["loaded_checkpoint_path"] = sacred_info.get("loaded_checkpoint_path")
        run_reference["loaded_checkpoint_step"] = sacred_info.get("loaded_checkpoint_step")
        run_reference["latest_model_path"] = (
            str((self.repo_root / Path(latest_model_path)).resolve())
            if isinstance(latest_model_path, str) and latest_model_path and not Path(latest_model_path).is_absolute()
            else latest_model_path
        )
        return run_reference
