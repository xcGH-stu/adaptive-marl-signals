from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import os
from types import ModuleType
from typing import Any, Dict, Optional


@dataclass
class RuntimeContext:
    t_env: int
    episode_idx: int
    episode_step: int
    test_mode: bool
    obs_before: Any
    obs_after: Any
    state_before: Any
    state_after: Any
    actions: Any
    sparse_reward: Any
    done: bool
    truncated: bool
    info: Dict[str, Any]
    raw_env: Any
    lbf_state_before: Any = None
    lbf_state_after: Any = None


@dataclass
class RewardComputationResult:
    sparse_reward: Any
    dense_reward: Any
    mixed_reward: Any
    alpha: float
    dense_reward_applied: bool
    reward_breakdown: Dict[str, Any] = field(default_factory=dict)
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class RuntimeState:
    workflow_id: Optional[str] = None
    round_id: Optional[int] = None
    reward_module_path: Optional[str] = None
    alpha_policy_config: Optional[Dict[str, Any]] = None
    policy_guidance_config: Optional[Dict[str, Any]] = None
    last_t_env: int = 0
    last_episode_idx: int = 0
    last_test_mode: bool = False


class RewardRuntimeWrapper:
    """
    Execute the current round's dense reward function and alpha policy at runtime.

    This class does not decide how reward shaping should evolve across workflow
    rounds. It only interprets the externally supplied reward module and alpha
    policy configuration for the current training run.
    """

    def __init__(
        self,
        reward_module_path: Optional[str] = None,
        alpha_policy_config: Optional[Dict[str, Any]] = None,
        policy_guidance_config: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        round_id: Optional[int] = None,
        apply_dense_reward_in_eval: bool = False,
        fail_open: bool = True,
    ):
        self.state = RuntimeState(
            workflow_id=workflow_id,
            round_id=round_id,
            reward_module_path=reward_module_path,
            alpha_policy_config=alpha_policy_config,
            policy_guidance_config=policy_guidance_config,
        )
        self.apply_dense_reward_in_eval = apply_dense_reward_in_eval
        self.fail_open = fail_open

        self._reward_module: Optional[ModuleType] = None
        self._reward_fn = None
        self._load_errors: list[str] = []
        self._module_loaded = False

    def set_training_context(
        self,
        t_env: int,
        episode_idx: int,
        test_mode: bool,
    ) -> None:
        self.state.last_t_env = int(t_env)
        self.state.last_episode_idx = int(episode_idx)
        self.state.last_test_mode = bool(test_mode)

    def process_transition(
        self,
        *,
        raw_env: Any,
        obs_before: Any,
        obs_after: Any,
        state_before: Any,
        state_after: Any,
        actions: Any,
        sparse_reward: Any,
        done: bool,
        truncated: bool,
        info: Optional[Dict[str, Any]] = None,
        episode_step: int = 0,
        lbf_state_before: Any = None,
        lbf_state_after: Any = None,
    ) -> RewardComputationResult:
        runtime_info = dict(info or {})
        context = RuntimeContext(
            t_env=self.state.last_t_env,
            episode_idx=self.state.last_episode_idx,
            episode_step=int(episode_step),
            test_mode=self.state.last_test_mode,
            obs_before=obs_before,
            obs_after=obs_after,
            state_before=state_before,
            state_after=state_after,
            actions=actions,
            sparse_reward=sparse_reward,
            done=bool(done),
            truncated=bool(truncated),
            info=runtime_info,
            raw_env=raw_env,
            lbf_state_before=lbf_state_before,
            lbf_state_after=lbf_state_after,
        )

        if context.test_mode and not self.apply_dense_reward_in_eval:
            return RewardComputationResult(
                sparse_reward=sparse_reward,
                dense_reward=self._zero_like(sparse_reward),
                mixed_reward=sparse_reward,
                alpha=0.0,
                dense_reward_applied=False,
                reward_breakdown={"mode": "eval_sparse_only"},
                runtime_metadata=self._runtime_metadata(),
            )

        try:
            dense_reward, breakdown = self._compute_dense_reward(context)
            alpha = self._resolve_alpha(context)
            mixed_reward = self._mix_rewards(sparse_reward, dense_reward, alpha)
            return RewardComputationResult(
                sparse_reward=sparse_reward,
                dense_reward=dense_reward,
                mixed_reward=mixed_reward,
                alpha=alpha,
                dense_reward_applied=bool(alpha != 0.0),
                reward_breakdown=breakdown,
                runtime_metadata=self._runtime_metadata(),
                errors=list(self._load_errors),
            )
        except Exception as exc:
            error_message = f"runtime_wrapper_error: {exc}"
            if not self.fail_open:
                raise
            return self._fallback_result(sparse_reward, error_message)

    def _compute_dense_reward(
        self,
        context: RuntimeContext,
    ) -> tuple[Any, Dict[str, Any]]:
        reward_fn = self._get_reward_fn()
        if reward_fn is None:
            return self._zero_like(context.sparse_reward), {"mode": "no_reward_module"}

        result = reward_fn(context)
        if isinstance(result, tuple):
            dense_reward = result[0]
            breakdown = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
        else:
            dense_reward = result
            breakdown = {}

        dense_reward = self._normalise_reward_shape(dense_reward, context.sparse_reward)
        return dense_reward, breakdown

    def _resolve_alpha(
        self,
        context: RuntimeContext,
    ) -> float:
        config = self.state.alpha_policy_config or {}
        if not config:
            return 1.0

        policy_type = config.get("type", "constant")
        if policy_type == "constant":
            return float(config.get("value", 1.0))

        if policy_type == "piecewise_by_t_env":
            return self._piecewise_value(
                current_value=context.t_env,
                segments=config.get("segments", []),
                default=float(config.get("default", 1.0)),
            )

        if policy_type == "piecewise_by_episode":
            return self._piecewise_value(
                current_value=context.episode_idx,
                segments=config.get("segments", []),
                default=float(config.get("default", 1.0)),
            )

        if policy_type == "linear_decay":
            start = float(config.get("start", 1.0))
            end = float(config.get("end", 0.0))
            start_t = int(config.get("start_t", 0))
            end_t = int(config.get("end_t", start_t))
            return self._linear_decay(context.t_env, start, end, start_t, end_t)

        raise ValueError(f"Unsupported alpha policy type: {policy_type}")

    def _mix_rewards(
        self,
        sparse_reward: Any,
        dense_reward: Any,
        alpha: float,
    ) -> Any:
        if isinstance(sparse_reward, (list, tuple)):
            return [
                sparse_component + alpha * dense_component
                for sparse_component, dense_component in zip(sparse_reward, dense_reward)
            ]
        return sparse_reward + alpha * dense_reward

    def _normalise_reward_shape(self, reward: Any, sparse_reward: Any) -> Any:
        if isinstance(sparse_reward, tuple):
            if isinstance(reward, tuple):
                return reward
            if isinstance(reward, list):
                return tuple(reward)
            if self._is_scalar(reward):
                return tuple(float(reward) for _ in sparse_reward)
        elif isinstance(sparse_reward, list):
            if isinstance(reward, list):
                return reward
            if isinstance(reward, tuple):
                return list(reward)
            if self._is_scalar(reward):
                return [float(reward) for _ in sparse_reward]
        else:
            if isinstance(reward, (list, tuple)):
                if len(reward) != 1:
                    raise ValueError(
                        "Dense reward shape mismatch: scalar sparse reward received "
                        f"multi-value dense reward of length {len(reward)}"
                    )
                return reward[0]
        return reward

    def _fallback_result(
        self,
        sparse_reward: Any,
        error_message: str,
    ) -> RewardComputationResult:
        errors = list(self._load_errors)
        errors.append(error_message)
        return RewardComputationResult(
            sparse_reward=sparse_reward,
            dense_reward=self._zero_like(sparse_reward),
            mixed_reward=sparse_reward,
            alpha=0.0,
            dense_reward_applied=False,
            reward_breakdown={"mode": "fallback_sparse_only"},
            runtime_metadata=self._runtime_metadata(),
            errors=errors,
        )

    def _get_reward_fn(self):
        if self._module_loaded:
            return self._reward_fn

        self._module_loaded = True
        module_path = self.state.reward_module_path
        if not module_path:
            return None

        if not os.path.isfile(module_path):
            self._load_errors.append(f"reward module not found: {module_path}")
            return None

        module_name = self._module_name_from_path(module_path)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            self._load_errors.append(f"unable to load module spec: {module_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reward_fn = getattr(module, "compute_dense_reward", None)
        if reward_fn is None or not callable(reward_fn):
            self._load_errors.append(
                "reward module does not define callable compute_dense_reward(context)"
            )
            return None

        self._reward_module = module
        self._reward_fn = reward_fn
        return self._reward_fn

    def _module_name_from_path(self, module_path: str) -> str:
        basename = os.path.basename(module_path)
        stem, _ = os.path.splitext(basename)
        workflow_id = self.state.workflow_id or "workflow"
        round_id = self.state.round_id if self.state.round_id is not None else "round"
        return f"reward_module_{workflow_id}_{round_id}_{stem}"

    def _piecewise_value(
        self,
        *,
        current_value: int,
        segments: list[Dict[str, Any]],
        default: float,
    ) -> float:
        for segment in segments:
            start = int(segment.get("start", 0))
            end = segment.get("end")
            if end is None:
                if current_value >= start:
                    return float(segment["alpha"])
                continue
            if start <= current_value <= int(end):
                return float(segment["alpha"])
        return default

    def _linear_decay(
        self,
        current_value: int,
        start: float,
        end: float,
        start_t: int,
        end_t: int,
    ) -> float:
        if current_value <= start_t:
            return start
        if current_value >= end_t:
            return end
        if end_t <= start_t:
            return end
        ratio = (current_value - start_t) / float(end_t - start_t)
        return start + ratio * (end - start)

    def _runtime_metadata(self) -> Dict[str, Any]:
        policy_guidance = self.state.policy_guidance_config or {}
        return {
            "workflow_id": self.state.workflow_id,
            "round_id": self.state.round_id,
            "reward_module_path": self.state.reward_module_path,
            "alpha_policy_type": (self.state.alpha_policy_config or {}).get(
                "type", "constant"
            ),
            "policy_guidance_enabled": any(
                bool(value.get("enabled"))
                for value in policy_guidance.values()
                if isinstance(value, dict)
            ),
            "policy_guidance_keys": sorted(
                key
                for key, value in policy_guidance.items()
                if isinstance(value, dict)
            ),
        }

    def _zero_like(self, reward: Any) -> Any:
        if isinstance(reward, tuple):
            return tuple(0.0 for _ in reward)
        if isinstance(reward, list):
            return [0.0 for _ in reward]
        return 0.0

    def _is_scalar(self, value: Any) -> bool:
        return isinstance(value, (int, float))
