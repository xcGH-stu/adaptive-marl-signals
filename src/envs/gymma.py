from collections.abc import Iterable
from copy import deepcopy
import json
import warnings

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import numpy as np

from .multiagentenv import MultiAgentEnv
from .wrappers import FlattenObservation
import envs.pretrained as pretrained  # noqa
from rewarding.lbf_pbrs import build_lbf_pbrs, extract_lbf_state
from rewarding.lbf_pbrs_v2 import LBFPBRSV2Runtime, normalize_lbf_pbrs_v2_config
from rewarding.runtime_wrapper import RewardRuntimeWrapper

try:
    import rware  # noqa: F401
except ImportError:
    warnings.warn(
        "RWARE is not installed, so robotic warehouse environments will not be available!"
    )

try:
    from .pz_wrapper import PettingZooWrapper  # noqa
except ImportError:
    warnings.warn(
        "PettingZoo is not installed, so these environments will not be available! To install, run `pip install pettingzoo`"
    )

try:
    from .vmas_wrapper import VMASWrapper  # noqa
except ImportError:
    warnings.warn(
        "VMAS is not installed, so these environments will not be available! To install, run `pip install 'vmas[gymnasium]'`"
    )


class GymmaWrapper(MultiAgentEnv):
    def __init__(
        self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        use_pbrs=False,
        eval_use_pbrs=False,
        pbrs_beta=1.0,
        pbrs_wc=0.6,
        pbrs_wp=0.4,
        pbrs_version="",
        pbrs_mode="",
        pbrs_active_terms=None,
        pbrs_weights=None,
        pbrs_gamma=0.99,
        pbrs_variant="original",
        pbrs_semi_strict_gate_radius=2,
        pbrs_beta_schedule_enabled=False,
        pbrs_beta_schedule_first=0.3,
        pbrs_beta_schedule_second=0.9,
        pbrs_beta_schedule_switch_ratio=0.5,
        pbrs_stage_schedule_enabled=False,
        pbrs_stage_boundary_1=875923,
        pbrs_stage_boundary_2=1451234,
        pbrs_stage_beta_early=0.3,
        pbrs_stage_beta_mid=1.0,
        pbrs_stage_beta_late=0.5,
        pbrs_stage_wc_early=0.1,
        pbrs_stage_wc_mid=1.0,
        pbrs_stage_wc_late=0.7,
        pbrs_stage_wp_early=0.7,
        pbrs_stage_wp_mid=0.1,
        pbrs_stage_wp_late=0.3,
        use_dense_reward=False,
        reward_module_path=None,
        alpha_policy_config=None,
        policy_guidance_config=None,
        workflow_id=None,
        workflow_round=None,
        apply_dense_reward_in_eval=False,
        log_dense_reward_details=True,
        **kwargs,
    ):
        self._env = gym.make(f"{key}", **kwargs)
        self._env = TimeLimit(self._env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if isinstance(pretrained_wrapper, str) and pretrained_wrapper.strip().lower() in {
            "",
            "null",
            "none",
        }:
            pretrained_wrapper = None

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.unwrapped.n_agents
        self.episode_limit = time_limit
        self._obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        try:
            self._env.unwrapped.seed(self._seed)
        except:
            self._env.reset(seed=self._seed)

        self.common_reward = common_reward
        self.key = key
        self.use_pbrs = bool(use_pbrs)
        self.eval_use_pbrs = bool(eval_use_pbrs)
        self._native_pbrs_version = str(pbrs_version or "").strip()
        self.use_dense_reward = use_dense_reward
        self.log_dense_reward_details = log_dense_reward_details
        self._episode_step = 0
        self._training_context = {
            "t_env": 0,
            "episode_idx": 0,
            "test_mode": False,
        }
        if self.use_pbrs and self.use_dense_reward:
            raise ValueError(
                "use_pbrs and use_dense_reward cannot both be enabled; "
                "native original PBRS should bypass runtime dense reward modules."
            )
        self._base_pbrs_beta = float(pbrs_beta)
        self._pbrs_beta_schedule_enabled = bool(pbrs_beta_schedule_enabled)
        self._pbrs_beta_schedule_first = float(pbrs_beta_schedule_first)
        self._pbrs_beta_schedule_second = float(pbrs_beta_schedule_second)
        self._pbrs_beta_schedule_switch_ratio = float(
            np.clip(pbrs_beta_schedule_switch_ratio, 0.0, 1.0)
        )
        self._pbrs_stage_schedule_enabled = bool(pbrs_stage_schedule_enabled)
        self._pbrs_stage_boundary_1 = int(max(0, pbrs_stage_boundary_1))
        self._pbrs_stage_boundary_2 = int(max(self._pbrs_stage_boundary_1, pbrs_stage_boundary_2))
        self._pbrs_stage_values = {
            "early_exploration": {
                "beta": float(pbrs_stage_beta_early),
                "wc": float(pbrs_stage_wc_early),
                "wp": float(pbrs_stage_wp_early),
            },
            "mid_progress": {
                "beta": float(pbrs_stage_beta_mid),
                "wc": float(pbrs_stage_wc_mid),
                "wp": float(pbrs_stage_wp_mid),
            },
            "late_plateau": {
                "beta": float(pbrs_stage_beta_late),
                "wc": float(pbrs_stage_wc_late),
                "wp": float(pbrs_stage_wp_late),
            },
        }
        self._current_global_t_env = 0
        self._current_t_max = None
        self._pbrs_episode_stats = {}
        self._current_pbrs_stage_label = "early_exploration"
        self._cached_lbf_state = None
        self._cached_lbf_phi = None
        self._cached_pbrs_signature = None
        self._cached_lbf_terms = None
        self.pbrs = None
        if self.use_pbrs:
            if not str(self.key).startswith("lbforaging:"):
                raise NotImplementedError(
                    "Native original PBRS is currently implemented only for LBF environments"
                )
            if self._native_pbrs_version == "lbf_pbrs_v2":
                parsed_active_terms = pbrs_active_terms
                if isinstance(parsed_active_terms, str):
                    try:
                        parsed_active_terms = json.loads(parsed_active_terms)
                    except Exception:
                        parsed_active_terms = []
                parsed_weights = pbrs_weights
                if isinstance(parsed_weights, str):
                    try:
                        parsed_weights = json.loads(parsed_weights)
                    except Exception:
                        parsed_weights = {}
                self.pbrs = LBFPBRSV2Runtime(
                    {
                        "pbrs_version": "lbf_pbrs_v2",
                        "mode": pbrs_mode,
                        "beta": pbrs_beta,
                        "gamma": pbrs_gamma,
                        "active_terms": list(parsed_active_terms or []),
                        "weights": dict(parsed_weights or {}),
                    }
                )
            else:
                self.pbrs = build_lbf_pbrs(
                    variant=pbrs_variant,
                    beta=pbrs_beta,
                    gamma=pbrs_gamma,
                    wc=pbrs_wc,
                    wp=pbrs_wp,
                    semi_strict_gate_radius=pbrs_semi_strict_gate_radius,
                )
            self._reset_pbrs_episode_stats()
            self._update_pbrs_parameters()
        self.reward_runtime = None
        if self.use_dense_reward:
            self.reward_runtime = RewardRuntimeWrapper(
                reward_module_path=reward_module_path,
                alpha_policy_config=alpha_policy_config,
                policy_guidance_config=policy_guidance_config,
                workflow_id=workflow_id,
                round_id=workflow_round,
                apply_dense_reward_in_eval=apply_dense_reward_in_eval,
            )
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )

    def _pad_observation(self, obs):
        return [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def _scalarise_reward_for_logging(self, reward):
        if isinstance(reward, np.ndarray):
            if reward.ndim == 0:
                return float(reward.item())
            reward = reward.tolist()
        if isinstance(reward, Iterable) and not isinstance(reward, (str, bytes, dict)):
            reward_values = [float(r) for r in reward]
            if self.common_reward:
                return float(self.reward_agg_fn(reward_values))
            return float(sum(reward_values))
        return float(reward)

    def _reset_pbrs_episode_stats(self):
        self._pbrs_episode_stats = {
            "reward_env": 0.0,
            "reward_shape": 0.0,
            "reward_total": 0.0,
            "phi_t": 0.0,
            "phi_tp1": 0.0,
            "pbrs_steps": 0,
        }

    def _apply_common_reward_if_needed(self, reward):
        if self.common_reward and isinstance(reward, Iterable) and not isinstance(
            reward, (str, bytes)
        ):
            return float(self.reward_agg_fn(reward))
        if not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )
        return reward

    def _combine_reward_with_shaping(self, env_reward, reward_shape):
        env_reward_for_algo = self._apply_common_reward_if_needed(env_reward)
        if self.common_reward:
            reward_total = float(env_reward_for_algo) + float(reward_shape)
        else:
            reward_total = [float(reward) + float(reward_shape) for reward in env_reward]
        return env_reward_for_algo, reward_total

    def _pbrs_active(self):
        if self.pbrs is None:
            return False
        return not self._training_context["test_mode"] or self.eval_use_pbrs

    def _current_pbrs_stage(self):
        if not self._pbrs_stage_schedule_enabled:
            return "single_stage"
        if self._current_global_t_env < self._pbrs_stage_boundary_1:
            return "early_exploration"
        if self._current_global_t_env < self._pbrs_stage_boundary_2:
            return "mid_progress"
        return "late_plateau"

    def _scheduled_pbrs_beta(self):
        if not self._pbrs_beta_schedule_enabled:
            return self._base_pbrs_beta
        if self._current_t_max is None or self._current_t_max <= 0:
            return self._pbrs_beta_schedule_first
        switch_t_env = self._pbrs_beta_schedule_switch_ratio * float(self._current_t_max)
        if float(self._current_global_t_env) < switch_t_env:
            return self._pbrs_beta_schedule_first
        return self._pbrs_beta_schedule_second

    def _scheduled_pbrs_parameters(self):
        stage_label = self._current_pbrs_stage()
        if self._pbrs_stage_schedule_enabled:
            values = dict(self._pbrs_stage_values[stage_label])
            self._current_pbrs_stage_label = stage_label
            return values
        self._current_pbrs_stage_label = stage_label
        if self._native_pbrs_version == "lbf_pbrs_v2":
            config = getattr(self.pbrs, "config", {}) if self.pbrs is not None else {}
            return {
                "beta": float(self._scheduled_pbrs_beta()),
                "wc": float(config.get("wc", 0.5)),
                "wp": float(config.get("wp", 0.5)),
                "mode": str(config.get("mode", "")),
                "active_terms": list(config.get("active_terms") or []),
                "weights": deepcopy(config.get("weights") or {}),
            }
        return {
            "beta": float(self._scheduled_pbrs_beta()),
            "wc": float(self.pbrs.wc if self.pbrs is not None else 0.6),
            "wp": float(self.pbrs.wp if self.pbrs is not None else 0.4),
        }

    def _update_pbrs_parameters(self):
        if self.pbrs is None:
            return
        values = self._scheduled_pbrs_parameters()
        if self._native_pbrs_version == "lbf_pbrs_v2":
            config = normalize_lbf_pbrs_v2_config(
                {
                    "pbrs_version": "lbf_pbrs_v2",
                    "mode": values.get("mode", getattr(self.pbrs, "mode", "")),
                    "beta": float(values["beta"]),
                    "gamma": float(getattr(self.pbrs, "gamma", 0.99)),
                    "active_terms": list(values.get("active_terms") or []),
                    "weights": deepcopy(values.get("weights") or {}),
                }
            )
            self.pbrs.apply_config(config)
        else:
            self.pbrs.beta = float(values["beta"])
            self.pbrs.wc = float(values["wc"])
            self.pbrs.wp = float(values["wp"])

    def _update_pbrs_episode_stats(
        self, reward_env, reward_shape, reward_total, phi_t, phi_tp1
    ):
        self._pbrs_episode_stats["reward_env"] += self._scalarise_reward_for_logging(
            reward_env
        )
        self._pbrs_episode_stats["reward_shape"] += float(reward_shape)
        self._pbrs_episode_stats["reward_total"] += self._scalarise_reward_for_logging(
            reward_total
        )
        self._pbrs_episode_stats["phi_t"] += float(phi_t)
        self._pbrs_episode_stats["phi_tp1"] += float(phi_tp1)
        self._pbrs_episode_stats["pbrs_steps"] += 1

    def _build_pbrs_info(self):
        steps = max(self._pbrs_episode_stats["pbrs_steps"], 1)
        return {
            "reward_env": self._pbrs_episode_stats["reward_env"],
            "reward_shape": self._pbrs_episode_stats["reward_shape"],
            "reward_total": self._pbrs_episode_stats["reward_total"],
            "phi_t": self._pbrs_episode_stats["phi_t"] / steps,
            "phi_tp1": self._pbrs_episode_stats["phi_tp1"] / steps,
            "pbrs_beta": 0.0 if self.pbrs is None else float(self.pbrs.beta),
            "pbrs_wc": 0.0 if self.pbrs is None else float(getattr(self.pbrs, "wc", getattr(self.pbrs, "config", {}).get("wc", 0.0))),
            "pbrs_wp": 0.0 if self.pbrs is None else float(getattr(self.pbrs, "wp", getattr(self.pbrs, "config", {}).get("wp", 0.0))),
            "pbrs_stage_label": self._current_pbrs_stage_label,
            "pbrs_v2_runtime_used": 1.0 if self._native_pbrs_version == "lbf_pbrs_v2" else 0.0,
        }

    def _current_pbrs_signature(self):
        if self.pbrs is None:
            return None
        if self._native_pbrs_version == "lbf_pbrs_v2":
            weights = getattr(self.pbrs, "weights", {})
            return (
                "lbf_pbrs_v2",
                float(self.pbrs.beta),
                str(getattr(self.pbrs, "mode", "")),
                tuple((term, round(float(weights.get(term, 0.0)), 6)) for term in sorted(weights)),
                self._current_pbrs_stage_label,
            )
        return (
            float(self.pbrs.beta),
            float(self.pbrs.wc),
            float(self.pbrs.wp),
            self._current_pbrs_stage_label,
        )

    def _invalidate_pbrs_cache(self):
        self._cached_lbf_state = None
        self._cached_lbf_phi = None
        self._cached_pbrs_signature = None
        self._cached_lbf_terms = None

    def _get_pbrs_prev_state_and_phi(self):
        if self.pbrs is None:
            return None, None
        signature = self._current_pbrs_signature()
        state = self._cached_lbf_state
        if state is None:
            state = self._capture_lbf_state()
            if state is None:
                return None, None
        if self._cached_lbf_phi is not None and self._cached_pbrs_signature == signature:
            phi = float(self._cached_lbf_phi)
        else:
            phi = float(self.pbrs.compute_lbf_phi(state))
        if self._native_pbrs_version == "lbf_pbrs_v2":
            self._cached_lbf_terms = deepcopy(self.pbrs.compute_terms(state))
        self._cached_lbf_state = state
        self._cached_lbf_phi = phi
        self._cached_pbrs_signature = signature
        return state, phi

    def _update_pbrs_cache_with_next_state(self, state):
        if self.pbrs is None or state is None:
            self._invalidate_pbrs_cache()
            return None
        signature = self._current_pbrs_signature()
        phi = float(self.pbrs.compute_lbf_phi(state))
        if self._native_pbrs_version == "lbf_pbrs_v2":
            self._cached_lbf_terms = deepcopy(self.pbrs.compute_terms(state))
        self._cached_lbf_state = state
        self._cached_lbf_phi = phi
        self._cached_pbrs_signature = signature
        return phi

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        obs_before = self._obs
        state_before = (
            self.get_state() if (self.reward_runtime is not None and self._obs is not None) else None
        )
        lbf_state_before = None
        phi_t = None
        if self.pbrs is not None:
            lbf_state_before, phi_t = self._get_pbrs_prev_state_and_phi()
        actions = [int(a) for a in actions]
        obs, reward, done, truncated, self._info = self._env.step(actions)
        self._obs = self._pad_observation(obs)
        state_after = self.get_state() if self.reward_runtime is not None else None
        lbf_state_after = self._capture_lbf_state() if self.pbrs is not None else None
        sparse_reward = reward

        if (
            self.pbrs is not None
            and lbf_state_before is not None
            and lbf_state_after is not None
            and phi_t is not None
        ):
            phi_tp1 = self._update_pbrs_cache_with_next_state(lbf_state_after)
            if self._pbrs_active():
                reward_shape = self.pbrs.beta * (self.pbrs.gamma * phi_tp1 - phi_t)
                reward_env, reward_total = self._combine_reward_with_shaping(
                    sparse_reward, reward_shape
                )
            else:
                reward_shape = 0.0
                reward_env = self._apply_common_reward_if_needed(sparse_reward)
                reward_total = reward_env
            reward = reward_total
            self._update_pbrs_episode_stats(
                reward_env=reward_env,
                reward_shape=reward_shape,
                reward_total=reward_total,
                phi_t=phi_t,
                phi_tp1=phi_tp1,
            )
            self._info = dict(self._info or {})
            self._info.update(self._build_pbrs_info())
            if self._native_pbrs_version == "lbf_pbrs_v2":
                terms_tp1 = deepcopy(self._cached_lbf_terms or {})
                self._info.update(
                    self.pbrs.build_info(
                        phi_t=float(phi_t),
                        phi_tp1=float(phi_tp1),
                        reward_shape=float(reward_shape),
                        reward_env=self._scalarise_reward_for_logging(reward_env),
                        reward_total=self._scalarise_reward_for_logging(reward_total),
                        terms_tp1=terms_tp1,
                    )
                )
            self._info["sparse_reward_raw"] = sparse_reward
            self._info["sparse_reward_scalar"] = self._scalarise_reward_for_logging(
                reward_env
            )
            self._info["dense_reward_scalar"] = float(reward_shape)
            self._info["mixed_reward_scalar"] = self._scalarise_reward_for_logging(
                reward_total
            )

        if self.reward_runtime is not None:
            self.reward_runtime.set_training_context(
                t_env=self._training_context["t_env"],
                episode_idx=self._training_context["episode_idx"],
                test_mode=self._training_context["test_mode"],
            )
            reward_result = self.reward_runtime.process_transition(
                raw_env=self._env,
                obs_before=obs_before,
                obs_after=self._obs,
                state_before=state_before,
                state_after=state_after,
                actions=actions,
                sparse_reward=sparse_reward,
                done=done,
                truncated=truncated,
                info=self._info,
                episode_step=self._episode_step,
                lbf_state_before=lbf_state_before,
                lbf_state_after=lbf_state_after,
            )
            reward = reward_result.mixed_reward
            if self.log_dense_reward_details:
                self._info = dict(self._info or {})
                self._info["sparse_reward_raw"] = reward_result.sparse_reward
                self._info["dense_reward"] = reward_result.dense_reward
                self._info["mixed_reward"] = reward_result.mixed_reward
                self._info["sparse_reward_scalar"] = self._scalarise_reward_for_logging(
                    reward_result.sparse_reward
                )
                self._info["dense_reward_scalar"] = self._scalarise_reward_for_logging(
                    reward_result.dense_reward
                )
                self._info["mixed_reward_scalar"] = self._scalarise_reward_for_logging(
                    reward_result.mixed_reward
                )
                self._info["reward_alpha"] = reward_result.alpha
                self._info["dense_reward_applied"] = (
                    reward_result.dense_reward_applied
                )
                self._info["reward_breakdown"] = reward_result.reward_breakdown
                self._info["reward_runtime_metadata"] = (
                    reward_result.runtime_metadata
                )
                self._info["reward_runtime_errors"] = reward_result.errors

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )

        if isinstance(done, Iterable):
            done = all(done)
        self._episode_step += 1
        return self._obs, reward, done, truncated, self._info

    def _capture_lbf_state(self):
        if not str(self.key).startswith("lbforaging:"):
            return None
        try:
            return extract_lbf_state(self._env)
        except Exception:
            return None

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        raise self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the shape of the state"""
        if hasattr(self._env.unwrapped, "state_size"):
            return self._env.unwrapped.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs, info = self._env.reset(seed=seed, options=options)
        self._obs = self._pad_observation(obs)
        self._episode_step = 0
        self._invalidate_pbrs_cache()
        if self.pbrs is not None:
            self._reset_pbrs_episode_stats()
            initial_state = self._capture_lbf_state()
            if initial_state is not None:
                self.pbrs.reset(initial_state)
                self._update_pbrs_cache_with_next_state(initial_state)
        return self._obs, info

    def set_training_context(self, t_env, episode_idx, test_mode):
        self._training_context = {
            "t_env": int(t_env),
            "episode_idx": int(episode_idx),
            "test_mode": bool(test_mode),
        }
        self._current_global_t_env = self._training_context["t_env"]
        self._update_pbrs_parameters()
        if self.reward_runtime is not None:
            self.reward_runtime.set_training_context(
                t_env=self._training_context["t_env"],
                episode_idx=self._training_context["episode_idx"],
                test_mode=self._training_context["test_mode"],
            )

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self._env.unwrapped.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
