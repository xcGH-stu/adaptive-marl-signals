from functools import partial
import json

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        self.episode_count = 0

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        env_kwargs = dict(self.args.env_args)
        env_kwargs["common_reward"] = self.args.common_reward
        env_kwargs["reward_scalarisation"] = self.args.reward_scalarisation
        if self.args.env == "gymma":
            alpha_policy_config = getattr(self.args, "alpha_policy_config", None)
            if isinstance(alpha_policy_config, str):
                alpha_policy_config = json.loads(alpha_policy_config)
            policy_guidance_config = getattr(self.args, "policy_guidance_config", None)
            if isinstance(policy_guidance_config, str):
                policy_guidance_config = json.loads(policy_guidance_config)
            env_kwargs["use_dense_reward"] = getattr(
                self.args, "use_dense_reward", False
            )
            env_kwargs["reward_module_path"] = getattr(
                self.args, "reward_module_path", None
            )
            env_kwargs["alpha_policy_config"] = alpha_policy_config
            env_kwargs["policy_guidance_config"] = policy_guidance_config
            env_kwargs["workflow_id"] = getattr(self.args, "workflow_id", None)
            env_kwargs["workflow_round"] = getattr(self.args, "workflow_round", None)
            env_kwargs["apply_dense_reward_in_eval"] = getattr(
                self.args, "apply_dense_reward_in_eval", False
            )
            env_kwargs["log_dense_reward_details"] = getattr(
                self.args, "log_dense_reward_details", True
            )

        self.env = env_REGISTRY[self.args.env](**env_kwargs)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self._reward_diagnostics_enabled = bool(
            getattr(self.args, "use_dense_reward", False)
            or getattr(self.args, "env_args", {}).get("use_pbrs", False)
        )

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self._sync_env_training_context(test_mode=test_mode)
        self.reset()

        terminated = False
        if self.args.common_reward:
            episode_return = 0.0
        else:
            episode_return = np.zeros(self.args.n_agents)
        episode_sparse_return = 0.0
        episode_dense_return = 0.0
        episode_mixed_return = 0.0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward
            if self._reward_diagnostics_enabled:
                episode_sparse_return += float(
                    env_info.get("sparse_reward_scalar", reward)
                )
                episode_dense_return += float(
                    env_info.get("dense_reward_scalar", 0.0)
                )
                episode_mixed_return += float(env_info.get("mixed_reward_scalar", reward))

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        numeric_env_info = self._extract_numeric_env_info(env_info)
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + numeric_env_info.get(k, 0)
                for k in set(cur_stats) | set(numeric_env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        if self._reward_diagnostics_enabled:
            cur_stats["sparse_return"] = episode_sparse_return + cur_stats.get(
                "sparse_return", 0
            )
            cur_stats["dense_return"] = episode_dense_return + cur_stats.get(
                "dense_return", 0
            )
            cur_stats["mixed_return"] = episode_mixed_return + cur_stats.get(
                "mixed_return", 0
            )

        if not test_mode:
            self.t_env += self.t
            self.episode_count += self.batch_size
        else:
            self.episode_count += self.batch_size

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _sync_env_training_context(self, test_mode=False):
        if hasattr(self.env, "set_training_context"):
            self.env.set_training_context(
                t_env=self.t_env,
                episode_idx=self.episode_count,
                test_mode=test_mode,
            )

    def _extract_numeric_env_info(self, env_info, prefix=""):
        numeric_env_info = {}
        for key, value in (env_info or {}).items():
            metric_key = f"{prefix}__{key}" if prefix else key
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_env_info[metric_key] = float(value)
            elif isinstance(value, dict):
                numeric_env_info.update(
                    self._extract_numeric_env_info(value, prefix=metric_key)
                )
        return numeric_env_info

    def _log(self, returns, stats, prefix):
        if self.args.common_reward:
            self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
            self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_mean",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_std",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                prefix + "total_return_mean", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                prefix + "total_return_std", total_returns.std(), self.t_env
            )
        if "sparse_return" in stats:
            self.logger.log_stat(
                prefix + "sparse_return_mean",
                stats["sparse_return"] / stats["n_episodes"],
                self.t_env,
            )
        if "dense_return" in stats:
            self.logger.log_stat(
                prefix + "dense_return_mean",
                stats["dense_return"] / stats["n_episodes"],
                self.t_env,
            )
        if "mixed_return" in stats:
            self.logger.log_stat(
                prefix + "mixed_return_mean",
                stats["mixed_return"] / stats["n_episodes"],
                self.t_env,
            )
        returns.clear()

        for k, v in stats.items():
            if k not in {"n_episodes", "sparse_return", "dense_return", "mixed_return"}:
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()
