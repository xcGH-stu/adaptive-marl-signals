from functools import partial
from multiprocessing import Pipe, Process
import json

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2


def _coerce_numeric_info_scalar(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            try:
                return value.item()
            except ValueError:
                return None
        return None
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        return value
    return None


def _safe_sum_numeric_info_values(infos, prefix=""):
    aggregated = {}
    skipped = {}
    for key, value in (infos or {}).items():
        metric_key = f"{prefix}__{key}" if prefix else key
        scalar = _coerce_numeric_info_scalar(value)
        if scalar is not None:
            aggregated[metric_key] = float(scalar)
            continue
        if isinstance(value, dict):
            nested_aggregated, nested_skipped = _safe_sum_numeric_info_values(
                value,
                prefix=metric_key,
            )
            aggregated.update(nested_aggregated)
            skipped.update(nested_skipped)
            continue
        skipped.setdefault(metric_key, set()).add(type(value).__name__)
    return aggregated, skipped


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i
            env_args[i]["common_reward"] = self.args.common_reward
            env_args[i]["reward_scalarisation"] = self.args.reward_scalarisation
            if self.args.env == "gymma":
                alpha_policy_config = getattr(self.args, "alpha_policy_config", None)
                if isinstance(alpha_policy_config, str):
                    alpha_policy_config = json.loads(alpha_policy_config)
                policy_guidance_config = getattr(self.args, "policy_guidance_config", None)
                if isinstance(policy_guidance_config, str):
                    policy_guidance_config = json.loads(policy_guidance_config)
                env_args[i]["use_dense_reward"] = getattr(self.args, "use_dense_reward", False)
                env_args[i]["reward_module_path"] = getattr(self.args, "reward_module_path", None)
                env_args[i]["alpha_policy_config"] = alpha_policy_config
                env_args[i]["policy_guidance_config"] = policy_guidance_config
                env_args[i]["workflow_id"] = getattr(self.args, "workflow_id", None)
                env_args[i]["workflow_round"] = getattr(self.args, "workflow_round", None)
                env_args[i]["apply_dense_reward_in_eval"] = getattr(self.args, "apply_dense_reward_in_eval", False)
                env_args[i]["log_dense_reward_details"] = getattr(self.args, "log_dense_reward_details", True)
        self.ps = [
            Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
            )
            for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

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
        self._info_aggregation_debug_emitted = False
        self._last_skipped_info_key_types = {}

        self.log_train_stats_t = -100000

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
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        self.parent_conns[0].send(("save_replay", None))

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self._sync_env_training_context(test_mode=test_mode)
        self.reset()

        all_terminated = False
        if self.args.common_reward:
            episode_returns = [0 for _ in range(self.batch_size)]
        else:
            episode_returns = [
                np.zeros(self.args.n_agents) for _ in range(self.batch_size)
            ]
        episode_sparse_returns = [0.0 for _ in range(self.batch_size)]
        episode_dense_returns = [0.0 for _ in range(self.batch_size)]
        episode_mixed_returns = [0.0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = []

        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                bs=envs_not_terminated,
                test_mode=test_mode,
            )
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {"actions": actions.unsqueeze(1)}
            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[
                        idx
                    ]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env
                    if idx == 0 and test_mode and self.args.render:
                        parent_conn.send(("render", None))

            # Update envs_not_terminated
            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {"reward": [], "terminated": []}
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    if self._reward_diagnostics_enabled:
                        env_info = data.get("info") or {}
                        episode_sparse_returns[idx] += float(env_info.get("sparse_reward_scalar", data["reward"]))
                        episode_dense_returns[idx] += float(env_info.get("dense_reward_scalar", 0.0))
                        episode_mixed_returns[idx] += float(env_info.get("mixed_reward_scalar", data["reward"]))
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        numeric_env_infos = []
        skipped_info_key_types = {}
        for item in final_env_infos:
            numeric_info, skipped_info = _safe_sum_numeric_info_values(item)
            numeric_env_infos.append(numeric_info)
            for key, type_names in skipped_info.items():
                skipped_info_key_types.setdefault(key, set()).update(type_names)
        infos = [cur_stats] + numeric_env_infos
        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos])
            }
        )
        self._last_skipped_info_key_types = {
            key: sorted(type_names) for key, type_names in skipped_info_key_types.items()
        }
        if self._last_skipped_info_key_types and not self._info_aggregation_debug_emitted:
            console_logger = getattr(self.logger, "console_logger", None)
            if console_logger is not None:
                preview_items = []
                for key in sorted(self._last_skipped_info_key_types.keys())[:12]:
                    preview_items.append(
                        f"{key}:{'/'.join(self._last_skipped_info_key_types[key])}"
                    )
                console_logger.info(
                    "ParallelRunner skipped non-numeric env info fields during aggregation: %s",
                    ", ".join(preview_items),
                )
            self._info_aggregation_debug_emitted = True
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        if self._reward_diagnostics_enabled:
            cur_stats["sparse_return"] = sum(episode_sparse_returns) + cur_stats.get("sparse_return", 0)
            cur_stats["dense_return"] = sum(episode_dense_returns) + cur_stats.get("dense_return", 0)
            cur_stats["mixed_return"] = sum(episode_mixed_returns) + cur_stats.get("mixed_return", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = (
            max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        )
        if test_mode and (len(self.test_returns) == n_test_runs):
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
        for idx, parent_conn in enumerate(self.parent_conns):
            parent_conn.send(
                (
                    "set_training_context",
                    {
                        "t_env": int(self.t_env),
                        "episode_idx": int(self.t_env + idx),
                        "test_mode": bool(test_mode),
                    },
                )
            )
        for parent_conn in self.parent_conns:
            parent_conn.recv()

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


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            _, reward, terminated, truncated, env_info = env.step(actions)
            terminated = terminated or truncated
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send(
                {
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                }
            )
        elif cmd == "reset":
            env.reset()
            remote.send(
                {
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs(),
                }
            )
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "set_training_context":
            if hasattr(env, "set_training_context"):
                env.set_training_context(**(data or {}))
            remote.send({"ok": True})
        elif cmd == "render":
            env.render()
        elif cmd == "save_replay":
            env.save_replay()
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)
