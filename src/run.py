import datetime
import os
from os.path import dirname, abspath
import pprint
import random
import shutil
import time
import threading
from types import SimpleNamespace as SN

import torch as th
import numpy as np

from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import Logger
from utils.timehelper import time_left, time_str


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    assert test_alg_config_supports_reward(
        args
    ), "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = (
        f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"
    )

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    if args.use_wandb:
        logger.setup_wandb(
            _config, args.wandb_team, args.wandb_project, args.wandb_mode
        )

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger, sacred_run=_run)

    # Finish logging
    logger.finish()

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger, sacred_run=None):
    if sacred_run is not None:
        model_root_path = os.path.join(args.local_results_path, "models", args.unique_token)
        sacred_run.info["unique_token"] = args.unique_token
        sacred_run.info["model_root_path"] = model_root_path
        sacred_run.info.setdefault("saved_model_steps", [])

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    # For individual rewards in gymmai reward is of shape (1, n_agents)
    if args.common_reward:
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (args.n_agents,)}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    restored_training_state = None

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load
        if sacred_run is not None:
            sacred_run.info["loaded_checkpoint_path"] = model_path
            sacred_run.info["loaded_checkpoint_step"] = timestep_to_load

        training_state_path = os.path.join(model_path, "training_state.th")
        if os.path.exists(training_state_path):
            logger.console_logger.info(
                "Loading training state from {}".format(training_state_path)
            )
            restored_training_state = th.load(
                training_state_path,
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
            buffer_state = restored_training_state.get("buffer_state")
            if buffer_state is not None:
                buffer.load_state_dict(buffer_state)
            learner_state = restored_training_state.get("learner_state")
            if learner_state is not None and hasattr(learner, "load_checkpoint_state"):
                learner.load_checkpoint_state(learner_state)

            runner_state = restored_training_state.get("runner_state", {})
            runner.t_env = int(runner_state.get("t_env", runner.t_env))
            if hasattr(runner, "episode_count"):
                runner.episode_count = int(
                    runner_state.get(
                        "episode_count", getattr(runner, "episode_count", 0)
                    )
                )
            if hasattr(runner, "log_train_stats_t"):
                runner.log_train_stats_t = int(
                    runner_state.get(
                        "log_train_stats_t",
                        getattr(runner, "log_train_stats_t", runner.t_env),
                    )
                )
            if hasattr(runner, "train_returns"):
                runner.train_returns = list(
                    runner_state.get("train_returns", getattr(runner, "train_returns", []))
                )
            if hasattr(runner, "test_returns"):
                runner.test_returns = list(
                    runner_state.get("test_returns", getattr(runner, "test_returns", []))
                )
            if hasattr(runner, "train_stats"):
                runner.train_stats = dict(
                    runner_state.get("train_stats", getattr(runner, "train_stats", {}))
                )
            if hasattr(runner, "test_stats"):
                runner.test_stats = dict(
                    runner_state.get("test_stats", getattr(runner, "test_stats", {}))
                )

            rng_state = restored_training_state.get("rng_state", {})
            if rng_state.get("python") is not None:
                random.setstate(rng_state["python"])
            if rng_state.get("numpy") is not None:
                np.random.set_state(rng_state["numpy"])
            if rng_state.get("torch") is not None:
                th.set_rng_state(rng_state["torch"])
            if th.cuda.is_available() and rng_state.get("torch_cuda") is not None:
                th.cuda.set_rng_state_all(rng_state["torch_cuda"])

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    if restored_training_state is not None:
        loop_state = restored_training_state.get("loop_state", {})
        episode = int(loop_state.get("episode", episode))
        last_test_T = int(loop_state.get("last_test_T", last_test_T))
        last_log_T = int(loop_state.get("last_log_T", last_log_T))
        model_save_time = int(loop_state.get("model_save_time", model_save_time))

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    def _save_training_checkpoint(save_step: int) -> str:
        save_path = os.path.join(
            args.local_results_path, "models", args.unique_token, str(save_step)
        )
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))
        learner.save_models(save_path)
        training_state = {
            "runner_state": {
                "t_env": runner.t_env,
                "episode_count": getattr(runner, "episode_count", 0),
                "log_train_stats_t": getattr(runner, "log_train_stats_t", 0),
                "train_returns": getattr(runner, "train_returns", []),
                "test_returns": getattr(runner, "test_returns", []),
                "train_stats": getattr(runner, "train_stats", {}),
                "test_stats": getattr(runner, "test_stats", {}),
            },
            "loop_state": {
                "episode": episode,
                "last_test_T": last_test_T,
                "last_log_T": last_log_T,
                "model_save_time": save_step,
            },
            "buffer_state": buffer.state_dict(),
            "learner_state": (
                learner.save_checkpoint_state()
                if hasattr(learner, "save_checkpoint_state")
                else None
            ),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": th.get_rng_state(),
                "torch_cuda": th.cuda.get_rng_state_all()
                if th.cuda.is_available()
                else None,
            },
        }
        th.save(training_state, os.path.join(save_path, "training_state.th"))
        if sacred_run is not None:
            saved_steps = list(sacred_run.info.get("saved_model_steps", []))
            if save_step not in saved_steps:
                saved_steps.append(save_step)
                saved_steps.sort()
            sacred_run.info["saved_model_steps"] = saved_steps
            sacred_run.info["latest_model_path"] = save_path

        if args.use_wandb and args.wandb_save_model:
            wandb_save_dir = os.path.join(
                logger.wandb.dir, "models", args.unique_token, str(save_step)
            )
            os.makedirs(wandb_save_dir, exist_ok=True)
            for f in os.listdir(save_path):
                shutil.copyfile(
                    os.path.join(save_path, f), os.path.join(wandb_save_dir, f)
                )
        return save_path

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or (
                model_save_time == 0
                and not bool(getattr(args, "defer_initial_model_save", False))
            )
        ):
            model_save_time = runner.t_env
            _save_training_checkpoint(model_save_time)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    if args.save_model and getattr(args, "save_final_model", False) and runner.t_env != model_save_time:
        model_save_time = runner.t_env
        _save_training_checkpoint(model_save_time)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
