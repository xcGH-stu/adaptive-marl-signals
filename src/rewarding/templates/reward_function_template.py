"""
Standard dense reward template for LLM-generated reward shaping modules.

This file defines the interface expected by RewardRuntimeWrapper.

Required entrypoint:
    compute_dense_reward(context)

Expected input:
    context is a RuntimeContext instance created by
    src/rewarding/runtime_wrapper.py.

Expected output:
    Either:
        dense_reward
    Or:
        (dense_reward, breakdown_dict)

Notes:
    - This module should only compute dense shaping reward.
    - Do not re-implement the environment's sparse task reward here.
    - Do not apply alpha mixing here. That is handled by the runtime wrapper.
    - The returned dense reward should be shape-compatible with
      context.sparse_reward.
"""


def compute_dense_reward(context):
    """
    Compute a dense shaping reward for the current transition.

    The example below is intentionally conservative. It demonstrates how a
    reward function can use LBF environment state without overriding the sparse
    task objective.
    """
    sparse_reward = context.sparse_reward
    raw_env = context.raw_env.unwrapped if hasattr(context.raw_env, "unwrapped") else context.raw_env

    dense_reward = _zero_like(sparse_reward)
    breakdown = {
        "approach_food_bonus": 0.0,
        "idle_penalty": 0.0,
        "episode_completion_bonus": 0.0,
    }

    players = getattr(raw_env, "players", [])
    field = getattr(raw_env, "field", None)

    if not players or field is None:
        return dense_reward, breakdown

    previous_food_distances = _nearest_food_distances(context.obs_before, players)
    current_food_distances = _nearest_food_distances(context.obs_after, players)

    per_agent_dense = []
    for agent_idx, player in enumerate(players):
        agent_reward = 0.0

        previous_distance = previous_food_distances[agent_idx]
        current_distance = current_food_distances[agent_idx]
        if previous_distance is not None and current_distance is not None:
            if current_distance < previous_distance:
                agent_reward += 0.05
                breakdown["approach_food_bonus"] += 0.05

        if agent_idx < len(context.actions):
            if int(context.actions[agent_idx]) == 0:
                agent_reward -= 0.01
                breakdown["idle_penalty"] -= 0.01

        if context.done and _has_collected_all_food(raw_env):
            agent_reward += 0.1
            breakdown["episode_completion_bonus"] += 0.1

        per_agent_dense.append(agent_reward)

    dense_reward = _match_sparse_reward_shape(per_agent_dense, sparse_reward)
    return dense_reward, breakdown


def _nearest_food_distances(obs_after, players):
    """
    Compute each agent's Manhattan distance to the nearest visible food.

    This helper intentionally uses the padded observation representation coming
    from GymmaWrapper, so Generator-produced reward functions can rely on a
    stable interface instead of environment internals only.
    """
    if obs_after is None:
        return [None for _ in players]

    distances = []
    for agent_idx, player in enumerate(players):
        if agent_idx >= len(obs_after):
            distances.append(None)
            continue
        food_positions = _extract_food_positions_from_obs(obs_after[agent_idx])
        if not food_positions or player.position is None:
            distances.append(None)
            continue
        row, col = player.position
        distances.append(
            min(abs(row - food_row) + abs(col - food_col) for food_row, food_col in food_positions)
        )
    return distances


def _extract_food_positions_from_obs(agent_obs):
    """
    Parse the flattened LBF observation layout used by EPyMARL's gymma wrapper.

    For LBF vector observations, foods are encoded first as repeated triples:
        [food_y, food_x, food_level, ...]

    Empty food slots are represented by (-1, -1, 0).
    """
    food_positions = []
    if agent_obs is None:
        return food_positions

    obs_len = len(agent_obs)
    for index in range(0, obs_len, 3):
        if index + 2 >= obs_len:
            break
        y = agent_obs[index]
        x = agent_obs[index + 1]
        level = agent_obs[index + 2]
        if y < 0 or x < 0:
            continue
        if level <= 0:
            continue
        food_positions.append((int(y), int(x)))
    return food_positions


def _has_collected_all_food(raw_env):
    field = getattr(raw_env, "field", None)
    if field is None:
        return False
    return field.sum() == 0


def _match_sparse_reward_shape(per_agent_dense, sparse_reward):
    if isinstance(sparse_reward, tuple):
        return tuple(float(value) for value in per_agent_dense)
    if isinstance(sparse_reward, list):
        return [float(value) for value in per_agent_dense]
    if not per_agent_dense:
        return 0.0
    return float(sum(per_agent_dense))


def _zero_like(reward):
    if isinstance(reward, tuple):
        return tuple(0.0 for _ in reward)
    if isinstance(reward, list):
        return [0.0 for _ in reward]
    return 0.0
