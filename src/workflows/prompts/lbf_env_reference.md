# LBF Environment Reference

This document describes the Level-Based Foraging (LBF) environment as used in this repository, with emphasis on the information available to dense reward generation.

## Environment Scope

- Underlying environment source:
  [`/home/epymarl/code/lbforaging_local/lbforaging/foraging/environment.py`](/home/epymarl/code/lbforaging_local/lbforaging/foraging/environment.py)
- EPyMARL wrapper used during training:
  [`/home/epymarl/code/epymarl/src/envs/gymma.py`](/home/epymarl/code/epymarl/src/envs/gymma.py)

Dense reward generation should treat the original LBF reward as the sparse task reward and only add shaping reward on top of it.

## Action Space

The LBF action enum defines 6 discrete actions:

- `0`: `NONE`
- `1`: `NORTH`
- `2`: `SOUTH`
- `3`: `WEST`
- `4`: `EAST`
- `5`: `LOAD`

Operational meaning:

- `NONE`: stay in place
- `NORTH/SOUTH/WEST/EAST`: move by one grid cell if the move is valid
- `LOAD`: attempt to load adjacent food

Important constraints:

- Invalid actions are replaced by `NONE`
- Movement conflicts can cause agents to fail to move
- `LOAD` only succeeds when enough adjacent agent levels sum to the target food level

## Sparse Reward Semantics

The original sparse reward is generated inside LBF environment logic.

Key properties:

- Reward is reset to zero for all players at each environment step
- If a `LOAD` attempt fails due to insufficient combined level, participating agents receive `-penalty`
- If a `LOAD` attempt succeeds, each participating agent receives reward proportional to its level and the food level
- Optional reward normalization may divide the reward by `adjacent_player_level * total_spawned_food`
- The environment returns per-agent rewards as a list

Dense reward modules must not replace this sparse reward logic.

## Native LBF State Elements

The underlying environment exposes useful state through `raw_env`:

- `raw_env.players`: list of player objects
- `raw_env.field`: 2D grid containing food levels
- `raw_env.current_step`: current environment step
- `raw_env.sight`: sight radius
- `raw_env.max_num_food`: maximum number of food items
- `raw_env.penalty`: penalty used for failed load attempts

Each player typically exposes:

- `player.position`: `(row, col)`
- `player.level`: integer level
- `player.reward`: most recent sparse reward
- `player.score`: accumulated reward

These are suitable for shaping terms such as:

- distance-to-food reduction
- coordination incentives
- inactivity penalties
- completion bonuses

## Observation Space in Native LBF

LBF supports two observation styles:

- vector observation
- grid observation

In this project, the common setting is vector observation, and EPyMARL uses flattened observations.

For vector observations, the native observation space is structured as:

- food entries first
- player entries second

Food entries are repeated triples:

- `(food_y, food_x, food_level)`

Player entries are repeated tuples:

- if `observe_agent_levels=True`:
  `(player_y, player_x, player_level)`
- otherwise:
  `(player_y, player_x)`

Empty or unused food/player slots use placeholder values such as:

- `(-1, -1, 0)` for absent food

## EPyMARL / Gymma Representation

EPyMARL wraps the environment through Gymnasium and `GymmaWrapper`.

Relevant wrapper behavior:

- `gym.make(key, **kwargs)` instantiates the registered LBF environment
- `TimeLimit` enforces the episode horizon
- `FlattenObservation` converts observations to flat vectors
- observations may be padded to the longest observation shape across agents
- per-agent action availability is represented as a binary vector

In `GymmaWrapper`:

- `get_obs()` returns a list of per-agent observations
- `get_state()` returns concatenated per-agent observations
- `step(actions)` returns:
  - padded observations
  - reward
  - done
  - truncated
  - info

If `common_reward=True`, EPyMARL may aggregate the per-agent reward list using:

- `sum`
- `mean`

If `common_reward=False`, the reward remains per-agent.

## Dense Reward Runtime Inputs

Dense reward functions are called through:

- `compute_dense_reward(context)`

The runtime context can include:

- `context.t_env`
- `context.episode_idx`
- `context.episode_step`
- `context.test_mode`
- `context.obs_before`
- `context.obs_after`
- `context.state_before`
- `context.state_after`
- `context.actions`
- `context.sparse_reward`
- `context.done`
- `context.truncated`
- `context.info`
- `context.raw_env`

This means a dense reward function can use both:

- wrapper-level observations and actions
- native LBF state through `context.raw_env`

## Shape Compatibility Requirement

The dense reward returned by `compute_dense_reward(context)` must be shape-compatible with `context.sparse_reward`:

- if sparse reward is per-agent, return per-agent dense reward
- if sparse reward is scalar, return scalar dense reward

Alpha mixing is handled outside the reward module by the runtime wrapper.
