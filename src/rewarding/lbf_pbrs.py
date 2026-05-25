from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class AgentState:
    position: tuple[int, int]
    level: int


@dataclass(frozen=True)
class FoodState:
    position: tuple[int, int]
    level: int


@dataclass(frozen=True)
class LBFStateSnapshot:
    foods: List[FoodState]
    agents: List[AgentState]
    field_size: tuple[int, int]


def extract_lbf_state(env) -> LBFStateSnapshot:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    foods: List[FoodState] = []
    food_rows, food_cols = np.nonzero(base_env.field)
    for row, col in zip(food_rows, food_cols):
        foods.append(
            FoodState(position=(int(row), int(col)), level=int(base_env.field[row, col]))
        )

    agents = [
        AgentState(
            position=(int(player.position[0]), int(player.position[1])),
            level=int(player.level),
        )
        for player in getattr(base_env, "players", [])
    ]

    return LBFStateSnapshot(
        foods=foods,
        agents=agents,
        field_size=(int(base_env.rows), int(base_env.cols)),
    )


class LBFPBRS:
    def __init__(
        self,
        beta: float,
        gamma: float,
        wc: float = 0.6,
        wp: float = 0.4,
        phi_clip: tuple[float, float] = (0.0, 1.0),
        closeness_mode: str = "topk_mean",
    ):
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.wc = float(wc)
        self.wp = float(wp)
        self.phi_clip = phi_clip
        self.closeness_mode = str(closeness_mode)

        self._initial_total_food_value = 0.0
        self._agent_levels: List[int] = []

    def reset(self, state: LBFStateSnapshot) -> None:
        self._initial_total_food_value = float(sum(food.level for food in state.foods))
        self._agent_levels = [agent.level for agent in state.agents]

    def compute_lbf_phi(self, state: LBFStateSnapshot) -> float:
        if self._initial_total_food_value <= 0:
            self.reset(state)
        if self._initial_total_food_value <= 0:
            return 0.0

        remaining_food_value = float(sum(food.level for food in state.foods))
        collected_food_value = max(
            self._initial_total_food_value - remaining_food_value,
            0.0,
        )
        phi_col = collected_food_value / self._initial_total_food_value

        phi_prog_numerator = 0.0
        for food in state.foods:
            topk_agents = self._select_topk_agents(food, state.agents)
            gate = self._compute_gate(food, topk_agents)
            closeness = self._compute_closeness(food, topk_agents, state.field_size)
            phi_prog_numerator += float(food.level) * gate * closeness

        phi_prog = phi_prog_numerator / self._initial_total_food_value
        phi = self.wc * phi_col + self.wp * phi_prog
        return float(np.clip(phi, self.phi_clip[0], self.phi_clip[1]))

    def compute_pbrs_reward(
        self,
        prev_state: LBFStateSnapshot,
        next_state: LBFStateSnapshot,
        env_reward,
    ) -> dict:
        if self._initial_total_food_value <= 0:
            self.reset(prev_state)
        phi_t = self.compute_lbf_phi(prev_state)
        phi_tp1 = self.compute_lbf_phi(next_state)
        reward_shape = self.beta * (self.gamma * phi_tp1 - phi_t)
        reward_total = self._apply_shaping(env_reward, reward_shape)
        return {
            "reward_env": env_reward,
            "reward_shape": reward_shape,
            "reward_total": reward_total,
            "phi_t": phi_t,
            "phi_tp1": phi_tp1,
        }

    def _select_topk_agents(
        self,
        food: FoodState,
        agents: Sequence[AgentState],
    ) -> List[AgentState]:
        k_required = self._estimate_required_agents(food.level)
        ranked_agents = sorted(
            agents,
            key=lambda agent: (
                self._manhattan_distance(agent.position, food.position),
                -agent.level,
            ),
        )
        return ranked_agents[:k_required]

    def _estimate_required_agents(self, food_level: int) -> int:
        if not self._agent_levels:
            return 1
        cumulative_level = 0
        for index, level in enumerate(sorted(self._agent_levels, reverse=True), start=1):
            cumulative_level += level
            if cumulative_level >= food_level:
                return index
        return len(self._agent_levels)

    def _compute_gate(
        self,
        food: FoodState,
        topk_agents: Sequence[AgentState],
    ) -> float:
        total_level = sum(agent.level for agent in topk_agents)
        return 1.0 if total_level >= food.level else 0.0

    def _compute_closeness(
        self,
        food: FoodState,
        topk_agents: Sequence[AgentState],
        field_size: tuple[int, int],
    ) -> float:
        if not topk_agents:
            return 0.0
        rows, cols = field_size
        d_max = max((rows - 1) + (cols - 1), 1)
        values = []
        for agent in topk_agents:
            distance = self._manhattan_distance(agent.position, food.position)
            values.append(max(0.0, 1.0 - (distance / d_max)))
        if self.closeness_mode == "topk_nearest":
            return float(max(values))
        return float(sum(values) / len(values))

    def _apply_shaping(self, env_reward, reward_shape: float):
        if isinstance(env_reward, np.ndarray):
            return env_reward + reward_shape
        if isinstance(env_reward, Iterable) and not isinstance(env_reward, (str, bytes)):
            return [float(reward) + reward_shape for reward in env_reward]
        return float(env_reward) + reward_shape

    @staticmethod
    def _manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


class SemiStrictGateLBFPBRS(LBFPBRS):
    def __init__(
        self,
        beta: float,
        gamma: float,
        wc: float = 0.6,
        wp: float = 0.4,
        phi_clip: tuple[float, float] = (0.0, 1.0),
        gate_radius: int = 2,
        closeness_mode: str = "assignment_mean",
    ):
        super().__init__(
            beta=beta,
            gamma=gamma,
            wc=wc,
            wp=wp,
            phi_clip=phi_clip,
            closeness_mode=closeness_mode,
        )
        self.gate_radius = int(gate_radius)

    def compute_lbf_phi(self, state: LBFStateSnapshot) -> float:
        if self._initial_total_food_value <= 0:
            self.reset(state)
        if self._initial_total_food_value <= 0:
            return 0.0

        remaining_food_value = float(sum(food.level for food in state.foods))
        collected_food_value = max(
            self._initial_total_food_value - remaining_food_value,
            0.0,
        )
        phi_col = collected_food_value / self._initial_total_food_value

        food_positions = {food.position for food in state.foods}
        phi_prog_numerator = 0.0
        for food in state.foods:
            assignment = self._find_best_feasible_assignment(
                food=food,
                agents=state.agents,
                field_size=state.field_size,
                food_positions=food_positions,
            )
            gate = 1.0 if assignment is not None else 0.0
            closeness = self._compute_assignment_closeness(
                assignment=assignment,
                field_size=state.field_size,
            )
            phi_prog_numerator += float(food.level) * gate * closeness

        phi_prog = phi_prog_numerator / self._initial_total_food_value
        phi = self.wc * phi_col + self.wp * phi_prog
        return float(np.clip(phi, self.phi_clip[0], self.phi_clip[1]))

    def _find_best_feasible_assignment(
        self,
        food: FoodState,
        agents: Sequence[AgentState],
        field_size: tuple[int, int],
        food_positions: set[tuple[int, int]],
    ) -> list[tuple[AgentState, tuple[int, int], int]] | None:
        load_positions = self._get_valid_loading_positions(
            food=food,
            field_size=field_size,
            food_positions=food_positions,
        )
        if not load_positions:
            return None

        max_team_size = min(len(agents), len(load_positions))
        best_assignment = None
        best_cost = None

        for team_size in range(1, max_team_size + 1):
            for team in combinations(agents, team_size):
                if sum(agent.level for agent in team) < food.level:
                    continue
                assignment = self._minimum_cost_assignment(team, load_positions)
                if assignment is None:
                    continue
                total_cost = sum(distance for _, _, distance in assignment)
                if total_cost > self.gate_radius * team_size:
                    continue
                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_assignment = assignment

        return best_assignment

    def _get_valid_loading_positions(
        self,
        food: FoodState,
        field_size: tuple[int, int],
        food_positions: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        rows, cols = field_size
        row, col = food.position
        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]
        valid_positions = []
        for candidate in candidates:
            cand_row, cand_col = candidate
            if not (0 <= cand_row < rows and 0 <= cand_col < cols):
                continue
            if candidate in food_positions:
                continue
            valid_positions.append(candidate)
        return valid_positions

    def _minimum_cost_assignment(
        self,
        team: Sequence[AgentState],
        load_positions: Sequence[tuple[int, int]],
    ) -> list[tuple[AgentState, tuple[int, int], int]] | None:
        if not team:
            return []
        if len(team) > len(load_positions):
            return None

        best_assignment = None
        best_cost = None
        for slot_choice in combinations(load_positions, len(team)):
            for slot_perm in permutations(slot_choice):
                assignment = []
                total_cost = 0
                for agent, slot in zip(team, slot_perm):
                    distance = self._manhattan_distance(agent.position, slot)
                    assignment.append((agent, slot, distance))
                    total_cost += distance
                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_assignment = assignment
        return best_assignment

    def _compute_assignment_closeness(
        self,
        assignment: Sequence[tuple[AgentState, tuple[int, int], int]] | None,
        field_size: tuple[int, int],
    ) -> float:
        if not assignment:
            return 0.0

        rows, cols = field_size
        d_max = max((rows - 1) + (cols - 1), 1)
        values = [max(0.0, 1.0 - (distance / d_max)) for _, _, distance in assignment]
        if self.closeness_mode == "assignment_nearest":
            return float(max(values))
        return float(sum(values) / len(values))


def build_lbf_pbrs(
    *,
    variant: str,
    beta: float,
    gamma: float,
    wc: float = 0.6,
    wp: float = 0.4,
    phi_clip: tuple[float, float] = (0.0, 1.0),
    semi_strict_gate_radius: int = 2,
    gate_mode: str | None = None,
    closeness_mode: str | None = None,
):
    normalized_variant = str(variant).strip().lower()
    normalized_gate_mode = str(gate_mode).strip().lower() if gate_mode is not None else None
    normalized_closeness_mode = (
        str(closeness_mode).strip().lower() if closeness_mode is not None else None
    )
    if normalized_gate_mode == "feasible_assignment" or (
        normalized_closeness_mode is not None and normalized_closeness_mode.startswith("assignment_")
    ):
        return SemiStrictGateLBFPBRS(
            beta=beta,
            gamma=gamma,
            wc=wc,
            wp=wp,
            phi_clip=phi_clip,
            gate_radius=semi_strict_gate_radius,
            closeness_mode=normalized_closeness_mode or "assignment_mean",
        )
    if normalized_variant == "original":
        return LBFPBRS(
            beta=beta,
            gamma=gamma,
            wc=wc,
            wp=wp,
            phi_clip=phi_clip,
            closeness_mode=normalized_closeness_mode or "topk_mean",
        )
    if normalized_variant in {"semi_strict_gate", "semi-strict-gate"}:
        return SemiStrictGateLBFPBRS(
            beta=beta,
            gamma=gamma,
            wc=wc,
            wp=wp,
            phi_clip=phi_clip,
            gate_radius=semi_strict_gate_radius,
            closeness_mode=normalized_closeness_mode or "assignment_mean",
        )
    raise ValueError(
        f"Unknown PBRS variant '{variant}'. Supported variants: 'original', 'semi_strict_gate'."
    )


def build_lbf_pbrs_from_config(config: dict | None):
    if not isinstance(config, dict) or not config.get("enabled", False):
        return None
    phi_clip = tuple(config.get("phi_clip", [0.0, 1.0]))
    gate_config = config.get("gate", {}) if isinstance(config.get("gate"), dict) else {}
    closeness_config = (
        config.get("closeness", {}) if isinstance(config.get("closeness"), dict) else {}
    )
    return build_lbf_pbrs(
        variant=config.get("variant", "original"),
        beta=float(config.get("beta", 0.3)),
        gamma=float(config.get("gamma", 0.99)),
        wc=float(config.get("wc", 0.6)),
        wp=float(config.get("wp", 0.4)),
        phi_clip=phi_clip,
        semi_strict_gate_radius=int(
            gate_config.get("radius", config.get("semi_strict_gate_radius", 2))
        ),
        gate_mode=gate_config.get("mode"),
        closeness_mode=closeness_config.get("mode"),
    )
