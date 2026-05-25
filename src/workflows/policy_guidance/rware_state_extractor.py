from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple


ACTION_NAMES = {
    0: "NOOP",
    1: "FORWARD",
    2: "LEFT",
    3: "RIGHT",
    4: "TOGGLE_LOAD",
}


def extract_rware_state(
    env_or_wrapper: Any,
    prev_state: Dict[str, Any] | None = None,
    actions: Any = None,
    reward: Any = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    base_env = _resolve_rware_env(env_or_wrapper)
    if base_env is None:
        return {
            "env_type": "rware",
            "grid_size": None,
            "episode_step": 0,
            "max_steps": None,
            "agents": [],
            "shelves": [],
            "requested_shelves": [],
            "goals": [],
            "highways_shape": None,
            "grid_shape": None,
            "events": {
                "pickup_events": [],
                "delivery_events": [],
                "blocked_forward_events": [],
            },
            "feature_flags": {
                "direct_env_state": False,
                "delivery_inferred": False,
                "blocked_inferred": False,
            },
            "warnings": [
                "Unable to resolve a Warehouse-like RWARE environment from the input object."
            ],
            "actions": _normalize_actions(actions, None),
            "reward": _scalarize_reward(reward),
        }

    normalized_actions = _normalize_actions(actions, getattr(base_env, "n_agents", None))
    requested_ids = {
        int(getattr(shelf, "id", -1))
        for shelf in list(getattr(base_env, "request_queue", []) or [])
    }
    agents = []
    for agent in list(getattr(base_env, "agents", []) or []):
        carrying = getattr(agent, "carrying_shelf", None)
        agent_id = int(getattr(agent, "id", -1))
        requested_action = None
        if normalized_actions and 0 <= agent_id - 1 < len(normalized_actions):
            requested_action = normalized_actions[agent_id - 1]
        elif getattr(agent, "req_action", None) is not None:
            requested_action = getattr(getattr(agent, "req_action"), "name", None)
        prev_position = None
        prev_x = getattr(agent, "prev_x", None)
        prev_y = getattr(agent, "prev_y", None)
        if prev_x is not None and prev_y is not None:
            prev_position = [int(prev_x), int(prev_y)]
        agents.append(
            {
                "id": agent_id,
                "position": [int(getattr(agent, "x", -1)), int(getattr(agent, "y", -1))],
                "direction": getattr(getattr(agent, "dir", None), "name", "UNKNOWN"),
                "requested_action": requested_action,
                "carrying_shelf_id": (
                    None if carrying is None else int(getattr(carrying, "id", -1))
                ),
                "has_delivered": bool(getattr(agent, "has_delivered", False)),
                "prev_position": prev_position,
            }
        )

    shelves = []
    for shelf in list(getattr(base_env, "shelfs", []) or []):
        shelf_id = int(getattr(shelf, "id", -1))
        shelves.append(
            {
                "id": shelf_id,
                "position": [int(getattr(shelf, "x", -1)), int(getattr(shelf, "y", -1))],
                "requested": shelf_id in requested_ids,
            }
        )

    requested_shelves = [
        {
            "id": int(getattr(shelf, "id", -1)),
            "position": [int(getattr(shelf, "x", -1)), int(getattr(shelf, "y", -1))],
        }
        for shelf in list(getattr(base_env, "request_queue", []) or [])
    ]
    goals = [
        [int(goal[0]), int(goal[1])]
        for goal in list(getattr(base_env, "goals", []) or [])
        if isinstance(goal, (list, tuple)) and len(goal) == 2
    ]

    current_state = {
        "env_type": "rware",
        "grid_size": _shape_pair(getattr(base_env, "grid_size", None)),
        "episode_step": int(getattr(base_env, "_cur_steps", 0) or 0),
        "max_steps": getattr(base_env, "max_steps", None),
        "agents": agents,
        "shelves": shelves,
        "requested_shelves": requested_shelves,
        "goals": goals,
        "highways_shape": _shape_of(getattr(base_env, "highways", None)),
        "grid_shape": _shape_of(getattr(base_env, "grid", None)),
        "events": {
            "pickup_events": [],
            "delivery_events": [],
            "blocked_forward_events": [],
        },
        "feature_flags": {
            "direct_env_state": True,
            "delivery_inferred": prev_state is not None,
            "blocked_inferred": prev_state is not None and bool(normalized_actions),
        },
        "warnings": warnings,
        "actions": normalized_actions,
        "reward": _scalarize_reward(reward),
    }

    previous = _normalize_previous_state(prev_state)
    current_state["events"]["pickup_events"] = _infer_pickup_events(previous, current_state)
    delivery_events = _infer_delivery_events(previous, current_state)
    if previous is None:
        current_state["feature_flags"]["delivery_inferred"] = False
    elif not delivery_events and _previous_request_ids(previous) and not current_state["requested_shelves"]:
        warnings.append("Delivery inference ran but found no confident request-queue transition.")
    current_state["events"]["delivery_events"] = delivery_events

    blocked_events, blocked_warning = _infer_blocked_forward_events(
        previous,
        current_state,
        normalized_actions,
    )
    if blocked_warning:
        warnings.append(blocked_warning)
    current_state["events"]["blocked_forward_events"] = blocked_events
    if previous is None or not normalized_actions:
        current_state["feature_flags"]["blocked_inferred"] = False
    return current_state


def _resolve_rware_env(env_or_wrapper: Any) -> Any:
    candidates = [env_or_wrapper]
    visited = set()
    while candidates:
        current = candidates.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if _looks_like_rware_env(current):
            return current
        for attr in ("_env", "env", "unwrapped"):
            nested = getattr(current, attr, None)
            if nested is not None and id(nested) not in visited:
                candidates.append(nested)
    return None


def _looks_like_rware_env(candidate: Any) -> bool:
    return all(
        hasattr(candidate, name)
        for name in ("agents", "shelfs", "request_queue", "goals", "grid_size")
    )


def _normalize_actions(actions: Any, n_agents: int | None) -> List[str]:
    if actions is None:
        return []
    normalized: List[str] = []
    if isinstance(actions, (list, tuple)):
        raw_actions = list(actions)
    else:
        raw_actions = [actions]
    for item in raw_actions:
        if hasattr(item, "name"):
            normalized.append(str(item.name))
            continue
        try:
            normalized.append(ACTION_NAMES.get(int(item), str(item)))
        except Exception:
            normalized.append(str(item))
    if n_agents is not None and len(normalized) < int(n_agents):
        normalized.extend(["UNKNOWN"] * (int(n_agents) - len(normalized)))
    return normalized


def _normalize_previous_state(prev_state: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(prev_state, dict):
        return None
    if str(prev_state.get("env_type") or "") != "rware":
        return None
    return deepcopy(prev_state)


def _infer_pickup_events(
    prev_state: Dict[str, Any] | None,
    current_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if prev_state is None:
        return []
    previous_agents = {
        int(agent.get("id", -1)): agent for agent in list(prev_state.get("agents") or [])
    }
    events = []
    for agent in list(current_state.get("agents") or []):
        agent_id = int(agent.get("id", -1))
        prev_agent = previous_agents.get(agent_id, {})
        prev_carry = prev_agent.get("carrying_shelf_id")
        carry = agent.get("carrying_shelf_id")
        if prev_carry is None and carry is not None:
            events.append(
                {
                    "agent_id": agent_id,
                    "shelf_id": int(carry),
                    "inferred": True,
                }
            )
    return events


def _infer_delivery_events(
    prev_state: Dict[str, Any] | None,
    current_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if prev_state is None:
        return []
    prev_request_ids = _previous_request_ids(prev_state)
    curr_request_ids = {
        int(shelf.get("id", -1))
        for shelf in list(current_state.get("requested_shelves") or [])
    }
    removed_ids = sorted(prev_request_ids - curr_request_ids)
    positive_reward = bool(float(current_state.get("reward") or 0.0) > 0.0)
    delivered_agents = [
        int(agent.get("id", -1))
        for agent in list(current_state.get("agents") or [])
        if bool(agent.get("has_delivered"))
    ]
    events = []
    for shelf_id in removed_ids:
        low_confidence = not (positive_reward or delivered_agents)
        events.append(
            {
                "shelf_id": shelf_id,
                "positive_reward_observed": positive_reward,
                "delivered_agent_ids": delivered_agents,
                "inferred": True,
                "inferred_with_low_confidence": low_confidence,
            }
        )
    return events


def _previous_request_ids(prev_state: Dict[str, Any]) -> set[int]:
    return {
        int(shelf.get("id", -1))
        for shelf in list(prev_state.get("requested_shelves") or [])
    }


def _infer_blocked_forward_events(
    prev_state: Dict[str, Any] | None,
    current_state: Dict[str, Any],
    normalized_actions: List[str],
) -> Tuple[List[Dict[str, Any]], str | None]:
    if prev_state is None:
        return [], None
    if not normalized_actions:
        return [], "Blocked-forward inference skipped because actions were unavailable."
    previous_agents = {
        int(agent.get("id", -1)): agent for agent in list(prev_state.get("agents") or [])
    }
    events = []
    for agent in list(current_state.get("agents") or []):
        agent_id = int(agent.get("id", -1))
        prev_agent = previous_agents.get(agent_id)
        if prev_agent is None:
            continue
        if not (0 <= agent_id - 1 < len(normalized_actions)):
            continue
        if normalized_actions[agent_id - 1] != "FORWARD":
            continue
        if list(prev_agent.get("position") or []) == list(agent.get("position") or []):
            events.append(
                {
                    "agent_id": agent_id,
                    "requested_action": "FORWARD",
                    "position": deepcopy(agent.get("position")),
                    "inferred": True,
                }
            )
    return events, None


def _shape_pair(value: Any) -> List[int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    return None


def _shape_of(value: Any) -> List[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _scalarize_reward(reward: Any) -> float:
    if reward is None:
        return 0.0
    if isinstance(reward, (list, tuple)):
        return float(sum(float(item) for item in reward))
    return float(reward)
