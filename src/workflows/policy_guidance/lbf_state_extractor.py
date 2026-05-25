from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from rewarding.lbf_pbrs import extract_lbf_state


def extract_policy_guidance_step_context(
    *,
    env: Any = None,
    obs: Any = None,
    actions: Any = None,
    reward: Any = None,
    terminated: bool = False,
    info: Any = None,
) -> Dict[str, Any]:
    env_state_payload = _extract_lbf_env_state(env)
    extraction_status = env_state_payload["extraction_status"]
    env_state = env_state_payload["env_state"]

    if env_state is None and isinstance(obs, dict):
        env_state = _normalize_lbf_like_state_dict(obs)
        if env_state is not None:
            extraction_status = "obs_action"

    if env_state is None:
        extraction_status = "insufficient_evidence"

    return {
        "extraction_status": extraction_status,
        "env_state": env_state,
        "obs": obs,
        "actions": actions,
        "reward": reward,
        "terminated": bool(terminated),
        "info": dict(info or {}),
    }


def _extract_lbf_env_state(env: Any) -> Dict[str, Any]:
    if env is None:
        return {"extraction_status": "insufficient_evidence", "env_state": None}
    candidates = [env]
    for attr in ("_env", "env", "unwrapped"):
        nested = getattr(env, attr, None)
        if nested is not None and nested not in candidates:
            candidates.append(nested)

    for candidate in candidates:
        try:
            snapshot = extract_lbf_state(candidate)
            normalized = _normalize_snapshot(snapshot)
            if normalized is not None:
                return {"extraction_status": "env_state", "env_state": normalized}
        except Exception:
            continue

    cached = getattr(env, "_cached_lbf_state", None)
    normalized_cached = _normalize_snapshot(cached)
    if normalized_cached is not None:
        return {"extraction_status": "env_state", "env_state": normalized_cached}
    return {"extraction_status": "insufficient_evidence", "env_state": None}


def _normalize_snapshot(snapshot: Any) -> Dict[str, Any] | None:
    if snapshot is None:
        return None
    if is_dataclass(snapshot):
        payload = asdict(snapshot)
    elif isinstance(snapshot, dict):
        payload = snapshot
    else:
        payload = None
    if not isinstance(payload, dict):
        return None
    return _normalize_lbf_like_state_dict(payload)


def _normalize_lbf_like_state_dict(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    field_size = payload.get("field_size") or payload.get("grid_size")
    agents = payload.get("agents")
    foods = payload.get("foods")
    if not isinstance(field_size, (list, tuple)) or len(field_size) != 2:
        return None
    if not isinstance(agents, list) or not isinstance(foods, list):
        return None
    normalized_agents = []
    for index, agent in enumerate(agents):
        if not isinstance(agent, dict):
            return None
        position = agent.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            return None
        normalized_agents.append(
            {
                "id": int(agent.get("id", index)),
                "position": [int(position[0]), int(position[1])],
                "level": int(agent.get("level", 1)),
            }
        )
    normalized_foods = []
    for index, food in enumerate(foods):
        if not isinstance(food, dict):
            return None
        position = food.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            return None
        normalized_foods.append(
            {
                "id": str(food.get("id", f"food_{index}")),
                "position": [int(position[0]), int(position[1])],
                "level": int(food.get("level", 1)),
            }
        )
    return {
        "field_size": [int(field_size[0]), int(field_size[1])],
        "agents": normalized_agents,
        "foods": normalized_foods,
        "current_step": int(payload.get("current_step", 0)),
    }
