from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict


DEFAULT_COMPAT_BASE_URL = "https://www.iuseapi.com/v1"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _config_value(
    *,
    cli_value: Any,
    profile_value: Any,
    env_name: str,
    default: Any,
) -> tuple[Any, str]:
    if cli_value not in (None, ""):
        return cli_value, "cli"
    if profile_value not in (None, ""):
        return profile_value, "profile"
    env_value = os.environ.get(env_name)
    if env_value not in (None, ""):
        return env_value, "env"
    return default, "default"


def redact_base_url(base_url: str) -> str:
    text = _string(base_url).strip()
    if not text:
        return ""
    if "://" not in text:
        return text
    scheme, _, remainder = text.partition("://")
    host = remainder.split("/", 1)[0]
    return f"{scheme}://{host}"


def llm_routing_artifact_fields(routing: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = deepcopy(routing or {})
    api_key_env = str(payload.get("api_key_env") or "")
    return {
        "llm_model_tier": str(payload.get("tier") or ""),
        "llm_provider": str(payload.get("provider") or ""),
        "llm_model": str(payload.get("model") or ""),
        "llm_base_url": redact_base_url(str(payload.get("base_url") or "")),
        "base_url_label_or_redacted": redact_base_url(str(payload.get("base_url") or "")),
        "api_key_env_name": api_key_env,
        "api_key_env_exists": bool(api_key_env and os.environ.get(api_key_env)),
        "llm_routing_reason": str(payload.get("reason") or ""),
        "llm_routing_source": str(payload.get("source") or ""),
        "formal_model_allowed": bool(payload.get("formal_model_allowed", False)),
        "validation_model_used": bool(payload.get("tier") == "validation"),
    }


def resolve_llm_model_tier(context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ctx = deepcopy(context or {})
    routing_config = deepcopy(ctx.get("llm_model_routing") or {})
    enabled = bool(
        routing_config.get("enabled", True)
        if routing_config.get("enabled") is not None
        else _env_bool("LLM_MODEL_ROUTING_ENABLED", True)
    )
    explicit_tier = _string(
        ctx.get("llm_model_tier_override")
        or ctx.get("explicit_llm_model_tier")
        or ctx.get("llm_model_tier")
    ).strip().lower()
    dry_run = bool(ctx.get("dry_run") or ctx.get("planned_only"))
    execute = bool(ctx.get("execute"))
    allow_formal_execute = bool(ctx.get("allow_formal_execute"))
    profile_name = _string(ctx.get("profile_name") or ctx.get("workflow_profile")).lower()
    workflow_id = _string(ctx.get("workflow_id")).lower()
    stage_name = _string(ctx.get("stage_name")).lower()
    script_name = _string(ctx.get("script_name")).lower()
    force_allow_formal_model_in_validation = bool(
        ctx.get("force_allow_formal_model_in_validation")
        or _env_bool("FORCE_ALLOW_FORMAL_MODEL_IN_VALIDATION", False)
    )
    formal_only_when_allow_formal_execute = bool(
        routing_config.get("formal_only_when_allow_formal_execute", True)
    )

    smoke_like = any(
        token in " ".join([profile_name, workflow_id, stage_name, script_name])
        for token in ("smoke", "mock", "tiny", "dryrun", "dry-run", "planning", "regression")
    )
    formal_profile = (
        "formal_pilot" in profile_name
        or "formal_seed" in profile_name
        or "formal" in profile_name
        or bool(ctx.get("formal_execute_guard_required"))
        or "formal" in _string(ctx.get("budget_profile")).lower()
    )

    if dry_run or smoke_like:
        tier = "validation"
        reason = "dry_run_or_smoke_validation"
    elif explicit_tier == "formal":
        if not (execute and allow_formal_execute and formal_profile):
            if not force_allow_formal_model_in_validation:
                raise ValueError(
                    "explicit formal LLM tier is blocked outside true formal execute; "
                    "set FORCE_ALLOW_FORMAL_MODEL_IN_VALIDATION=true only for intentional exceptions"
                )
        tier = "formal"
        reason = "explicit_formal_override"
    elif explicit_tier == "validation":
        tier = "validation"
        reason = "explicit_validation_override"
    elif execute and formal_profile and (
        allow_formal_execute or not formal_only_when_allow_formal_execute
    ):
        tier = "formal"
        reason = "formal_execute_with_allow_formal_execute"
    else:
        tier = "validation"
        reason = "default_validation_routing"

    validation_provider, validation_provider_source = _config_value(
        cli_value=ctx.get("llm_validation_provider"),
        profile_value=routing_config.get("validation_provider"),
        env_name="LLM_VALIDATION_PROVIDER",
        default="deepseek",
    )
    validation_model, validation_model_source = _config_value(
        cli_value=ctx.get("llm_validation_model"),
        profile_value=routing_config.get("validation_model"),
        env_name="LLM_VALIDATION_MODEL",
        default="deepseek-chat",
    )
    validation_base_url, validation_base_url_source = _config_value(
        cli_value=ctx.get("llm_validation_base_url"),
        profile_value=routing_config.get("validation_base_url"),
        env_name="LLM_VALIDATION_BASE_URL",
        default=DEFAULT_COMPAT_BASE_URL,
    )
    validation_api_key_env, validation_api_key_env_source = _config_value(
        cli_value=ctx.get("llm_validation_api_key_env"),
        profile_value=routing_config.get("validation_api_key_env"),
        env_name="LLM_VALIDATION_API_KEY_ENV",
        default="IUSEAPI_API_KEY",
    )

    formal_provider, formal_provider_source = _config_value(
        cli_value=ctx.get("llm_formal_provider"),
        profile_value=routing_config.get("formal_provider"),
        env_name="LLM_FORMAL_PROVIDER",
        default="openai",
    )
    formal_model, formal_model_source = _config_value(
        cli_value=ctx.get("llm_formal_model"),
        profile_value=routing_config.get("formal_model"),
        env_name="LLM_FORMAL_MODEL",
        default="gpt-5.2",
    )
    formal_base_url, formal_base_url_source = _config_value(
        cli_value=ctx.get("llm_formal_base_url"),
        profile_value=routing_config.get("formal_base_url"),
        env_name="LLM_FORMAL_BASE_URL",
        default=DEFAULT_COMPAT_BASE_URL,
    )
    formal_api_key_env, formal_api_key_env_source = _config_value(
        cli_value=ctx.get("llm_formal_api_key_env"),
        profile_value=routing_config.get("formal_api_key_env"),
        env_name="LLM_FORMAL_API_KEY_ENV",
        default="IUSEAPI_API_KEY",
    )

    if tier == "validation":
        provider = _string(validation_provider)
        model = _string(validation_model)
        base_url = _string(validation_base_url)
        api_key_env = _string(validation_api_key_env)
        source = validation_model_source
        if model == "gpt-5.2" and not force_allow_formal_model_in_validation:
            raise ValueError("validation tier must not use gpt-5.2 without FORCE_ALLOW_FORMAL_MODEL_IN_VALIDATION=true")
    else:
        provider = _string(formal_provider)
        model = _string(formal_model)
        base_url = _string(formal_base_url)
        api_key_env = _string(formal_api_key_env)
        source = formal_model_source
        if model != "gpt-5.2" and not force_allow_formal_model_in_validation:
            raise ValueError("formal tier must use gpt-5.2 unless an explicit override is intentionally allowed")
        if provider.lower() == "deepseek" and not force_allow_formal_model_in_validation:
            raise ValueError("formal tier must not silently use DeepSeek; provide an explicit override only if intended")

    return {
        "routing_enabled": enabled,
        "tier": tier,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "reason": reason,
        "source": source,
        "formal_model_allowed": bool(tier == "formal"),
        "validation_model_used": bool(tier == "validation"),
        "api_key_env_exists": bool(api_key_env and os.environ.get(api_key_env)),
        "base_url_label_or_redacted": redact_base_url(base_url),
        "formal_profile_detected": bool(formal_profile),
        "execute": execute,
        "allow_formal_execute": allow_formal_execute,
        "dry_run": dry_run,
        "profile_name": _string(ctx.get("profile_name") or ctx.get("workflow_profile")),
        "workflow_id": _string(ctx.get("workflow_id")),
        "validation_provider": _string(validation_provider),
        "validation_model": _string(validation_model),
        "validation_base_url": _string(validation_base_url),
        "validation_api_key_env": _string(validation_api_key_env),
        "formal_provider": _string(formal_provider),
        "formal_model": _string(formal_model),
        "formal_base_url": _string(formal_base_url),
        "formal_api_key_env": _string(formal_api_key_env),
        "validation_provider_source": validation_provider_source,
        "validation_model_source": validation_model_source,
        "validation_base_url_source": validation_base_url_source,
        "validation_api_key_env_source": validation_api_key_env_source,
        "formal_provider_source": formal_provider_source,
        "formal_model_source": formal_model_source,
        "formal_base_url_source": formal_base_url_source,
        "formal_api_key_env_source": formal_api_key_env_source,
    }
