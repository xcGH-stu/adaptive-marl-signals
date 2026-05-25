from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional


DEFAULT_BASE_URL = "https://www.iuseapi.com/v1"
DEFAULT_MODEL = "gpt-5.5"


def probe_openai_backend(
    *,
    api_key_env: str = "IUSEAPI_API_KEY",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    api_key_detected = bool(
        os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
    )
    import_available = True
    import_error = None
    try:
        from openai import OpenAI  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-dependent
        import_available = False
        import_error = f"{exc.__class__.__name__}: {exc}"
    return {
        "api_key_env": str(api_key_env),
        "api_key_detected": bool(api_key_detected),
        "openai_import_available": bool(import_available),
        "openai_import_error": import_error,
        "base_url": str(base_url),
        "model": str(model),
        "backend_ready": bool(api_key_detected and import_available),
    }


class OpenAIChatBackend:
    """
    Thin OpenAI-compatible chat backend used by both Generator and Critic.

    API keys are intentionally read from environment variables at runtime rather
    than stored in source files.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_key_env: str = "IUSEAPI_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 5.0,
    ):
        self.api_key = api_key or os.environ.get(api_key_env) or os.environ.get(
            "OPENAI_API_KEY"
        )
        if not self.api_key:
            raise ValueError(
                f"Missing API key. Set {api_key_env} (or OPENAI_API_KEY) before running."
            )

        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                OpenAI,
                RateLimitError,
            )
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for the real LLM backend. "
                "Install it in the active environment before running."
            ) from exc

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._retryable_errors = (
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
            InternalServerError,
        )

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                break
            except self._retryable_errors as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_seconds * attempt)
        else:
            raise last_error

        text = response.choices[0].message.content or ""

        usage = None
        if getattr(response, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }

        raw_response = (
            response.model_dump()
            if hasattr(response, "model_dump")
            else {
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
                "metadata": metadata,
            }
        )

        return {
            "text": text,
            "raw_response": raw_response,
            "model": self.model,
            "request_id": getattr(response, "id", None),
            "usage": usage,
            "backend_metadata": {
                "provider": "openai" if "gpt-" in str(self.model).lower() else None,
                "base_url": self.base_url,
                "api_key_env": self.api_key_env,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "metadata": metadata or {},
            },
        }


class FallbackLLMBackend:
    """
    Wrap a primary backend with a deterministic fallback backend.

    This is especially useful for Critic calls in long workflows: training may
    have already completed, so an API timeout should degrade gracefully rather
    than wasting the whole round.
    """

    def __init__(
        self,
        *,
        primary_backend: Any,
        fallback_backend: Any,
        fallback_label: str = "fallback",
    ):
        self.primary_backend = primary_backend
        self.fallback_backend = fallback_backend
        self.fallback_label = fallback_label

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            return self.primary_backend.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata=metadata,
            )
        except Exception as exc:
            fallback_result = self.fallback_backend.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata=metadata,
            )
            raw_response = fallback_result.get("raw_response")
            if not isinstance(raw_response, dict):
                raw_response = {"fallback": True}
            raw_response = dict(raw_response)
            raw_response["fallback_triggered"] = True
            raw_response["fallback_label"] = self.fallback_label
            raw_response["primary_error"] = str(exc)
            fallback_result["raw_response"] = raw_response
            fallback_result["model"] = (
                f"{fallback_result.get('model') or 'fallback'}::{self.fallback_label}"
            )
            backend_metadata = fallback_result.get("backend_metadata")
            if not isinstance(backend_metadata, dict):
                backend_metadata = {}
            backend_metadata = dict(backend_metadata)
            backend_metadata["fallback_triggered"] = True
            backend_metadata["primary_error"] = str(exc)
            backend_metadata["fallback_label"] = self.fallback_label
            fallback_result["backend_metadata"] = backend_metadata
            return fallback_result
