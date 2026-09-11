from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Type

import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import Settings, is_free_model_id, normalize_model_id
from app.storage.db import Database

logger = logging.getLogger(__name__)


class LLMQuotaExceeded(RuntimeError):
    """Legacy exception retained for compatibility."""


class OpenRouterQuotaExceeded(RuntimeError):
    """Raised when OpenRouter itself says the account quota is exhausted."""


class LLMRequestGate:
    """Shared request spacing + local usage telemetry for the OpenRouter daily budget."""

    def __init__(self, settings: Settings, db: Database):
        self.db = db
        # Keep 50 as the local UI/account-budget reference. It is telemetry, not
        # a hard stop: OpenRouter itself is the authoritative quota source.
        self.daily_budget = min(max(1, settings.llm_daily_request_budget), 50)
        self.min_interval = max(0.0, settings.llm_min_request_interval_seconds)
        self._lock = asyncio.Lock()
        # Free OpenRouter model routes can return malformed/empty responses when
        # several requests are in flight at once. The quota gate previously spaced
        # reservations, but released the lock before the HTTP call, so the research
        # wave could still hit the same model concurrently. Serialize actual model
        # requests through one shared lock for deterministic provider behavior.
        self._request_lock = asyncio.Lock()
        stored_day = self.db.get_control_value("llm_quota_day", "") if self.db else ""
        stored_count = self.db.get_control_value("llm_quota_used", "0") if self.db else "0"
        self._day = stored_day or self._today_key()
        try:
            self._count = int(stored_count) if self._day == self._today_key() else 0
        except ValueError:
            self._count = 0
        self._last_request_at = 0.0
        self._persist()

    @staticmethod
    def _today_key() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _persist(self) -> None:
        if not self.db:
            return
        self.db.set_control_value("llm_quota_day", self._day)
        self.db.set_control_value("llm_quota_used", str(self._count))

    def _rollover_if_needed(self) -> None:
        day = self._today_key()
        if day != self._day:
            self._day = day
            self._count = 0
            self._last_request_at = 0.0
            self._persist()

    async def acquire(self) -> None:
        async with self._lock:
            self._rollover_if_needed()
            # This counter is telemetry only. OpenRouter is the sole authority for
            # when the account/model has actually exhausted its request allowance.
            wait = self.min_interval - (time.monotonic() - self._last_request_at)
            if wait > 0:
                await asyncio.sleep(wait)
            self._rollover_if_needed()
            # Count actual provider attempts for display/telemetry.
            self._count += 1
            self._last_request_at = time.monotonic()
            self._persist()

    def remaining(self) -> int:
        self._rollover_if_needed()
        return max(0, self.daily_budget - self._count)

    def snapshot(self) -> dict[str, Any]:
        self._rollover_if_needed()
        return {
            "used": self._count,
            "budget": self.daily_budget,
            "remaining": max(0, self.daily_budget - self._count),
            "day_utc": self._day,
            "min_interval_seconds": self.min_interval,
        }

    @property
    def request_lock(self) -> asyncio.Lock:
        """Shared lock guarding the actual provider HTTP request."""
        return self._request_lock


class Agent:
    """One LLM-backed desk role.

    Model selection is entirely config-driven — nothing here hardcodes a
    specific model ID. Each agent resolves a primary model (its own
    ``MODEL_<NAME>`` override if set, else ``MODEL_DEFAULT``) and a
    single fallback model (``MODEL_FALLBACK``, tried only if the
    primary is exhausted and differs from it). See ``_candidate_models``.
    """

    def __init__(
            self,
            name: str,
            role: str,
            prompt_path: Path,
            model: str,
            settings: Settings,
            db: Database,
            gate: LLMRequestGate,
    ):
        self.name = name
        self.role = role
        self.prompt_path = prompt_path
        self.model_override = normalize_model_id(model) if model else ""
        if self.model_override and not is_free_model_id(self.model_override):
            raise ValueError(
                f"{self.name} model override must be a free OpenRouter model; got {self.model_override!r}"
            )
        self.default_model = normalize_model_id(settings.model_default)
        self.fallback_model = normalize_model_id(settings.model_fallback) if settings.model_fallback else ""
        self.model = self.model_override or self.default_model
        self.settings = settings
        self.db = db
        self.gate = gate
        self.prompt = self._load_prompt()
        self.status = "IDLE"
        self.enabled = True
        self.last_run = None
        self.last_summary = ""
        self.last_error = ""
        self.last_served_provider = ""
        self.last_requested_model = self.model
        # The OpenAI-compatible SDK retries 429s (and other transient errors)
        # automatically by default. That conflicts with our account-level quota
        # scheduler because a single logical agent request can become multiple
        # provider requests. Keep retry policy exclusively in this application.
        self.client = AsyncOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            max_retries=0,
            default_headers={
                "HTTP-Referer": settings.site_url,
                "X-OpenRouter-Title": settings.site_name,
            },
        )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.status = "IDLE"

    def _load_prompt(self) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Agent prompt missing: {self.prompt_path}")
        data = yaml.safe_load(self.prompt_path.read_text(encoding="utf-8")) or {}
        return str(data.get("system_prompt", "")).strip()

    def _candidate_models(self) -> list[str]:
        """Ordered models to try for one logical request: primary, then fallback.

        Primary is this agent's own override if set, else the shared default.
        The fallback is only appended when it differs from the primary, so an
        agent with no override and a fallback equal to the default (the normal
        zero-config case) tries exactly one model, same as before.
        """
        primary = self.model_override or self.default_model
        chain = [primary]
        if self.fallback_model and self.fallback_model != primary:
            chain.append(self.fallback_model)
        return chain

    async def run(self, input_payload: dict[str, Any], response_model: Type[BaseModel]) -> BaseModel:
        if not self.enabled:
            raise RuntimeError(f"{self.name} is disabled while autopilot is OFF")

        self.last_error = ""
        self.status = "THINKING"
        schema = response_model.model_json_schema()
        schema_text = json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": (
                        self.prompt
                        + "\n\nREQUIRED OUTPUT CONTRACT: Return ONLY one valid JSON object. "
                        + "Do not use Markdown fences, prose, commentary, or tool calls. "
                        + f"The JSON object MUST validate against this JSON Schema for {response_model.__name__}: {schema_text}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(input_payload, default=str, ensure_ascii=False),
            },
        ]

        candidates = self._candidate_models()
        self.last_requested_model = candidates[0]
        self.status = "WORKING"

        response_tool = {
            "type": "function",
            "function": {
                "name": "submit_response",
                "description": f"Return the validated {response_model.__name__} result as JSON arguments.",
                "parameters": schema,
            },
        }

        async def request(model_name: str):
            # Configured free-tier models are expected to support tool calling but
            # not necessarily response_format. Force the structured response
            # through one tool so the model does not have to reliably emit raw
            # JSON in normal assistant text.
            return await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=[response_tool],
                tool_choice={"type": "function", "function": {"name": "submit_response"}},
                temperature=0.0,
                max_tokens=2200,
            )

        # A free-route provider can occasionally return an empty choices array or
        # malformed plain-text content even when the HTTP request itself succeeded.
        # Permit one application-level retry per model for those transient
        # formatting failures, then move on to the next candidate model (if a
        # fallback is configured). Quota/routing errors and schema validation
        # errors are never retried.
        attempts_per_model = 2
        last_error: Exception | None = None
        attempted: list[str] = []
        response = None
        parsed = None
        for model_name in candidates:
            for attempt in range(attempts_per_model):
                if not self.enabled:
                    raise RuntimeError(f"{self.name} was stopped because autopilot was turned OFF")
                try:
                    await self.gate.acquire()
                    attempted.append(f"{model_name}:tool:{attempt + 1}")
                    # Keep the entire provider interaction serialized, not just the
                    # quota reservation. This prevents concurrent free-route requests
                    # from producing intermittent empty-choice/garbled responses.
                    async with self.gate.request_lock:
                        response = await request(model_name)
                    parsed = _parse_agent_response(response, response_model)
                    self.last_requested_model = model_name
                    break
                except (LLMQuotaExceeded, OpenRouterQuotaExceeded):
                    raise
                except Exception as exc:
                    last_error = exc
                    if _is_openrouter_daily_quota_error(exc):
                        logger.error("%s hit OpenRouter's free-model daily quota; stopping this AI cycle: %s",
                                     self.name, _compact_error(exc))
                        raise OpenRouterQuotaExceeded(
                            f"OpenRouter daily quota exhausted: {_compact_error(exc)}"
                        ) from exc
                    transient = _is_transient_structured_response_error(exc)
                    if transient and attempt + 1 < attempts_per_model:
                        logger.warning(
                            "%s %s structured response failed on attempt %s/%s; retrying once: %s",
                            self.name,
                            model_name,
                            attempt + 1,
                            attempts_per_model,
                            _compact_error(exc),
                        )
                        await asyncio.sleep(0.75)
                    else:
                        logger.warning(
                            "%s %s attempt %s/%s failed: %s",
                            self.name,
                            model_name,
                            attempt + 1,
                            attempts_per_model,
                            _compact_error(exc),
                        )
                    response = None
                    parsed = None
            if parsed is not None:
                break

        if parsed is None or response is None:
            error_text = _build_agent_failure(self.name, response_model, attempted, last_error)
            self.status = "IDLE"
            self.last_error = error_text
            self.last_summary = error_text[:320]
            self._record_run(input_payload, False, {}, error_text)
            raise RuntimeError(error_text) from last_error

        served_model = getattr(response, "model", None) or self.last_requested_model
        provider = getattr(response, "provider", None) or ""
        self.model = str(served_model)
        self.last_served_provider = str(provider)
        self.last_run = getattr(response, "created", None)
        self.last_summary = _summarize(parsed)
        self.last_error = ""
        self.status = "IDLE"
        self._record_run(input_payload, True, parsed.model_dump())
        return parsed

    def _record_run(self, input_payload: dict[str, Any], ok: bool, output: dict[str, Any],
                    error: str | None = None) -> None:
        # Telemetry persistence must never turn a valid agent result into a failed
        # agent call. Trading state is authoritative; logging is best-effort.
        if not self.db:
            return
        try:
            self.db.agent_run(
                self.name,
                input_payload.get("symbol"),
                self.model,
                ok,
                input_payload,
                output,
                error,
            )
        except Exception as exc:
            logger.warning("%s agent_run telemetry failed: %s", self.name, _compact_error(exc))


def _summarize(model: BaseModel) -> str:
    data = model.model_dump()
    for key in ("summary", "thesis", "reason"):
        if data.get(key):
            return str(data[key])[:320]
    return json.dumps(data, default=str)[:320]


def _response_diagnostics(response: Any) -> str:
    """Return compact, non-sensitive diagnostics for malformed provider responses."""
    parts: list[str] = []
    for attr in ("id", "object", "model"):
        value = getattr(response, attr, None)
        if value is not None:
            parts.append(f"{attr}={value}")
    usage = getattr(response, "usage", None)
    if usage is not None:
        try:
            data = usage.model_dump() if hasattr(usage, "model_dump") else usage
            if isinstance(data, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if key in data:
                        parts.append(f"{key}={data[key]}")
        except Exception:
            pass
    return ", ".join(parts) or "response metadata unavailable"


def _extract_message_content(response: Any) -> str:
    """Extract the JSON payload from an OpenRouter Chat Completions response."""
    try:
        choices = getattr(response, "choices", None) or []
        if not choices:
            details = _response_diagnostics(response)
            raise ValueError(f"provider returned no choices ({details})")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("provider response contained no message")

        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(f"provider refused the request: {str(refusal)[:240]}")

        tool_calls = getattr(message, "tool_calls", None) or []
        for tool_call in tool_calls:
            function = getattr(tool_call, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
            if name == "submit_response" and arguments is not None:
                text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
                text = text.strip()
                if text:
                    return text

        # Some OpenAI-compatible gateways return the tool payload in a dict-like form.
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                function = tool_call.get("function") or {}
                if function.get("name") == "submit_response" and function.get("arguments") is not None:
                    text = function["arguments"]
                    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
                    if text.strip():
                        return text.strip()

        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise ValueError("provider returned neither a submit_response tool call nor message content")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"unable to read provider response: {exc}") from exc


def _parse_agent_response(response: Any, response_model: Type[BaseModel]) -> BaseModel:
    """Extract and strictly validate the agent's structured response.

    Nemotron is instructed to answer through the ``submit_response`` tool, but
    some OpenAI-compatible gateways may still return plain content. Accept a
    small set of safe JSON wrappers so transient formatting differences do not
    break an otherwise valid response. The final authority remains Pydantic.
    """
    raw = _extract_message_content(response)
    if not raw or not raw.strip():
        raise ValueError("provider returned empty message content")

    text = raw.strip()
    candidates = [text]

    # Strip fenced JSON when a gateway/model wraps the object in Markdown.
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            body = "\n".join(lines[1:-1]).strip()
            candidates.insert(0, body)

    # Find the first complete JSON object/array in prefixed text.
    for start_char in ("{", "["):
        idx = text.find(start_char)
        if idx > 0:
            candidates.append(text[idx:])

    last_error: Exception | None = None
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                raise ValueError("provider JSON root must be an object")
            return response_model.model_validate(payload)
        except Exception as exc:
            last_error = exc

    detail = str(last_error) if last_error else "unknown JSON validation error"
    if isinstance(last_error, json.JSONDecodeError):
        raise ValueError(f"provider returned invalid JSON: {detail}") from last_error
    raise ValueError(
        f"provider JSON failed {response_model.__name__} validation: {detail}"
    ) from last_error


def _is_transient_structured_response_error(exc: Exception) -> bool:
    """Identify empty/malformed structured output that is safe to retry once."""
    text = _compact_error(exc).lower()
    transient_markers = (
        "provider returned no choices",
        "provider returned invalid json",
        "provider returned empty message content",
        "provider returned neither a submit_response tool call nor message content",
        "provider response contained no message",
    )
    return any(marker in text for marker in transient_markers)


def _is_openrouter_daily_quota_error(exc: Exception) -> bool:
    """Return True only for OpenRouter's account-level free-model daily quota error."""
    status = getattr(exc, "status_code", None)
    text = _compact_error(exc).lower()
    return status == 429 and "free-models-per-day" in text


def _compact_error(exc: Exception | None) -> str:
    if exc is None:
        return "unknown failure"
    text = str(exc).replace("\n", " ").strip()
    return text[:500] or exc.__class__.__name__


def _build_agent_failure(
        agent_name: str,
        response_model: Type[BaseModel],
        attempted: list[str],
        last_error: Exception | None,
) -> str:
    attempts = ", ".join(attempted) if attempted else "none"
    detail = _compact_error(last_error)
    return (
        f"{agent_name} failed to produce a valid {response_model.__name__} response. "
        f"Attempts: {attempts}. Last error: {detail}"
    )
