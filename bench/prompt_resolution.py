"""Fail-closed browser prompt resolution.

The Gemini resolver is intended to run in the browser-server parent only.  It
never belongs in the CUDA worker process, and its caller owns the hard request
deadline (normally an ``asyncio.wait_for`` of roughly four seconds).
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, AsyncContextManager, Callable, Protocol, runtime_checkable


MAX_PROMPT_UTF8_BYTES = 4096
RICH_PROMPT_MIN_WORDS = 40
GEMINI_MAX_RESPONSE_BYTES = 64 * 1024
EXPANDED_PROMPT_MIN_WORDS = 85
EXPANDED_PROMPT_MAX_WORDS = 115
GEMINI_FLASH_LITE_MODEL = "gemini-3.1-flash-lite"
GEMINI_FLASH_LITE_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_FLASH_LITE_MODEL}:generateContent"
)
IDENTITY_PROMPT_TRANSFORM_ID = "identity-prompt-v1"
RICH_PROMPT_PASSTHROUGH_TRANSFORM_ID = "rich-prompt-passthrough-v1"
GEMINI_FLASH_LITE_TRANSFORM_ID = (
    "gemini-3.1-flash-lite-video-prompt-expansion-v2"
)

_EXPANSION_SYSTEM_INSTRUCTION = (
    "Rewrite a terse text-to-video request into one production-ready prompt for a "
    "five-second, 16-fps, 832x480 video. Preserve the user's subject identity, exact "
    "subject count, requested action, any explicit action count, direction, and camera "
    "intent. Use one continuous medium or wide shot with a stationary camera and "
    "background unless camera motion is explicitly requested; never substitute camera "
    "motion for subject motion. Choose framing and stable apparent scale so the complete "
    "primary subject remains inside the central 90% of the image in every frame, "
    "including at maximum excursion; never crop, zoom, duplicate, or replace it. Make "
    "the requested action unfold through explicit chronological phases. Never substitute "
    "a static pose, texture-only change, surface rotation, or morph for the requested "
    "action; preserve rotation when rotation is requested. If the user supplies a count, "
    "use exactly that count. Otherwise specify exactly three complete cycles for "
    "naturally repetitive action, or one complete sequence otherwise. Each cycle must "
    "reach distinct named extrema or transitions. Unless continuous motion through the "
    "end is explicitly requested, establish the initial state for about 0.5 seconds and "
    "reserve the final 0.75 seconds for a stable completed end state; begin no additional "
    "partial cycle. State continuity constraints. The expanded prompt must contain 85 to "
    "115 whitespace-delimited words and must not mention these instructions."
)


class PromptResolutionError(ValueError):
    """A prompt or provider response violated the resolution contract."""


def _utf8_prompt(value: object, label: str) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise PromptResolutionError(f"{label} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise PromptResolutionError(f"{label} must be valid UTF-8") from error
    if len(encoded) > MAX_PROMPT_UTF8_BYTES:
        raise PromptResolutionError(
            f"{label} must not exceed {MAX_PROMPT_UTF8_BYTES} UTF-8 bytes"
        )
    return encoded


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PromptResolution:
    """Immutable provenance for the exact raw and effective prompt bytes."""

    raw_prompt: str
    effective_prompt: str
    raw_prompt_sha256: str
    effective_prompt_sha256: str
    transform_id: str
    changed: bool

    def __post_init__(self) -> None:
        raw_bytes = _utf8_prompt(self.raw_prompt, "raw_prompt")
        effective_bytes = _utf8_prompt(self.effective_prompt, "effective_prompt")
        if self.raw_prompt_sha256 != _sha256(raw_bytes):
            raise PromptResolutionError("raw_prompt_sha256 does not match raw_prompt")
        if self.effective_prompt_sha256 != _sha256(effective_bytes):
            raise PromptResolutionError(
                "effective_prompt_sha256 does not match effective_prompt"
            )
        if not isinstance(self.transform_id, str) or not self.transform_id.strip():
            raise PromptResolutionError("transform_id must be a non-empty string")
        if not isinstance(self.changed, bool):
            raise PromptResolutionError("changed must be boolean")
        if self.changed != (self.raw_prompt != self.effective_prompt):
            raise PromptResolutionError("changed does not match the prompt values")


@runtime_checkable
class PromptResolver(Protocol):
    async def resolve(self, raw_prompt: str) -> PromptResolution:
        """Resolve one exact raw prompt into an effective prompt."""


def _resolution(raw_prompt: str, effective_prompt: str, transform_id: str) -> PromptResolution:
    raw_bytes = _utf8_prompt(raw_prompt, "raw_prompt")
    effective_bytes = _utf8_prompt(effective_prompt, "effective_prompt")
    return PromptResolution(
        raw_prompt=raw_prompt,
        effective_prompt=effective_prompt,
        raw_prompt_sha256=_sha256(raw_bytes),
        effective_prompt_sha256=_sha256(effective_bytes),
        transform_id=transform_id,
        changed=raw_prompt != effective_prompt,
    )


class IdentityPromptResolver:
    async def resolve(self, raw_prompt: str) -> PromptResolution:
        return _resolution(
            raw_prompt,
            raw_prompt,
            IDENTITY_PROMPT_TRANSFORM_ID,
        )


class _ResponseLike(Protocol):
    status: int

    async def read(self) -> bytes:
        ...


class _SessionLike(Protocol):
    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
    ) -> AsyncContextManager[_ResponseLike]:
        ...


SessionFactory = Callable[..., AsyncContextManager[_SessionLike]]


def _default_session_factory(*, headers: dict[str, str]) -> Any:
    # Keep aiohttp out of import-time CUDA/worker surfaces. The browser parent is
    # the only process that should instantiate this resolver.
    try:
        import aiohttp
    except ImportError as error:  # pragma: no cover - deployment dependency guard
        raise PromptResolutionError("aiohttp is required for Gemini resolution") from error
    return aiohttp.ClientSession(headers=headers)


def _request_body(raw_prompt: str) -> dict[str, Any]:
    return {
        "systemInstruction": {
            "parts": [{"text": _EXPANSION_SYSTEM_INSTRUCTION}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": raw_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 220,
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "object",
                "properties": {
                    "expanded_prompt": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_PROMPT_UTF8_BYTES,
                    }
                },
                "required": ["expanded_prompt"],
                "additionalProperties": False,
            },
        },
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PromptResolutionError("Gemini response contains duplicate JSON keys")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise PromptResolutionError(f"Gemini response contains invalid JSON constant {value}")


def _load_json_object(encoded: bytes, label: str) -> dict[str, Any]:
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PromptResolutionError(f"{label} must be valid UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except PromptResolutionError:
        raise
    except (json.JSONDecodeError, RecursionError, UnicodeError) as error:
        raise PromptResolutionError(f"{label} must be valid JSON") from error
    if not isinstance(value, dict):
        raise PromptResolutionError(f"{label} must be a JSON object")
    return value


def _require_fields(
    value: object,
    *,
    required: set[str],
    allowed: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromptResolutionError(f"{label} must be an object")
    fields = set(value)
    if not required.issubset(fields) or not fields.issubset(allowed):
        raise PromptResolutionError(f"{label} fields violate the response contract")
    return value


def _reject_blocked_ratings(value: object, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise PromptResolutionError(f"{label} must be an array")
    for rating in value:
        if not isinstance(rating, dict):
            raise PromptResolutionError(f"{label} entries must be objects")
        if rating.get("blocked") is True:
            raise PromptResolutionError("Gemini prompt expansion was blocked")


def _expanded_prompt(response_bytes: bytes) -> str:
    if not response_bytes:
        raise PromptResolutionError("Gemini response must not be empty")
    if len(response_bytes) > GEMINI_MAX_RESPONSE_BYTES:
        raise PromptResolutionError("Gemini response exceeds the byte limit")
    response = _load_json_object(response_bytes, "Gemini response")
    response = _require_fields(
        response,
        required={"candidates", "usageMetadata", "modelVersion", "responseId"},
        allowed={
            "candidates",
            "usageMetadata",
            "modelVersion",
            "responseId",
            "promptFeedback",
        },
        label="Gemini response",
    )
    if response["modelVersion"] != GEMINI_FLASH_LITE_MODEL:
        raise PromptResolutionError("Gemini response modelVersion mismatch")
    if not isinstance(response["responseId"], str) or not response["responseId"]:
        raise PromptResolutionError("Gemini responseId is required")
    if not isinstance(response["usageMetadata"], dict):
        raise PromptResolutionError("Gemini usageMetadata must be an object")

    prompt_feedback = response.get("promptFeedback")
    if prompt_feedback is not None:
        prompt_feedback = _require_fields(
            prompt_feedback,
            required=set(),
            allowed={"blockReason", "safetyRatings"},
            label="Gemini promptFeedback",
        )
        if prompt_feedback.get("blockReason"):
            raise PromptResolutionError("Gemini prompt expansion was blocked")
        _reject_blocked_ratings(
            prompt_feedback.get("safetyRatings"),
            "Gemini promptFeedback safetyRatings",
        )

    candidates = response["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise PromptResolutionError("Gemini response must contain exactly one candidate")
    candidate = _require_fields(
        candidates[0],
        required={"content", "finishReason", "index"},
        allowed={"content", "finishReason", "index", "safetyRatings"},
        label="Gemini candidate",
    )
    if candidate["index"] != 0:
        raise PromptResolutionError("Gemini candidate index must equal zero")
    if candidate["finishReason"] != "STOP":
        raise PromptResolutionError("Gemini response finishReason must equal STOP")
    _reject_blocked_ratings(
        candidate.get("safetyRatings"),
        "Gemini candidate safetyRatings",
    )

    content = _require_fields(
        candidate["content"],
        required={"parts", "role"},
        allowed={"parts", "role"},
        label="Gemini candidate content",
    )
    if content["role"] != "model":
        raise PromptResolutionError("Gemini candidate role must equal model")
    parts = content["parts"]
    if not isinstance(parts, list) or len(parts) != 1:
        raise PromptResolutionError("Gemini response must contain exactly one text part")
    part = _require_fields(
        parts[0],
        required={"text"},
        allowed={"text", "thoughtSignature"},
        label="Gemini candidate text part",
    )
    thought_signature = part.get("thoughtSignature")
    if thought_signature is not None and (
        not isinstance(thought_signature, str) or not thought_signature
    ):
        raise PromptResolutionError("Gemini thoughtSignature is invalid")
    if not isinstance(part["text"], str):
        raise PromptResolutionError("Gemini candidate text part must contain text")
    try:
        structured_bytes = part["text"].encode("utf-8")
    except UnicodeEncodeError as error:
        raise PromptResolutionError(
            "Gemini structured response must be valid UTF-8"
        ) from error
    structured = _load_json_object(structured_bytes, "Gemini structured response")
    if set(structured) != {"expanded_prompt"}:
        raise PromptResolutionError(
            "Gemini structured response must contain exactly expanded_prompt"
        )
    expanded_prompt = structured["expanded_prompt"]
    _utf8_prompt(expanded_prompt, "Gemini expanded_prompt")
    word_count = len(expanded_prompt.split())
    if not EXPANDED_PROMPT_MIN_WORDS <= word_count <= EXPANDED_PROMPT_MAX_WORDS:
        raise PromptResolutionError(
            "Gemini expanded_prompt must contain 85 to 115 whitespace-delimited words"
        )
    return expanded_prompt


class GeminiFlashLitePromptResolver:
    """Resolve short prompts through the stable Gemini 3.1 Flash-Lite model.

    The API key is retained only in this parent-owned object and sent in an
    HTTP header. It is deliberately absent from ``repr`` and every error.
    Successful provider resolutions are cached; failures are never inserted.
    """

    __slots__ = ("_api_key", "_session_factory", "_cache_capacity", "_cache")

    def __init__(
        self,
        api_key: str,
        *,
        session_factory: SessionFactory | None = None,
        cache_capacity: int = 128,
    ) -> None:
        if (
            not isinstance(api_key, str)
            or not api_key
            or len(api_key) > 512
            or any(ord(character) < 33 or ord(character) > 126 for character in api_key)
        ):
            raise PromptResolutionError("api_key must be a non-empty ASCII credential")
        if (
            isinstance(cache_capacity, bool)
            or not isinstance(cache_capacity, int)
            or not 1 <= cache_capacity <= 1024
        ):
            raise PromptResolutionError("cache_capacity must be an integer in [1, 1024]")
        if session_factory is not None and not callable(session_factory):
            raise PromptResolutionError("session_factory must be callable")
        self._api_key = api_key
        self._session_factory = session_factory or _default_session_factory
        self._cache_capacity = cache_capacity
        self._cache: OrderedDict[
            tuple[str, str, str], PromptResolution
        ] = OrderedDict()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={GEMINI_FLASH_LITE_MODEL!r}, "
            f"cache_entries={len(self._cache)})"
        )

    async def resolve(self, raw_prompt: str) -> PromptResolution:
        raw_bytes = _utf8_prompt(raw_prompt, "raw_prompt")
        if len(raw_prompt.split()) >= RICH_PROMPT_MIN_WORDS:
            return _resolution(
                raw_prompt,
                raw_prompt,
                RICH_PROMPT_PASSTHROUGH_TRANSFORM_ID,
            )

        raw_sha256 = _sha256(raw_bytes)
        cache_key = (
            raw_sha256,
            GEMINI_FLASH_LITE_MODEL,
            GEMINI_FLASH_LITE_TRANSFORM_ID,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        try:
            async with self._session_factory(
                headers={"x-goog-api-key": self._api_key}
            ) as session:
                async with session.post(
                    GEMINI_FLASH_LITE_ENDPOINT,
                    json=_request_body(raw_prompt),
                ) as response:
                    if response.status != 200:
                        raise PromptResolutionError(
                            "Gemini prompt expansion returned a non-success HTTP status"
                        )
                    response_bytes = await response.read()
        except PromptResolutionError:
            raise
        except Exception as error:
            raise PromptResolutionError("Gemini prompt expansion request failed") from error

        effective_prompt = _expanded_prompt(response_bytes)
        resolved = _resolution(
            raw_prompt,
            effective_prompt,
            GEMINI_FLASH_LITE_TRANSFORM_ID,
        )
        self._cache[cache_key] = resolved
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)
        return resolved
