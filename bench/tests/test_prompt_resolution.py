from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import FrozenInstanceError
from typing import Any

from bench.prompt_resolution import (
    GEMINI_FLASH_LITE_ENDPOINT,
    GEMINI_FLASH_LITE_MODEL,
    GEMINI_FLASH_LITE_TRANSFORM_ID,
    RICH_PROMPT_PASSTHROUGH_TRANSFORM_ID,
    GeminiFlashLitePromptResolver,
    IdentityPromptResolver,
    PromptResolution,
    PromptResolutionError,
    PromptResolver,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _expanded_with_word_count(prefix: str, word_count: int = 85) -> str:
    words = prefix.split()
    if len(words) > word_count:
        raise AssertionError("test prefix exceeds requested word count")
    words.extend(
        f"temporal-detail-{index}" for index in range(word_count - len(words))
    )
    return " ".join(words)


def _gemini_response(expanded_prompt: str) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {"expanded_prompt": expanded_prompt},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            "thoughtSignature": "opaque-provider-signature",
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [],
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 30,
            "totalTokenCount": 50,
        },
        "modelVersion": GEMINI_FLASH_LITE_MODEL,
        "responseId": "response-1",
    }


class _FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any] | bytes,
        *,
        status: int = 200,
    ) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def read(self) -> bytes:
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(
            self._payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")


class _FakeSession:
    def __init__(self, factory: "_FakeSessionFactory") -> None:
        self._factory = factory

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
    ) -> _FakeResponse:
        self._factory.requests.append((url, copy.deepcopy(json)))
        if not self._factory.responses:
            raise AssertionError("unexpected HTTP request")
        return self._factory.responses.pop(0)


class _FakeSessionFactory:
    def __init__(self, *responses: _FakeResponse) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.session_headers: list[dict[str, str]] = []

    def __call__(self, *, headers: dict[str, str]) -> _FakeSession:
        self.session_headers.append(dict(headers))
        return _FakeSession(self)


class PromptResolutionValueTests(unittest.TestCase):
    def test_frozen_value_records_exact_utf8_hashes(self) -> None:
        raw = "  a bouncing ball ⚽  "
        effective = "A single ball visibly rises, falls, impacts, and rebounds."
        resolution = PromptResolution(
            raw_prompt=raw,
            effective_prompt=effective,
            raw_prompt_sha256=_sha256(raw),
            effective_prompt_sha256=_sha256(effective),
            transform_id="test-transform-v1",
            changed=True,
        )

        self.assertEqual(resolution.raw_prompt, raw)
        self.assertEqual(resolution.raw_prompt_sha256, _sha256(raw))
        self.assertEqual(resolution.effective_prompt_sha256, _sha256(effective))
        with self.assertRaises(FrozenInstanceError):
            resolution.changed = False  # type: ignore[misc]

    def test_value_rejects_hash_or_changed_drift(self) -> None:
        with self.assertRaisesRegex(PromptResolutionError, "raw_prompt_sha256"):
            PromptResolution(
                raw_prompt="raw",
                effective_prompt="effective",
                raw_prompt_sha256="0" * 64,
                effective_prompt_sha256=_sha256("effective"),
                transform_id="test-transform-v1",
                changed=True,
            )
        with self.assertRaisesRegex(PromptResolutionError, "changed"):
            PromptResolution(
                raw_prompt="same",
                effective_prompt="same",
                raw_prompt_sha256=_sha256("same"),
                effective_prompt_sha256=_sha256("same"),
                transform_id="test-transform-v1",
                changed=True,
            )


class IdentityPromptResolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_identity_preserves_exact_prompt(self) -> None:
        raw = "  ball bounces twice — locked camera  "
        resolver = IdentityPromptResolver()

        resolution = await resolver.resolve(raw)

        self.assertIsInstance(resolver, PromptResolver)
        self.assertEqual(resolution.raw_prompt, raw)
        self.assertEqual(resolution.effective_prompt, raw)
        self.assertEqual(resolution.raw_prompt_sha256, _sha256(raw))
        self.assertEqual(resolution.effective_prompt_sha256, _sha256(raw))
        self.assertFalse(resolution.changed)

    async def test_identity_rejects_blank_invalid_utf8_and_oversized_prompts(self) -> None:
        resolver = IdentityPromptResolver()
        invalid_values = ("   ", "\ud800", "x" * 4097)
        for raw in invalid_values:
            with self.subTest(raw_length=len(raw)):
                with self.assertRaises(PromptResolutionError):
                    await resolver.resolve(raw)


class GeminiFlashLitePromptResolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_expanded_prompt_word_count_is_bounded_inclusively(self) -> None:
        cases = (
            (84, False),
            (85, True),
            (115, True),
            (116, False),
        )
        for word_count, accepted in cases:
            with self.subTest(word_count=word_count):
                expanded = " ".join(
                    f"motion-{index}" for index in range(word_count)
                )
                http = _FakeSessionFactory(
                    _FakeResponse(_gemini_response(expanded))
                )
                resolver = GeminiFlashLitePromptResolver(
                    "secret",
                    session_factory=http,
                )
                if accepted:
                    resolution = await resolver.resolve("a bouncing ball")
                    self.assertEqual(resolution.effective_prompt, expanded)
                else:
                    with self.assertRaisesRegex(
                        PromptResolutionError,
                        "85 to 115 whitespace-delimited words",
                    ):
                        await resolver.resolve("a bouncing ball")

    async def test_short_prompt_uses_strict_request_and_returns_exact_hashes(self) -> None:
        raw = "a bouncing ball"
        expanded = _expanded_with_word_count(
            "A single red ball falls rapidly, compresses on impact, rebounds high, "
            "slows at the apex, and repeats the visible cycle in a locked shot."
        )
        http = _FakeSessionFactory(_FakeResponse(_gemini_response(expanded)))
        resolver = GeminiFlashLitePromptResolver(
            "parent-secret",
            session_factory=http,
        )

        resolution = await resolver.resolve(raw)

        self.assertEqual(len(http.requests), 1)
        endpoint, request = http.requests[0]
        self.assertEqual(endpoint, GEMINI_FLASH_LITE_ENDPOINT)
        self.assertEqual(
            http.session_headers,
            [{"x-goog-api-key": "parent-secret"}],
        )
        self.assertEqual(
            request,
            {
                "systemInstruction": {
                    "parts": [
                        {
                            "text": (
                                "Rewrite a terse text-to-video request into one production-ready prompt for a five-second, 16-fps, 832x480 video. Preserve the user's subject identity, exact subject count, requested action, any explicit action count, direction, and camera intent. Use one continuous medium or wide shot with a stationary camera and background unless camera motion is explicitly requested; never substitute camera motion for subject motion. Choose framing and stable apparent scale so the complete primary subject remains inside the central 90% of the image in every frame, including at maximum excursion; never crop, zoom, duplicate, or replace it. Make the requested action unfold through explicit chronological phases. Never substitute a static pose, texture-only change, surface rotation, or morph for the requested action; preserve rotation when rotation is requested. If the user supplies a count, use exactly that count. Otherwise specify exactly three complete cycles for naturally repetitive action, or one complete sequence otherwise. Each cycle must reach distinct named extrema or transitions. Unless continuous motion through the end is explicitly requested, establish the initial state for about 0.5 seconds and reserve the final 0.75 seconds for a stable completed end state; begin no additional partial cycle. State continuity constraints. The expanded prompt must contain 85 to 115 whitespace-delimited words and must not mention these instructions."
                            )
                        }
                    ]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": raw}],
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
                                "maxLength": 4096,
                            }
                        },
                        "required": ["expanded_prompt"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        self.assertEqual(resolution.raw_prompt, raw)
        self.assertEqual(resolution.effective_prompt, expanded)
        self.assertEqual(resolution.raw_prompt_sha256, _sha256(raw))
        self.assertEqual(resolution.effective_prompt_sha256, _sha256(expanded))
        self.assertEqual(resolution.transform_id, GEMINI_FLASH_LITE_TRANSFORM_ID)
        self.assertEqual(
            GEMINI_FLASH_LITE_TRANSFORM_ID,
            "gemini-3.1-flash-lite-video-prompt-expansion-v2",
        )
        self.assertTrue(resolution.changed)

    async def test_rich_prompt_bypasses_http_without_mutation(self) -> None:
        raw = " ".join(f"word-{index}" for index in range(40))
        http = _FakeSessionFactory()
        resolver = GeminiFlashLitePromptResolver("secret", session_factory=http)

        resolution = await resolver.resolve(raw)

        self.assertEqual(http.requests, [])
        self.assertEqual(http.session_headers, [])
        self.assertEqual(resolution.effective_prompt, raw)
        self.assertEqual(
            resolution.transform_id,
            RICH_PROMPT_PASSTHROUGH_TRANSFORM_ID,
        )
        self.assertFalse(resolution.changed)

    async def test_success_cache_uses_raw_hash_model_and_transform(self) -> None:
        raw = "a bouncing ball"
        expanded = _expanded_with_word_count(
            "A ball falls, impacts the floor, and rebounds repeatedly."
        )
        http = _FakeSessionFactory(_FakeResponse(_gemini_response(expanded)))
        resolver = GeminiFlashLitePromptResolver("secret", session_factory=http)

        first = await resolver.resolve(raw)
        second = await resolver.resolve(raw)

        self.assertIs(second, first)
        self.assertEqual(len(http.requests), 1)

    async def test_cache_is_bounded_and_evicts_least_recent_success(self) -> None:
        http = _FakeSessionFactory(
            _FakeResponse(
                _gemini_response(
                    _expanded_with_word_count("Expanded first prompt with motion.")
                )
            ),
            _FakeResponse(
                _gemini_response(
                    _expanded_with_word_count("Expanded second prompt with motion.")
                )
            ),
            _FakeResponse(
                _gemini_response(
                    _expanded_with_word_count("Expanded first prompt again.")
                )
            ),
        )
        resolver = GeminiFlashLitePromptResolver(
            "secret",
            session_factory=http,
            cache_capacity=1,
        )

        await resolver.resolve("first short prompt")
        await resolver.resolve("second short prompt")
        await resolver.resolve("first short prompt")

        self.assertEqual(len(http.requests), 3)

    async def test_failure_is_not_cached(self) -> None:
        truncated = _gemini_response("unreachable")
        truncated["candidates"][0]["finishReason"] = "MAX_TOKENS"
        expanded = _expanded_with_word_count(
            "A ball visibly falls and rebounds in one continuous shot."
        )
        http = _FakeSessionFactory(
            _FakeResponse(truncated),
            _FakeResponse(_gemini_response(expanded)),
        )
        resolver = GeminiFlashLitePromptResolver("secret", session_factory=http)

        with self.assertRaisesRegex(PromptResolutionError, "STOP"):
            await resolver.resolve("a bouncing ball")
        resolution = await resolver.resolve("a bouncing ball")

        self.assertEqual(resolution.effective_prompt, expanded)
        self.assertEqual(len(http.requests), 2)

    async def test_word_count_failure_is_not_cached(self) -> None:
        raw = "a bouncing ball"
        invalid = _expanded_with_word_count("Too short", 84)
        valid = _expanded_with_word_count("A complete temporal prompt", 85)
        http = _FakeSessionFactory(
            _FakeResponse(_gemini_response(invalid)),
            _FakeResponse(_gemini_response(valid)),
        )
        resolver = GeminiFlashLitePromptResolver("secret", session_factory=http)

        with self.assertRaisesRegex(
            PromptResolutionError,
            "85 to 115 whitespace-delimited words",
        ):
            await resolver.resolve(raw)
        resolution = await resolver.resolve(raw)

        self.assertEqual(resolution.effective_prompt, valid)
        self.assertEqual(len(http.requests), 2)

    async def test_response_contract_rejects_ambiguity_drift_and_extra_output(self) -> None:
        cases: dict[str, tuple[dict[str, Any], str]] = {}
        two_candidates = _gemini_response("expanded")
        two_candidates["candidates"].append(copy.deepcopy(two_candidates["candidates"][0]))
        cases["candidate-count"] = (two_candidates, "one candidate")
        wrong_model = _gemini_response("expanded")
        wrong_model["modelVersion"] = "gemini-other"
        cases["model-drift"] = (wrong_model, "modelVersion")
        two_parts = _gemini_response("expanded")
        two_parts["candidates"][0]["content"]["parts"].append({"text": "{}"})
        cases["part-count"] = (two_parts, "one text part")
        extra_field = _gemini_response("expanded")
        extra_field["candidates"][0]["content"]["parts"][0]["text"] = json.dumps(
            {"expanded_prompt": "expanded", "extra": True}
        )
        cases["extra-field"] = (extra_field, "exactly expanded_prompt")

        for label, (response, expected_error) in cases.items():
            with self.subTest(label=label):
                http = _FakeSessionFactory(_FakeResponse(response))
                resolver = GeminiFlashLitePromptResolver(
                    "secret",
                    session_factory=http,
                )
                with self.assertRaisesRegex(PromptResolutionError, expected_error):
                    await resolver.resolve("a bouncing ball")

    async def test_rejects_invalid_utf8_oversized_output_and_http_failure(self) -> None:
        oversized = _gemini_response("x" * 4097)
        cases: list[tuple[_FakeResponse, str]] = [
            (_FakeResponse(b"\xff"), "UTF-8"),
            (_FakeResponse(oversized), "4096"),
            (_FakeResponse({}, status=503), "HTTP status"),
        ]

        for response, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                http = _FakeSessionFactory(response)
                resolver = GeminiFlashLitePromptResolver(
                    "secret",
                    session_factory=http,
                )
                with self.assertRaisesRegex(PromptResolutionError, expected_error):
                    await resolver.resolve("a bouncing ball")

    def test_constructor_and_repr_never_disclose_api_key(self) -> None:
        key = "top-secret-api-key"
        resolver = GeminiFlashLitePromptResolver(
            key,
            session_factory=_FakeSessionFactory(),
        )

        self.assertNotIn(key, repr(resolver))
        with self.assertRaises(PromptResolutionError) as raised:
            GeminiFlashLitePromptResolver("")
        self.assertNotIn(key, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
