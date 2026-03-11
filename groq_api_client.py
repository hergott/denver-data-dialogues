"""
Simplified Groq API Client — OpenAI Responses API with Structured Outputs

Async client for calling OpenAI LLM models hosted on Groq.
Uses ``AsyncOpenAI`` with ``DefaultAioHttpClient`` and
``responses.parse`` with ``text_format`` for Pydantic structured outputs.

Two levels of validation failure are handled identically via retry:
    1. API-level  — Groq rejects during server-side schema validation.
    2. Local-level — ``output_parsed`` is None or local Pydantic validation fails.

Usage:
    from openai import AsyncOpenAI, DefaultAioHttpClient
    from groq_api_client import GroqAPIClient

    openai_client = AsyncOpenAI(
        api_key="gsk_...",
        base_url="https://api.groq.com/openai/v1",
        max_retries=0,
        http_client=DefaultAioHttpClient(),
    )
    groq = GroqAPIClient(openai_client)
    result = await groq.call(
        developer_instructions="You are a helpful assistant.",
        prompt="Analyze this text...",
        response_model=MyPydanticModel,
    )
"""

import asyncio
import logging
from typing import Any, TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError as PydanticValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

VALID_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
MAX_ATTEMPTS = 6
RETRY_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 180.0
MAX_OUTPUT_TOKENS = 12000


class GroqAPIClient:
    """
    Async structured-output client for Groq via ``responses.parse``
    using ``AsyncOpenAI`` with ``DefaultAioHttpClient``.

    All errors are treated identically: log, sleep one second, retry.
    On retries the full error and raw output are appended to the original
    prompt so the model can self-correct.
    """

    def __init__(self, client: AsyncOpenAI):
        super().__init__()
        self.client = client
        logger.info("GroqAPIClient initialized")

    async def call(
        self,
        developer_instructions: str,
        prompt: str,
        response_model: type[T],
        model: str = "openai/gpt-oss-20b",
    ) -> T:
        """
        Call Groq API and return a validated Pydantic object.

        Args:
            developer_instructions: System-level instructions.
            prompt: The user prompt.
            response_model: Pydantic class for the expected output schema.
            model: ``"openai/gpt-oss-20b"`` (default) or ``"openai/gpt-oss-120b"``.

        Returns:
            An instance of *response_model*.

        Raises:
            ValueError: If *model* is not one of the two supported models.
            RuntimeError: If all attempts are exhausted.
        """
        if model not in VALID_MODELS:
            raise ValueError(f"model must be one of {VALID_MODELS}, got {model!r}")

        schema_name = response_model.__name__
        accumulated_errors: list[str] = []
        last_exception: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            response: Any | None = None
            current_prompt = _augment_prompt(prompt, accumulated_errors, schema_name) if accumulated_errors else prompt

            logger.info(
                "Attempt %d/%d for %s (model=%s, prompt_len=%d)",
                attempt,
                MAX_ATTEMPTS,
                schema_name,
                model,
                len(current_prompt),
            )

            try:
                response = await self.client.responses.parse(
                    model=model,
                    input=[
                        {"role": "system", "content": developer_instructions},
                        {"role": "user", "content": current_prompt},
                    ],
                    text_format=response_model,
                    temperature=0.1,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )

                refusal = _extract_refusal(response)
                if refusal:
                    raise RuntimeError(f"Model refused: {refusal}")

                parsed: T | None = response.output_parsed
                if parsed is None:
                    raw = _extract_raw_text(response)
                    raise RuntimeError(f"output_parsed is None for {schema_name}.\n" + f"Raw output ({len(raw)} chars):\n{raw}")

                logger.info(
                    "Success on attempt %d/%d for %s",
                    attempt,
                    MAX_ATTEMPTS,
                    schema_name,
                )

                logger.info(f"\n\nParsed output: {str(parsed)}\n\n")
                return parsed

            except (openai.APITimeoutError, openai.APIConnectionError) as exc:
                last_exception = exc
                logger.warning(
                    "Transient API transport error on attempt %d/%d for %s: %s: %s",
                    attempt,
                    MAX_ATTEMPTS,
                    schema_name,
                    type(exc).__name__,
                    exc,
                )

                entry = f"[Attempt {attempt} — {type(exc).__name__}]\n{_format_error(exc)}"
                accumulated_errors.append(entry)

                if attempt < MAX_ATTEMPTS:
                    logger.info("Sleeping %.1fs before retry…", RETRY_DELAY_SECONDS)
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                continue

            except Exception as exc:
                last_exception = exc
                error_detail = _format_error(exc)

                raw_output = ""
                try:
                    if response is not None:
                        raw_output = _extract_raw_text(response)
                except Exception:
                    pass

                logger.error(
                    "Attempt %d/%d FAILED for %s.\n  Type : %s\n  Error: %s\n  Raw  : %s",
                    attempt,
                    MAX_ATTEMPTS,
                    schema_name,
                    type(exc).__name__,
                    error_detail,
                    raw_output[:2000] if raw_output else "(unavailable)",
                )

                entry = f"[Attempt {attempt} — {type(exc).__name__}]\n{error_detail}"
                if raw_output:
                    entry += f"\n\n[Raw output from attempt {attempt}]:\n{raw_output}"
                accumulated_errors.append(entry)

                if attempt < MAX_ATTEMPTS:
                    logger.info("Sleeping %.1fs before retry…", RETRY_DELAY_SECONDS)
                    await asyncio.sleep(RETRY_DELAY_SECONDS)

        raise RuntimeError(f"All {MAX_ATTEMPTS} attempts failed for {schema_name}. Last error: {last_exception}")

    async def close(self) -> None:
        """Close the underlying async HTTP client."""
        await self.client.close()


# ======================================================================
# Module-level helpers (no state needed → plain functions, not methods)
# ======================================================================


def _extract_refusal(response: Any) -> str | None:
    """Return the refusal string if the model refused, else None."""
    try:
        for item in getattr(response, "output", []):
            for block in getattr(item, "content", []):
                if getattr(block, "type", None) == "refusal":
                    return getattr(block, "refusal", "")
    except Exception:
        pass
    return None


def _extract_raw_text(response: Any) -> str:
    """Best-effort extraction of the raw text the model produced."""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    try:
        parts: list[str] = []
        for item in getattr(response, "output", []):
            for block in getattr(item, "content", []):
                t = getattr(block, "text", None)
                if isinstance(t, str) and t:
                    parts.append(t)
        if parts:
            return "\n".join(parts)
    except Exception:
        pass
    try:
        return str(response)
    except Exception:
        return ""


def _format_error(exc: Exception) -> str:
    """Return a thorough string representation of any exception."""
    parts: list[str] = [f"{type(exc).__name__}: {exc}"]
    if isinstance(exc, PydanticValidationError):
        for err in exc.errors():
            parts.append(f"  • loc={err.get('loc')}  type={err.get('type')}  msg={err.get('msg')}")
    body = getattr(exc, "body", None) or getattr(exc, "response", None)
    if body is not None:
        parts.append(f"  Response body: {body}")
    return "\n".join(parts)


def _augment_prompt(original: str, errors: list[str], schema_name: str) -> str:
    """Append the full history of errors to the original prompt."""
    sep = "=" * 70
    error_block = "\n\n".join(errors)

    return_str: str = f"{original}\n\n{sep}\nPREVIOUS ATTEMPTS FAILED — FIX THE ERRORS BELOW\n{sep}\n\n{error_block}\n\n{sep}\nProduce valid JSON conforming exactly to {schema_name}.\nALL fields required. Correct data types. Start with {{ — no markdown.\n{sep}\n"
    return return_str
