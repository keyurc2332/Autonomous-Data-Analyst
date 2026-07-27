"""Provider-agnostic LLM factory.

Every agent must obtain its model through get_llm(). Nothing else in the
codebase should import a provider-specific class. That is what makes
switching Google -> Groq -> Ollama a single env-var change.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import LLMProvider, settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)


class LLMConfigError(RuntimeError):
    """Raised when the selected provider is missing required configuration."""


def _build_google() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.GOOGLE_API_KEY:
        raise LLMConfigError(
            "LLM_PROVIDER=google but GOOGLE_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return ChatGoogleGenerativeAI(
        model=settings.GOOGLE_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def _build_groq() -> BaseChatModel:
    from langchain_groq import ChatGroq

    if not settings.GROQ_API_KEY:
        raise LLMConfigError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com/keys"
        )
    return ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def _build_ollama() -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
    )


_BUILDERS = {
    "google": _build_google,
    "groq": _build_groq,
    "ollama": _build_ollama,
}


@lru_cache
def get_llm(provider: LLMProvider | None = None) -> BaseChatModel:
    """Return a chat model for `provider`, defaulting to settings.LLM_PROVIDER.

    Cached: building a client per agent invocation wastes connections.
    Pass an explicit provider to override, e.g. a cheap model for the
    Planner and a stronger one for the Reflection agent.
    """
    chosen: LLMProvider = provider or settings.LLM_PROVIDER
    builder = _BUILDERS.get(chosen)
    if builder is None:
        raise LLMConfigError(
            f"Unknown LLM_PROVIDER {chosen!r}. Expected one of {sorted(_BUILDERS)}."
        )
    logger.info("Initialising LLM", extra={"provider": chosen})
    return builder()


def configure_tracing() -> None:
    """Wire LangSmith. Called once at application startup.

    LangChain reads these from the process environment, so we set them
    rather than passing them around.
    """
    import os

    if not settings.LANGSMITH_TRACING:
        os.environ["LANGSMITH_TRACING"] = "false"
        return
    if not settings.LANGSMITH_API_KEY:
        logger.warning("LANGSMITH_TRACING is on but no API key is set; disabling.")
        os.environ["LANGSMITH_TRACING"] = "false"
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    logger.info("LangSmith tracing enabled", extra={"project": settings.LANGSMITH_PROJECT})


def message_text(response: object) -> str:
    """Extract plain text from a chat response.

    Providers do not agree on the shape of `.content`. Older models return a
    string; Gemini 3.x returns a list of content blocks, each a dict like
    {"type": "text", "text": "...", "extras": {"signature": "<base64>"}}.
    Calling str() on that list dumps the signature blob into user-facing
    output, so blocks must be walked and their text concatenated.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            # "text" blocks carry the prose; tool_use/thinking blocks do not.
            elif (
                isinstance(block, dict)
                and block.get("type") in (None, "text")
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
        if parts:
            return "".join(parts).strip()

    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"].strip()

    return str(content).strip()
