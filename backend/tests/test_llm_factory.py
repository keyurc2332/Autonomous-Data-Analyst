import pytest

from app.core.llm import LLMConfigError, get_llm


def test_unknown_provider_raises():
    get_llm.cache_clear()
    with pytest.raises(LLMConfigError, match="Unknown LLM_PROVIDER"):
        get_llm("anthropic-typo")  # type: ignore[arg-type]
