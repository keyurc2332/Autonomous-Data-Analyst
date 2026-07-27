"""Response text extraction across provider content shapes."""
from types import SimpleNamespace

import pytest

from app.core.llm import message_text


def _msg(content):
    return SimpleNamespace(content=content)


def test_plain_string():
    assert message_text(_msg("  Hello  ")) == "Hello"


def test_gemini_3_content_blocks():
    """Gemini 3.x wraps text in blocks and attaches a signature blob.

    str() on this list would leak the base64 signature into user output.
    """
    content = [{
        "type": "text",
        "text": "OK",
        "extras": {"signature": "EooDCocDARFNMg80rnJn/tOXDT9Mc/ov48Nn"},
    }]
    assert message_text(_msg(content)) == "OK"
    assert "signature" not in message_text(_msg(content))


def test_multiple_blocks_are_joined():
    content = [
        {"type": "text", "text": "First part. "},
        {"type": "text", "text": "Second part."},
    ]
    assert message_text(_msg(content)) == "First part. Second part."


def test_non_text_blocks_are_skipped():
    content = [
        {"type": "thinking", "thinking": "internal reasoning"},
        {"type": "text", "text": "Visible answer."},
    ]
    assert message_text(_msg(content)) == "Visible answer."


def test_list_of_raw_strings():
    assert message_text(_msg(["a", "b"])) == "ab"


def test_bare_dict():
    assert message_text(_msg({"text": "hi"})) == "hi"


@pytest.mark.parametrize("content", [None, 42, []])
def test_unexpected_shapes_do_not_raise(content):
    assert isinstance(message_text(_msg(content)), str)
