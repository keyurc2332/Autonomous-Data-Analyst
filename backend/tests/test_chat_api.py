"""Chat response serialisation.

The bug this exists for produced a 500 on every chat message and nothing else:
the endpoint worked, the agent worked, the row was written -- only the response
model failed, so the conversation was invisible.
"""
import datetime
import json
import uuid

import pytest

from app.api.routes.chat import ChatMessage
from app.db.models import ConversationMessage, MessageRole


def _message(**kwargs) -> ConversationMessage:
    msg = ConversationMessage(
        id=uuid.uuid4(), project_id=uuid.uuid4(),
        role=kwargs.pop("role", MessageRole.ASSISTANT),
        content=kwargs.pop("content", "text"),
        **kwargs,
    )
    msg.created_at = datetime.datetime.now(datetime.UTC)
    return msg


def test_metadata_reads_the_column_not_sqlalchemys_registry():
    """Regression: `alias="metadata"` is used for reading as well as writing,
    so Pydantic fetched `msg.metadata` -- the MetaData registry present on every
    declarative model -- and every response failed with "Input should be a
    valid dictionary"."""
    tools = [{"tool": "Aggregate", "arguments": {"group_by": "sex"}}]
    validated = ChatMessage.model_validate(_message(metadata_={"tools": tools}))
    assert validated.metadata == {"tools": tools}


def test_metadata_is_named_metadata_in_json():
    """The frontend reads `m.metadata.tools` to render tool attribution."""
    payload = json.loads(
        ChatMessage.model_validate(
            _message(metadata_={"tools": [{"tool": "CountRows"}]})
        ).model_dump_json()
    )
    assert "metadata" in payload
    assert "metadata_" not in payload
    assert payload["metadata"]["tools"][0]["tool"] == "CountRows"


def test_message_without_metadata_serialises():
    """User messages carry none, and must not fail."""
    payload = json.loads(
        ChatMessage.model_validate(
            _message(role=MessageRole.USER, content="hi")
        ).model_dump_json()
    )
    assert payload["metadata"] is None
    assert payload["role"] == "user"


@pytest.mark.parametrize("role", list(MessageRole))
def test_every_role_serialises(role):
    ChatMessage.model_validate(_message(role=role))
