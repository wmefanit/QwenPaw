# -*- coding: utf-8 -*-
"""Unit tests for console batch send (multiple user messages per request).

Covers the pi-mono model: several queued messages are submitted as
independent user messages in one request instead of being concatenated
into a single message.
"""
# pylint: disable=protected-access,redefined-outer-name,unused-argument
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qwenpaw.app.routers.console import _extract_session_and_payload
from qwenpaw.schemas import AgentRequest, Message, Role, TextContent


@pytest.fixture
def console_channel(tmp_path):
    from qwenpaw.app.channels.console.channel import ConsoleChannel

    return ConsoleChannel(
        process=MagicMock(),
        enabled=True,
        bot_prefix="",
        media_dir=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# _extract_session_and_payload — message grouping
# ---------------------------------------------------------------------------


class TestExtractMessageGroups:
    def test_single_message_one_group(self):
        payload = _extract_session_and_payload(
            AgentRequest(
                input=[
                    Message(
                        role=Role.USER,
                        content=[TextContent(text="hello")],
                    ),
                ],
            ),
        )

        assert len(payload["message_groups"]) == 1
        assert payload["message_groups"][0]["content"][0].text == "hello"
        assert len(payload["content_parts"]) == 1

    def test_two_messages_two_groups(self):
        payload = _extract_session_and_payload(
            AgentRequest(
                input=[
                    Message(
                        role=Role.USER,
                        content=[TextContent(text="first")],
                    ),
                    Message(
                        role=Role.USER,
                        content=[TextContent(text="second")],
                    ),
                ],
            ),
        )

        assert len(payload["message_groups"]) == 2
        assert payload["message_groups"][0]["content"][0].text == "first"
        assert payload["message_groups"][1]["content"][0].text == "second"
        # Flattened view stays available for title/placeholder logic.
        assert len(payload["content_parts"]) == 2

    def test_per_message_metadata_preserved(self):
        payload = _extract_session_and_payload(
            AgentRequest(
                input=[
                    Message(
                        role=Role.USER,
                        content=[TextContent(text="a")],
                        metadata={"qwenpaw_client_message_id": "id-a"},
                    ),
                    Message(
                        role=Role.USER,
                        content=[TextContent(text="b")],
                        metadata={"qwenpaw_client_message_id": "id-b"},
                    ),
                ],
            ),
        )

        groups = payload["message_groups"]
        assert groups[0]["metadata"] == {
            "qwenpaw_client_message_id": "id-a",
        }
        assert groups[1]["metadata"] == {
            "qwenpaw_client_message_id": "id-b",
        }

    def test_empty_input_yields_no_groups(self):
        payload = _extract_session_and_payload(AgentRequest(input=[]))

        assert payload["message_groups"] == []
        assert payload["content_parts"] == []

    def test_dict_payload_two_groups(self):
        payload = _extract_session_and_payload(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "x"}],
                        "metadata": {"qwenpaw_client_message_id": "x-id"},
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "y"}],
                    },
                ],
                "session_id": "s1",
                "user_id": "u1",
            },
        )

        assert len(payload["message_groups"]) == 2
        assert payload["message_groups"][0]["metadata"] == {
            "qwenpaw_client_message_id": "x-id",
        }


# ---------------------------------------------------------------------------
# build_agent_request_from_native — multi-message request
# ---------------------------------------------------------------------------


class TestBuildMultiMessageRequest:
    def test_two_groups_build_two_messages(self, console_channel):
        payload = {
            "sender_id": "u1",
            "content_parts": [TextContent(text="a"), TextContent(text="b")],
            "message_groups": [
                {"content": [TextContent(text="a")], "metadata": None},
                {"content": [TextContent(text="b")], "metadata": None},
            ],
            "meta": {},
        }

        req = console_channel.build_agent_request_from_native(payload)

        assert len(req.input) == 2
        assert req.input[0].content[0].text == "a"
        assert req.input[1].content[0].text == "b"
        assert req.input[0].role == Role.USER
        assert req.input[1].role == Role.USER

    def test_single_group_keeps_legacy_path(self, console_channel):
        payload = {
            "sender_id": "u1",
            "content_parts": [TextContent(text="solo")],
            "message_groups": [
                {
                    "content": [TextContent(text="solo")],
                    "metadata": {"qwenpaw_client_message_id": "solo-id"},
                },
            ],
            "meta": {},
        }

        req = console_channel.build_agent_request_from_native(payload)

        assert len(req.input) == 1
        assert req.input[0].content[0].text == "solo"
        assert req.input[0].metadata == {
            "qwenpaw_client_message_id": "solo-id",
        }

    def test_legacy_message_metadata_used_when_no_groups(
        self,
        console_channel,
    ):
        payload = {
            "sender_id": "u1",
            "content_parts": [TextContent(text="solo")],
            "message_metadata": {"qwenpaw_client_message_id": "solo-id"},
            "meta": {},
        }

        req = console_channel.build_agent_request_from_native(payload)

        assert len(req.input) == 1
        assert req.input[0].metadata == {
            "qwenpaw_client_message_id": "solo-id",
        }

    def test_no_message_groups_keeps_legacy_path(self, console_channel):
        payload = {
            "sender_id": "u1",
            "content_parts": [TextContent(text="legacy")],
            "meta": {},
        }

        req = console_channel.build_agent_request_from_native(payload)

        assert len(req.input) == 1
        assert req.input[0].content[0].text == "legacy"

    def test_per_message_metadata_lands_on_each_message(
        self,
        console_channel,
    ):
        payload = {
            "sender_id": "u1",
            "content_parts": [TextContent(text="a"), TextContent(text="b")],
            "message_groups": [
                {
                    "content": [TextContent(text="a")],
                    "metadata": {"qwenpaw_client_message_id": "id-a"},
                },
                {
                    "content": [TextContent(text="b")],
                    "metadata": {"qwenpaw_client_message_id": "id-b"},
                },
            ],
            "meta": {},
        }

        req = console_channel.build_agent_request_from_native(payload)

        assert req.input[0].metadata == {
            "qwenpaw_client_message_id": "id-a",
        }
        assert req.input[1].metadata == {
            "qwenpaw_client_message_id": "id-b",
        }

    def test_empty_group_content_gets_space_placeholder(
        self,
        console_channel,
    ):
        payload = {
            "sender_id": "u1",
            "content_parts": [TextContent(text="a")],
            "message_groups": [
                {"content": [TextContent(text="a")], "metadata": None},
                {"content": [], "metadata": None},
            ],
            "meta": {},
        }

        req = console_channel.build_agent_request_from_native(payload)

        assert len(req.input) == 2
        assert req.input[1].content[0].text == " "

    def test_session_and_channel_carried_over(self, console_channel):
        payload = {
            "channel_id": "console",
            "sender_id": "u1",
            "content_parts": [TextContent(text="a"), TextContent(text="b")],
            "message_groups": [
                {"content": [TextContent(text="a")], "metadata": None},
                {"content": [TextContent(text="b")], "metadata": None},
            ],
            "meta": {"session_id": "console:u1"},
        }

        req = console_channel.build_agent_request_from_native(payload)

        assert req.channel == "console"
        assert req.user_id == "u1"
        assert req.session_id == "console:u1"


# ---------------------------------------------------------------------------
# stream_one — batch send skips the no-text debounce
# ---------------------------------------------------------------------------


def _streaming_channel(tmp_path, received, *, no_text_debounce=False):
    """ConsoleChannel whose process handler records the request it receives."""
    from qwenpaw.app.channels.console.channel import ConsoleChannel
    from qwenpaw.schemas import Event

    async def process(request):
        received.append(request)
        yield Event(object="response", status="completed", output=[])

    channel = ConsoleChannel(
        process=process,
        enabled=True,
        bot_prefix="",
        media_dir=str(tmp_path),
    )
    # __init__ hard-codes False; tests that exercise buffering opt in.
    channel._no_text_debounce = no_text_debounce
    return channel


def _attachment(tmp_path, name="a.txt"):
    from qwenpaw.schemas import FileContent

    return FileContent(
        file_url=str(tmp_path / name),
        file_name=name,
        file_size=1,
    )


class TestStreamOneBatchDebounce:
    @pytest.mark.asyncio
    async def test_multi_group_skips_debounce(self, tmp_path):
        """A confirmed batch is complete input: no-text content runs now."""
        received = []
        channel = _streaming_channel(
            tmp_path,
            received,
            no_text_debounce=True,
        )
        attachment = _attachment(tmp_path)
        payload = {
            "sender_id": "u1",
            "content_parts": [attachment, attachment],
            "message_groups": [
                {"content": [attachment], "metadata": None},
                {"content": [attachment], "metadata": None},
            ],
            "meta": {"session_id": "s1"},
        }

        events = [event async for event in channel.stream_one(payload)]

        assert events
        assert len(received) == 1
        assert len(received[0].input) == 2

    @pytest.mark.asyncio
    async def test_single_group_no_text_still_buffered(self, tmp_path):
        """One no-text message keeps the legacy buffering behaviour."""
        received = []
        channel = _streaming_channel(
            tmp_path,
            received,
            no_text_debounce=True,
        )
        attachment = _attachment(tmp_path)
        payload = {
            "sender_id": "u1",
            "content_parts": [attachment],
            "message_groups": [
                {"content": [attachment], "metadata": None},
            ],
            "meta": {"session_id": "s1"},
        }

        events = [event async for event in channel.stream_one(payload)]

        assert events == []
        assert received == []

    @pytest.mark.asyncio
    async def test_debounce_merge_keeps_client_message_id(self, tmp_path):
        """Buffered parts must not drop the receipt id of the merged turn."""
        from qwenpaw.constant import QWENPAW_CLIENT_MESSAGE_ID_KEY

        received = []
        channel = _streaming_channel(
            tmp_path,
            received,
            no_text_debounce=True,
        )
        attachment = _attachment(tmp_path)
        # First payload carries no text, so it is buffered.
        first = {
            "sender_id": "u1",
            "content_parts": [attachment],
            "meta": {"session_id": "s1"},
        }
        assert [event async for event in channel.stream_one(first)] == []

        # Second payload has text, releasing the buffer.
        second = {
            "sender_id": "u1",
            "content_parts": [TextContent(text="read this")],
            "message_metadata": {
                QWENPAW_CLIENT_MESSAGE_ID_KEY: "client-1",
            },
            "meta": {"session_id": "s1"},
        }
        events = [event async for event in channel.stream_one(second)]

        assert events
        assert len(received) == 1
        message = received[0].input[0]
        assert len(message.content) == 2
        # merged = buffered parts + current parts, so the attachment leads.
        assert message.content[0].file_name == "a.txt"
        assert message.content[1].text == "read this"
        assert message.metadata == {
            QWENPAW_CLIENT_MESSAGE_ID_KEY: "client-1",
        }
