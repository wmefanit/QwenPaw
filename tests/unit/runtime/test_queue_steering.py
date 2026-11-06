"""Safe-boundary tests for console queue steering."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from agentscope.event import ReplyEndEvent, ReplyFinishedReason
from agentscope.message import Msg, TextBlock
from agentscope.state import ReplyContext

from qwenpaw.app.task_tracker import TaskTracker
from qwenpaw.runtime.queue_steering import QueueSteeringMiddleware


class _FakeState:
    def __init__(self, *, awaiting: bool = False) -> None:
        self.awaiting = awaiting
        self.reply_context = ReplyContext(cur_iter=4)
        self.received = []

    def has_awaiting_tool_calls(self, _agent_name: str) -> bool:
        return self.awaiting


class _FakeAgent:
    name = "test-agent"

    def __init__(self, *, awaiting: bool = False) -> None:
        self.state = _FakeState(awaiting=awaiting)

    async def _handle_incoming_messages(self, messages) -> None:
        self.state.received.extend(messages)


def _message(receipt: str) -> Msg:
    return Msg(
        name="user",
        role="user",
        content=[TextBlock(type="text", text=f"queued {receipt}")],
        metadata={"qwenpaw_client_message_id": receipt},
    )


async def _finish(queue: asyncio.Queue, release: asyncio.Event) -> None:
    release.set()
    while await asyncio.wait_for(queue.get(), timeout=1) is not None:
        pass


async def _active_tracker():
    tracker = TaskTracker()
    started = asyncio.Event()
    release = asyncio.Event()

    async def stream(_payload):
        started.set()
        await release.wait()
        yield "data: done\\n\\n"

    queue, _ = await tracker.attach_or_start("chat-1", None, stream)
    await asyncio.wait_for(started.wait(), timeout=1)
    return tracker, queue, release


@pytest.mark.asyncio
async def test_task_tracker_steering_is_idempotent_and_taken_atomically():
    tracker, queue, release = await _active_tracker()
    first = _message("receipt-1")
    second = _message("receipt-2")

    assert await tracker.enqueue_steering("chat-1", [first, second]) == [
        "receipt-1",
        "receipt-2",
    ]
    # A retried POST acknowledges both receipts but queues neither twice.
    assert await tracker.enqueue_steering("chat-1", [first, second]) == [
        "receipt-1",
        "receipt-2",
    ]
    assert await tracker.take_steering("chat-1") == [first, second]
    assert await tracker.take_steering("chat-1") == []

    await _finish(queue, release)


@pytest.mark.asyncio
async def test_steering_waits_while_a_tool_confirmation_is_pending():
    tracker, queue, release = await _active_tracker()
    message = _message("receipt-pending")
    assert await tracker.enqueue_steering("chat-1", [message]) == [
        "receipt-pending",
    ]
    agent = _FakeAgent(awaiting=True)
    middleware = QueueSteeringMiddleware(tracker, "chat-1")

    async def next_handler(**_kwargs):
        yield "reasoning"

    assert [event async for event in middleware.on_reasoning(
        agent,
        {},
        next_handler,
    )] == ["reasoning"]
    assert agent.state.received == []
    assert await tracker.take_steering("chat-1") == [message]

    await _finish(queue, release)


@pytest.mark.asyncio
async def test_completed_reply_end_is_swallowed_only_when_message_injected():
    tracker, queue, release = await _active_tracker()
    message = _message("receipt-at-end")
    await tracker.enqueue_steering("chat-1", [message])
    agent = _FakeAgent()
    middleware = QueueSteeringMiddleware(tracker, "chat-1")
    completed = ReplyEndEvent(
        session_id="session-1",
        reply_id="reply-1",
        finished_reason=ReplyFinishedReason.COMPLETED,
    )

    async def next_handler(**_kwargs):
        yield completed

    events = [
        event
        async for event in middleware.on_reply(agent, {}, next_handler)
    ]

    assert events == []
    assert agent.state.received == [message]
    assert agent.state.reply_context.cur_iter == 0

    await _finish(queue, release)


@pytest.mark.asyncio
async def test_non_completed_reply_end_is_never_swallowed():
    tracker, queue, release = await _active_tracker()
    await tracker.enqueue_steering("chat-1", [_message("receipt-interrupted")])
    agent = _FakeAgent()
    middleware = QueueSteeringMiddleware(tracker, "chat-1")
    interrupted = ReplyEndEvent(
        session_id="session-1",
        reply_id="reply-2",
        finished_reason=ReplyFinishedReason.INTERRUPTED,
    )

    async def next_handler(**_kwargs):
        yield interrupted

    events = [
        event
        async for event in middleware.on_reply(agent, {}, next_handler)
    ]

    assert events == [interrupted]
    assert agent.state.received == []
    assert await tracker.take_steering("chat-1")

    await _finish(queue, release)
