"""Safe-boundary tests for console queue steering."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Awaitable, Callable
from unittest.mock import AsyncMock

import pytest
from agentscope.agent import Agent, ReActConfig
from agentscope.event import ReplyEndEvent, ReplyFinishedReason
from agentscope.message import Msg, TextBlock, ToolCallBlock
from agentscope.model import ChatModelBase
from agentscope.model._model_response import ChatResponse
from agentscope.permission import PermissionMode
from agentscope.state import ReplyContext
from agentscope.tool import FunctionTool, Toolkit

from qwenpaw.agents.react_agent import QwenPawAgent
from qwenpaw.app.task_tracker import TaskTracker
from qwenpaw.loop.gates import StopAction, StopHandlerResult
from qwenpaw.runtime.builder import AgentBuilder
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


def _message(receipt: str, text: str | None = None) -> Msg:
    return Msg(
        name="user",
        role="user",
        content=[TextBlock(type="text", text=text or f"queued {receipt}")],
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


class _QueueSteeringModel(ChatModelBase):
    """Inject steering during the first provider request, then finish."""

    def __init__(
        self,
        tracker: TaskTracker,
        during_first_request: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(
            credential=None,
            model="queue-steering-test",
            parameters=ChatModelBase.Parameters(),
            stream=False,
            context_size=32_768,
        )
        self.formatter = SimpleNamespace(supported_input_media_types=[])
        self._tracker = tracker
        self._during_first_request = during_first_request
        self.requests: list[list[Msg]] = []

    async def __call__(
        self,
        messages: list[Msg],
        tools=None,
        tool_choice=None,
        **_kwargs,
    ) -> ChatResponse:
        self.requests.append(list(messages))
        if len(self.requests) == 1:
            if self._during_first_request is not None:
                await self._during_first_request()
            else:
                accepted = await self._tracker.enqueue_steering(
                    "chat-1",
                    [_message("receipt-model-boundary", "STEERING_MARKER")],
                )
                assert accepted == ["receipt-model-boundary"]
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        id="tool-call-1",
                        name="echo_tool",
                        input='{"value":"tool completed"}',
                    ),
                ],
                is_last=True,
            )
        return ChatResponse(
            content=[TextBlock(type="text", text="completed")],
            is_last=True,
        )


def _bare_qwenpaw_agent(
    model: ChatModelBase,
    tracker: TaskTracker,
    tool_results: list[str],
):
    """Initialize the real AgentScope chain without workspace services."""
    agent = object.__new__(QwenPawAgent)
    middleware = QueueSteeringMiddleware(tracker, "chat-1")

    async def record_tool(value: str) -> str:
        tool_results.append(value)
        return value

    Agent.__init__(
        agent,
        name="test-agent",
        system_prompt="test prompt",
        model=model,
        toolkit=Toolkit(
            tools=[
                FunctionTool(
                    record_tool,
                    name="echo_tool",
                ),
            ],
        ),
        middlewares=[middleware],
        react_config=ReActConfig(max_iters=4),
    )
    from agentscope.permission import PermissionMode

    agent.state.permission_context.mode = PermissionMode.BYPASS
    agent._agent_config = SimpleNamespace(language="en-US")
    agent._context_manager = None
    agent._request_context = {}
    agent._model_session_id = "queue-steering-test"
    agent._gate_pending_stop = None
    agent._gate_pending_continue = None
    agent._inject_pending_hints = _async_noop
    agent._model_rejects_media = lambda: False
    agent._model_rejects_audio = lambda: False
    agent._uses_request_time_media_normalization = lambda: False

    async def stop_handlers(final_msg):
        return StopHandlerResult(
            action=StopAction.TERMINATE,
            final_message=final_msg,
        )

    agent._run_stop_handlers = stop_handlers
    return agent


async def _async_noop() -> None:
    return None


def test_builder_attaches_steering_to_workspace_tracker(monkeypatch):
    tracker = TaskTracker()
    workspace = SimpleNamespace(
        task_tracker=tracker,
        workspace_dir="",
        memory_manager=None,
    )
    ctx = SimpleNamespace(
        request=SimpleNamespace(
            channel_meta={"qwenpaw_run_key": "chat-1"},
        ),
        workspace=workspace,
        app_services=None,
    )
    agent_config = SimpleNamespace(
        running=SimpleNamespace(
            light_context_config=SimpleNamespace(
                visual_compact_config=SimpleNamespace(
                    enabled=False,
                    effort="low",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        AgentBuilder,
        "_build_tool_result_pruning_middleware",
        staticmethod(lambda _ctx, _config: None),
    )
    monkeypatch.setattr(
        "qwenpaw.plugins.registry.PluginRegistry.get_middleware_factories",
        lambda _registry: [],
    )

    middlewares = AgentBuilder._build_middlewares(ctx, agent_config)
    steering = next(
        middleware
        for middleware in middlewares
        if isinstance(middleware, QueueSteeringMiddleware)
    )

    assert steering._task_tracker is tracker
    assert steering._run_key == "chat-1"


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
async def test_queued_message_is_injected_into_next_real_model_request():
    tracker, queue, release = await _active_tracker()
    model = _QueueSteeringModel(tracker)
    tool_results: list[str] = []
    agent = _bare_qwenpaw_agent(model, tracker, tool_results)

    try:
        events = [
            event
            async for event in agent.reply_stream(
                inputs=Msg(
                    name="user",
                    role="user",
                    content=[TextBlock(type="text", text="start")],
                ),
            )
        ]

        assert len(model.requests) == 2
        assert tool_results == ["tool completed"]
        assert any(
            getattr(block, "text", None) == "STEERING_MARKER"
            for message in model.requests[1]
            for block in getattr(message, "content", [])
        )
        assert not await tracker.take_steering("chat-1")
        assert events
    finally:
        await _finish(queue, release)


@pytest.mark.asyncio
async def test_http_batch_arrives_in_order_at_next_model_request(monkeypatch):
    """Exercise the HTTP route and real agent loop with a gated local model."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from qwenpaw.app.routers import console

    tracker = TaskTracker()
    first_request_started = asyncio.Event()
    release_first_request = asyncio.Event()

    async def hold_first_request():
        first_request_started.set()
        await release_first_request.wait()

    model = _QueueSteeringModel(tracker, hold_first_request)
    tool_results: list[str] = []
    agent = _bare_qwenpaw_agent(model, tracker, tool_results)
    workspace = SimpleNamespace(
        task_tracker=tracker,
        chat_manager=SimpleNamespace(
            get_chat=AsyncMock(
                return_value=SimpleNamespace(id="chat-1", channel="console"),
            ),
        ),
    )
    monkeypatch.setattr(
        console,
        "get_agent_for_request",
        AsyncMock(return_value=workspace),
    )
    app = FastAPI()
    app.include_router(console.router, prefix="/api")
    events = []

    async def stream(_payload):
        async for event in agent.reply_stream(
            inputs=_message("receipt-start", "start"),
        ):
            events.append(event)
            yield "data: {}\n\n"

    queue, is_new_run = await tracker.attach_or_start("chat-1", None, stream)
    assert is_new_run
    body = {
        "chat_id": "chat-1",
        "input": [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
                "metadata": {"qwenpaw_client_message_id": receipt},
            }
            for receipt, text in [
                ("receipt-first", "QUEUE_FIRST"),
                ("receipt-second", "QUEUE_SECOND"),
            ]
        ],
    }
    try:
        await asyncio.wait_for(first_request_started.wait(), timeout=2)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            for _ in range(2):
                response = await client.post(
                    "/api/console/chat/steer",
                    headers={"X-Agent-Id": "default"},
                    json=body,
                )
                assert response.status_code == 200, response.text
                assert response.json() == {
                    "accepted_client_message_ids": [
                        "receipt-first",
                        "receipt-second",
                    ],
                }
        # Accepting the batch must not interrupt the in-flight model request.
        assert len(model.requests) == 1
        assert tool_results == []
        assert not any(
            message.metadata.get("qwenpaw_client_message_id")
            in {"receipt-first", "receipt-second"}
            for message in model.requests[0]
        )
    finally:
        await _finish(queue, release_first_request)

    assert len(model.requests) == 2
    assert tool_results == ["tool completed"]
    received = [
        message
        for message in model.requests[1]
        if message.metadata.get("qwenpaw_client_message_id")
        in {"receipt-first", "receipt-second"}
    ]
    assert [message.get_text_content() for message in received] == [
        "QUEUE_FIRST",
        "QUEUE_SECOND",
    ]
    assert [
        message.metadata["qwenpaw_client_message_id"] for message in received
    ] == ["receipt-first", "receipt-second"]
    assert not await tracker.take_steering("chat-1")
    assert events


@pytest.mark.asyncio
async def test_unconsumed_steering_is_reported_when_run_finishes(caplog):
    tracker, queue, release = await _active_tracker()
    message = _message("receipt-leftover")
    assert await tracker.enqueue_steering("chat-1", [message]) == [
        "receipt-leftover",
    ]

    with caplog.at_level(logging.WARNING, logger="qwenpaw.app.task_tracker"):
        await _finish(queue, release)

    assert "run ended with unconsumed queue steering messages" in caplog.text
    assert "receipt-leftover" in caplog.text


@pytest.mark.asyncio
async def test_steering_waits_while_a_tool_confirmation_is_pending():
    tracker, queue, release = await _active_tracker()
    message = _message("receipt-pending")
    assert await tracker.enqueue_steering("chat-1", [message]) == [
        "receipt-pending",
    ]
    agent = _FakeAgent(awaiting=True)
    middleware = QueueSteeringMiddleware(tracker, "chat-1")

    call = {}

    async def next_handler(**kwargs):
        call.update(kwargs)
        return "model-call"

    assert await middleware.on_model_call(
        agent,
        {"messages": [], "tools": []},
        next_handler,
    ) == "model-call"
    assert call == {"messages": [], "tools": []}
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
