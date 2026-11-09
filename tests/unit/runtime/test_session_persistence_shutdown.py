# -*- coding: utf-8 -*-
"""Shutdown checkpoints preserve accepted turns and completed agent state."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from agentscope.event import ReplyStartEvent
from agentscope.message import Msg, TextBlock

from qwenpaw.app.task_tracker import TaskTracker
from qwenpaw.runtime import runtime as runtime_module
from qwenpaw.runtime.hooks import HookResult
from qwenpaw.runtime.runtime import Runtime
from qwenpaw.schemas import AgentRequest, Message, TextContent


class _Session:
    def __init__(self):
        self.snapshots = []
        self.reply_started_saved = asyncio.Event()

    async def save_session_state(self, **kwargs):
        self.snapshots.append(deepcopy(kwargs["agent"].state_dict()))
        if len(self.snapshots) >= 2:
            self.reply_started_saved.set()


class _Hooks:
    async def run(self, _phase, _ctx):
        return HookResult()


class _Commands:
    async def dispatch(self, _text, _ctx):
        return None


class _Agent:
    name = "checkpoint-agent"

    def __init__(self, started):
        self.state = SimpleNamespace(context=[])
        self.started = started

    def state_dict(self):
        return {
            "state": {
                "context": [msg.model_dump(mode="json") for msg in self.state.context],
            },
        }

    async def reply_stream(self, inputs):
        self.state.context.extend(inputs)
        self.started.set()
        yield ReplyStartEvent(
            session_id="session-1",
            reply_id="reply-1",
            name=self.name,
        )
        await asyncio.Event().wait()

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_shutdown_cancel_checkpoints_and_closes_runtime_stream(monkeypatch):
    started = asyncio.Event()
    session = _Session()
    agent = _Agent(started)

    class Builder:
        def __init__(self, **_kwargs):
            pass

        async def build(self, _ctx):
            return agent

    monkeypatch.setattr(runtime_module, "AgentBuilder", Builder)
    monkeypatch.setattr(
        runtime_module,
        "record_agent_activity",
        lambda: asyncio.sleep(0),
    )

    owner = SimpleNamespace(
        agent_id="test-agent",
        workspace_dir="/tmp/test-agent",
        session=session,
        plugins=SimpleNamespace(
            hook_registry=_Hooks(),
            slash_command_registry=_Commands(),
            modes=[],
        ),
    )
    request = AgentRequest(
        session_id="session-1",
        user_id="user-1",
        channel="console",
        input=[
            Message(
                role="user",
                content=[TextContent(type="text", text="keep this turn")],
                metadata={"qwenpaw_client_message_id": "client-1"},
            ),
        ],
    )
    runtime = Runtime(workspace=owner, app_services=None)

    async def stream_fn(_payload):
        stream = runtime.run(request)
        try:
            async for _event in stream:
                yield "data: event\\n\\n"
        except asyncio.CancelledError:
            # Mirrors BaseChannel's cancellation cleanup of its process stream.
            await stream.aclose()
            raise

    tracker = TaskTracker()
    await tracker.attach_or_start(
        "run-1",
        None,
        stream_fn,
        owner=owner,
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    await asyncio.wait_for(session.reply_started_saved.wait(), timeout=2)

    assert len(session.snapshots) >= 2
    before_cancel = session.snapshots[-1]
    assert any(
        msg.get("role") == "user"
        and "keep this turn" in str(msg.get("content"))
        for msg in before_cancel["state"]["context"]
    )

    assert await tracker.cancel_owner(owner, timeout=2)
    assert len(session.snapshots) >= 3
    final_snapshot = session.snapshots[-1]
    last_user = next(
        msg
        for msg in reversed(final_snapshot["state"]["context"])
        if msg.get("role") == "user"
    )
    assert last_user["metadata"]["qwenpaw_turn_state"]["status"] == "canceled"
    assert await tracker.get_status("run-1") == "idle"
