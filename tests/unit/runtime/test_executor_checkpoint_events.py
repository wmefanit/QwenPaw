# -*- coding: utf-8 -*-
"""Checkpoint only at complete AgentScope reply/tool boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from agentscope.event import EventType

from qwenpaw.runtime.executor import AgentExecutor


class _Agent:
    def __init__(self, events):
        self.events = events

    async def reply_stream(self, inputs):
        for event in self.events:
            yield event


class _Envelope:
    async def translate_event(self, _event):
        if False:
            yield None

    async def heartbeat(self):
        if False:
            yield None


@pytest.mark.asyncio
async def test_executor_checkpoints_only_durable_event_boundaries(monkeypatch):
    events = [
        SimpleNamespace(type=EventType.REPLY_START),
        SimpleNamespace(type="MODEL_CALL_START"),
        SimpleNamespace(type=EventType.TOOL_RESULT_END),
        SimpleNamespace(type=EventType.REPLY_END),
    ]
    checkpoints = []

    async def save(agent, reason):
        checkpoints.append((agent, reason))
        return True

    monkeypatch.setattr(
        "qwenpaw.runtime.executor._iter_with_heartbeat",
        lambda iterator, _interval: iterator,
    )
    agent = _Agent(events)
    executor = AgentExecutor(agent, _Envelope(), checkpoint=save)

    assert [event async for event in executor.run([])] == []
    assert checkpoints == [
        (agent, EventType.REPLY_START),
        (agent, EventType.TOOL_RESULT_END),
        (agent, EventType.REPLY_END),
    ]
