# -*- coding: utf-8 -*-
"""Cancellation persistence must respect ephemeral internal requests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from qwenpaw.agents.acp.meta import ACP_EPHEMERAL_META_KEY
from qwenpaw.runtime.runtime import Runtime


@pytest.mark.asyncio
async def test_ephemeral_cancel_does_not_create_hidden_session() -> None:
    class FailIfSaved:
        async def save_session_state(self, **_kwargs) -> None:
            raise AssertionError("ephemeral request must not be saved")

    ctx = SimpleNamespace(
        request=SimpleNamespace(
            request_context={ACP_EPHEMERAL_META_KEY: True},
        ),
        agent=SimpleNamespace(state_dict=lambda: {"state": {}}),
        workspace=SimpleNamespace(session=FailIfSaved()),
    )
    runtime = Runtime(workspace=object(), app_services=None)

    await runtime._try_save_on_cancel(ctx)  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_cancel_save_waits_for_disk_write_after_repeated_cancel() -> None:
    save_started = asyncio.Event()
    allow_save = asyncio.Event()
    save_finished = asyncio.Event()

    class SlowSession:
        async def save_session_state(self, **_kwargs) -> None:
            save_started.set()
            await allow_save.wait()
            save_finished.set()

    ctx = SimpleNamespace(
        request=SimpleNamespace(request_context={}, user_id="u1", channel=""),
        agent=SimpleNamespace(state_dict=lambda: {"state": {"context": []}}),
        workspace=SimpleNamespace(session=SlowSession()),
        session_id="s1",
        error=asyncio.CancelledError("shutdown"),
    )
    runtime = Runtime(workspace=object(), app_services=None)
    task = asyncio.create_task(runtime._try_save_on_cancel(ctx))

    await asyncio.wait_for(save_started.wait(), timeout=1)
    task.cancel("repeated shutdown cancel")
    await asyncio.sleep(0)
    assert not task.done()
    assert not save_finished.is_set()

    allow_save.set()
    await asyncio.wait_for(task, timeout=1)
    assert save_finished.is_set()
