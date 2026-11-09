# -*- coding: utf-8 -*-
"""Process-level session persistence during an actual Uvicorn shutdown."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGTERM behavior")
def test_sigterm_drains_task_tracker_runtime_and_persists_session(tmp_path):
    ready = tmp_path / "ready"
    request_started = tmp_path / "request_started"
    saved = tmp_path / "saved"
    state_path = tmp_path / "session.json"
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]

    script = textwrap.dedent(
        """
        import asyncio
        import json
        import signal
        import sys
        from contextlib import asynccontextmanager
        from copy import deepcopy
        from pathlib import Path
        from types import SimpleNamespace

        import uvicorn
        from agentscope.event import ReplyStartEvent
        from agentscope.message import Msg, TextBlock
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse

        from qwenpaw.app.task_tracker import TaskTracker
        from qwenpaw.runtime import runtime as runtime_module
        from qwenpaw.runtime.hooks import HookResult
        from qwenpaw.runtime.runtime import Runtime
        from qwenpaw.schemas import AgentRequest, Message, TextContent

        ready, request_started, saved, state_path = map(Path, sys.argv[1:5])
        port = int(sys.argv[5])

        class Session:
            async def save_session_state(self, **kwargs):
                data = deepcopy(kwargs["agent"].state_dict())
                state_path.write_text(json.dumps(data), encoding="utf-8")
                context = data["state"]["context"]
                if any(m.get("role") == "user" for m in context):
                    saved.write_text("saved", encoding="utf-8")

        class Hooks:
            async def run(self, _phase, _ctx):
                return HookResult()

        class Commands:
            async def dispatch(self, _text, _ctx):
                return None

        class Agent:
            name = "shutdown-test-agent"
            def __init__(self):
                self.state = SimpleNamespace(context=[])
            def state_dict(self):
                return {"state": {"context": [
                    msg.model_dump(mode="json") for msg in self.state.context
                ]}}
            async def reply_stream(self, inputs):
                self.state.context.extend(inputs)
                request_started.write_text("started", encoding="utf-8")
                yield ReplyStartEvent(
                    session_id="test-session", reply_id="reply-1", name=self.name,
                )
                await asyncio.Event().wait()
            async def close(self):
                return None

        agent = Agent()
        class Builder:
            def __init__(self, **_kwargs):
                pass
            async def build(self, _ctx):
                return agent

        runtime_module.AgentBuilder = Builder
        session = Session()
        tracker = TaskTracker()
        owner = SimpleNamespace(
            agent_id="default",
            workspace_dir=None,
            session=session,
            task_tracker=tracker,
            plugins=SimpleNamespace(
                hook_registry=Hooks(), slash_command_registry=Commands(), modes=[],
            ),
        )

        async def response_stream(_payload):
            request = AgentRequest(
                session_id="test-session", user_id="user-1", channel="console",
                input=[Message(
                    role="user",
                    content=[TextContent(type="text", text="persist this turn")],
                    metadata={"qwenpaw_client_message_id": "client-1"},
                )],
            )
            runtime = Runtime(workspace=owner, app_services=None)
            async for _event in runtime.run(request):
                yield "data: event\\n\\n"

        app = FastAPI()
        @app.post("/run")
        async def run(_body: dict):
            async def consume():
                queue, is_new = await tracker.attach_or_start(
                    "test-run", None, response_stream, owner=owner,
                )
                if is_new:
                    async for _item in tracker.stream_from_queue(queue, "test-run"):
                        pass
            # Like the console endpoint, the HTTP request attaches to a
            # TaskTracker-owned producer which outlives the SSE socket.
            asyncio.create_task(consume())
            async def events():
                yield "data: accepted\\n\\n"
                await asyncio.Event().wait()
            return StreamingResponse(events(), media_type="text/event-stream")

        @asynccontextmanager
        async def lifespan(app):
            app.state.multi_agent_manager = SimpleNamespace(stop_all=stop_all)
            ready.write_text("ready", encoding="utf-8")
            try:
                yield
            finally:
                await stop_all()

        async def stop_all():
            if not await tracker.cancel_owner(owner, timeout=4):
                raise TimeoutError("tracker did not drain")

        app.router.lifespan_context = lifespan
        uvicorn.run(
            app, host="127.0.0.1", port=port,
            timeout_graceful_shutdown=0.4, log_level="error",
        )
        """,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[3] / "src"), env.get("PYTHONPATH", "")],
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(ready), str(request_started),
         str(saved), str(state_path), str(port)],
        env=env,
        start_new_session=True,
    )
    client = None
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline and not ready.exists():
            if proc.poll() is not None:
                pytest.fail(f"server exited early: {proc.returncode}")
            time.sleep(0.05)
        assert ready.exists()
        client = socket.create_connection(("127.0.0.1", port), timeout=1)
        client.sendall(
            b"POST /run HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: 2\r\nContent-Type: application/json\r\n\r\n{}",
        )
        while time.monotonic() < deadline and not request_started.exists():
            time.sleep(0.05)
        assert request_started.exists(), "background Runtime producer did not start"
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=8)
        assert saved.exists(), "shutdown did not persist the active session"
        snapshot = json.loads(state_path.read_text(encoding="utf-8"))
        context = snapshot["state"]["context"]
        user = next(m for m in context if m.get("role") == "user")
        assert user["metadata"]["qwenpaw_turn_state"]["status"] == "canceled"
        # Uvicorn re-raises the captured signal after orderly lifespan teardown.
        assert proc.returncode in (0, -signal.SIGTERM)
    finally:
        if client is not None:
            client.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
