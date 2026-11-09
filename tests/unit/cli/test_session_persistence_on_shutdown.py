# -*- coding: utf-8 -*-
"""End-to-end regression coverage for session persistence on shutdown."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from qwenpaw.cli import shutdown_cmd


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal behavior")
def test_running_sse_request_checkpoints_before_graceful_timeout(
    tmp_path,
    monkeypatch,
):
    ready = tmp_path / "ready"
    user_saved = tmp_path / "user_saved"
    checkpoint_saved = tmp_path / "checkpoint_saved"
    request_started = tmp_path / "request_started"
    release_model = tmp_path / "release_model"
    state_path = tmp_path / "session.json"
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]

    script = textwrap.dedent(
        """
        import asyncio
        import json
        import sys
        from contextlib import asynccontextmanager
        from pathlib import Path

        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from qwenpaw.runtime import runtime as runtime_module
        from qwenpaw.runtime.hooks import HookContext
        from qwenpaw.runtime.runtime import Runtime
        from qwenpaw.schemas import AgentRequest, Message, TextContent
        from agentscope.message import Msg, TextBlock

        ready = Path(sys.argv[1])
        user_saved = Path(sys.argv[2])
        checkpoint_saved = Path(sys.argv[3])
        request_started = Path(sys.argv[4])
        release_model = Path(sys.argv[5])
        state_path = Path(sys.argv[6])
        port = int(sys.argv[7])

        class Session:
            async def save_session_state(self, **kwargs):
                data = kwargs["agent"].state_dict()
                state_path.write_text(json.dumps(data), encoding="utf-8")
                context = data.get("state", {}).get("context", [])
                if context and any(
                    m.get("role") == "user" for m in context
                ) and not user_saved.exists():
                    user_saved.write_text("saved", encoding="utf-8")
                if any(
                    "completed model round" in str(m)
                    for m in context
                ):
                    checkpoint_saved.write_text("saved", encoding="utf-8")

        class Hooks:
            async def run(self, phase, ctx):
                return type("Result", (), {"action": "continue", "payload": None})()

        class Commands:
            async def dispatch(self, text, ctx):
                return None

        class Workspace:
            workspace_dir = None
            agent_id = "default"
            session = Session()
            plugins = type("Plugins", (), {
                "hook_registry": Hooks(),
                "slash_command_registry": Commands(),
                "modes": [],
            })()

        class FakeAgent:
            name = "fake"
            def __init__(self):
                self.state = type("State", (), {"context": []})()
                self.saved = []
            async def reply_stream(self, inputs):
                self.state.context.extend(inputs)
                self.state.context.append(Msg(
                    name="fake", role="assistant",
                    content=[TextBlock(text="completed model round")],
                ))
                yield type("Event", (), {
                    "type": "REPLY_END", "reply_id": None,
                    "created_at": "2026-11-09T00:00:00",
                })()
                request_started.write_text("started", encoding="utf-8")
                while not release_model.exists():
                    await asyncio.sleep(0.02)
                await asyncio.Event().wait()
            def state_dict(self):
                return {"state": {"context": [
                    {"role": m.role, "content": [
                        getattr(b, "text", "") for b in m.content
                    ]} for m in self.state.context
                ]}}
            async def close(self):
                pass

        fake_agent = FakeAgent()
        class Builder:
            def __init__(self, **kwargs):
                pass
            async def build(self, ctx):
                return fake_agent

        runtime_module.AgentBuilder = Builder
        app = FastAPI()

        @app.post("/run")
        async def run(body: dict):
            request = AgentRequest(
                session_id="s1", user_id="u1", channel="console",
                input=[Message(role="user", content=[TextContent(type="text", text="save me")])],
            )
            rt = Runtime(workspace=Workspace(), app_services=None)
            async def stream():
                async for item in rt.run(request):
                    yield b"data: {}\\n\\n"
            return StreamingResponse(stream(), media_type="text/event-stream")

        @asynccontextmanager
        async def lifespan(app):
            ready.write_text("ready", encoding="utf-8")
            yield

        app.router.lifespan_context = lifespan
        uvicorn.run(app, host="127.0.0.1", port=port,
                    timeout_graceful_shutdown=0.4, log_level="error")
        """,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[3] / "src"), env.get("PYTHONPATH", "")],
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(ready), str(user_saved),
         str(checkpoint_saved), str(request_started), str(release_model),
         str(state_path), str(port)],
        env=env,
        start_new_session=True,
    )
    client = None
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not ready.exists():
            if proc.poll() is not None:
                pytest.fail(f"server exited early: {proc.returncode}")
            time.sleep(0.05)
        assert ready.exists()
        client = socket.create_connection(("127.0.0.1", port), timeout=1)
        client.sendall(
            b"POST /run HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: 2\r\nContent-Type: application/json\r\n"
            b"Connection: keep-alive\r\n\r\n{}",
        )
        while time.monotonic() < deadline and not request_started.exists():
            time.sleep(0.05)
        assert request_started.exists()

        original_signal = shutdown_cmd._signal_process_tree_unix
        signals = []
        def record_signal(pid, sig):
            signals.append(sig)
            original_signal(pid, sig)
        monkeypatch.setattr(shutdown_cmd, "_signal_process_tree_unix", record_signal)
        reaper = threading.Thread(target=proc.wait, daemon=True)
        reaper.start()
        assert shutdown_cmd._terminate_pid(proc.pid, timeout_sec=5)
        reaper.join(timeout=1)
        assert proc.returncode in (0, -signal.SIGTERM)
        assert user_saved.exists(), "incoming user message was not durable"
        assert checkpoint_saved.exists(), "completed agent round was not durable"
        snapshot = json.loads(state_path.read_text(encoding="utf-8"))
        context = snapshot["state"]["context"]
        assert any(m["role"] == "user" for m in context)
        assert any("completed model round" in str(m) for m in context)
        assert signals == [signal.SIGTERM]
    finally:
        if client is not None:
            client.close()
        if proc.poll() is None:
            release_model.touch()
            proc.kill()
            proc.wait(timeout=5)
