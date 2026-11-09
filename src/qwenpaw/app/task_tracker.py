# -*- coding: utf-8 -*-
"""Task tracker for background runs: streaming, reconnect, multi-subscriber.

``run_key`` is typically ``ChatSpec.id`` (chat_id). Per run: task, queues,
event buffer. Reconnects get buffer replay + new events. Cleanup when task
completes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
)

logger = logging.getLogger(__name__)

_SENTINEL = None

# Indirection so tests can simulate a failing task creation without
# patching the process-wide ``asyncio`` module.
_create_task = asyncio.create_task

# Emitted to reconnect subscribers right after the buffered events, so
# the client can render the replayed part instantly (no token-by-token
# re-animation) and switch to live streaming afterwards.
REPLAY_END_SSE = f"data: {json.dumps({'type': 'replay_end'})}\n\n"


@dataclass
class _RunState:
    """Per-run state (task, queues, buffer), guarded by tracker lock."""

    task: asyncio.Future
    queues: list[asyncio.Queue] = field(default_factory=list)
    buffer: list[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    finish_time: Optional[datetime] = None
    owner: object | None = None
    steering_messages: list[Any] = field(default_factory=list)
    steering_receipts: set[str] = field(default_factory=set)
    steering_closed: bool = False
    steering_window_open: bool = True


class TaskTracker:
    """Per-agent tracker: run_key -> RunState.

    All mutations to _runs under _lock. Producer broadcasts under lock.
    Subscribers use unbounded per-connection queues; disconnect removes them
    via :meth:`detach_subscriber`. A workspace reload reuses the same tracker
    so active runs remain reconnectable.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._runs: dict[str, _RunState] = {}
        self._global_last_run_at: Optional[datetime] = None
        self._global_last_finish_at: Optional[datetime] = None

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    async def get_status(self, run_key: str) -> str:
        """Return ``'idle'`` or ``'running'``."""
        async with self._lock:
            state = self._runs.get(run_key)
        if state is None or state.task.done():
            return "idle"
        return "running"

    @staticmethod
    def _steering_receipt(message: Any) -> str | None:
        metadata = getattr(message, "metadata", None)
        if isinstance(metadata, dict):
            value = metadata.get("qwenpaw_client_message_id")
            return str(value) if value else None
        return None

    async def enqueue_steering(
        self,
        run_key: str,
        messages: list[Any],
    ) -> list[str]:
        """Add user messages to an active run's next-model-request queue.

        Receipt IDs make repeated HTTP submissions idempotent for the lifetime
        of the run, including retries after the middleware has taken a batch.
        """
        async with self._lock:
            state = self._runs.get(run_key)
            if (
                state is None
                or state.task.done()
                or state.steering_closed
                or not state.steering_window_open
            ):
                return []
            accepted: list[str] = []
            for message in messages:
                receipt = self._steering_receipt(message)
                if not receipt:
                    continue
                accepted.append(receipt)
                if receipt in state.steering_receipts:
                    continue
                state.steering_receipts.add(receipt)
                state.steering_messages.append(message)
            return accepted

    async def open_steering_window(self, run_key: str) -> None:
        async with self._lock:
            state = self._runs.get(run_key)
            if state is not None and not state.task.done() and not state.steering_closed:
                state.steering_window_open = True

    async def close_steering_window(self, run_key: str) -> None:
        async with self._lock:
            state = self._runs.get(run_key)
            if state is not None and not state.task.done() and not state.steering_closed:
                state.steering_window_open = False

    async def take_steering(
        self,
        run_key: str,
        *,
        allow_closed_window: bool = False,
    ) -> list[Any]:
        """Atomically take messages before a normal reasoning call."""
        async with self._lock:
            state = self._runs.get(run_key)
            if (
                state is None
                or state.task.done()
                or state.steering_closed
                or (not allow_closed_window and not state.steering_window_open)
                or not state.steering_messages
            ):
                return []
            messages = state.steering_messages
            state.steering_messages = []
            return messages

    async def restore_steering(
        self,
        run_key: str,
        messages: list[Any],
    ) -> None:
        """Restore a batch if runtime conversion failed before injection."""
        async with self._lock:
            state = self._runs.get(run_key)
            if (
                state is not None
                and not state.task.done()
                and not state.steering_closed
            ):
                state.steering_messages = messages + state.steering_messages

    async def get_global_status(self) -> dict:
        """Get global agent status summary.

        Returns:
            dict with keys:
                - status: 'idle' | 'running'
                - running_task_count: int
                - last_run_at: Optional[datetime]
                - last_finish_at: Optional[datetime]
        """
        async with self._lock:
            running_count = sum(
                1 for state in self._runs.values() if not state.task.done()
            )
            status = "running" if running_count > 0 else "idle"

            return {
                "status": status,
                "running_task_count": running_count,
                "last_run_at": self._global_last_run_at,
                "last_finish_at": self._global_last_finish_at,
            }

    async def has_active_tasks(self) -> bool:
        """Check if any tasks are currently running.

        Returns:
            bool: True if any tasks are active, False otherwise
        """
        async with self._lock:
            for state in self._runs.values():
                if not state.task.done():
                    return True
            return False

    async def has_active_tasks_excluding(
        self,
        excluded_task: asyncio.Future | None,
    ) -> bool:
        """Check for active tasks other than ``excluded_task``."""
        async with self._lock:
            return any(
                not state.task.done() and state.task is not excluded_task
                for state in self._runs.values()
            )

    async def list_active_tasks(self) -> list[str]:
        """List all currently running task keys.

        Returns:
            list[str]: List of active run_keys
        """
        async with self._lock:
            return [
                run_key
                for run_key, state in self._runs.items()
                if not state.task.done()
            ]

    async def snapshot_active_tasks(
        self,
        owner: object | None = None,
    ) -> dict[str, asyncio.Future]:
        """Return the currently active tasks without tracking later runs."""
        async with self._lock:
            return {
                run_key: state.task
                for run_key, state in self._runs.items()
                if not state.task.done()
                and (owner is None or state.owner is owner)
            }

    async def cancel_owner(
        self,
        owner: object,
        *,
        timeout: float = 10.0,
    ) -> bool:
        """Cancel and drain runs owned by a workspace before it is stopped.

        The owner filter matters during zero-downtime reload: old and new
        workspace instances may share a tracker while only the old instance
        is being torn down.
        """
        tasks = list(
            (await self.snapshot_active_tasks(owner=owner)).values(),
        )
        if not tasks:
            return True

        logger.info(
            "stopping %d active run(s) for workspace owner %s",
            len(tasks),
            getattr(owner, "agent_id", "unknown"),
        )
        for task in tasks:
            task.cancel("shutdown")

        _, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            logger.error(
                "timed out draining %d run(s) for workspace owner %s",
                len(pending),
                getattr(owner, "agent_id", "unknown"),
            )
            return False
        return True

    async def wait_tasks_done(
        self,
        tasks: list[asyncio.Future],
        timeout: float = 300.0,
    ) -> bool:
        """Wait for a fixed task snapshot without cancelling on timeout."""
        if not tasks:
            return True

        async def _wait_snapshot() -> None:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks),
                return_exceptions=True,
            )

        try:
            await asyncio.wait_for(_wait_snapshot(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_all_done(self, timeout: float = 300.0) -> bool:
        """Wait for all active tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds (default: 300s = 5min)

        Returns:
            bool: True if all tasks completed, False if timeout occurred
        """

        async def _wait_loop() -> None:
            while await self.has_active_tasks():
                await asyncio.sleep(0.5)

        try:
            await asyncio.wait_for(_wait_loop(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def attach(self, run_key: str) -> asyncio.Queue | None:
        """Attach to an existing run.

        Returns a new queue pre-filled with the event buffer plus a
        ``replay_end`` marker, or ``None`` if no run is active for
        *run_key*.
        """
        async with self._lock:
            state = self._runs.get(run_key)
            if state is None or state.task.done():
                return None
            q: asyncio.Queue = asyncio.Queue()
            for sse in state.buffer:
                q.put_nowait(sse)
            q.put_nowait(REPLAY_END_SSE)
            state.queues.append(q)
            return q

    async def detach_subscriber(
        self,
        run_key: str,
        queue: asyncio.Queue,
    ) -> None:
        """Remove *queue* from *run_key*'s subscriber list.

        Idempotent if the run ended or *queue* was already removed.
        """
        async with self._lock:
            state = self._runs.get(run_key)
            if state is None:
                return
            try:
                state.queues.remove(queue)
            except ValueError:
                pass

    async def request_stop(
        self,
        run_key: str,
        *,
        reason: str = "user",
    ) -> bool:
        """Cancel the run. Returns ``True`` if it was running."""
        logger.debug("[STOP] request_stop called for run_key=%s", run_key)
        async with self._lock:
            state = self._runs.get(run_key)
            logger.debug(
                "[STOP] run_key=%s state=%s done=%s",
                run_key,
                "found" if state else "not_found",
                state.task.done() if state else "N/A",
            )
            if state is None or state.task.done():
                logger.debug(
                    "[STOP] Cannot stop run_key=%s (not running)",
                    run_key,
                )
                return False
            logger.debug(
                "[STOP] Calling task.cancel() for run_key=%s",
                run_key,
            )
            task = state.task
            task.cancel(reason)
            logger.debug("[STOP] task.cancel() called for run_key=%s", run_key)
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    async def attach_or_start(
        self,
        run_key: str,
        payload: Any,
        stream_fn: Callable[[Any], AsyncIterator[str]],
        owner: object | None = None,
        on_finished: Callable[[str, datetime], Awaitable[Any]] | None = None,
    ) -> tuple[asyncio.Queue, bool]:
        """Attach to an existing run or start a new one.

        Returns ``(queue, is_new_run)``.
        """
        async with self._lock:
            state = self._runs.get(run_key)
            if state is not None and not state.task.done():
                q: asyncio.Queue = asyncio.Queue()
                for sse in state.buffer:
                    q.put_nowait(sse)
                state.queues.append(q)
                return q, False

            my_queue: asyncio.Queue = asyncio.Queue()
            run = _RunState(
                task=asyncio.Future(),  # placeholder, replaced below
                queues=[my_queue],
                buffer=[],
                owner=owner,
            )

            tracker_ref = weakref.ref(self)

            async def _producer() -> None:
                start_time = datetime.now(timezone.utc)
                stream_iterator = None

                try:
                    tracker = tracker_ref()
                    if tracker is not None:
                        async with tracker.lock:
                            run.start_time = start_time
                            # pylint: disable=protected-access
                            tracker._global_last_run_at = start_time

                    stream_iterator = stream_fn(payload).__aiter__()
                    async for sse in stream_iterator:
                        tracker = tracker_ref()
                        if tracker is None:
                            return
                        async with tracker.lock:
                            run.buffer.append(sse)
                            for q in run.queues:
                                q.put_nowait(sse)
                except asyncio.CancelledError as exc:
                    from ..runtime.cancellation import CURRENT_CANCEL_REASON

                    reason = str(exc.args[0]) if exc.args else None
                    CURRENT_CANCEL_REASON.set(reason)
                    logger.debug(
                        "run cancelled run_key=%s reason=%s",
                        run_key,
                        reason or "unknown",
                    )
                except Exception:
                    logger.exception("run error run_key=%s", run_key)
                    err_sse = (
                        "data: "
                        f"{json.dumps({'error': 'internal server error'})}\n\n"
                    )
                    tracker = tracker_ref()
                    if tracker is not None:
                        async with tracker.lock:
                            run.buffer.append(err_sse)
                            for q in run.queues:
                                q.put_nowait(err_sse)
                finally:
                    # ``async for`` does not guarantee that an async
                    # generator is closed when its consumer task is cancelled.
                    # Close it explicitly so Runtime's GeneratorExit cleanup
                    # can persist the interrupted session before shutdown.
                    close = getattr(stream_iterator, "aclose", None)
                    if close is not None:
                        try:
                            await close()
                        except asyncio.CancelledError:
                            logger.warning(
                                "run stream close cancelled run_key=%s",
                                run_key,
                            )
                        except Exception:
                            logger.exception(
                                "run stream close failed run_key=%s",
                                run_key,
                            )

                    finish_time = datetime.now(timezone.utc)
                    if on_finished is not None:
                        try:
                            await on_finished(run_key, finish_time)
                        except Exception:  # pylint: disable=broad-except
                            logger.exception(
                                "run completion callback failed run_key=%s",
                                run_key,
                            )
                    tracker = tracker_ref()
                    if tracker is not None:
                        async with tracker.lock:
                            run.finish_time = finish_time
                            # pylint: disable=protected-access
                            tracker._global_last_finish_at = finish_time
                            run.steering_closed = True
                            if run.steering_messages:
                                receipts = [
                                    receipt
                                    for message in run.steering_messages
                                    if (
                                        receipt := self._steering_receipt(
                                            message,
                                        )
                                    )
                                ]
                                logger.warning(
                                    "run ended with unconsumed queue "
                                    "steering messages run_key=%s "
                                    "message_count=%d receipts=%s",
                                    run_key,
                                    len(run.steering_messages),
                                    receipts,
                                )
                            for q in run.queues:
                                q.put_nowait(_SENTINEL)
                            # pylint: disable=protected-access
                            tracker._runs.pop(
                                run_key,
                                None,
                            )

            # Register the run only once the producer task exists.
            # Registering a placeholder first would leak a permanent
            # "running" entry -- the status counters read
            # ``not state.task.done()``, and a bare Future is never done --
            # if task creation ever failed. This is defensive hardening:
            # the window is synchronous and a running loop cannot be
            # closed, so the failure is not reachable in practice.
            run.task = _create_task(_producer())
            self._runs[run_key] = run
            return my_queue, True

    async def stream_from_queue(
        self,
        queue: asyncio.Queue,
        run_key: str,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE strings from *queue* until the sentinel ``None``.

        Always detaches *queue* from *run_key* when this stream ends or is
        closed (including client disconnect), so reconnects do not leak queues.
        """
        try:
            while True:
                try:
                    event = await queue.get()
                    if event is _SENTINEL:
                        break
                    yield event
                except asyncio.CancelledError:
                    break
        finally:
            await self.detach_subscriber(run_key, queue)
