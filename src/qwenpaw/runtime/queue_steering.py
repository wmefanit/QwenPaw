"""Inject queued console messages only at AgentScope's safe model boundaries."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Callable

from agentscope.middleware import MiddlewareBase

logger = logging.getLogger(__name__)


class QueueSteeringMiddleware(MiddlewareBase):
    """Consume chat steering messages before reasoning or after a full reply.

    A completed ``ReplyEndEvent`` is swallowed only after its assistant output
    and tool results are complete. Messages arriving during a model call are
    therefore injected before the next model request, never during a tool
    execution or while a user-confirmation call is parked.
    """

    def __init__(self, task_tracker: Any, run_key: str) -> None:
        self._task_tracker = task_tracker
        self._run_key = run_key

    async def _inject_pending(
        self,
        agent: Any,
        *,
        allow_closed_window: bool = False,
    ) -> bool:
        if agent.state.has_awaiting_tool_calls(agent.name):
            return False
        messages = await self._task_tracker.take_steering(
            self._run_key,
            allow_closed_window=allow_closed_window,
        )
        if not messages:
            return False
        try:
            await agent._handle_incoming_messages(messages)
        except Exception:
            await self._task_tracker.open_steering_window(self._run_key)
            await self._task_tracker.restore_steering(self._run_key, messages)
            logger.exception(
                "queue steering injection failed run_key=%s",
                self._run_key,
            )
            return False
        return True

    async def on_reasoning(
        self,
        agent: Any,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        await self._task_tracker.open_steering_window(self._run_key)
        await self._inject_pending(agent)
        async for event in next_handler(**input_kwargs):
            yield event

    async def on_reply(
        self,
        agent: Any,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        from agentscope.event import ReplyEndEvent, ReplyFinishedReason
        from agentscope.state import ReplyContext

        async for event in next_handler(**input_kwargs):
            if (
                isinstance(event, ReplyEndEvent)
                and event.finished_reason == ReplyFinishedReason.COMPLETED
                and not agent.state.has_awaiting_tool_calls(agent.name)
            ):
                await self._task_tracker.close_steering_window(self._run_key)
                injected = await self._inject_pending(
                    agent,
                    allow_closed_window=True,
                )
                if injected:
                    # ``_reply_impl`` resumes after the swallowed end event.
                    # Start a fresh reply budget so a newly-arrived user
                    # message does not inherit an exhausted iteration counter.
                    agent.state.reply_context = ReplyContext(
                        structured_schema=agent.state.reply_context.structured_schema,
                    )
                    continue
            yield event


__all__ = ["QueueSteeringMiddleware"]
