"""Inject queued console messages at the model-request boundary."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Callable

from agentscope.middleware import MiddlewareBase

logger = logging.getLogger(__name__)


class QueueSteeringMiddleware(MiddlewareBase):
    """Consume chat steering messages at AgentScope's safe model boundaries.

    Two consumption points, both of which keep the provider request and the
    agent context consistent:

    * ``on_model_call`` — the queued messages are appended to the context and
      the outgoing request is rebuilt, so they reach the provider on the very
      request that follows. This is what "before the next model request"
      means; injecting only into ``agent.state.context`` from ``on_reasoning``
      relied on the agent re-preparing its input afterwards and silently did
      nothing when it did not.
    * ``on_reply`` — a completed ``ReplyEndEvent`` is swallowed only after its
      assistant output and tool results are complete, which resumes the reply
      loop for messages that arrived too late for any model call.

    Messages are never injected while a user-confirmation or external
    execution is parked: the agent is waiting for that specific answer and a
    queued user message would corrupt the HITL protocol.
    """

    def __init__(self, task_tracker: Any, run_key: str) -> None:
        self._task_tracker = task_tracker
        self._run_key = run_key

    async def _inject_pending(
        self,
        agent: Any,
        *,
        allow_closed_window: bool = False,
    ) -> list[Any]:
        """Take and append queued messages, returning the ones injected."""
        if agent.state.has_awaiting_tool_calls(agent.name):
            return []
        messages = await self._task_tracker.take_steering(
            self._run_key,
            allow_closed_window=allow_closed_window,
        )
        if not messages:
            return []
        try:
            await agent._handle_incoming_messages(messages)
        except Exception:
            # Put the batch back so a later boundary can retry it, and keep the
            # window open: the run is still healthy, only this injection failed.
            await self._task_tracker.open_steering_window(self._run_key)
            await self._task_tracker.restore_steering(self._run_key, messages)
            logger.exception(
                "queue steering injection failed run_key=%s receipts=%s",
                self._run_key,
                self._receipts(messages),
            )
            return []
        return messages

    @staticmethod
    def _receipts(messages: list[Any]) -> list[str]:
        receipts: list[str] = []
        for message in messages:
            metadata = getattr(message, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            receipt = metadata.get("qwenpaw_client_message_id")
            if receipt:
                receipts.append(str(receipt))
        return receipts

    async def on_reasoning(
        self,
        agent: Any,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        # Acceptance is re-opened at every reasoning boundary; consumption is
        # deferred to on_model_call so the messages land in a real request.
        await self._task_tracker.open_steering_window(self._run_key)
        async for event in next_handler(**input_kwargs):
            yield event

    async def on_model_call(
        self,
        agent: Any,
        input_kwargs: dict,
        next_handler: Callable[..., Any],
    ) -> Any:
        """Inject pending messages into the exact request being sent."""
        await self._task_tracker.open_steering_window(self._run_key)
        messages = await self._inject_pending(agent)
        if messages:
            # ``_reasoning_impl`` prepared its input before this hook ran, so
            # the request must be rebuilt or the injected messages would only
            # affect a later turn.
            prepared = await agent._prepare_model_input()
            input_kwargs = {**input_kwargs, **prepared}
            logger.info(
                "queue steering included in model request run_key=%s "
                "message_count=%d receipts=%s",
                self._run_key,
                len(messages),
                self._receipts(messages),
            )
        return await next_handler(**input_kwargs)

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
                    logger.info(
                        "queue steering resumed reply run_key=%s receipts=%s",
                        self._run_key,
                        self._receipts(injected),
                    )
                    continue
            yield event


__all__ = ["QueueSteeringMiddleware"]
