# -*- coding: utf-8 -*-
"""Cancellation metadata shared between run tracking and runtime cleanup."""

from __future__ import annotations

from contextvars import ContextVar

CURRENT_CANCEL_REASON: ContextVar[str | None] = ContextVar(
    "qwenpaw_current_cancel_reason",
    default=None,
)

__all__ = ["CURRENT_CANCEL_REASON"]
