"""Detect transactions whose absolute locktime is zero."""

from .base import detector


@detector("nlocktime_zero")
def detect(ctx):
    """Return whether nLockTime is zero."""
    return ctx.tx.locktime == 0
