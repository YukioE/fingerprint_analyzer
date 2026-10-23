"""Detect Bitcoin transactions that use transaction version 2."""

from .base import detector


@detector("txversion_2")
def detect(ctx):
    """Return whether the transaction version is exactly 2."""
    return ctx.tx.version == 2
