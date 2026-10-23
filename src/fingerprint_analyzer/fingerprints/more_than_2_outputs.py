"""Detect transactions with more than two outputs."""

from .base import detector


@detector("more_than_2_outputs")
def detect(ctx):
    """Return whether the transaction has more than two outputs."""
    return len(ctx.outputs) > 2
