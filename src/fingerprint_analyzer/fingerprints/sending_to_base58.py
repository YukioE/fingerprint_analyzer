"""Detect whether any output pays to a Base58 address type."""

from .base import detector


@detector("sending_to_base58")
def detect(ctx):
    """Return whether the transaction has at least one Base58 output."""
    return "base58" in ctx.output_type_set
