"""Detect whether any output pays to a Bech32 address type."""

from .base import detector


@detector("sending_to_bech32")
def detect(ctx):
    """Return whether the transaction has at least one Bech32 output."""
    return "bech32" in ctx.output_type_set
