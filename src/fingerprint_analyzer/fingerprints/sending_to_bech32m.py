"""Detect whether any output pays to a Bech32m address type."""

from .base import detector


@detector("sending_to_bech32m")
def detect(ctx):
    """Return whether the transaction has at least one Bech32m output."""
    return "bech32m" in ctx.output_type_set
