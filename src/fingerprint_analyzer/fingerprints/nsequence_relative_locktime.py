"""Detect inputs that enable BIP-68 relative locktime."""

from .base import detector


@detector("nsequence_relative_locktime")
def detect(ctx):
    """Return whether any non-coinbase sequence has its disable flag cleared."""
    return any(
        inp.sequence & 0x80000000 == 0
        for inp in ctx.inputs
        if not inp.is_coinbase
    )
