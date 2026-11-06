"""Detect transactions with more than one input sequence value."""

from .base import detector


@detector("nsequence_mixed")
def detect(ctx):
    """Return whether non-coinbase inputs use different sequence values."""
    sequences = {inp.sequence for inp in ctx.inputs if not inp.is_coinbase}
    return len(sequences) > 1
