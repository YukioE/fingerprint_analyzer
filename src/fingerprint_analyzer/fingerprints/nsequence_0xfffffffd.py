"""Detect inputs signaling opt-in Replace-By-Fee."""

from .base import detector


@detector("nsequence_0xfffffffd")
def detect(ctx):
    """Return whether any non-coinbase input signals BIP-125 opt-in RBF."""
    return any(
        inp.sequence < 0xFFFFFFFE for inp in ctx.inputs if not inp.is_coinbase
    )
