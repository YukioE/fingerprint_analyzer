"""Detect inputs that enable transaction locktime without opt-in RBF."""

from .base import detector


@detector("nsequence_0xfffffffe")
def detect(ctx):
    """Return whether any non-coinbase input uses the locktime-only sequence."""
    return any(
        inp.sequence == 0xFFFFFFFE for inp in ctx.inputs if not inp.is_coinbase
    )
