"""Detect inputs that disable both locktime and opt-in RBF."""

from .base import detector


@detector("nsequence_0xffffffff")
def detect(ctx):
    """Return whether any non-coinbase input uses the final sequence value."""
    return any(
        inp.sequence == 0xFFFFFFFF for inp in ctx.inputs if not inp.is_coinbase
    )
