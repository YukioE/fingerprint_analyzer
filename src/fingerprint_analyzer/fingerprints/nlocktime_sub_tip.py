"""Detect nonzero locktimes below the transaction's block height."""

from .base import detector


@detector("nlocktime_sub_tip")
def detect(ctx):
    """Return whether a confirmed transaction's locktime is below its block."""
    if ctx.tx.locktime == 0:
        return False
    height = ctx.tx.status.block_height
    if height is None:
        return None
    return ctx.tx.locktime < height
