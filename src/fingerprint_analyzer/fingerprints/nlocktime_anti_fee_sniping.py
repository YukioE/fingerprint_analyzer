"""Detect locktimes matching the transaction's inclusion block."""

from .base import detector


@detector("nlocktime_anti_fee_sniping")
def detect(ctx):
    """Return whether locktime is the confirmed block height or one below it."""
    if ctx.tx.locktime == 0:
        return False
    height = ctx.tx.status.block_height
    if height is None:
        return None
    return ctx.tx.locktime in (height, height - 1)
