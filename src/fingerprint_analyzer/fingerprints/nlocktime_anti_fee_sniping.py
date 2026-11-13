"""Detect locktimes within 110 blocks of the transaction's inclusion block."""

from .base import detector


@detector("nlocktime_anti_fee_sniping")
def detect(ctx):
    """Return whether locktime is within 110 blocks before the confirmed height."""
    if ctx.tx.locktime == 0:
        return False
    height = ctx.tx.status.block_height
    if height is None:
        return None
    return height - 110 <= ctx.tx.locktime <= height
