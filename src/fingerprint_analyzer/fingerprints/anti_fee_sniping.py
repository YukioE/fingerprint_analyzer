"""Detect transactions whose locktime appears to avoid fee sniping."""

from .base import FingerprintContext, detector


@detector("anti_fee_sniping")
def detect(ctx: FingerprintContext) -> bool | None:
    """Return whether locktime is set below the confirmed block height."""
    height = ctx.tx.status.block_height
    return (ctx.tx.locktime > 0 and ctx.tx.locktime <= height) if height is not None else None
