"""Heuristically detect inputs ordered from newest UTXO to oldest."""

from .base import detector


@detector("coin_selection_lifo")
def detect(ctx):
    """Match strictly descending input prevout heights when all are known."""
    inputs = [inp for inp in ctx.inputs if not inp.is_coinbase]
    heights = [inp.prevout_block_height for inp in inputs]
    if len(heights) < 2 or any(height is None for height in heights):
        return None
    known_heights = [height for height in heights if height is not None]
    if len(set(known_heights)) != len(known_heights):
        return None
    return known_heights == sorted(known_heights, reverse=True)
