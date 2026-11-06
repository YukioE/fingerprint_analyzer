"""Detect transactions matching the requested unnecessary-input value rule."""

from .base import detector


@detector("unnecessary_input")
def detect(ctx):
    """Match when the minimum output is at least the minimum input value."""
    inputs = [inp for inp in ctx.inputs if not inp.is_coinbase]
    if not inputs or not ctx.outputs or any(inp.prevout is None for inp in inputs):
        return None
    return min(output.value for output in ctx.outputs) >= min(
        inp.prevout.value for inp in inputs if inp.prevout is not None
    )
