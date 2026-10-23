"""Detect whether a transaction includes an OP_RETURN output."""

from .base import detector


@detector("op_return_outputs")
def detect(ctx):
    """Return whether any output uses the OP_RETURN script type."""
    return "op_return" in ctx.output_type_set
