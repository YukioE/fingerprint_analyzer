"""Detect transactions that spend inputs from multiple script types."""

from .base import detector


@detector("multi_type_vin")
def detect(ctx):
    """Return unknown if an unclassified input prevents a confident result."""
    known_types = ctx.input_type_set - {"unknown"}
    if len(known_types) > 1:
        return True
    if "unknown" in ctx.input_type_set:
        return None
    return False
