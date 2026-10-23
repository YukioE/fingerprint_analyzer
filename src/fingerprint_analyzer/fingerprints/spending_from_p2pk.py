"""Detect transactions spending outputs protected by P2PK."""

from .base import detector, has_input_type


@detector("spending_from_p2pk")
def detect(ctx):
    """Check whether a P2PK input is present or its type is unknown."""
    return has_input_type(ctx, {"p2pk"})
