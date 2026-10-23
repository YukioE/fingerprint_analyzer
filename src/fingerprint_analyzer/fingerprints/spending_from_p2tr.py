"""Detect transactions spending Taproot (P2TR) outputs."""

from .base import detector, has_input_type


@detector("spending_from_p2tr")
def detect(ctx):
    """Check whether a Taproot input is present or its type is unknown."""
    return has_input_type(ctx, {"p2tr"})
