"""Detect transactions spending native P2WPKH outputs."""

from .base import detector, has_input_type


@detector("spending_from_p2wpkh")
def detect(ctx):
    """Check whether a native P2WPKH input is present or its type is unknown."""
    return has_input_type(ctx, {"p2wpkh"})
