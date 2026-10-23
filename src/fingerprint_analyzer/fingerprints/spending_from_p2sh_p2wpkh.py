"""Detect transactions spending nested P2SH-wrapped P2WPKH outputs."""

from .base import detector, has_input_type


@detector("spending_from_p2sh_p2wpkh")
def detect(ctx):
    """Check whether a P2SH-wrapped P2WPKH input is present."""
    return has_input_type(ctx, {"p2sh_p2wpkh"})
