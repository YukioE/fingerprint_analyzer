"""Detect transactions spending outputs protected by P2PKH."""

from .base import detector, has_input_type


@detector("spending_from_p2pkh")
def detect(ctx):
    """Check whether a P2PKH input is present or its type is unknown."""
    return has_input_type(ctx, {"p2pkh"})
