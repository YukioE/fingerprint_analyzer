"""Detect Taproot key-path spends."""

from .base import detector, has_input_type


@detector("spending_from_p2tr_keypath")
def detect(ctx):
    """Check for a Taproot key-path input."""
    return has_input_type(ctx, {"p2tr_keypath"})
