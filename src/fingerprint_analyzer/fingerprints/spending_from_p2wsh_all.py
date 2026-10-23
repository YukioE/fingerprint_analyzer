"""Detect transactions spending any supported P2WSH output."""

from .base import detector, has_input_type


@detector("spending_from_p2wsh_all")
def detect(ctx):
    """Match P2WSH inputs, including multisignature and unknown subtypes."""
    return has_input_type(ctx, {"p2wsh_all", "p2wsh_multisig", "p2wsh_unknown"})
