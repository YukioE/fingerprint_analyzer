"""Detect transactions spending P2WSH multisignature outputs."""

from .base import detector, has_input_type


@detector("spending_from_p2wsh_multisig")
def detect(ctx):
    """Check whether a P2WSH multisignature input is present."""
    return has_input_type(ctx, {"p2wsh_multisig"})
