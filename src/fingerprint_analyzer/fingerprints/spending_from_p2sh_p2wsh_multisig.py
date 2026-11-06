"""Detect nested P2SH-P2WSH multisignature spends."""

from .base import detector, has_input_type


@detector("spending_from_p2sh_p2wsh_multisig")
def detect(ctx):
    """Check for a P2SH-wrapped P2WSH multisignature input."""
    return has_input_type(ctx, {"p2sh_p2wsh_multisig"})
