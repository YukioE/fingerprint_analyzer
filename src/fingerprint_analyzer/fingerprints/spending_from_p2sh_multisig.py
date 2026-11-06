"""Detect transactions spending legacy P2SH multisignature outputs."""

from .base import detector, has_input_type


@detector("spending_from_p2sh_multisig")
def detect(ctx):
    """Check for a P2SH multisignature spend, excluding nested witness scripts."""
    return has_input_type(ctx, {"p2sh_multisig"})
