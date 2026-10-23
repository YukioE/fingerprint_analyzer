"""Detect transactions spending P2SH multisignature outputs."""

from .base import detector, has_input_type


@detector("spending_from_p2sh_multisig")
def detect(ctx):
    """Check whether a P2SH multisignature input is present."""
    return has_input_type(ctx, {"p2sh_multisig"})
