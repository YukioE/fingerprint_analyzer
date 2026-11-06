"""Detect Taproot script-path threshold-signature spends."""

from .base import detector, has_input_type


@detector("spending_from_p2tr_scriptpath_multisig")
def detect(ctx):
    """Check for a Taproot script-path multisignature input."""
    return has_input_type(ctx, {"p2tr_scriptpath_multisig"})
