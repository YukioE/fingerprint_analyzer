"""Detect Taproot script-path spends with non-multisig scripts."""

from .base import detector, has_input_type


@detector("spending_from_p2tr_scriptpath_custom")
def detect(ctx):
    """Check for a Taproot script-path input not classified as multisig."""
    return has_input_type(ctx, {"p2tr_scriptpath_custom"})
