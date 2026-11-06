"""Detect custom P2WSH script spends."""

from .base import detector, has_input_type


@detector("spending_from_p2wsh_custom")
def detect(ctx):
    """Check for native or nested P2WSH scripts that are not multisig."""
    return has_input_type(ctx, {"p2wsh_custom", "p2sh_p2wsh_custom"})
