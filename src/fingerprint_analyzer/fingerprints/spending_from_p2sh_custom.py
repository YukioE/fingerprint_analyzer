"""Detect custom P2SH script spends."""

from .base import detector, has_input_type


@detector("spending_from_p2sh_custom")
def detect(ctx):
    """Check for P2SH scripts that are neither multisig nor wrapped P2WPKH."""
    return has_input_type(ctx, {"p2sh_custom", "p2sh_p2wsh_custom"})
