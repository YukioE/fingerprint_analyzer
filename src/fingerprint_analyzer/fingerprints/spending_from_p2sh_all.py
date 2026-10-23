"""Detect transactions spending any supported P2SH output."""

from .base import detector, has_input_type


@detector("spending_from_p2sh_all")
def detect(ctx):
    """Match P2SH inputs, including nested witness and unknown subtypes."""
    return has_input_type(
        ctx,
        {
            "p2sh",
            "p2sh_all",
            "p2sh_unknown",
            "p2sh_multisig",
            "p2sh_p2wpkh",
            "p2sh_p2wsh",
        },
    )
