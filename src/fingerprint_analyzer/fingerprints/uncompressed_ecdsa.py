"""Detect whether a transaction uses uncompressed ECDSA public keys."""

from .base import detector
from .helpers import public_keys


@detector("uncompressed_ecdsa_public_keys")
def detect(ctx):
    """Return whether at least one discovered public key is uncompressed."""
    keys = list(public_keys(ctx.tx))
    if not keys:
        return None
    return any(len(key) == 65 for key in keys)
