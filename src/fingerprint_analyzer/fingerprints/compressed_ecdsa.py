"""Detect whether a transaction uses compressed ECDSA public keys."""

from .base import detector
from .helpers import public_keys


@detector("compressed_ecdsa_public_keys")
def detect(ctx):
    """Return whether at least one discovered public key is compressed."""
    keys = list(public_keys(ctx.tx))
    if not keys:
        return None
    return any(len(key) == 33 for key in keys)
