from .base import detector
from .helpers import public_keys


@detector("compressed_ecdsa_public_keys")
def detect(ctx):
    keys = list(public_keys(ctx.tx))
    if not keys:
        return None
    return any(len(key) == 33 for key in keys)
