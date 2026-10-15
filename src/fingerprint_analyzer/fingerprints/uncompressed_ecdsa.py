from .base import detector
from .helpers import public_keys


@detector("uncompressed_ecdsa_public_keys")
def detect(ctx):
    keys = list(public_keys(ctx.tx))
    if not keys:
        return None
    return any(len(key) == 65 for key in keys)
