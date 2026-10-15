from .base import detector
from .helpers import is_ecdsa_signature, script_bytes, script_pushes


@detector("low_r_grinding")
def detect(ctx):
    signatures = []
    for inp in ctx.tx.inputs:
        signatures.extend(script_bytes(item) for item in (inp.witness or []))
        if inp.scriptsig:
            signatures.extend(script_pushes(script_bytes(inp.scriptsig)))

    ecdsa_signatures = [
        signature for signature in signatures if is_ecdsa_signature(signature)
    ]
    if not ecdsa_signatures:
        return None
    return all(signature[3] == 32 for signature in ecdsa_signatures)
