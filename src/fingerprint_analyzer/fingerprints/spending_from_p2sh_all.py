from .base import detector
@detector("spending_from_p2sh_all")
def detect(ctx): return bool(ctx.input_type_set & {"p2sh", "p2sh_all", "p2sh_multisig", "p2sh_p2wpkh"})
