from .base import detector
@detector("spending_from_p2sh_multisig")
def detect(ctx): return "p2sh_multisig" in ctx.input_type_set
