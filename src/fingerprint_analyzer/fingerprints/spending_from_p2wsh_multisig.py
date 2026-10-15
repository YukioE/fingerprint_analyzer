from .base import detector
@detector("spending_from_p2wsh_multisig")
def detect(ctx): return "p2wsh_multisig" in ctx.input_type_set
