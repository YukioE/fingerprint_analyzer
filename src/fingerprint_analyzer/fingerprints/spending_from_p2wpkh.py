from .base import detector
@detector("spending_from_p2wpkh")
def detect(ctx): return "p2wpkh" in ctx.input_type_set
