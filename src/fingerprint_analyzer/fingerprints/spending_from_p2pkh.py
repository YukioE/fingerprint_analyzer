from .base import detector
@detector("spending_from_p2pkh")
def detect(ctx): return "p2pkh" in ctx.input_type_set
