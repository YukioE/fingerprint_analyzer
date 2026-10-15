from .base import detector
@detector("spending_from_p2tr")
def detect(ctx): return "p2tr" in ctx.input_type_set
