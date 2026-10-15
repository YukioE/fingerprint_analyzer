from .base import detector
@detector("spending_from_p2pk")
def detect(ctx): return "p2pk" in ctx.input_type_set
