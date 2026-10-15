from .base import detector
@detector("sending_to_base58")
def detect(ctx): return "base58" in ctx.output_type_set
