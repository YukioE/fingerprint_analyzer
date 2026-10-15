from .base import detector
@detector("sending_to_bech32")
def detect(ctx): return "bech32" in ctx.output_type_set
