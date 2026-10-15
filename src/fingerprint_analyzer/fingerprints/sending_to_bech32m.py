from .base import detector
@detector("sending_to_bech32m")
def detect(ctx): return "bech32m" in ctx.output_type_set
