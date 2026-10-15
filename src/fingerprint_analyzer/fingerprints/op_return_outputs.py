from .base import detector
@detector("op_return_outputs")
def detect(ctx): return "op_return" in ctx.output_type_set
