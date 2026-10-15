from .base import detector
@detector("more_than_2_outputs")
def detect(ctx): return len(ctx.outputs) > 2
