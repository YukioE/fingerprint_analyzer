from .base import detector
@detector("txversion_2")
def detect(ctx): return ctx.tx.version == 2
