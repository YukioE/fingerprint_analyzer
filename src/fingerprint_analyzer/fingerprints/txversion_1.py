from .base import detector
@detector("txversion_1")
def detect(ctx): return ctx.tx.version == 1
