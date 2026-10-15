from .base import detector
@detector("multi_type_vin")
def detect(ctx): return len(ctx.input_type_set - {"unknown"}) > 1
