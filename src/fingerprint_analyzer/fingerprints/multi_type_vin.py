from .base import detector


@detector("multi_type_vin")
def detect(ctx):
    known_types = ctx.input_type_set - {"unknown"}
    if len(known_types) > 1:
        return True
    if "unknown" in ctx.input_type_set:
        return None
    return False
