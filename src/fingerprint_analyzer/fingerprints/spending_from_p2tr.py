from .base import detector, has_input_type


@detector("spending_from_p2tr")
def detect(ctx):
    return has_input_type(ctx, {"p2tr"})
