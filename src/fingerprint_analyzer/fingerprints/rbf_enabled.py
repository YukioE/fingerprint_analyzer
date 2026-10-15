from .base import detector
@detector("default_opt_in_rbf_set")
def detect(ctx): return ctx.rbf
