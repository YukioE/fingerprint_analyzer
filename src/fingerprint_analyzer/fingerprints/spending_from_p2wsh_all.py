from .base import detector
@detector("spending_from_p2wsh_all")
def detect(ctx): return bool(ctx.input_type_set & {"p2wsh_all", "p2wsh_multisig"})
