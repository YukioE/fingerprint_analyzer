from .base import detector
from .helpers import script_bytes


@detector("bip69")
def detect(ctx):
    inputs = [(script_bytes(inp.txid), inp.vout) for inp in ctx.tx.inputs]
    outputs = [
        (output.value, script_bytes(output.scriptpubkey)) for output in ctx.tx.outputs
    ]
    return inputs == sorted(inputs) and outputs == sorted(outputs)
