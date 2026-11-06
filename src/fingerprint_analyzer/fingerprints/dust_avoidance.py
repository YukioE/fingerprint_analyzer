"""Detect whether supported spendable outputs avoid standard dust values."""

from .base import detector
from .helpers import dust_threshold, output_type


@detector("dust_avoidance")
def detect(ctx):
    """Return whether every output is above its script-specific dust cutoff."""
    outputs = [output for output in ctx.outputs if output_type(output) != "op_return"]
    if not outputs:
        return None
    thresholds = [dust_threshold(output) for output in outputs]
    if any(threshold is None for threshold in thresholds):
        return None
    return all(output.value >= threshold for output, threshold in zip(outputs, thresholds))
