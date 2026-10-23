"""Detect fee rates that may have been selected manually."""

from decimal import Decimal

from .base import FingerprintContext, detector


@detector("manual_fee_rate_setting")
def detect(ctx: FingerprintContext) -> bool | None:
    """Treat fee rates more than 0.1 sat/vB from an integer as manual."""
    if ctx.fee_rate is None:
        return None
    fee_rate = Decimal(str(ctx.fee_rate))
    distance = abs(fee_rate - fee_rate.to_integral_value())
    return distance > Decimal("0.1")
