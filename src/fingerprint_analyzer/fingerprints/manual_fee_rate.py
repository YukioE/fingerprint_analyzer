from decimal import Decimal

from .base import FingerprintContext, detector


@detector("manual_fee_rate_setting")
def detect(ctx: FingerprintContext) -> bool | None:
    if ctx.fee_rate is None:
        return None
    fee_rate = Decimal(str(ctx.fee_rate))
    distance = abs(fee_rate - fee_rate.to_integral_value())
    return distance > Decimal("0.1")
