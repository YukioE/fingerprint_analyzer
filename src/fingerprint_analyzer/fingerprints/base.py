from collections.abc import Callable
from dataclasses import dataclass

from ..model import Transaction, TxInput, TxOutput

FingerprintDetector = Callable[["FingerprintContext"], bool | None]


@dataclass(frozen=True)
class FingerprintContext:
    tx: Transaction
    input_type_set: frozenset[str]
    output_type_set: frozenset[str]
    fee_rate: float | None

    @property
    def inputs(self) -> list[TxInput]:
        return self.tx.inputs

    @property
    def outputs(self) -> list[TxOutput]:
        return self.tx.outputs

    @property
    def rbf(self) -> bool:
        return any(
            inp.sequence < 0xFFFFFFFE for inp in self.tx.inputs if not inp.is_coinbase
        )


def has_input_type(ctx: FingerprintContext, input_types: set[str]) -> bool | None:
    if ctx.input_type_set & input_types:
        return True
    if ctx.input_type_set & {"unknown", "p2sh_unknown", "p2wsh_unknown"}:
        return None
    return False


def detector(name: str):
    def register(function: FingerprintDetector) -> FingerprintDetector:
        function.fingerprint_name = name
        return function

    return register
