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


def detector(name: str):
    def register(function: FingerprintDetector) -> FingerprintDetector:
        function.fingerprint_name = name
        return function

    return register
