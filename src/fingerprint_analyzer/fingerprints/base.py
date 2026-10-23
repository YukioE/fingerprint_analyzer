"""Shared context, types, and registration helpers for fingerprint detectors."""

from collections.abc import Callable
from dataclasses import dataclass

from ..model import Transaction, TxInput, TxOutput

FingerprintDetector = Callable[["FingerprintContext"], bool | None]


@dataclass(frozen=True)
class FingerprintContext:
    """Transaction data precomputed for use by individual detectors."""

    tx: Transaction
    input_type_set: frozenset[str]
    output_type_set: frozenset[str]
    fee_rate: float | None

    @property
    def inputs(self) -> list[TxInput]:
        """Expose the transaction inputs to detector implementations."""
        return self.tx.inputs

    @property
    def outputs(self) -> list[TxOutput]:
        """Expose the transaction outputs to detector implementations."""
        return self.tx.outputs

    @property
    def rbf(self) -> bool:
        """Return whether any non-coinbase input signals opt-in RBF."""
        return any(
            inp.sequence < 0xFFFFFFFE for inp in self.tx.inputs if not inp.is_coinbase
        )


def has_input_type(ctx: FingerprintContext, input_types: set[str]) -> bool | None:
    """Match input types while preserving unknown when classification is incomplete."""
    if ctx.input_type_set & input_types:
        return True
    if ctx.input_type_set & {"unknown", "p2sh_unknown", "p2wsh_unknown"}:
        return None
    return False


def detector(name: str):
    """Register a detector under its stable feature name."""
    def register(function: FingerprintDetector) -> FingerprintDetector:
        """Attach the detector's public feature name for registry discovery."""
        function.fingerprint_name = name
        return function

    return register
