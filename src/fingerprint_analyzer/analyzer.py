"""Build a structured fingerprint report from a Bitcoin transaction."""

from dataclasses import asdict, dataclass

from .fingerprints import DETECTORS, FingerprintContext
from .fingerprints.helpers import input_type, output_type
from .model import Transaction


@dataclass
class FingerprintReport:
    """Analysis results and transaction details ready for JSON output."""

    txid: str
    version: int
    locktime: int
    confirmed: bool
    block_height: int | None
    input_count: int
    output_count: int
    fee_sats: int | None
    vbytes: int
    fee_rate_sat_vb: float | None
    fingerprints: dict[str, bool | None]

    def to_dict(self) -> dict:
        """Convert the report to a flat dictionary for the command-line output."""
        return asdict(self)


def analyze(tx: Transaction) -> FingerprintReport:
    """Run all registered detectors and summarize the transaction."""
    input_types = {input_type(inp) for inp in tx.inputs if not inp.is_coinbase}
    output_types = {output_type(output) for output in tx.outputs}
    fee_rate = tx.fee / tx.vbytes if tx.fee is not None and tx.vbytes else None

    context = FingerprintContext(
        tx=tx,
        input_type_set=frozenset(input_types),
        output_type_set=frozenset(output_types),
        fee_rate=fee_rate,
    )
    fingerprints = {
        detector.fingerprint_name: detector(context) for detector in DETECTORS
    }

    return FingerprintReport(
        txid=tx.txid,
        version=tx.version,
        locktime=tx.locktime,
        confirmed=tx.status.confirmed,
        block_height=tx.status.block_height,
        input_count=len(tx.inputs),
        output_count=len(tx.outputs),
        fee_sats=tx.fee,
        vbytes=tx.vbytes,
        fee_rate_sat_vb=round(fee_rate, 3) if fee_rate is not None else None,
        fingerprints=fingerprints,
    )
