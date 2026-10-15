from dataclasses import asdict, dataclass

from .fingerprints import DETECTORS, FingerprintContext
from .fingerprints.helpers import input_type, output_type
from .model import Transaction


@dataclass
class FingerprintReport:
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
    input_type_set: list[str]
    output_type_set: list[str]
    fingerprints: dict[str, bool | None]
    details: dict[str, object]

    def to_dict(self) -> dict:
        report = asdict(self)
        details = report.pop("details")
        flattened = {}
        for name, value in report.items():
            flattened[name] = value
            if name == "output_type_set":
                flattened.update(details)
        return flattened


def _signature_byte_from_script(script: bytes) -> int | None:
    if not script or len(script) < 2:
        return None
    push_length = script[0]
    if push_length == 0 or push_length >= len(script):
        return None
    signature = script[1 : 1 + push_length]
    if not signature:
        return None
    return signature[-1]


def get_sighash_bytes(tx: Transaction) -> set[int]:
    sighashes: set[int] = set()

    for txin in tx.inputs:
        if txin.scriptsig:
            script = bytes.fromhex(txin.scriptsig)
            signature_byte = _signature_byte_from_script(script)
            if signature_byte is not None:
                sighashes.add(signature_byte)

        if txin.witness:
            for item in txin.witness:
                data = bytes.fromhex(item)
                if len(data) > 1 and data[0] == 0x30:
                    sighashes.add(data[-1])

    return sighashes


def analyze(tx: Transaction) -> FingerprintReport:
    input_types = {input_type(inp) for inp in tx.inputs}
    output_types = {output_type(output) for output in tx.outputs}
    sequence_hex = [f"0x{inp.sequence:08x}" for inp in tx.inputs if not inp.is_coinbase]
    sequence_set = sorted(set(sequence_hex))
    sighash_set = get_sighash_bytes(tx)
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
        input_type_set=sorted(input_types),
        output_type_set=sorted(output_types),
        fingerprints=fingerprints,
        details={
            "sequence_set": sequence_set,
            "sighash_set_detail": sorted(sighash_set),
        },
    )
