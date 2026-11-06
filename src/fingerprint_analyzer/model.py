"""Small data classes representing the transaction fields used by the analyzer."""

from dataclasses import dataclass


@dataclass
class PrevOut:
    """The previous output referenced by a transaction input."""

    scriptpubkey: str
    scriptpubkey_asm: str
    scriptpubkey_type: str
    value: int
    scriptpubkey_address: str | None = None


@dataclass
class TxInput:
    """An input, including its unlocking data and referenced previous output."""

    txid: str
    vout: int
    sequence: int
    is_coinbase: bool = False
    prevout: PrevOut | None = None
    scriptsig: str = ""
    scriptsig_asm: str = ""
    witness: list[str] | None = None


@dataclass
class TxOutput:
    """An output's value and locking-script information."""

    value: int
    scriptpubkey: str
    scriptpubkey_asm: str
    scriptpubkey_type: str
    scriptpubkey_address: str = ""


@dataclass
class TxStatus:
    """Confirmation status and optional block metadata."""

    confirmed: bool
    block_height: int | None = None
    block_hash: str | None = None
    block_time: int | None = None


@dataclass
class Transaction:
    """The transaction fields needed for fingerprint analysis."""

    txid: str
    version: int
    locktime: int
    inputs: list[TxInput]
    outputs: list[TxOutput]
    size: int
    weight: int
    fee: int | None
    status: TxStatus
    sigops: int | None = None

    @property
    def vbytes(self) -> int:
        """Return virtual size, calculated from weight when it is available."""
        if self.weight:
            return (self.weight + 3) // 4
        return self.size
