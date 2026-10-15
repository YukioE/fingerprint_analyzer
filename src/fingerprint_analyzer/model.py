from dataclasses import dataclass


@dataclass
class PrevOut:
    scriptpubkey: str
    scriptpubkey_asm: str
    scriptpubkey_type: str
    value: int
    scriptpubkey_address: str | None = None


@dataclass
class TxInput:
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
    value: int
    scriptpubkey: str
    scriptpubkey_asm: str
    scriptpubkey_type: str
    scriptpubkey_address: str = ""


@dataclass
class TxStatus:
    confirmed: bool
    block_height: int | None = None
    block_hash: str | None = None
    block_time: int | None = None


@dataclass
class Transaction:
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
        if self.weight:
            return (self.weight + 3) // 4
        return self.size

