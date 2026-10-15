import asyncio
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from .model import PrevOut, Transaction, TxInput, TxOutput, TxStatus


class ProviderError(Exception):
    """Raised when a transaction cannot be fetched or parsed."""


class TransactionNotFoundError(ProviderError):
    pass


class TransactionProvider(ABC):
    @abstractmethod
    async def get_transaction(self, txid: str) -> Transaction:
        raise NotImplementedError


class ElectrsProvider(TransactionProvider):
    def __init__(self, base_url: str = "https://mempool.space/api", timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def get_transaction(self, txid: str) -> Transaction:
        url = f"{self.base_url}/tx/{txid}"
        raw = await asyncio.to_thread(self._fetch_json, url)
        return self._parse(raw)

    def _fetch_json(self, url: str) -> dict:
        request = urllib.request.Request(
            url, headers={"User-Agent": "fingerprint-analyzer/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise TransactionNotFoundError(f"transaction not found: {url}") from exc
            raise ProviderError(f"HTTP {exc.code} while fetching transaction") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderError(f"request failed: {exc}") from exc

    def _parse(self, raw: dict) -> Transaction:
        inputs = []
        for vin in raw.get("vin", []):
            prevout = vin.get("prevout")
            inputs.append(
                TxInput(
                    txid=vin.get("txid", ""),
                    vout=vin.get("vout", 0),
                    sequence=vin.get("sequence", 0xFFFFFFFF),
                    is_coinbase=vin.get("is_coinbase", "coinbase" in vin),
                    prevout=PrevOut(**prevout) if prevout else None,
                    scriptsig=vin.get("scriptsig", ""),
                    scriptsig_asm=vin.get("scriptsig_asm", ""),
                    witness=vin.get("witness", []),
                )
            )

        outputs = []
        for vout in raw.get("vout", []):
            outputs.append(
                TxOutput(
                    value=vout.get("value", 0),
                    scriptpubkey=vout.get("scriptpubkey", ""),
                    scriptpubkey_asm=vout.get("scriptpubkey_asm", ""),
                    scriptpubkey_type=vout.get("scriptpubkey_type", "unknown"),
                    scriptpubkey_address=vout.get("scriptpubkey_address", ""),
                )
            )

        status = raw.get("status", {})
        return Transaction(
            txid=raw["txid"],
            version=raw.get("version", 1),
            locktime=raw.get("locktime", 0),
            inputs=inputs,
            outputs=outputs,
            size=raw.get("size", 0),
            weight=raw.get("weight", 0),
            fee=raw.get("fee"),
            sigops=raw.get("sigops"),
            status=TxStatus(
                confirmed=status.get("confirmed", False),
                block_height=status.get("block_height"),
                block_hash=status.get("block_hash"),
                block_time=status.get("block_time"),
            ),
        )

