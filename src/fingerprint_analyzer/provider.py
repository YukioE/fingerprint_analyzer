"""Fetch Bitcoin transaction and block data from an Electrs-compatible API."""

import asyncio
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from .model import PrevOut, Transaction, TxInput, TxOutput, TxStatus


class ProviderError(Exception):
    """Raised when a transaction cannot be fetched or parsed."""


class TransactionNotFoundError(ProviderError):
    """Raised when the requested transaction does not exist at the provider."""

    pass


class TransactionProvider(ABC):
    """Interface implemented by asynchronous transaction data providers."""

    @abstractmethod
    async def get_transaction(self, txid: str) -> Transaction:
        """Fetch and parse a transaction by its transaction ID."""
        raise NotImplementedError


class ElectrsProvider(TransactionProvider):
    """HTTP client for transaction and block endpoints exposed by Electrs APIs."""

    def __init__(self, base_url: str = "https://mempool.space/api", timeout: int = 15):
        """Set the API base URL and request timeout."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def get_transaction(self, txid: str) -> Transaction:
        """Fetch transaction JSON and convert it to the internal data model."""
        url = f"{self.base_url}/tx/{txid}"
        raw = await asyncio.to_thread(self._fetch_json, url)
        if not isinstance(raw, dict):
            raise ProviderError(f"invalid transaction response for {txid}")
        return self._parse(raw)

    async def get_output_spends(self, txid: str) -> list[str | None]:
        """Return the spending transaction ID for each output, if it is spent."""
        url = f"{self.base_url}/tx/{txid}/outspends"
        raw = await asyncio.to_thread(self._fetch_json, url)
        if not isinstance(raw, list):
            raise ProviderError(f"invalid output-spends response for {txid}")

        spending_txids = []
        for output_index, spend in enumerate(raw):
            if not isinstance(spend, dict) or not isinstance(spend.get("spent"), bool):
                raise ProviderError(
                    f"invalid output-spends entry for {txid} at output {output_index}"
                )
            if not spend["spent"]:
                spending_txids.append(None)
                continue

            spending_txid = spend.get("txid")
            if not isinstance(spending_txid, str):
                raise ProviderError(
                    f"missing spending transaction ID for {txid} output {output_index}"
                )
            spending_txids.append(spending_txid)

        return spending_txids

    async def get_tip_height(self) -> int:
        """Return the height of the most recently mined block."""
        raw = await asyncio.to_thread(
            self._fetch_text, f"{self.base_url}/blocks/tip/height"
        )
        try:
            height = int(raw.strip())
        except (TypeError, ValueError) as exc:
            raise ProviderError("invalid tip-height response") from exc
        if height < 0:
            raise ProviderError("invalid tip-height response")
        return height

    async def get_block_hash(self, height: int) -> str:
        """Return the block hash at a given height."""
        raw = await asyncio.to_thread(
            self._fetch_text, f"{self.base_url}/block-height/{height}"
        )
        block_hash = raw.strip()
        if not block_hash:
            raise ProviderError(f"invalid block-hash response at height {height}")
        return block_hash

    async def get_block_timestamp(self, block_hash: str) -> int:
        """Return the Unix timestamp recorded for a block."""
        raw = await asyncio.to_thread(
            self._fetch_json, f"{self.base_url}/block/{block_hash}"
        )
        if not isinstance(raw, dict) or not isinstance(raw.get("timestamp"), int):
            raise ProviderError(f"invalid block response for {block_hash}")
        return raw["timestamp"]

    async def get_block_txids(self, block_hash: str) -> list[str]:
        """Return all transaction IDs included in a block."""
        raw = await asyncio.to_thread(
            self._fetch_json, f"{self.base_url}/block/{block_hash}/txids"
        )
        if not isinstance(raw, list) or not raw or any(
            not isinstance(txid, str) or not txid for txid in raw
        ):
            raise ProviderError(f"invalid transaction list for block {block_hash}")
        return raw

    def _fetch_json(self, url: str) -> object:
        """Fetch a URL and decode its response as JSON."""
        body = self._fetch_text(url)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"invalid JSON response from {url}") from exc

    def _fetch_text(self, url: str) -> str:
        """Fetch a URL as UTF-8 text and translate network errors to provider errors."""
        request = urllib.request.Request(
            url, headers={"User-Agent": "fingerprint-analyzer/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                if url.startswith(f"{self.base_url}/tx/"):
                    raise TransactionNotFoundError(
                        f"transaction not found: {url}"
                    ) from exc
                raise ProviderError(f"resource not found: {url}") from exc
            raise ProviderError(f"HTTP {exc.code} while fetching transaction") from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise ProviderError(f"request failed: {exc}") from exc

    def _parse(self, raw: dict) -> Transaction:
        """Convert a provider transaction object into typed transaction data."""
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
