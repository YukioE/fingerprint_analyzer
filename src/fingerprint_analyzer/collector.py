"""Collect confirmed transactions and persist their fingerprints by block."""

import asyncio
import sys
from datetime import datetime, timezone

from .analyzer import FingerprintReport, analyze
from .database import FingerprintStore
from .provider import ElectrsProvider

MAX_CONCURRENCY = 64


async def collect_forever(
    provider: ElectrsProvider,
    store: FingerprintStore,
    start_height: int | None,
    poll_interval: float,
    block_interval: int = 1,
) -> None:
    """Continuously analyze blocks, resuming after the latest stored height."""
    if block_interval <= 0:
        raise ValueError("block_interval must be greater than zero")

    tip_height = await provider.get_tip_height()
    latest_height = store.get_latest_height()
    if latest_height is None:
        next_height = tip_height if start_height is None else start_height
        if next_height > tip_height:
            raise ValueError(
                f"start height {next_height} is above current tip {tip_height}"
            )
    else:
        next_height = latest_height + 1

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def analyze_tx(txid: str) -> FingerprintReport:
        """Fetch and analyze one transaction while respecting the concurrency limit."""
        async with semaphore:
            transaction = await provider.get_transaction(txid)
        return analyze(transaction)

    while True:
        tip_height = await provider.get_tip_height()
        while next_height <= tip_height:
            # Save a complete block before advancing so a failed block can be retried.
            block_hash = await provider.get_block_hash(next_height)
            timestamp = datetime.fromtimestamp(
                await provider.get_block_timestamp(block_hash), tz=timezone.utc
            )
            txids = await provider.get_block_txids(block_hash)
            reports = await asyncio.gather(*(analyze_tx(txid) for txid in txids))
            store.save_block(next_height, block_hash, timestamp, reports)
            print(
                f"Stored block {next_height}: {len(reports)} transactions",
                file=sys.stderr,
            )
            next_height += block_interval
        await asyncio.sleep(poll_interval)
