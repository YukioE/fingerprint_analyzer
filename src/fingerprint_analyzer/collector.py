"""Collect confirmed transactions and persist their fingerprints by block."""

import asyncio
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .analyzer import FingerprintReport, analyze
from .database import FingerprintStore
from .provider import ElectrsProvider

MAX_CONCURRENCY = 64
MONTHLY_TRANSACTION_CAP = 500_000


@dataclass
class _MonthBatch:
    year: int
    month: int
    remaining_capacity: int
    blocks: list[tuple[int, str, datetime]] = field(default_factory=list)
    selected_txids: list[tuple[str, int]] = field(default_factory=list)
    transactions_seen: int = 0


def _add_to_reservoir(
    reservoir: list[tuple[str, int]],
    candidate: tuple[str, int],
    transactions_seen: int,
    capacity: int,
) -> None:
    """Keep a uniform random sample of at most ``capacity`` candidates."""
    if len(reservoir) < capacity:
        reservoir.append(candidate)
        return
    replacement = random.randrange(transactions_seen)
    if replacement < capacity:
        reservoir[replacement] = candidate


async def collect_forever(
    provider: ElectrsProvider,
    store: FingerprintStore,
    start_height: int | None,
    poll_interval: float,
) -> None:
    """Continuously analyze blocks, resuming after the latest stored height."""
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

    async def flush_month(batch: _MonthBatch) -> None:
        """Fetch the month's reservoir sample and atomically save its blocks."""
        selected_heights: dict[str, int] = {}
        for txid, height in batch.selected_txids:
            selected_heights[txid] = height
        selected_txids = [txid for txid, _ in batch.selected_txids]
        reports: list[FingerprintReport] = []
        batch_size = MAX_CONCURRENCY * 4
        for offset in range(0, len(selected_txids), batch_size):
            reports.extend(
                await asyncio.gather(
                    *(
                        analyze_tx(txid)
                        for txid in selected_txids[offset : offset + batch_size]
                    )
                )
            )
        reports_by_height: dict[int, list[FingerprintReport]] = {}
        for report in reports:
            height = selected_heights[report.txid]
            reports_by_height.setdefault(height, []).append(report)
        store.save_blocks(batch.blocks, reports_by_height)
        print(
            f"Stored {batch.year:04d}-{batch.month:02d}: "
            f"{len(reports)} transactions (cap {MONTHLY_TRANSACTION_CAP})",
            file=sys.stderr,
        )

    month_batch: _MonthBatch | None = None

    while True:
        tip_height = await provider.get_tip_height()
        while next_height <= tip_height:
            block_hash = await provider.get_block_hash(next_height)
            timestamp = datetime.fromtimestamp(
                await provider.get_block_timestamp(block_hash), tz=timezone.utc
            )

            if (
                month_batch is not None
                and (timestamp.year, timestamp.month)
                != (month_batch.year, month_batch.month)
            ):
                await flush_month(month_batch)
                month_batch = None

            if month_batch is None:
                existing_count = store.get_month_transaction_count(
                    timestamp.year, timestamp.month
                )
                month_batch = _MonthBatch(
                    timestamp.year,
                    timestamp.month,
                    max(0, MONTHLY_TRANSACTION_CAP - existing_count),
                )

            txids = await provider.get_block_txids(block_hash)
            month_batch.blocks.append((next_height, block_hash, timestamp))
            for txid in txids:
                if month_batch.remaining_capacity == 0:
                    break
                month_batch.transactions_seen += 1
                _add_to_reservoir(
                    month_batch.selected_txids,
                    (txid, next_height),
                    month_batch.transactions_seen,
                    month_batch.remaining_capacity,
                )
            next_height += 1
        await asyncio.sleep(poll_interval)
