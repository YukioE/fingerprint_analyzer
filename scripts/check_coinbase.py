"""Fetch transactions from a CSV and report whether each is a coinbase."""

import argparse
import asyncio
import csv
import itertools
import sys
from pathlib import Path
from typing import Iterator

from fingerprint_analyzer.provider import ElectrsProvider

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "taproot_scriptpath_backfill.csv"
DEFAULT_API_URL = "https://mempool.space/api"
DEFAULT_CONCURRENCY = 16


def iter_txids(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row_number, row in enumerate(csv.reader(source), start=1):
            if not row:
                continue
            txid = row[0].strip()
            if len(txid) != 64 or any(
                char not in "0123456789abcdefABCDEF" for char in txid
            ):
                raise ValueError(f"Invalid transaction ID in CSV row {row_number}")
            yield txid


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--mempool-url", default=DEFAULT_API_URL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    provider = ElectrsProvider(base_url=args.mempool_url)
    writer = csv.writer(sys.stdout)
    writer.writerow(["txid", "is_coinbase"])
    txids = iter_txids(args.csv_file)

    while batch := list(itertools.islice(txids, args.concurrency)):
        transactions = await asyncio.gather(
            *(provider.get_transaction(txid) for txid in batch)
        )
        for txid, transaction in zip(batch, transactions):
            is_coinbase = (
                len(transaction.inputs) == 1
                and transaction.inputs[0].is_coinbase
            )
            writer.writerow([txid, str(is_coinbase).lower()])


if __name__ == "__main__":
    asyncio.run(main())
