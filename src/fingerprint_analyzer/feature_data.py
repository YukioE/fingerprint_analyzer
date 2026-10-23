"""Read labeled transaction lists and create machine-learning feature datasets."""

import asyncio
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .analyzer import FingerprintReport, analyze
from .provider import ElectrsProvider, TransactionNotFoundError

LABEL_COLUMN = "wallet"
TXID_COLUMN = "txid"
NETWORK_COLUMN = "network"

NETWORK_URLS = {
    "mainnet": "https://mempool.space/api",
    "bitcoin": "https://mempool.space/api",
    "bitcoin-mainnet": "https://mempool.space/api",
    "testnet": "https://mempool.space/testnet/api",
    "testnet3": "https://mempool.space/testnet/api",
    "testnet4": "https://mempool.space/testnet4/api",
}
MAX_CONCURRENCY = 64


@dataclass(frozen=True)
class TransactionRow:
    """One transaction reference and its optional wallet label."""

    txid: str
    network: str
    wallet_label: str | None = None


def _normalise_column(name: str) -> str:
    """Normalize CSV header spelling so equivalent names match."""
    return re.sub(r"[\s-]+", "_", name.strip().lower())


def _is_blank_row(row: dict[str, str | None]) -> bool:
    """Check whether a CSV row contains only empty values."""
    return all(value is None or not value.strip() for value in row.values())


def load_transaction_list(
    path: Path, *, require_label: bool
) -> list[TransactionRow]:
    """Load and validate transaction IDs, networks, and optional wallet labels."""
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        fieldnames = [_normalise_column(name) for name in reader.fieldnames]
        if len(set(fieldnames)) != len(fieldnames):
            raise ValueError(f"CSV has duplicate column names: {path}")
        reader.fieldnames = fieldnames

        required = {TXID_COLUMN, NETWORK_COLUMN}
        if require_label:
            required.add(LABEL_COLUMN)

        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(
                f"CSV must contain columns: {', '.join(sorted(required))}; "
                f"missing: {', '.join(missing)}"
            )

        rows: list[TransactionRow] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Too many CSV values in {path} row {row_number}")
            if _is_blank_row(row):
                continue

            txid = (row.get(TXID_COLUMN) or "").strip()
            if not re.fullmatch(r"[0-9a-fA-F]{64}", txid):
                raise ValueError(f"Invalid transaction ID in {path} row {row_number}")

            network = (row.get(NETWORK_COLUMN) or "").strip().lower()
            if network not in NETWORK_URLS:
                supported = ", ".join(NETWORK_URLS)
                raise ValueError(
                    f"Unsupported network {network!r} in {path} row {row_number}; "
                    f"supported values: {supported}"
                )

            wallet_label = (row.get(LABEL_COLUMN) or "").strip() or None
            if require_label and wallet_label is None:
                raise ValueError(f"Missing wallet label in {path} row {row_number}")

            rows.append(TransactionRow(txid, network, wallet_label))

    if not rows:
        raise ValueError(f"No transactions found in {path}")
    return rows


def fingerprint_features(report: FingerprintReport) -> dict[str, int]:
    """Map detector results to 1, 0, or -1 when the result is unknown."""
    features: dict[str, int] = {}
    for name, value in report.fingerprints.items():
        if value is True:
            features[name] = 1
        elif value is False:
            features[name] = 0
        else:
            features[name] = -1
    return features


def feature_vector(report: FingerprintReport, feature_names: list[str]) -> list[int]:
    """Return report features in the order expected by a trained classifier."""
    features = fingerprint_features(report)
    return [features.get(name, -1) for name in feature_names]


async def analyze_transaction_rows(
    rows: list[TransactionRow],
    mempool_url: str | None = None,
) -> list[tuple[TransactionRow, FingerprintReport]]:
    """Fetch and analyze rows concurrently, skipping transactions that are missing."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    base_urls = (
        {mempool_url}
        if mempool_url
        else {NETWORK_URLS[row.network] for row in rows}
    )
    providers = {base_url: ElectrsProvider(base_url) for base_url in base_urls}

    async def analyze_row(row: TransactionRow) -> tuple[TransactionRow, FingerprintReport] | None:
        """Analyze one row using its network-specific provider."""
        base_url = mempool_url or NETWORK_URLS[row.network]
        try:
            async with semaphore:
                transaction = await providers[base_url].get_transaction(row.txid)
        except TransactionNotFoundError:
            print(f"Warning: transaction {row.txid} not found; skipping", file=sys.stderr)
            return None
        return row, analyze(transaction)

    results = await asyncio.gather(*(analyze_row(row) for row in rows))
    return [result for result in results if result is not None]


def write_feature_dataset(
    output: TextIO,
    reports: list[tuple[TransactionRow, FingerprintReport]],
) -> list[str]:
    """Write labeled fingerprint rows as CSV and return the feature column names."""
    if not reports:
        raise ValueError("no transaction reports to write")

    feature_names = list(reports[0][1].fingerprints)
    fieldnames = [LABEL_COLUMN, *feature_names]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for row, report in reports:
        if row.wallet_label is None:
            raise ValueError(f"Missing wallet label for transaction {row.txid}")
        writer.writerow({LABEL_COLUMN: row.wallet_label, **fingerprint_features(report)})

    return feature_names
