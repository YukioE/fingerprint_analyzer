#!/usr/bin/env python3
"""Recheck transactions omitted by the old input-type plot filter."""

import argparse
import asyncio
import csv
import os
import sys
from collections.abc import Iterator
from pathlib import Path

from psycopg import sql

from fingerprint_analyzer.analyzer import FingerprintReport, analyze
from fingerprint_analyzer.database import (
    FingerprintStore,
    TRANSACTIONS_TABLE,
)
from fingerprint_analyzer.fingerprints import DETECTORS
from fingerprint_analyzer.fingerprints.helpers import input_type, script_bytes
from fingerprint_analyzer.model import TxInput
from fingerprint_analyzer.provider import ElectrsProvider

CUSTOM_TAPROOT_FEATURE = "spending_from_p2tr_scriptpath_custom"
INPUT_TYPE_FEATURES = tuple(
    detector.fingerprint_name
    for detector in DETECTORS
    if detector.fingerprint_name.startswith("spending_from_")
    and detector.fingerprint_name != CUSTOM_TAPROOT_FEATURE
)


def _script_instructions(script: bytes) -> Iterator[tuple[int, bytes | None]]:
    """Yield each script opcode and any data pushed by that opcode."""
    index = 0
    while index < len(script):
        opcode = script[index]
        index += 1
        if opcode == 0:
            yield opcode, b""
            continue
        if opcode <= 75:
            length = opcode
        elif opcode == 0x4C:
            if index >= len(script):
                return
            length = script[index]
            index += 1
        elif opcode == 0x4D:
            if index + 2 > len(script):
                return
            length = int.from_bytes(script[index : index + 2], "little")
            index += 2
        elif opcode == 0x4E:
            if index + 4 > len(script):
                return
            length = int.from_bytes(script[index : index + 4], "little")
            index += 4
        else:
            yield opcode, None
            continue

        if index + length > len(script):
            return
        yield opcode, script[index : index + length]
        index += length


def _has_inscription_envelope(script: bytes) -> bool:
    """Check for the Ordinals OP_FALSE OP_IF "ord" envelope marker."""
    previous_false = False
    expecting_ord = False
    for opcode, pushed_data in _script_instructions(script):
        if expecting_ord:
            if pushed_data == b"ord":
                return True
            expecting_ord = False
        if previous_false and opcode == 0x63:
            expecting_ord = True
        previous_false = opcode == 0
    return False


def _input_has_inscription(txin: TxInput) -> bool:
    """Check the Taproot script-path witness for an inscription envelope."""
    if not input_type(txin).startswith("p2tr_scriptpath_"):
        return False
    witness = list(txin.witness or [])
    if witness and script_bytes(witness[-1]).startswith(b"\x50"):
        witness.pop()
    return len(witness) >= 2 and _has_inscription_envelope(
        script_bytes(witness[-2])
    )


def _candidate_query(
    after_txid: str, limit: int
) -> tuple[sql.Composed, tuple[str, int]]:
    """Find old rows with no recognized input-type flag and no backfill yet."""
    all_false = sql.SQL(" AND ").join(
        sql.SQL("{} IS FALSE").format(sql.Identifier(name))
        for name in INPUT_TYPE_FEATURES
    )
    query = sql.SQL(
        """
        SELECT txid, block_height
        FROM {}
        WHERE {} AND {} IS NULL AND {} > %s
        ORDER BY {}
        LIMIT %s
        """
    ).format(
        sql.Identifier(TRANSACTIONS_TABLE),
        all_false,
        sql.Identifier(CUSTOM_TAPROOT_FEATURE),
        sql.Identifier("txid"),
        sql.Identifier("txid"),
    )
    return query, (after_txid, limit)


async def _inspect_transaction(
    provider: ElectrsProvider,
    txid: str,
    height: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int, FingerprintReport, list[str], bool]:
    async with semaphore:
        transaction = await provider.get_transaction(txid)
    report = analyze(transaction)
    input_types = sorted(
        {
            input_type(txin)
            for txin in transaction.inputs
            if not txin.is_coinbase
        }
    )
    has_inscription = any(_input_has_inscription(txin) for txin in transaction.inputs)
    return txid, height, report, input_types, has_inscription


async def _run(args: argparse.Namespace) -> int:
    store = FingerprintStore(args.database_url)
    try:
        store.ensure_schema()
        print("Searching database for unclassified transactions...", file=sys.stderr)

        provider = ElectrsProvider(base_url=args.mempool_url)
        semaphore = asyncio.Semaphore(args.concurrency)
        inscription_count = 0
        taproot_scriptpath_count = 0
        completed_count = 0
        progress_interval = min(32, max(1, args.concurrency // 4))
        after_txid = ""
        report_path = args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with report_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(
                [
                    "txid",
                    "block_height",
                    "input_types",
                    "taproot_scriptpath_custom",
                    "has_inscription",
                ]
            )
            while args.limit is None or completed_count < args.limit:
                batch_limit = args.concurrency
                if args.limit is not None:
                    batch_limit = min(batch_limit, args.limit - completed_count)
                query, parameters = _candidate_query(after_txid, batch_limit)
                batch = store.connection.execute(query, parameters).fetchall()
                if not batch:
                    break
                after_txid = batch[-1][0]
                tasks = [
                    asyncio.create_task(
                        _inspect_transaction(
                            provider, txid, height, semaphore
                        )
                    )
                    for txid, height in batch
                ]
                updates = []
                for completed in asyncio.as_completed(tasks):
                    txid, height, report, input_types, has_inscription = await completed
                    completed_count += 1
                    taproot_custom = report.fingerprints[CUSTOM_TAPROOT_FEATURE]
                    taproot_scriptpath_count += taproot_custom is True
                    inscription_count += has_inscription
                    writer.writerow(
                        [
                            txid,
                            height,
                            ";".join(input_types),
                            taproot_custom,
                            has_inscription,
                        ]
                    )
                    if args.apply:
                        updates.append((taproot_custom, txid))
                    if (
                        completed_count % progress_interval == 0
                    ):
                        output.flush()
                        print(
                            f"Inspected {completed_count} transactions",
                            file=sys.stderr,
                        )

                if args.apply and updates:
                    update = sql.SQL(
                        "UPDATE {} SET {} = %s "
                        "WHERE txid = %s AND {} IS NULL"
                    ).format(
                        sql.Identifier(TRANSACTIONS_TABLE),
                        sql.Identifier(CUSTOM_TAPROOT_FEATURE),
                        sql.Identifier(CUSTOM_TAPROOT_FEATURE),
                    )
                    with store.connection.transaction():
                        with store.connection.cursor() as cursor:
                            cursor.executemany(update, updates)
                if completed_count % progress_interval:
                    output.flush()
                    print(
                        f"Inspected {completed_count} transactions",
                        file=sys.stderr,
                    )

        print(
            f"Inspected {completed_count} transactions; "
            f"{taproot_scriptpath_count} use custom Taproot script paths; "
            f"{inscription_count} contain an inscription envelope. "
            f"Report: {report_path}"
        )
        if args.apply:
            print(f"Updated {CUSTOM_TAPROOT_FEATURE} in the database.")
        else:
            print("Database unchanged; rerun with --apply to save classifications.")
    finally:
        store.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL (defaults to DATABASE_URL)",
    )
    parser.add_argument(
        "--mempool-url",
        default="http://100.126.92.24:3000",
        help="Electrs-compatible API base URL",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("taproot_scriptpath_backfill.csv"),
        help="CSV report path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="inspect at most this many candidates",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=128,
        help="maximum concurrent API requests (default: 8)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the new Taproot script-path classification to PostgreSQL",
    )
    args = parser.parse_args()
    if not args.database_url:
        parser.error("provide --database-url or set DATABASE_URL")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must not be negative")
    if args.concurrency <= 0:
        parser.error("--concurrency must be greater than zero")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
