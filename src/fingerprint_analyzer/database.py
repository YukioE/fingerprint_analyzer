"""PostgreSQL storage for per-transaction fingerprints and block metadata."""

from datetime import datetime, timezone

from psycopg import Connection, connect, sql

from .analyzer import FingerprintReport
from .fingerprints import DETECTORS

TRANSACTIONS_TABLE = "fingerprint_transactions"
BLOCKS_TABLE = "fingerprint_blocks"


class FingerprintStore:
    """Create the database schema and save analyzed blocks."""

    def __init__(self, database_url: str):
        """Open a database connection and cache the registered feature names."""
        self.connection: Connection = connect(database_url, autocommit=True)
        self.fingerprint_names = [
            detector.fingerprint_name for detector in DETECTORS
        ]

    def close(self) -> None:
        """Close the underlying PostgreSQL connection."""
        self.connection.close()

    def ensure_schema(self) -> None:
        """Create the transaction and block tables if they do not exist."""
        with self.connection.transaction():
            self.connection.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} ({})"
                ).format(
                    sql.Identifier(TRANSACTIONS_TABLE),
                    sql.SQL(", ").join(
                        [
                            sql.SQL("txid TEXT NOT NULL PRIMARY KEY"),
                            sql.SQL("block_height BIGINT NOT NULL"),
                            *(
                                sql.SQL("{} BOOLEAN").format(sql.Identifier(name))
                                for name in self.fingerprint_names
                            ),
                        ]
                    )
                )
            )
            self.connection.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        block_height BIGINT PRIMARY KEY,
                        block_hash TEXT NOT NULL,
                        "timestamp" TIMESTAMPTZ NOT NULL
                    )
                    """
                ).format(sql.Identifier(BLOCKS_TABLE))
            )
            for name in self.fingerprint_names:
                self.connection.execute(
                    sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} BOOLEAN").format(
                        sql.Identifier(TRANSACTIONS_TABLE),
                        sql.Identifier(name),
                    )
                )

    def get_latest_height(self) -> int | None:
        """Return the newest stored block height, or ``None`` for an empty store."""
        row = self.connection.execute(
            sql.SQL("SELECT MAX(block_height) FROM {}").format(
                sql.Identifier(BLOCKS_TABLE)
            ),
        ).fetchone()
        return row[0]

    def get_month_transaction_count(self, year: int, month: int) -> int:
        """Return the number of stored transactions in a UTC calendar month."""
        month_start = datetime(year, month, 1, tzinfo=timezone.utc)
        next_month_year = year + (1 if month == 12 else 0)
        next_month_number = 1 if month == 12 else month + 1
        next_month = datetime(
            next_month_year, next_month_number, 1, tzinfo=timezone.utc
        )
        row = self.connection.execute(
            sql.SQL(
                """
                SELECT COUNT(*)
                FROM {} AS t
                JOIN {} AS b USING (block_height)
                WHERE b."timestamp" >= %s AND b."timestamp" < %s
                """
            ).format(
                sql.Identifier(TRANSACTIONS_TABLE),
                sql.Identifier(BLOCKS_TABLE),
            ),
            (month_start, next_month),
        ).fetchone()
        return row[0]

    def save_block(
        self,
        height: int,
        block_hash: str,
        timestamp: datetime,
        reports: list[FingerprintReport],
    ) -> None:
        """Write one block and its transaction fingerprints atomically."""
        self.save_blocks([(height, block_hash, timestamp)], {height: reports})

    def save_blocks(
        self,
        blocks: list[tuple[int, str, datetime]],
        reports_by_height: dict[int, list[FingerprintReport]],
    ) -> None:
        """Atomically save a batch of blocks and their selected transactions."""
        columns = [
            "txid",
            "block_height",
            *self.fingerprint_names,
        ]
        insert = sql.SQL(
            "INSERT INTO {} ({}) VALUES ({}) "
            "ON CONFLICT (txid) DO UPDATE SET {}"
        ).format(
            sql.Identifier(TRANSACTIONS_TABLE),
            sql.SQL(", ").join(sql.Identifier(column) for column in columns),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
            sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(
                    sql.Identifier(column), sql.Identifier(column)
                )
                for column in columns[1:]
            ),
        )
        values = [
            (
                report.txid,
                height,
                *(report.fingerprints[name] for name in self.fingerprint_names),
            )
            for height, reports in reports_by_height.items()
            for report in reports
        ]

        with self.connection.transaction():
            if blocks:
                block_insert = sql.SQL(
                    """
                    INSERT INTO {} (block_height, block_hash, "timestamp")
                    VALUES (%s, %s, %s)
                    ON CONFLICT (block_height) DO UPDATE SET
                        block_hash = EXCLUDED.block_hash,
                        "timestamp" = EXCLUDED."timestamp"
                    """
                ).format(sql.Identifier(BLOCKS_TABLE))
                with self.connection.cursor() as cursor:
                    cursor.executemany(block_insert, blocks)
            if values:
                with self.connection.cursor() as cursor:
                    cursor.executemany(insert, values)
