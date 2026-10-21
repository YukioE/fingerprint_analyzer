from datetime import datetime

from psycopg import Connection, connect, sql

from .analyzer import FingerprintReport
from .fingerprints import DETECTORS

TRANSACTIONS_TABLE = "fingerprint_transactions"
BLOCKS_TABLE = "fingerprint_blocks"


class FingerprintStore:
    def __init__(self, database_url: str):
        self.connection: Connection = connect(database_url, autocommit=True)
        self.fingerprint_names = [
            detector.fingerprint_name for detector in DETECTORS
        ]

    def close(self) -> None:
        self.connection.close()

    def ensure_schema(self) -> None:
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

    def get_latest_height(self) -> int | None:
        row = self.connection.execute(
            sql.SQL("SELECT MAX(block_height) FROM {}").format(
                sql.Identifier(BLOCKS_TABLE)
            ),
        ).fetchone()
        return row[0]

    def save_block(
        self,
        height: int,
        block_hash: str,
        timestamp: datetime,
        reports: list[FingerprintReport],
    ) -> None:
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
            for report in reports
        ]

        with self.connection.transaction():
            self.connection.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (block_height, block_hash, "timestamp")
                    VALUES (%s, %s, %s)
                    ON CONFLICT (block_height) DO UPDATE SET
                        block_hash = EXCLUDED.block_hash,
                        "timestamp" = EXCLUDED."timestamp"
                    """
                ).format(sql.Identifier(BLOCKS_TABLE)),
                (height, block_hash, timestamp),
            )
            if values:
                with self.connection.cursor() as cursor:
                    cursor.executemany(insert, values)
