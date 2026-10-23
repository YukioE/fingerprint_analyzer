#!/usr/bin/env python3
"""Plot monthly transaction fingerprint and input-type statistics."""

import argparse
import os
import sys
from pathlib import Path

from psycopg import Error as PsycopgError
from psycopg import connect, sql

from fingerprint_analyzer.database import BLOCKS_TABLE, TRANSACTIONS_TABLE

BOOLEAN_FEATURES = (
    ("anti_fee_sniping", "Anti-fee-sniping"),
    ("bip69", "BIP69"),
    ("low_r_grinding", "Low-R grinding"),
    ("more_than_2_outputs", "More than 2 outputs"),
    ("multi_type_vin", "Multiple input types"),
    ("sending_to_base58", "Base58 outputs"),
    ("sending_to_bech32", "Bech32 outputs"),
    ("sending_to_bech32m", "Bech32m outputs"),
    ("op_return_outputs", "OP_RETURN outputs"),
    ("compressed_ecdsa_public_keys", "Compressed ECDSA public keys"),
    ("default_opt_in_rbf_set", "Replace by Fee"),
)


def _available_columns(connection):
    rows = connection.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (TRANSACTIONS_TABLE,),
    )
    return {row[0] for row in rows}


def _fetch_monthly_statistics(database_url):
    """Return monthly aggregates and input types present in this database."""
    with connect(database_url) as connection:
        columns = _available_columns(connection)
        input_features = sorted(
            name for name in columns if name.startswith("spending_from_")
        )
        feature_names = [name for name, _ in BOOLEAN_FEATURES] + input_features
        feature_names = [name for name in feature_names if name in columns]

        # Percentages exclude NULLs so older or partial feature collections remain meaningful.
        metrics = []
        if "txversion_2" in columns:
            metrics.append(("txversion_2", "tx_version_2_percent"))
        metrics.extend((name, name) for name in feature_names)

        fields = [
            sql.SQL("date_trunc('month', b.\"timestamp\")::date AS month"),
            sql.SQL("count(*)::bigint AS transaction_count"),
        ]
        for column, alias in metrics:
            fields.append(
                sql.SQL(
                    "100.0 * count(*) FILTER (WHERE t.{column} IS TRUE) "
                    "/ NULLIF(count(*) FILTER (WHERE t.{column} IS NOT NULL), 0) "
                    "AS {alias}"
                ).format(
                    column=sql.Identifier(column),
                    alias=sql.Identifier(alias),
                )
            )

        query = sql.SQL(
            """
            SELECT {}
            FROM {} AS t
            JOIN {} AS b USING (block_height)
            GROUP BY month
            ORDER BY month
            """
        ).format(
            sql.SQL(", ").join(fields),
            sql.Identifier(TRANSACTIONS_TABLE),
            sql.Identifier(BLOCKS_TABLE),
        )
        rows = connection.execute(query).fetchall()

    monthly = {
        row[0]: {
            "transaction_count": row[1],
            **dict(zip((alias for _, alias in metrics), row[2:], strict=True)),
        }
        for row in rows
    }
    months = list(monthly)
    input_types = [name.removeprefix("spending_from_") for name in input_features]
    return months, monthly, input_types


def _plot_percentages(axis, months, monthly, features, ylabel, show_legend=True):
    """Draw labeled monthly percentage series on a shared 0–100 scale."""
    for index, (feature, label) in enumerate(features):
        values = []
        for month in months:
            value = monthly[month].get(feature)
            values.append(value if value is not None else float("nan"))
        axis.plot(months, values, color=f"C{index}", label=label)

    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 100)
    axis.set_yticks(range(0, 101, 10))
    axis.grid(axis="both", linestyle=":", alpha=0.45)
    if show_legend:
        axis.legend()


def _save_plots(output_dir, months, monthly, input_types, plt):
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    # Save each collected boolean feature separately, including empty charts for absent columns.
    for feature, label in BOOLEAN_FEATURES:
        figure, axis = plt.subplots(figsize=(12, 6))
        _plot_percentages(
            axis,
            months,
            monthly,
            ((feature, label),),
            "Transactions with feature (%)",
            show_legend=False,
        )
        axis.set_title(label)
        axis.set_xlabel("Year")
        axis.grid(axis="x", which="major", linestyle=":", alpha=0.45)
        figure.tight_layout()
        plots.append((f"{feature}.png", figure))

    figure, axis = plt.subplots(figsize=(12, 6))
    _plot_percentages(
        axis,
        months,
        monthly,
        (("tx_version_2_percent", "Version 2"),),
        "Transactions using version 2 (%)",
        show_legend=False,
    )
    axis.set_title("Transaction version 2")
    axis.set_xlabel("Year")
    figure.tight_layout()
    plots.append(("transaction_version_2.png", figure))

    figure, axis = plt.subplots(figsize=(12, 6))
    for index, input_type in enumerate(input_types):
        feature = f"spending_from_{input_type}"
        axis.plot(
            months,
            [monthly[month].get(feature, float("nan")) for month in months],
            color=plt.get_cmap("tab10")(index % 10),
            label=input_type,
        )
    axis.set_title("Input-type usage")
    axis.set_xlabel("Year")
    axis.set_ylabel("Transactions containing input type (%)")
    axis.set_ylim(0, 100)
    axis.set_yticks(range(0, 101, 10))
    axis.grid(axis="both", linestyle=":", alpha=0.45)
    if input_types:
        axis.legend(title="Input type")
    else:
        axis.text(0.5, 0.5, "No input-type data collected", ha="center", va="center")
    figure.tight_layout()
    plots.append(("input_types.png", figure))

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        months,
        [monthly[month]["transaction_count"] for month in months],
        width=20,
        color="C4",
    )
    axis.set_title("Transactions sampled per month")
    axis.set_xlabel("Year")
    axis.set_ylabel("Collected transaction count")
    axis.grid(axis="y", linestyle=":", alpha=0.45)
    figure.tight_layout()
    plots.append(("monthly_sample_count.png", figure))

    for filename, figure in plots:
        path = output_dir / filename
        try:
            figure.savefig(path, dpi=150)
        finally:
            plt.close(figure)
        print(f"Plot saved to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL (defaults to DATABASE_URL)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("monthly_statistics"),
        help="directory for generated PNG plots (default: monthly_statistics)",
    )
    args = parser.parse_args()
    if not args.database_url:
        parser.error("provide --database-url or set DATABASE_URL")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        statistics = _fetch_monthly_statistics(args.database_url)
        months, monthly, input_types = statistics
        if not months:
            raise ValueError("No collected transactions found")
        _save_plots(args.output_dir, months, monthly, input_types, plt)
    except ImportError:
        print(
            "Error: plotting requires matplotlib; install with "
            "`python -m pip install -e '.[pruning-eval]'`",
            file=sys.stderr,
        )
        return 1
    except (PsycopgError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
