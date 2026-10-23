#!/usr/bin/env python3
"""Plot monthly transaction fingerprint and input-type statistics."""

import argparse
import os
import sys
from pathlib import Path

from psycopg import Error as PsycopgError
from psycopg import connect, sql

from fingerprint_analyzer.database import BLOCKS_TABLE, TRANSACTIONS_TABLE

BOOLEAN_PLOTS = {
    "anti_fee_sniping_bip69_low_r": (
        "Anti-fee-sniping, BIP69, and low-R grinding",
        (
            ("anti_fee_sniping", "Anti-fee-sniping"),
            ("bip69", "BIP69"),
            ("low_r_grinding", "Low-R grinding"),
        ),
    ),
    "outputs_and_input_mixture": (
        "More than two outputs and multiple input types",
        (
            ("more_than_2_outputs", "More than 2 outputs"),
            ("multi_type_vin", "Multiple input types"),
        ),
    ),
    "address_usage": (
        "Output address/script usage",
        (
            ("sending_to_base58", "Base58 outputs"),
            ("sending_to_bech32", "Bech32 outputs"),
            ("sending_to_bech32m", "Bech32m outputs"),
            ("op_return_outputs", "OP_RETURN outputs"),
        ),
    ),
}


def _get_available_columns(connection):
    rows = connection.execute(
        sql.SQL(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
            """
        ),
        (TRANSACTIONS_TABLE,),
    ).fetchall()
    return {row[0] for row in rows}


def _fetch_monthly_statistics(database_url: str):
    with connect(database_url) as connection:
        available_columns = _get_available_columns(connection)

    boolean_features = [
        feature
        for _, features in BOOLEAN_PLOTS.values()
        for feature, _ in features
    ] + ["compressed_ecdsa_public_keys", "default_opt_in_rbf_set"]
    input_type_features = sorted(
        column for column in available_columns if column.startswith("spending_from_")
    )
    boolean_features.extend(input_type_features)
    boolean_features = [feature for feature in boolean_features if feature in available_columns]

    version_column = "txversion_2" if "txversion_2" in available_columns else None
    with connect(database_url) as connection:
        select_fields = [
            sql.SQL("date_trunc('month', b.\"timestamp\")::date AS month"),
            sql.SQL("count(*)::bigint AS transaction_count"),
        ]
        if version_column is not None:
            select_fields.append(
                sql.SQL(
                    "100.0 * count(*) FILTER (WHERE t.{feature} IS TRUE) / NULLIF(count(*) FILTER (WHERE t.{feature} IS NOT NULL), 0) AS tx_version_2"
                ).format(feature=sql.Identifier(version_column))
            )
        else:
            select_fields.append(sql.SQL("NULL::numeric AS tx_version_2"))

        select_fields.append(sql.SQL("count(*)::bigint AS input_type_sample_count"))
        for feature in boolean_features:
            select_fields.append(
                sql.SQL(
                    "100.0 * count(*) FILTER (WHERE t.{feature} IS TRUE) / NULLIF(count(*) FILTER (WHERE t.{feature} IS NOT NULL), 0) AS {feature}"
                ).format(feature=sql.Identifier(feature))
            )

        averages = connection.execute(
            sql.SQL("SELECT {} FROM {} AS t JOIN {} AS b USING (block_height) GROUP BY month ORDER BY month").format(
                sql.SQL(", ").join(select_fields),
                sql.Identifier(TRANSACTIONS_TABLE),
                sql.Identifier(BLOCKS_TABLE),
            )
        ).fetchall()

    months = [row[0] for row in averages]
    monthly = {
        row[0]: {
            "transaction_count": row[1],
            "average_tx_version": row[2] if version_column is not None else None,
            "input_type_sample_count": row[3],
            **dict(zip(boolean_features, row[4:], strict=True)),
        }
        for row in averages
    }
    input_types = {
        feature.removeprefix("spending_from_"): {
            month: monthly[month][feature] for month in months
        }
        for feature in input_type_features
    }
    return months, monthly, input_types


def _plot_boolean_averages(axis, months, monthly, features, show_legend=True):
    for index, (feature, label) in enumerate(features):
        values = [
            monthly[month][feature]
            if monthly[month][feature] is not None
            else float("nan")
            for month in months
        ]
        axis.plot(
            months,
            values,
            color=f"C{index}",
            label=label,
        )
    axis.set_ylabel("Transactions with feature (%)")
    axis.set_ylim(0, 100)
    if show_legend:
        axis.legend()


def main() -> int:
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
        help="directory for the generated PNG plots (default: monthly_statistics)",
    )
    args = parser.parse_args()

    if not args.database_url:
        parser.error("provide --database-url or set DATABASE_URL")

    try:
        months, monthly, input_types = _fetch_monthly_statistics(args.database_url)
        if not months:
            raise ValueError("No collected transactions found")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(
            "Error: plotting requires matplotlib; install with "
            "`python -m pip install -e '.[pruning-eval]'`",
            file=sys.stderr,
        )
        return 1
    except (PsycopgError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Error creating output directory: {error}", file=sys.stderr)
        return 1
    plots = []

    individual_features = [
        (feature, label, f"{feature}.png")
        for _, features in BOOLEAN_PLOTS.values()
        for feature, label in features
    ] + [
        ("compressed_ecdsa_public_keys", "Compressed ECDSA public keys", "compressed_ecdsa_public_keys.png"),
        ("default_opt_in_rbf_set", "Replace by Fee", "default_opt_in_rbf_set.png"),
    ]
    for feature, label, filename in individual_features:
        figure, axis = plt.subplots(figsize=(12, 6))
        _plot_boolean_averages(
            axis,
            months,
            monthly,
            ((feature, label),),
            show_legend=False,
        )
        axis.set_title(label)
        axis.set_xlabel("Year")
        figure.tight_layout()
        plots.append((filename, figure))

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        months,
        [
            float(monthly[month]["average_tx_version"])
            if monthly[month]["average_tx_version"] is not None
            else float("nan")
            for month in months
        ],
        color="C0",
    )
    axis.set_ylabel("Transactions using version 2 (%)")
    axis.set_xlabel("Year")
    axis.set_title("Transaction version 2")
    axis.set_ylim(0, 100)
    figure.tight_layout()
    plots.append(("transaction_version_2.png", figure))

    figure, axis = plt.subplots(figsize=(12, 6))
    palette = plt.get_cmap("tab10")
    for index, input_type in enumerate(sorted(input_types)):
        axis.plot(
            months,
            [input_types[input_type].get(month, float("nan")) for month in months],
            color=palette(index % 10),
            label=input_type,
        )
    axis.set_title("Input-type usage")
    axis.set_xlabel("Year")
    axis.set_ylabel("Transactions containing input type (%)")
    axis.set_ylim(0, 100)
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
    figure.tight_layout()
    plots.append(("monthly_sample_count.png", figure))

    for filename, figure in plots:
        output_path = args.output_dir / filename
        try:
            figure.savefig(output_path, dpi=150)
        except OSError as error:
            print(f"Error saving {output_path}: {error}", file=sys.stderr)
            return 1
        finally:
            plt.close(figure)
        print(f"Plot saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
