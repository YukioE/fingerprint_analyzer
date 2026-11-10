#!/usr/bin/env python3
"""Cluster collected transaction fingerprints to estimate distinct wallet groups."""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from psycopg import Error as PsycopgError
from psycopg import connect, sql
from sklearn.cluster import AgglomerativeClustering

from fingerprint_analyzer.database import BLOCKS_TABLE, TRANSACTIONS_TABLE
from fingerprint_analyzer.fingerprints import DETECTORS
from fingerprint_analyzer.ml import first_dataset, train_classifier

FINGERPRINT_NAMES = tuple(detector.fingerprint_name for detector in DETECTORS)
UNKNOWN_FEATURE = 2
EPS = 0.0
THRESHOLD_CHECKPOINTS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)


@dataclass
class _ClusterStats:
    transaction_count: int
    profile_count: int
    true_counts: list[int]
    known_counts: list[int]


def _available_columns(connection) -> set[str]:
    """Return fingerprint columns present in this database."""
    rows = connection.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (TRANSACTIONS_TABLE,),
    )
    return {row[0] for row in rows}


def _fetch_fingerprint_profiles(database_url: str, year: int):
    """Return counts of distinct fingerprints collected during a UTC calendar year."""
    year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    next_year_start = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    with connect(database_url) as connection:
        available_columns = _available_columns(connection)
        feature_names = [
            name
            for name in FINGERPRINT_NAMES
            if name in available_columns
        ]
        if not feature_names:
            raise ValueError(
                f"No fingerprint columns found in {TRANSACTIONS_TABLE}"
            )

        columns = sql.SQL(", ").join(
            sql.SQL("t.{}").format(sql.Identifier(name)) for name in feature_names
        )
        query = sql.SQL(
            """
            SELECT {}, COUNT(*)::bigint, MIN(t.txid)
            FROM {} AS t
            JOIN {} AS b USING (block_height)
            WHERE b."timestamp" >= %s AND b."timestamp" < %s
            GROUP BY {}
            ORDER BY {}
            """
        ).format(
            columns,
            sql.Identifier(TRANSACTIONS_TABLE),
            sql.Identifier(BLOCKS_TABLE),
            columns,
            columns,
        )
        rows = connection.execute(query, (year_start, next_year_start)).fetchall()
    return (
        feature_names,
        [(tuple(row[:-2]), row[-2]) for row in rows],
        [row[-1] for row in rows],
    )


def _build_cluster_tree(profiles):
    """Build the complete-linkage hierarchy once for threshold comparisons."""
    if not profiles:
        raise ValueError("No collected fingerprints found")

    informative_indices = [
        index
        for index, (profile, _) in enumerate(profiles)
        if any(value is not None for value in profile)
    ]
    if not informative_indices:
        return informative_indices, None

    vectors = [
        [
            UNKNOWN_FEATURE if value is None else int(value)
            for value in profiles[index][0]
        ]
        for index in informative_indices
    ]
    values = np.asarray(vectors, dtype=np.int8)
    known = (values != UNKNOWN_FEATURE).astype(np.int32)
    true = (values == 1).astype(np.int32)
    false = (values == 0).astype(np.int32)
    distances = np.ones((len(values), len(values)), dtype=float)
    block_size = max(1, min(256, len(values)))
    starts = range(0, len(values), block_size)
    block_count = len(starts)
    progress_interval = max(1, block_count // 20)
    for block_index, start in enumerate(starts, start=1):
        stop = min(start + block_size, len(values))
        comparable = known[start:stop] @ known.T
        matching = true[start:stop] @ true.T
        matching += false[start:stop] @ false.T
        mismatches = comparable - matching
        np.divide(
            mismatches,
            comparable,
            out=distances[start:stop],
            where=comparable != 0,
        )
        if block_index % progress_interval == 0 or block_index == block_count:
            percentage = 100 * block_index // block_count
            print(
                f"Pairwise distances: {percentage}% "
                f"({block_index:,}/{block_count:,} blocks)",
                file=sys.stderr,
                flush=True,
            )

    model = AgglomerativeClustering(
        n_clusters=1,
        compute_full_tree=True,
        compute_distances=True,
        linkage="complete",
        metric="precomputed",
    )
    print("Starting complete-linkage hierarchy...", file=sys.stderr, flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as executor:
        fit = executor.submit(model.fit, distances)
        while True:
            try:
                tree = fit.result(timeout=10)
                break
            except TimeoutError:
                elapsed = time.monotonic() - started
                print(
                    f"Complete-linkage hierarchy still running "
                    f"({elapsed:.0f}s elapsed)...",
                    file=sys.stderr,
                    flush=True,
                )
    return informative_indices, tree


def _labels_at_threshold(
    profiles, informative_indices, tree, *, eps: float, min_samples: int
) -> list[int]:
    """Cut the hierarchy at ``eps`` and mark undersized clusters as noise."""
    labels = [-1] * len(profiles)
    if tree is None:
        return labels

    profile_count = len(informative_indices)
    parents = list(range(2 * profile_count - 1))

    def find_root(node: int) -> int:
        while parents[node] != node:
            node = parents[node]
        return node

    for merge_index, (left, right) in enumerate(tree.children_):
        if tree.distances_[merge_index] > eps:
            break
        parent = profile_count + merge_index
        parents[left] = parent
        parents[right] = parent

    roots = [find_root(index) for index in range(profile_count)]
    root_labels = {root: label for label, root in enumerate(dict.fromkeys(roots))}
    cluster_sizes: dict[int, int] = {}
    for index, root in zip(informative_indices, roots, strict=True):
        label = root_labels[root]
        cluster_sizes[label] = cluster_sizes.get(label, 0) + profiles[index][1]

    for index, root in zip(informative_indices, roots, strict=True):
        label = root_labels[root]
        if cluster_sizes[label] >= min_samples:
            labels[index] = label
    return labels


def _cluster_summary(
    feature_names, profiles, labels, sample_txids, cluster_wallet_counts
):
    """Summarize cluster sizes and features that differ from global rates."""
    totals = [0] * len(feature_names)
    known_totals = [0] * len(feature_names)
    cluster_data: dict[int, _ClusterStats] = {}
    cluster_sample_txids: dict[int, str] = {}

    for (profile, count), label, sample_txid in zip(
        profiles, labels, sample_txids, strict=True
    ):
        for index, value in enumerate(profile):
            if value is not None:
                known_totals[index] += count
                totals[index] += count * int(value)
        if label == -1:
            continue
        data = cluster_data.setdefault(
            label,
            _ClusterStats(0, 0, [0] * len(feature_names), [0] * len(feature_names)),
        )
        data.transaction_count += count
        data.profile_count += 1
        cluster_sample_txids.setdefault(label, sample_txid)
        for index, value in enumerate(profile):
            if value is not None:
                data.known_counts[index] += count
                data.true_counts[index] += count * int(value)

    summaries = []
    for label, data in cluster_data.items():
        wallet_counts = cluster_wallet_counts[label]
        classified_count = sum(wallet_counts.values())
        ranked_wallets = sorted(
            wallet_counts.items(), key=lambda item: (-item[1], item[0])
        )
        wallet_predictions = "; ".join(
            f"{wallet}: {count / classified_count:.1%}"
            for wallet, count in ranked_wallets
        )
        differences = []
        for index, name in enumerate(feature_names):
            known_count = data.known_counts[index]
            if not known_count:
                differences.append((0, f"{name}: no known values"))
                continue
            if not known_totals[index]:
                continue
            cluster_rate = data.true_counts[index] / known_count
            overall_rate = totals[index] / known_totals[index]
            differences.append(
                (
                    abs(cluster_rate - overall_rate),
                    f"{name}: {cluster_rate:.0%} true "
                    f"(overall {overall_rate:.0%})",
                )
            )
        feature_details = "; ".join(
            description for _, description in sorted(differences, reverse=True)
        )
        summaries.append(
            {
                "cluster_id": label,
                "transaction_count": data.transaction_count,
                "profile_count": data.profile_count,
                "sample_txid": cluster_sample_txids[label],
                "dominant_wallet": ranked_wallets[0][0],
                "dominant_wallet_share_percent": (
                    100 * ranked_wallets[0][1] / classified_count
                ),
                "wallet_predictions": wallet_predictions,
                "features": feature_details,
            }
        )
    return sorted(
        summaries,
        key=lambda summary: (
            -summary["transaction_count"],
            summary["cluster_id"],
        ),
    )


def _classify_cluster_profiles(
    feature_names,
    profiles,
    labels,
    classifier,
    classifier_feature_names,
) -> dict[int, dict[str, int]]:
    """Classify each distinct profile and count predictions by cluster."""
    vectors = []
    classified_profiles = []
    for (profile, count), label in zip(profiles, labels, strict=True):
        if label == -1:
            continue
        profile_features = dict(zip(feature_names, profile, strict=True))
        vectors.append(
            [
                -1
                if profile_features.get(name) is None
                else int(profile_features[name])
                for name in classifier_feature_names
            ]
        )
        classified_profiles.append((label, count))

    wallet_predictions: dict[int, dict[str, int]] = {}
    if not vectors:
        return wallet_predictions
    predicted_wallets = classifier.predict(vectors)
    for (label, count), wallet in zip(
        classified_profiles, predicted_wallets, strict=True
    ):
        wallet_counts = wallet_predictions.setdefault(label, {})
        wallet_name = str(wallet)
        wallet_counts[wallet_name] = wallet_counts.get(wallet_name, 0) + count
    return wallet_predictions


def _write_summary(output_path: Path, summaries) -> None:
    """Write one size and distinguishing-feature row per cluster."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "cluster_id",
                "transaction_count",
                "profile_count",
                "sample_txid",
                "dominant_wallet",
                "dominant_wallet_share_percent",
                "wallet_predictions",
                "features",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)


def _wallet_market_shares(cluster_wallet_counts):
    """Aggregate classified cluster transactions into per-wallet market shares."""
    wallet_totals: dict[str, dict[str, int]] = {}
    for cluster_counts in cluster_wallet_counts.values():
        for wallet, transaction_count in cluster_counts.items():
            totals = wallet_totals.setdefault(
                wallet, {"transaction_count": 0, "cluster_count": 0}
            )
            totals["transaction_count"] += transaction_count
            totals["cluster_count"] += 1

    classified_transactions = sum(
        totals["transaction_count"] for totals in wallet_totals.values()
    )
    return [
        {
            "wallet": wallet,
            "transaction_count": totals["transaction_count"],
            "cluster_count": totals["cluster_count"],
            "market_share_percent": (
                100 * totals["transaction_count"] / classified_transactions
            ),
        }
        for wallet, totals in sorted(
            wallet_totals.items(),
            key=lambda item: (-item[1]["transaction_count"], item[0]),
        )
    ]


def _write_wallet_market_shares(output_path: Path, market_shares) -> None:
    """Write one transaction-weighted market-share row per predicted wallet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "wallet",
                "transaction_count",
                "cluster_count",
                "market_share_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(market_shares)


def _print_threshold_sweep(
    profiles, informative_indices, tree, *, selected_eps: float, min_samples: int
) -> None:
    """Show how cluster counts and coverage change across distance thresholds."""
    thresholds = sorted({*THRESHOLD_CHECKPOINTS, selected_eps})
    total_transactions = sum(count for _, count in profiles)
    print("Threshold sensitivity (threshold, clusters, noise %, largest cluster %):")
    for threshold in thresholds:
        labels = _labels_at_threshold(
            profiles,
            informative_indices,
            tree,
            eps=threshold,
            min_samples=min_samples,
        )
        counts: dict[int, int] = {}
        noise = 0
        for (_, count), label in zip(profiles, labels, strict=True):
            if label == -1:
                noise += count
            else:
                counts[label] = counts.get(label, 0) + count
        largest = max(counts.values(), default=0)
        print(
            f"  {threshold:>5.2f}: {len(counts):>5,} clusters, "
            f"{noise / total_transactions:>6.1%} noise, "
            f"{largest / total_transactions:>6.1%} largest"
        )


def _known_feature_distance(left, right) -> float:
    """Compare only features known for both profiles; no overlap means no match."""
    comparable = [
        (left_value, right_value)
        for left_value, right_value in zip(left, right, strict=True)
        if left_value != UNKNOWN_FEATURE and right_value != UNKNOWN_FEATURE
    ]
    if not comparable:
        return 1.0
    mismatches = sum(
        left_value != right_value for left_value, right_value in comparable
    )
    return mismatches / len(comparable)


def _write_clusters(output_path: Path, feature_names, profiles, labels) -> None:
    """Write one row per distinct fingerprint profile with its cluster assignment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["cluster_id", "transaction_count", *feature_names])
        for (profile, count), label in zip(profiles, labels, strict=True):
            writer.writerow(
                [
                    label,
                    count,
                    *("" if value is None else str(value).lower() for value in profile),
                ]
            )


def main() -> int:
    """Load fingerprints from PostgreSQL, cluster them, and write a CSV report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL (defaults to DATABASE_URL)",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="UTC calendar year to cluster",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="minimum transactions required in a cluster (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fingerprint_clusters.csv"),
        help="CSV report path (default: fingerprint_clusters.csv)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="cluster summary CSV (defaults to <output>_summary.csv)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="decision-tree training CSV (defaults to the first bundled dataset)",
    )
    args = parser.parse_args()
    if not args.database_url:
        parser.error("provide --database-url or set DATABASE_URL")
    if args.min_samples < 1:
        parser.error("--min-samples must be at least 1")

    try:
        dataset_path = args.dataset or first_dataset()
        classifier, classifier_feature_names = train_classifier(
            dataset_path, "decision-tree"
        )
        print(f"Decision tree trained from {dataset_path}", file=sys.stderr)
        feature_names, profiles, sample_txids = _fetch_fingerprint_profiles(
            args.database_url, args.year
        )
        informative_count = sum(
            any(value is not None for value in profile)
            for profile, _ in profiles
        )
        print(
            f"Building complete-linkage tree for {informative_count:,} "
            "informative profiles...",
            file=sys.stderr,
            flush=True,
        )
        informative_indices, tree = _build_cluster_tree(profiles)
        labels = _labels_at_threshold(
            profiles,
            informative_indices,
            tree,
            eps=EPS,
            min_samples=args.min_samples,
        )
        _write_clusters(args.output, feature_names, profiles, labels)
        summary_path = args.summary_output or args.output.with_name(
            f"{args.output.stem}_summary.csv"
        )
        cluster_wallet_counts = _classify_cluster_profiles(
            feature_names,
            profiles,
            labels,
            classifier,
            classifier_feature_names,
        )
        summaries = _cluster_summary(
            feature_names, profiles, labels, sample_txids, cluster_wallet_counts
        )
        _write_summary(summary_path, summaries)
        market_shares = _wallet_market_shares(cluster_wallet_counts)
        market_share_path = summary_path.with_name(
            f"{summary_path.stem}_wallet_market_share.csv"
        )
        _write_wallet_market_shares(market_share_path, market_shares)

        noise_count = sum(
            count
            for (_, count), label in zip(profiles, labels, strict=True)
            if label == -1
        )
        transaction_count = sum(count for _, count in profiles)
        print(
            f"{transaction_count:,} transactions from {args.year}, "
            f"{len(profiles):,} distinct "
            f"fingerprint profiles, {len(summaries):,} clusters, "
            f"{noise_count:,} noise transactions (eps={EPS:g})"
        )
        for summary in summaries[:20]:
            print(
                f"Cluster {summary['cluster_id']}: "
                f"{summary['transaction_count']:,} transactions "
                f"({summary['profile_count']:,} profiles); "
                f"sample txid: {summary['sample_txid']}; "
                f"dominant decision-tree wallet: {summary['dominant_wallet']} "
                f"({summary['dominant_wallet_share_percent']:.1f}%)"
            )
            print(f"  Wallet predictions: {summary['wallet_predictions']}")
            if summary["features"]:
                print(f"  Features: {summary['features']}")
        print("Wallet market share (classified transactions; noise excluded):")
        for share in market_shares:
            print(
                f"  {share['wallet']}: {share['market_share_percent']:.2f}% "
                f"({share['transaction_count']:,} transactions, "
                f"{share['cluster_count']:,} clusters)"
            )
        if len(summaries) > 20:
            print(f"... {len(summaries) - 20:,} smaller clusters in the summary CSV")
        _print_threshold_sweep(
            profiles,
            informative_indices,
            tree,
            selected_eps=EPS,
            min_samples=args.min_samples,
        )
        print(f"Profile assignments saved to {args.output}")
        print(f"Cluster summary saved to {summary_path}")
        print(f"Wallet market share saved to {market_share_path}")
    except (PsycopgError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
