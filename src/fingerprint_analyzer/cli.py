"""Define command-line commands for transaction analysis and model workflows."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from psycopg import Error as PsycopgError

from .analyzer import analyze
from .collector import collect_forever
from .database import FingerprintStore
from .feature_data import (
    NETWORK_URLS,
    analyze_transaction_rows,
    feature_vector,
    load_transaction_list,
    write_feature_dataset,
)
from .identification import (
    identify_fingerprint_features,
    identify_transaction_changes,
)
from .ml import first_dataset, train_classifier, train_models
from .provider import ElectrsProvider, ProviderError

MODEL_NAMES = {
    "decision-tree": "Decision Tree",
    "random-forest": "Random Forest",
    "knn": "K-Nearest Neighbors",
}


def _bold(text: str) -> str:
    """Wrap text in terminal bold formatting when color output is available."""
    if (
        sys.stdout.isatty()
        and not os.environ.get("NO_COLOR")
        and os.environ.get("TERM") != "dumb"
    ):
        return f"\033[1m{text}\033[0m"
    return text


def _mempool_url_argument(parser: argparse.ArgumentParser) -> None:
    """Add the option used to override the default API endpoint."""
    parser.add_argument(
        "--mempool-url",
        help="base URL of an Electrs-compatible mempool API (overrides the network default)",
    )


def _network_arguments(parser: argparse.ArgumentParser) -> None:
    """Add network selection and the optional API URL to a parser."""
    network_group = parser.add_mutually_exclusive_group()
    network_group.add_argument("--testnet", action="store_true", help="use Bitcoin testnet3")
    network_group.add_argument("--testnet4", action="store_true", help="use Bitcoin testnet4")
    _mempool_url_argument(parser)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all supported commands."""
    parser = argparse.ArgumentParser(
        description="Analyze Bitcoin transaction fingerprints and identify wallets"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tx_parser = subparsers.add_parser(
        "tx", aliases=["transaction"], help="analyze one or more transactions"
    )
    tx_parser.add_argument("txid", nargs="+", help="one or more transaction IDs")
    _network_arguments(tx_parser)

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="convert labeled transactions from a CSV to fingerprint features",
    )
    dataset_parser.add_argument("input_csv", type=Path, help="CSV with txid,wallet_label,network")
    _mempool_url_argument(dataset_parser)

    subparsers.add_parser(
        "train", help="train all classifiers using the first CSV in the data directory"
    )

    identify_parser = subparsers.add_parser(
        "identify", help="identify the wallet for one or more transactions"
    )
    identify_parser.add_argument("txid", nargs="+", help="one or more transaction IDs")
    _network_arguments(identify_parser)

    change_parser = subparsers.add_parser(
        "change", help="identify likely change outputs by analyzing their spenders"
    )
    change_parser.add_argument("txid", help="transaction ID to analyze")
    _network_arguments(change_parser)

    collect_parser = subparsers.add_parser(
        "collect",
        help="continuously analyze confirmed transactions and store fingerprints in PostgreSQL",
    )
    _network_arguments(collect_parser)
    collect_parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL (defaults to DATABASE_URL)",
    )
    collect_parser.add_argument(
        "--start-height",
        type=int,
        help="first block to collect; by default starts at the current tip",
    )
    collect_parser.add_argument(
        "--poll-interval",
        type=float,
        default=10,
        help="seconds between checks for new blocks (default: 10)",
    )
    collect_parser.add_argument(
        "--block-interval",
        type=int,
        default=1,
        help="store every Nth block (default: 1, store every block)",
    )
    return parser


def _provider(args) -> ElectrsProvider:
    """Create a provider using the URL override or selected Bitcoin network."""
    if args.mempool_url:
        base_url = args.mempool_url
    elif args.testnet4:
        base_url = NETWORK_URLS["testnet4"]
    elif args.testnet:
        base_url = NETWORK_URLS["testnet"]
    else:
        base_url = NETWORK_URLS["mainnet"]
    return ElectrsProvider(base_url)


async def _fetch_transactions(args):
    """Fetch all transaction IDs provided to a transaction-oriented command."""
    provider = _provider(args)
    return await asyncio.gather(*(provider.get_transaction(txid) for txid in args.txid))


async def _run_tx(args) -> int:
    """Fetch transactions and print their fingerprint reports as JSON."""
    try:
        transactions = await _fetch_transactions(args)
        reports = [analyze(transaction) for transaction in transactions]
    except (ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if len(reports) == 1:
        output = reports[0].to_dict()
    else:
        output = [report.to_dict() for report in reports]

    print(json.dumps(output, indent=2))
    return 0


async def _run_dataset(args) -> int:
    """Convert a labeled transaction CSV into a fingerprint feature CSV."""
    try:
        rows = load_transaction_list(args.input_csv, require_label=True)
        reports = await analyze_transaction_rows(rows, args.mempool_url)
        if not reports:
            raise ValueError("No transactions could be fetched; no CSV was written")
        feature_names = write_feature_dataset(sys.stdout, reports)
    except (OSError, ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Created {len(reports)} rows with {len(feature_names)} fingerprint features",
        file=sys.stderr,
    )
    return 0


def _run_train() -> int:
    """Train and print evaluation results for each supported classifier."""
    try:
        dataset_path = first_dataset()
        results, _, _ = train_models(dataset_path, mode="evaluate")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    statistics = results[0]["statistics"]
    print(f"Training dataset: {dataset_path}")
    print(
        f"Samples: {statistics['samples']} | Features: {statistics['features']} | "
        f"Wallets: {statistics['wallet_labels']}"
    )

    class_counts = ", ".join(
        f"{label}: {count}" for label, count in statistics["class_distribution"].items()
    )
    print(f"Class distribution: {class_counts}")
    print(
        f"Random baseline (uniform guess): "
        f"{statistics['random_baseline_accuracy']:.1%} "
        f"(1/{statistics['wallet_labels']} labels)"
    )
    print(
        f"Unknown feature values: {statistics['unknown_feature_values']} / "
        f"{statistics['known_feature_values'] + statistics['unknown_feature_values']}"
    )

    ignored_features = statistics["ignored_dataset_features"]
    if ignored_features:
        print(
            "Ignored dataset features not produced by the analyzer: "
            + ", ".join(ignored_features)
        )

    for result in results:
        evaluation = result["evaluation"]
        model_name = MODEL_NAMES.get(result["algorithm"], result["algorithm"])
        print(f"\n{_bold(model_name)}")

        if evaluation is None:
            print("  Holdout evaluation unavailable (no test samples).")
        else:
            print(
                f"  Holdout: {evaluation['training_samples']} training / "
                f"{evaluation['test_samples']} test samples"
            )
            print(
                _bold(
                    f"  Accuracy: {evaluation['accuracy']:.1%} | "
                    f"Balanced accuracy: {evaluation['balanced_accuracy']:.1%} | "
                    f"Macro F1: {evaluation['f1_macro']:.1%} | "
                    f"Weighted F1: {evaluation['f1_weighted']:.1%}"
                )
            )
            print("  Per-class one-vs-rest metrics:")
            report = evaluation["classification_report"]
            labels = evaluation["confusion_matrix_labels"]
            matrix = evaluation["confusion_matrix"]
            total = sum(sum(row) for row in matrix)
            label_width = max(len("Class"), *(len(label) for label in labels))
            metric_columns = ("TPr", "TNr", "FPr", "FNr", "Precision", "Recall", "F1")
            metric_widths = {
                "TPr": 7,
                "TNr": 7,
                "FPr": 7,
                "FNr": 7,
                "Precision": 9,
                "Recall": 7,
                "F1": 7,
            }
            print(
                f"    {'Class':<{label_width}}  "
                + "  ".join(
                    f"{column:>{metric_widths[column]}}"
                    for column in metric_columns
                )
            )
            print(
                f"    {'-' * label_width}  "
                + "  ".join("-" * metric_widths[column] for column in metric_columns)
            )
            for index, label in enumerate(labels):
                metrics = report[label]
                # Derive each one-vs-rest count from the confusion matrix.
                true_positive = matrix[index][index]
                false_negative = sum(matrix[index]) - true_positive
                false_positive = (
                    sum(row[index] for row in matrix) - true_positive
                )
                true_negative = total - true_positive - false_negative - false_positive
                actual_positives = true_positive + false_negative
                actual_negatives = true_negative + false_positive
                true_positive_rate = (
                    f"{100 * true_positive / actual_positives:.1f}%"
                    if actual_positives
                    else "n/a"
                )
                false_negative_rate = (
                    f"{100 * false_negative / actual_positives:.1f}%"
                    if actual_positives
                    else "n/a"
                )
                true_negative_rate = (
                    f"{100 * true_negative / actual_negatives:.1f}%"
                    if actual_negatives
                    else "n/a"
                )
                false_positive_rate = (
                    f"{100 * false_positive / actual_negatives:.1f}%"
                    if actual_negatives
                    else "n/a"
                )
                print(
                    f"    {label:<{label_width}}  "
                    f"{true_positive_rate:>7}  {true_negative_rate:>7}  "
                    f"{false_positive_rate:>7}  {false_negative_rate:>7}  "
                    f"{metrics['precision']:>9.1%}  {metrics['recall']:>7.1%}  "
                    f"{metrics['f1-score']:>7.1%}"
                )

            column_width = max(5, *(len(label) for label in labels))
            axis_label = "Actual / Predicted"
            row_label_width = max(len(axis_label), column_width)
            print("  Confusion matrix:")
            print(
                f"    {axis_label:<{row_label_width}} | "
                + " | ".join(f"{label:>{column_width}}" for label in labels)
            )
            print(
                f"    {'-' * row_label_width}-+-"
                + "-+-".join("-" * column_width for _ in labels)
            )
            for label, row in zip(labels, matrix):
                print(
                    f"    {label:<{row_label_width}} | "
                    + " | ".join(f"{count:>{column_width}}" for count in row)
                )

        importances = result.get("feature_importances")
        if importances:
            top_features = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:5]
            print("  Top features:")
            for feature, importance in top_features:
                print(f"    {feature}: {importance:.1%}")

    return 0


def _model_reasons(
    feature_names: list[str],
    features: list[int],
    feature_importances: dict[str, float] | None,
) -> str:
    """Choose a short list of known features that explain a model prediction."""
    known = [
        (feature, value)
        for feature, value in zip(feature_names, features)
        if value != -1
    ]

    if feature_importances:
        ranked = []
        for feature, value in known:
            importance = feature_importances.get(feature, 0.0)
            if value == 1:
                ranked.append((importance, feature, 1))
            elif value == 0:
                ranked.append((importance * 0.3, feature, 0))
        if ranked:
            ranked.sort(key=lambda item: item[0], reverse=True)
            return ", ".join(f"{feature}={value}" for _, feature, value in ranked[:3])

    if known:
        strongest = sorted(known, key=lambda item: abs(item[1]), reverse=True)[:3]
        return ", ".join(f"{feature}={value}" for feature, value in strongest)

    return "no usable signal"


async def _run_identify(args) -> int:
    """Classify each requested transaction using all trained models."""
    try:
        dataset_path = first_dataset()
        results, classifiers, feature_names = train_models(dataset_path, mode="identify")
        transactions = await _fetch_transactions(args)
        reports = [analyze(transaction) for transaction in transactions]
    except (OSError, ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    missing_features = sorted(set(feature_names) - set(reports[0].fingerprints) if reports else set())
    if missing_features:
        print(
            "Warning: dataset features unavailable from the analyzer are treated "
            f"as unknown (-1): {', '.join(missing_features)}",
            file=sys.stderr,
        )

    model_importances = {
        result["algorithm"]: result.get("feature_importances")
        for result in results
    }

    predictions = []
    for transaction, report in zip(transactions, reports):
        features = feature_vector(report, feature_names)
        model_predictions = {}
        for algorithm, classifier in classifiers.items():
            probabilities = classifier.predict_proba([features])[0]
            scores = sorted(
                enumerate(probabilities),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
            model_predictions[algorithm] = [
                {
                    "wallet_label": str(classifier.classes_[index]),
                    "score": float(score),
                }
                for index, score in scores
            ]
        identified_label = identify_fingerprint_features(features, classifiers)
        predictions.append(
            {
                "txid": transaction.txid,
                "features": features,
                "identified_label": identified_label,
                "predictions": model_predictions,
            }
        )

    print("Wallet identification")
    print(f"Dataset: {dataset_path.name}")
    for prediction in predictions:
        print(f"\nTransaction: {prediction['txid']}")
        label = prediction["identified_label"]
        print(
            "  Majority-vote label: "
            + (label if label is not None else "no majority")
        )
        for model_index, (algorithm, items) in enumerate(prediction["predictions"].items(), start=1):
            model_name = MODEL_NAMES.get(algorithm, algorithm)
            print(f"\n  [{model_index}] {model_name}")
            for rank, item in enumerate(items, start=1):
                contender = (
                    f"      {rank}. {item['wallet_label']} "
                    f"({item['score']:.1%})"
                )
                print(_bold(contender) if rank == 1 else contender)
            reasons = _model_reasons(
                feature_names=feature_names,
                features=prediction["features"],
                feature_importances=model_importances.get(algorithm),
            )
            print(f"    Reasons: {reasons}")

    print("\n")
    return 0


async def _run_change(args) -> int:
    """Identify likely change outputs by comparing spending wallet labels."""
    try:
        dataset_path = first_dataset()
        decision_tree, feature_names = train_classifier(
            dataset_path, "decision-tree"
        )

        provider = _provider(args)
        transaction = await provider.get_transaction(args.txid)
        result = await identify_transaction_changes(
            transaction, provider, {"decision-tree": decision_tree}, feature_names
        )
    except (OSError, ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset:          {dataset_path.name}")
    print(f"Transaction:      {result.transaction_id}")
    print(f"Decision-tree wallet: {result.wallet_label or 'no prediction'}")

    likely_change_outputs = [
        output.output_index
        for output in result.output_spends
        if output.is_change is True
    ]
    print("\nLikely change outputs:")
    if likely_change_outputs:
        print("".join(f"  - Output {index}\n" for index in likely_change_outputs))
    else:
        print("None identified")

    if not result.output_spends:
        print("No outputs found.")

    for output in result.output_spends:
        output_heading = f"Output {output.output_index}"
        if output.is_change is True:
            output_heading = _bold(output_heading)
        print(f"\n{output_heading}")
        if output.spending_transaction_id is None:
            print("  Status:       Unspent")
            continue

        wallet_label = output.wallet_label or "no majority"
        if output.is_change is True:
            change_label = "Likely change"
        elif output.is_change is False:
            change_label = "Not identified as change"
        else:
            change_label = "Undetermined"
        if output.is_change is True:
            change_label = _bold(change_label)
        print(f"  Status:       {change_label}")
        print(f"  Spent by:     {output.spending_transaction_id}")
        print(f"  Wallet label: {wallet_label}")

    return 0


async def _run_collect(args) -> int:
    """Validate collector settings and persist confirmed transaction data."""
    if not args.database_url:
        print(
            "Error: provide --database-url or set DATABASE_URL",
            file=sys.stderr,
        )
        return 1
    if args.start_height is not None and args.start_height < 0:
        print("Error: --start-height must not be negative", file=sys.stderr)
        return 1
    if args.poll_interval <= 0:
        print("Error: --poll-interval must be greater than zero", file=sys.stderr)
        return 1
    if args.block_interval <= 0:
        print("Error: --block-interval must be greater than zero", file=sys.stderr)
        return 1

    store = None
    try:
        store = FingerprintStore(args.database_url)
        store.ensure_schema()
        await collect_forever(
            _provider(args),
            store,
            args.start_height,
            args.poll_interval,
            args.block_interval,
        )
    except (OSError, PsycopgError, ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()
    return 0


def main() -> None:
    """Parse command-line arguments and dispatch to the matching command."""
    args = build_parser().parse_args()
    if args.command in {"tx", "transaction"}:
        exit_code = asyncio.run(_run_tx(args))
    elif args.command == "dataset":
        exit_code = asyncio.run(_run_dataset(args))
    elif args.command == "train":
        exit_code = _run_train()
    elif args.command == "change":
        exit_code = asyncio.run(_run_change(args))
    elif args.command == "collect":
        try:
            exit_code = asyncio.run(_run_collect(args))
        except KeyboardInterrupt:
            print("Collector stopped.", file=sys.stderr)
            exit_code = 130
    else:
        exit_code = asyncio.run(_run_identify(args))
    raise SystemExit(exit_code)
