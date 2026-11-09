"""Define command-line commands for transaction analysis and model workflows."""

import argparse
import asyncio
import json
import math
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
    compare_transaction_fingerprints,
    identify_transaction_changes,
)
from .ml import first_dataset, train_classifier, train_models
from .provider import ElectrsProvider, ProviderError

MODEL_NAMES = {
    "decision-tree": "Decision Tree",
    "random-forest": "Random Forest",
    "extra-trees": "Extra Trees",
    "gradient-boosting": "Gradient Boosting",
    "logistic-regression": "Logistic Regression",
    "svm-linear": "Linear SVM",
    "svm-rbf": "RBF SVM",
    "knn": "K-Nearest Neighbors",
    "gaussian-nb": "Gaussian Naive Bayes",
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


def _wallet_with_probability(
    wallet_label: str | None, probability: float | None
) -> str:
    """Format an identified wallet with its prediction probability."""
    label = wallet_label or "no prediction"
    if probability is None:
        return f"{label} (probability unavailable)"
    return f"{label} ({probability:.1%})"


def _tree_feature_importances(classifier, feature_names: list[str]) -> dict[str, float]:
    """Map a tree classifier's feature importances to feature names."""
    importance_model = (
        classifier[-1] if hasattr(classifier, "steps") else classifier
    )
    return dict(
        zip(
            feature_names,
            getattr(importance_model, "feature_importances_", []),
        )
    )


def _fingerprint_value(value: int) -> str:
    """Render a known fingerprint feature value for terminal output."""
    return "yes" if value == 1 else "no"


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


def _dataset_argument(parser: argparse.ArgumentParser) -> None:
    """Add an optional path to the fingerprint dataset used to train models."""
    parser.add_argument(
        "--dataset",
        type=Path,
        help="CSV training dataset (defaults to the first CSV in the data directory)",
    )


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

    train_parser = subparsers.add_parser(
        "train", help="train all classifiers using the first CSV in the data directory"
    )
    _dataset_argument(train_parser)

    identify_parser = subparsers.add_parser(
        "identify", help="identify the wallet for one or more transactions"
    )
    identify_parser.add_argument("txid", nargs="+", help="one or more transaction IDs")
    _dataset_argument(identify_parser)
    _network_arguments(identify_parser)

    change_parser = subparsers.add_parser(
        "change",
        help="compare transaction fingerprints with output spenders using model importance",
    )
    change_parser.add_argument("txid", help="transaction ID to analyze")
    _dataset_argument(change_parser)
    _network_arguments(change_parser)

    ml_change_parser = subparsers.add_parser(
        "ml-change",
        help="identify likely change outputs by comparing predicted wallet labels",
    )
    ml_change_parser.add_argument("txid", help="transaction ID to analyze")
    _dataset_argument(ml_change_parser)
    _network_arguments(ml_change_parser)

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


def _run_train(args) -> int:
    """Train and print evaluation results for each supported classifier."""
    try:
        dataset_path = args.dataset or first_dataset()
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
            print("  Cross-validation evaluation unavailable.")
        else:
            print(
                f"  {evaluation['cross_validation_folds']}-fold stratified "
                f"cross-validation: {evaluation['test_samples']} out-of-fold "
                "predictions"
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


def _classifier_scores(classifier, features: list[int]) -> list[float]:
    """Return class scores, normalizing decision values for display when needed."""
    if hasattr(classifier, "predict_proba"):
        return list(classifier.predict_proba([features])[0])

    decision_values = classifier.decision_function([features])[0]
    if len(classifier.classes_) == 2:
        decision_values = [-float(decision_values), float(decision_values)]
    else:
        decision_values = [float(value) for value in decision_values]

    exponentials = [
        math.exp(value - max(decision_values)) for value in decision_values
    ]
    total = sum(exponentials)
    return [value / total for value in exponentials]


async def _run_identify(args) -> int:
    """Classify transactions with the decision tree and show model comparisons."""
    try:
        dataset_path = args.dataset or first_dataset()
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
            probabilities = _classifier_scores(classifier, features)
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
        predictions.append(
            {
                "txid": transaction.txid,
                "features": features,
                "predictions": model_predictions,
            }
        )

    print("Wallet identification")
    print(f"Dataset: {dataset_path.name}")
    for prediction in predictions:
        print(f"\nTransaction: {prediction['txid']}")
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


async def _run_ml_change(args) -> int:
    """Identify likely change outputs by comparing spending wallet labels."""
    try:
        dataset_path = args.dataset or first_dataset()
        extra_trees, feature_names = train_classifier(dataset_path, "extra-trees")

        provider = _provider(args)
        transaction = await provider.get_transaction(args.txid)
        result = await identify_transaction_changes(
            transaction,
            provider,
            {"extra-trees": extra_trees},
            feature_names,
            classifier_name="extra-trees",
        )
    except (OSError, ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    feature_importances = _tree_feature_importances(extra_trees, feature_names)
    importance_ranks = {
        feature_name: rank
        for rank, feature_name in enumerate(
            sorted(
                feature_names,
                key=lambda name: feature_importances.get(name, 0),
                reverse=True,
            ),
            start=1,
        )
    }

    print(f"Dataset:          {dataset_path.name}")
    print(f"Transaction:      {result.transaction_id}")
    print(
        "Extra-trees wallet: "
        f"{_wallet_with_probability(result.wallet_label, result.wallet_probability)}"
    )

    change_candidates = [
        output
        for output in result.output_spends
        if output.is_change is True
    ]
    print("\nChange candidates:")
    if change_candidates:
        print(
            "".join(
                f"  - Output {output.output_index}: "
                f"{_wallet_with_probability(output.wallet_label, output.wallet_probability)}\n"
                for output in change_candidates
            )
        )
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

        if output.is_change is True:
            change_label = "Same wallet as source"
        elif output.is_change is False:
            change_label = "Different wallet from source"
        else:
            change_label = "Undetermined"
        if output.is_change is True:
            change_label = _bold(change_label)
        print(f"  Status:       {change_label}")
        print(f"  Spent by:     {output.spending_transaction_id}")
        print(
            "  Wallet label: "
            f"{_wallet_with_probability(output.wallet_label, output.wallet_probability)}"
        )
        if output.is_change is True:
            top_conflicts = sorted(
                output.fingerprint_differences,
                key=lambda difference: importance_ranks.get(
                    difference.feature_name, len(feature_names) + 1
                ),
            )[:3]
            if top_conflicts:
                print("  top 3 Fingerprint conflicts:")
                for difference in top_conflicts:
                    rank = importance_ranks.get(
                        difference.feature_name, len(feature_names)
                    )
                    print(
                        f"    - {difference.feature_name}: "
                        f"source={_fingerprint_value(difference.source_value)}, "
                        f"spending={_fingerprint_value(difference.spending_value)} "
                        f"(importance rank {rank}/{len(feature_names)})"
                    )
            else:
                print("  Fingerprint conflicts: none among known features")

    return 0


async def _run_change(args) -> int:
    """Compare each output spender's fingerprint with the source transaction."""
    try:
        dataset_path = args.dataset or first_dataset()
        extra_trees, feature_names = train_classifier(dataset_path, "extra-trees")
        feature_importances = _tree_feature_importances(extra_trees, feature_names)
        provider = _provider(args)
        transaction = await provider.get_transaction(args.txid)
        result = await compare_transaction_fingerprints(
            transaction, provider, feature_importances
        )
    except (OSError, ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset:    {dataset_path.name}")
    print(f"Transaction: {result.transaction_id}")
    if not result.output_matches:
        print("No outputs found.")

    scored_outputs = [
        (output, output.matching_score)
        for output in result.output_matches
        if output.matching_score is not None
    ]
    print("\nChange candidates (highest importance-weighted match):")
    if scored_outputs:
        highest_score = max(score for _, score in scored_outputs)
        for output, score in scored_outputs:
            if score == highest_score:
                print(
                    f"  - Output {output.output_index}: "
                    f"{score:.1%}"
                )
    else:
        print("None (no comparable fingerprint scores)")

    for output in result.output_matches:
        print(f"\nOutput {output.output_index}")
        if output.spending_transaction_id is None:
            print("  Status:          Unspent")
            continue

        print(f"  Spent by:        {output.spending_transaction_id}")
        if output.matching_score is None:
            if output.compared_features:
                print(
                    "  Weighted fingerprint match: n/a "
                    "(comparable features have zero total importance)"
                )
            else:
                print(
                    "  Weighted fingerprint match: n/a "
                    "(no comparable known features)"
                )
        else:
            print(
                f"  Weighted fingerprint match: {output.matching_score:.1%} "
                f"({output.matching_features}/{output.compared_features} fingerprints matching)"
            )

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
    store = None
    try:
        store = FingerprintStore(args.database_url)
        store.ensure_schema()
        await collect_forever(
            _provider(args),
            store,
            args.start_height,
            args.poll_interval,
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
        exit_code = _run_train(args)
    elif args.command == "ml-change":
        exit_code = asyncio.run(_run_ml_change(args))
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
