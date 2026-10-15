import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .fingerprints import DETECTORS

LABEL_COLUMN = "wallet"
METADATA_COLUMNS = {"txid", LABEL_COLUMN}
ALGORITHMS = ("decision-tree", "random-forest", "knn")
TrainingMode = Literal["evaluate", "identify"]
DATA_DIRECTORY = Path(__file__).parent / "data"
ANALYZER_FEATURES = frozenset(detector.fingerprint_name for detector in DETECTORS)


def first_dataset() -> Path:
    datasets = sorted(DATA_DIRECTORY.glob("*.csv"))
    if not datasets:
        raise ValueError(f"No CSV training dataset found in {DATA_DIRECTORY}")
    return datasets[0]


def load_feature_dataset(path: Path) -> tuple[list[list[int]], list[str], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or LABEL_COLUMN not in reader.fieldnames:
            raise ValueError(f"Dataset must contain a {LABEL_COLUMN!r} column")

        feature_names = [
            name
            for name in reader.fieldnames
            if name not in METADATA_COLUMNS and name in ANALYZER_FEATURES
        ]
        if not feature_names:
            raise ValueError(
                "Dataset must contain at least one feature supported by the analyzer"
            )
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Dataset contains duplicate column names")

        features: list[list[int]] = []
        labels: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Too many CSV values in {path} row {row_number}")

            label = (row.get(LABEL_COLUMN) or "").strip()
            if not label:
                raise ValueError(f"Missing wallet label in {path} row {row_number}")

            try:
                vector = [int(row[name]) for name in feature_names]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid feature value in {path} row {row_number}"
                ) from error

            if any(value not in {-1, 0, 1} for value in vector):
                raise ValueError(
                    f"Fingerprint features must be -1, 0, or 1 in {path} row {row_number}"
                )

            features.append(vector)
            labels.append(label)

    if not features:
        raise ValueError(f"Dataset is empty: {path}")
    return features, labels, feature_names


def _new_classifier(algorithm: str, training_size: int):
    if algorithm == "decision-tree":
        return DecisionTreeClassifier(random_state=22)
    if algorithm == "random-forest":
        return RandomForestClassifier(n_estimators=100, random_state=22)
    if algorithm == "knn":
        if training_size < 1:
            raise ValueError("KNN requires at least one training sample")
        return KNeighborsClassifier(n_neighbors=min(5, training_size))
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _holdout_indices(labels: list[str]) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)

    randomizer = random.Random(22)
    training_indices: list[int] = []
    testing_indices: list[int] = []

    for indices in groups.values():
        randomizer.shuffle(indices)
        if len(indices) >= 2:
            test_count = max(1, round(len(indices) * 0.2))
            testing_indices.extend(indices[:test_count])
            training_indices.extend(indices[test_count:])
        else:
            training_indices.extend(indices)

    return training_indices, testing_indices


def dataset_statistics(
    features: list[list[int]], labels: list[str], feature_names: list[str]
) -> dict:
    wallet_count = len(set(labels))
    return {
        "samples": len(features),
        "features": len(feature_names),
        "wallet_labels": wallet_count,
        "random_baseline_accuracy": 1 / wallet_count,
        "class_distribution": dict(sorted(Counter(labels).items())),
        "unknown_feature_values": sum(value == -1 for row in features for value in row),
        "known_feature_values": sum(value != -1 for row in features for value in row),
    }


def _evaluate(
    classifier,
    test_features: list[list[int]],
    test_labels: list[str],
    all_labels: list[str],
) -> dict:
    predictions = classifier.predict(test_features)
    return _evaluate_predictions(test_labels, predictions, all_labels)


def _evaluate_predictions(
    test_labels: list[str],
    predictions: list[str],
    all_labels: list[str],
) -> dict:
    return {
        "test_samples": len(test_labels),
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
        "precision_macro": float(
            precision_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(test_labels, predictions, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=all_labels,
            target_names=all_labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            test_labels, predictions, labels=all_labels
        ).tolist(),
        "confusion_matrix_labels": all_labels,
    }


def train_models(
    dataset_path: Path,
    *,
    mode: TrainingMode = "evaluate",
) -> tuple[list[dict], dict, list[str]]:
    if mode not in {"evaluate", "identify"}:
        raise ValueError(f"Unsupported training mode: {mode}")

    features, labels, feature_names = load_feature_dataset(dataset_path)
    with dataset_path.open(newline="", encoding="utf-8-sig") as source:
        dataset_fields = csv.DictReader(source).fieldnames or []

    ignored_features = sorted(set(dataset_fields) - METADATA_COLUMNS - ANALYZER_FEATURES)
    training_indices, testing_indices = (
        _holdout_indices(labels) if mode == "evaluate" else ([], [])
    )
    all_labels = sorted(set(labels))

    results: list[dict] = []
    classifiers: dict[str, object] = {}

    for algorithm in ALGORITHMS:
        evaluation = None
        classifier = None

        if mode == "evaluate" and testing_indices:
            classifier = _new_classifier(algorithm, len(training_indices))
            training_features = [features[index] for index in training_indices]
            training_labels = [labels[index] for index in training_indices]
            classifier.fit(training_features, training_labels)

            test_features = [features[index] for index in testing_indices]
            test_labels = [labels[index] for index in testing_indices]
            evaluation = _evaluate(classifier, test_features, test_labels, all_labels)
            evaluation["training_samples"] = len(training_indices)
        elif mode == "identify":
            classifier = _new_classifier(algorithm, len(features))
            classifier.fit(features, labels)
            classifiers[algorithm] = classifier

        statistics = dataset_statistics(features, labels, feature_names)
        statistics["ignored_dataset_features"] = ignored_features
        result = {
            "algorithm": algorithm,
            "statistics": statistics,
            "evaluation": evaluation,
        }

        if classifier is not None and hasattr(classifier, "feature_importances_"):
            result["feature_importances"] = dict(
                sorted(
                    zip(feature_names, map(float, classifier.feature_importances_)),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

        results.append(result)

    return results, classifiers, feature_names
