"""Load fingerprint datasets, train classifiers, and evaluate their predictions."""

import csv
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .fingerprints import DETECTORS

LABEL_COLUMN = "wallet"
METADATA_COLUMNS = {"txid", LABEL_COLUMN}
ALGORITHMS = (
    "decision-tree",
    "random-forest",
    "extra-trees",
    "gradient-boosting",
    "logistic-regression",
    "svm-linear",
    "svm-rbf",
    "knn",
    "gaussian-nb",
)
TrainingMode = Literal["evaluate", "identify"]
DATA_DIRECTORY = Path(__file__).parent / "data"
ANALYZER_FEATURES = frozenset(detector.fingerprint_name for detector in DETECTORS)


def first_dataset() -> Path:
    """Return the alphabetically first bundled CSV dataset."""
    datasets = sorted(DATA_DIRECTORY.glob("*.csv"))
    if not datasets:
        raise ValueError(f"No CSV training dataset found in {DATA_DIRECTORY}")
    return datasets[0]


def load_feature_dataset(path: Path) -> tuple[list[list[int]], list[str], list[str]]:
    """Load supported fingerprint columns and wallet labels from a CSV file."""
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


def _new_classifier(
    algorithm: str,
    training_size: int,
):
    """Create a configured classifier, adapting KNN to the available samples."""
    if algorithm == "decision-tree":
        return DecisionTreeClassifier(
            max_features="sqrt",
            random_state=22,
        )
    if algorithm == "random-forest":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_features="sqrt",
            random_state=22,
        )
    if algorithm == "extra-trees":
        return ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_features="sqrt",
            random_state=22,
        )
    if algorithm == "gradient-boosting":
        return GradientBoostingClassifier(random_state=22)
    if algorithm == "knn":
        if training_size < 1:
            raise ValueError("KNN requires at least one training sample")
        return KNeighborsClassifier(n_neighbors=min(5, training_size))
    if algorithm == "logistic-regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=22),
        )
    if algorithm == "svm-linear":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", random_state=22),
        )
    if algorithm == "svm-rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", random_state=22),
        )
    if algorithm == "gaussian-nb":
        return GaussianNB()
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def train_classifier(dataset_path: Path, algorithm: str) -> tuple[object, list[str]]:
    """Train one classifier on the full dataset for identification."""
    features, labels, feature_names = load_feature_dataset(dataset_path)
    classifier = _new_classifier(algorithm, len(features))
    classifier.fit(features, labels)
    return classifier, feature_names


def _stratified_folds(labels: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic stratified folds, requiring at least two samples per class."""
    class_counts = Counter(labels)
    if not class_counts:
        raise ValueError("Dataset must contain at least one labeled sample")
    min_class_count = min(class_counts.values())
    if min_class_count < 2:
        raise ValueError(
            "Stratified cross-validation requires at least two samples per wallet label"
        )

    splitter = StratifiedKFold(
        n_splits=min(5, min_class_count),
        shuffle=True,
        random_state=22,
    )
    return list(splitter.split(np.zeros(len(labels)), labels))


def dataset_statistics(
    features: list[list[int]], labels: list[str], feature_names: list[str]
) -> dict:
    """Summarize dataset size, label balance, and known/unknown feature values."""
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


def _evaluate_predictions(
    test_labels: list[str],
    predictions: list[str],
    all_labels: list[str],
) -> dict:
    """Calculate metrics and a confusion matrix from known labels and predictions."""
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
    """Train all supported models for evaluation or full-dataset identification."""
    if mode not in {"evaluate", "identify"}:
        raise ValueError(f"Unsupported training mode: {mode}")

    features, labels, feature_names = load_feature_dataset(dataset_path)
    with dataset_path.open(newline="", encoding="utf-8-sig") as source:
        dataset_fields = csv.DictReader(source).fieldnames or []

    ignored_features = sorted(set(dataset_fields) - METADATA_COLUMNS - ANALYZER_FEATURES)
    folds = _stratified_folds(labels) if mode == "evaluate" else []
    all_labels = sorted(set(labels))

    results: list[dict] = []
    classifiers: dict[str, object] = {}

    for algorithm in ALGORITHMS:
        evaluation = None
        classifier = None
        fold_importances: list[np.ndarray] = []

        if mode == "evaluate":
            predictions: list[str | None] = [None] * len(labels)
            for training_indices, testing_indices in folds:
                training_features = [features[index] for index in training_indices]
                training_labels = [labels[index] for index in training_indices]
                classifier = _new_classifier(algorithm, len(training_indices))
                classifier.fit(training_features, training_labels)

                model = classifier[-1] if hasattr(classifier, "steps") else classifier
                if hasattr(model, "feature_importances_"):
                    fold_importances.append(model.feature_importances_)

                test_features = [features[index] for index in testing_indices]
                fold_predictions = classifier.predict(test_features)
                for index, prediction in zip(testing_indices, fold_predictions):
                    predictions[index] = prediction

            if any(prediction is None for prediction in predictions):
                raise RuntimeError("Cross-validation did not predict every sample")
            evaluation = _evaluate_predictions(
                labels,
                [prediction for prediction in predictions if prediction is not None],
                all_labels,
            )
            evaluation["cross_validation_folds"] = len(folds)
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

        if fold_importances:
            mean_importances = np.mean(fold_importances, axis=0)
            importance_names = feature_names
            result["feature_importances"] = dict(
                sorted(
                    zip(
                        importance_names,
                        map(float, mean_importances),
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
        elif classifier is not None:
            importance_model = (
                classifier[-1] if hasattr(classifier, "steps") else classifier
            )
            if hasattr(importance_model, "feature_importances_"):
                result["feature_importances"] = dict(
                    sorted(
                        zip(
                            feature_names,
                            map(float, importance_model.feature_importances_),
                        ),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
        results.append(result)

    return results, classifiers, feature_names
