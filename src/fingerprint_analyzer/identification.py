"""Identify a wallet from a transaction's fingerprint features."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from .analyzer import analyze
from .feature_data import feature_vector
from .model import Transaction
from .provider import ElectrsProvider


@dataclass(frozen=True)
class FingerprintDifference:
    """A known fingerprint feature with different values in two transactions."""

    feature_name: str
    source_value: int
    spending_value: int


@dataclass(frozen=True)
class OutputSpendIdentification:
    """Wallet identification for the transaction that spends one output."""

    output_index: int
    spending_transaction_id: str | None
    wallet_label: str | None
    is_change: bool | None
    wallet_probability: float | None = None
    fingerprint_differences: tuple[FingerprintDifference, ...] = ()
    matching_score: float | None = None
    matching_features: int = 0
    compared_features: int = 0


@dataclass(frozen=True)
class ChangeDetectionResult:
    """Identification results for a transaction and its spent outputs."""

    transaction_id: str
    wallet_label: str | None
    output_spends: list[OutputSpendIdentification]
    wallet_probability: float | None = None


@dataclass(frozen=True)
class OutputFingerprintMatch:
    """Fingerprint similarity between a transaction and an output's spender."""

    output_index: int
    spending_transaction_id: str | None
    matching_score: float | None
    matching_features: int
    compared_features: int


@dataclass(frozen=True)
class FingerprintChangeResult:
    """Fingerprint similarity results for a transaction's spent outputs."""

    transaction_id: str
    output_matches: list[OutputFingerprintMatch]


def _classifier_label(
    feature_values: list[int],
    classifiers: Mapping[str, object],
    classifier_name: str,
) -> str | None:
    """Return a wallet prediction from the selected trained classifier."""
    classifier = classifiers.get(classifier_name)
    if classifier is None:
        raise ValueError(f"A trained {classifier_name!r} classifier is required")

    predicted_label = getattr(classifier, "predict")([feature_values])[0]
    return str(predicted_label)


def _classifier_predictions(
    feature_vectors: list[list[int]],
    classifiers: Mapping[str, object],
    classifier_name: str,
    *,
    include_probabilities: bool,
) -> list[tuple[str | None, float | None]]:
    """Predict wallets in one batch, optionally returning class probabilities."""
    classifier = classifiers.get(classifier_name)
    if classifier is None:
        raise ValueError(f"A trained {classifier_name!r} classifier is required")

    predicted_labels = getattr(classifier, "predict")(feature_vectors)
    probability_rows = None
    if include_probabilities and hasattr(classifier, "predict_proba"):
        probability_rows = getattr(classifier, "predict_proba")(feature_vectors)
    classes = list(getattr(classifier, "classes_", []))

    predictions = []
    for index, predicted_label in enumerate(predicted_labels):
        probability = None
        if probability_rows is not None:
            probability = next(
                (
                    float(score)
                    for label, score in zip(classes, probability_rows[index])
                    if label == predicted_label
                ),
                None,
            )
        predictions.append((str(predicted_label), probability))
    return predictions


async def _fetch_output_spenders(
    transaction: Transaction,
    provider: ElectrsProvider,
    request_semaphore: asyncio.Semaphore | None,
) -> tuple[list[str | None], dict[str, Transaction]]:
    """Fetch each unique spending transaction and align spends with outputs."""
    if request_semaphore is None:
        spending_txids = await provider.get_output_spends(transaction.txid)
    else:
        async with request_semaphore:
            spending_txids = await provider.get_output_spends(transaction.txid)
    if len(spending_txids) != len(transaction.outputs):
        raise ValueError(
            f"Expected spend information for {len(transaction.outputs)} outputs "
            f"of transaction {transaction.txid}, received {len(spending_txids)}"
        )

    unique_spending_txids = list(
        dict.fromkeys(txid for txid in spending_txids if txid is not None)
    )

    async def fetch_spending_transaction(txid: str) -> Transaction:
        if request_semaphore is None:
            return await provider.get_transaction(txid)
        async with request_semaphore:
            return await provider.get_transaction(txid)

    spending_transactions = await asyncio.gather(
        *(fetch_spending_transaction(txid) for txid in unique_spending_txids)
    )
    return spending_txids, dict(zip(unique_spending_txids, spending_transactions))


def _fingerprint_match_score(
    source: Mapping[str, bool | None],
    spending: Mapping[str, bool | None],
    feature_importances: Mapping[str, float],
) -> tuple[float | None, int, int]:
    """Score equal known features by their share of comparable feature importance."""
    comparable_values = [
        (
            source_value,
            spending_value,
            feature_importances.get(feature_name, 0.0),
        )
        for feature_name, source_value in source.items()
        if feature_name in feature_importances
        and source_value is not None
        and (spending_value := spending.get(feature_name)) is not None
    ]
    compared_features = len(comparable_values)
    matching_features = sum(
        source_value == spending_value
        for source_value, spending_value, _ in comparable_values
    )
    comparable_importance = sum(
        importance for _, _, importance in comparable_values
    )
    if not compared_features or comparable_importance == 0:
        return None, matching_features, compared_features
    matching_importance = sum(
        importance
        for source_value, spending_value, importance in comparable_values
        if source_value == spending_value
    )
    return (
        matching_importance / comparable_importance,
        matching_features,
        compared_features,
    )


def identify_transaction(
    transaction: Transaction,
    classifiers: Mapping[str, object],
    feature_names: list[str],
    *,
    classifier_name: str = "decision-tree",
) -> str | None:
    """Analyze a transaction and identify its wallet with the selected model.

    The transaction is first converted into the requested feature order. Each
    feature is passed to the selected trained classifier, which predicts a
    wallet label.
    """
    report = analyze(transaction)
    ordered_features = feature_vector(report, feature_names)
    return _classifier_label(ordered_features, classifiers, classifier_name)


async def identify_transaction_changes(
    transaction: Transaction,
    provider: ElectrsProvider,
    classifiers: Mapping[str, object],
    feature_names: list[str],
    *,
    feature_importances: Mapping[str, float] | None = None,
    include_probabilities: bool = True,
    include_fingerprint_differences: bool = True,
    request_semaphore: asyncio.Semaphore | None = None,
    classifier_name: str = "decision-tree",
) -> ChangeDetectionResult:
    """Identify a transaction and check which spent outputs return to its wallet.

    For every spent output, the corresponding spending transaction is fetched
    from the provider and classified with the same selected model as the original
    transaction. An output is considered change when both transactions have the
    same identified wallet label. This is a heuristic; transactions without a
    prediction cannot be assessed. An optional semaphore can cap concurrent
    provider requests when processing multiple transactions.
    """
    report = analyze(transaction)
    ordered_features = feature_vector(report, feature_names)
    spending_txids, spending_transactions = await _fetch_output_spenders(
        transaction, provider, request_semaphore
    )
    spending_reports = {
        txid: analyze(spending_transaction)
        for txid, spending_transaction in spending_transactions.items()
    }
    spending_features = {
        txid: feature_vector(report, feature_names)
        for txid, report in spending_reports.items()
    }
    predicted_values = _classifier_predictions(
        [ordered_features, *spending_features.values()],
        classifiers,
        classifier_name,
        include_probabilities=include_probabilities,
    )
    wallet_label, wallet_probability = predicted_values[0]
    spending_predictions = dict(
        zip(spending_features, predicted_values[1:])
    )

    spending_matches = {}
    for txid, spending_report in spending_reports.items():
        if feature_importances is None:
            matching_score, matching_features, compared_features = None, 0, 0
        else:
            (
                matching_score,
                matching_features,
                compared_features,
            ) = _fingerprint_match_score(
                report.fingerprints,
                spending_report.fingerprints,
                feature_importances,
            )
        spending_matches[txid] = (
            matching_score,
            matching_features,
            compared_features,
            tuple(
                FingerprintDifference(feature_name, source_value, spending_value)
                for feature_name, source_value, spending_value in zip(
                    feature_names, ordered_features, spending_features[txid]
                )
                if source_value != -1
                and spending_value != -1
                and source_value != spending_value
            )
            if include_fingerprint_differences
            else (),
        )

    output_spends = []
    for output_index, spending_txid in enumerate(spending_txids):
        spending_label, spending_probability = (
            spending_predictions[spending_txid]
            if spending_txid is not None
            else (None, None)
        )
        (
            matching_score,
            matching_features,
            compared_features,
            fingerprint_differences,
        ) = (
            spending_matches[spending_txid]
            if spending_txid is not None
            else (None, 0, 0, ())
        )
        if wallet_label is None or spending_label is None:
            is_change = None
        else:
            is_change = wallet_label == spending_label

        output_spends.append(
            OutputSpendIdentification(
                output_index=output_index,
                spending_transaction_id=spending_txid,
                wallet_label=spending_label,
                wallet_probability=spending_probability,
                is_change=is_change,
                fingerprint_differences=fingerprint_differences,
                matching_score=matching_score,
                matching_features=matching_features,
                compared_features=compared_features,
            )
        )

    return ChangeDetectionResult(
        transaction_id=transaction.txid,
        wallet_label=wallet_label,
        wallet_probability=wallet_probability,
        output_spends=output_spends,
    )


async def compare_transaction_fingerprints(
    transaction: Transaction,
    provider: ElectrsProvider,
    feature_importances: Mapping[str, float],
    *,
    request_semaphore: asyncio.Semaphore | None = None,
) -> FingerprintChangeResult:
    """Compare each output spender's fingerprint using model importance weights.

    The match score is the share of importance for features known in both
    reports that have the same value. Unknown features are excluded.
    """
    source_fingerprints = analyze(transaction).fingerprints
    spending_txids, spending_transactions = await _fetch_output_spenders(
        transaction, provider, request_semaphore
    )
    spending_scores = {
        txid: _fingerprint_match_score(
            source_fingerprints,
            analyze(spending_transaction).fingerprints,
            feature_importances,
        )
        for txid, spending_transaction in spending_transactions.items()
    }

    output_matches = []
    for output_index, spending_txid in enumerate(spending_txids):
        if spending_txid is None:
            matching_score, matching_features, compared_features = None, 0, 0
        else:
            (
                matching_score,
                matching_features,
                compared_features,
            ) = spending_scores[spending_txid]
        output_matches.append(
            OutputFingerprintMatch(
                output_index=output_index,
                spending_transaction_id=spending_txid,
                matching_score=matching_score,
                matching_features=matching_features,
                compared_features=compared_features,
            )
        )

    return FingerprintChangeResult(
        transaction_id=transaction.txid,
        output_matches=output_matches,
    )


def identify_fingerprint_features(
    feature_values: list[int],
    classifiers: Mapping[str, object],
) -> str | None:
    """Identify a wallet from an ordered feature vector using the decision tree."""
    return _classifier_label(feature_values, classifiers, "decision-tree")
