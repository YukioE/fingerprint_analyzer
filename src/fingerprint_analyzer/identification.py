"""Identify a wallet from a transaction's fingerprint features."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from .analyzer import analyze
from .feature_data import feature_vector
from .model import Transaction
from .provider import ElectrsProvider


@dataclass(frozen=True)
class OutputSpendIdentification:
    """Wallet identification for the transaction that spends one output."""

    output_index: int
    spending_transaction_id: str | None
    wallet_label: str | None
    is_change: bool | None


@dataclass(frozen=True)
class ChangeDetectionResult:
    """Identification results for a transaction and its spent outputs."""

    transaction_id: str
    wallet_label: str | None
    output_spends: list[OutputSpendIdentification]


def _decision_tree_label(
    feature_values: list[int],
    classifiers: Mapping[str, object],
) -> str | None:
    """Return the decision-tree prediction from a trained classifier set."""
    classifier = classifiers.get("decision-tree")
    if classifier is None:
        raise ValueError("A trained decision-tree classifier is required")

    predicted_label = getattr(classifier, "predict")([feature_values])[0]
    return str(predicted_label)


def identify_transaction(
    transaction: Transaction,
    classifiers: Mapping[str, object],
    feature_names: list[str],
) -> str | None:
    """Analyze a transaction and identify its wallet with the decision tree.

    The transaction is first converted into the requested feature order. Each
    feature is passed to the trained decision-tree classifier, which predicts
    a wallet label.
    """
    report = analyze(transaction)
    ordered_features = feature_vector(report, feature_names)
    return _decision_tree_label(ordered_features, classifiers)


async def identify_transaction_changes(
    transaction: Transaction,
    provider: ElectrsProvider,
    classifiers: Mapping[str, object],
    feature_names: list[str],
) -> ChangeDetectionResult:
    """Identify a transaction and check which spent outputs return to its wallet.

    For every spent output, the corresponding spending transaction is fetched
    from the provider and classified with the same decision-tree model as the
    original transaction. An output is considered change when both
    transactions have the same identified wallet label. This is a heuristic;
    transactions without a prediction cannot be assessed.
    """
    wallet_label = identify_transaction(transaction, classifiers, feature_names)
    spending_txids = await provider.get_output_spends(transaction.txid)
    if len(spending_txids) != len(transaction.outputs):
        raise ValueError(
            f"Expected spend information for {len(transaction.outputs)} outputs "
            f"of transaction {transaction.txid}, received {len(spending_txids)}"
        )

    unique_spending_txids = list(
        dict.fromkeys(txid for txid in spending_txids if txid is not None)
    )
    spending_transactions = await asyncio.gather(
        *(provider.get_transaction(txid) for txid in unique_spending_txids)
    )
    spending_labels = {
        txid: identify_transaction(spending_transaction, classifiers, feature_names)
        for txid, spending_transaction in zip(
            unique_spending_txids, spending_transactions
        )
    }

    output_spends = []
    for output_index, spending_txid in enumerate(spending_txids):
        spending_label = (
            spending_labels[spending_txid] if spending_txid is not None else None
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
                is_change=is_change,
            )
        )

    return ChangeDetectionResult(
        transaction_id=transaction.txid,
        wallet_label=wallet_label,
        output_spends=output_spends,
    )


def identify_fingerprint_features(
    feature_values: list[int],
    classifiers: Mapping[str, object],
) -> str | None:
    """Identify a wallet from an ordered feature vector using the decision tree."""
    return _decision_tree_label(feature_values, classifiers)
