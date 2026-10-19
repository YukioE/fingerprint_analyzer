"""Identify a wallet from a transaction's fingerprint features."""

import asyncio
from collections import Counter
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


def _majority_label(
    feature_values: list[int],
    classifiers: Mapping[str, object],
) -> str | None:
    """Return a label only when more than half of the classifiers agree."""
    if not classifiers:
        raise ValueError("At least one trained classifier is required")

    votes: Counter[str] = Counter()
    for classifier in classifiers.values():
        # Classifiers expect a batch of samples, so wrap the single vector.
        predicted_label = getattr(classifier, "predict")([feature_values])[0]
        votes[str(predicted_label)] += 1

    majority_label, vote_count = votes.most_common(1)[0]
    required_votes = len(classifiers) / 2
    if vote_count > required_votes:
        return majority_label

    return None


def identify_transaction(
    transaction: Transaction,
    classifiers: Mapping[str, object],
    feature_names: list[str],
) -> str | None:
    """Analyze a transaction and identify its wallet using a strict majority.

    The transaction is first converted into the requested feature order. Each
    classifier then predicts a wallet label, and a label is returned only if
    more than half of the classifiers agree.
    """
    report = analyze(transaction)
    ordered_features = feature_vector(report, feature_names)
    return _majority_label(ordered_features, classifiers)


async def identify_transaction_changes(
    transaction: Transaction,
    provider: ElectrsProvider,
    classifiers: Mapping[str, object],
    feature_names: list[str],
) -> ChangeDetectionResult:
    """Identify a transaction and check which spent outputs return to its wallet.

    For every spent output, the corresponding spending transaction is fetched
    from the provider and classified with the same majority-vote method as the
    original transaction. An output is considered change when both
    transactions have the same identified wallet label. This is a heuristic;
    transactions with no majority label cannot be assessed.
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
    """Identify a wallet from an already ordered fingerprint feature vector."""
    return _majority_label(feature_values, classifiers)
