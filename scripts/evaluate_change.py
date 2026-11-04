"""Evaluate fingerprint-analyzer change detection against labeled outputs."""

import argparse
import asyncio
import csv
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from fingerprint_analyzer.identification import identify_transaction_changes
from fingerprint_analyzer.ml import first_dataset, train_classifier
from fingerprint_analyzer.provider import ElectrsProvider, ProviderError

MEMPOOL_API_URL = "http://100.126.92.24:3000"
MAX_CONCURRENT_REQUESTS = 1024
TICKER_INTERVAL = 1_00
EVALUATION_INTERVAL = 10_000


@dataclass(frozen=True)
class GroundTruth:
    txid: str
    change_output_index: int


def iter_ground_truth(path: Path) -> Iterator[GroundTruth]:
    """Stream every labeled transaction from the ground-truth CSV."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"txhash", "change_output_idx"} <= set(
            reader.fieldnames
        ):
            raise ValueError(
                "Ground truth CSV must contain txhash and change_output_idx"
            )

        for row_number, row in enumerate(reader, start=2):
            txid = (row.get("txhash") or "").strip()
            try:
                output_index = int(row["change_output_idx"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid change_output_idx in CSV row {row_number}"
                ) from exc
            if not txid or output_index < 0:
                raise ValueError(f"Invalid ground truth in CSV row {row_number}")
            yield GroundTruth(txid, output_index)


async def evaluate_one(
    ground_truth: GroundTruth,
    provider: ElectrsProvider,
    classifiers: dict[str, object],
    feature_names: list[str],
    request_semaphore: asyncio.Semaphore,
) -> tuple[GroundTruth, list[bool] | None, str | None]:
    """Fetch and assess one transaction; each output's prediction is a bool."""
    try:
        async with request_semaphore:
            transaction = await provider.get_transaction(ground_truth.txid)
        if ground_truth.change_output_index >= len(transaction.outputs):
            return (
                ground_truth,
                None,
                f"change output index {ground_truth.change_output_index} "
                f"outside {len(transaction.outputs)} transaction outputs",
            )

        result = await identify_transaction_changes(
            transaction,
            provider,
            classifiers,
            feature_names,
            request_semaphore=request_semaphore,
        )
        return (
            ground_truth,
            [output.is_change is True for output in result.output_spends],
            None,
        )
    except (ProviderError, ValueError) as exc:
        return ground_truth, None, str(exc)


def print_evaluation(
    completed: int,
    errors_count: int,
    correct: int,
    incorrect: int,
    abstained: int,
    elapsed: float,
) -> None:
    evaluated = completed - errors_count
    unique_guesses = correct + incorrect
    coverage = unique_guesses / evaluated if evaluated else 0.0
    throughput = evaluated / elapsed if elapsed else 0.0

    print(f"\nEvaluation after {completed:,} transactions")
    print(f"Transactions evaluated: {evaluated}/{completed}")
    print(f"Transactions skipped: {errors_count}")
    print(
        f"Unique predictions: correct={correct}, "
        f"incorrect={incorrect}, abstained={abstained}"
    )
    print(
        f"Coverage: {coverage:.4f} "
        f"({unique_guesses}/{evaluated} evaluated transactions)"
    )
    print(
        f"True positive rate (TPR): {correct / unique_guesses:.4f}"
        if unique_guesses
        else "True positive rate (TPR): n/a (no unique predictions)"
    )
    print(
        f"False positive rate (FPR): {incorrect / unique_guesses:.4f}"
        if unique_guesses
        else "False positive rate (FPR): n/a (no unique predictions)"
    )
    print(f"Elapsed: {elapsed:.1f}s ({throughput:.2f} transactions/s)")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path, help="ground-truth CSV dataset")
    args = parser.parse_args()

    dataset_path = first_dataset()
    decision_tree, feature_names = train_classifier(dataset_path, "decision-tree")
    classifiers = {"decision-tree": decision_tree}
    provider = ElectrsProvider(base_url=MEMPOOL_API_URL, cache_responses=True)
    request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    print(f"Ground truth: {args.ground_truth}")
    print("Rows requested: all")
    print(f"Fingerprint dataset: {dataset_path}")
    print(
        f"API: {MEMPOOL_API_URL}; "
        f"max concurrent requests: {MAX_CONCURRENT_REQUESTS}"
    )
    print("Unspent or unclassified outputs are treated as negative predictions.")

    started = time.perf_counter()
    ground_truth_rows = iter_ground_truth(args.ground_truth)
    completed_count = errors_count = 0
    correct = incorrect = abstained = 0
    errors = []
    while True:
        batch = list(itertools.islice(ground_truth_rows, MAX_CONCURRENT_REQUESTS))
        if not batch:
            break
        tasks = [
            asyncio.create_task(
                evaluate_one(
                    row, provider, classifiers, feature_names, request_semaphore
                )
            )
            for row in batch
        ]
        for completed in asyncio.as_completed(tasks):
            ground_truth, predictions, error = await completed
            completed_count += 1
            if error is not None:
                errors_count += 1
                if len(errors) < 20:
                    errors.append((ground_truth.txid, error))
            else:
                assert predictions is not None
                candidates = [
                    index
                    for index, predicted_change in enumerate(predictions)
                    if predicted_change
                ]
                if len(candidates) != 1:
                    abstained += 1
                elif candidates[0] == ground_truth.change_output_index:
                    correct += 1
                else:
                    incorrect += 1

            if completed_count % TICKER_INTERVAL == 0:
                print(
                    f"\rCompleted {completed_count:,} transactions",
                    end="",
                    flush=True,
                )
            if completed_count % EVALUATION_INTERVAL == 0:
                print_evaluation(
                    completed_count,
                    errors_count,
                    correct,
                    incorrect,
                    abstained,
                    time.perf_counter() - started,
                )

    if completed_count == 0:
        raise ValueError(f"No ground-truth rows found in {args.ground_truth}")

    print()
    elapsed = time.perf_counter() - started
    if completed_count % EVALUATION_INTERVAL:
        print_evaluation(
            completed_count,
            errors_count,
            correct,
            incorrect,
            abstained,
            elapsed,
        )

    for txid, error in errors:
        print(f"Skipped {txid}: {error}", file=sys.stderr)
    if errors_count > len(errors):
        print(
            f"... and {errors_count - len(errors)} more skipped transactions",
            file=sys.stderr,
        )


if __name__ == "__main__":
    asyncio.run(main())
