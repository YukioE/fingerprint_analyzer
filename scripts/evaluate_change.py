"""Evaluate fingerprint-analyzer change detection against labeled outputs."""

import argparse
import asyncio
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from fingerprint_analyzer.identification import identify_transaction_changes
from fingerprint_analyzer.ml import first_dataset, train_classifier
from fingerprint_analyzer.provider import ElectrsProvider, ProviderError

MEMPOOL_API_URL = "http://100.126.92.24:3000"
MAX_CONCURRENT_REQUESTS = 128
EVALUATION_INTERVAL = 1_000


@dataclass(frozen=True)
class GroundTruth:
    txid: str
    change_output_index: int


def score_prediction(
    predictions: list[bool], change_output_index: int
) -> tuple[bool, bool]:
    """Return whether a transaction was scored and whether it was an exact match."""
    candidate_count = sum(predictions)
    if candidate_count == 0 or candidate_count == len(predictions):
        return False, False
    return True, candidate_count == 1 and predictions[change_output_index]


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
    scored_transactions: int,
    correct_transactions: int,
    *,
    final: bool = False,
) -> None:
    is_terminal = sys.stdout.isatty()
    if not is_terminal and not final:
        return

    evaluated = completed - errors_count
    scored_percentage = (
        f"{scored_transactions / evaluated * 100:.2f}%" if evaluated else "n/a"
    )
    correct_percentage = (
        f"{correct_transactions / scored_transactions * 100:.2f}%"
        if scored_transactions
        else "n/a"
    )
    report = (
        f"Scored/evaluated: {scored_percentage} "
        f"({scored_transactions}/{evaluated}) | "
        f"Correct/scored: {correct_percentage} "
        f"({correct_transactions}/{scored_transactions})"
    )
    if is_terminal:
        sys.stdout.write(f"\r\033[2K{report}")
        if final:
            sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        print(report, flush=True)


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
    print(f"Fingerprint dataset: {dataset_path}")
    print(
        f"API: {MEMPOOL_API_URL}; "
        f"max concurrent requests: {MAX_CONCURRENT_REQUESTS}"
    )
    ground_truth_rows = iter_ground_truth(args.ground_truth)
    completed_count = errors_count = 0
    scored_transactions = correct_transactions = 0
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
                scored, correct = score_prediction(
                    predictions, ground_truth.change_output_index
                )
                if scored:
                    scored_transactions += 1
                    correct_transactions += int(correct)

            if completed_count % EVALUATION_INTERVAL == 0:
                print_evaluation(
                    completed_count,
                    errors_count,
                    scored_transactions,
                    correct_transactions,
                )

    if completed_count == 0:
        raise ValueError(f"No ground-truth rows found in {args.ground_truth}")

    print_evaluation(
        completed_count,
        errors_count,
        scored_transactions,
        correct_transactions,
        final=True,
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
