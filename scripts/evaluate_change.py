"""Evaluate fingerprint-analyzer change detection against labeled outputs."""

import argparse
import asyncio
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from fingerprint_analyzer.fingerprints.helpers import input_type, output_type
from fingerprint_analyzer.identification import identify_transaction_changes
from fingerprint_analyzer.ml import first_dataset, train_classifier
from fingerprint_analyzer.model import Transaction
from fingerprint_analyzer.provider import ElectrsProvider, ProviderError

MEMPOOL_API_URL = "http://100.126.92.24:3000"
MAX_CONCURRENT_REQUESTS = 128
EVALUATION_INTERVAL = 1_0
# Treat outputs divisible by 1,000 satoshis as round payment amounts.
ROUND_AMOUNT_INCREMENT = 1_000

INPUT_TYPE_TO_OUTPUT_FAMILY = {
    "p2pk": "base58",
    "p2pkh": "base58",
    "p2sh_all": "base58",
    "p2sh_multisig": "base58",
    "p2sh_p2wpkh": "base58",
    "p2sh_p2wsh": "base58",
    "p2sh_unknown": "base58",
    "p2tr": "bech32m",
    "p2wpkh": "bech32",
    "p2wsh_all": "bech32",
    "p2wsh_multisig": "bech32",
    "p2wsh_unknown": "bech32",
}

HEURISTIC_NAMES = (
    "Input/output type match",
    "Non-round amount",
    "Optimal change (UIH)",
    "Universal heuristics (majority)",
    "Wallet fingerprinting",
)


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


def universal_predictions(transaction: Transaction) -> dict[str, list[bool]]:
    """Return one candidate-output vector per universal change heuristic."""
    input_families = {
        INPUT_TYPE_TO_OUTPUT_FAMILY[type_name]
        for txin in transaction.inputs
        if not txin.is_coinbase
        if (type_name := input_type(txin)) in INPUT_TYPE_TO_OUTPUT_FAMILY
    }
    output_families = [output_type(output) for output in transaction.outputs]

    type_match = [
        family in input_families and family not in {"op_return", "unknown"}
        for family in output_families
    ]
    non_round_amount = [
        output.value % ROUND_AMOUNT_INCREMENT != 0
        and family != "op_return"
        for output, family in zip(transaction.outputs, output_families)
    ]

    input_values = [
        txin.prevout.value
        for txin in transaction.inputs
        if not txin.is_coinbase and txin.prevout is not None
    ]
    non_coinbase_inputs = [txin for txin in transaction.inputs if not txin.is_coinbase]
    has_complete_input_values = (
        len(non_coinbase_inputs) >= 2
        and len(input_values) == len(non_coinbase_inputs)
    )
    optimal_change = [False] * len(transaction.outputs)
    if has_complete_input_values:
        smallest_input = min(input_values)
        if transaction.fee is not None:
            optimal_change = [
                output.value + transaction.fee < smallest_input
                and family != "op_return"
                for output, family in zip(transaction.outputs, output_families)
            ]

    heuristics = {
        HEURISTIC_NAMES[0]: type_match,
        HEURISTIC_NAMES[1]: non_round_amount,
        HEURISTIC_NAMES[2]: optimal_change,
    }
    heuristics[HEURISTIC_NAMES[3]] = [
        sum(predictions[index] for predictions in heuristics.values()) >= 2
        for index in range(len(transaction.outputs))
    ]
    return heuristics


def iter_ground_truth(path: Path) -> Iterator[GroundTruth]:
    """Stream every labeled transaction from the ground-truth CSV."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if header is None or len(header) != 2:
            raise ValueError("Ground truth CSV must contain exactly 2 columns")

        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(
                    f"Expected 2 columns in ground truth CSV row {row_number}"
                )
            txid = row[0].strip()
            try:
                output_index = int(row[1])
            except ValueError as exc:
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
) -> tuple[
    GroundTruth,
    list[bool] | None,
    dict[str, list[bool]] | None,
    str | None,
]:
    """Fetch and assess one transaction; each output's prediction is a bool."""
    try:
        async with request_semaphore:
            transaction = await provider.get_transaction(ground_truth.txid)
        if ground_truth.change_output_index >= len(transaction.outputs):
            return (
                ground_truth,
                None,
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
            universal_predictions(transaction),
            None,
        )
    except (ProviderError, ValueError) as exc:
        return ground_truth, None, None, str(exc)


def print_evaluation(
    completed: int,
    errors_count: int,
    statistics: dict[str, dict[str, int]],
    *,
    overwrite: bool = False,
) -> None:
    evaluated = completed - errors_count
    lines = [
        f"Evaluation progress: {completed} completed, {errors_count} skipped",
        f"{'Detector':<36} {'Scored/evaluated':>18} {'Correct/scored':>17}",
    ]
    for name in HEURISTIC_NAMES:
        scored = statistics[name]["scored"]
        correct = statistics[name]["correct"]
        coverage = f"{scored / evaluated * 100:.2f}%" if evaluated else "n/a"
        accuracy = f"{correct / scored * 100:.2f}%" if scored else "n/a"
        lines.append(
            f"{name:<36} {coverage:>8} ({scored}/{evaluated}) "
            f"{accuracy:>7} ({correct}/{scored})"
        )
    report = "\n".join(lines)
    if overwrite and sys.stdout.isatty():
        sys.stdout.write(f"\033[{len(lines)}F\033[J{report}\n")
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
    statistics = {name: {"scored": 0, "correct": 0} for name in HEURISTIC_NAMES}
    errors = []
    has_progress_report = False
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
            ground_truth, predictions, heuristic_predictions, error = await completed
            completed_count += 1
            if error is not None:
                errors_count += 1
                if len(errors) < 20:
                    errors.append((ground_truth.txid, error))
            else:
                assert predictions is not None
                assert heuristic_predictions is not None
                all_predictions = {
                    **heuristic_predictions,
                    HEURISTIC_NAMES[4]: predictions,
                }
                for name, candidates in all_predictions.items():
                    scored, correct = score_prediction(
                        candidates, ground_truth.change_output_index
                    )
                    if scored:
                        statistics[name]["scored"] += 1
                        statistics[name]["correct"] += int(correct)

            if completed_count % EVALUATION_INTERVAL == 0:
                print_evaluation(
                    completed_count,
                    errors_count,
                    statistics,
                    overwrite=has_progress_report,
                )
                has_progress_report = True

    if completed_count == 0:
        raise ValueError(f"No ground-truth rows found in {args.ground_truth}")

    print_evaluation(
        completed_count,
        errors_count,
        statistics,
        overwrite=has_progress_report,
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
