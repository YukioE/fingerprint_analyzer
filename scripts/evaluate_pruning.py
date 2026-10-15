#!/usr/bin/env python3
"""Plot decision-tree training and holdout accuracy across pruning alphas."""

import argparse
import sys
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier

from fingerprint_analyzer.ml import (
    _holdout_indices,
    first_dataset,
    load_feature_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=None,
        help="fingerprint feature CSV (defaults to the first CSV in the data directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="save the plot to this file instead of only displaying it",
    )
    args = parser.parse_args()

    try:
        dataset_path = args.dataset or first_dataset()
        features, labels, _ = load_feature_dataset(dataset_path)
        training_indices, testing_indices = _holdout_indices(labels)
        if not testing_indices:
            raise ValueError("Dataset has no holdout samples to evaluate")

        training_features = [features[index] for index in training_indices]
        training_labels = [labels[index] for index in training_indices]
        testing_features = [features[index] for index in testing_indices]
        testing_labels = [labels[index] for index in testing_indices]

        base_tree = DecisionTreeClassifier(random_state=22)
        pruning_path = base_tree.cost_complexity_pruning_path(
            training_features, training_labels
        )
        ccp_alphas = pruning_path.ccp_alphas
        classifiers = [
            DecisionTreeClassifier(random_state=22, ccp_alpha=alpha).fit(
                training_features, training_labels
            )
            for alpha in ccp_alphas
        ]
        train_scores = [
            classifier.score(training_features, training_labels)
            for classifier in classifiers
        ]
        test_scores = [
            classifier.score(testing_features, testing_labels)
            for classifier in classifiers
        ]

        try:
            import matplotlib
            from matplotlib.backends import BackendFilter, backend_registry
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "Plotting requires matplotlib; install with "
                "`python -m pip install -e '.[pruning-eval]'`"
            ) from error

        best_index = max(range(len(test_scores)), key=test_scores.__getitem__)
        print(
            f"Best holdout accuracy: {test_scores[best_index]:.1%} "
            f"at ccp_alpha={ccp_alphas[best_index]:.6g}"
        )

        figure, axis = plt.subplots()
        axis.set_xlabel("alpha")
        axis.set_ylabel("accuracy")
        axis.set_title("Accuracy vs alpha for training and testing sets")
        axis.plot(
            ccp_alphas,
            train_scores,
            marker="o",
            label="train",
            drawstyle="steps-post",
        )
        axis.plot(
            ccp_alphas,
            test_scores,
            marker="o",
            label="test",
            drawstyle="steps-post",
        )
        axis.legend()
        figure.tight_layout()
        interactive_backends = {
            backend.lower()
            for backend in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
        }
        if matplotlib.get_backend().lower() in interactive_backends:
            if args.output:
                figure.savefig(args.output)
            plt.show()
        else:
            output_path = args.output or Path("decision_tree_pruning.png")
            figure.savefig(output_path)
            print(f"Non-interactive matplotlib backend; plot saved to {output_path}")
        plt.close(figure)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
