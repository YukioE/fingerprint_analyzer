#!/usr/bin/env python3
"""Evaluate classifier behavior and feature importance on a deterministic holdout."""

import argparse
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from fingerprint_analyzer.ml import (
    _holdout_indices,
    first_dataset,
    load_feature_dataset,
)

PLOT_DPI = 300


def main() -> int:
    """Train evaluation models, create plots, and report the best holdout scores."""
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
        help="save plots beside this path (the pruning plot uses this exact filename)",
    )
    args = parser.parse_args()

    try:
        dataset_path = args.dataset or first_dataset()
        features, labels, feature_names = load_feature_dataset(dataset_path)
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
        decision_tree = DecisionTreeClassifier(random_state=22).fit(
            training_features, training_labels
        )
        random_forest = RandomForestClassifier(
            n_estimators=100, random_state=22
        ).fit(training_features, training_labels)

        max_k = min(25, len(training_features))
        knn_classifiers = [
            KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(
                training_features, training_labels
            )
            for k in range(1, max_k + 1)
        ]
        knn_train_scores = [
            classifier.score(training_features, training_labels)
            for classifier in knn_classifiers
        ]
        knn_test_scores = [
            classifier.score(testing_features, testing_labels)
            for classifier in knn_classifiers
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

        best_k_index = max(range(len(knn_test_scores)), key=knn_test_scores.__getitem__)
        print(
            f"Best KNN holdout accuracy: {knn_test_scores[best_k_index]:.1%} "
            f"at k={best_k_index + 1} (cosine distance)"
        )

        figures = []
        figure, axis = plt.subplots()
        axis.set_xlabel("alpha")
        axis.set_ylabel("accuracy")
        axis.set_title("Accuracy vs alpha for training and testing sets")
        axis.plot(
            ccp_alphas,
            train_scores,
            label="train",
            drawstyle="steps-post",
        )
        axis.plot(
            ccp_alphas,
            test_scores,
            label="test",
            drawstyle="steps-post",
        )
        axis.legend()
        figure.tight_layout()
        figures.append(("decision_tree_pruning", figure))

        figure, axis = plt.subplots(figsize=(20, 12))
        plot_tree(
            decision_tree,
            feature_names=feature_names,
            class_names=decision_tree.classes_,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=axis,
        )
        axis.set_title("Full decision tree structure")
        figure.tight_layout()
        figures.append(("decision_tree_structure", figure))

        tree_importances = decision_tree.feature_importances_
        forest_importances = random_forest.feature_importances_
        averaged_importances = (tree_importances + forest_importances) / 2
        importance_plots = (
            ("feature_importance_decision_tree", "Decision tree", tree_importances),
            ("feature_importance_random_forest", "Random forest", forest_importances),
            ("feature_importance_average", "Average importance", averaged_importances),
        )
        for filename, title, values in importance_plots:
            top_indices = sorted(
                range(len(feature_names)),
                key=values.__getitem__,
                reverse=True,
            )[:15]
            top_indices.reverse()
            figure, axis = plt.subplots(figsize=(10, 7))
            axis.barh(
                range(len(top_indices)),
                [values[index] for index in top_indices],
            )
            axis.set_yticks(range(len(top_indices)))
            axis.set_yticklabels([feature_names[index] for index in top_indices])
            axis.set_xlabel("feature importance")
            axis.set_title(f"Feature importance: {title}")
            figure.tight_layout()
            figures.append((filename, figure))

        figure, axis = plt.subplots()
        k_values = range(1, max_k + 1)
        axis.plot(k_values, knn_train_scores, label="train")
        axis.plot(k_values, knn_test_scores, label="holdout")
        axis.set_xticks(list(k_values))
        axis.set_xlabel("number of neighbors (k)")
        axis.set_ylabel("accuracy")
        axis.set_title("KNN accuracy by k (cosine distance)")
        axis.legend()
        figure.tight_layout()
        figures.append(("knn_effect_of_k", figure))

        interactive_backends = {
            backend.lower()
            for backend in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
        }
        if matplotlib.get_backend().lower() in interactive_backends:
            for name, plot_figure in figures:
                if args.output:
                    output_path = (
                        args.output
                        if name == "decision_tree_pruning"
                        else args.output.parent / f"{name}.png"
                    )
                    plot_figure.savefig(output_path, dpi=PLOT_DPI)
            plt.show()
        else:
            for name, plot_figure in figures:
                output_path = (
                    args.output
                    if args.output and name == "decision_tree_pruning"
                    else (
                        args.output.parent / f"{name}.png"
                        if args.output
                        else Path(f"{name}.png")
                    )
                )
                plot_figure.savefig(output_path, dpi=PLOT_DPI)
                print(f"Plot saved to {output_path}")
        for _, plot_figure in figures:
            plt.close(plot_figure)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
