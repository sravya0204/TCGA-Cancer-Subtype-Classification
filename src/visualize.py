"""
visualize.py — Generate publication-quality plots for TCGA classification results.

Outputs:
  - PCA scatter plot (2D projection by cancer type)
  - Confusion matrix heatmap (per model)
  - Feature importance bar chart (top 20 genes, Random Forest)
  - Model comparison bar chart
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder,
    title: str = "PCA Projection of TCGA Samples by Cancer Type",
) -> str:
    """
    2D PCA scatter plot of all samples, colored by cancer type.
    Uses the FULL scaled dataset (train + test combined) for visualization.
    """
    _ensure_output_dir()

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    target_names = label_encoder.classes_
    palette = sns.color_palette("husl", len(target_names))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(target_names):
        mask = y == i
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=name,
            color=palette[i],
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            s=60,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title="Cancer Type", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "pca_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: list[str],
    model_name: str,
) -> str:
    """Confusion matrix heatmap for a single model."""
    _ensure_output_dir()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_all_confusion_matrices(eval_results: list[dict]) -> list[str]:
    """Plot confusion matrices for all evaluated models."""
    paths = []
    for er in eval_results:
        p = plot_confusion_matrix(
            cm=er["confusion_matrix"],
            target_names=er["target_names"],
            model_name=er["name"],
        )
        paths.append(p)
    return paths


def plot_feature_importance(
    model_results: list[dict],
    feature_names: list[str],
    top_n: int = 20,
) -> str:
    """
    Bar chart of top N most important genes from the Random Forest model.
    """
    _ensure_output_dir()

    # Find the Random Forest result
    rf_result = None
    for mr in model_results:
        if "random forest" in mr["name"].lower():
            rf_result = mr
            break

    if rf_result is None:
        print("No Random Forest model found — skipping feature importance plot.")
        return ""

    importances = rf_result["best_estimator"].feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", top_n)
    ax.barh(range(top_n), top_importances[::-1], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title(
        f"Top {top_n} Most Important Genes (Random Forest)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_model_comparison(eval_results: list[dict]) -> str:
    """Grouped bar chart comparing accuracy, precision, recall, F1 across models."""
    _ensure_output_dir()

    models = [er["name"] for er in eval_results]
    metrics = {
        "Accuracy": [er["accuracy"] for er in eval_results],
        "Precision": [er["precision_macro"] for er in eval_results],
        "Recall": [er["recall_macro"] for er in eval_results],
        "F1 Score": [er["f1_macro"] for er in eval_results],
    }

    df = pd.DataFrame(metrics, index=models)

    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", width=0.7)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison (Test Set)", fontsize=14, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def generate_all_plots(
    preprocessed: dict,
    model_results: list[dict],
    eval_results: list[dict],
) -> dict:
    """Generate all visualizations and return a dict of file paths."""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Combine train + test for PCA visualization
    X_all = np.vstack([preprocessed["X_train"], preprocessed["X_test"]])
    y_all = np.concatenate([preprocessed["y_train"], preprocessed["y_test"]])

    paths = {
        "pca_scatter": plot_pca_scatter(
            X_all,
            y_all,
            preprocessed["label_encoder"],
        ),
        "confusion_matrices": plot_all_confusion_matrices(eval_results),
        "feature_importance": plot_feature_importance(
            model_results,
            preprocessed["feature_names"],
        ),
        "model_comparison": plot_model_comparison(eval_results),
    }

    print(f"\nAll plots saved to {OUTPUT_DIR}/\n")
    return paths
