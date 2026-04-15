#!/usr/bin/env python3
"""
main.py — TCGA Cancer Subtype Classification Pipeline

End-to-end ML pipeline:
  1. Load TCGA Pan-Cancer RNA-Seq data (5 cancer types, 20k+ genes)
  2. Preprocess (filter, scale, encode, split, PCA)
  3. Train models (Random Forest, SVM, Logistic Regression)
  4. Evaluate on held-out test set
  5. Generate publication-quality visualizations

Usage:
    python main.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, get_dataset_info
from src.preprocessing import preprocess_pipeline
from src.models import train_all_models
from src.evaluate import evaluate_all_models
from src.visualize import generate_all_plots


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   TCGA Cancer Subtype Classification Pipeline           ║")
    print("║   Dataset: Pan-Cancer RNA-Seq (UCI ML Repository)       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Step 1: Load Data ──────────────────────────────────────────
    print("\n[Step 1/5] Loading dataset...")
    X, y = load_data()
    info = get_dataset_info(X, y)
    print(f"  Samples: {info['n_samples']}")
    print(f"  Features (genes): {info['n_features']}")
    print(f"  Classes: {info['classes']}")
    print(f"  Distribution: {info['class_distribution']}")

    # ── Step 2: Preprocess ─────────────────────────────────────────
    print("\n[Step 2/5] Preprocessing data...")
    preprocessed = preprocess_pipeline(
        X, y,
        use_pca=True,
        pca_variance=0.95,
        test_size=0.2,
        random_state=42,
    )

    # ── Step 3: Train Models ───────────────────────────────────────
    # Use PCA-reduced features for faster training
    print("\n[Step 3/5] Training models...")
    X_train = preprocessed["X_train_pca"]
    X_test = preprocessed["X_test_pca"]
    y_train = preprocessed["y_train"]
    y_test = preprocessed["y_test"]

    model_results = train_all_models(X_train, y_train)

    # ── Step 4: Evaluate ───────────────────────────────────────────
    print("\n[Step 4/5] Evaluating models on test set...")
    eval_results = evaluate_all_models(
        model_results,
        X_test,
        y_test,
        preprocessed["label_encoder"],
    )

    # ── Step 5: Visualize ──────────────────────────────────────────
    # Use full (non-PCA) features for feature importance & PCA viz
    print("\n[Step 5/5] Generating visualizations...")

    # For feature importance, retrain RF on full features
    from src.models import train_random_forest
    rf_full = train_random_forest(preprocessed["X_train"], preprocessed["y_train"])

    # Merge full-feature RF into model_results for visualization
    model_results_for_viz = [
        rf_full if "random forest" in mr["name"].lower() else mr
        for mr in model_results
    ]

    plot_paths = generate_all_plots(
        preprocessed,
        model_results_for_viz,
        eval_results,
    )

    # ── Summary ────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   PIPELINE COMPLETE                                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\nGenerated outputs:")
    for key, val in plot_paths.items():
        if isinstance(val, list):
            for v in val:
                print(f"  📊 {v}")
        elif val:
            print(f"  📊 {val}")

    # Find best model
    best = max(eval_results, key=lambda x: x["f1_macro"])
    print(f"\n🏆 Best model: {best['name']} "
          f"(F1={best['f1_macro']:.4f}, Accuracy={best['accuracy']:.4f})")


if __name__ == "__main__":
    main()
