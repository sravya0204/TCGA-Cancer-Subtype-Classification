"""
evaluate.py — Evaluate trained models on the test set.

Produces per-class and macro-averaged metrics,
classification reports, and a comparison summary table.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
) -> dict:
    """
    Evaluate a single model on the test set.

    Returns a dict containing predictions, metrics, confusion matrix,
    and the full classification report.
    """
    y_pred = model.predict(X_test)
    target_names = label_encoder.classes_.tolist()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    return {
        "name": model_name,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "target_names": target_names,
    }


def evaluate_all_models(
    model_results: list[dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
) -> list[dict]:
    """
    Evaluate all trained models and print results.

    Parameters
    ----------
    model_results : list[dict]
        Output from models.train_all_models().
    """
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    eval_results = []
    for mr in model_results:
        er = evaluate_model(
            model=mr["best_estimator"],
            model_name=mr["name"],
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder,
        )
        eval_results.append(er)

        print(f"\n{'─' * 40}")
        print(f"Model: {er['name']}")
        print(f"{'─' * 40}")
        print(er["classification_report"])

    # Summary table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 60)
    summary = pd.DataFrame(
        [
            {
                "Model": er["name"],
                "Accuracy": f"{er['accuracy']:.4f}",
                "Precision": f"{er['precision_macro']:.4f}",
                "Recall": f"{er['recall_macro']:.4f}",
                "F1 (macro)": f"{er['f1_macro']:.4f}",
            }
            for er in eval_results
        ]
    )
    print(summary.to_string(index=False))
    print("=" * 60 + "\n")

    return eval_results
