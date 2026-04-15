"""
models.py — Train and tune classifiers on TCGA gene expression data.

Models:
  - Random Forest
  - Support Vector Machine (RBF kernel)
  - Logistic Regression

Each model is tuned via GridSearchCV with 5-fold stratified CV.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def _train_model(
    name: str,
    estimator,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> dict:
    """Train a single model with GridSearchCV and return results."""
    print(f"\n  Training {name}...")
    start = time.time()

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")
    print(f"  Training time: {elapsed:.1f}s")

    return {
        "name": name,
        "best_estimator": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
        "training_time": elapsed,
    }


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Train a Random Forest classifier with hyperparameter tuning."""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    return _train_model(
        name="Random Forest",
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
    )


def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Train an SVM classifier (RBF kernel) with hyperparameter tuning."""
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
    }
    return _train_model(
        name="SVM (RBF)",
        estimator=SVC(kernel="rbf", random_state=42),
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
    )


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Train a Logistic Regression classifier with hyperparameter tuning."""
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "max_iter": [1000],
    }
    return _train_model(
        name="Logistic Regression",
        estimator=LogisticRegression(random_state=42, solver="lbfgs", multi_class="multinomial"),
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
    )


def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> list[dict]:
    """
    Train all three classifiers and return a list of result dicts.
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING (GridSearchCV, 5-fold stratified CV)")
    print("=" * 60)

    results = [
        train_random_forest(X_train, y_train),
        train_svm(X_train, y_train),
        train_logistic_regression(X_train, y_train),
    ]

    print("\n" + "-" * 40)
    print("Cross-Validation Summary:")
    for r in results:
        print(f"  {r['name']:25s}  CV Accuracy: {r['best_cv_score']:.4f}")
    print("-" * 40 + "\n")

    return results
