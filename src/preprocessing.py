"""
preprocessing.py — Data preprocessing pipeline for gene expression data.

Steps:
  1. Drop zero-variance genes
  2. StandardScaler normalization
  3. Label encoding
  4. Stratified train/test split
  5. Optional PCA dimensionality reduction
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


def remove_zero_variance(X: pd.DataFrame) -> pd.DataFrame:
    """Remove features (genes) with zero variance across all samples."""
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X)
    mask = selector.get_support()
    X_filtered = X.loc[:, mask]
    n_removed = X.shape[1] - X_filtered.shape[1]
    print(f"Removed {n_removed} zero-variance genes. "
          f"Remaining features: {X_filtered.shape[1]}")
    return X_filtered


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode string cancer-type labels to integers."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return y_encoded, le


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train set: {X_train.shape[0]} samples | "
          f"Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize features to zero mean, unit variance (fit on train only)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def apply_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    variance_ratio: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Reduce dimensionality via PCA, retaining `variance_ratio` of variance.
    """
    pca = PCA(n_components=variance_ratio, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA: reduced to {pca.n_components_} components "
          f"(explaining {variance_ratio*100:.0f}% variance)")
    return X_train_pca, X_test_pca, pca


def preprocess_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    use_pca: bool = True,
    pca_variance: float = 0.95,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Run the full preprocessing pipeline.

    Returns a dict with all preprocessed data and fitted transformers.
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Remove zero-variance genes
    X_filtered = remove_zero_variance(X)

    # 2. Encode labels
    y_encoded, label_encoder = encode_labels(y)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = split_data(
        X_filtered, y_encoded, test_size=test_size, random_state=random_state
    )

    # 4. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    result = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "feature_names": X_filtered.columns.tolist(),
    }

    # 5. Optional PCA
    if use_pca:
        X_train_pca, X_test_pca, pca = apply_pca(
            X_train_scaled, X_test_scaled, variance_ratio=pca_variance
        )
        result["X_train_pca"] = X_train_pca
        result["X_test_pca"] = X_test_pca
        result["pca"] = pca

    print("Preprocessing complete.\n")
    return result
