"""
data_loader.py — Fetch and cache the TCGA Pan-Cancer RNA-Seq dataset.

Dataset: UCI ML Repository ID 401
  - 801 samples × 20,531 gene features
  - 5 cancer types: BRCA, KIRC, COAD, LUAD, PRAD

Download URL:
  https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip
"""

import io
import os
import tarfile
import zipfile
import urllib.request

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FEATURES_PATH = os.path.join(DATA_DIR, "data.csv")
LABELS_PATH = os.path.join(DATA_DIR, "labels.csv")
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/401/"
    "gene+expression+cancer+rna+seq.zip"
)


def _download_and_extract():
    """Download the TCGA dataset from UCI and extract CSVs to DATA_DIR."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading TCGA dataset from UCI archive...")

    # Download zip
    with urllib.request.urlopen(DATASET_URL) as response:
        zip_bytes = response.read()

    # Extract the tar.gz from inside the zip
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        tar_name = [n for n in zf.namelist() if n.endswith(".tar.gz")][0]
        tar_bytes = zf.read(tar_name)

    # Extract CSVs from the tar.gz
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            if member.name.endswith("data.csv"):
                f = tf.extractfile(member)
                with open(FEATURES_PATH, "wb") as out:
                    out.write(f.read())
            elif member.name.endswith("labels.csv"):
                f = tf.extractfile(member)
                with open(LABELS_PATH, "wb") as out:
                    out.write(f.read())

    print(f"Data saved to {DATA_DIR}/")


def load_data(use_cache: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the TCGA gene expression dataset.

    Downloads from UCI ML Repository on first call and caches locally.
    Subsequent calls read from the cached CSV files.

    Returns
    -------
    X : pd.DataFrame
        Gene expression features (801 × 20531).
    y : pd.Series
        Cancer type labels.
    """
    if not (use_cache and os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH)):
        _download_and_extract()
    else:
        print("Loading cached data from disk...")

    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(LABELS_PATH)

    # The labels CSV has columns like "Unnamed: 0" (index) and "Class"
    # Normalize: extract just the label column
    if "Class" in y.columns:
        y = y["Class"]
    elif y.shape[1] == 2:
        y = y.iloc[:, 1]
    else:
        y = y.squeeze()

    # Drop any unnamed index columns from X
    unnamed_cols = [c for c in X.columns if "unnamed" in c.lower()]
    if unnamed_cols:
        X = X.drop(columns=unnamed_cols)

    return X, y


def get_dataset_info(X: pd.DataFrame, y: pd.Series) -> dict:
    """Return a summary dict describing the dataset."""
    return {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "classes": sorted(y.unique().tolist()),
        "n_classes": y.nunique(),
        "class_distribution": y.value_counts().to_dict(),
    }
