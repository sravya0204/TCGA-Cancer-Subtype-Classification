# TCGA Cancer Subtype Classification

A machine learning pipeline for classifying cancer subtypes — **Breast (BRCA)**, **Kidney (KIRC)**, **Colon (COAD)**, **Lung (LUAD)**, and **Prostate (PRAD)** carcinomas — from high-dimensional RNA-Seq gene expression profiles using The Cancer Genome Atlas (TCGA) Pan-Cancer dataset.

> 📓 **[View the full analysis notebook with inline results →](notebooks/TCGA_Analysis.ipynb)**

## Motivation

Cancer subtype classification from transcriptomic data is a core problem in computational genomics. Accurate classification enables precision medicine by guiding treatment decisions based on a tumor's molecular profile rather than tissue of origin alone. This project applies scalable ML techniques to a real-world high-dimensional biomedical dataset, addressing key challenges such as:

- **High dimensionality** — 20,531 gene expression features per sample (p >> n problem)
- **Class imbalance** — BRCA (300 samples) is ~4× more represented than COAD (78 samples)
- **Biological interpretability** — identifying which genes drive classification across cancer types

## Dataset

| Property | Value |
|---|---|
| Source | [TCGA Pan-Cancer RNA-Seq (UCI ML Repository)](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq) |
| Samples | 801 tumor samples |
| Features | 20,531 gene expression levels (RNA-Seq) |
| Classes | 5 cancer subtypes |

| Cancer Subtype | Full Name | Samples |
|---|---|---|
| **BRCA** | Breast Invasive Carcinoma | 300 |
| **KIRC** | Kidney Renal Clear Cell Carcinoma | 146 |
| **LUAD** | Lung Adenocarcinoma | 141 |
| **PRAD** | Prostate Adenocarcinoma | 136 |
| **COAD** | Colon Adenocarcinoma | 78 |

## Results

### Model Performance (Test Set, 161 samples)

| Model | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| Random Forest | 92.55% | 96.16% | 90.43% | 92.70% |
| SVM (RBF) | 98.14% | 98.61% | 97.32% | 97.94% |
| **Logistic Regression** | **99.38%** | **99.67%** | **99.29%** | **99.47%** |

🏆 **Best model: Logistic Regression** — 99.4% accuracy, 99.5% macro F1-score.

Logistic Regression outperformed ensemble and kernel-based methods, suggesting BRCA, KIRC, COAD, LUAD, and PRAD are **linearly separable** in the normalized gene expression space — consistent with these cancer types originating from biologically distinct tissues with markedly different transcriptomic profiles.

### Visualizations

#### PCA Projection — Cancer Subtypes in Gene Expression Space

<p align="center">
  <img src="outputs/pca_scatter.png" width="600" alt="PCA scatter plot">
</p>

The 2D PCA projection reveals clear cluster separation between all five subtypes, with **BRCA** and **KIRC** forming the most compact, well-separated clusters. This confirms that gene expression profiles carry strong discriminative signals across cancer types.

#### Model Comparison

<p align="center">
  <img src="outputs/model_comparison.png" width="600" alt="Model comparison">
</p>

#### Top 20 Most Important Genes (Random Forest)

<p align="center">
  <img src="outputs/feature_importance.png" width="700" alt="Feature importance">
</p>

Feature importance analysis from the Random Forest classifier identifies the genes with the highest discriminative power across cancer subtypes. These top-ranked genes could serve as candidate biomarkers for further biological validation.

#### Confusion Matrices

<p align="center">
  <img src="outputs/confusion_matrix_logistic_regression.png" width="400" alt="Logistic Regression CM">
  <img src="outputs/confusion_matrix_svm_rbf.png" width="400" alt="SVM CM">
</p>

## Methods

### Data Preprocessing — Handling High-Dimensional Genomic Data

Working with 20,531 gene expression features across only 801 samples (a classic **p >> n** problem) required careful preprocessing to avoid overfitting and enable tractable model training:

1. **Zero-variance gene filtering** — Removed 267 genes with no expression variation across samples, reducing noise without losing information
2. **StandardScaler normalization** — Standardized features to zero mean and unit variance, fit exclusively on the training set to prevent data leakage
3. **Class-aware stratified splitting** — 80/20 train/test split preserving the natural class imbalance (BRCA:COAD ratio of ~4:1), ensuring minority classes like COAD are adequately represented in both sets
4. **Label encoding** — Mapped cancer subtypes (BRCA, KIRC, COAD, LUAD, PRAD) to integer labels

### Dimensionality Reduction — PCA

Applied **Principal Component Analysis (PCA)** to address the high dimensionality:

- Reduced from **20,264 → 434 components** while retaining **95% of cumulative variance**
- ~47× dimensionality reduction, enabling faster model training and reducing overfitting risk
- The PCA 2D projection confirms that cancer subtypes occupy distinct regions in gene expression space

> **Note:** This project uses PCA for linear dimensionality reduction. A natural extension would be **Variational Autoencoders (VAEs)** for nonlinear manifold learning — capturing more complex gene expression patterns that linear methods may miss.

### Classification Models

All models tuned via **GridSearchCV** with **5-fold stratified cross-validation**:

| Model | Hyperparameters Tuned | Best CV Accuracy |
|---|---|---|
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split` | 91.41% |
| **SVM (RBF kernel)** | `C`, `gamma` | 98.91% |
| **Logistic Regression (L2)** | `C`, regularization strength | 99.84% |

## Reproduce

### Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt`

### Setup & Run

```bash
# Clone the repository
git clone https://github.com/sravya0204/TCGA-Cancer-Subtype-Classification.git
cd TCGA-Cancer-Subtype-Classification

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

The pipeline will:
1. Download the TCGA dataset (~70 MB, cached after first run)
2. Preprocess: filter zero-variance genes, normalize, PCA
3. Train and tune 3 classifiers with 5-fold CV
4. Evaluate on held-out test set and print classification reports
5. Generate all visualizations to `outputs/`

## Project Structure

```
TCGA-Cancer-Subtype-Classification/
├── main.py                 # End-to-end pipeline entry point
├── requirements.txt        # Python dependencies
├── .gitignore
├── src/
│   ├── data_loader.py      # Download & cache TCGA dataset from UCI
│   ├── preprocessing.py    # Zero-variance filter, scaling, PCA
│   ├── models.py           # RF, SVM, LR with GridSearchCV tuning
│   ├── evaluate.py         # Metrics, classification reports
│   └── visualize.py        # PCA, confusion matrices, feature importance
├── data/                   # Cached dataset (gitignored)
├── outputs/                # Generated figures
└── notebooks/
    └── TCGA_Analysis.ipynb # Full EDA & analysis with inline plots
```

## Tech Stack

- **Python** — Core language
- **Scikit-Learn** — ML models, preprocessing, cross-validation, evaluation
- **Pandas / NumPy** — Data manipulation and numerical computing
- **Matplotlib / Seaborn** — Publication-quality visualization
- **UCI ML Repository** — Data source (TCGA Pan-Cancer RNA-Seq)

## Future Work

- **Variational Autoencoder (VAE)** — Replace PCA with a VAE for nonlinear dimensionality reduction and synthetic data generation, following approaches similar to those used in genomic data augmentation research
- **Expand to all 33 TCGA cancer types** — Scale the pipeline to the full Pan-Cancer Atlas for a more comprehensive multi-class classification challenge
- **Gene set enrichment analysis (GSEA)** — Investigate whether the top-ranked features from the Random Forest map to known oncogenic pathways
- **Galaxy Platform integration** — Explore running the analysis pipeline through [Galaxy](https://galaxyproject.org/) for reproducibility, and programmatic access via the [BioBlend](https://bioblend.readthedocs.io/) Python API
- **Deep learning** — 1D CNNs or transformer architectures for end-to-end feature learning from raw expression profiles
- **Survival prediction** — Extend from classification to predicting patient outcomes using the same gene expression features

## References

- [TCGA Pan-Cancer Analysis Project](https://www.cell.com/pb-assets/consortium/pancanceratlas/pancani3/index.html)
- [UCI ML Repository: Gene Expression Cancer RNA-Seq](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq)
- Weinstein, J.N., et al. "The Cancer Genome Atlas Pan-Cancer analysis project." *Nature Genetics*, 2013.
- [Galaxy Project](https://galaxyproject.org/) — Open-source platform for reproducible computational biology

## License

This project is for educational and research purposes. The TCGA dataset is publicly available under open access.
