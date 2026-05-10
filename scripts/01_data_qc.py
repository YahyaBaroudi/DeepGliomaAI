import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = "/mnt/d/projects/DeepGliomaAI"
EXPR_PATH = f"{BASE_DIR}/data/expr_mat_log2.csv"
CLIN_PATH = f"{BASE_DIR}/data/clinical_matched.csv"
OUTDIR = f"{BASE_DIR}/results/01_data_qc"
FIGDIR = f"{BASE_DIR}/figures/01_data_qc"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

print("Loading data...")
expr = pd.read_csv(EXPR_PATH, index_col=0)
clin = pd.read_csv(CLIN_PATH)

print("Expression shape:", expr.shape)
print("Clinical shape:", clin.shape)
print("Clinical columns:")
print(clin.columns.tolist())

# expr: genes x samples -> samples x genes
expr_t = expr.T.copy()
# Convert sample IDs to patient IDs (TCGA format)
expr_t.index = expr_t.index.astype(str)

# Keep only first 12 characters (TCGA patient ID)
expr_t.index = expr_t.index.str.slice(0, 12)
# Detect patient/sample column
possible_id_cols = ["PATIENT_ID", "patient_id", "sample", "Sample", "barcode", "ID"]
id_col = None
for col in possible_id_cols:
    if col in clin.columns:
        id_col = col
        break

if id_col is None:
    raise ValueError("No patient/sample ID column found in clinical file.")

clin[id_col] = clin[id_col].astype(str)

clin[id_col] = clin[id_col].str.slice(0, 12)

common = sorted(set(expr_t.index).intersection(set(clin[id_col])))
print("Common samples:", len(common))

if len(common) == 0:
    raise ValueError("No common samples between expression and clinical data.")

X = expr_t.loc[common].copy()
clin2 = clin.set_index(id_col).loc[common].copy()

# Save matched data
X.to_csv(f"{OUTDIR}/X_matched_samples_by_genes.csv")
clin2.to_csv(f"{OUTDIR}/clinical_matched_for_project.csv")

# Missing values
qc = pd.DataFrame({
    "n_samples": [X.shape[0]],
    "n_genes": [X.shape[1]],
    "missing_values": [int(X.isna().sum().sum())],
    "duplicated_samples": [int(X.index.duplicated().sum())],
    "duplicated_genes": [int(pd.Index(X.columns).duplicated().sum())]
})
qc.to_csv(f"{OUTDIR}/qc_summary.csv", index=False)
print(qc)

# Gene variance
gene_var = X.var(axis=0).sort_values(ascending=False)
gene_var.to_csv(f"{OUTDIR}/gene_variance.csv")

plt.figure(figsize=(8, 5))
plt.hist(gene_var.values, bins=80)
plt.xlabel("Gene variance")
plt.ylabel("Number of genes")
plt.title("Distribution of gene variance")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/gene_variance_distribution.png", dpi=300)
plt.close()

# Keep top variable genes for PCA
TOP_N = min(5000, X.shape[1])
top_genes = gene_var.head(TOP_N).index.tolist()
X_top = X[top_genes]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top)

pca = PCA(n_components=10, random_state=42)
pcs = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    pcs[:, :5],
    index=X.index,
    columns=[f"PC{i}" for i in range(1, 6)]
)
pca_df = pca_df.join(clin2, how="left")
pca_df.to_csv(f"{OUTDIR}/pca_top_variable_genes.csv")

explained = pd.DataFrame({
    "PC": [f"PC{i}" for i in range(1, 11)],
    "explained_variance_ratio": pca.explained_variance_ratio_
})
explained.to_csv(f"{OUTDIR}/pca_explained_variance.csv", index=False)

# PCA plot function
def plot_pca(color_col):
    if color_col not in pca_df.columns:
        return

    tmp = pca_df.dropna(subset=[color_col]).copy()
    if tmp.empty:
        return

    plt.figure(figsize=(7, 6))

    if tmp[color_col].dtype == "object" or tmp[color_col].nunique() <= 10:
        groups = tmp[color_col].astype(str).unique()
        for g in groups:
            sub = tmp[tmp[color_col].astype(str) == g]
            plt.scatter(sub["PC1"], sub["PC2"], label=g, alpha=0.75)
        plt.legend(fontsize=8)
    else:
        sc = plt.scatter(tmp["PC1"], tmp["PC2"], c=tmp[color_col], alpha=0.75)
        plt.colorbar(sc, label=color_col)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(f"PCA colored by {color_col}")
    plt.tight_layout()
    safe = str(color_col).replace("/", "_").replace(" ", "_")
    plt.savefig(f"{FIGDIR}/PCA_by_{safe}.png", dpi=300)
    plt.close()

for col in ["SUBTYPE", "tumor_type", "CANCER_TYPE_ACRONYM", "grade_group", "OS_event", "OS_STATUS", "AGE", "SEX"]:
    plot_pca(col)

print("Done.")
print("Results saved in:", OUTDIR)
print("Figures saved in:", FIGDIR)
