# scripts/04_latent_biological_interpretation.py

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================

BASE_DIR = "/mnt/d/projects/DeepGliomaAI"

EXPR_PATH = os.path.join(BASE_DIR, "data", "expr_mat_log2.csv")
CLIN_PATH = os.path.join(BASE_DIR, "data", "clinical_matched.csv")

LATENT_PATH = os.path.join(BASE_DIR, "results", "02_vae", "vae_latent_space.csv")
SURV_PATH = os.path.join(BASE_DIR, "results", "03_vae_latent_survival", "latent_risk_scores.csv")

OUTDIR = os.path.join(BASE_DIR, "results", "04_latent_biology")
FIGDIR = os.path.join(BASE_DIR, "figures", "04_latent_biology")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

SELECTED_LATENTS = ["VAE_Z9", "VAE_Z8", "VAE_Z10", "VAE_Z13", "VAE_Z11"]

TOP_N_GENES = 30


# =========================
# HELPERS
# =========================

def normalize_tcga_id(x):
    """
    Harmonise TCGA barcodes:
    TCGA-XX-XXXX-01 -> TCGA-XX-XXXX
    TCGA-XX-XXXX    -> TCGA-XX-XXXX
    """
    if pd.isna(x):
        return x
    x = str(x).strip()
    parts = x.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return x


def load_expression(path):
    expr = pd.read_csv(path, index_col=0)

    # Si genes en lignes et samples en colonnes
    if expr.shape[0] > expr.shape[1]:
        print("Expression appears to be genes x samples. Transposing...")
        expr = expr.T

    expr.index = expr.index.map(normalize_tcga_id)
    expr = expr[~expr.index.duplicated(keep="first")]

    return expr


def load_latent(path):
    latent = pd.read_csv(path, index_col=0)
    latent.index = latent.index.map(normalize_tcga_id)
    latent = latent[~latent.index.duplicated(keep="first")]
    return latent


def plot_top_correlated(df, latent_dim, direction="positive"):
    if direction == "positive":
        sub = df.sort_values("spearman_rho", ascending=False).head(TOP_N_GENES)
        title = f"{latent_dim} - Top positively correlated genes"
        fname = f"{latent_dim}_top_positive.png"
    else:
        sub = df.sort_values("spearman_rho", ascending=True).head(TOP_N_GENES)
        title = f"{latent_dim} - Top negatively correlated genes"
        fname = f"{latent_dim}_top_negative.png"

    if sub.empty:
        print(f"[WARNING] No genes to plot for {latent_dim} {direction}")
        return

    plt.figure(figsize=(8, 8))
    plt.barh(sub["gene"], sub["spearman_rho"])
    plt.axvline(0, linestyle="--")
    plt.xlabel("Spearman rho")
    plt.ylabel("Gene")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, fname), dpi=300)
    plt.close()


def compute_gene_correlations(expr, latent_vector, latent_dim):
    results = []

    for gene in expr.columns:
        x = expr[gene].values
        y = latent_vector.values

        if np.std(x) == 0 or np.std(y) == 0:
            continue

        rho, p = spearmanr(x, y)

        if np.isnan(rho):
            continue

        results.append({
            "latent_dimension": latent_dim,
            "gene": gene,
            "spearman_rho": rho,
            "p_value": p
        })

    res = pd.DataFrame(results)

    if not res.empty:
        res["abs_rho"] = res["spearman_rho"].abs()
        res = res.sort_values("abs_rho", ascending=False)

    return res


def run_gsea_if_available(corr_df, latent_dim):
    try:
        import gseapy as gp
    except ImportError:
        print("gseapy not installed. Skipping GSEA.")
        print("Install with: pip install gseapy")
        return None

    ranked = corr_df[["gene", "spearman_rho"]].dropna()
    ranked = ranked.sort_values("spearman_rho", ascending=False)

    rnk_path = os.path.join(OUTDIR, f"{latent_dim}_ranked_genes.rnk")
    ranked.to_csv(rnk_path, sep="\t", index=False, header=False)

    try:
        pre_res = gp.prerank(
            rnk=rnk_path,
            gene_sets="MSigDB_Hallmark_2020",
            outdir=os.path.join(OUTDIR, f"{latent_dim}_GSEA"),
            min_size=10,
            max_size=500,
            permutation_num=1000,
            seed=42,
            verbose=False
        )

        gsea_res = pre_res.res2d
        gsea_res.to_csv(os.path.join(OUTDIR, f"{latent_dim}_GSEA_results.csv"))
        return gsea_res

    except Exception as e:
        print(f"GSEA failed for {latent_dim}: {e}")
        return None


# =========================
# MAIN
# =========================

def main():

    print("=" * 60)
    print("Loading data")
    print("=" * 60)

    expr = load_expression(EXPR_PATH)
    latent = load_latent(LATENT_PATH)

    print("Expression shape:", expr.shape)
    print("Latent shape:", latent.shape)

    print("\nExpression ID examples:")
    print(expr.index[:10].tolist())

    print("\nLatent ID examples:")
    print(latent.index[:10].tolist())

    common_samples = sorted(set(expr.index).intersection(set(latent.index)))

    print("\nCommon samples after TCGA ID normalization:", len(common_samples))

    if len(common_samples) == 0:
        raise ValueError(
            "No common samples between expression matrix and VAE latent space. "
            "Check that both files come from the same cohort."
        )

    expr = expr.loc[common_samples]
    latent = latent.loc[common_samples]

    print("Final expression shape:", expr.shape)
    print("Final latent shape:", latent.shape)

    all_corrs = []

    for latent_dim in SELECTED_LATENTS:

        if latent_dim not in latent.columns:
            print(f"[WARNING] {latent_dim} not found in latent matrix. Skipping.")
            continue

        print("\n" + "=" * 60)
        print(f"Processing {latent_dim}")
        print("=" * 60)

        latent_vector = latent[latent_dim]

        corr_df = compute_gene_correlations(expr, latent_vector, latent_dim)

        if corr_df.empty:
            print(f"[WARNING] No valid correlations for {latent_dim}")
            continue

        out_corr = os.path.join(OUTDIR, f"{latent_dim}_gene_correlations.csv")
        corr_df.to_csv(out_corr, index=False)

        print(f"Saved correlations: {out_corr}")

        plot_top_correlated(corr_df, latent_dim, direction="positive")
        plot_top_correlated(corr_df, latent_dim, direction="negative")

        gsea_res = run_gsea_if_available(corr_df, latent_dim)

        if gsea_res is not None:
            print(f"GSEA done for {latent_dim}")

        all_corrs.append(corr_df)

    if all_corrs:
        all_corrs = pd.concat(all_corrs, axis=0)
        all_corrs.to_csv(os.path.join(OUTDIR, "all_latent_gene_correlations.csv"), index=False)

    print("\nDone.")
    print("Results saved in:", OUTDIR)
    print("Figures saved in:", FIGDIR)


if __name__ == "__main__":
    main()
