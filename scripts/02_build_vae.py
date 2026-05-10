#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepGliomaAI - Script 02
Professional VAE pipeline for glioma transcriptomic representation learning

Goal:
- Load normalized transcriptomic data
- Match with clinical metadata
- Select high-variance genes
- Train a Variational Autoencoder
- Extract latent space
- Visualize latent space using PCA and UMAP
- Compare latent representation with clinical variables
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = "/mnt/d/projects/DeepGliomaAI"

EXPR_PATH = os.path.join(BASE_DIR, "data", "expr_mat_log2.csv")
CLIN_PATH = os.path.join(BASE_DIR, "data", "clinical_matched.csv")

OUTDIR = os.path.join(BASE_DIR, "results", "02_vae")
FIGDIR = os.path.join(BASE_DIR, "figures", "02_vae")
MODELDIR = os.path.join(BASE_DIR, "models", "02_vae")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

SEED = 42
TOP_VAR_GENES = 5000

LATENT_DIM = 20
BATCH_SIZE = 64
EPOCHS = 120
LR = 1e-3
WEIGHT_DECAY = 1e-5
BETA_KL = 0.001
TEST_SIZE = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# DATASET
# =========================================================

class ExpressionDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# =========================================================
# VAE MODEL
# =========================================================

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


# =========================================================
# LOSS
# =========================================================

def vae_loss(recon_x, x, mu, logvar, beta=0.001):
    reconstruction_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    print("=" * 60)
    print("Loading data")
    print("=" * 60)

    expr = pd.read_csv(EXPR_PATH, index_col=0)
    clin = pd.read_csv(CLIN_PATH)

    print(f"Raw expression shape: {expr.shape}")
    print(f"Clinical shape: {clin.shape}")
    print("Clinical columns:")
    print(list(clin.columns))

    if "PATIENT_ID" not in clin.columns:
        raise ValueError("PATIENT_ID column not found in clinical file.")

    # Expression expected as genes x samples
    expr.columns = expr.columns.astype(str)
    clin["PATIENT_ID"] = clin["PATIENT_ID"].astype(str)

    common = sorted(list(set(expr.columns) & set(clin["PATIENT_ID"])))

    if len(common) == 0:
        print("No exact common samples found.")
        print("Trying TCGA-like truncation strategy...")

        expr_short = expr.columns.str.slice(0, 12)
        mapping = dict(zip(expr.columns, expr_short))
        expr = expr.rename(columns=mapping)

        common = sorted(list(set(expr.columns) & set(clin["PATIENT_ID"])))

    print(f"Common samples: {len(common)}")

    if len(common) == 0:
        raise ValueError("No common samples between expression and clinical data.")

    expr = expr[common]
    clin = clin[clin["PATIENT_ID"].isin(common)].copy()
    clin = clin.drop_duplicates("PATIENT_ID")
    clin = clin.set_index("PATIENT_ID").loc[common].reset_index()

    # transpose: samples x genes
    X = expr.T.copy()

    print(f"Matched expression matrix: {X.shape}")
    print(f"Matched clinical table: {clin.shape}")

    return X, clin


# =========================================================
# FEATURE SELECTION
# =========================================================

def select_high_variance_genes(X, top_n=5000):
    print("=" * 60)
    print("Selecting high-variance genes")
    print("=" * 60)

    gene_var = X.var(axis=0).sort_values(ascending=False)
    selected_genes = gene_var.head(top_n).index.tolist()

    X_sel = X[selected_genes].copy()

    gene_var.to_csv(os.path.join(OUTDIR, "gene_variance_all.csv"))
    pd.Series(selected_genes, name="selected_genes").to_csv(
        os.path.join(OUTDIR, "selected_high_variance_genes.csv"),
        index=False
    )

    print(f"Selected genes: {len(selected_genes)}")
    print(f"Selected matrix: {X_sel.shape}")

    plt.figure(figsize=(8, 5))
    plt.hist(gene_var.values, bins=80)
    plt.xlabel("Gene variance")
    plt.ylabel("Number of genes")
    plt.title("Distribution of gene variance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "gene_variance_distribution.png"), dpi=300)
    plt.close()

    return X_sel, selected_genes


# =========================================================
# TRAINING
# =========================================================

def train_vae(X_scaled):
    print("=" * 60)
    print("Training VAE")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    X_train, X_val = train_test_split(
        X_scaled,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    train_loader = DataLoader(
        ExpressionDataset(X_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        ExpressionDataset(X_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    input_dim = X_scaled.shape[1]
    model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    history = {
        "epoch": [],
        "train_total_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "val_total_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": []
    }

    best_val_loss = np.inf
    best_model_path = os.path.join(MODELDIR, "best_vae_model.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_total = []
        train_recon = []
        train_kl = []

        for batch in train_loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            recon, mu, logvar, _ = model(batch)
            loss, recon_loss, kl_loss = vae_loss(
                recon, batch, mu, logvar, beta=BETA_KL
            )

            loss.backward()
            optimizer.step()

            train_total.append(loss.item())
            train_recon.append(recon_loss.item())
            train_kl.append(kl_loss.item())

        model.eval()

        val_total = []
        val_recon = []
        val_kl = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)

                recon, mu, logvar, _ = model(batch)
                loss, recon_loss, kl_loss = vae_loss(
                    recon, batch, mu, logvar, beta=BETA_KL
                )

                val_total.append(loss.item())
                val_recon.append(recon_loss.item())
                val_kl.append(kl_loss.item())

        mean_train_total = np.mean(train_total)
        mean_train_recon = np.mean(train_recon)
        mean_train_kl = np.mean(train_kl)

        mean_val_total = np.mean(val_total)
        mean_val_recon = np.mean(val_recon)
        mean_val_kl = np.mean(val_kl)

        history["epoch"].append(epoch)
        history["train_total_loss"].append(mean_train_total)
        history["train_recon_loss"].append(mean_train_recon)
        history["train_kl_loss"].append(mean_train_kl)
        history["val_total_loss"].append(mean_val_total)
        history["val_recon_loss"].append(mean_val_recon)
        history["val_kl_loss"].append(mean_val_kl)

        if mean_val_total < best_val_loss:
            best_val_loss = mean_val_total
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {mean_train_total:.4f} | "
                f"Val loss: {mean_val_total:.4f} | "
                f"Recon: {mean_val_recon:.4f} | "
                f"KL: {mean_val_kl:.4f}"
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(OUTDIR, "vae_training_history.csv"), index=False)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    return model, history_df


# =========================================================
# LATENT EXTRACTION
# =========================================================

def extract_latent(model, X_scaled, sample_ids):
    print("=" * 60)
    print("Extracting latent space")
    print("=" * 60)

    model.eval()

    loader = DataLoader(
        ExpressionDataset(X_scaled),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_mu = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            mu, logvar = model.encode(batch)
            all_mu.append(mu.cpu().numpy())

    latent = np.vstack(all_mu)

    latent_df = pd.DataFrame(
        latent,
        index=sample_ids,
        columns=[f"VAE_Z{i+1}" for i in range(latent.shape[1])]
    )

    latent_df.to_csv(os.path.join(OUTDIR, "vae_latent_space.csv"))

    print(f"Latent matrix: {latent_df.shape}")

    return latent_df


# =========================================================
# PLOTS
# =========================================================

def plot_training(history_df):
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_total_loss"], label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_total_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "vae_training_loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["val_recon_loss"], label="Reconstruction loss")
    plt.plot(history_df["epoch"], history_df["val_kl_loss"], label="KL loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation reconstruction and KL loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "vae_recon_kl_loss.png"), dpi=300)
    plt.close()


def plot_latent_pca(latent_df, clin):
    print("=" * 60)
    print("Plotting latent PCA")
    print("=" * 60)

    sample_ids = latent_df.index.tolist()

    clin2 = clin.copy()
    clin2["PATIENT_ID"] = clin2["PATIENT_ID"].astype(str)
    clin2 = clin2.set_index("PATIENT_ID").loc[sample_ids]

    pca = PCA(n_components=2, random_state=SEED)
    pcs = pca.fit_transform(latent_df.values)

    pca_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1]
    }, index=sample_ids)

    pca_df = pd.concat([pca_df, clin2], axis=1)

    pca_df.to_csv(os.path.join(OUTDIR, "vae_latent_pca_coordinates.csv"))

    variables_to_plot = [
        "CANCER_TYPE_ACRONYM",
        "SUBTYPE",
        "grade_group",
        "AGE",
        "SEX",
        "OS_event",
        "OS_STATUS"
    ]

    for var in variables_to_plot:
        if var not in pca_df.columns:
            continue

        plt.figure(figsize=(8, 6))

        if pd.api.types.is_numeric_dtype(pca_df[var]):
            sc = plt.scatter(
                pca_df["PC1"],
                pca_df["PC2"],
                c=pca_df[var],
                alpha=0.75
            )
            plt.colorbar(sc, label=var)

        else:
            groups = pca_df[var].astype(str).fillna("NA")
            for g in sorted(groups.unique()):
                idx = groups == g
                plt.scatter(
                    pca_df.loc[idx, "PC1"],
                    pca_df.loc[idx, "PC2"],
                    label=g,
                    alpha=0.75
                )
            plt.legend(fontsize=8)

        plt.xlabel(f"Latent PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"Latent PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.title(f"VAE latent PCA colored by {var}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGDIR, f"vae_latent_pca_by_{var}.png"), dpi=300)
        plt.close()


def try_umap_plot(latent_df, clin):
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return

    print("=" * 60)
    print("Plotting UMAP")
    print("=" * 60)

    sample_ids = latent_df.index.tolist()

    clin2 = clin.copy()
    clin2["PATIENT_ID"] = clin2["PATIENT_ID"].astype(str)
    clin2 = clin2.set_index("PATIENT_ID").loc[sample_ids]

    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.2,
        metric="euclidean",
        random_state=SEED
    )

    emb = reducer.fit_transform(latent_df.values)

    umap_df = pd.DataFrame({
        "UMAP1": emb[:, 0],
        "UMAP2": emb[:, 1]
    }, index=sample_ids)

    umap_df = pd.concat([umap_df, clin2], axis=1)
    umap_df.to_csv(os.path.join(OUTDIR, "vae_latent_umap_coordinates.csv"))

    variables_to_plot = [
        "CANCER_TYPE_ACRONYM",
        "SUBTYPE",
        "grade_group",
        "AGE",
        "SEX",
        "OS_event",
        "OS_STATUS"
    ]

    for var in variables_to_plot:
        if var not in umap_df.columns:
            continue

        plt.figure(figsize=(8, 6))

        if pd.api.types.is_numeric_dtype(umap_df[var]):
            sc = plt.scatter(
                umap_df["UMAP1"],
                umap_df["UMAP2"],
                c=umap_df[var],
                alpha=0.75
            )
            plt.colorbar(sc, label=var)

        else:
            groups = umap_df[var].astype(str).fillna("NA")
            for g in sorted(groups.unique()):
                idx = groups == g
                plt.scatter(
                    umap_df.loc[idx, "UMAP1"],
                    umap_df.loc[idx, "UMAP2"],
                    label=g,
                    alpha=0.75
                )
            plt.legend(fontsize=8)

        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title(f"VAE latent UMAP colored by {var}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGDIR, f"vae_latent_umap_by_{var}.png"), dpi=300)
        plt.close()


# =========================================================
# LATENT ASSOCIATION WITH CLINICAL VARIABLES
# =========================================================

def latent_clinical_associations(latent_df, clin):
    print("=" * 60)
    print("Computing latent-clinical associations")
    print("=" * 60)

    sample_ids = latent_df.index.tolist()

    clin2 = clin.copy()
    clin2["PATIENT_ID"] = clin2["PATIENT_ID"].astype(str)
    clin2 = clin2.set_index("PATIENT_ID").loc[sample_ids]

    merged = pd.concat([latent_df, clin2], axis=1)

    results = []

    clinical_vars = ["AGE", "OS_MONTHS", "OS_event", "PFS_MONTHS", "PFS_event"]

    for z in latent_df.columns:
        for var in clinical_vars:
            if var not in merged.columns:
                continue

            x = pd.to_numeric(merged[z], errors="coerce")
            y = pd.to_numeric(merged[var], errors="coerce")

            valid = x.notna() & y.notna()

            if valid.sum() < 20:
                continue

            rho = pd.Series(x[valid]).corr(pd.Series(y[valid]), method="spearman")

            results.append({
                "latent_dimension": z,
                "clinical_variable": var,
                "spearman_rho": rho,
                "n": int(valid.sum())
            })

    assoc_df = pd.DataFrame(results)
    assoc_df.to_csv(os.path.join(OUTDIR, "latent_clinical_correlations.csv"), index=False)

    if assoc_df.shape[0] > 0:
        print(assoc_df.sort_values("spearman_rho", key=lambda x: abs(x), ascending=False).head(20))

    return assoc_df


# =========================================================
# MAIN
# =========================================================

def main():
    print("\nDeepGliomaAI - VAE representation learning")
    print("=" * 60)

    X, clin = load_data()

    X_sel, selected_genes = select_high_variance_genes(X, top_n=TOP_VAR_GENES)

    sample_ids = X_sel.index.astype(str).tolist()

    print("=" * 60)
    print("Scaling data")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel.values)

    pd.DataFrame(
        X_scaled,
        index=sample_ids,
        columns=selected_genes
    ).to_csv(os.path.join(OUTDIR, "scaled_expression_selected_genes.csv"))

    model, history_df = train_vae(X_scaled)

    plot_training(history_df)

    latent_df = extract_latent(model, X_scaled, sample_ids)

    plot_latent_pca(latent_df, clin)

    try_umap_plot(latent_df, clin)

    latent_clinical_associations(latent_df, clin)

    print("=" * 60)
    print("Done.")
    print(f"Results saved in: {OUTDIR}")
    print(f"Figures saved in: {FIGDIR}")
    print(f"Model saved in: {MODELDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
