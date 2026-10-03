#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepGliomaAI - leakage-free internal validation of the VAE + Cox pipeline.

Purpose
-------
Estimate prognostic performance without allowing the outer test folds to
influence:
  - high-variance gene selection,
  - expression scaling,
  - VAE fitting / early stopping,
  - latent-feature scaling,
  - survival-based latent-dimension selection,
  - Cox model fitting.

The script performs an outer stratified K-fold cross-validation. Within each
outer training fold, the VAE uses an inner validation split only to select the
number of training epochs. A fresh VAE is then retrained on the complete outer
training fold for that number of epochs, before any transformation of the
outer test fold.

Outputs
-------
results/10_leakage_free_internal_validation/
    fold_metrics.csv
    fold_selected_latent_dims.csv
    oof_predictions.csv
    summary.txt
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_outer_splits: int = 5
    inner_val_fraction: float = 0.15

    top_var_genes: int = 5000
    latent_dim: int = 20

    batch_size: int = 64
    max_epochs: int = 120
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    beta_kl: float = 0.001

    n_selected_latent: int = 5
    latent_fdr_threshold: float = 0.05
    cox_penalizer: float = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_sample_id(x: object) -> str:
    """Convert TCGA aliquot/sample IDs to patient IDs."""
    x = str(x).replace(".", "-").strip()
    parts = x.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return x


def load_expression(path: Path) -> pd.DataFrame:
    expr = pd.read_csv(path, index_col=0)

    if expr.shape[0] > expr.shape[1]:
        expr = expr.T

    expr.index = [clean_sample_id(i) for i in expr.index]
    expr = expr.groupby(expr.index).mean()
    expr = expr.apply(pd.to_numeric, errors="coerce")

    if expr.isna().any().any():
        n_missing = int(expr.isna().sum().sum())
        raise ValueError(
            f"Expression matrix contains {n_missing} missing values. "
            "Resolve missingness before running the validation pipeline."
        )

    return expr


def load_clinical(path: Path) -> pd.DataFrame:
    clin = pd.read_csv(path)
    if "PATIENT_ID" not in clin.columns:
        raise ValueError("clinical file must contain PATIENT_ID")

    clin["PATIENT_ID"] = clin["PATIENT_ID"].map(clean_sample_id)
    clin = clin.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")
    return clin


def prepare_survival_dataset(
    expr: pd.DataFrame, clin: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = sorted(set(expr.index) & set(clin.index))
    if not common:
        raise ValueError("No common patients between expression and clinical data.")

    expr = expr.loc[common].copy()
    clin = clin.loc[common].copy()

    required = ["OS_MONTHS", "OS_event"]
    missing_cols = [c for c in required if c not in clin.columns]
    if missing_cols:
        raise ValueError(f"Missing required clinical columns: {missing_cols}")

    clin["OS_MONTHS"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    clin["OS_event"] = pd.to_numeric(clin["OS_event"], errors="coerce")

    keep = (
        clin["OS_MONTHS"].notna()
        & clin["OS_event"].notna()
        & (clin["OS_MONTHS"] > 0)
        & clin["OS_event"].isin([0, 1])
    )

    expr = expr.loc[keep].copy()
    clin = clin.loc[keep].copy()
    clin["OS_event"] = clin["OS_event"].astype(int)

    return expr, clin


class ExpressionDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
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
            nn.ReLU(),
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
            nn.Linear(1024, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float,
) -> torch.Tensor:
    recon = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta_kl * kl


def make_loader(
    x: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ExpressionDataset(x),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def select_best_epoch(
    x_outer_train: np.ndarray,
    cfg: Config,
    device: str,
    fold_seed: int,
) -> int:
    """
    Use an inner validation split only to choose the number of epochs.
    The final VAE for the fold is retrained from scratch on all outer-training
    samples for the selected number of epochs.
    """
    idx = np.arange(x_outer_train.shape[0])
    inner_train_idx, inner_val_idx = train_test_split(
        idx,
        test_size=cfg.inner_val_fraction,
        random_state=fold_seed,
        shuffle=True,
    )

    set_seed(fold_seed)
    model = VAE(x_outer_train.shape[1], cfg.latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    train_loader = make_loader(
        x_outer_train[inner_train_idx], cfg.batch_size, shuffle=True
    )
    val_loader = make_loader(
        x_outer_train[inner_val_idx], cfg.batch_size, shuffle=False
    )

    best_epoch = 1
    best_val_loss = np.inf

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, cfg.beta_kl)
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss = vae_loss(recon, batch, mu, logvar, cfg.beta_kl)
                losses.append(float(loss.item()))

        mean_val = float(np.mean(losses))
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_epoch = epoch

    return best_epoch


def train_final_vae(
    x_outer_train: np.ndarray,
    cfg: Config,
    device: str,
    fold_seed: int,
    n_epochs: int,
) -> VAE:
    set_seed(fold_seed)
    model = VAE(x_outer_train.shape[1], cfg.latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    loader = make_loader(x_outer_train, cfg.batch_size, shuffle=True)

    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, cfg.beta_kl)
            loss.backward()
            optimizer.step()

    return model


def encode_mu(
    model: VAE,
    x: np.ndarray,
    cfg: Config,
    device: str,
) -> np.ndarray:
    model.eval()
    loader = make_loader(x, cfg.batch_size, shuffle=False)
    all_mu = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            all_mu.append(mu.cpu().numpy())
    return np.vstack(all_mu)


def select_latent_dimensions(
    z_train: pd.DataFrame,
    clin_train: pd.DataFrame,
    cfg: Config,
) -> tuple[list[str], pd.DataFrame]:
    rows = []

    for col in z_train.columns:
        tmp = pd.DataFrame(
            {
                "OS_MONTHS": clin_train["OS_MONTHS"],
                "OS_event": clin_train["OS_event"],
                col: z_train[col],
            }
        ).dropna()

        try:
            cph = CoxPHFitter(penalizer=0.0)
            cph.fit(tmp, duration_col="OS_MONTHS", event_col="OS_event")
            rows.append(
                {
                    "latent_dim": col,
                    "coef": float(cph.params_[col]),
                    "HR": float(np.exp(cph.params_[col])),
                    "p": float(cph.summary.loc[col, "p"]),
                }
            )
        except Exception:
            rows.append(
                {
                    "latent_dim": col,
                    "coef": np.nan,
                    "HR": np.nan,
                    "p": np.nan,
                }
            )

    uni = pd.DataFrame(rows)
    valid = uni["p"].notna()
    uni["FDR"] = np.nan

    if valid.any():
        uni.loc[valid, "FDR"] = multipletests(
            uni.loc[valid, "p"].values,
            method="fdr_bh",
        )[1]

    ranked = uni.dropna(subset=["p"]).sort_values(["FDR", "p"])

    significant = ranked[
        ranked["FDR"] < cfg.latent_fdr_threshold
    ]["latent_dim"].tolist()

    if len(significant) >= cfg.n_selected_latent:
        selected = significant[: cfg.n_selected_latent]
    else:
        selected = ranked["latent_dim"].head(cfg.n_selected_latent).tolist()

    if not selected:
        raise RuntimeError("No latent dimensions could be selected.")

    return selected, uni.sort_values("p", na_position="last")


def fit_and_score_cox(
    z_train: pd.DataFrame,
    z_test: pd.DataFrame,
    clin_train: pd.DataFrame,
    clin_test: pd.DataFrame,
    selected: list[str],
    cfg: Config,
) -> tuple[float, np.ndarray, list[str]]:
    """
    Fit latent-feature scaling and Cox model on outer train only, then score the
    untouched outer test fold.
    """
    use_age = "AGE" in clin_train.columns and "AGE" in clin_test.columns

    x_train = z_train[selected].copy()
    x_test = z_test[selected].copy()
    features = selected.copy()

    if use_age:
        age_train = pd.to_numeric(clin_train["AGE"], errors="coerce")
        age_test = pd.to_numeric(clin_test["AGE"], errors="coerce")

        if age_train.notna().all() and age_test.notna().all():
            x_train["AGE"] = age_train.values
            x_test["AGE"] = age_test.values
            features.append("AGE")

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        index=x_train.index,
        columns=features,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        index=x_test.index,
        columns=features,
    )

    train_cox = x_train_scaled.copy()
    train_cox["OS_MONTHS"] = clin_train["OS_MONTHS"].values
    train_cox["OS_event"] = clin_train["OS_event"].values

    cph = CoxPHFitter(penalizer=cfg.cox_penalizer)
    cph.fit(
        train_cox,
        duration_col="OS_MONTHS",
        event_col="OS_event",
    )

    risk_test = cph.predict_partial_hazard(x_test_scaled).values.ravel()

    c_index = concordance_index(
        clin_test["OS_MONTHS"].values,
        -risk_test,
        clin_test["OS_event"].values,
    )

    return float(c_index), risk_test, features


def run_validation(
    expr: pd.DataFrame,
    clin: pd.DataFrame,
    outdir: Path,
    cfg: Config,
    device: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedKFold(
        n_splits=cfg.n_outer_splits,
        shuffle=True,
        random_state=cfg.seed,
    )

    fold_metrics = []
    selected_rows = []
    oof_rows = []

    y_strat = clin["OS_event"].values

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(expr, y_strat), start=1
    ):
        fold_seed = cfg.seed + fold

        train_ids = expr.index[train_idx]
        test_ids = expr.index[test_idx]

        expr_train = expr.loc[train_ids]
        expr_test = expr.loc[test_ids]
        clin_train = clin.loc[train_ids]
        clin_test = clin.loc[test_ids]

        # 1) Feature selection on OUTER TRAIN ONLY.
        gene_variance = expr_train.var(axis=0).sort_values(ascending=False)
        selected_genes = gene_variance.head(
            min(cfg.top_var_genes, expr_train.shape[1])
        ).index.tolist()

        # 2) Expression scaling on OUTER TRAIN ONLY.
        expr_scaler = StandardScaler()
        x_train = expr_scaler.fit_transform(expr_train[selected_genes].values)
        x_test = expr_scaler.transform(expr_test[selected_genes].values)

        # 3) Inner validation chooses epoch count; final VAE is then fitted on
        #    all outer-training patients. Outer test is still untouched.
        best_epoch = select_best_epoch(
            x_train,
            cfg=cfg,
            device=device,
            fold_seed=fold_seed,
        )
        vae = train_final_vae(
            x_train,
            cfg=cfg,
            device=device,
            fold_seed=fold_seed,
            n_epochs=best_epoch,
        )

        z_train_arr = encode_mu(vae, x_train, cfg, device)
        z_test_arr = encode_mu(vae, x_test, cfg, device)

        latent_cols = [f"VAE_Z{i+1}" for i in range(cfg.latent_dim)]
        z_train = pd.DataFrame(z_train_arr, index=train_ids, columns=latent_cols)
        z_test = pd.DataFrame(z_test_arr, index=test_ids, columns=latent_cols)

        # 4) Latent selection uses survival outcomes from OUTER TRAIN ONLY.
        selected, uni = select_latent_dimensions(z_train, clin_train, cfg)

        for _, row in uni.iterrows():
            selected_rows.append(
                {
                    "fold": fold,
                    "latent_dim": row["latent_dim"],
                    "coef_univariate": row["coef"],
                    "HR_univariate": row["HR"],
                    "p_univariate": row["p"],
                    "FDR_univariate": row["FDR"],
                    "selected_for_multivariate": row["latent_dim"] in selected,
                }
            )

        # 5) Cox fit on OUTER TRAIN; evaluate once on untouched OUTER TEST.
        c_index, risk_test, cox_features = fit_and_score_cox(
            z_train,
            z_test,
            clin_train,
            clin_test,
            selected,
            cfg,
        )

        fold_metrics.append(
            {
                "fold": fold,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "events_train": int(clin_train["OS_event"].sum()),
                "events_test": int(clin_test["OS_event"].sum()),
                "best_vae_epoch": best_epoch,
                "selected_latent_dims": ";".join(selected),
                "cox_features": ";".join(cox_features),
                "c_index": c_index,
            }
        )

        for patient_id, risk in zip(test_ids, risk_test):
            oof_rows.append(
                {
                    "PATIENT_ID": patient_id,
                    "fold": fold,
                    "OS_MONTHS": float(clin_test.loc[patient_id, "OS_MONTHS"]),
                    "OS_event": int(clin_test.loc[patient_id, "OS_event"]),
                    "risk_score": float(risk),
                }
            )

        print(
            f"Fold {fold}/{cfg.n_outer_splits} | "
            f"best_epoch={best_epoch:03d} | "
            f"selected={selected} | "
            f"C-index={c_index:.3f}"
        )

        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(fold_metrics)
    selected_df = pd.DataFrame(selected_rows)
    oof_df = pd.DataFrame(oof_rows).sort_values("PATIENT_ID")

    metrics_df.to_csv(outdir / "fold_metrics.csv", index=False)
    selected_df.to_csv(outdir / "fold_selected_latent_dims.csv", index=False)
    oof_df.to_csv(outdir / "oof_predictions.csv", index=False)

    fold_mean = float(metrics_df["c_index"].mean())
    fold_std = float(metrics_df["c_index"].std(ddof=1))

    selection_frequency = (
        selected_df[selected_df["selected_for_multivariate"]]
        .groupby("latent_dim")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    summary = {
        "n_patients": int(expr.shape[0]),
        "n_events": int(clin["OS_event"].sum()),
        "outer_folds": cfg.n_outer_splits,
        "fold_cindex_mean": fold_mean,
        "fold_cindex_sd": fold_std,
        "latent_selection_frequency": selection_frequency,
        "device": device,
        "config": cfg.__dict__,
    }

    with open(outdir / "summary.txt", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2))

    print("\nLeakage-free internal validation completed.")
    print(f"Fold C-index: {fold_mean:.3f} ± {fold_std:.3f}")
    print(f"Results: {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-free outer CV for DeepGliomaAI VAE + Cox survival."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="DeepGliomaAI repository root.",
    )
    parser.add_argument("--expr", type=Path, default=None)
    parser.add_argument("--clinical", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    expr_path = args.expr or root / "data" / "expr_mat_log2.csv"
    clinical_path = args.clinical or root / "data" / "clinical_matched.csv"
    outdir = args.outdir or root / "results" / "10_leakage_free_internal_validation"

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")

    cfg = Config(
        seed=args.seed,
        n_outer_splits=args.outer_folds,
        max_epochs=args.max_epochs,
    )

    print("Loading data...")
    expr = load_expression(expr_path)
    clin = load_clinical(clinical_path)
    expr, clin = prepare_survival_dataset(expr, clin)

    print(f"Patients with complete OS: {expr.shape[0]}")
    print(f"Genes: {expr.shape[1]}")
    print(f"Events: {int(clin['OS_event'].sum())}")
    print(f"Device: {device}")

    run_validation(
        expr=expr,
        clin=clin,
        outdir=outdir,
        cfg=cfg,
        device=device,
    )


if __name__ == "__main__":
    main()
