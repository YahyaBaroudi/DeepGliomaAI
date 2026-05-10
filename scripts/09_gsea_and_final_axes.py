#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09_gsea_and_final_axes.py

Objectifs :
1) Corriger/régénérer proprement la GSEA des dimensions latentes VAE_Z1–VAE_Z20.
2) Construire une figure finale synthétique en 4 axes biologiques :
   - Proliferation / high grade
   - Mesenchymal / invasion / angiogenesis
   - Neuronal / proneural / protective
   - Immune / microenvironment

À lancer depuis : /mnt/d/projects/DeepGliomaAI
    python scripts/09_fix_gsea_and_final_axes.py

Dépendances : pandas, numpy, scipy, matplotlib, gseapy
Si besoin :
    conda install -c conda-forge gseapy scipy matplotlib pandas numpy -y
"""

import os
import re
import glob
import json
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = "/mnt/d/projects/DeepGliomaAI"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

EXPR_PATH = os.path.join(DATA_DIR, "expr_mat_log2.csv")
CLIN_PATH = os.path.join(DATA_DIR, "clinical_matched.csv")
LATENT_PATH = os.path.join(RESULTS_DIR, "02_vae", "vae_latent_space.csv")

CORR_PATH = os.path.join(
    RESULTS_DIR, "08_latent_biological_annotation", "all_latent_gene_correlations_used.csv"
)
if not os.path.exists(CORR_PATH):
    CORR_PATH = os.path.join(
        RESULTS_DIR, "04_latent_biology", "all_latent_gene_correlations.csv"
    )

BIOMARKER_CORR_PATH = os.path.join(
    RESULTS_DIR, "08_latent_biological_annotation", "latent_dimension_biomarker_correlations.csv"
)
CLINICAL_ASSOC_PATH = os.path.join(
    RESULTS_DIR, "08_latent_biological_annotation", "latent_dimension_clinical_subtype_associations.csv"
)
SURVIVAL_PATH = os.path.join(
    RESULTS_DIR, "08_latent_biological_annotation", "latent_dimension_survival_summary.csv"
)
TOP_GENES_PATH = os.path.join(
    RESULTS_DIR, "08_latent_biological_annotation", "latent_top_positive_negative_genes.csv"
)

OUTDIR = os.path.join(RESULTS_DIR, "09_gsea_final_four_axes")
FIGDIR = os.path.join(FIGURES_DIR, "09_gsea_final_four_axes")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

LATENT_DIMS = [f"VAE_Z{i}" for i in range(1, 21)]
RANDOM_STATE = 42

# Les 4 axes finaux : keywords gènes + voies Hallmark/MSigDB
AXES = {
    "Proliferation / high grade": {
        "genes": [
            "MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "BIRC5", "AURKA", "AURKB",
            "CDC20", "MCM2", "MCM4", "MCM6", "UBE2C", "KIF2C", "KIF20A", "MELK"
        ],
        "pathway_keywords": [
            "E2F", "G2M", "G2-M", "MITOTIC", "MITOTIC_SPINDLE", "MYC", "DNA_REPAIR",
            "CELL_CYCLE", "PROLIFERATION"
        ],
    },
    "Mesenchymal / invasion / angiogenesis": {
        "genes": [
            "CHI3L1", "IGFBP2", "POSTN", "COL1A1", "COL1A2", "FN1", "VIM", "VEGFA",
            "CA9", "SERPINE1", "THBS1", "TNC", "ANXA1", "CD44", "ITGA5", "MMP2", "MMP9"
        ],
        "pathway_keywords": [
            "EPITHELIAL_MESENCHYMAL", "EPITHELIAL MESENCHYMAL", "EMT", "ANGIOGENESIS",
            "HYPOXIA", "COAGULATION", "ECM", "EXTRACELLULAR", "INVASION"
        ],
    },
    "Immune / microenvironment": {
        "genes": [
            "PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "CD8B", "HLA-DRA", "HLA-A",
            "CXCL10", "CXCL9", "STAT1", "IRF1", "CD274", "PDCD1", "CTLA4",
            "AIF1", "ITGAM", "C1QA", "C1QB", "C1QC", "LST1", "TYROBP"
        ],
        "pathway_keywords": [
            "INTERFERON", "INFLAMMATORY", "TNFA", "TNF", "NFKB", "NF-KB", "COMPLEMENT",
            "IL6", "JAK", "STAT", "ALLOGRAFT", "IMMUNE", "ANTIGEN"
        ],
    },
    "Neuronal / proneural / protective": {
        "genes": [
            "OLIG2", "SOX2", "DLL3", "ASCL1", "DCX", "MAP2", "SYT1", "SNAP25",
            "NEFL", "NEFM", "GABRA1", "SLC17A7", "RBFOX3", "TUBB3", "GRIN1"
        ],
        "pathway_keywords": [
            "NEURON", "NEURONAL", "SYNAP", "PRONEURAL", "OLIGODENDRO", "AXON", "DENDRITE"
        ],
    },
}


# ============================================================
# HELPERS
# ============================================================
def clean_patient_id(x: str) -> str:
    return str(x).replace(".", "-")[:12]


def save_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    df.to_csv(os.path.join(OUTDIR, filename), index=index)


def plot_and_save(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean(skipna=True)) / sd


def minmax_abs(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    m = s.abs().max()
    if m == 0 or pd.isna(m):
        return pd.Series(0.0, index=s.index)
    return s / m


def load_expression(path: str) -> pd.DataFrame:
    expr = pd.read_csv(path, index_col=0)
    rownames = expr.index.astype(str)
    tcga_like = rownames.str.startswith("TCGA-").mean()
    if tcga_like < 0.3:
        expr = expr.T
    expr.index = expr.index.astype(str).str.replace(r"\.", "-", regex=True)
    expr.columns = expr.columns.astype(str)
    expr["PATIENT_ID"] = expr.index.str[:12]
    expr = expr.groupby("PATIENT_ID").mean(numeric_only=True)
    expr.index.name = "PATIENT_ID"
    return expr


def load_latent(path: str) -> pd.DataFrame:
    latent = pd.read_csv(path)
    first_col = latent.columns[0]
    if first_col not in LATENT_DIMS:
        latent = latent.rename(columns={first_col: "PATIENT_ID"})
    else:
        raise ValueError("Le fichier latent ne contient pas de colonne PATIENT_ID.")
    latent["PATIENT_ID"] = latent["PATIENT_ID"].apply(clean_patient_id)
    latent = latent.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")
    return latent[[c for c in LATENT_DIMS if c in latent.columns]].copy()


def find_gmt() -> Optional[str]:
    """Trouve un GMT local. Priorité aux gene_sets.gmt déjà utilisés par ton projet."""
    patterns = [
        os.path.join(RESULTS_DIR, "04_latent_biology", "VAE_Z*_gsea", "gene_sets.gmt"),
        os.path.join(RESULTS_DIR, "04_latent_biology", "*.gmt"),
        os.path.join(DATA_DIR, "*.gmt"),
        os.path.join(BASE_DIR, "*.gmt"),
    ]
    hits = []
    for p in patterns:
        hits.extend(glob.glob(p))
    hits = sorted(list(set(hits)))
    return hits[0] if hits else None


def normalize_gsea_output(gsea: pd.DataFrame) -> pd.DataFrame:
    """
    Correction principale : dans gseapy res2d, 'Name' vaut souvent 'prerank'.
    Le vrai nom de pathway est 'Term' ou l'index. On ne doit donc JAMAIS renommer 'Name' en pathway.
    """
    if gsea is None or gsea.empty:
        return pd.DataFrame()

    gsea = gsea.copy()

    # Si le pathway est dans l'index
    if gsea.index.name is not None or not isinstance(gsea.index, pd.RangeIndex):
        gsea = gsea.reset_index()

    gsea = gsea.loc[:, ~gsea.columns.duplicated()].copy()

    rename = {}
    for c in gsea.columns:
        lc = str(c).lower().strip()
        if lc in ["term", "terms", "gene_set", "geneset", "gene set", "index"]:
            rename[c] = "pathway"
        elif lc in ["nes", "normalized enrichment score", "normalized_enrichment_score"]:
            rename[c] = "NES"
        elif lc in ["es", "enrichment score", "enrichment_score"]:
            rename[c] = "ES"
        elif "fdr" in lc:
            rename[c] = "FDR"
        elif lc in ["nom p-val", "nom p val", "pval", "p-value", "p_value", "p_value"]:
            rename[c] = "p_value"
        elif lc in ["lead_genes", "lead genes", "leading_edge", "ledge_genes"]:
            rename[c] = "Lead_genes"
        # Important: 'Name' n'est PAS un pathway dans gseapy prerank.

    gsea = gsea.rename(columns=rename)
    gsea = gsea.loc[:, ~gsea.columns.duplicated()].copy()

    # Si pathway absent, essayer de récupérer une colonne textuelle qui n'est pas Name/prerank
    if "pathway" not in gsea.columns:
        text_cols = [c for c in gsea.columns if gsea[c].dtype == "object" and str(c).lower() != "name"]
        candidate = None
        for c in text_cols:
            vals = gsea[c].dropna().astype(str)
            if len(vals) > 0 and vals.nunique() > 1 and not (vals == "prerank").all():
                candidate = c
                break
        if candidate is not None:
            gsea["pathway"] = gsea[candidate]
        else:
            gsea["pathway"] = np.nan

    for col in ["NES", "ES", "FDR", "p_value"]:
        if col not in gsea.columns:
            gsea[col] = np.nan
        gsea[col] = pd.to_numeric(gsea[col], errors="coerce")

    if "Lead_genes" not in gsea.columns:
        gsea["Lead_genes"] = ""

    gsea["pathway"] = gsea["pathway"].astype(str)

    # Enlever les lignes où pathway == prerank ou nan si possible
    bad = gsea["pathway"].str.lower().isin(["prerank", "nan", "none"])
    if bad.all():
        print("[WARNING] Tous les pathways valent 'prerank'. Il faut rerun GSEA depuis gseapy avec la colonne Term.")
    else:
        gsea = gsea[~bad].copy()

    return gsea


# ============================================================
# GSEA RERUN
# ============================================================
def run_gsea_all_dimensions(corr_df: pd.DataFrame, gmt_path: str) -> pd.DataFrame:
    try:
        import gseapy as gp
    except Exception as e:
        raise ImportError(
            "gseapy n'est pas installé. Installe-le avec : conda install -c conda-forge gseapy -y"
        ) from e

    all_res = []

    for dim in LATENT_DIMS:
        sub = corr_df[corr_df["latent_dimension"] == dim].copy()
        if sub.empty:
            print(f"[WARNING] No correlations for {dim}, skipped.")
            continue

        sub = sub[["gene", "spearman_rho"]].dropna()
        sub = sub.sort_values("spearman_rho", ascending=False)

        rnk_path = os.path.join(OUTDIR, f"{dim}_ranked_genes.rnk")
        sub.to_csv(rnk_path, sep="\t", index=False, header=False)

        dim_out = os.path.join(OUTDIR, f"{dim}_gsea")
        os.makedirs(dim_out, exist_ok=True)

        print(f"[GSEA] Running {dim}...")
        res = gp.prerank(
            rnk=rnk_path,
            gene_sets=gmt_path,
            outdir=dim_out,
            permutation_num=1000,
            min_size=10,
            max_size=500,
            seed=RANDOM_STATE,
            verbose=False,
        )

        out = res.res2d.copy()
        out = normalize_gsea_output(out)
        out["latent_dimension"] = dim
        all_res.append(out)

    if len(all_res) == 0:
        return pd.DataFrame()

    gsea = pd.concat(all_res, axis=0, ignore_index=True)
    gsea = normalize_gsea_output(gsea)
    return gsea


def summarize_gsea(gsea: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if gsea.empty:
        return pd.DataFrame()

    gsea = normalize_gsea_output(gsea)
    rows = []

    for dim in LATENT_DIMS:
        sub = gsea[gsea["latent_dimension"] == dim].copy()
        if sub.empty:
            continue

        sig = sub[sub["FDR"] <= 0.25].copy()

        pos = sig[sig["NES"] > 0].sort_values(["FDR", "NES"], ascending=[True, False]).head(top_n)
        neg = sig[sig["NES"] < 0].sort_values(["FDR", "NES"], ascending=[True, True]).head(top_n)

        rows.append({
            "latent_dimension": dim,
            "top_positive_pathways": ";".join(pos["pathway"].astype(str).tolist()),
            "top_negative_pathways": ";".join(neg["pathway"].astype(str).tolist()),
            "n_sig_pathways_FDR_0_25": int(sig.shape[0]),
            "max_positive_NES": float(pos["NES"].max()) if not pos.empty else np.nan,
            "max_negative_NES": float(neg["NES"].min()) if not neg.empty else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# FOUR AXES SCORES
# ============================================================
def compute_axis_gene_scores(corr_df: pd.DataFrame, biomarker_corr: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Source 1 : corrélations genome-wide si les gènes existent dans corr_df
    for dim in LATENT_DIMS:
        sub = corr_df[corr_df["latent_dimension"] == dim].copy()
        sub["gene_upper"] = sub["gene"].astype(str).str.upper()

        for axis, info in AXES.items():
            genes = [g.upper() for g in info["genes"]]
            hit = sub[sub["gene_upper"].isin(genes)].copy()

            if not hit.empty:
                raw = hit["spearman_rho"].mean()
                abs_score = hit["spearman_rho"].abs().mean()
                n = hit.shape[0]
                source = "genome_wide_corr"
            else:
                raw = np.nan
                abs_score = np.nan
                n = 0
                source = "genome_wide_corr"

            rows.append({
                "latent_dimension": dim,
                "axis": axis,
                "gene_axis_score_signed": raw,
                "gene_axis_score_abs": abs_score,
                "n_axis_genes_found": n,
                "source": source,
            })

    out = pd.DataFrame(rows)

    # Source 2 : biomarker_corr fallback/renforcement
    if not biomarker_corr.empty:
        rows2 = []
        bc = biomarker_corr.copy()
        bc["gene_upper"] = bc["biomarker_gene"].astype(str).str.upper()

        for dim in LATENT_DIMS:
            sub = bc[bc["latent_dimension"] == dim].copy()
            for axis, info in AXES.items():
                genes = [g.upper() for g in info["genes"]]
                hit = sub[sub["gene_upper"].isin(genes)].copy()
                rows2.append({
                    "latent_dimension": dim,
                    "axis": axis,
                    "biomarker_axis_score_signed": hit["spearman_rho"].mean() if not hit.empty else np.nan,
                    "biomarker_axis_score_abs": hit["spearman_rho"].abs().mean() if not hit.empty else np.nan,
                    "n_biomarkers_found": hit.shape[0],
                })
        bcout = pd.DataFrame(rows2)
        out = out.merge(bcout, on=["latent_dimension", "axis"], how="left")
    else:
        out["biomarker_axis_score_signed"] = np.nan
        out["biomarker_axis_score_abs"] = np.nan
        out["n_biomarkers_found"] = 0

    return out


def compute_axis_gsea_scores(gsea: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if gsea.empty:
        return pd.DataFrame(columns=["latent_dimension", "axis", "gsea_axis_score", "top_axis_pathways"])

    gsea = normalize_gsea_output(gsea)
    gsea["pathway_upper"] = gsea["pathway"].astype(str).str.upper()

    for dim in LATENT_DIMS:
        sub = gsea[gsea["latent_dimension"] == dim].copy()

        for axis, info in AXES.items():
            keywords = [k.upper() for k in info["pathway_keywords"]]
            mask = pd.Series(False, index=sub.index)
            for kw in keywords:
                mask = mask | sub["pathway_upper"].str.contains(kw, regex=False, na=False)

            hit = sub[mask].copy()
            hit_sig = hit[hit["FDR"] <= 0.25].copy()
            use = hit_sig if not hit_sig.empty else hit

            if not use.empty:
                # score signé : moyenne NES des pathways de l'axe
                score = use["NES"].mean()
                top_paths = use.sort_values("FDR").head(5)["pathway"].astype(str).tolist()
            else:
                score = np.nan
                top_paths = []

            rows.append({
                "latent_dimension": dim,
                "axis": axis,
                "gsea_axis_score": score,
                "top_axis_pathways": ";".join(top_paths),
                "n_axis_pathways": int(use.shape[0]) if not use.empty else 0,
            })

    return pd.DataFrame(rows)


def compute_final_axis_scores(
    corr_df: pd.DataFrame,
    biomarker_corr: pd.DataFrame,
    gsea: pd.DataFrame,
    survival: pd.DataFrame,
    clinical_assoc: pd.DataFrame,
) -> pd.DataFrame:
    gene_scores = compute_axis_gene_scores(corr_df, biomarker_corr)
    gsea_scores = compute_axis_gsea_scores(gsea)

    scores = gene_scores.merge(gsea_scores, on=["latent_dimension", "axis"], how="left")

    # Intensité biologique = abs(gene corr) + abs(biomarker corr) + abs(GSEA NES), standardisés
    scores["gene_component"] = scores.groupby("axis")["gene_axis_score_abs"].transform(lambda x: zscore_series(x).fillna(0))
    scores["biomarker_component"] = scores.groupby("axis")["biomarker_axis_score_abs"].transform(lambda x: zscore_series(x).fillna(0))
    scores["gsea_component"] = scores.groupby("axis")["gsea_axis_score"].transform(lambda x: zscore_series(x.abs()).fillna(0))

    scores["axis_score"] = (
        scores["gene_component"] + scores["biomarker_component"] + scores["gsea_component"]
    ) / 3.0

    # Ajouter survie
    if not survival.empty and "latent_dim" in survival.columns:
        surv = survival.copy()
        surv = surv.rename(columns={"latent_dim": "latent_dimension"})
        if "HR" in surv.columns:
            surv["HR"] = pd.to_numeric(surv["HR"], errors="coerce")
            surv["log2_HR"] = np.log2(surv["HR"])
        scores = scores.merge(
            surv[[c for c in ["latent_dimension", "HR", "log2_HR", "FDR", "c_index"] if c in surv.columns]],
            on="latent_dimension",
            how="left"
        )

    # Ajouter associations cliniques clés
    if not clinical_assoc.empty:
        ca = clinical_assoc.copy()
        keep_vars = ["SUBTYPE", "IDH_like", "GBM_like", "codel_like", "grade_group"]
        ca = ca[ca["clinical_variable"].isin(keep_vars)].copy()
        piv = ca.pivot_table(
            index="latent_dimension",
            columns="clinical_variable",
            values="epsilon_squared",
            aggfunc="max"
        ).reset_index()
        piv.columns = [str(c) if c == "latent_dimension" else f"eps2_{c}" for c in piv.columns]
        scores = scores.merge(piv, on="latent_dimension", how="left")

    return scores


def build_axis_summary(axis_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis in AXES.keys():
        sub = axis_scores[axis_scores["axis"] == axis].copy()
        sub = sub.sort_values("axis_score", ascending=False)
        top = sub.head(5)

        rows.append({
            "axis": axis,
            "top_latent_dimensions": ";".join(top["latent_dimension"].tolist()),
            "top_latent_dimensions_with_scores": ";".join([
                f"{r['latent_dimension']}({r['axis_score']:.2f})" for _, r in top.iterrows()
            ]),
            "main_pathways": ";".join(
                [x for x in top["top_axis_pathways"].dropna().astype(str).tolist() if x != ""][:5]
            ),
            "mean_log2_HR_top_dims": top["log2_HR"].mean() if "log2_HR" in top.columns else np.nan,
        })
    return pd.DataFrame(rows)


# ============================================================
# FIGURES
# ============================================================
def plot_four_axes_heatmap(axis_scores: pd.DataFrame) -> None:
    mat = axis_scores.pivot_table(
        index="latent_dimension",
        columns="axis",
        values="axis_score",
        aggfunc="mean"
    ).reindex(LATENT_DIMS)

    mat = mat[[a for a in AXES.keys() if a in mat.columns]]

    plt.figure(figsize=(11, 8))
    im = plt.imshow(mat.values, aspect="auto")
    plt.colorbar(im, label="Composite biological axis score")
    plt.xticks(range(mat.shape[1]), mat.columns, rotation=35, ha="right")
    plt.yticks(range(mat.shape[0]), mat.index)
    plt.title("DeepGliomaAI latent dimensions mapped to four biological axes")
    plot_and_save(os.path.join(FIGDIR, "final_four_axes_heatmap.png"))


def plot_four_axes_with_survival(axis_scores: pd.DataFrame) -> None:
    if "log2_HR" not in axis_scores.columns:
        return

    # une figure compacte : pour chaque axe, top 5 dims et leur log2HR
    rows = []
    for axis in AXES.keys():
        sub = axis_scores[axis_scores["axis"] == axis].sort_values("axis_score", ascending=False).head(5)
        for _, r in sub.iterrows():
            rows.append({
                "axis": axis,
                "latent_dimension": r["latent_dimension"],
                "axis_score": r["axis_score"],
                "log2_HR": r.get("log2_HR", np.nan),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    labels = [f"{r['axis']}\n{r['latent_dimension']}" for _, r in df.iterrows()]
    x = np.arange(len(df))

    plt.figure(figsize=(14, 6))
    plt.bar(x, df["log2_HR"])
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels, rotation=75, ha="right", fontsize=8)
    plt.ylabel("log2(Hazard Ratio)")
    plt.title("Survival direction of the top latent dimensions within each biological axis")
    plot_and_save(os.path.join(FIGDIR, "final_four_axes_survival_direction.png"))


def plot_final_conceptual_panel(axis_summary: pd.DataFrame) -> None:
    """Figure texte simple utilisable dans le mémoire/README."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    title = "DeepGliomaAI: four interpretable latent biological axes"
    ax.text(0.5, 0.95, title, ha="center", va="center", fontsize=18, fontweight="bold")

    y_positions = [0.75, 0.55, 0.35, 0.15]
    for y, (_, row) in zip(y_positions, axis_summary.iterrows()):
        axis = row["axis"]
        dims = row["top_latent_dimensions_with_scores"]
        hr = row.get("mean_log2_HR_top_dims", np.nan)
        direction = "worse survival" if pd.notna(hr) and hr > 0 else "better/less aggressive survival" if pd.notna(hr) else "survival unclear"

        text = f"{axis}\nTop latent dimensions: {dims}\nSurvival tendency: {direction}"
        ax.text(
            0.05, y, text,
            ha="left", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", alpha=0.15)
        )

    plot_and_save(os.path.join(FIGDIR, "final_four_axes_conceptual_panel.png"))


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== 09_fix_gsea_and_final_axes.py ===")

    print("[1/7] Loading latent-gene correlations...")
    if not os.path.exists(CORR_PATH):
        raise FileNotFoundError(f"Missing correlation file: {CORR_PATH}")
    corr_df = pd.read_csv(CORR_PATH)
    required = {"latent_dimension", "gene", "spearman_rho"}
    if not required.issubset(set(corr_df.columns)):
        raise ValueError(f"Correlation file must contain columns: {required}")
    corr_df = corr_df[corr_df["latent_dimension"].isin(LATENT_DIMS)].copy()

    print("[2/7] Finding GMT file...")
    gmt = find_gmt()
    if gmt is None:
        raise FileNotFoundError(
            "No GMT file found. Place an MSigDB/Hallmark .gmt file in data/ or results/04_latent_biology/."
        )
    print(f"GMT used: {gmt}")

    print("[3/7] Re-running GSEA correctly for all VAE dimensions...")
    gsea = run_gsea_all_dimensions(corr_df, gmt)
    if gsea.empty:
        raise RuntimeError("GSEA returned empty results.")

    # Vérification anti-bug prerank
    if gsea["pathway"].str.lower().eq("prerank").any():
        raise RuntimeError("GSEA still contains pathway='prerank'. Check gseapy output columns.")

    save_csv(gsea, "latent_dimension_gsea_all_results_FIXED.csv")
    gsea_summary = summarize_gsea(gsea, top_n=10)
    save_csv(gsea_summary, "latent_dimension_pathway_annotation_FIXED.csv")

    print("[4/7] Loading annotation inputs...")
    biomarker_corr = pd.read_csv(BIOMARKER_CORR_PATH) if os.path.exists(BIOMARKER_CORR_PATH) else pd.DataFrame()
    clinical_assoc = pd.read_csv(CLINICAL_ASSOC_PATH) if os.path.exists(CLINICAL_ASSOC_PATH) else pd.DataFrame()
    survival = pd.read_csv(SURVIVAL_PATH) if os.path.exists(SURVIVAL_PATH) else pd.DataFrame()
    top_genes = pd.read_csv(TOP_GENES_PATH) if os.path.exists(TOP_GENES_PATH) else pd.DataFrame()

    print("[5/7] Computing four biological axis scores...")
    axis_scores = compute_final_axis_scores(
        corr_df=corr_df,
        biomarker_corr=biomarker_corr,
        gsea=gsea,
        survival=survival,
        clinical_assoc=clinical_assoc,
    )
    save_csv(axis_scores, "final_four_axes_latent_dimension_scores.csv")

    axis_summary = build_axis_summary(axis_scores)
    save_csv(axis_summary, "final_four_axes_summary.csv")

    print("[6/7] Updating final annotation table with corrected GSEA...")
    # Table simple par dimension avec pathways + meilleur axe
    best_axis = axis_scores.sort_values("axis_score", ascending=False).groupby("latent_dimension").head(1)
    best_axis = best_axis[["latent_dimension", "axis", "axis_score"]].rename(
        columns={"axis": "dominant_biological_axis", "axis_score": "dominant_axis_score"}
    )

    final_dim = gsea_summary.merge(best_axis, on="latent_dimension", how="left")
    if not survival.empty and "latent_dim" in survival.columns:
        surv = survival.rename(columns={"latent_dim": "latent_dimension"}).copy()
        final_dim = final_dim.merge(
            surv[[c for c in ["latent_dimension", "HR", "FDR", "c_index"] if c in surv.columns]],
            on="latent_dimension",
            how="left"
        )
    save_csv(final_dim, "latent_dimension_final_annotation_GSEA_FIXED.csv")

    print("[7/7] Generating final figures...")
    plot_four_axes_heatmap(axis_scores)
    plot_four_axes_with_survival(axis_scores)
    plot_final_conceptual_panel(axis_summary)

    summary = {
        "gmt_used": gmt,
        "n_gsea_rows": int(gsea.shape[0]),
        "n_latent_dimensions": int(gsea["latent_dimension"].nunique()),
        "outputs_dir": OUTDIR,
        "figures_dir": FIGDIR,
        "main_outputs": [
            "latent_dimension_gsea_all_results_FIXED.csv",
            "latent_dimension_pathway_annotation_FIXED.csv",
            "final_four_axes_latent_dimension_scores.csv",
            "final_four_axes_summary.csv",
            "latent_dimension_final_annotation_GSEA_FIXED.csv",
            "final_four_axes_heatmap.png",
            "final_four_axes_survival_direction.png",
            "final_four_axes_conceptual_panel.png",
        ],
    }
    with open(os.path.join(OUTDIR, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Results saved in: {OUTDIR}")
    print(f"Figures saved in: {FIGDIR}")


if __name__ == "__main__":
    main()
