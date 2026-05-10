import os
import re
import glob
import json
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kruskal

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

LATENT_GENE_CORR_PATH = os.path.join(
    RESULTS_DIR,
    "04_latent_biology",
    "all_latent_gene_correlations.csv"
)

LATENT_SURVIVAL_PATH = os.path.join(
    RESULTS_DIR,
    "03_vae_latent_survival",
    "univariate_cox_latent_dims.csv"
)

LATENT_COEF_PATH = os.path.join(
    RESULTS_DIR,
    "07_model_interpretation",
    "latent_logistic_coefficients.csv"
)

OUTDIR = os.path.join(RESULTS_DIR, "08_latent_biological_annotation")
FIGDIR = os.path.join(FIGURES_DIR, "08_latent_biological_annotation")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

LATENT_DIMS = [f"VAE_Z{i}" for i in range(1, 21)]

BIOMARKER_GENES = [
    "IDH1", "IDH2", "ATRX", "TP53", "TERT", "EGFR", "CDKN2A", "CDKN2B",
    "CIC", "FUBP1", "MGMT", "PTEN", "NF1", "RB1", "MDM2", "MDM4",
    "OLIG2", "SOX2", "SOX9", "GFAP", "AQP4", "NES", "VIM",
    "MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "BIRC5", "AURKA", "AURKB",
    "CHI3L1", "IGFBP2", "POSTN", "COL1A1", "COL1A2", "FN1", "VEGFA", "CA9",
    "CD3D", "CD3E", "CD8A", "CD4", "PTPRC", "HLA-DRA", "HLA-A",
    "CXCL10", "CXCL9", "STAT1", "IRF1", "CD274", "PDCD1", "CTLA4",
    "AIF1", "C1QA", "C1QB", "C1QC", "ITGAM",
    "DLL3", "ASCL1", "DCX", "MAP2", "SYT1", "SNAP25", "NEFL", "NEFM",
    "UQCRBP1", "UQCRB", "NACA2", "C1orf51", "CIART"
]


PROGRAM_RULES = {
    "Cell cycle / proliferation": [
        "E2F", "G2M", "MITOTIC", "MYC", "DNA REPAIR",
        "MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "BIRC5", "AURKA", "AURKB"
    ],
    "Immune / interferon / inflammation": [
        "INTERFERON", "INFLAMMATORY", "TNFA", "NF-KB", "COMPLEMENT",
        "IL6", "JAK", "STAT", "HLA", "CXCL", "CD3", "CD8",
        "PTPRC", "CD274", "AIF1", "C1QA", "C1QB", "C1QC"
    ],
    "Mesenchymal / ECM / invasion": [
        "EPITHELIAL MESENCHYMAL", "ANGIOGENESIS", "HYPOXIA",
        "COL1A1", "COL1A2", "POSTN", "FN1", "VIM", "CHI3L1", "IGFBP2",
        "VEGFA", "CA9"
    ],
    "Neuronal / proneural differentiation": [
        "NEURON", "SYNAP", "OLIG2", "SOX2", "DLL3", "ASCL1", "DCX",
        "MAP2", "SYT1", "SNAP25", "NEFL", "NEFM"
    ],
    "Metabolism / mitochondrial / oxidative": [
        "OXIDATIVE PHOSPHORYLATION", "GLYCOLYSIS", "MTORC1", "FATTY ACID",
        "UQCR", "NDUF", "COX", "ATP5"
    ],
    "p53 / stress / apoptosis": [
        "P53", "APOPTOSIS", "UV RESPONSE", "REACTIVE OXYGEN",
        "TP53", "CDKN2A", "CDKN2B"
    ],
    "RTK / EGFR-like": [
        "EGFR", "KRAS", "PI3K", "AKT", "MTOR", "PTEN", "PDGF", "MAPK"
    ]
}


# ============================================================
# HELPERS
# ============================================================
def clean_patient_id(x):
    return str(x).replace(".", "-")[:12]


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def save_csv(df, filename, index=False):
    df.to_csv(os.path.join(OUTDIR, filename), index=index)


def bh_fdr(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)

    order = np.argsort(pvalues)
    ranked = pvalues[order]

    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty_like(q)
    out[order] = q

    return out


def fdr_correction(df, p_col="p_value", out_col="FDR"):
    df = df.copy()
    df[out_col] = np.nan

    valid = df[p_col].notna()
    if valid.sum() > 0:
        df.loc[valid, out_col] = bh_fdr(df.loc[valid, p_col].values)

    return df


def plot_and_save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def load_expression(path):
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


def load_clinical(path):
    clin = pd.read_csv(path)
    clin.columns = [c.strip() for c in clin.columns]

    if "PATIENT_ID" not in clin.columns:
        raise ValueError("PATIENT_ID absent du fichier clinique.")

    clin["PATIENT_ID"] = clin["PATIENT_ID"].apply(clean_patient_id)
    clin = clin.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")

    for c in ["OS_MONTHS", "OS_event", "AGE", "PFS_MONTHS", "PFS_event"]:
        if c in clin.columns:
            clin[c] = pd.to_numeric(clin[c], errors="coerce")

    return clin


def load_latent(path):
    latent = pd.read_csv(path)

    first_col = latent.columns[0]
    if first_col not in LATENT_DIMS:
        latent = latent.rename(columns={first_col: "PATIENT_ID"})
    else:
        raise ValueError("Le fichier latent ne contient pas de colonne patient.")

    latent["PATIENT_ID"] = latent["PATIENT_ID"].apply(clean_patient_id)
    latent = latent.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")

    keep = [c for c in LATENT_DIMS if c in latent.columns]
    latent = latent[keep].copy()

    return latent


# ============================================================
# GSEA
# ============================================================
def normalize_gsea_columns(gsea):
    if gsea.empty:
        return gsea

    gsea = gsea.copy()
    gsea = gsea.loc[:, ~gsea.columns.duplicated()].copy()

    rename_map = {}

    for c in gsea.columns:
        lc = c.lower().strip()

        if lc in ["term", "terms", "gene_set", "geneset", "name"]:
            rename_map[c] = "pathway"
        elif lc in ["nes", "normalized_enrichment_score"]:
            rename_map[c] = "NES"
        elif "fdr" in lc:
            rename_map[c] = "FDR"
        elif lc in ["pval", "p-value", "p_value", "nom p-val", "nom_p_val"]:
            rename_map[c] = "p_value"

    gsea = gsea.rename(columns=rename_map)
    gsea = gsea.loc[:, ~gsea.columns.duplicated()].copy()

    if "pathway" not in gsea.columns:
        possible = [
            c for c in gsea.columns
            if "term" in c.lower() or "name" in c.lower()
        ]
        gsea["pathway"] = gsea[possible[0]] if possible else np.nan

    if "NES" not in gsea.columns:
        possible = [c for c in gsea.columns if c.lower() == "nes"]
        gsea["NES"] = gsea[possible[0]] if possible else np.nan

    if "FDR" not in gsea.columns:
        possible = [c for c in gsea.columns if "fdr" in c.lower()]
        gsea["FDR"] = gsea[possible[0]] if possible else np.nan

    gsea["pathway"] = gsea["pathway"].astype(str)
    gsea["NES"] = pd.to_numeric(gsea["NES"], errors="coerce")
    gsea["FDR"] = pd.to_numeric(gsea["FDR"], errors="coerce")

    return gsea


def find_hallmark_gmt():
    candidates = glob.glob(
        os.path.join(RESULTS_DIR, "04_latent_biology", "VAE_Z*_gsea", "gene_sets.gmt")
    )
    return candidates[0] if len(candidates) > 0 else None


def run_prerank_gsea_for_dimension(dim, corr_df, gmt_path):
    if gmt_path is None:
        return pd.DataFrame()

    try:
        import gseapy as gp
    except Exception:
        print("[WARNING] gseapy non installé. GSEA ignorée.")
        return pd.DataFrame()

    sub = corr_df[corr_df["latent_dimension"] == dim].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub[["gene", "spearman_rho"]].dropna()
    sub = sub.sort_values("spearman_rho", ascending=False)

    rnk_path = os.path.join(OUTDIR, f"{dim}_ranked_genes.rnk")
    sub.to_csv(rnk_path, sep="\t", index=False, header=False)

    gsea_out = os.path.join(OUTDIR, f"{dim}_gsea")
    os.makedirs(gsea_out, exist_ok=True)

    try:
        res = gp.prerank(
            rnk=rnk_path,
            gene_sets=gmt_path,
            outdir=gsea_out,
            permutation_num=1000,
            min_size=10,
            max_size=500,
            seed=42,
            verbose=False
        )

        out = res.res2d.copy()
        out["latent_dimension"] = dim
        return normalize_gsea_columns(out)

    except Exception as e:
        print(f"[WARNING] GSEA failed for {dim}: {e}")
        return pd.DataFrame()


def load_or_run_gsea(corr_df):
    existing = []

    for dim in LATENT_DIMS:
        patterns = [
            os.path.join(RESULTS_DIR, "04_latent_biology", f"{dim}_GSEA_results.csv"),
            os.path.join(RESULTS_DIR, "04_latent_biology", f"{dim}_gsea", "*.csv"),
        ]

        found = []
        for p in patterns:
            found += glob.glob(p)

        if len(found) > 0:
            try:
                tmp = pd.read_csv(found[0])
                tmp["latent_dimension"] = dim
                existing.append(tmp)
            except Exception:
                pass

    if len(existing) > 0:
        gsea = pd.concat(existing, ignore_index=True)
        gsea = normalize_gsea_columns(gsea)

        existing_dims = set(gsea["latent_dimension"].dropna().astype(str))
        missing_dims = [d for d in LATENT_DIMS if d not in existing_dims]

        print(f"[INFO] Existing GSEA found for {len(existing_dims)} dimensions.")

        if len(missing_dims) > 0:
            print(f"[INFO] Running missing GSEA for: {missing_dims}")
            gmt = find_hallmark_gmt()

            new_res = []
            for dim in missing_dims:
                res = run_prerank_gsea_for_dimension(dim, corr_df, gmt)
                if not res.empty:
                    new_res.append(res)

            if len(new_res) > 0:
                new_res = pd.concat(new_res, ignore_index=True)
                new_res = normalize_gsea_columns(new_res)
                gsea = pd.concat([gsea, new_res], ignore_index=True)

        return normalize_gsea_columns(gsea)

    print("[INFO] No existing GSEA found. Trying to run GSEA for all dimensions.")

    gmt = find_hallmark_gmt()
    all_res = []

    for dim in LATENT_DIMS:
        res = run_prerank_gsea_for_dimension(dim, corr_df, gmt)
        if not res.empty:
            all_res.append(res)

    if len(all_res) == 0:
        return pd.DataFrame()

    return normalize_gsea_columns(pd.concat(all_res, ignore_index=True))


def summarize_gsea(gsea, top_n=10):
    if gsea.empty:
        return pd.DataFrame()

    gsea = normalize_gsea_columns(gsea)

    rows = []

    for dim in LATENT_DIMS:
        sub = gsea[gsea["latent_dimension"] == dim].copy()

        if sub.empty:
            continue

        sub["NES"] = pd.to_numeric(sub["NES"], errors="coerce")
        sub["FDR"] = pd.to_numeric(sub["FDR"], errors="coerce")
        sub["pathway"] = sub["pathway"].astype(str)

        sig = sub[sub["FDR"] <= 0.25].copy()

        pos = (
            sig[sig["NES"] > 0]
            .sort_values(["FDR", "NES"], ascending=[True, False])
            .head(top_n)
        )

        neg = (
            sig[sig["NES"] < 0]
            .sort_values(["FDR", "NES"], ascending=[True, True])
            .head(top_n)
        )

        rows.append({
            "latent_dimension": dim,
            "top_positive_pathways": ";".join(pos["pathway"].astype(str).values.tolist()) if not pos.empty else "",
            "top_negative_pathways": ";".join(neg["pathway"].astype(str).values.tolist()) if not neg.empty else "",
            "n_sig_pathways_FDR_0_25": int(sig.shape[0]),
            "max_positive_NES": pos["NES"].max() if not pos.empty else np.nan,
            "max_negative_NES": neg["NES"].min() if not neg.empty else np.nan
        })

    return pd.DataFrame(rows)


# ============================================================
# ANALYSES
# ============================================================
def summarize_top_genes(corr_df, top_n=25):
    rows = []

    for dim in LATENT_DIMS:
        sub = corr_df[corr_df["latent_dimension"] == dim].copy()

        if sub.empty:
            continue

        pos = sub.sort_values("spearman_rho", ascending=False).head(top_n)
        neg = sub.sort_values("spearman_rho", ascending=True).head(top_n)

        rows.append({
            "latent_dimension": dim,
            "top_positive_genes": ";".join(pos["gene"].astype(str).tolist()),
            "top_negative_genes": ";".join(neg["gene"].astype(str).tolist()),
            "max_positive_rho": pos["spearman_rho"].max(),
            "max_negative_rho": neg["spearman_rho"].min()
        })

    return pd.DataFrame(rows)


def compute_biomarker_correlations(latent, expr):
    common = latent.index.intersection(expr.index)
    latent2 = latent.loc[common]
    expr2 = expr.loc[common]

    available = [g for g in BIOMARKER_GENES if g in expr2.columns]
    missing = [g for g in BIOMARKER_GENES if g not in expr2.columns]

    pd.DataFrame({"missing_biomarker_gene": missing}).to_csv(
        os.path.join(OUTDIR, "missing_biomarker_genes.csv"),
        index=False
    )

    rows = []

    for dim in latent2.columns:
        for gene in available:
            tmp = pd.DataFrame({
                "z": latent2[dim],
                "expr": expr2[gene]
            }).dropna()

            if len(tmp) < 10 or tmp["expr"].nunique() < 3:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(tmp["z"], tmp["expr"])

            rows.append({
                "latent_dimension": dim,
                "biomarker_gene": gene,
                "spearman_rho": rho,
                "p_value": p,
                "n": len(tmp)
            })

    out = pd.DataFrame(rows)
    out = fdr_correction(out, "p_value", "FDR")
    out["abs_rho"] = out["spearman_rho"].abs()

    return out.sort_values(["latent_dimension", "abs_rho"], ascending=[True, False])


def compute_clinical_associations(latent, clin):
    common = latent.index.intersection(clin.index)
    data = latent.loc[common].join(clin.loc[common], how="inner")

    if "SUBTYPE" in data.columns:
        subtype = data["SUBTYPE"].astype(str)

        data["IDH_like"] = np.where(
            subtype.str.contains("IDHmut", case=False, na=False),
            "IDHmut-like",
            np.where(
                subtype.str.contains("IDHwt", case=False, na=False),
                "IDHwt-like",
                "Unknown"
            )
        )

        data["codel_like"] = np.where(
            subtype.str.contains("codel", case=False, na=False)
            & ~subtype.str.contains("non-codel", case=False, na=False),
            "1p19q-codel-like",
            np.where(
                subtype.str.contains("non-codel", case=False, na=False),
                "non-codel-like",
                "Unknown"
            )
        )

        data["GBM_like"] = np.where(
            subtype.str.contains("GBM", case=False, na=False),
            "GBM-like",
            "non-GBM-like"
        )

    categorical_cols = [
        "SUBTYPE", "CANCER_TYPE_ACRONYM", "grade_group", "SEX",
        "OS_event", "PFS_event", "IDH_like", "codel_like", "GBM_like"
    ]

    categorical_cols = [c for c in categorical_cols if c in data.columns]

    rows = []

    for dim in latent.columns:
        for cat in categorical_cols:
            tmp = data[[dim, cat]].dropna()
            tmp = tmp[tmp[cat].astype(str) != "Unknown"]

            groups = []
            names = []

            for name, sub in tmp.groupby(cat):
                vals = sub[dim].dropna().values
                if len(vals) >= 5:
                    groups.append(vals)
                    names.append(str(name))

            if len(groups) >= 2:
                try:
                    H, p = kruskal(*groups)
                    n = sum(len(g) for g in groups)
                    k = len(groups)
                    eps2 = (H - k + 1) / (n - k) if n > k else np.nan
                except Exception:
                    H, p, eps2 = np.nan, np.nan, np.nan
            else:
                H, p, eps2 = np.nan, np.nan, np.nan

            row = {
                "latent_dimension": dim,
                "clinical_variable": cat,
                "groups": ";".join(names),
                "n_groups": len(groups),
                "kruskal_H": H,
                "p_value": p,
                "epsilon_squared": eps2
            }

            for name, vals in zip(names, groups):
                row[f"mean_{safe_name(name)}"] = float(np.mean(vals))

            rows.append(row)

    out = pd.DataFrame(rows)
    out = fdr_correction(out, "p_value", "FDR")

    return out.sort_values(["latent_dimension", "FDR"])


def load_latent_coefficients():
    if not os.path.exists(LATENT_COEF_PATH):
        return pd.DataFrame()

    coef = pd.read_csv(LATENT_COEF_PATH)

    rename = {}
    for c in coef.columns:
        lc = c.lower()
        if "class" in lc or "subtype" in lc:
            rename[c] = "class"
        elif "feature" in lc or "latent" in lc:
            rename[c] = "feature"
        elif "coef" in lc:
            rename[c] = "coefficient"

    coef = coef.rename(columns=rename)

    if not {"class", "feature", "coefficient"}.issubset(coef.columns):
        return pd.DataFrame()

    coef = coef[coef["feature"].isin(LATENT_DIMS)].copy()
    coef["abs_coefficient"] = coef["coefficient"].abs()

    return coef


def summarize_coefficients(coef):
    if coef.empty:
        return pd.DataFrame()

    rows = []

    for dim in LATENT_DIMS:
        sub = coef[coef["feature"] == dim].copy()

        if sub.empty:
            continue

        pos = sub.sort_values("coefficient", ascending=False).head(5)
        neg = sub.sort_values("coefficient", ascending=True).head(5)

        rows.append({
            "latent_dimension": dim,
            "top_positive_classes_by_coef": ";".join(
                [f"{r['class']}({r['coefficient']:.3f})" for _, r in pos.iterrows()]
            ),
            "top_negative_classes_by_coef": ";".join(
                [f"{r['class']}({r['coefficient']:.3f})" for _, r in neg.iterrows()]
            ),
            "max_abs_classifier_coefficient": sub["abs_coefficient"].max()
        })

    return pd.DataFrame(rows)


def infer_program_label(text):
    text = str(text).upper()

    scores = {}

    for label, keywords in PROGRAM_RULES.items():
        score = 0
        for kw in keywords:
            if kw.upper() in text:
                score += 1
        scores[label] = score

    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if len(best) == 0 or best[0][1] == 0:
        return "Unannotated / mixed latent program"

    labels = [label for label, score in best if score == best[0][1] and score > 0]

    return " + ".join(labels[:2])


def build_final_table(top_genes, gsea_summary, biomarker_corr, clinical_assoc, survival, coef_summary):
    rows = []

    for dim in LATENT_DIMS:
        row = {"latent_dimension": dim}

        tg = top_genes[top_genes["latent_dimension"] == dim]
        if not tg.empty:
            row.update(tg.iloc[0].to_dict())

        gs = gsea_summary[gsea_summary["latent_dimension"] == dim] if not gsea_summary.empty else pd.DataFrame()
        if not gs.empty:
            row.update(gs.iloc[0].to_dict())

        bc = biomarker_corr[biomarker_corr["latent_dimension"] == dim].copy()
        if not bc.empty:
            bc_top = bc.sort_values("abs_rho", ascending=False).head(10)
            row["top_correlated_biomarkers"] = ";".join([
                f"{r['biomarker_gene']}({r['spearman_rho']:.2f})"
                for _, r in bc_top.iterrows()
            ])
        else:
            row["top_correlated_biomarkers"] = ""

        ca = clinical_assoc[clinical_assoc["latent_dimension"] == dim].copy()
        if not ca.empty:
            ca_top = ca.sort_values("FDR").head(5)
            row["top_clinical_associations"] = ";".join([
                f"{r['clinical_variable']}(FDR={r['FDR']:.2e}, eps2={r['epsilon_squared']:.2f})"
                for _, r in ca_top.iterrows()
                if pd.notna(r["FDR"])
            ])
        else:
            row["top_clinical_associations"] = ""

        if not survival.empty and "latent_dim" in survival.columns:
            sv = survival[survival["latent_dim"] == dim].copy()
        else:
            sv = pd.DataFrame()

        if not sv.empty:
            row["survival_HR"] = sv.iloc[0].get("HR", np.nan)
            row["survival_p"] = sv.iloc[0].get("p", np.nan)
            row["survival_FDR"] = sv.iloc[0].get("FDR", np.nan)
            row["survival_c_index"] = sv.iloc[0].get("c_index", np.nan)
        else:
            row["survival_HR"] = np.nan
            row["survival_p"] = np.nan
            row["survival_FDR"] = np.nan
            row["survival_c_index"] = np.nan

        cs = coef_summary[coef_summary["latent_dimension"] == dim].copy() if not coef_summary.empty else pd.DataFrame()
        if not cs.empty:
            row.update(cs.iloc[0].to_dict())

        global_text = " ".join([
            row.get("top_positive_genes", ""),
            row.get("top_negative_genes", ""),
            row.get("top_positive_pathways", ""),
            row.get("top_negative_pathways", ""),
            row.get("top_correlated_biomarkers", "")
        ])

        row["inferred_biological_program"] = infer_program_label(global_text)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# FIGURES
# ============================================================
def plot_biomarker_heatmap(biomarker_corr):
    if biomarker_corr.empty:
        return

    selected = [
        "IDH1", "IDH2", "ATRX", "TP53", "TERT", "EGFR", "CDKN2A", "CDKN2B",
        "MGMT", "OLIG2", "SOX2", "GFAP", "MKI67", "TOP2A",
        "CHI3L1", "IGFBP2", "POSTN", "VEGFA", "CD274", "CXCL10",
        "HLA-DRA", "UQCRBP1", "NACA2"
    ]

    df = biomarker_corr[
        biomarker_corr["biomarker_gene"].isin(selected)
    ].pivot_table(
        index="latent_dimension",
        columns="biomarker_gene",
        values="spearman_rho",
        aggfunc="mean"
    )

    df = df.reindex(LATENT_DIMS)
    df = df[[c for c in selected if c in df.columns]]

    if df.empty:
        return

    plt.figure(figsize=(max(10, 0.5 * df.shape[1]), 8))
    im = plt.imshow(df.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, label="Spearman rho")
    plt.xticks(range(df.shape[1]), df.columns, rotation=90)
    plt.yticks(range(df.shape[0]), df.index)
    plt.title("Latent dimensions vs glioma biomarkers")
    plot_and_save(os.path.join(FIGDIR, "latent_biomarker_correlation_heatmap.png"))


def plot_clinical_heatmap(clinical_assoc):
    if clinical_assoc.empty:
        return

    df = clinical_assoc.pivot_table(
        index="latent_dimension",
        columns="clinical_variable",
        values="epsilon_squared",
        aggfunc="max"
    )

    df = df.reindex(LATENT_DIMS)

    if df.empty:
        return

    plt.figure(figsize=(max(8, 0.8 * df.shape[1]), 8))
    im = plt.imshow(df.values, aspect="auto")
    plt.colorbar(im, label="Epsilon squared")
    plt.xticks(range(df.shape[1]), df.columns, rotation=90)
    plt.yticks(range(df.shape[0]), df.index)
    plt.title("Latent dimensions vs clinical / molecular labels")
    plot_and_save(os.path.join(FIGDIR, "latent_clinical_association_heatmap.png"))


def plot_survival_bar(survival):
    if survival.empty:
        return

    if "latent_dim" not in survival.columns or "HR" not in survival.columns:
        return

    surv = survival[survival["latent_dim"].isin(LATENT_DIMS)].copy()
    surv["HR"] = pd.to_numeric(surv["HR"], errors="coerce")
    surv["log2_HR"] = np.log2(surv["HR"])
    surv = surv.sort_values("log2_HR")

    plt.figure(figsize=(8, 7))
    plt.barh(surv["latent_dim"], surv["log2_HR"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("log2(Hazard Ratio)")
    plt.title("Survival association of VAE latent dimensions")
    plot_and_save(os.path.join(FIGDIR, "latent_survival_log2HR.png"))


def plot_gsea_matrix(gsea):
    if gsea.empty:
        return

    gsea = normalize_gsea_columns(gsea)

    pathways = [
        "E2F", "G2M", "MYC", "INTERFERON", "INFLAMMATORY",
        "TNFA", "COMPLEMENT", "EPITHELIAL", "ANGIOGENESIS",
        "HYPOXIA", "GLYCOLYSIS", "OXIDATIVE", "MTORC1", "P53", "APOPTOSIS"
    ]

    rows = []

    for dim in LATENT_DIMS:
        sub = gsea[gsea["latent_dimension"] == dim].copy()

        for p in pathways:
            hit = sub[sub["pathway"].str.upper().str.contains(p, na=False)]
            if not hit.empty:
                best = hit.sort_values("FDR").iloc[0]
                rows.append({
                    "latent_dimension": dim,
                    "program": p,
                    "NES": best["NES"]
                })

    mat = pd.DataFrame(rows)
    if mat.empty:
        return

    df = mat.pivot_table(
        index="latent_dimension",
        columns="program",
        values="NES",
        aggfunc="mean"
    ).reindex(LATENT_DIMS)

    plt.figure(figsize=(max(10, 0.7 * df.shape[1]), 8))
    im = plt.imshow(df.values, aspect="auto")
    plt.colorbar(im, label="NES")
    plt.xticks(range(df.shape[1]), df.columns, rotation=90)
    plt.yticks(range(df.shape[0]), df.index)
    plt.title("GSEA program matrix across latent dimensions")
    plot_and_save(os.path.join(FIGDIR, "latent_gsea_program_matrix.png"))


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== 08_latent_biological_annotation.py ===")

    print("[1/9] Loading data...")
    latent = load_latent(LATENT_PATH)
    clin = load_clinical(CLIN_PATH)
    expr = load_expression(EXPR_PATH)

    common = latent.index.intersection(clin.index).intersection(expr.index)

    latent = latent.loc[common].copy()
    clin = clin.loc[common].copy()
    expr = expr.loc[common].copy()

    print(f"Patients used: {len(common)}")
    print(f"Latent dimensions: {latent.shape[1]}")
    print(f"Expression genes: {expr.shape[1]}")

    print("[2/9] Loading latent-gene correlations...")
    if not os.path.exists(LATENT_GENE_CORR_PATH):
        raise FileNotFoundError(f"Missing file: {LATENT_GENE_CORR_PATH}")

    corr_df = pd.read_csv(LATENT_GENE_CORR_PATH)

    required = {"latent_dimension", "gene", "spearman_rho"}
    if not required.issubset(set(corr_df.columns)):
        raise ValueError(f"Correlation file must contain: {required}")

    corr_df = corr_df[corr_df["latent_dimension"].isin(LATENT_DIMS)].copy()
    save_csv(corr_df, "all_latent_gene_correlations_used.csv")

    print("[3/9] Summarizing top positive/negative genes...")
    top_genes = summarize_top_genes(corr_df, top_n=25)
    save_csv(top_genes, "latent_top_positive_negative_genes.csv")

    print("[4/9] Computing biomarker correlations...")
    biomarker_corr = compute_biomarker_correlations(latent, expr)
    save_csv(biomarker_corr, "latent_dimension_biomarker_correlations.csv")

    print("[5/9] Computing clinical/subtype associations...")
    clinical_assoc = compute_clinical_associations(latent, clin)
    save_csv(clinical_assoc, "latent_dimension_clinical_subtype_associations.csv")

    print("[6/9] Loading or running GSEA...")
    gsea = load_or_run_gsea(corr_df)

    if not gsea.empty:
        save_csv(gsea, "latent_dimension_gsea_all_results.csv")
        gsea_summary = summarize_gsea(gsea, top_n=10)
        save_csv(gsea_summary, "latent_dimension_pathway_annotation.csv")
    else:
        print("[WARNING] No GSEA results available.")
        gsea_summary = pd.DataFrame()

    print("[7/9] Loading survival and classifier coefficients...")
    survival = pd.read_csv(LATENT_SURVIVAL_PATH) if os.path.exists(LATENT_SURVIVAL_PATH) else pd.DataFrame()
    if not survival.empty:
        save_csv(survival, "latent_dimension_survival_summary.csv")

    coef = load_latent_coefficients()
    if not coef.empty:
        save_csv(coef, "latent_dimension_classifier_coefficients.csv")
        coef_summary = summarize_coefficients(coef)
        save_csv(coef_summary, "latent_dimension_classifier_coefficient_summary.csv")
    else:
        coef_summary = pd.DataFrame()

    print("[8/9] Building final annotation table...")
    final_table = build_final_table(
        top_genes=top_genes,
        gsea_summary=gsea_summary,
        biomarker_corr=biomarker_corr,
        clinical_assoc=clinical_assoc,
        survival=survival,
        coef_summary=coef_summary
    )

    save_csv(final_table, "latent_dimension_final_annotation_table.csv")

    print("[9/9] Generating figures...")
    plot_biomarker_heatmap(biomarker_corr)
    plot_clinical_heatmap(clinical_assoc)
    plot_survival_bar(survival)
    plot_gsea_matrix(gsea)

    summary = {
        "n_patients": int(len(common)),
        "n_latent_dimensions": int(latent.shape[1]),
        "n_expression_genes": int(expr.shape[1]),
        "n_biomarker_genes_available": int(
            biomarker_corr["biomarker_gene"].nunique()
            if not biomarker_corr.empty else 0
        ),
        "n_gsea_rows": int(gsea.shape[0]) if not gsea.empty else 0,
        "results_dir": OUTDIR,
        "figures_dir": FIGDIR
    }

    with open(os.path.join(OUTDIR, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Results saved in: {OUTDIR}")
    print(f"Figures saved in: {FIGDIR}")


if __name__ == "__main__":
    main()
