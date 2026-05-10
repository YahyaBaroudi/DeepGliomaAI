import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


# =========================
# PATHS
# =========================
BASE_DIR = "/mnt/d/projects/DeepGliomaAI"

LATENT_PATH = os.path.join(BASE_DIR, "results/02_vae/vae_latent_space.csv")
CLIN_PATH = os.path.join(BASE_DIR, "data/clinical_matched.csv")

OUTDIR = os.path.join(BASE_DIR, "results/03_vae_latent_survival")
FIGDIR = os.path.join(BASE_DIR, "figures/03_vae_latent_survival")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


# =========================
# HELPERS
# =========================
def clean_id(x):
    return str(x).strip().replace(".", "-")


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def zscore_df(df):
    scaler = StandardScaler()
    arr = scaler.fit_transform(df)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def fit_cox(df, duration_col, event_col, features):
    cox_df = df[[duration_col, event_col] + features].dropna().copy()
    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(cox_df, duration_col=duration_col, event_col=event_col)
    return cph, cox_df


def plot_km(df, time_col, event_col, risk_col, title, outpath):
    d = df[[time_col, event_col, risk_col]].dropna().copy()
    d["risk_group"] = np.where(
        d[risk_col] >= d[risk_col].median(),
        "High latent risk",
        "Low latent risk"
    )

    high = d[d["risk_group"] == "High latent risk"]
    low = d[d["risk_group"] == "Low latent risk"]

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(8, 6))

    kmf.fit(low[time_col], low[event_col], label="Low latent risk")
    ax = kmf.plot(ci_show=True)

    kmf.fit(high[time_col], high[event_col], label="High latent risk")
    kmf.plot(ax=ax, ci_show=True)

    res = logrank_test(
        high[time_col], low[time_col],
        event_observed_A=high[event_col],
        event_observed_B=low[event_col]
    )

    plt.title(title)
    plt.xlabel("Time (months)")
    plt.ylabel("Survival probability")
    plt.text(0.05, 0.10, f"log-rank p = {res.p_value:.2e}", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    return res.p_value


def cross_validated_cindex(df, time_col, event_col, features, n_splits=5):
    d = df[[time_col, event_col] + features].dropna().copy()
    X = d[features]
    y_time = d[time_col].values
    y_event = d[event_col].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cindices = []

    for train_idx, test_idx in kf.split(X):
        train = d.iloc[train_idx].copy()
        test = d.iloc[test_idx].copy()

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train[[time_col, event_col] + features],
                duration_col=time_col,
                event_col=event_col)

        risk = cph.predict_partial_hazard(test[features]).values.ravel()

        ci = concordance_index(
            test[time_col],
            -risk,
            test[event_col]
        )
        cindices.append(ci)

    return np.array(cindices)


# =========================
# LOAD DATA
# =========================
print("=" * 60)
print("Loading data")
print("=" * 60)

latent = pd.read_csv(LATENT_PATH)
clin = pd.read_csv(CLIN_PATH)

print("Latent shape:", latent.shape)
print("Clinical shape:", clin.shape)

id_col_latent = find_col(latent, ["PATIENT_ID", "patient_id", "sample_id", "Sample", "sample"])
id_col_clin = find_col(clin, ["PATIENT_ID", "patient_id", "sample_id", "Sample", "sample"])

if id_col_latent is None:
    latent = latent.rename(columns={latent.columns[0]: "PATIENT_ID"})
    id_col_latent = "PATIENT_ID"

if id_col_clin is None:
    raise ValueError("No patient ID column found in clinical file.")

latent[id_col_latent] = latent[id_col_latent].apply(clean_id)
clin[id_col_clin] = clin[id_col_clin].apply(clean_id)

latent = latent.rename(columns={id_col_latent: "PATIENT_ID"})
clin = clin.rename(columns={id_col_clin: "PATIENT_ID"})

df = latent.merge(clin, on="PATIENT_ID", how="inner")

print("Merged shape:", df.shape)

time_col = find_col(df, ["OS_MONTHS", "OS_months", "os_months", "OS.time"])
event_col = find_col(df, ["OS_event", "OS_EVENT", "os_event"])

if time_col is None or event_col is None:
    raise ValueError("OS time/event columns not found.")

df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

latent_cols = [c for c in df.columns if c.startswith("VAE_Z")]

if len(latent_cols) == 0:
    raise ValueError("No VAE_Z latent columns found.")

print("Latent dimensions:", len(latent_cols))


# =========================
# STANDARDIZE LATENT FEATURES
# =========================
df[latent_cols] = zscore_df(df[latent_cols])


# =========================
# UNIVARIATE COX FOR EACH LATENT DIM
# =========================
print("=" * 60)
print("Univariate Cox on latent dimensions")
print("=" * 60)

uni_results = []

for z in latent_cols:
    tmp = df[[time_col, event_col, z]].dropna().copy()

    if tmp[z].std() == 0:
        continue

    try:
        cph = CoxPHFitter()
        cph.fit(tmp, duration_col=time_col, event_col=event_col)

        row = cph.summary.loc[z]
        uni_results.append({
            "latent_dim": z,
            "coef": row["coef"],
            "HR": np.exp(row["coef"]),
            "p": row["p"],
            "c_index": cph.concordance_index_
        })

    except Exception as e:
        print(f"Skipped {z}: {e}")

uni = pd.DataFrame(uni_results)
uni["FDR"] = pd.Series(uni["p"]).rank(method="first") / len(uni) * 0
uni["FDR"] = np.minimum(1, uni["p"] * len(uni) / uni["p"].rank(method="first"))

uni = uni.sort_values("p")
uni.to_csv(os.path.join(OUTDIR, "univariate_cox_latent_dims.csv"), index=False)

print(uni.head(20))


# =========================
# SELECT LATENT DIMENSIONS
# =========================
selected = uni.query("p < 0.01").head(5)["latent_dim"].tolist()

if len(selected) < 3:
    selected = uni.head(10)["latent_dim"].tolist()

print("Selected latent dimensions:", selected)


# =========================
# MULTIVARIATE COX
# =========================
print("=" * 60)
print("Multivariate Cox latent survival model")
print("=" * 60)

cox_features = selected.copy()

candidate_covars = ["AGE"]
available_covars = [c for c in candidate_covars if c in df.columns]

cox_df = df[[time_col, event_col] + cox_features + available_covars].copy()

for c in available_covars:
    if c in ["SEX", "CANCER_TYPE_ACRONYM", "grade_group", "SUBTYPE"]:
        cox_df[c] = cox_df[c].astype(str)

cox_df = pd.get_dummies(cox_df, columns=[c for c in available_covars if c != "AGE"], drop_first=True)

cox_df = cox_df.replace([np.inf, -np.inf], np.nan).dropna()

cph = CoxPHFitter(penalizer=1.0)
cph.fit(cox_df, duration_col=time_col, event_col=event_col)

cph.summary.to_csv(os.path.join(OUTDIR, "multivariate_cox_latent_model.csv"))

print(cph.summary)

risk_score = cph.predict_partial_hazard(cox_df.drop(columns=[time_col, event_col]))
cox_df["latent_risk_score"] = risk_score.values.ravel()

cox_df[["latent_risk_score", time_col, event_col]].to_csv(
    os.path.join(OUTDIR, "latent_risk_scores.csv"),
    index=False
)


# =========================
# C-INDEX
# =========================
risk = cox_df["latent_risk_score"].values
cindex = concordance_index(
    cox_df[time_col],
    -risk,
    cox_df[event_col]
)

print(f"Apparent C-index: {cindex:.3f}")

cv_cindex = cross_validated_cindex(
    cox_df,
    time_col,
    event_col,
    [c for c in cox_df.columns if c not in [time_col, event_col, "latent_risk_score"]],
    n_splits=5
)

pd.DataFrame({
    "fold": range(1, len(cv_cindex) + 1),
    "c_index": cv_cindex
}).to_csv(os.path.join(OUTDIR, "cross_validated_cindex.csv"), index=False)

print(f"CV C-index: {cv_cindex.mean():.3f} ± {cv_cindex.std():.3f}")


# =========================
# KM CURVE
# =========================
print("=" * 60)
print("Kaplan-Meier by latent risk score")
print("=" * 60)

pval = plot_km(
    cox_df,
    time_col,
    event_col,
    "latent_risk_score",
    "Overall survival by VAE latent risk score",
    os.path.join(FIGDIR, "KM_latent_risk_score.png")
)

print(f"KM log-rank p = {pval:.2e}")


# =========================
# COX COEFFICIENT PLOT
# =========================
summary = cph.summary.copy()
summary["feature"] = summary.index
summary = summary.sort_values("coef")

plt.figure(figsize=(8, max(6, 0.25 * len(summary))))
plt.barh(summary["feature"], summary["coef"])
plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("Cox coefficient")
plt.title("Multivariate Cox coefficients - VAE latent survival model")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "cox_coefficients_latent_model.png"), dpi=300)
plt.close()


# =========================
# SAVE SUMMARY
# =========================
summary_txt = f"""
VAE latent survival model summary

Samples used: {cox_df.shape[0]}
Number of selected latent dimensions: {len(selected)}
Selected latent dimensions: {selected}

Apparent C-index: {cindex:.3f}
Cross-validated C-index: {cv_cindex.mean():.3f} ± {cv_cindex.std():.3f}

KM log-rank p-value: {pval:.3e}

Interpretation:
The VAE-derived latent variables were used as compressed transcriptomic features in a Cox proportional hazards model.
A significant Kaplan-Meier separation and a good cross-validated C-index suggest that the latent representation captures prognostic transcriptomic structure.
"""

with open(os.path.join(OUTDIR, "summary.txt"), "w") as f:
    f.write(summary_txt)

print("=" * 60)
print("Done.")
print("Results saved in:", OUTDIR)
print("Figures saved in:", FIGDIR)
print("=" * 60)
