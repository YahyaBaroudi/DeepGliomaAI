# scripts/06_supervised_models.py

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/mnt/d/projects/DeepGliomaAI"

EXPR_PATH = os.path.join(BASE_DIR, "data", "expr_mat_log2.csv")
CLIN_PATH = os.path.join(BASE_DIR, "data", "clinical_matched.csv")
LATENT_PATH = os.path.join(BASE_DIR, "results", "02_vae", "vae_latent_space.csv")

OUTDIR = os.path.join(BASE_DIR, "results", "06_supervised_models")
FIGDIR = os.path.join(BASE_DIR, "figures", "06_supervised_models")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

TARGET_COL = "SUBTYPE"
N_TOP_GENES = 2000
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# UTILS
# ============================================================

def clean_sample_id(x):
    x = str(x).replace(".", "-").strip()

    # TCGA sample ID -> patient ID
    # TCGA-CS-4938-01 -> TCGA-CS-4938
    parts = x.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])

    return x

def load_expression(path):
    expr = pd.read_csv(path, index_col=0)

    # genes x samples -> samples x genes
    if expr.shape[0] > expr.shape[1]:
        print("Expression appears to be genes x samples. Transposing...")
        expr = expr.T

    expr.index = [clean_sample_id(i) for i in expr.index]
    expr = expr.groupby(expr.index).mean()  # si plusieurs aliquots par patient

    expr = expr.apply(pd.to_numeric, errors="coerce")
    return expr

def load_clinical(path):
    clin = pd.read_csv(path)
    clin["PATIENT_ID"] = clin["PATIENT_ID"].apply(clean_sample_id)
    clin = clin.set_index("PATIENT_ID")
    return clin


def load_latent(path):
    latent = pd.read_csv(path, index_col=0)
    latent.index = [clean_sample_id(i) for i in latent.index]
    latent = latent.groupby(latent.index).mean()
    latent = latent.apply(pd.to_numeric, errors="coerce")
    return latent

def align_data(expr, clin, latent):
    common = sorted(set(expr.index) & set(clin.index) & set(latent.index))

    if len(common) == 0:
        print("Expression examples:", expr.index[:5].tolist())
        print("Clinical examples:", clin.index[:5].tolist())
        print("Latent examples:", latent.index[:5].tolist())
        raise ValueError("No common samples between expression, clinical and latent data.")

    expr = expr.loc[common]
    clin = clin.loc[common]
    latent = latent.loc[common]

    print(f"Common samples: {len(common)}")
    print(f"Matched expression: {expr.shape}")
    print(f"Matched clinical: {clin.shape}")
    print(f"Matched latent: {latent.shape}")

    return expr, clin, latent


def prepare_target(clin, target_col):
    y_raw = clin[target_col].astype(str)

    y_raw = y_raw.replace({
        "nan": np.nan,
        "None": np.nan,
        "NA": np.nan,
        "": np.nan
    })

    keep = y_raw.notna()
    y_raw = y_raw[keep]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print("\nTarget distribution:")
    print(y_raw.value_counts())

    print("\nEncoded labels:")
    for cls, code in zip(le.classes_, le.transform(le.classes_)):
        print(f"{cls} -> {code}")

    return y_raw.index, y, le


def select_top_variable_genes(expr, n_top=2000):
    variances = expr.var(axis=0).sort_values(ascending=False)
    top_genes = variances.head(n_top).index.tolist()

    selected = expr[top_genes].copy()

    pd.DataFrame({
        "gene": variances.index,
        "variance": variances.values
    }).to_csv(os.path.join(OUTDIR, "gene_variance_ranking.csv"), index=False)

    pd.Series(top_genes, name="top_variable_genes").to_csv(
        os.path.join(OUTDIR, "selected_top_variable_genes.csv"),
        index=False
    )

    return selected


def short_model_name(name):
    name = name.replace("Expression_top_variable_genes_", "Expr ")
    name = name.replace("VAE_latent_space_", "Latent ")
    name = name.replace("ElasticNet_LogisticRegression", "ElasticNet")
    name = name.replace("RandomForest", "RF")
    return name


def plot_confusion_matrix_clean(cm, labels, model_name):
    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        ax=ax,
        cmap="viridis",
        colorbar=True,
        values_format="d"
    )

    ax.set_title(short_model_name(model_name), fontsize=14, pad=14)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    ax.tick_params(axis="x", labelrotation=40, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    plt.tight_layout()

    outpath = os.path.join(FIGDIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results_df):
    df = results_df.copy()
    df["short_name"] = df["model"].apply(short_model_name)
    df = df.sort_values("F1_macro", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(df["short_name"], df["F1_macro"])

    ax.set_title("Supervised model comparison", fontsize=16, pad=15)
    ax.set_ylabel("Macro-F1 score", fontsize=13)
    ax.set_ylim(0, 1.05)

    ax.tick_params(axis="x", labelrotation=30, labelsize=10)
    ax.tick_params(axis="y", labelsize=11)

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGDIR, "supervised_model_comparison_clean.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, class_names):
    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    auc = np.nan
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            auc = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="macro"
            )
        except Exception:
            auc = np.nan

    print(
        f"{model_name} | "
        f"ACC={acc:.3f} | "
        f"F1_macro={f1_macro:.3f} | "
        f"F1_weighted={f1_weighted:.3f} | "
        f"AUC={auc:.3f}"
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).T
    report_df.to_csv(os.path.join(OUTDIR, f"classification_report_{model_name}.csv"))

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(OUTDIR, f"confusion_matrix_{model_name}.csv")
    )

    plot_confusion_matrix_clean(cm, class_names, model_name)

    return {
        "model": model_name,
        "accuracy": acc,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
        "AUC": auc
    }


def cross_validate_model(model_name, model, X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    print(
        f"{model_name} CV F1_macro = "
        f"{scores.mean():.3f} ± {scores.std():.3f}"
    )

    return scores.mean(), scores.std()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("Loading data")
    print("=" * 80)

    expr = load_expression(EXPR_PATH)
    clin = load_clinical(CLIN_PATH)
    latent = load_latent(LATENT_PATH)

    print(f"Raw expression shape: {expr.shape}")
    print(f"Clinical shape: {clin.shape}")
    print(f"Latent shape: {latent.shape}")

    expr, clin, latent = align_data(expr, clin, latent)

    sample_ids, y, le = prepare_target(clin, TARGET_COL)

    expr = expr.loc[sample_ids]
    latent = latent.loc[sample_ids]

    class_names = list(le.classes_)

    print(f"\nFinal matched expression: {expr.shape}")
    print(f"Final matched latent: {latent.shape}")
    print(f"Final y: {y.shape}")

    X_expr = select_top_variable_genes(expr, N_TOP_GENES)
    X_latent = latent.copy()

    feature_sets = {
        "Expression_top_variable_genes": X_expr,
        "VAE_latent_space": X_latent
    }

    all_results = []
    cv_results = []

    for feature_set_name, X in feature_sets.items():
        print("\n" + "=" * 80)
        print(f"FEATURE SET: {feature_set_name}")
        print("=" * 80)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        models = {
            f"{feature_set_name}_ElasticNet_LogisticRegression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                ))
            ]),

            f"{feature_set_name}_RandomForest": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ))
            ])
        }

        for model_name, model in models.items():
            result = evaluate_model(
                model_name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                class_names
            )

            cv_mean, cv_std = cross_validate_model(
                model_name,
                model,
                X,
                y
            )

            result["CV_F1_macro_mean"] = cv_mean
            result["CV_F1_macro_std"] = cv_std

            all_results.append(result)

            cv_results.append({
                "model": model_name,
                "CV_F1_macro_mean": cv_mean,
                "CV_F1_macro_std": cv_std
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTDIR, "supervised_model_results.csv"), index=False)

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(OUTDIR, "supervised_model_cv_results.csv"), index=False)

    plot_model_comparison(results_df)

    print("\nDone.")
    print(f"Results saved in: {OUTDIR}")
    print(f"Figures saved in: {FIGDIR}")


if __name__ == "__main__":
    main()
