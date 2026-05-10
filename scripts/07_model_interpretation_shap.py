# scripts/07_model_interpretation_shap.py

import os
import re
import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
BASE_DIR = "/mnt/d/projects/DeepGliomaAI"

EXPR_PATH = os.path.join(BASE_DIR, "data", "expr_mat_log2.csv")
CLIN_PATH = os.path.join(BASE_DIR, "data", "clinical_matched.csv")
LATENT_PATH = os.path.join(BASE_DIR, "results", "02_vae", "vae_latent_space.csv")

OUTDIR = os.path.join(BASE_DIR, "results", "07_model_interpretation")
FIGDIR = os.path.join(BASE_DIR, "figures", "07_model_interpretation")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

TARGET_COL = "SUBTYPE"
N_TOP_GENES = 1000
TEST_SIZE = 0.20
RANDOM_STATE = 42

TOP_N_COEFFICIENTS_PER_SIDE = 20
TOP_N_SHAP_DISPLAY = 25


# =========================
# HELPERS
# =========================
def clean_sample_id(x: str) -> str:
    """Convert sample/barcode IDs to TCGA patient-level IDs: TCGA-XX-YYYY."""
    x = str(x).replace(".", "-").strip()
    parts = x.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return x


def safe_filename(x: str) -> str:
    """Make class names safe for filenames."""
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    x = x.strip("_")
    return x or "unknown"


def save_plot(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def load_expression(path: str) -> pd.DataFrame:
    expr = pd.read_csv(path, index_col=0)

    # If rows are genes and columns are samples, transpose.
    # In TCGA expression matrices, there are usually many more genes than samples.
    if expr.shape[0] > expr.shape[1]:
        print("Expression appears to be genes x samples. Transposing...")
        expr = expr.T

    expr.index = [clean_sample_id(i) for i in expr.index]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.groupby(expr.index).mean(numeric_only=True)

    # Remove completely empty columns if present.
    expr = expr.loc[:, expr.notna().sum(axis=0) > 0]
    expr.index.name = "PATIENT_ID"

    return expr


def load_clinical(path: str) -> pd.DataFrame:
    clin = pd.read_csv(path)
    clin.columns = [c.strip() for c in clin.columns]

    if "PATIENT_ID" not in clin.columns:
        raise ValueError("PATIENT_ID column is missing from clinical file.")
    if TARGET_COL not in clin.columns:
        raise ValueError(f"{TARGET_COL} column is missing from clinical file.")

    clin["PATIENT_ID"] = clin["PATIENT_ID"].apply(clean_sample_id)
    clin = clin.drop_duplicates(subset="PATIENT_ID").set_index("PATIENT_ID")

    return clin


def load_latent(path: str) -> pd.DataFrame:
    latent = pd.read_csv(path, index_col=0)
    latent.index = [clean_sample_id(i) for i in latent.index]
    latent = latent.apply(pd.to_numeric, errors="coerce")
    latent = latent.groupby(latent.index).mean(numeric_only=True)
    latent.index.name = "PATIENT_ID"

    return latent


def align_data(
    expr: pd.DataFrame,
    clin: pd.DataFrame,
    latent: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = sorted(set(expr.index) & set(clin.index) & set(latent.index))

    if len(common) == 0:
        raise ValueError("No common patients found between expression, clinical and latent data.")

    return expr.loc[common], clin.loc[common], latent.loc[common]


def prepare_target(clin: pd.DataFrame) -> Tuple[pd.Index, np.ndarray, LabelEncoder, pd.Series]:
    y_raw = clin[TARGET_COL].astype(str).str.strip()
    y_raw = y_raw.replace({"nan": np.nan, "None": np.nan, "NA": np.nan, "": np.nan})
    keep = y_raw.notna()
    y_raw = y_raw[keep]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print("\nTarget distribution:")
    print(y_raw.value_counts())

    print("\nEncoded labels:")
    for cls, code in zip(le.classes_, le.transform(le.classes_)):
        print(f"{cls} -> {code}")

    target_map = pd.DataFrame({
        "class_name": le.classes_,
        "class_code": le.transform(le.classes_)
    })
    target_map.to_csv(os.path.join(OUTDIR, "target_label_encoding.csv"), index=False)

    return y_raw.index, y, le, y_raw


def select_top_variable_genes(expr: pd.DataFrame, n_top: int) -> pd.DataFrame:
    variances = expr.var(axis=0, skipna=True).sort_values(ascending=False)
    genes = variances.head(min(n_top, expr.shape[1])).index.tolist()

    pd.DataFrame({
        "gene": variances.index,
        "variance": variances.values
    }).to_csv(os.path.join(OUTDIR, "gene_variance_ranking.csv"), index=False)

    return expr[genes].copy()


def train_elasticnet_logistic_model(X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=10000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    prefix: str
) -> None:
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    metrics = {
        "prefix": prefix,
        "train_accuracy": float(accuracy_score(y_train, pred_train)),
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "train_balanced_accuracy": float(balanced_accuracy_score(y_train, pred_train)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
        "train_f1_macro": float(f1_score(y_train, pred_train, average="macro")),
        "test_f1_macro": float(f1_score(y_test, pred_test, average="macro")),
    }

    with open(os.path.join(OUTDIR, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_test,
        pred_test,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).T.to_csv(os.path.join(OUTDIR, f"{prefix}_classification_report_test.csv"))

    pred_df = pd.DataFrame({
        "PATIENT_ID": X_test.index,
        "true_class": [class_names[i] for i in y_test],
        "predicted_class": [class_names[i] for i in pred_test]
    })
    pred_df.to_csv(os.path.join(OUTDIR, f"{prefix}_test_predictions.csv"), index=False)

    print(f"\n{prefix} model test metrics:")
    print(f"  Accuracy          : {metrics['test_accuracy']:.3f}")
    print(f"  Balanced accuracy : {metrics['test_balanced_accuracy']:.3f}")
    print(f"  F1 macro          : {metrics['test_f1_macro']:.3f}")


def transformed_data(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]

    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=feature_names)

    return X_train_df, X_test_df


def logistic_coefficients(
    model: Pipeline,
    feature_names: List[str],
    class_names: List[str],
    prefix: str
) -> None:
    """Export class-specific ElasticNet logistic coefficients and make one plot per class."""
    clf = model.named_steps["clf"]

    coefs = pd.DataFrame(
        clf.coef_,
        index=class_names,
        columns=feature_names
    )
    coefs.to_csv(os.path.join(OUTDIR, f"{prefix}_logistic_coefficients_matrix.csv"))

    long_rows = []

    for cls in class_names:
        s = coefs.loc[cls].sort_values()
        cls_safe = safe_filename(cls)

        cls_df = pd.DataFrame({
            "class": cls,
            "feature": s.index,
            "coefficient": s.values,
            "abs_coefficient": np.abs(s.values)
        }).sort_values("abs_coefficient", ascending=False)

        cls_df.to_csv(
            os.path.join(OUTDIR, f"{prefix}_coefficients_{cls_safe}.csv"),
            index=False
        )
        long_rows.append(cls_df)

        if len(s) <= 2 * TOP_N_COEFFICIENTS_PER_SIDE:
            top = s
        else:
            top = pd.concat([
                s.head(TOP_N_COEFFICIENTS_PER_SIDE),
                s.tail(TOP_N_COEFFICIENTS_PER_SIDE)
            ])

        plt.figure(figsize=(9, 10))
        top.plot(kind="barh")
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.title(f"{prefix} - ElasticNet coefficients for {cls}")
        plt.xlabel("Model coefficient")
        plt.ylabel("Feature")
        save_plot(os.path.join(FIGDIR, f"{prefix}_coefficients_{cls_safe}.png"))

    pd.concat(long_rows, ignore_index=True).to_csv(
        os.path.join(OUTDIR, f"{prefix}_logistic_coefficients_long.csv"),
        index=False
    )


def extract_class_shap_values(shap_values, class_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Robustly converts SHAP output to {class_name: matrix[n_samples, n_features]}.
    Compatible with old list output and newer Explanation output.
    """
    if isinstance(shap_values, list):
        return {cls: shap_values[i] for i, cls in enumerate(class_names)}

    values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)

    # Binary or single-output case: n_samples x n_features
    if values.ndim == 2:
        return {class_names[-1]: values}

    # Multiclass SHAP commonly returns: n_samples x n_features x n_classes
    if values.ndim == 3 and values.shape[2] == len(class_names):
        return {cls: values[:, :, i] for i, cls in enumerate(class_names)}

    # Some older versions may return: n_classes x n_samples x n_features
    if values.ndim == 3 and values.shape[0] == len(class_names):
        return {cls: values[i, :, :] for i, cls in enumerate(class_names)}

    raise ValueError(f"Unexpected SHAP values shape: {values.shape}")


def run_shap_linear(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    class_names: List[str],
    prefix: str
) -> None:
    print(f"\nRunning SHAP for {prefix}...")

    clf = model.named_steps["clf"]
    X_train_scaled, X_test_scaled = transformed_data(model, X_train, X_test, feature_names)

    # Use the linear model directly, not model.predict. This avoids the slow generic permutation explainer.
    explainer = shap.LinearExplainer(clf, X_train_scaled)
    shap_values = explainer(X_test_scaled)
    shap_by_class = extract_class_shap_values(shap_values, class_names)

    all_importance = []

    for cls, sv in shap_by_class.items():
        cls_safe = safe_filename(cls)

        # Export full SHAP matrix for the class.
        shap_df = pd.DataFrame(sv, index=X_test_scaled.index, columns=feature_names)
        shap_df.index.name = "PATIENT_ID"
        shap_df.to_csv(os.path.join(OUTDIR, f"{prefix}_SHAP_values_{cls_safe}.csv"))

        mean_abs = np.abs(sv).mean(axis=0)
        mean_signed = sv.mean(axis=0)

        imp = pd.DataFrame({
            "class": cls,
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed
        }).sort_values("mean_abs_shap", ascending=False)

        imp.to_csv(
            os.path.join(OUTDIR, f"{prefix}_SHAP_importance_{cls_safe}.csv"),
            index=False
        )
        all_importance.append(imp)

        # Beeswarm / summary plot for the class.
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv,
            X_test_scaled,
            feature_names=feature_names,
            show=False,
            max_display=TOP_N_SHAP_DISPLAY
        )
        plt.title(f"{prefix} SHAP summary - {cls}")
        save_plot(os.path.join(FIGDIR, f"{prefix}_SHAP_summary_{cls_safe}.png"))

        # Custom top SHAP bar plot.
        top = imp.head(TOP_N_SHAP_DISPLAY).iloc[::-1]
        plt.figure(figsize=(8, 8))
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.xlabel("Mean |SHAP value|")
        plt.ylabel("Feature")
        plt.title(f"{prefix} top SHAP features - {cls}")
        save_plot(os.path.join(FIGDIR, f"{prefix}_top_SHAP_{cls_safe}.png"))

    pd.concat(all_importance, ignore_index=True).to_csv(
        os.path.join(OUTDIR, f"{prefix}_SHAP_importance_all_classes.csv"),
        index=False
    )


def main() -> None:
    print("=" * 80)
    print("07 - Model interpretation: ElasticNet coefficients + SHAP")
    print("=" * 80)

    print("\nLoading data...")
    expr = load_expression(EXPR_PATH)
    clin = load_clinical(CLIN_PATH)
    latent = load_latent(LATENT_PATH)

    expr, clin, latent = align_data(expr, clin, latent)
    sample_ids, y, le, y_raw = prepare_target(clin)

    expr = expr.loc[sample_ids]
    latent = latent.loc[sample_ids]
    class_names = list(le.classes_)

    X_expr = select_top_variable_genes(expr, N_TOP_GENES)
    X_latent = latent.copy()

    print(f"\nExpression feature matrix: {X_expr.shape[0]} patients x {X_expr.shape[1]} genes")
    print(f"Latent feature matrix:     {X_latent.shape[0]} patients x {X_latent.shape[1]} latent dimensions")

    # Same stratified split for expression and latent matrices.
    train_idx, test_idx, y_train, y_test = train_test_split(
        X_expr.index,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_expr_train = X_expr.loc[train_idx]
    X_expr_test = X_expr.loc[test_idx]
    X_latent_train = X_latent.loc[train_idx]
    X_latent_test = X_latent.loc[test_idx]

    split_df = pd.DataFrame({
        "PATIENT_ID": list(train_idx) + list(test_idx),
        "split": ["train"] * len(train_idx) + ["test"] * len(test_idx),
        "class": list(y_raw.loc[train_idx].values) + list(y_raw.loc[test_idx].values)
    })
    split_df.to_csv(os.path.join(OUTDIR, "train_test_split_patients.csv"), index=False)

    print("\nTraining expression ElasticNet logistic model...")
    expr_model = train_elasticnet_logistic_model(X_expr_train, y_train)
    evaluate_model(expr_model, X_expr_train, X_expr_test, y_train, y_test, class_names, "expression")

    print("\nTraining latent ElasticNet logistic model...")
    latent_model = train_elasticnet_logistic_model(X_latent_train, y_train)
    evaluate_model(latent_model, X_latent_train, X_latent_test, y_train, y_test, class_names, "latent")

    print("\nExporting class-specific coefficients...")
    logistic_coefficients(expr_model, X_expr.columns.tolist(), class_names, prefix="expression")
    logistic_coefficients(latent_model, X_latent.columns.tolist(), class_names, prefix="latent")

    print("\nComputing SHAP values...")
    run_shap_linear(expr_model, X_expr_train, X_expr_test, X_expr.columns.tolist(), class_names, prefix="expression")
    run_shap_linear(latent_model, X_latent_train, X_latent_test, X_latent.columns.tolist(), class_names, prefix="latent")

    config = {
        "target_col": TARGET_COL,
        "n_top_genes": N_TOP_GENES,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_expression_features": int(X_expr.shape[1]),
        "n_latent_features": int(X_latent.shape[1]),
        "n_patients": int(len(sample_ids)),
        "classes": class_names
    }
    with open(os.path.join(OUTDIR, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Results saved in: {OUTDIR}")
    print(f"Figures saved in: {FIGDIR}")


if __name__ == "__main__":
    main()
