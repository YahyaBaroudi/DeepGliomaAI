# DeepGliomaAI

> **DeepGliomaAI is an interpretable deep learning framework for glioma transcriptomics. It combines Variational Autoencoders, survival modelling, molecular subtype classification, SHAP explainability, and GSEA-based biological annotation to uncover biologically meaningful glioma latent programs.**

---

## Overview

**DeepGliomaAI** is a computational oncology project designed to learn biologically meaningful latent representations from glioma transcriptomic profiles.

The project uses a **Variational Autoencoder (VAE)** to compress high-dimensional bulk RNA-seq expression data into a compact latent space. These latent dimensions are then evaluated for their association with patient survival, molecular subtype structure, supervised classification performance, known glioma biomarkers, and biological pathway activity.

The central hypothesis is that glioma transcriptomes reflect underlying biological states such as proliferation, mesenchymal invasion, immune microenvironment activation, angiogenesis, and neuronal/proneural differentiation. DeepGliomaAI explores whether deep learning can recover these axes in an unsupervised and interpretable manner.

---

## Scientific Question

Can a Variational Autoencoder learn biologically meaningful latent dimensions from glioma transcriptomes that are associated with patient survival, molecular subtype structure, established glioma biomarkers, supervised subtype classification, and major biological programs?

More specifically, this project asks whether VAE-derived latent dimensions can capture glioma-relevant programs such as:

| Biological process | Relevance in glioma |
|---|---|
| Proliferation / cell cycle | Associated with high-grade tumors and poor prognosis |
| Mesenchymal activation | Linked to invasion, extracellular matrix remodeling, and aggressive glioma states |
| Angiogenesis | Important in glioblastoma progression and tumor vascularization |
| Immune microenvironment | Reflects inflammatory activation, antigen presentation, and immune infiltration |
| Neuronal / proneural differentiation | Often associated with less aggressive and more differentiated tumor states |

---

## Project Rationale

Gliomas are highly heterogeneous brain tumors. Their clinical behavior is influenced by histological grade, molecular subtype, and transcriptomic state. Modern glioma classification integrates molecular markers such as **IDH**, **1p/19q codeletion**, **ATRX**, **TP53**, **TERT**, **EGFR**, **CDKN2A/B**, and **MGMT**.

However, these molecular markers do not fully capture the complexity of glioma biology. Tumor aggressiveness also emerges from broader transcriptomic programs, including cell-cycle activation, mesenchymal transition, extracellular matrix remodeling, immune activation, metabolic reprogramming, angiogenesis, and loss of neuronal/proneural differentiation.

This project investigates whether a deep learning model can learn these biological programs directly from transcriptomic data and organize them into interpretable latent dimensions.

---

## Main Objectives

| Objective | Description |
|---|---|
| Data preparation | Build a clean glioma transcriptomic and clinical dataset |
| Representation learning | Train a Variational Autoencoder to compress gene expression into latent dimensions |
| Survival modelling | Evaluate whether VAE latent dimensions are associated with overall survival |
| Subtype structure | Assess whether the latent space captures molecular subtype and grade structure |
| Supervised classification | Compare models based on raw expression features and VAE latent features |
| Model interpretation | Use coefficients and SHAP to interpret subtype classification models |
| Biological annotation | Annotate latent dimensions using gene correlations, biomarkers, clinical associations, survival, and GSEA |
| Final synthesis | Summarize the learned latent space into major biological axes |

---

## Dataset

The project uses matched glioma bulk transcriptomic and clinical data.

Expected input files:

```text
data/expr_mat_log2.csv
data/clinical_matched.csv
```

The expression matrix should contain log2-transformed gene expression values.

Accepted expression formats:

```text
samples x genes
```

or:

```text
genes x samples
```

The scripts automatically detect and transpose the expression matrix when required.

The clinical file should contain at least:

| Column | Description |
|---|---|
| `PATIENT_ID` | Patient identifier |
| `OS_MONTHS` | Overall survival time in months |
| `OS_event` | Survival event, coded as 1 = death and 0 = censored/alive |

Recommended additional clinical columns:

| Column | Description |
|---|---|
| `SUBTYPE` | Molecular subtype label |
| `CANCER_TYPE_ACRONYM` | Cancer type label, for example LGG or GBM |
| `grade_group` | Tumor grade group |
| `AGE` | Patient age |
| `SEX` | Patient sex |

Raw transcriptomic and clinical matrices are not included in this repository due to file size and data redistribution considerations. See `data/README.md` for the expected file format.

---

## Workflow

The project is organized into nine computational steps:

| Step | Script | Purpose |
|---|---|---|
| 01 | `01_data_qc.py` | Data quality control and exploratory analysis |
| 02 | `02_build_vae.py` | Training of the Variational Autoencoder |
| 03 | `03_vae_latent_survival.py` | Survival analysis of VAE latent dimensions |
| 04 | `04_latent_biological_interpretation.py` | Gene correlation and initial latent interpretation |
| 05 | `05_latent_structure.py` | Latent space clustering and UMAP analysis |
| 06 | `06_supervised_models.py` | Molecular subtype classification |
| 07 | `07_model_interpretation_shap.py` | SHAP and coefficient-based model interpretation |
| 08 | `08_latent_biological_annotation.py` | Systematic biological annotation of latent dimensions |
| 09 | `09_fix_gsea_and_final_axes.py` | GSEA correction and final biological axis construction |

---

## Step 01 — Data Quality Control

The first step performs quality control of the expression and clinical matrices.

Main checks include:

| QC check | Purpose |
|---|---|
| Patient/sample matching | Ensure that expression and clinical data are aligned |
| Missing values | Detect incomplete data |
| Duplicate patients | Avoid duplicated patient-level observations |
| Duplicate genes | Avoid redundant gene-level features |
| Gene variance distribution | Identify informative genes |
| PCA exploration | Visualize large-scale transcriptomic structure |

Final matched dataset:

| Metric | Value |
|---|---:|
| Patients | 663 |
| Genes | 20,228 |
| Missing values | 0 |
| Duplicate patients | 0 |
| Duplicate genes | 0 |

Main outputs:

```text
results/01_data_qc/qc_summary.csv
figures/01_data_qc/
```

---

## Step 02 — Variational Autoencoder

A Variational Autoencoder is trained on high-variance genes to learn a compact transcriptomic representation.

The model compresses gene expression profiles into 20 latent dimensions:

```text
Gene expression matrix → VAE → VAE_Z1 to VAE_Z20
```

Each latent dimension can be interpreted as a learned transcriptomic program.

| Component | Description |
|---|---|
| Input | High-dimensional gene expression matrix |
| Model | Variational Autoencoder |
| Output | 20 latent dimensions |
| Latent features | `VAE_Z1` to `VAE_Z20` |
| Downstream use | Survival, classification, SHAP, biomarker annotation, GSEA |

Main outputs:

```text
results/02_vae/vae_latent_space.csv
results/02_vae/vae_training_history.csv
results/02_vae/latent_clinical_correlations.csv
models/02_vae/best_vae_model.pt
```

The trained model file is not included in the public repository because it can be regenerated by running the pipeline.

---

## Step 03 — Survival Analysis of VAE Latent Dimensions

The prognostic value of the VAE latent space is evaluated using Cox proportional hazards models.

Analyses include:

| Analysis | Purpose |
|---|---|
| Univariate Cox models | Test each latent dimension individually |
| Multivariate Cox model | Combine selected latent dimensions into a risk model |
| Cross-validated C-index | Estimate survival prediction performance |
| Kaplan-Meier stratification | Compare survival between latent risk groups |

Key survival performance:

| Metric | Value |
|---|---:|
| Cross-validated C-index | 0.842 ± 0.032 |
| Kaplan-Meier log-rank p-value | 7.545e-29 |

High-risk latent dimensions:

| Latent dimension | Interpretation |
|---|---|
| `VAE_Z10` | Proliferation / cell cycle / high-grade axis |
| `VAE_Z13` | Proliferation and mesenchymal activation |
| `VAE_Z11` | GBM-like aggressive program |
| `VAE_Z20` | Loss of proneural features and inflammatory/aggressive state |
| `VAE_Z8` | Mesenchymal / ECM / inflammation / angiogenesis |

Protective latent dimensions:

| Latent dimension | Interpretation |
|---|---|
| `VAE_Z9` | Protective axis, inversely associated with inflammatory and mesenchymal programs |
| `VAE_Z3` | Favorable survival-associated latent program |
| `VAE_Z16` | Protective / less aggressive latent program |
| `VAE_Z14` | Protective latent program |
| `VAE_Z1` | Protective latent program |

Main outputs:

```text
results/03_vae_latent_survival/univariate_cox_latent_dims.csv
results/03_vae_latent_survival/multivariate_cox_latent_model.csv
results/03_vae_latent_survival/cross_validated_cindex.csv
```

---

## Step 04 — Latent Gene Correlation and Initial Biological Interpretation

Each VAE latent dimension is correlated with gene expression to identify genes positively or negatively associated with the latent program.

This step helps interpret each latent dimension biologically.

Representative interpretations:

| Latent dimension | Biological interpretation |
|---|---|
| `VAE_Z10` | Proliferation / cell cycle / high-grade program |
| `VAE_Z8` | Mesenchymal / extracellular matrix / inflammation / angiogenesis |
| `VAE_Z11` | GBM-like aggressive program involving proliferation, EMT, metabolism and angiogenesis |
| `VAE_Z13` | Mixed proliferation, mesenchymal and inflammatory activation |
| `VAE_Z9` | Protective axis inversely associated with inflammatory and mesenchymal programs |

Main outputs:

```text
results/04_latent_biology/all_latent_gene_correlations.csv
results/04_latent_biology/latent_top_gene_summary.csv
results/04_latent_biology/VAE_Z*_gene_correlations.csv
```

---

## Step 05 — Latent Space Structure

The VAE latent space is explored using dimensionality reduction and clustering.

Analyses include:

| Analysis | Purpose |
|---|---|
| PCA on latent dimensions | Explore broad latent structure |
| UMAP projection | Visualize nonlinear patient organization |
| K-means clustering | Identify patient clusters in latent space |
| Elbow analysis | Estimate appropriate number of clusters |
| Silhouette analysis | Evaluate cluster separation |

This step evaluates whether the VAE latent space organizes patients into biologically meaningful structures.

Main outputs:

```text
results/05_latent_structure/latent_clusters.csv
results/05_latent_structure/umap_coordinates.csv
figures/05_latent_structure/
```

---

## Step 06 — Supervised Molecular Subtype Classification

Supervised models are trained to predict molecular subtype labels using either raw expression features or VAE latent dimensions.

Models compared:

| Feature set | Model |
|---|---|
| Raw expression | ElasticNet Logistic Regression |
| Raw expression | Random Forest |
| VAE latent space | ElasticNet Logistic Regression |
| VAE latent space | Random Forest |

Main classification results:

| Feature set | Model | Accuracy | Macro F1 | AUC |
|---|---|---:|---:|---:|
| Expression | ElasticNet Logistic Regression | 0.976 | 0.821 | 0.956 |
| Expression | Random Forest | 0.976 | 0.655 | 0.990 |
| VAE latent space | ElasticNet Logistic Regression | 0.808 | 0.532 | 0.838 |
| VAE latent space | Random Forest | 0.888 | 0.572 | 0.894 |

Interpretation:

| Observation | Meaning |
|---|---|
| Raw expression features achieved the best subtype classification | Gene-level information preserves fine subtype-specific signals |
| VAE latent features performed lower in classification | Compression reduces some discriminative detail |
| VAE latent features remain biologically valuable | They provide compact, interpretable and survival-associated representations |

Main outputs:

```text
results/06_supervised_models/supervised_model_results.csv
figures/06_supervised_models/
```

---

## Step 07 — SHAP Model Interpretation

SHAP was used to interpret supervised classification models.

Two levels of interpretation were performed:

| Interpretation level | Purpose |
|---|---|
| Expression-level SHAP | Identify genes contributing to molecular subtype prediction |
| Latent-level SHAP | Identify VAE dimensions contributing to molecular subtype prediction |

Important latent dimensions repeatedly identified by SHAP:

| Latent dimension | Biological relevance |
|---|---|
| `VAE_Z8` | Mesenchymal / inflammation / angiogenesis |
| `VAE_Z9` | Protective / anti-inflammatory / proneural-like axis |
| `VAE_Z10` | Proliferation / high-grade axis |
| `VAE_Z11` | GBM-like aggressive program |
| `VAE_Z13` | Proliferation and mesenchymal activation |
| `VAE_Z20` | Loss of proneural features and aggressive state |

These dimensions were also strongly involved in survival and biological annotation, suggesting that the same latent axes encode clinically and molecularly relevant information.

Main outputs:

```text
figures/07_model_interpretation/
results/07_model_interpretation/
```

Only selected SHAP figures and summary outputs should be included in the public repository to keep it lightweight.

---

## Step 08 — Biological Annotation of Latent Dimensions

Each VAE dimension was systematically annotated using:

| Annotation layer | Description |
|---|---|
| Top positive genes | Genes positively correlated with a latent dimension |
| Top negative genes | Genes negatively correlated with a latent dimension |
| Biomarker correlations | Association with known glioma markers |
| Clinical associations | Association with subtype, grade, IDH-like, GBM-like and codel-like labels |
| Survival associations | Hazard ratio and survival significance |
| GSEA | Pathway enrichment for ranked gene correlations |

Glioma-relevant markers used for annotation included:

| Marker group | Genes |
|---|---|
| Molecular classification | `IDH1`, `IDH2`, `ATRX`, `TP53`, `TERT`, `EGFR`, `CDKN2A`, `CDKN2B`, `MGMT` |
| Lineage / differentiation | `OLIG2`, `SOX2`, `GFAP`, `DLL3`, `MAP2`, `DCX`, `SNAP25` |
| Proliferation | `MKI67`, `TOP2A`, `PCNA`, `AURKA`, `AURKB`, `BIRC5`, `CCNB1`, `CDK1` |
| Mesenchymal / invasion | `CHI3L1`, `IGFBP2`, `POSTN`, `COL1A1`, `COL1A2`, `FN1`, `VEGFA` |
| Immune / inflammation | `CD274`, `CXCL10`, `CXCL9`, `HLA-DRA`, `PTPRC`, `C1QA`, `C1QB`, `C1QC` |

Main outputs:

```text
results/08_latent_biological_annotation/latent_dimension_biomarker_correlations.csv
results/08_latent_biological_annotation/latent_dimension_clinical_subtype_associations.csv
results/08_latent_biological_annotation/latent_dimension_final_annotation_table.csv
results/08_latent_biological_annotation/latent_dimension_survival_summary.csv
```

---

## Step 09 — GSEA Correction and Final Biological Axes

The final step corrects and summarizes GSEA results, then integrates the latent dimensions into four major biological axes.

Final biological axes:

| Axis | Main latent dimensions | Biological interpretation |
|---|---|---|
| Proliferation / high grade | `VAE_Z10`, `VAE_Z13`, `VAE_Z11` | Cell cycle, E2F, G2M, MYC, high-risk survival |
| Mesenchymal / invasion / angiogenesis | `VAE_Z8`, `VAE_Z11`, `VAE_Z13` | EMT, ECM, POSTN, CHI3L1, VEGFA, invasion |
| Immune / microenvironment | `VAE_Z8`, `VAE_Z10`, `VAE_Z13` | IFN response, TNF/NF-kB, antigen presentation, inflammatory activation |
| Neuronal / proneural / protective | `VAE_Z9`, `VAE_Z3`, `VAE_Z16` | Neuronal/proneural markers, reduced inflammation, better prognosis |

Main outputs:

```text
results/09_gsea_final_four_axes/final_four_axes_summary.csv
results/09_gsea_final_four_axes/final_four_axes_latent_dimension_scores.csv
```

---

## Key Findings

### 1. The VAE latent space is prognostic

The VAE-derived latent risk score stratified patients by overall survival.

| Metric | Value |
|---|---:|
| Cross-validated C-index | 0.842 ± 0.032 |
| Kaplan-Meier log-rank p-value | 7.545e-29 |

This suggests that the learned latent representation captures survival-relevant transcriptomic programs.

---

### 2. Latent dimensions recover molecular glioma structure

Several latent dimensions were associated with:

| Variable | Interpretation |
|---|---|
| `SUBTYPE` | Molecular subtype structure |
| `IDH_like` | IDH-like transcriptomic pattern |
| `GBM_like` | GBM-like aggressive tumor state |
| `codel_like` | 1p/19q-codeletion-like structure |
| `grade_group` | Tumor grade-related biology |

---

### 3. High-risk latent dimensions represent aggressive tumor programs

High-risk dimensions were associated with biological programs related to proliferation, cell cycle, mesenchymal activation, extracellular matrix remodeling, angiogenesis, immune/inflammatory activation, and GBM-like aggressive states.

| Latent dimension | Survival direction | Main biological interpretation |
|---|---|---|
| `VAE_Z10` | High risk | Proliferation / cell cycle / high grade |
| `VAE_Z13` | High risk | Proliferation + mesenchymal + inflammation |
| `VAE_Z11` | High risk | GBM-like aggressive program |
| `VAE_Z20` | High risk | Loss of proneural features and inflammatory/aggressive state |
| `VAE_Z8` | High risk | Mesenchymal / ECM / angiogenesis / inflammation |

---

### 4. Protective latent dimensions reflect less aggressive programs

Protective dimensions were associated with improved survival and reduced inflammatory or mesenchymal activation.

| Latent dimension | Survival direction | Main biological interpretation |
|---|---|---|
| `VAE_Z9` | Protective | Inversely associated with inflammatory and mesenchymal activation |
| `VAE_Z3` | Protective | Favorable latent program |
| `VAE_Z16` | Protective | Less aggressive latent program |
| `VAE_Z14` | Protective | Favorable survival-associated latent program |
| `VAE_Z1` | Protective | Favorable survival-associated latent program |

---

## Detailed Interpretation of Key Latent Dimensions

### VAE_Z10 — Proliferation / high-grade axis

`VAE_Z10` is one of the strongest high-risk dimensions.

Associated markers include:

| Marker | Biological meaning |
|---|---|
| `MKI67` | Proliferation |
| `TOP2A` | DNA replication and cell division |
| `AURKA` / `AURKB` | Mitotic regulation |
| `BIRC5` | Cell survival and mitosis |
| `CCNB1` | Cell-cycle progression |
| `CDK1` | Cell-cycle control |
| `PCNA` | DNA replication |

Interpretation:

```text
VAE_Z10 = proliferation / cell-cycle / high-grade axis
```

---

### VAE_Z8 — Mesenchymal / ECM / inflammation / angiogenesis axis

`VAE_Z8` is associated with aggressive microenvironmental and invasive programs.

Associated markers include:

| Marker | Biological meaning |
|---|---|
| `COL1A1` / `COL1A2` | Extracellular matrix remodeling |
| `FN1` | Matrix adhesion and invasion |
| `POSTN` | Mesenchymal activation |
| `IGFBP2` | Glioma aggressiveness and invasion |
| `VEGFA` | Angiogenesis |
| `CXCL10` / `CXCL9` | Inflammatory chemokine signaling |
| `IRF1` | Interferon-associated transcriptional activation |

Interpretation:

```text
VAE_Z8 = mesenchymal / ECM / invasion / inflammation / angiogenesis axis
```

---

### VAE_Z11 — GBM-like aggressive program

`VAE_Z11` is associated with aggressive tumor biology, including proliferation, EMT, metabolism and angiogenesis.

Associated pathways include:

| Pathway | Interpretation |
|---|---|
| E2F Targets | Cell-cycle activation |
| G2-M Checkpoint | Proliferation and mitosis |
| Epithelial Mesenchymal Transition | Mesenchymal and invasive phenotype |
| Glycolysis | Metabolic reprogramming |
| mTORC1 Signaling | Growth and metabolic signalling |
| Angiogenesis | Tumor vascularization |

Interpretation:

```text
VAE_Z11 = GBM-like aggressive program involving proliferation, invasion, metabolism and angiogenesis
```

---

### VAE_Z13 — Mixed proliferation, mesenchymal and inflammatory program

`VAE_Z13` is another high-risk latent dimension.

It integrates several aggressive glioma programs:

| Program | Biological interpretation |
|---|---|
| Proliferation | High mitotic activity |
| Mesenchymal activation | Invasion and extracellular matrix remodeling |
| Inflammation | Immune and interferon-associated activation |
| Coagulation / vascular biology | Potential tumor microenvironment remodeling |

Interpretation:

```text
VAE_Z13 = high-risk mixed program combining proliferation, mesenchymal activation and inflammation
```

---

### VAE_Z9 — Protective anti-inflammatory / proneural-like axis

`VAE_Z9` shows a protective survival association.

It is negatively associated with inflammatory and mesenchymal programs.

Interpretation:

```text
VAE_Z9 = protective axis inversely associated with inflammatory and mesenchymal activation
```

This dimension may reflect a more neuronal or proneural-like transcriptomic state.

---

## Selected Results Included in This Repository

The repository includes selected summary result tables:

| File | Description |
|---|---|
| `results/01_data_qc/qc_summary.csv` | Data QC summary |
| `results/03_vae_latent_survival/univariate_cox_latent_dims.csv` | Univariate Cox analysis of latent dimensions |
| `results/03_vae_latent_survival/multivariate_cox_latent_model.csv` | Multivariate latent Cox model |
| `results/03_vae_latent_survival/cross_validated_cindex.csv` | Cross-validated survival performance |
| `results/06_supervised_models/supervised_model_results.csv` | Supervised classification results |
| `results/08_latent_biological_annotation/latent_dimension_final_annotation_table.csv` | Final annotation of VAE dimensions |
| `results/08_latent_biological_annotation/latent_dimension_survival_summary.csv` | Survival summary of latent dimensions |
| `results/09_gsea_final_four_axes/final_four_axes_summary.csv` | Summary of final biological axes |
| `results/09_gsea_final_four_axes/final_four_axes_latent_dimension_scores.csv` | Axis scores per latent dimension |

Large intermediate files, raw datasets, trained models, and full SHAP matrices are intentionally excluded.

---

## Repository Structure

```text
DeepGliomaAI/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── requirements.txt
├── environment.yml
├── run_pipeline.sh
├── DO_NOT_UPLOAD.md
│
├── data/
│   └── README.md
│
├── scripts/
│   ├── 01_data_qc.py
│   ├── 02_build_vae.py
│   ├── 03_vae_latent_survival.py
│   ├── 04_latent_biological_interpretation.py
│   ├── 05_latent_structure.py
│   ├── 06_supervised_models.py
│   ├── 07_model_interpretation_shap.py
│   ├── 08_latent_biological_annotation.py
│   └── 09_fix_gsea_and_final_axes.py
│
├── results/
│   ├── 01_data_qc/
│   ├── 03_vae_latent_survival/
│   ├── 06_supervised_models/
│   ├── 08_latent_biological_annotation/
│   └── 09_gsea_final_four_axes/
│
├── figures/
│   ├── main/
│   └── supplementary/
│
└── docs/
    ├── project_overview.md
    ├── results_summary.md
    └── biological_interpretation.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Yhy-Brd/DeepGliomaAI.git
cd DeepGliomaAI
```

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate deepgliomaai
```

Alternatively, install dependencies using pip:

```bash
pip install -r requirements.txt
```

---

## Requirements

Main Python packages:

| Package | Use |
|---|---|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scipy` | Statistical analysis |
| `scikit-learn` | Machine learning models |
| `matplotlib` | Plotting |
| `seaborn` | Statistical plotting |
| `torch` | VAE deep learning model |
| `lifelines` | Survival analysis |
| `shap` | Model explainability |
| `gseapy` | GSEA pathway analysis |
| `umap-learn` | UMAP latent projection |
| `statsmodels` | Statistical testing and correction |
| `openpyxl` | Excel file support |

---

## How to Reproduce the Analysis

Place the required input files in the `data/` directory:

```text
data/expr_mat_log2.csv
data/clinical_matched.csv
```

Then run the full pipeline:

```bash
bash run_pipeline.sh
```

Or run each step manually:

```bash
python scripts/01_data_qc.py
python scripts/02_build_vae.py
python scripts/03_vae_latent_survival.py
python scripts/04_latent_biological_interpretation.py
python scripts/05_latent_structure.py
python scripts/06_supervised_models.py
python scripts/07_model_interpretation_shap.py
python scripts/08_latent_biological_annotation.py
python scripts/09_fix_gsea_and_final_axes.py
```

---

## Files Not Included

The following files are intentionally excluded from the public repository:

| File type | Reason |
|---|---|
| `data/expr_mat_log2.csv` | Raw expression matrix, large file |
| `data/clinical_matched.csv` | Clinical matrix, redistribution considerations |
| `models/*.pt` | Trained models can be regenerated |
| `models/*.pkl` | Trained models can be regenerated |
| `results/**/*SHAP_values*.csv` | Large intermediate SHAP matrices |
| `results/**/*scaled*.csv` | Large intermediate matrices |
| `results/**/*X_matched*.csv` | Large intermediate matrices |
| `results/**/*ranked_genes*.rnk` | Intermediate GSEA files |
| `results/**/*prerank*.rnk` | Intermediate GSEA files |

---

## Limitations

| Limitation | Explanation |
|---|---|
| No external validation yet | Independent validation using CGGA or another glioma cohort is still required |
| VAE stochasticity | Latent dimensions may vary depending on random seed and model training configuration |
| Transcriptomic inference | Some molecular features are inferred from expression patterns rather than directly measured genomic alterations |
| Survival adjustment | Survival models should be further adjusted for clinical covariates such as age, grade, IDH status and treatment |
| Classification performance | Raw expression features outperformed VAE latent features for supervised subtype prediction |
| Research only | The current pipeline is not intended for clinical use |

---

## Future Work

| Direction | Description |
|---|---|
| External validation | Validate survival and latent axes in CGGA or other independent glioma cohorts |
| Multi-omics integration | Integrate mutations, copy number alterations, methylation and transcriptomics |
| Model comparison | Compare VAE with PCA, NMF, standard autoencoders and other representation learning methods |
| Deep survival modelling | Evaluate Cox-nnet, DeepSurv or multimodal survival models |
| Stability analysis | Test robustness of latent dimensions across random seeds and training splits |
| Model calibration | Assess calibration of prognostic risk scores |
| Workflow management | Convert pipeline to Snakemake or Nextflow |
| Interactive visualization | Build a dashboard for patient projection into latent space |

---

## Project Summary

DeepGliomaAI demonstrates that Variational Autoencoders can learn compact and biologically meaningful representations from glioma transcriptomes.

The learned latent dimensions capture survival-relevant programs, molecular subtype structure, glioma biomarker associations, proliferation and high-grade tumor biology, mesenchymal invasion and angiogenesis, immune microenvironment activation, and neuronal/proneural protective patterns.

The project combines deep learning, survival analysis, supervised classification, SHAP explainability, GSEA, and biomarker-based biological annotation to explore glioma biology from high-dimensional transcriptomic data.

---

## Author

**Yahya Baroudi**  
Master's student in Genomics and Medical Bioinformatics  
GitHub: [Yhy-Brd](https://github.com/Yhy-Brd)

---

## Citation

If you use or adapt this project, please cite this repository.

```bibtex
@misc{deepgliomaai2026,
  author = {Baroudi, Yahya},
  title = {DeepGliomaAI: Interpretable Deep Learning for Glioma Transcriptomics},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Yhy-Brd/DeepGliomaAI}
}
```

---

## License

This project is distributed under the MIT License.

---

## Disclaimer

This project is for research and educational purposes only. It is not intended for clinical diagnosis, treatment decision-making, or direct medical use.
