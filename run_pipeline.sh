#!/bin/bash

set -e

echo "============================================================"
echo "Running DeepGliomaAI pipeline"
echo "============================================================"

echo "[1/9] Data quality control"
python scripts/01_data_qc.py

echo "[2/9] Training Variational Autoencoder"
python scripts/02_build_vae.py

echo "[3/9] Survival analysis of VAE latent dimensions"
python scripts/03_vae_latent_survival.py

echo "[4/9] Latent gene correlation and biological interpretation"
python scripts/04_latent_biological_interpretation.py

echo "[5/9] Latent space structure and clustering"
python scripts/05_latent_structure.py

echo "[6/9] Supervised molecular subtype classification"
python scripts/06_supervised_models.py

echo "[7/9] SHAP and model interpretation"
python scripts/07_model_interpretation_shap.py

echo "[8/9] Systematic biological annotation of latent dimensions"
python scripts/08_latent_biological_annotation.py

echo "[9/9] GSEA correction and final biological axes"
python scripts/09_gsea_and_final_axes.py

echo "============================================================"
echo "DeepGliomaAI pipeline completed successfully."
echo "============================================================"





