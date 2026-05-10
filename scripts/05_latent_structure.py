# scripts/05_latent_structure.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP

# =========================
# CONFIG
# =========================

BASE_DIR = "/mnt/d/projects/DeepGliomaAI"

INPUT_LATENT = os.path.join(BASE_DIR, "results/02_vae/vae_latent_space.csv")

OUTDIR = os.path.join(BASE_DIR, "results/05_latent_structure")
os.makedirs(OUTDIR, exist_ok=True)

RANDOM_STATE = 42
MAX_K = 10


# =========================
# LOAD DATA
# =========================

print("🔹 Loading latent space...")

latent_df = pd.read_csv(INPUT_LATENT, index_col=0)

print(f"Latent shape: {latent_df.shape}")


# =========================
# STANDARDIZE
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(latent_df)


# =========================
# ELBOW METHOD
# =========================

print("🔹 Computing elbow curve...")

inertias = []
K_range = range(2, MAX_K + 1)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertias, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "kmeans_elbow.png"), dpi=300)
plt.close()


# =========================
# BEST K (SILHOUETTE)
# =========================

print("🔹 Finding best K with silhouette...")

best_k = 2
best_score = -1

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, labels)
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"✅ Best K: {best_k} (silhouette={best_score:.3f})")


# =========================
# FINAL CLUSTERING
# =========================

print("🔹 Running final clustering...")

kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
clusters = kmeans.fit_predict(X_scaled)

latent_df["cluster"] = clusters

latent_df.to_csv(os.path.join(OUTDIR, "latent_clusters.csv"))


# =========================
# UMAP
# =========================

print("🔹 Computing UMAP...")

umap = UMAP(
    n_components=2,
    random_state=RANDOM_STATE,
    n_neighbors=15,
    min_dist=0.3
)

embedding = umap.fit_transform(X_scaled)

umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=latent_df.index)
umap_df["cluster"] = clusters

umap_df.to_csv(os.path.join(OUTDIR, "umap_coordinates.csv"))


# =========================
# PLOT UMAP
# =========================

print("🔹 Plotting UMAP clusters...")

plt.figure(figsize=(7, 6))
sns.scatterplot(
    x="UMAP1",
    y="UMAP2",
    hue="cluster",
    data=umap_df,
    palette="tab10",
    s=60,
    alpha=0.8
)

plt.title("UMAP of latent space (clusters)")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig(os.path.join(OUTDIR, "umap_clusters.png"), dpi=300)
plt.close()


print("✅ DONE: Latent structure analysis complete.")
