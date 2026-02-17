# =====================================
# 1. LIBRERÍAS
# =====================================

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score


# =====================================
# 2. COPIA DEL DATAFRAME
# =====================================

df = pd.read_csv('10.000_Empresas_mas_Grandes_del_País_20260210.csv')

# Normalización de nombres de columnas
df.columns.to_list()
nuevos_nombres = {
    'INGRESOS OPERACIONALES' : 'INGRESOS',
    'GANANCIA (PÉRDIDA)' : 'GANANCIA',
    'TOTAL ACTIVOS' : 'ACTIVOS',
    'TOTAL PASIVOS' : 'PASIVOS',
    'TOTAL PATRIMONIO' : 'PATRIMONIO',
}
df = df.rename(columns=nuevos_nombres)

df = df.copy()

cols = ["INGRESOS", "GANANCIA", "ACTIVOS", "PASIVOS", "PATRIMONIO"]

# Limpieza si vienen como string
for col in cols:
    if df[col].dtype == "object":
        df[col] = (
            df[col]
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

df.replace([np.inf, -np.inf], np.nan, inplace=True)


# =====================================
# 3. FEATURE ENGINEERING (RATIOS)
# =====================================

df["ratio_deuda_activos"] = df["PASIVOS"] / df["ACTIVOS"]
df["ratio_deuda_patrimonio"] = df["PASIVOS"] / df["PATRIMONIO"]
df["ratio_apalancamiento_total"] = df["ACTIVOS"] / df["PATRIMONIO"]

df["ROA"] = df["GANANCIA"] / df["ACTIVOS"]

df.replace([np.inf, -np.inf], np.nan, inplace=True)

ratios = [
    "ratio_deuda_activos",
    "ratio_deuda_patrimonio",
    "ratio_apalancamiento_total"
]

df_model = df.dropna(subset=ratios).copy()


# =====================================
# 4. WINSORIZACIÓN (1% - 99%)
# =====================================

for col in ["ratio_deuda_patrimonio", "ratio_apalancamiento_total"]:
    lower = df_model[col].quantile(0.01)
    upper = df_model[col].quantile(0.99)
    df_model[col] = df_model[col].clip(lower, upper)


# =====================================
# 5. COLUMN TRANSFORMER
# =====================================

preprocessor = ColumnTransformer(
    transformers=[
        ("robust_scaler", RobustScaler(), ratios)
    ],
    remainder="drop"
)


# =====================================
# 6. PIPELINE COMPLETO
# =====================================

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("clustering", KMeans(n_clusters=3, random_state=42, n_init=10))
])


# =====================================
# 7. ENTRENAMIENTO
# =====================================

df_model["cluster"] = pipeline.fit_predict(df_model)


# =====================================
# 8. EVALUACIÓN
# =====================================

# Silhouette score
X_transformed = pipeline.named_steps["preprocessing"].transform(df_model)
labels = df_model["cluster"]

score = silhouette_score(X_transformed, labels)
print(f"Silhouette Score: {score:.4f}")


# ROA por cluster
print("\nROA promedio por cluster:")
print(df_model.groupby("cluster")["ROA"].mean())


# Perfil financiero promedio
print("\nPerfil financiero por cluster:")
print(df_model.groupby("cluster")[ratios].mean())

# =====================================
# 9. CREAR CARPETA OUT
# =====================================

os.makedirs("out", exist_ok=True)


# =====================================
# 10. DISTANCIA DE MAHALANOBIS
# =====================================

# Matriz de covarianza
cov_matrix = np.cov(X_transformed.T)
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Media global
mean_vec = np.mean(X_transformed, axis=0)

# Calcular distancia Mahalanobis
mahal_distances = [
    mahalanobis(x, mean_vec, inv_cov_matrix)
    for x in X_transformed
]

df_model["mahalanobis_dist"] = mahal_distances

print("\nResumen distancia Mahalanobis:")
print(df_model["mahalanobis_dist"].describe())

# Guardar top observaciones más extremas
df_model.sort_values("mahalanobis_dist", ascending=False)\
        .head(20)\
        .to_csv("out/top_mahalanobis_extremos.csv", index=False)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_transformed)

df_model["PCA1"] = X_pca[:, 0]
df_model["PCA2"] = X_pca[:, 1]

plt.figure()
for cluster in df_model["cluster"].unique():
    subset = df_model[df_model["cluster"] == cluster]
    plt.scatter(subset["PCA1"], subset["PCA2"])

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Clusters - Visualización PCA")
plt.savefig("out/pca_clusters.png")
plt.close()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_transformed)

df_model["PCA1"] = X_pca[:, 0]
df_model["PCA2"] = X_pca[:, 1]

plt.figure()
for cluster in df_model["cluster"].unique():
    subset = df_model[df_model["cluster"] == cluster]
    plt.scatter(subset["PCA1"], subset["PCA2"])

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Clusters - Visualización PCA")
plt.savefig("out/pca_clusters.png")
plt.close()

roa_means = df_model.groupby("cluster")["ROA"].mean()

plt.figure()
roa_means.plot(kind="bar")
plt.title("ROA promedio por Cluster")
plt.ylabel("ROA")
plt.xlabel("Cluster")
plt.savefig("out/roa_por_cluster.png")
plt.close()
