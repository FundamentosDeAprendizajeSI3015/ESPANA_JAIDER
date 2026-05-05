"""
EDA, Preprocesamiento y Clustering – 10,000 Empresas más Grandes del País
Objetivo: Segmentar empresas por perfil de apalancamiento financiero mediante K-Means
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
FILE_PATH = "C:\\Users\\User\\Documents\\ml\\ML\\10.000_Empresas_mas_Grandes_del_País_20260210.csv"

df_raw = pd.read_csv(FILE_PATH)
print("=" * 60)
print("CARGA DE DATOS")
print("=" * 60)
print(f"  Filas   : {df_raw.shape[0]:,}")
print(f"  Columnas: {df_raw.shape[1]}")
print(f"\nColumnas:\n  {df_raw.columns.tolist()}")

# ─────────────────────────────────────────────────────────────
# 2. LIMPIEZA Y CONVERSIÓN DE TIPOS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LIMPIEZA Y CONVERSIÓN DE TIPOS")
print("=" * 60)

df = df_raw.copy()

monetary_cols = ["INGRESOS OPERACIONALES", "GANANCIA (PÉRDIDA)",
                 "TOTAL ACTIVOS", "TOTAL PASIVOS", "TOTAL PATRIMONIO"]

for col in monetary_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )

df["Año de Corte"] = df["Año de Corte"].astype(str).str.replace(",", "").astype(int)
df["NIT"] = df["NIT"].astype(str).str.replace(",", "").str.strip()

print(f"  Valores nulos después de limpieza:\n{df[monetary_cols].isnull().sum().to_string()}")
print(f"\n  Años disponibles: {sorted(df['Año de Corte'].unique())}")
print(f"  Macrosectores  : {df['MACROSECTOR'].nunique()}")

# ─────────────────────────────────────────────────────────────
# 3. ANÁLISIS EXPLORATORIO (EDA)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA")
print("=" * 60)
print(df[monetary_cols + ["Año de Corte"]].describe().T.to_string())

# ── 3.1 Distribución de empresas por año y macrosector ────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df["Año de Corte"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("Empresas por Año de Corte")
axes[0].set_xlabel("Año"); axes[0].set_ylabel("Cantidad")
axes[0].tick_params(axis="x", rotation=0)
df["MACROSECTOR"].value_counts().plot(
    kind="barh", ax=axes[1], color="coral", edgecolor="white")
axes[1].set_title("Empresas por Macrosector"); axes[1].set_xlabel("Cantidad")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_01_distribucion_empresas.png"), dpi=150)
plt.close()
print("\n  [Gráfica guardada] out/eda_01_distribucion_empresas.png")

# ── 3.2 Distribución de variables financieras (log scale) ─────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(monetary_cols):
    data = df[col][df[col] > 0]
    axes[i].hist(np.log1p(data), bins=60, color="teal", edgecolor="white", alpha=0.8)
    axes[i].set_title(f"log(1 + {col})")
    axes[i].set_xlabel("log(1 + valor)"); axes[i].set_ylabel("Frecuencia")
axes[5].axis("off")
plt.suptitle("Distribución de Variables Financieras (escala logarítmica)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_02_distribuciones_financieras.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/eda_02_distribuciones_financieras.png")

# ── 3.3 Boxplots por macrosector – Pasivos/Activos ────────────
fig, ax = plt.subplots(figsize=(14, 6))
sector_order = (df.groupby("MACROSECTOR")["TOTAL PASIVOS"]
                .median().sort_values(ascending=False).index)
df_pos = df[df["TOTAL ACTIVOS"] > 0].copy()
df_pos["Pasivos/Activos"] = df_pos["TOTAL PASIVOS"] / df_pos["TOTAL ACTIVOS"]
df_pos_clip = df_pos[df_pos["Pasivos/Activos"].between(0, 5)]
sns.boxplot(data=df_pos_clip, x="MACROSECTOR", y="Pasivos/Activos",
            order=sector_order, ax=ax, palette="Set2", linewidth=0.8)
ax.set_title("Ratio Pasivos/Activos por Macrosector")
ax.set_xlabel(""); ax.set_ylabel("Pasivos / Activos")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_03_boxplot_leverage_macrosector.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/eda_03_boxplot_leverage_macrosector.png")

# ── 3.4 Heatmap de correlación ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[monetary_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Correlación entre Variables Financieras")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_04_correlacion.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/eda_04_correlacion.png")

# ── 3.5 Evolución temporal de activos y pasivos promedio ───────
evol = (df.groupby("Año de Corte")[["TOTAL ACTIVOS", "TOTAL PASIVOS", "TOTAL PATRIMONIO"]]
        .median().reset_index())
fig, ax = plt.subplots(figsize=(10, 5))
for col, color in zip(["TOTAL ACTIVOS", "TOTAL PASIVOS", "TOTAL PATRIMONIO"],
                      ["steelblue", "coral", "seagreen"]):
    ax.plot(evol["Año de Corte"], evol[col], marker="o", label=col, color=color)
ax.set_title("Mediana de Activos, Pasivos y Patrimonio por Año")
ax.set_xlabel("Año"); ax.set_ylabel("Billones COP (mediana)")
ax.legend(); ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_05_evolucion_temporal.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/eda_05_evolucion_temporal.png")

# ─────────────────────────────────────────────────────────────
# 4. INGENIERÍA DE CARACTERÍSTICAS DE APALANCAMIENTO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("INGENIERÍA DE CARACTERÍSTICAS DE APALANCAMIENTO")
print("=" * 60)

df_feat = df.copy()
eps = 1e-9

df_feat["deuda_activos"]      = df_feat["TOTAL PASIVOS"]  / (df_feat["TOTAL ACTIVOS"]   + eps)
df_feat["deuda_patrimonio"]   = df_feat["TOTAL PASIVOS"]  / (df_feat["TOTAL PATRIMONIO"].abs() + eps)
df_feat["multiplicador_cap"]  = df_feat["TOTAL ACTIVOS"]  / (df_feat["TOTAL PATRIMONIO"].abs() + eps)
df_feat["cobertura_ingresos"] = df_feat["INGRESOS OPERACIONALES"] / (df_feat["TOTAL PASIVOS"] + eps)
df_feat["margen_neto"]        = df_feat["GANANCIA (PÉRDIDA)"] / (df_feat["INGRESOS OPERACIONALES"].abs() + eps)
df_feat["roa"]                = df_feat["GANANCIA (PÉRDIDA)"] / (df_feat["TOTAL ACTIVOS"] + eps)
df_feat["roe"]                = df_feat["GANANCIA (PÉRDIDA)"] / (df_feat["TOTAL PATRIMONIO"].abs() + eps)

leverage_features = ["deuda_activos", "deuda_patrimonio", "multiplicador_cap",
                     "cobertura_ingresos", "margen_neto", "roa", "roe"]

print(f"\n  Características creadas: {leverage_features}")
print(f"\n  Estadísticas descriptivas:")
print(df_feat[leverage_features].describe().T[["mean","std","min","50%","max"]].to_string())

# ─────────────────────────────────────────────────────────────
# 5. PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PREPROCESAMIENTO")
print("=" * 60)

n_before = len(df_feat)
df_clean = df_feat[df_feat["TOTAL ACTIVOS"] > 0].copy()
print(f"\n  Filas con TOTAL ACTIVOS > 0: {len(df_clean):,} (removidas: {n_before - len(df_clean):,})")

def remove_outliers_iqr(df, cols, factor=5.0):
    mask = pd.Series(True, index=df.index)
    for col in cols:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask &= df[col].between(lower, upper)
    return df[mask]

df_clean = remove_outliers_iqr(df_clean, leverage_features, factor=5.0)
print(f"  Filas tras eliminar outliers extremos (IQR×5): {len(df_clean):,}")

for col in leverage_features:
    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

print(f"  NaN/Inf residuales tras imputación: {df_clean[leverage_features].isnull().sum().sum()}")

def log_modulus(x):
    return np.sign(x) * np.log1p(np.abs(x))

df_transformed = df_clean.copy()
for col in leverage_features:
    df_transformed[col + "_lm"] = log_modulus(df_transformed[col])

transformed_cols = [c + "_lm" for c in leverage_features]

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_transformed[transformed_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=transformed_cols, index=df_transformed.index)

print(f"\n  Shape final para clustering: {X_scaled_df.shape}")

# ── 5.6 Distribuciones post-procesamiento ─────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(transformed_cols):
    axes[i].hist(X_scaled_df[col], bins=60, color="mediumpurple", edgecolor="white", alpha=0.85)
    axes[i].set_title(col.replace("_lm", "").replace("_", " ").title())
    axes[i].set_xlabel("Valor escalado"); axes[i].set_ylabel("Frecuencia")
axes[-1].axis("off")
plt.suptitle("Distribuciones Post-Procesamiento (log-modulus + RobustScaler)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prepro_01_distribuciones_finales.png"), dpi=150)
plt.close()
print("\n  [Gráfica guardada] out/prepro_01_distribuciones_finales.png")

# ── 5.7 Pairplot ───────────────────────────────────────────────
sample = X_scaled_df[transformed_cols[:4]].sample(min(2000, len(X_scaled_df)), random_state=42)
g = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10})
g.figure.suptitle("Pairplot – Features de Apalancamiento (muestra 2,000)", y=1.02)
g.figure.savefig(os.path.join(OUTPUT_DIR, "prepro_02_pairplot_features.png"),
                 dpi=130, bbox_inches="tight")
plt.close()
print("  [Gráfica guardada] out/prepro_02_pairplot_features.png")

# ── 5.8 Correlación post-procesamiento ────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr_post = X_scaled_df.corr()
mask2 = np.triu(np.ones_like(corr_post, dtype=bool))
sns.heatmap(corr_post, mask=mask2, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax,
            linewidths=0.4, cbar_kws={"shrink": 0.8})
ax.set_title("Correlación – Features Procesadas")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prepro_03_correlacion_post.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/prepro_03_correlacion_post.png")

# ─────────────────────────────────────────────────────────────
# 6. EXPORTAR DATASET PROCESADO
# ─────────────────────────────────────────────────────────────
output_df = df_transformed[
    ["NIT", "RAZÓN SOCIAL", "MACROSECTOR", "REGIÓN", "Año de Corte"]
    + leverage_features
].copy()
output_df = output_df.loc[X_scaled_df.index]
output_df[transformed_cols] = X_scaled_df.values
output_df.to_csv(os.path.join(OUTPUT_DIR, "dataset_procesado_apalancamiento.csv"), index=False)
print("\n" + "=" * 60)
print("DATASET EXPORTADO → out/dataset_procesado_apalancamiento.csv")
print(f"  Shape: {output_df.shape}")
print("=" * 60)

# ═════════════════════════════════════════════════════════════
# 7. CLUSTERING – K-MEANS
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CLUSTERING K-MEANS")
print("=" * 60)

from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA

# ── 7.1 Método del codo + Silhouette Score para elegir K ──────
K_RANGE = range(2, 11)
inertias, sil_scores = [], []

print("\n  Calculando inercia y silhouette para K=2..10 ...")
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled_df)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled_df, labels, sample_size=5000, random_state=42))
    print(f"    K={k}  inercia={km.inertia_:,.0f}  silhouette={sil_scores[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(K_RANGE), inertias, marker="o", color="steelblue")
axes[0].set_title("Método del Codo"); axes[0].set_xlabel("K"); axes[0].set_ylabel("Inercia")
axes[0].axvline(x=3, color="red", linestyle="--", alpha=0.6, label="K elegido")
axes[0].legend()

axes[1].plot(list(K_RANGE), sil_scores, marker="s", color="darkorange")
axes[1].set_title("Silhouette Score"); axes[1].set_xlabel("K"); axes[1].set_ylabel("Score")
best_k_idx = int(np.argmax(sil_scores))
best_k = list(K_RANGE)[best_k_idx]
axes[1].axvline(x=best_k, color="red", linestyle="--", alpha=0.6,
                label=f"K={best_k} (mejor)")
axes[1].legend()
plt.suptitle("Selección de K Óptimo", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_01_codo_silhouette.png"), dpi=150)
plt.close()
print(f"\n  [Gráfica guardada] out/cluster_01_codo_silhouette.png")
print(f"  K con mejor silhouette: {best_k}")

# ── 7.2 Entrenar K-Means con K óptimo ─────────────────────────
K_FINAL = 3          # ajusta si el método del codo sugiere otro valor
print(f"\n  Entrenando K-Means con K={K_FINAL} ...")
kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=20, max_iter=500)
cluster_labels = kmeans.fit_predict(X_scaled_df)
df_transformed = df_transformed.loc[X_scaled_df.index].copy()
df_transformed["cluster"] = cluster_labels
X_scaled_df["cluster"] = cluster_labels

sil_final = silhouette_score(X_scaled_df.drop(columns="cluster"), cluster_labels,
                             sample_size=5000, random_state=42)
print(f"  Silhouette score final (K={K_FINAL}): {sil_final:.4f}")

# Perfil de cada cluster
print("\n  Perfil de clusters (mediana de features originales):")
profile = df_transformed.groupby("cluster")[leverage_features].median()
print(profile.T.to_string())

# ── 7.3 Análisis Silhouette por muestra ───────────────────────
X_sil = X_scaled_df.drop(columns="cluster")
sil_vals = silhouette_samples(X_sil, cluster_labels)
fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10
colors = sns.color_palette("Set2", K_FINAL)
for i in range(K_FINAL):
    ith_sil = np.sort(sil_vals[cluster_labels == i])
    size_cluster_i = ith_sil.shape[0]
    y_upper = y_lower + size_cluster_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil,
                     facecolor=colors[i], edgecolor=colors[i], alpha=0.7,
                     label=f"Cluster {i} (n={size_cluster_i:,})")
    y_lower = y_upper + 10

ax.axvline(x=sil_final, color="red", linestyle="--", label=f"Media={sil_final:.3f}")
ax.set_title("Análisis Silhouette por Cluster")
ax.set_xlabel("Coeficiente Silhouette"); ax.set_ylabel("Muestra")
ax.legend(loc="upper right"); ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_02_silhouette_analisis.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/cluster_02_silhouette_analisis.png")

# ── 7.4 Visualización PCA ─────────────────────────────────────
print("\n  Reduciendo dimensionalidad con PCA ...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sil)
var_exp = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(10, 7))
for i in range(K_FINAL):
    mask = cluster_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               s=8, alpha=0.4, color=colors[i], label=f"Cluster {i}")
ax.set_title("Clusters – Visualización PCA")
ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)")
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_03_pca.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/cluster_03_pca.png")

# ── 7.5 Visualización UMAP ────────────────────────────────────
try:
    import umap
    print("  Reduciendo dimensionalidad con UMAP (puede tomar ~2 min) ...")
    UMAP_SAMPLE = min(15000, len(X_sil))
    idx_sample = np.random.default_rng(42).choice(len(X_sil), UMAP_SAMPLE, replace=False)
    X_umap_input = X_sil.values[idx_sample]
    labels_umap  = cluster_labels[idx_sample]

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30,
                        min_dist=0.1, metric="euclidean")
    X_umap = reducer.fit_transform(X_umap_input)

    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(K_FINAL):
        mask = labels_umap == i
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   s=8, alpha=0.4, color=colors[i], label=f"Cluster {i}")
    ax.set_title(f"Clusters – Visualización UMAP (muestra {UMAP_SAMPLE:,})")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_04_umap.png"), dpi=150)
    plt.close()
    print("  [Gráfica guardada] out/cluster_04_umap.png")
    UMAP_AVAILABLE = True
except ImportError:
    print("  [AVISO] umap-learn no instalado. Ejecuta: pip install umap-learn")
    print("          Omitiendo gráfica UMAP.")
    UMAP_AVAILABLE = False

# ── 7.6 Heatmap de perfil de clusters ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
profile_norm = (profile - profile.mean()) / (profile.std() + 1e-9)
sns.heatmap(profile_norm.T, annot=profile.T.round(3), fmt=".3f",
            cmap="RdYlGn_r", center=0, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Perfil de Clusters – Mediana de Ratios de Apalancamiento (normalizado Z)")
ax.set_xlabel("Cluster"); ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_05_perfil_heatmap.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/cluster_05_perfil_heatmap.png")

# ── 7.7 Distribución de clusters por macrosector ──────────────
fig, ax = plt.subplots(figsize=(12, 5))
cross = pd.crosstab(df_transformed["MACROSECTOR"],
                    df_transformed["cluster"],
                    normalize="index") * 100
cross.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
ax.set_title("Distribución de Clusters por Macrosector (%)")
ax.set_xlabel(""); ax.set_ylabel("% dentro del macrosector")
ax.tick_params(axis="x", rotation=30)
ax.legend(title="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_06_distribucion_macrosector.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/cluster_06_distribucion_macrosector.png")

# ─────────────────────────────────────────────────────────────
# 8. CORRECCIÓN DE ETIQUETAS Y MÉTRICAS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CORRECCIÓN DE ETIQUETAS")
print("=" * 60)

# Etiqueta sintética de referencia basada en deuda_activos (proxy de ground truth)
# Bajo apalancamiento: deuda/activos < 0.4 → 0
# Medio              : 0.4 – 0.7         → 1
# Alto               : > 0.7             → 2
def label_leverage(x):
    if x < 0.4:   return 0
    elif x < 0.7: return 1
    else:          return 2

df_transformed["true_label"] = df_transformed["deuda_activos"].apply(label_leverage)

# ── Sin corrección ─────────────────────────────────────────────
y_true = df_transformed["true_label"].values
y_pred_raw = df_transformed["cluster"].values

acc_raw = accuracy_score(y_true, y_pred_raw)
f1_raw  = f1_score(y_true, y_pred_raw, average="weighted", zero_division=0)
prec_raw = precision_score(y_true, y_pred_raw, average="weighted", zero_division=0)
rec_raw  = recall_score(y_true, y_pred_raw, average="weighted", zero_division=0)

print(f"\n  SIN corrección  →  Accuracy={acc_raw:.4f}  F1={f1_raw:.4f}")

# ── Corrección por votación mayoritaria ───────────────────────
from scipy.stats import mode as scipy_mode

mapping = {}
for cluster_id in range(K_FINAL):
    true_labels_in_cluster = y_true[y_pred_raw == cluster_id]
    if len(true_labels_in_cluster) == 0:
        mapping[cluster_id] = cluster_id
    else:
        majority = scipy_mode(true_labels_in_cluster, keepdims=True).mode[0]
        mapping[cluster_id] = majority

print(f"  Mapeo cluster → etiqueta real: {mapping}")
y_pred_corr = np.array([mapping[c] for c in y_pred_raw])

acc_corr  = accuracy_score(y_true, y_pred_corr)
f1_corr   = f1_score(y_true, y_pred_corr, average="weighted", zero_division=0)
prec_corr = precision_score(y_true, y_pred_corr, average="weighted", zero_division=0)
rec_corr  = recall_score(y_true, y_pred_corr, average="weighted", zero_division=0)

print(f"  CON corrección  →  Accuracy={acc_corr:.4f}  F1={f1_corr:.4f}")

# ── 8.1 Comparación de métricas (sin vs con corrección) ───────
metrics = ["Accuracy", "F1", "Precisión", "Recall"]
vals_raw  = [acc_raw,  f1_raw,  prec_raw,  rec_raw]
vals_corr = [acc_corr, f1_corr, prec_corr, rec_corr]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, vals_raw,  width, label="Sin corrección",  color="steelblue",  alpha=0.85)
bars2 = ax.bar(x + width/2, vals_corr, width, label="Con corrección",  color="darkorange", alpha=0.85)
ax.set_ylabel("Score"); ax.set_title("Comparación de Métricas – Sin vs Con Corrección de Etiquetas")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_07_metricas_comparacion.png"), dpi=150)
plt.close()
print("\n  [Gráfica guardada] out/cluster_07_metricas_comparacion.png")

# ── 8.2 Matriz de confusión ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels_names = ["Bajo (0)", "Medio (1)", "Alto (2)"]

for ax, y_pred, title in zip(
        axes,
        [y_pred_raw, y_pred_corr],
        ["Sin corrección", "Con corrección"]):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de Confusión – {title}")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_08_confusion_matrix.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/cluster_08_confusion_matrix.png")

# ── 8.3 Evolución temporal de clusters ────────────────────────
evol_cluster = (df_transformed.groupby(["Año de Corte", "cluster"])
                .size().reset_index(name="count"))
evol_pivot = evol_cluster.pivot(index="Año de Corte", columns="cluster", values="count").fillna(0)
evol_pivot_pct = evol_pivot.div(evol_pivot.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))
for col in evol_pivot_pct.columns:
    ax.plot(evol_pivot_pct.index, evol_pivot_pct[col],
            marker="o", label=f"Cluster {col}", color=colors[col])
ax.set_title("Evolución Temporal de Clusters (% por año)")
ax.set_xlabel("Año"); ax.set_ylabel("% de empresas")
ax.legend(); ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_09_evolucion_temporal.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/cluster_09_evolucion_temporal.png")

# ─────────────────────────────────────────────────────────────
# 9. EXPORTAR DATASET FINAL CON CLUSTERS
# ─────────────────────────────────────────────────────────────
df_transformed["cluster_corregido"] = y_pred_corr
cluster_names = {0: "Bajo Apalancamiento", 1: "Apalancamiento Medio", 2: "Alto Apalancamiento"}
df_transformed["perfil_apalancamiento"] = df_transformed["cluster_corregido"].map(cluster_names)

final_cols = (["NIT", "RAZÓN SOCIAL", "MACROSECTOR", "REGIÓN", "Año de Corte"]
              + leverage_features
              + ["cluster", "cluster_corregido", "perfil_apalancamiento"])
df_final = df_transformed[final_cols]
df_final.to_csv(os.path.join(OUTPUT_DIR, "dataset_final_clusters.csv"), index=False)

print("\n" + "=" * 60)
print("DATASET FINAL EXPORTADO → out/dataset_final_clusters.csv")
print(f"  Shape: {df_final.shape}")
print(f"  Distribución de perfiles:")
print(df_final["perfil_apalancamiento"].value_counts().to_string())
print("=" * 60)
print("\n Pipeline completo (EDA → Preprocesamiento → Clustering) FINALIZADO.")

# ── Resumen final ──────────────────────────────────────────────
print("\n RESUMEN DE OUTPUTS:")
outputs = [
    "eda_01_distribucion_empresas.png",
    "eda_02_distribuciones_financieras.png",
    "eda_03_boxplot_leverage_macrosector.png",
    "eda_04_correlacion.png",
    "eda_05_evolucion_temporal.png",
    "prepro_01_distribuciones_finales.png",
    "prepro_02_pairplot_features.png",
    "prepro_03_correlacion_post.png",
    "cluster_01_codo_silhouette.png",
    "cluster_02_silhouette_analisis.png",
    "cluster_03_pca.png",
    "cluster_04_umap.png       (requiere: pip install umap-learn)",
    "cluster_05_perfil_heatmap.png",
    "cluster_06_distribucion_macrosector.png",
    "cluster_07_metricas_comparacion.png",
    "cluster_08_confusion_matrix.png",
    "cluster_09_evolucion_temporal.png",
    "dataset_procesado_apalancamiento.csv",
    "dataset_final_clusters.csv",
]
for f in outputs:
    print(f"  out/{f}")