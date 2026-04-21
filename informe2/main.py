"""
Pipeline Completo de ML – 10,000 Empresas más Grandes del País
=================================================================
Etapas:
  1. Carga y preprocesamiento (mismo que el script original)
  2. Clustering NO supervisado: K-Means, Fuzzy C-Means, Subtractive,
     DBSCAN y familia Gaussian Mixture (GMM)
  3. Re-evaluación de etiquetas (corrección ~30 % mal asignadas)
  4. Modelos supervisados: Árbol de Decisión, Regresión Logística,
     Regresión Lineal (OvR)
  5. Comparación: modelos entrenados con etiquetas corregidas vs.
     etiquetas originales del clustering
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, r2_score
)
from sklearn.neighbors import NearestNeighbors

try:
    import skfuzzy as fuzz
    HAS_SKFUZZY = True
except ImportError:
    HAS_SKFUZZY = False
    print("  [AVISO] skfuzzy no disponible – se omite Fuzzy C-Means")

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════
FILE_PATH = "C:/Users/jespa/OneDrive/Desktop/ESPANA_JAIDER/informe2/10.000_Empresas_mas_Grandes_del_País_20260210.csv"

print("=" * 65)
print("1. CARGA DE DATOS")
print("=" * 65)

df_raw = pd.read_csv(FILE_PATH)
print(f"  Filas   : {df_raw.shape[0]:,}")
print(f"  Columnas: {df_raw.shape[1]}")

# ═══════════════════════════════════════════════════════════════
# 2. LIMPIEZA Y CONVERSIÓN DE TIPOS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("2. LIMPIEZA Y CONVERSIÓN DE TIPOS")
print("=" * 65)

df = df_raw.copy()
monetary_cols = [
    "INGRESOS OPERACIONALES", "GANANCIA (PÉRDIDA)",
    "TOTAL ACTIVOS", "TOTAL PASIVOS", "TOTAL PATRIMONIO"
]

for col in monetary_cols:
    df[col] = (
        df[col].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )

df["Año de Corte"] = df["Año de Corte"].astype(str).str.replace(",", "").astype(int)
df["NIT"] = df["NIT"].astype(str).str.replace(",", "").str.strip()

print(f"  Nulos en columnas monetarias:\n{df[monetary_cols].isnull().sum().to_string()}")

# ═══════════════════════════════════════════════════════════════
# 3. INGENIERÍA DE CARACTERÍSTICAS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("3. INGENIERÍA DE CARACTERÍSTICAS DE APALANCAMIENTO")
print("=" * 65)

df_feat = df.copy()
eps = 1e-9

df_feat["deuda_activos"]      = df_feat["TOTAL PASIVOS"] / (df_feat["TOTAL ACTIVOS"] + eps)
df_feat["deuda_patrimonio"]   = df_feat["TOTAL PASIVOS"] / (df_feat["TOTAL PATRIMONIO"].abs() + eps)
df_feat["multiplicador_cap"]  = df_feat["TOTAL ACTIVOS"] / (df_feat["TOTAL PATRIMONIO"].abs() + eps)
df_feat["cobertura_ingresos"] = df_feat["INGRESOS OPERACIONALES"] / (df_feat["TOTAL PASIVOS"] + eps)
df_feat["margen_neto"]        = df_feat["GANANCIA (PÉRDIDA)"] / (df_feat["INGRESOS OPERACIONALES"].abs() + eps)
df_feat["roa"]                = df_feat["GANANCIA (PÉRDIDA)"] / (df_feat["TOTAL ACTIVOS"] + eps)
df_feat["roe"]                = df_feat["GANANCIA (PÉRDIDA)"] / (df_feat["TOTAL PATRIMONIO"].abs() + eps)

leverage_features = [
    "deuda_activos", "deuda_patrimonio", "multiplicador_cap",
    "cobertura_ingresos", "margen_neto", "roa", "roe"
]
print(f"  Características: {leverage_features}")

# ═══════════════════════════════════════════════════════════════
# 4. PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("4. PREPROCESAMIENTO")
print("=" * 65)

n_before = len(df_feat)
df_clean = df_feat[df_feat["TOTAL ACTIVOS"] > 0].copy()
print(f"  Filas con TOTAL ACTIVOS > 0: {len(df_clean):,}  (removidas: {n_before - len(df_clean):,})")

def remove_outliers_iqr(df, cols, factor=5.0):
    mask = pd.Series(True, index=df.index)
    for col in cols:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask &= df[col].between(lower, upper)
    return df[mask]

df_clean = remove_outliers_iqr(df_clean, leverage_features, factor=5.0)
print(f"  Filas tras outliers (IQR×5): {len(df_clean):,}")

for col in leverage_features:
    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

def log_modulus(x):
    return np.sign(x) * np.log1p(np.abs(x))

df_transformed = df_clean.copy()
for col in leverage_features:
    df_transformed[col + "_lm"] = log_modulus(df_transformed[col])

transformed_cols = [c + "_lm" for c in leverage_features]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_transformed[transformed_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=transformed_cols, index=df_transformed.index)

print(f"  Shape final para clustering: {X_scaled_df.shape}")

# PCA para visualizaciones 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"  Varianza explicada PCA 2D: {pca.explained_variance_ratio_.sum():.1%}")

# ═══════════════════════════════════════════════════════════════
# 5. CLUSTERING NO SUPERVISADO
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("5. CLUSTERING NO SUPERVISADO")
print("=" * 65)

N_CLUSTERS = 4   # ajustar según elbow/silhouette
cluster_results = {}   # nombre → etiquetas (array)

# ── 5.1 K-Means con búsqueda de k óptimo ──────────────────────
print("\n  [5.1] K-Means – búsqueda de k óptimo (2-9)…")
inertias, silhouettes = [], []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, lbl, sample_size=3000, random_state=42))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, "o-", color="steelblue")
axes[0].set_title("K-Means – Elbow (Inercia)")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inercia")
axes[1].plot(list(k_range), silhouettes, "o-", color="coral")
axes[1].set_title("K-Means – Silhouette Score")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_01_kmeans_elbow.png"), dpi=150)
plt.close()

best_k = list(k_range)[np.argmax(silhouettes)]
print(f"  Mejor k por silhouette: {best_k}  (silhouette={max(silhouettes):.3f})")
N_CLUSTERS = best_k

km_final = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
cluster_results["KMeans"] = km_final.fit_predict(X_scaled)
print(f"  K-Means final → silhouette={silhouette_score(X_scaled, cluster_results['KMeans'], sample_size=3000, random_state=42):.3f}")

# ── 5.2 Fuzzy C-Means ─────────────────────────────────────────
if HAS_SKFUZZY:
    print("\n  [5.2] Fuzzy C-Means…")
    X_t = X_scaled.T                       # skfuzzy requiere (features, samples)
    cntr, u, *_ = fuzz.cluster.cmeans(
        X_t, c=N_CLUSTERS, m=2, error=0.005, maxiter=500, init=None
    )
    fcm_labels = np.argmax(u, axis=0)
    cluster_results["FuzzyCMeans"] = fcm_labels
    print(f"  FCM → silhouette={silhouette_score(X_scaled, fcm_labels, sample_size=3000, random_state=42):.3f}")
else:
    print("\n  [5.2] Fuzzy C-Means omitido (skfuzzy no disponible).")

# ── 5.3 Subtractive Clustering (implementación propia) ────────
print("\n  [5.3] Subtractive Clustering…")

def subtractive_clustering(X, r_a=0.5, r_b=None, accept_ratio=0.5, reject_ratio=0.15):
    """
    Algoritmo de Chiu (1994) – versión simplificada sobre datos escalados.
    Devuelve centros y etiquetas por distancia mínima al centro.
    """
    if r_b is None:
        r_b = r_a * 1.5
    n, d = X.shape
    # Densidad potencial
    potential = np.array([
        np.sum(np.exp(-np.linalg.norm(X - X[i], axis=1) ** 2 / (r_a / 2) ** 2))
        for i in range(n)
    ])

    centers = []
    p_max_init = potential.max()
    p_current = potential.copy()
    threshold = accept_ratio * p_max_init

    for _ in range(20):   # máximo 20 centros
        best_idx = np.argmax(p_current)
        best_p = p_current[best_idx]
        if best_p < reject_ratio * p_max_init:
            break
        if best_p >= threshold:
            centers.append(X[best_idx])
        else:
            break
        # Reducir potencial de puntos cercanos
        dist2 = np.linalg.norm(X - X[best_idx], axis=1) ** 2
        p_current -= best_p * np.exp(-dist2 / (r_b / 2) ** 2)

    if len(centers) == 0:
        centers = [X[np.argmax(potential)]]

    centers = np.array(centers)
    labels = np.argmin(
        np.stack([np.linalg.norm(X - c, axis=1) for c in centers], axis=1), axis=1
    )
    return centers, labels

# Usar muestra para que sea viable en memoria
sample_idx = np.random.RandomState(42).choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)
X_sub = X_scaled[sample_idx]
_, sub_labels_sample = subtractive_clustering(X_sub, r_a=0.6)
n_sub_clusters = len(np.unique(sub_labels_sample))

# Extender a todo el dataset asignando al centroide más cercano
# (Re-compute centers from sample)
sub_centers = np.array([
    X_sub[sub_labels_sample == c].mean(axis=0)
    for c in np.unique(sub_labels_sample)
])
sub_labels_full = np.argmin(
    np.stack([np.linalg.norm(X_scaled - c, axis=1) for c in sub_centers], axis=1), axis=1
)
cluster_results["Subtractive"] = sub_labels_full
print(f"  Subtractive → {n_sub_clusters} clusters, silhouette={silhouette_score(X_scaled, sub_labels_full, sample_size=3000, random_state=42):.3f}")

# ── 5.4 DBSCAN ────────────────────────────────────────────────
print("\n  [5.4] DBSCAN – estimación de eps con k-NN…")
nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
knn_dist = np.sort(distances[:, 4])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(knn_dist, color="darkorchid")
ax.set_title("k-NN Distance (k=5) – Estimación eps para DBSCAN")
ax.set_xlabel("Puntos ordenados"); ax.set_ylabel("Distancia al 5° vecino")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_02_dbscan_knn.png"), dpi=150)
plt.close()

# Usar el percentil 90 como eps automático
eps_auto = float(np.percentile(knn_dist, 90))
db = DBSCAN(eps=eps_auto, min_samples=10, n_jobs=-1)
db_labels = db.fit_predict(X_scaled)
n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
noise_pct = (db_labels == -1).mean() * 100
print(f"  DBSCAN eps={eps_auto:.3f} → {n_db_clusters} clusters, ruido={noise_pct:.1f}%")

if n_db_clusters > 1:
    mask_valid = db_labels != -1
    sil_db = silhouette_score(X_scaled[mask_valid], db_labels[mask_valid], sample_size=3000, random_state=42)
    print(f"  DBSCAN silhouette (sin ruido)={sil_db:.3f}")
    cluster_results["DBSCAN"] = db_labels
else:
    print("  DBSCAN produjo un solo cluster – se omite del conjunto de resultados")

# ── 5.5 Familia GMM (Gaussian Mixture Models) ─────────────────
print("\n  [5.5] Gaussian Mixture Models (GMM)…")
covariance_types = ["full", "tied", "diag", "spherical"]
gmm_results = {}

for cov_type in covariance_types:
    gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type=cov_type,
                          random_state=42, max_iter=300)
    gmm.fit(X_scaled)
    lbl = gmm.predict(X_scaled)
    bic = gmm.bic(X_scaled)
    sil = silhouette_score(X_scaled, lbl, sample_size=3000, random_state=42)
    gmm_results[cov_type] = {"labels": lbl, "bic": bic, "silhouette": sil}
    print(f"  GMM ({cov_type:10s}) → BIC={bic:,.0f}  silhouette={sil:.3f}")

best_gmm_type = min(gmm_results, key=lambda k: gmm_results[k]["bic"])
cluster_results[f"GMM_{best_gmm_type}"] = gmm_results[best_gmm_type]["labels"]
print(f"\n  Mejor GMM (BIC mínimo): covariance_type='{best_gmm_type}'")

# ── 5.6 Resumen de métricas de clustering ─────────────────────
print("\n" + "-" * 65)
print("  RESUMEN CLUSTERING")
print(f"  {'Método':<20} {'Clusters':>8} {'Silhouette':>11} {'Davies-Bouldin':>15} {'Calinski-H':>12}")
print("  " + "-" * 68)

metrics_rows = []
for name, labels in cluster_results.items():
    valid_mask = labels != -1
    n_c = len(set(labels[valid_mask]))
    if n_c < 2:
        continue
    X_v = X_scaled[valid_mask]
    l_v = labels[valid_mask]
    sil = silhouette_score(X_v, l_v, sample_size=3000, random_state=42)
    db_s = davies_bouldin_score(X_v, l_v)
    ch  = calinski_harabasz_score(X_v, l_v)
    print(f"  {name:<20} {n_c:>8} {sil:>11.3f} {db_s:>15.3f} {ch:>12.1f}")
    metrics_rows.append({"Método": name, "Clusters": n_c, "Silhouette": sil,
                          "Davies-Bouldin": db_s, "Calinski-H": ch})

df_metrics = pd.DataFrame(metrics_rows)
df_metrics.to_csv(os.path.join(OUTPUT_DIR, "cluster_metricas.csv"), index=False)

# ── 5.7 Visualización PCA de todos los métodos ────────────────
n_methods = len(cluster_results)
fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(5 * ((n_methods + 1) // 2), 9))
axes = axes.flatten()

for idx, (name, labels) in enumerate(cluster_results.items()):
    scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                                 cmap="tab10", s=3, alpha=0.5)
    axes[idx].set_title(name)
    axes[idx].set_xlabel("PC1"); axes[idx].set_ylabel("PC2")

for j in range(idx + 1, len(axes)):
    axes[j].axis("off")

plt.suptitle("Clustering NO Supervisado – Proyección PCA 2D", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_03_pca_todos.png"), dpi=150)
plt.close()
print("\n  [Gráfica guardada] cluster_03_pca_todos.png")

# ═══════════════════════════════════════════════════════════════
# 6. RE-EVALUACIÓN DE ETIQUETAS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("6. RE-EVALUACIÓN DE ETIQUETAS (corrección ~30 % mal asignadas)")
print("=" * 65)

# Elegir el método con mejor silhouette como etiqueta base
best_method = df_metrics.sort_values("Silhouette", ascending=False).iloc[0]["Método"]
print(f"\n  Método base seleccionado: {best_method}")

labels_base = cluster_results[best_method].copy()

# --- Paso 1: Re-asignación por kNN --------------------------------
# Para cada punto, calculamos si su cluster asignado coincide con
# el cluster mayoritario de sus K vecinos más cercanos.
# Si no coincide (conf < umbral), reasignamos.
print("  Ejecutando re-evaluación por k-NN (k=15, umbral=0.40)…")

K_EVAL = 15
THRESHOLD = 0.40   # si el cluster propio tiene < 40 % del vecindario → reasignar

nbrs15 = NearestNeighbors(n_neighbors=K_EVAL + 1).fit(X_scaled)
nn_indices = nbrs15.kneighbors(X_scaled, return_distance=False)[:, 1:]  # excluir el propio

labels_corrected = labels_base.copy()
reassigned = 0

for i in range(len(X_scaled)):
    neighbor_labels = labels_base[nn_indices[i]]
    own_label = labels_base[i]
    counts = np.bincount(neighbor_labels[neighbor_labels >= 0],
                         minlength=int(labels_base.max()) + 1)
    own_conf = counts[own_label] / counts.sum() if counts.sum() > 0 else 0
    if own_conf < THRESHOLD:
        labels_corrected[i] = np.argmax(counts)
        reassigned += 1

pct_reassigned = 100 * reassigned / len(labels_corrected)
print(f"  Puntos re-asignados: {reassigned:,} ({pct_reassigned:.1f} %)")

# Almacenar ambas versiones
df_transformed["label_original"] = labels_base
df_transformed["label_corrected"] = labels_corrected

# Visualización antes/después
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, lbl, title in zip(
    axes,
    [labels_base, labels_corrected],
    ["Etiquetas Originales (clustering)", "Etiquetas Corregidas (k-NN)"]
):
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=lbl, cmap="tab10", s=3, alpha=0.5)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

plt.suptitle("Re-evaluación de Etiquetas", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "relabel_01_antes_despues.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] relabel_01_antes_despues.png")

# Métricas de coherencia post-corrección
sil_orig = silhouette_score(X_scaled, labels_base, sample_size=3000, random_state=42)
sil_corr = silhouette_score(X_scaled, labels_corrected, sample_size=3000, random_state=42)
print(f"\n  Silhouette antes: {sil_orig:.4f}  →  después: {sil_corr:.4f}")

# ═══════════════════════════════════════════════════════════════
# 7. MODELOS SUPERVISADOS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("7. MODELOS SUPERVISADOS")
print("=" * 65)

# Preparar X e y (dos versiones de etiquetas)
X_sup = X_scaled_df.loc[df_transformed.index].values
y_orig = labels_base
y_corr = labels_corrected

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_classifier(name, clf, X, y, cv):
    """Retorna métricas de clasificación por CV."""
    acc_scores  = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    f1_scores   = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted")
    return {
        "Modelo": name,
        "ACC_mean": acc_scores.mean(),
        "ACC_std":  acc_scores.std(),
        "F1_mean":  f1_scores.mean(),
        "F1_std":   f1_scores.std(),
    }

models = {
    "DecisionTree (depth=6)": DecisionTreeClassifier(max_depth=6, random_state=42),
    "DecisionTree (depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42),
}

# Regresión Lineal OvR (clasificación mediante umbral)
class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):
    """Regresión lineal OvR (One-vs-Rest) usada como clasificador."""

    def __init__(self):
        self.models_ = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}

        for c in self.classes_:
            lr = LinearRegression()
            lr.fit(X, (y == c).astype(float))
            self.models_[c] = lr

        return self

    def predict(self, X):
        scores = np.column_stack([
            self.models_[c].predict(X) for c in self.classes_
        ])
        return self.classes_[np.argmax(scores, axis=1)]

models["LinearRegression (OvR)"] = LinearRegressionClassifier()

# Evaluación con etiquetas CORREGIDAS y ORIGINALES
rows_corr, rows_orig = [], []

print(f"\n  {'Modelo':<30} {'ACC (corr)':>12} {'F1 (corr)':>11} {'ACC (orig)':>12} {'F1 (orig)':>11}")
print("  " + "-" * 78)

for model_name, clf in models.items():
    import copy
    r_corr = evaluate_classifier(model_name, copy.deepcopy(clf), X_sup, y_corr, CV)
    r_orig = evaluate_classifier(model_name, copy.deepcopy(clf), X_sup, y_orig, CV)
    rows_corr.append(r_corr)
    rows_orig.append(r_orig)
    print(
        f"  {model_name:<30} "
        f"{r_corr['ACC_mean']:>10.3f}±{r_corr['ACC_std']:.2f} "
        f"{r_corr['F1_mean']:>9.3f}±{r_corr['F1_std']:.2f} "
        f"{r_orig['ACC_mean']:>10.3f}±{r_orig['ACC_std']:.2f} "
        f"{r_orig['F1_mean']:>9.3f}±{r_orig['F1_std']:.2f}"
    )

df_results_corr = pd.DataFrame(rows_corr)
df_results_orig = pd.DataFrame(rows_orig)

# ── Guardar reporte de clasificación (fit completo) ───────────
from sklearn.model_selection import train_test_split

X_tr, X_te, yc_tr, yc_te = train_test_split(X_sup, y_corr, test_size=0.25,
                                               random_state=42, stratify=y_corr)
_, _, yo_tr, yo_te = train_test_split(X_sup, y_orig, test_size=0.25,
                                        random_state=42, stratify=y_orig)

report_lines = []
for model_name, clf in models.items():
    import copy
    # Etiquetas corregidas
    clf_c = copy.deepcopy(clf).fit(X_tr, yc_tr)
    yhat_c = clf_c.predict(X_te)
    # Etiquetas originales
    clf_o = copy.deepcopy(clf).fit(X_tr, yo_tr)
    yhat_o = clf_o.predict(X_te)

    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"Modelo: {model_name}")
    report_lines.append(f"\n--- ETIQUETAS CORREGIDAS ---")
    report_lines.append(classification_report(yc_te, yhat_c))
    report_lines.append(f"\n--- ETIQUETAS ORIGINALES ---")
    report_lines.append(classification_report(yo_te, yhat_o))

with open(os.path.join(OUTPUT_DIR, "sup_classification_reports.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\n  [Reporte guardado] sup_classification_reports.txt")

# ── Árbol de Decisión – reglas interpretables ─────────────────
dt_final = DecisionTreeClassifier(max_depth=6, random_state=42).fit(X_tr, yc_tr)
tree_rules = export_text(dt_final, feature_names=transformed_cols)
with open(os.path.join(OUTPUT_DIR, "sup_decision_tree_rules.txt"), "w", encoding="utf-8") as f:
    f.write(tree_rules)
print("  [Reglas guardadas] sup_decision_tree_rules.txt")

# ── Importancia de variables (Árbol) ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
importances = pd.Series(dt_final.feature_importances_, index=transformed_cols).sort_values()
importances.plot(kind="barh", ax=ax, color="teal", edgecolor="white")
ax.set_title("Importancia de Variables – Decision Tree (etiquetas corregidas)")
ax.set_xlabel("Importancia (Gini)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sup_01_feature_importance.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] sup_01_feature_importance.png")

# ═══════════════════════════════════════════════════════════════
# 8. COMPARACIÓN: etiquetas corregidas vs. originales
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("8. COMPARACIÓN – Etiquetas Corregidas vs. Originales")
print("=" * 65)

df_compare = df_results_corr.merge(
    df_results_orig, on="Modelo", suffixes=("_CORR", "_ORIG")
)[["Modelo", "ACC_mean_CORR", "F1_mean_CORR", "ACC_mean_ORIG", "F1_mean_ORIG"]]
df_compare["Delta_ACC"] = df_compare["ACC_mean_CORR"] - df_compare["ACC_mean_ORIG"]
df_compare["Delta_F1"]  = df_compare["F1_mean_CORR"]  - df_compare["F1_mean_ORIG"]

print(df_compare.to_string(index=False))
df_compare.to_csv(os.path.join(OUTPUT_DIR, "comparacion_resultados.csv"), index=False)

# Gráfica comparativa
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(df_compare))
width = 0.35

for ax, metric_pair, title in zip(
    axes,
    [("ACC_mean_CORR", "ACC_mean_ORIG"), ("F1_mean_CORR", "F1_mean_ORIG")],
    ["Accuracy", "F1 Weighted"]
):
    bars1 = ax.bar(x - width/2, df_compare[metric_pair[0]], width, label="Corregidas", color="steelblue")
    bars2 = ax.bar(x + width/2, df_compare[metric_pair[1]], width, label="Originales", color="coral")
    ax.set_title(f"Comparación – {title}")
    ax.set_xticks(x)
    ax.set_xticklabels(df_compare["Modelo"], rotation=15, ha="right")
    ax.set_ylabel(title)
    ax.legend()
    ax.set_ylim(0, 1.05)

plt.suptitle("Modelos Supervisados: Etiquetas Corregidas vs. Originales (CV 5-fold)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparacion_01_corregidas_vs_originales.png"), dpi=150)
plt.close()
print("\n  [Gráfica guardada] comparacion_01_corregidas_vs_originales.png")

# ── Matriz de confusión del mejor modelo (etiquetas corregidas) ──
best_model_name = df_results_corr.sort_values("F1_mean", ascending=False).iloc[0]["Modelo"]
print(f"\n  Mejor modelo (F1 corregidas): {best_model_name}")

import copy
best_clf = copy.deepcopy(models[best_model_name]).fit(X_tr, yc_tr)
yhat_best = best_clf.predict(X_te)
cm = confusion_matrix(yc_te, yhat_best)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title(f"Matriz de Confusión – {best_model_name}\n(etiquetas corregidas)")
ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparacion_02_confusion_matrix.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] comparacion_02_confusion_matrix.png")

# ═══════════════════════════════════════════════════════════════
# 9. EXPORTAR DATASET FINAL
# ═══════════════════════════════════════════════════════════════
output_df = df_transformed[
    ["NIT", "RAZÓN SOCIAL", "MACROSECTOR", "REGIÓN", "Año de Corte"]
    + leverage_features + ["label_original", "label_corrected"]
].copy()
output_df = output_df.loc[X_scaled_df.index]
output_df[transformed_cols] = X_scaled_df.values

output_df.to_csv(os.path.join(OUTPUT_DIR, "dataset_final_con_clusters.csv"), index=False)

print("\n" + "=" * 65)
print("PIPELINE COMPLETADO")
print("=" * 65)
print(f"  Dataset final → out/dataset_final_con_clusters.csv  ({output_df.shape})")
print("  Archivos generados en carpeta out/:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"    {f:<55} {size/1024:>6.1f} KB")