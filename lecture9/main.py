# =============================================================
#  FIRE-UdeA — Pipeline ML completo + Clustering K-Means
#  Objetivo: superar métricas baseline (test ROC-AUC=0.417,
#            log_loss=4.87, brier=0.257)
#  Clustering: K-Means con métricas Euclidiana, Manhattan,
#              Coseno y Mahalanobis (método del codo incluido)
# =============================================================

# ── 0. Instalación ───────────────────────────────────────────
# pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn scipy

import warnings
warnings.filterwarnings('ignore')

import os
OUT = 'out'
os.makedirs(OUT, exist_ok=True)
print(f"[OK] Carpeta de salida: '{OUT}/'")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, log_loss, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              silhouette_score, pairwise_distances)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.spatial import distance

plt.rc('font', family='serif', size=11)

# ─────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  FIRE-UdeA — Pipeline ML + Clustering K-Means")
print("=" * 60)

df1       = pd.read_csv('dataset_sintetico_FIRE_UdeA.csv')
df2       = pd.read_csv('dataset_sintetico_FIRE_UdeA_realista.csv')
baseline  = pd.read_csv('reporte_metricas_FIRE_UdeA_realista.csv')
scores_bl = pd.read_csv('scores_test_FIRE_UdeA_realista.csv')

df1 = df1.drop(columns=['participacion_ley30'], errors='ignore')
df2 = df2.drop(columns=['participacion_ley30'], errors='ignore')

print(f"\n[INFO] df1 (sintético):  {df1.shape}")
print(f"[INFO] df2 (realista):   {df2.shape}")

# ─────────────────────────────────────────────────────────────
# 2. PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────
print("\n--- Preprocesamiento ---")

cols_excluir   = ['anio', 'unidad', 'label']
num_cols_df2   = [c for c in df2.select_dtypes(include='number').columns
                  if c not in cols_excluir]

knn_imputer    = KNNImputer(n_neighbors=5)
df2_imp        = df2.copy()
df2_imp[num_cols_df2] = knn_imputer.fit_transform(df2[num_cols_df2])
print(f"[OK] KNN imputation — nulos restantes: {df2_imp[num_cols_df2].isnull().sum().sum()}")

def feature_engineering(df):
    d = df.copy()
    if 'ingresos_totales' in d.columns:
        d['cfo_ratio'] = d['cfo'] / d['ingresos_totales'].replace(0, np.nan)
    d['liquidez_x_dias']  = d['liquidez'] * d['dias_efectivo']
    d['tension_signal']   = (d['cfo'] < 0).astype(int) + (d['liquidez'] < 1).astype(int)
    d['diversificacion']  = 1 - d['hhi_fuentes']
    if 'participacion_servicios' in d.columns and 'participacion_matriculas' in d.columns:
        d['participacion_propia'] = d['participacion_servicios'] + d['participacion_matriculas']
    if 'ingresos_totales' in d.columns:
        d['log_ingresos']       = np.log1p(d['ingresos_totales'])
        d['log_gastos_personal'] = np.log1p(d['gastos_personal'])
    return d

df2_fe = feature_engineering(df2_imp)
df1_fe = feature_engineering(df1)

FEATURES_DF2 = [
    'liquidez', 'dias_efectivo', 'cfo', 'hhi_fuentes',
    'endeudamiento', 'tendencia_ingresos', 'gp_ratio',
    'participacion_regalias', 'participacion_servicios',
    'participacion_matriculas',
    'cfo_ratio', 'liquidez_x_dias', 'tension_signal',
    'diversificacion', 'participacion_propia',
    'log_ingresos', 'log_gastos_personal'
]
FEATURES_DF2 = [f for f in FEATURES_DF2 if f in df2_fe.columns]

FEATURES_DF1 = [
    'liquidez', 'dias_efectivo', 'cfo', 'hhi_fuentes',
    'gastos_personal', 'tendencia_ingresos',
    'tension_signal', 'diversificacion'
]
FEATURES_DF1 = [f for f in FEATURES_DF1 if f in df1_fe.columns]

print(f"[OK] Features df2: {len(FEATURES_DF2)} | Features df1: {len(FEATURES_DF1)}")

# Split temporal
anios      = sorted(df2_fe['anio'].unique())
anio_test  = anios[-1]
anio_valid = anios[-2]

mask_test  = df2_fe['anio'] == anio_test
mask_valid = df2_fe['anio'] == anio_valid
mask_train = ~(mask_test | mask_valid)

X2   = df2_fe[FEATURES_DF2]
y2   = df2_fe['label']
meta_test = df2_fe[mask_test][['anio', 'unidad']].reset_index(drop=True)

X_train_raw = X2[mask_train.values].reset_index(drop=True)
X_valid_raw = X2[mask_valid.values].reset_index(drop=True)
X_test_raw  = X2[mask_test.values].reset_index(drop=True)
y_train = y2[mask_train.values].reset_index(drop=True)
y_valid = y2[mask_valid.values].reset_index(drop=True)
y_test  = y2[mask_test.values].reset_index(drop=True)

scaler  = RobustScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=FEATURES_DF2)
X_valid = pd.DataFrame(scaler.transform(X_valid_raw),     columns=FEATURES_DF2)
X_test  = pd.DataFrame(scaler.transform(X_test_raw),      columns=FEATURES_DF2)

X1 = pd.DataFrame(RobustScaler().fit_transform(df1_fe[FEATURES_DF1]), columns=FEATURES_DF1)
y1 = df1_fe['label']

print(f"[OK] Train={X_train.shape} | Valid={X_valid.shape} | Test={X_test.shape}")
print(f"     Prevalencia  train={y_train.mean():.2f} | valid={y_valid.mean():.2f} | test={y_test.mean():.2f}")

# ─────────────────────────────────────────────────────────────
# 3. FUNCIONES DE EVALUACIÓN SUPERVISADA
# ─────────────────────────────────────────────────────────────
def evaluar(nombre, y_true, y_prob, y_pred, split, n):
    prevalencia = y_true.mean()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'modelo':      nombre,
        'split':       split,
        'n':           n,
        'prevalencia': round(prevalencia, 3),
        'roc_auc':     round(roc_auc_score(y_true, y_prob), 4),
        'pr_auc':      round(average_precision_score(y_true, y_prob), 4),
        'brier':       round(brier_score_loss(y_true, y_prob), 4),
        'log_loss':    round(log_loss(y_true, y_prob), 4),
        'precision':   round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':      round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1':          round(f1_score(y_true, y_pred, zero_division=0), 4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def evaluar_modelo(nombre, model, umbral=0.5):
    rows = []
    for split_name, Xs, ys in [('train', X_train, y_train),
                                ('valid', X_valid, y_valid),
                                ('test',  X_test,  y_test)]:
        prob = model.predict_proba(Xs)[:, 1]
        pred = (prob >= umbral).astype(int)
        rows.append(evaluar(nombre, ys, prob, pred, split_name, len(ys)))
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# 4. MODELOS SUPERVISADOS
# ─────────────────────────────────────────────────────────────
print("\n--- Entrenamiento de modelos supervisados ---")
resultados = []

lr     = LogisticRegression(C=0.1, max_iter=1000, random_state=42,
                             class_weight='balanced', solver='lbfgs')
lr_cal = CalibratedClassifierCV(lr, cv=3, method='isotonic')
lr_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('LogReg_calibrada', lr_cal))
print("[OK] Logistic Regression calibrada")

rf     = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                 max_features='sqrt', class_weight='balanced', random_state=42)
rf_cal = CalibratedClassifierCV(rf, cv=3, method='isotonic')
rf_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('RandomForest_calibrado', rf_cal))
print("[OK] Random Forest calibrado")

gb     = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                     min_samples_leaf=3, subsample=0.8, random_state=42)
gb_cal = CalibratedClassifierCV(gb, cv=3, method='isotonic')
gb_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('GradientBoosting_calibrado', gb_cal))
print("[OK] Gradient Boosting calibrado")

svm     = SVC(C=1.0, kernel='rbf', class_weight='balanced',
              probability=False, random_state=42)
svm_cal = CalibratedClassifierCV(svm, cv=3, method='isotonic')
svm_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('SVM_calibrado', svm_cal))
print("[OK] SVM calibrado")

knn     = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
knn_cal = CalibratedClassifierCV(knn, cv=3, method='isotonic')
knn_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('KNN_calibrado', knn_cal))
print("[OK] KNN calibrado")

lr_base = LogisticRegression(C=0.1, max_iter=1000, random_state=42, class_weight='balanced')
rf_base = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                  class_weight='balanced', random_state=42)
gb_base = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                      min_samples_leaf=3, subsample=0.8, random_state=42)
voting  = VotingClassifier(estimators=[('lr', lr_base), ('rf', rf_base), ('gb', gb_base)],
                            voting='soft', weights=[1, 2, 2])
voting_cal = CalibratedClassifierCV(voting, cv=3, method='isotonic')
voting_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('Voting_Ensemble', voting_cal))
print("[OK] Voting Ensemble calibrado")

# ─────────────────────────────────────────────────────────────
# 5. COMPARACIÓN SUPERVISADA
# ─────────────────────────────────────────────────────────────
print("\n--- Resultados supervisados ---")
df_res   = pd.concat(resultados, ignore_index=True)
test_res = df_res[df_res['split'] == 'test'].copy().sort_values('roc_auc', ascending=False)
bl_test  = baseline[baseline['split'] == 'test'].iloc[0]

print(f"\n{'─'*65}")
print(f"  BASELINE TEST → ROC-AUC={bl_test['roc_auc']:.4f} | "
      f"PR-AUC={bl_test['pr_auc']:.4f} | "
      f"Brier={bl_test['brier']:.4f} | "
      f"Log-Loss={bl_test['log_loss']:.4f}")
print(f"{'─'*65}")
print(f"\n{'Modelo':<30} {'ROC-AUC':>8} {'PR-AUC':>8} {'Brier':>7} {'LogLoss':>9} {'F1':>6}")
print("─" * 75)
for _, row in test_res.iterrows():
    supera = "✅" if row['roc_auc'] > bl_test['roc_auc'] else "❌"
    print(f"{row['modelo']:<30} {row['roc_auc']:>8.4f} {row['pr_auc']:>8.4f} "
          f"{row['brier']:>7.4f} {row['log_loss']:>9.4f} {row['f1']:>6.4f}  {supera}")

mejor = test_res.iloc[0]
modelos_map = {
    'LogReg_calibrada':           lr_cal,
    'RandomForest_calibrado':     rf_cal,
    'GradientBoosting_calibrado': gb_cal,
    'SVM_calibrado':              svm_cal,
    'KNN_calibrado':              knn_cal,
    'Voting_Ensemble':            voting_cal,
}
nombre_mejor = mejor['modelo']
best_model   = modelos_map[nombre_mejor]
prob_test    = best_model.predict_proba(X_test)[:, 1]
pred_test    = (prob_test >= 0.5).astype(int)

scores_nuevo        = meta_test.copy()
scores_nuevo['y_true'] = y_test.values
scores_nuevo['prob']   = prob_test.round(4)
scores_nuevo['pred']   = pred_test

print(f"\n🏆 Mejor modelo en test: {nombre_mejor}")
print(f"   ROC-AUC  : {mejor['roc_auc']:.4f}  (baseline: {bl_test['roc_auc']:.4f})")
print(f"   PR-AUC   : {mejor['pr_auc']:.4f}  (baseline: {bl_test['pr_auc']:.4f})")
print(f"   Brier    : {mejor['brier']:.4f}  (baseline: {bl_test['brier']:.4f})")
print(f"   Log-Loss : {mejor['log_loss']:.4f}  (baseline: {bl_test['log_loss']:.4f})")
print(f"   F1       : {mejor['f1']:.4f}  (baseline: {bl_test['f1']:.4f})")

# ─────────────────────────────────────────────────────────────
# 6. CLUSTERING K-MEANS — MÚLTIPLES MÉTRICAS DE DISTANCIA
# ─────────────────────────────────────────────────────────────
#
#  Estrategia:
#  • K-Means estándar opera sobre distancia Euclidiana.
#  • Para Manhattan y Coseno se pre-transforma el espacio de
#    features antes de aplicar K-Means (equivalente a minimizar
#    esa distancia en el espacio original).
#  • Para Mahalanobis se aplica una transformación de blanqueo
#    (whitening) sobre los datos; K-Means euclidiano en ese
#    espacio es equivalente a K-Means mahalanobis en el original.
#  • Se usa el conjunto de entrenamiento completo (train+valid)
#    para el clustering, ya que es no supervisado.
#  • Se evalúa con: Inercia, Silhouette Score y distribución
#    de etiquetas reales por cluster.
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CLUSTERING K-MEANS — MÚLTIPLES MÉTRICAS")
print("=" * 60)

# Datos de clustering: train + valid (sin test para no contaminar)
X_clust_raw = pd.concat([X_train, X_valid], ignore_index=True)
y_clust     = pd.concat([y_train, y_valid], ignore_index=True)
X_clust_arr = X_clust_raw.values

RANDOM_STATE = 42
K_MAX        = 10   # rango de K para método del codo

# ── Transformaciones por métrica ──────────────────────────────

def transform_euclidean(X):
    """Sin transformación; K-Means euclidiano directo."""
    return X.copy()

def transform_manhattan(X):
    """
    Aproximación: se normalizan las filas por su norma L1.
    K-Means sobre este espacio tiende a minimizar distancias L1.
    """
    norms = np.abs(X).sum(axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

def transform_cosine(X):
    """
    Normalizar filas a norma L2 = 1.
    La distancia euclidiana en este espacio es monótona con la
    distancia coseno (d_cos = 1 - cos_sim = eucl²/2).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

def transform_mahalanobis(X):
    """
    Blanqueo (whitening): X_w = X @ inv(L)^T  donde Σ = L L^T.
    K-Means euclidiano sobre X_w ≡ K-Means Mahalanobis sobre X.
    """
    cov   = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6   # regularización
    L     = np.linalg.cholesky(cov)
    X_w   = np.linalg.solve(L, X.T).T
    return X_w

METRICAS = {
    'Euclidiana':  transform_euclidean,
    'Manhattan':   transform_manhattan,
    'Coseno':      transform_cosine,
    'Mahalanobis': transform_mahalanobis,
}

# ── Función auxiliar: inercia real con cada métrica ───────────

def inercia_real(X_orig, labels, centroides_orig, metrica):
    """Calcula inercia (suma de distancias² al centroide) con la métrica dada."""
    total = 0.0
    for k in np.unique(labels):
        mask   = labels == k
        pts    = X_orig[mask]
        centro = centroides_orig[k]
        if metrica == 'Euclidiana':
            dists = np.linalg.norm(pts - centro, axis=1)
        elif metrica == 'Manhattan':
            dists = np.abs(pts - centro).sum(axis=1)
        elif metrica == 'Coseno':
            # distancia coseno = 1 - similitud coseno
            num   = (pts * centro).sum(axis=1)
            denom = np.linalg.norm(pts, axis=1) * np.linalg.norm(centro)
            dists = 1 - num / (denom + 1e-10)
        elif metrica == 'Mahalanobis':
            cov   = np.cov(X_orig.T) + np.eye(X_orig.shape[1]) * 1e-6
            VI    = np.linalg.inv(cov)
            dists = np.array([distance.mahalanobis(p, centro, VI) for p in pts])
        total += (dists ** 2).sum()
    return total

# ── Método del codo + Silhouette por métrica ─────────────────
print("\n--- Método del codo y Silhouette (K=2…10) ---")

resultados_clustering = {}   # {metrica: {k: {'labels':..., 'sil':..., 'inercia':...}}}
codo_data             = {}   # {metrica: {k: inercia}}

for nombre_metrica, transform_fn in METRICAS.items():
    X_t = transform_fn(X_clust_arr)
    inercias   = []
    silhouettes = []
    modelos_k  = {}

    for k in range(2, K_MAX + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_t)
        labels = km.labels_

        # Inercia real en espacio original con la métrica correcta
        # Los centroides en espacio original: promedio de puntos por cluster
        centroides_orig = np.array([X_clust_arr[labels == c].mean(axis=0)
                                     for c in range(k)])
        iner = inercia_real(X_clust_arr, labels, centroides_orig, nombre_metrica)

        sil  = silhouette_score(X_t, labels, metric='euclidean') if k > 1 else 0.0

        inercias.append(iner)
        silhouettes.append(sil)
        modelos_k[k] = {'labels': labels, 'sil': sil, 'inercia': iner, 'modelo': km}

    codo_data[nombre_metrica]             = {'k': list(range(2, K_MAX + 1)),
                                              'inercia': inercias,
                                              'silhouette': silhouettes}
    resultados_clustering[nombre_metrica] = modelos_k

    # K óptimo por Silhouette
    k_opt_sil = list(range(2, K_MAX + 1))[np.argmax(silhouettes)]
    print(f"  [{nombre_metrica:12s}]  K óptimo (silhouette): {k_opt_sil}"
          f"  | Silhouette max: {max(silhouettes):.4f}")

# ── K=2 fijo (equivalente al caso base del notebook) ─────────
K_FIJO = 2
print(f"\n--- Detalle con K={K_FIJO} (caso base, análogo al notebook) ---")

for nombre_metrica in METRICAS:
    info = resultados_clustering[nombre_metrica][K_FIJO]
    labels = info['labels']
    sil    = info['sil']
    iner   = info['inercia']
    # Distribución de etiquetas reales por cluster
    dist   = {c: y_clust[labels == c].value_counts().to_dict()
              for c in range(K_FIJO)}
    print(f"\n  Métrica: {nombre_metrica}")
    print(f"    Inercia     : {iner:.2f}")
    print(f"    Silhouette  : {sil:.4f}")
    for c, d in dist.items():
        n_c  = (labels == c).sum()
        n1   = d.get(1, 0)
        pct1 = 100 * n1 / n_c if n_c else 0
        print(f"    Cluster {c}   : {n_c:3d} puntos  |  label=1: {n1} ({pct1:.1f}%)")

# ── K óptimo por silhouette para cada métrica ─────────────────
print("\n--- Detalle con K óptimo (silhouette) por métrica ---")
for nombre_metrica in METRICAS:
    sils   = codo_data[nombre_metrica]['silhouette']
    k_opt  = list(range(2, K_MAX + 1))[np.argmax(sils)]
    info   = resultados_clustering[nombre_metrica][k_opt]
    labels = info['labels']
    sil    = info['sil']
    iner   = info['inercia']
    dist   = {c: y_clust[labels == c].value_counts().to_dict()
              for c in range(k_opt)}
    print(f"\n  Métrica: {nombre_metrica}  —  K óptimo: {k_opt}")
    print(f"    Inercia     : {iner:.2f}")
    print(f"    Silhouette  : {sil:.4f}")
    for c, d in dist.items():
        n_c  = (labels == c).sum()
        n1   = d.get(1, 0)
        pct1 = 100 * n1 / n_c if n_c else 0
        print(f"    Cluster {c:2d}  : {n_c:3d} puntos  |  label=1: {n1} ({pct1:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 7. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────
print("\n--- Generando visualizaciones ---")

# ── 7A. Panel supervisado ─────────────────────────────────────
from sklearn.metrics import roc_curve

colors_split = {'train': '#3498DB', 'valid': '#F39C12', 'test': '#E74C3C'}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(f'FIRE-UdeA — Resultados Modelo Supervisado: {nombre_mejor}',
             fontsize=13, fontweight='bold')

# 7A-1 ROC-AUC por modelo en test
ax = axes[0, 0]
bar_colors = ['#2ECC71' if a > bl_test['roc_auc'] else '#E74C3C'
              for a in test_res['roc_auc'].values]
bars = ax.barh(test_res['modelo'].values, test_res['roc_auc'].values,
               color=bar_colors, edgecolor='white')
ax.axvline(bl_test['roc_auc'], color='black', linestyle='--', linewidth=1.5,
           label=f"Baseline ({bl_test['roc_auc']:.3f})")
ax.set_title('ROC-AUC en Test\n(verde = supera baseline)', fontsize=10, fontweight='bold')
ax.set_xlabel('ROC-AUC');  ax.legend(fontsize=8)
for bar, val in zip(bars, test_res['roc_auc'].values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=8)
ax.set_xlim(0, 1.1)

# 7A-2 Log-Loss
ax = axes[0, 1]
ll_vals    = test_res['log_loss'].values
bar_colors2 = ['#2ECC71' if v < bl_test['log_loss'] else '#E74C3C' for v in ll_vals]
bars2 = ax.barh(test_res['modelo'].values, ll_vals, color=bar_colors2, edgecolor='white')
ax.axvline(bl_test['log_loss'], color='black', linestyle='--', linewidth=1.5,
           label=f"Baseline ({bl_test['log_loss']:.3f})")
ax.set_title('Log-Loss en Test\n(verde = supera baseline)', fontsize=10, fontweight='bold')
ax.set_xlabel('Log-Loss (menor es mejor)');  ax.legend(fontsize=8)
for bar, val in zip(bars2, ll_vals):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=8)

# 7A-3 Curva ROC
ax = axes[0, 2]
for split_name, Xs, ys in [('train', X_train, y_train),
                             ('valid', X_valid, y_valid),
                             ('test',  X_test,  y_test)]:
    prob = best_model.predict_proba(Xs)[:, 1]
    fpr, tpr, _ = roc_curve(ys, prob)
    auc = roc_auc_score(ys, prob)
    ax.plot(fpr, tpr, label=f'{split_name} (AUC={auc:.3f})',
            color=colors_split[split_name], linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('FPR');  ax.set_ylabel('TPR')
ax.set_title(f'Curva ROC — {nombre_mejor}', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)

# 7A-4 Distribución de probabilidades
ax = axes[1, 0]
for label, color, etiqueta in zip([0, 1], ['#2ECC71', '#E74C3C'],
                                    ['Sin tensión', 'Tensión']):
    mask = y_test == label
    ax.hist(prob_test[mask], bins=8, alpha=0.65, color=color,
            edgecolor='none', label=etiqueta)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Umbral=0.5')
ax.set_title('Distribución de Probabilidades\nen Test', fontsize=10, fontweight='bold')
ax.set_xlabel('Probabilidad predicha');  ax.set_ylabel('Frecuencia')
ax.legend(fontsize=9)

# 7A-5 Matriz de confusión
ax = axes[1, 1]
cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['Real 0', 'Real 1'],
            linewidths=1, linecolor='white')
ax.set_title(f'Matriz de Confusión — Test\n{nombre_mejor}', fontsize=10, fontweight='bold')

# 7A-6 Comparación nuevo vs baseline
ax = axes[1, 2]
metricas_labels = ['roc_auc', 'pr_auc', 'f1']
nuevo_vals = [mejor[m] for m in metricas_labels]
base_vals  = [bl_test[m] for m in metricas_labels]
x = np.arange(len(metricas_labels))
width = 0.35
b1 = ax.bar(x - width / 2, base_vals,  width, label='Baseline',      color='#95A5A6', edgecolor='white')
b2 = ax.bar(x + width / 2, nuevo_vals, width, label=nombre_mejor,    color='#2ECC71', edgecolor='white')
ax.set_xticks(x);  ax.set_xticklabels(['ROC-AUC', 'PR-AUC', 'F1'], fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_title('Comparación Nuevo vs Baseline\n(Test)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=8)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fire_udea_resultados.png'), dpi=130,
            bbox_inches='tight', facecolor='white')
print("[OK] fire_udea_resultados.png")
plt.close()

# ── 7B. Método del codo — 4 métricas ─────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle('K-Means — Método del Codo y Silhouette por Métrica de Distancia',
             fontsize=13, fontweight='bold')

colores = {'Euclidiana': '#3498DB', 'Manhattan': '#E67E22',
           'Coseno': '#9B59B6', 'Mahalanobis': '#27AE60'}

for col_idx, nombre_metrica in enumerate(METRICAS):
    ks      = codo_data[nombre_metrica]['k']
    iner    = codo_data[nombre_metrica]['inercia']
    sils    = codo_data[nombre_metrica]['silhouette']
    color   = colores[nombre_metrica]
    k_opt   = ks[np.argmax(sils)]

    # Codo
    ax = axes[0, col_idx]
    ax.plot(ks, iner, 'o-', color=color, linewidth=2, markersize=6)
    ax.set_title(f'Codo — {nombre_metrica}', fontsize=10, fontweight='bold')
    ax.set_xlabel('K');  ax.set_ylabel('Inercia')
    ax.grid(alpha=0.3)

    # Silhouette
    ax = axes[1, col_idx]
    ax.plot(ks, sils, 's-', color=color, linewidth=2, markersize=6)
    ax.axvline(k_opt, color='red', linestyle='--', linewidth=1.2,
               label=f'K óptimo={k_opt}')
    ax.set_title(f'Silhouette — {nombre_metrica}', fontsize=10, fontweight='bold')
    ax.set_xlabel('K');  ax.set_ylabel('Silhouette Score')
    ax.legend(fontsize=8);  ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'kmeans_codo_silhouette.png'), dpi=130,
            bbox_inches='tight', facecolor='white')
print("[OK] kmeans_codo_silhouette.png")
plt.close()

# ── 7C. Scatter clusters (PCA 2D) — K=2 y K óptimo ──────────
pca      = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca    = pca.fit_transform(X_clust_arr)
varianza = pca.explained_variance_ratio_

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle('K-Means — Clusters proyectados en PCA 2D\n'
             f'(PC1={varianza[0]*100:.1f}%, PC2={varianza[1]*100:.1f}%)',
             fontsize=13, fontweight='bold')

cmaps = {'Euclidiana': 'tab10', 'Manhattan': 'Set1',
         'Coseno': 'Set2', 'Mahalanobis': 'Dark2'}

for col_idx, nombre_metrica in enumerate(METRICAS):
    sils  = codo_data[nombre_metrica]['silhouette']
    k_opt = list(range(2, K_MAX + 1))[np.argmax(sils)]

    for row_idx, k_plot in enumerate([K_FIJO, k_opt]):
        ax     = axes[row_idx, col_idx]
        labels = resultados_clustering[nombre_metrica][k_plot]['labels']
        titulo = (f'{nombre_metrica}\nK={k_plot}'
                  + (f' (óptimo)' if k_plot == k_opt and k_plot != K_FIJO else ''))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                        cmap=cmaps[nombre_metrica], s=18, alpha=0.7, edgecolors='none')
        # Centroides en espacio PCA
        for k in range(k_plot):
            mask   = labels == k
            cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
            ax.scatter(cx, cy, marker='X', s=150, color='black', zorder=5)
        ax.set_title(titulo, fontsize=9, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=8);  ax.set_ylabel('PC2', fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'kmeans_clusters_pca.png'), dpi=130,
            bbox_inches='tight', facecolor='white')
print("[OK] kmeans_clusters_pca.png")
plt.close()

# ── 7D. Comparación de Silhouette por métrica (barras) ───────
fig, ax = plt.subplots(figsize=(10, 5))
k_range = list(range(2, K_MAX + 1))
bar_width = 0.2
offsets   = np.linspace(-(len(METRICAS) - 1) * bar_width / 2,
                         (len(METRICAS) - 1) * bar_width / 2,
                         len(METRICAS))
x_pos = np.arange(len(k_range))

for offset, (nombre_metrica, color) in zip(offsets,
                                            [(m, colores[m]) for m in METRICAS]):
    sils = codo_data[nombre_metrica]['silhouette']
    ax.bar(x_pos + offset, sils, width=bar_width, label=nombre_metrica,
           color=color, edgecolor='white', alpha=0.85)

ax.set_xticks(x_pos);  ax.set_xticklabels([f'K={k}' for k in k_range])
ax.set_title('Silhouette Score por K y por Métrica de Distancia',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Silhouette Score')
ax.legend(fontsize=10);  ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'kmeans_silhouette_comparacion.png'), dpi=130,
            bbox_inches='tight', facecolor='white')
print("[OK] kmeans_silhouette_comparacion.png")
plt.close()

# ── 7E. Feature importance del mejor modelo supervisado ──────
try:
    base_est = best_model.calibrated_classifiers_[0].estimator
    if hasattr(base_est, 'feature_importances_'):
        importances = base_est.feature_importances_
        fi = pd.Series(importances, index=FEATURES_DF2).sort_values(ascending=True)
        fig_fi, ax_fi = plt.subplots(figsize=(9, 6))
        fi.plot(kind='barh', ax=ax_fi, color='#3498DB', edgecolor='white')
        ax_fi.set_title(f'Feature Importance — {nombre_mejor}',
                        fontsize=12, fontweight='bold')
        ax_fi.set_xlabel('Importancia')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, 'fire_udea_feature_importance.png'), dpi=130,
                    bbox_inches='tight', facecolor='white')
        print("[OK] fire_udea_feature_importance.png")
        plt.close()
        print("\nTop 10 features más importantes:")
        print(fi.sort_values(ascending=False).head(10).round(4).to_string())
except Exception as e:
    print(f"[INFO] Feature importance no disponible: {e}")

# ─────────────────────────────────────────────────────────────
# 8. TABLA RESUMEN DE CLUSTERING
# ─────────────────────────────────────────────────────────────
print("\n--- Tabla resumen de clustering ---")
resumen_rows = []
for nombre_metrica in METRICAS:
    sils  = codo_data[nombre_metrica]['silhouette']
    iner  = codo_data[nombre_metrica]['inercia']
    for i, k in enumerate(range(2, K_MAX + 1)):
        resumen_rows.append({
            'metrica':    nombre_metrica,
            'k':          k,
            'inercia':    round(iner[i], 4),
            'silhouette': round(sils[i], 4),
        })
df_clustering = pd.DataFrame(resumen_rows)
print(df_clustering.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 9. EXPORTAR RESULTADOS
# ─────────────────────────────────────────────────────────────
df_res.to_csv(os.path.join(OUT, 'reporte_metricas_FIRE_nuevo.csv'), index=False)
scores_nuevo.to_csv(os.path.join(OUT, 'scores_test_FIRE_nuevo.csv'), index=False)
df_clustering.to_csv(os.path.join(OUT, 'reporte_clustering_kmeans.csv'), index=False)
print("\n[OK] Exportados:")
print("     out/reporte_metricas_FIRE_nuevo.csv")
print("     out/scores_test_FIRE_nuevo.csv")
print("     out/reporte_clustering_kmeans.csv")

print("\n" + "=" * 60)
print("  Pipeline completado. Revisa la carpeta out/")
print("=" * 60)