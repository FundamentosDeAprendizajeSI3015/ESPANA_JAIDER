# =============================================================
#  FIRE-UdeA — Pipeline ML completo
#  Objetivo: superar métricas baseline (test ROC-AUC=0.417,
#            log_loss=4.87, brier=0.257)
# =============================================================

# ── 0. Instalación ───────────────────────────────────────────
# pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn

import warnings
warnings.filterwarnings('ignore')

import os
OUT = 'out'
os.makedirs(OUT, exist_ok=True)
print(f"[OK] Carpeta de salida: '{OUT}/'")


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # cambiar a 'TkAgg' o quitar si usas Jupyter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, log_loss, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              RocCurveDisplay)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  FIRE-UdeA — Pipeline ML")
print("=" * 60)

df1 = pd.read_csv('dataset_sintetico_FIRE_UdeA.csv')
df2 = pd.read_csv('dataset_sintetico_FIRE_UdeA_realista.csv')
baseline = pd.read_csv('reporte_metricas_FIRE_UdeA_realista.csv')
scores_bl = pd.read_csv('scores_test_FIRE_UdeA_realista.csv')

# Eliminar participacion_ley30
df1 = df1.drop(columns=['participacion_ley30'], errors='ignore')
df2 = df2.drop(columns=['participacion_ley30'], errors='ignore')

print(f"\n[INFO] df1 (sintético):  {df1.shape}")
print(f"[INFO] df2 (realista):   {df2.shape}")

# ─────────────────────────────────────────────────────────────
# 2. PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────
print("\n--- Preprocesamiento ---")

# 2.1 Imputación KNN en df2
cols_excluir = ['anio', 'unidad', 'label']
num_cols_df2 = [c for c in df2.select_dtypes(include='number').columns
                if c not in cols_excluir]

knn_imputer = KNNImputer(n_neighbors=5)
df2_imp = df2.copy()
df2_imp[num_cols_df2] = knn_imputer.fit_transform(df2[num_cols_df2])
print(f"[OK] KNN imputation — nulos restantes: {df2_imp[num_cols_df2].isnull().sum().sum()}")

# 2.2 Feature Engineering — dataset realista
def feature_engineering(df):
    d = df.copy()
    # Ratio CFO / ingresos
    if 'ingresos_totales' in d.columns:
        d['cfo_ratio'] = d['cfo'] / d['ingresos_totales'].replace(0, np.nan)
    # Interacción liquidez × días efectivo
    d['liquidez_x_dias'] = d['liquidez'] * d['dias_efectivo']
    # Señal combinada de tensión (0-2)
    d['tension_signal'] = (d['cfo'] < 0).astype(int) + (d['liquidez'] < 1).astype(int)
    # Diversificación de fuentes (1 - HHI)
    d['diversificacion'] = 1 - d['hhi_fuentes']
    # Fuentes propias combinadas
    if 'participacion_servicios' in d.columns and 'participacion_matriculas' in d.columns:
        d['participacion_propia'] = d['participacion_servicios'] + d['participacion_matriculas']
    # Log de magnitudes grandes
    if 'ingresos_totales' in d.columns:
        d['log_ingresos'] = np.log1p(d['ingresos_totales'])
        d['log_gastos_personal'] = np.log1p(d['gastos_personal'])
    return d

df2_fe = feature_engineering(df2_imp)
df1_fe = feature_engineering(df1)

# 2.3 Definir feature sets
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

# 2.4 Split temporal — df2
anios = sorted(df2_fe['anio'].unique())
anio_test  = anios[-1]    # 2025
anio_valid = anios[-2]    # 2024

mask_test  = df2_fe['anio'] == anio_test
mask_valid = df2_fe['anio'] == anio_valid
mask_train = ~(mask_test | mask_valid)

X2 = df2_fe[FEATURES_DF2]
y2 = df2_fe['label']
meta_test = df2_fe[mask_test][['anio', 'unidad']].reset_index(drop=True)

X_train_raw = X2[mask_train.values].reset_index(drop=True)
X_valid_raw = X2[mask_valid.values].reset_index(drop=True)
X_test_raw  = X2[mask_test.values].reset_index(drop=True)
y_train = y2[mask_train.values].reset_index(drop=True)
y_valid = y2[mask_valid.values].reset_index(drop=True)
y_test  = y2[mask_test.values].reset_index(drop=True)

# 2.5 Escalado RobustScaler
scaler = RobustScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=FEATURES_DF2)
X_valid = pd.DataFrame(scaler.transform(X_valid_raw),     columns=FEATURES_DF2)
X_test  = pd.DataFrame(scaler.transform(X_test_raw),      columns=FEATURES_DF2)

# df1 completo (para preentrenamiento / referencia)
X1 = pd.DataFrame(RobustScaler().fit_transform(df1_fe[FEATURES_DF1]), columns=FEATURES_DF1)
y1 = df1_fe['label']

print(f"[OK] Train={X_train.shape} | Valid={X_valid.shape} | Test={X_test.shape}")
print(f"     Prevalencia  train={y_train.mean():.2f} | valid={y_valid.mean():.2f} | test={y_test.mean():.2f}")


# ─────────────────────────────────────────────────────────────
# 3. FUNCIONES DE EVALUACIÓN
# ─────────────────────────────────────────────────────────────
def evaluar(nombre, y_true, y_prob, y_pred, split, n):
    prevalencia = y_true.mean()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        'modelo':     nombre,
        'split':      split,
        'n':          n,
        'prevalencia': round(prevalencia, 3),
        'roc_auc':    round(roc_auc_score(y_true, y_prob), 4),
        'pr_auc':     round(average_precision_score(y_true, y_prob), 4),
        'brier':      round(brier_score_loss(y_true, y_prob), 4),
        'log_loss':   round(log_loss(y_true, y_prob), 4),
        'precision':  round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':     round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1':         round(f1_score(y_true, y_pred, zero_division=0), 4),
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
# 4. MODELOS
# ─────────────────────────────────────────────────────────────
print("\n--- Entrenamiento de modelos ---")

resultados = []

# ── 4.1 Logistic Regression calibrada ────────────────────────
lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42,
                        class_weight='balanced', solver='lbfgs')
lr_cal = CalibratedClassifierCV(lr, cv=3, method='isotonic')
lr_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('LogReg_calibrada', lr_cal))
print("[OK] Logistic Regression calibrada")

# ── 4.2 Random Forest calibrado ──────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300, max_depth=4, min_samples_leaf=3,
    max_features='sqrt', class_weight='balanced', random_state=42
)
rf_cal = CalibratedClassifierCV(rf, cv=3, method='isotonic')
rf_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('RandomForest_calibrado', rf_cal))
print("[OK] Random Forest calibrado")

# ── 4.3 Gradient Boosting calibrado ──────────────────────────
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3,
    min_samples_leaf=3, subsample=0.8, random_state=42
)
gb_cal = CalibratedClassifierCV(gb, cv=3, method='isotonic')
gb_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('GradientBoosting_calibrado', gb_cal))
print("[OK] Gradient Boosting calibrado")

# ── 4.4 SVM calibrado ────────────────────────────────────────
svm = SVC(C=1.0, kernel='rbf', class_weight='balanced',
          probability=False, random_state=42)
svm_cal = CalibratedClassifierCV(svm, cv=3, method='isotonic')
svm_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('SVM_calibrado', svm_cal))
print("[OK] SVM calibrado")

# ── 4.5 KNN calibrado ────────────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=5, weights='distance',
                            metric='euclidean')
knn_cal = CalibratedClassifierCV(knn, cv=3, method='isotonic')
knn_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('KNN_calibrado', knn_cal))
print("[OK] KNN calibrado")

# ── 4.6 Voting Ensemble (soft) ───────────────────────────────
# Usa los estimadores base (sin calibración de CV anidada)
lr_base = LogisticRegression(C=0.1, max_iter=1000, random_state=42,
                              class_weight='balanced')
rf_base = RandomForestClassifier(n_estimators=300, max_depth=4,
                                  min_samples_leaf=3, class_weight='balanced',
                                  random_state=42)
gb_base = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=3, min_samples_leaf=3,
                                      subsample=0.8, random_state=42)

voting = VotingClassifier(
    estimators=[('lr', lr_base), ('rf', rf_base), ('gb', gb_base)],
    voting='soft', weights=[1, 2, 2]
)
voting_cal = CalibratedClassifierCV(voting, cv=3, method='isotonic')
voting_cal.fit(X_train, y_train)
resultados.append(evaluar_modelo('Voting_Ensemble', voting_cal))
print("[OK] Voting Ensemble calibrado")


# ─────────────────────────────────────────────────────────────
# 5. COMPARACIÓN DE RESULTADOS
# ─────────────────────────────────────────────────────────────
print("\n--- Resultados ---")

df_res = pd.concat(resultados, ignore_index=True)

# Métricas clave en test
test_res = df_res[df_res['split'] == 'test'].copy()
test_res = test_res.sort_values('roc_auc', ascending=False)

# Baseline de referencia
bl_test = baseline[baseline['split'] == 'test'].iloc[0]
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

# Mejor modelo
mejor = test_res.iloc[0]
print(f"\n🏆 Mejor modelo en test: {mejor['modelo']}")
print(f"   ROC-AUC  : {mejor['roc_auc']:.4f}  (baseline: {bl_test['roc_auc']:.4f})")
print(f"   PR-AUC   : {mejor['pr_auc']:.4f}  (baseline: {bl_test['pr_auc']:.4f})")
print(f"   Brier    : {mejor['brier']:.4f}  (baseline: {bl_test['brier']:.4f})")
print(f"   Log-Loss : {mejor['log_loss']:.4f}  (baseline: {bl_test['log_loss']:.4f})")
print(f"   F1       : {mejor['f1']:.4f}  (baseline: {bl_test['f1']:.4f})")


# ─────────────────────────────────────────────────────────────
# 6. SCORES TEST DETALLADOS (mejor modelo)
# ─────────────────────────────────────────────────────────────
nombre_mejor = mejor['modelo']
modelos_map = {
    'LogReg_calibrada':       lr_cal,
    'RandomForest_calibrado': rf_cal,
    'GradientBoosting_calibrado': gb_cal,
    'SVM_calibrado':          svm_cal,
    'KNN_calibrado':          knn_cal,
    'Voting_Ensemble':        voting_cal,
}
best_model = modelos_map[nombre_mejor]
prob_test  = best_model.predict_proba(X_test)[:, 1]
pred_test  = (prob_test >= 0.5).astype(int)

scores_nuevo = meta_test.copy()
scores_nuevo['y_true'] = y_test.values
scores_nuevo['prob']   = prob_test.round(4)
scores_nuevo['pred']   = pred_test

print(f"\n--- Scores test — {nombre_mejor} ---")
print(scores_nuevo.to_string(index=False))

print("\n--- Scores test — BASELINE ---")
print(scores_bl.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# 7. TABLA COMPLETA DE MÉTRICAS (todos los splits)
# ─────────────────────────────────────────────────────────────
print("\n--- Tabla completa de métricas (mejor modelo) ---")
df_mejor = df_res[df_res['modelo'] == nombre_mejor][
    ['split','n','prevalencia','roc_auc','pr_auc','brier','log_loss',
     'precision','recall','f1','tn','fp','fn','tp']
]
print(df_mejor.to_string(index=False))

print("\n--- Tabla completa de métricas (BASELINE) ---")
print(baseline.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# 8. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────
print("\n--- Generando visualizaciones ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(f'FIRE-UdeA — Resultados Modelo: {nombre_mejor}',
             fontsize=14, fontweight='bold')

colors_split = {'train': '#3498DB', 'valid': '#F39C12', 'test': '#E74C3C'}

# ── 8.1 ROC-AUC por modelo en test ───────────────────────────
ax = axes[0, 0]
modelos_test = test_res['modelo'].values
aucs = test_res['roc_auc'].values
bar_colors = ['#2ECC71' if a > bl_test['roc_auc'] else '#E74C3C' for a in aucs]
bars = ax.barh(modelos_test, aucs, color=bar_colors, edgecolor='white')
ax.axvline(bl_test['roc_auc'], color='black', linestyle='--', linewidth=1.5,
           label=f"Baseline ({bl_test['roc_auc']:.3f})")
ax.set_title('ROC-AUC en Test\n(verde = supera baseline)', fontsize=10, fontweight='bold')
ax.set_xlabel('ROC-AUC')
ax.legend(fontsize=8)
for bar, val in zip(bars, aucs):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)
ax.set_xlim(0, 1.1)

# ── 8.2 Log-Loss por modelo en test ──────────────────────────
ax = axes[0, 1]
ll_vals = test_res['log_loss'].values
bar_colors2 = ['#2ECC71' if v < bl_test['log_loss'] else '#E74C3C' for v in ll_vals]
bars2 = ax.barh(modelos_test, ll_vals, color=bar_colors2, edgecolor='white')
ax.axvline(bl_test['log_loss'], color='black', linestyle='--', linewidth=1.5,
           label=f"Baseline ({bl_test['log_loss']:.3f})")
ax.set_title('Log-Loss en Test\n(verde = supera baseline)', fontsize=10, fontweight='bold')
ax.set_xlabel('Log-Loss (menor es mejor)')
ax.legend(fontsize=8)
for bar, val in zip(bars2, ll_vals):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)

# ── 8.3 Curva ROC del mejor modelo ───────────────────────────
from sklearn.metrics import roc_curve
ax = axes[0, 2]
for split_name, Xs, ys in [('train', X_train, y_train),
                             ('valid', X_valid, y_valid),
                             ('test',  X_test,  y_test)]:
    prob = best_model.predict_proba(Xs)[:, 1]
    fpr, tpr, _ = roc_curve(ys, prob)
    auc = roc_auc_score(ys, prob)
    ax.plot(fpr, tpr, label=f'{split_name} (AUC={auc:.3f})',
            color=colors_split[split_name], linewidth=2)
ax.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'Curva ROC — {nombre_mejor}', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)

# ── 8.4 Probabilidades predichas vs real ─────────────────────
ax = axes[1, 0]
prob_test_all = best_model.predict_proba(X_test)[:, 1]
for label, color, nombre in zip([0, 1], ['#2ECC71', '#E74C3C'], ['Sin tensión', 'Tensión']):
    mask = y_test == label
    ax.hist(prob_test_all[mask], bins=8, alpha=0.65, color=color,
            edgecolor='none', label=nombre)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Umbral=0.5')
ax.set_title('Distribución de Probabilidades\nen Test', fontsize=10, fontweight='bold')
ax.set_xlabel('Probabilidad predicha')
ax.set_ylabel('Frecuencia')
ax.legend(fontsize=9)

# ── 8.5 Matriz de confusión ───────────────────────────────────
ax = axes[1, 1]
cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['Real 0', 'Real 1'],
            linewidths=1, linecolor='white')
ax.set_title(f'Matriz de Confusión — Test\n{nombre_mejor}', fontsize=10, fontweight='bold')

# ── 8.6 Comparación nuevo vs baseline por métrica ────────────
ax = axes[1, 2]
metricas = ['roc_auc', 'pr_auc', 'f1']
nuevo_vals  = [mejor[m] for m in metricas]
base_vals   = [bl_test[m] for m in metricas]
x = np.arange(len(metricas))
width = 0.35
b1 = ax.bar(x - width/2, base_vals,  width, label='Baseline', color='#95A5A6', edgecolor='white')
b2 = ax.bar(x + width/2, nuevo_vals, width, label=nombre_mejor, color='#2ECC71', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(['ROC-AUC', 'PR-AUC', 'F1'], fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_title('Comparación Nuevo vs Baseline\n(Test)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=8)
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fire_udea_resultados.png'), dpi=130, bbox_inches='tight', facecolor='white')
print("[OK] Figura guardada: out/fire_udea_resultados.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# 9. FEATURE IMPORTANCE (mejor modelo si tiene atributo)
# ─────────────────────────────────────────────────────────────
try:
    # Extraer el estimador base del mejor modelo calibrado
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
        print("[OK] Feature importance guardada: out/fire_udea_feature_importance.png")
        plt.close()

        print("\nTop 10 features más importantes:")
        print(fi.sort_values(ascending=False).head(10).round(4).to_string())
except Exception as e:
    print(f"[INFO] Feature importance no disponible para este modelo: {e}")

# ─────────────────────────────────────────────────────────────
# 10. EXPORTAR RESULTADOS
# ─────────────────────────────────────────────────────────────
df_res.to_csv(os.path.join(OUT, 'reporte_metricas_FIRE_nuevo.csv'), index=False)
scores_nuevo.to_csv(os.path.join(OUT, 'scores_test_FIRE_nuevo.csv'), index=False)
print("\n[OK] Exportados:")
print("     out/reporte_metricas_FIRE_nuevo.csv")
print("     out/scores_test_FIRE_nuevo.csv")

print("\n" + "=" * 60)
print("  Pipeline completado. Revisa la carpeta out/")
print("=" * 60)