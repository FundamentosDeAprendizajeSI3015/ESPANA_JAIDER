"""
EDA y Preprocesamiento – 10,000 Empresas más Grandes del País
Objetivo: Preparar los datos para clustering por apalancamiento financiero
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

# Crear carpeta de salida si no existe
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

# Limpiar columnas monetarias (quitar $, comas → float)
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

# Limpiar año y NIT
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
    kind="bar", ax=axes[0], color="steelblue", edgecolor="white"
)
axes[0].set_title("Empresas por Año de Corte")
axes[0].set_xlabel("Año")
axes[0].set_ylabel("Cantidad")
axes[0].tick_params(axis="x", rotation=0)

df["MACROSECTOR"].value_counts().plot(
    kind="barh", ax=axes[1], color="coral", edgecolor="white"
)
axes[1].set_title("Empresas por Macrosector")
axes[1].set_xlabel("Cantidad")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_01_distribucion_empresas.png"), dpi=150)
plt.close()
print("\n  [Gráfica guardada] out/eda_01_distribucion_empresas.png")

# ── 3.2 Distribución de variables financieras (log scale) ─────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(monetary_cols):
    data = df[col][df[col] > 0]          # solo positivos para log
    axes[i].hist(np.log1p(data), bins=60, color="teal", edgecolor="white", alpha=0.8)
    axes[i].set_title(f"log(1 + {col})")
    axes[i].set_xlabel("log(1 + valor)")
    axes[i].set_ylabel("Frecuencia")

axes[5].axis("off")
plt.suptitle("Distribución de Variables Financieras (escala logarítmica)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_02_distribuciones_financieras.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/eda_02_distribuciones_financieras.png")

# ── 3.3 Boxplots por macrosector – Pasivos/Activos ────────────
fig, ax = plt.subplots(figsize=(14, 6))
sector_order = (
    df.groupby("MACROSECTOR")["TOTAL PASIVOS"].median().sort_values(ascending=False).index
)
df_pos = df[df["TOTAL ACTIVOS"] > 0].copy()
df_pos["Pasivos/Activos"] = df_pos["TOTAL PASIVOS"] / df_pos["TOTAL ACTIVOS"]
df_pos_clip = df_pos[df_pos["Pasivos/Activos"].between(0, 5)]

sns.boxplot(
    data=df_pos_clip,
    x="MACROSECTOR", y="Pasivos/Activos",
    order=sector_order, ax=ax,
    palette="Set2", linewidth=0.8
)
ax.set_title("Ratio Pasivos/Activos por Macrosector")
ax.set_xlabel("")
ax.set_ylabel("Pasivos / Activos")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_03_boxplot_leverage_macrosector.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/eda_03_boxplot_leverage_macrosector.png")

# ── 3.4 Heatmap de correlación ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[monetary_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, ax=ax,
    linewidths=0.5, cbar_kws={"shrink": 0.8}
)
ax.set_title("Correlación entre Variables Financieras")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_04_correlacion.png"), dpi=150)
plt.close()
print("  [Gráfica guardada] out/eda_04_correlacion.png")

# ── 3.5 Evolución temporal de activos y pasivos promedio ───────
evol = (
    df.groupby("Año de Corte")[["TOTAL ACTIVOS", "TOTAL PASIVOS", "TOTAL PATRIMONIO"]]
    .median()
    .reset_index()
)
fig, ax = plt.subplots(figsize=(10, 5))
for col, color in zip(["TOTAL ACTIVOS", "TOTAL PASIVOS", "TOTAL PATRIMONIO"],
                      ["steelblue", "coral", "seagreen"]):
    ax.plot(evol["Año de Corte"], evol[col], marker="o", label=col, color=color)
ax.set_title("Mediana de Activos, Pasivos y Patrimonio por Año")
ax.set_xlabel("Año")
ax.set_ylabel("Billones COP (mediana)")
ax.legend()
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
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

eps = 1e-9  # evitar divisiones por cero

# Ratios clásicos de leverage
df_feat["deuda_activos"]       = df_feat["TOTAL PASIVOS"]  / (df_feat["TOTAL ACTIVOS"]   + eps)
df_feat["deuda_patrimonio"]    = df_feat["TOTAL PASIVOS"]  / (df_feat["TOTAL PATRIMONIO"].abs() + eps)
df_feat["multiplicador_cap"]   = df_feat["TOTAL ACTIVOS"]  / (df_feat["TOTAL PATRIMONIO"].abs() + eps)
df_feat["cobertura_ingresos"]  = df_feat["INGRESOS OPERACIONALES"] / (df_feat["TOTAL PASIVOS"] + eps)
df_feat["margen_neto"]         = df_feat["GANANCIA (PÉRDIDA)"] / (df_feat["INGRESOS OPERACIONALES"].abs() + eps)
df_feat["roa"]                 = df_feat["GANANCIA (PÉRDIDA)"] / (df_feat["TOTAL ACTIVOS"] + eps)
df_feat["roe"]                 = df_feat["GANANCIA (PÉRDIDA)"] / (df_feat["TOTAL PATRIMONIO"].abs() + eps)

leverage_features = [
    "deuda_activos", "deuda_patrimonio", "multiplicador_cap",
    "cobertura_ingresos", "margen_neto", "roa", "roe"
]

print(f"\n  Características creadas: {leverage_features}")
print(f"\n  Estadísticas descriptivas:")
print(df_feat[leverage_features].describe().T[["mean","std","min","50%","max"]].to_string())

# ─────────────────────────────────────────────────────────────
# 5. PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PREPROCESAMIENTO")
print("=" * 60)

# ── 5.1 Filtrar registros con activos positivos y datos válidos
n_before = len(df_feat)
df_clean = df_feat[df_feat["TOTAL ACTIVOS"] > 0].copy()
print(f"\n  Filas con TOTAL ACTIVOS > 0: {len(df_clean):,} (removidas: {n_before - len(df_clean):,})")

# ── 5.2 Eliminar outliers extremos con IQR x5 ─────────────────
def remove_outliers_iqr(df, cols, factor=5.0):
    """Elimina filas fuera de [Q1 - factor*IQR, Q3 + factor*IQR]."""
    mask = pd.Series(True, index=df.index)
    for col in cols:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask &= df[col].between(lower, upper)
    return df[mask]

df_clean = remove_outliers_iqr(df_clean, leverage_features, factor=5.0)
print(f"  Filas tras eliminar outliers extremos (IQR×5): {len(df_clean):,}")

# ── 5.3 Rellenar NaN/Inf residuales con mediana ────────────────
for col in leverage_features:
    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

print(f"  NaN/Inf residuales tras imputación: {df_clean[leverage_features].isnull().sum().sum()}")

# ── 5.4 Transformación log-modulus para reducir sesgo ──────────
def log_modulus(x):
    """Signo(x) * log(1 + |x|) — maneja valores negativos."""
    return np.sign(x) * np.log1p(np.abs(x))

df_transformed = df_clean.copy()
for col in leverage_features:
    df_transformed[col + "_lm"] = log_modulus(df_transformed[col])

transformed_cols = [c + "_lm" for c in leverage_features]

# ── 5.5 Estandarización con RobustScaler ──────────────────────
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_transformed[transformed_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=transformed_cols, index=df_transformed.index)

print(f"\n  Shape final para clustering: {X_scaled_df.shape}")

# ── 5.6 Gráfica distribuciones post-procesamiento ─────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(transformed_cols):
    axes[i].hist(X_scaled_df[col], bins=60, color="mediumpurple", edgecolor="white", alpha=0.85)
    axes[i].set_title(col.replace("_lm", "").replace("_", " ").title())
    axes[i].set_xlabel("Valor escalado")
    axes[i].set_ylabel("Frecuencia")

axes[-1].axis("off")
plt.suptitle("Distribuciones Post-Procesamiento (log-modulus + RobustScaler)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prepro_01_distribuciones_finales.png"), dpi=150)
plt.close()
print("\n  [Gráfica guardada] out/prepro_01_distribuciones_finales.png")

# ── 5.7 Pairplot de características principales ────────────────
sample = X_scaled_df[transformed_cols[:4]].sample(min(2000, len(X_scaled_df)), random_state=42)
g = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10})
g.figure.suptitle("Pairplot – Features de Apalancamiento (muestra 2,000)", y=1.02)
g.figure.savefig(os.path.join(OUTPUT_DIR, "prepro_02_pairplot_features.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  [Gráfica guardada] out/prepro_02_pairplot_features.png")

# ── 5.8 Correlación post-procesamiento ────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr_post = X_scaled_df.corr()
mask2 = np.triu(np.ones_like(corr_post, dtype=bool))
sns.heatmap(
    corr_post, mask=mask2, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, ax=ax,
    linewidths=0.4, cbar_kws={"shrink": 0.8}
)
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
print("\n EDA y preprocesamiento completados.")
print("    Features escaladas en 'transformed_cols' listas para clustering.")