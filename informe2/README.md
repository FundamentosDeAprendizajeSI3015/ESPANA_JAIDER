# Pipeline ML – 10,000 Empresas más Grandes del País

Análisis de apalancamiento financiero mediante clustering no supervisado,
corrección de etiquetas y modelos supervisados de clasificación.

---

## Descripción general

El pipeline toma el dataset público de las 10,000 empresas más grandes
de Colombia y construye un flujo completo de Machine Learning que cubre:

| Etapa | Descripción |
|-------|-------------|
| Preprocesamiento | Limpieza monetaria, ratios de apalancamiento, log-modulus, RobustScaler |
| Clustering | K-Means, Fuzzy C-Means, Subtractive, DBSCAN, GMM (full/tied/diag/spherical) |
| Re-evaluación de etiquetas | Corrección k-NN de puntos mal asignados (~30 %) |
| Supervisado | Árbol de Decisión, Regresión Logística, Regresión Lineal OvR |
| Comparación | Métricas CV-5 con etiquetas corregidas vs. etiquetas originales |

---

## Estructura del proyecto

```
.
├── ml_pipeline.py               # Script principal (punto de entrada)
├── requirements.txt             # Dependencias Python
├── README.md                    # Este archivo
└── out/                         # Carpeta de salida (se crea automáticamente)
    ├── cluster_01_kmeans_elbow.png
    ├── cluster_02_dbscan_knn.png
    ├── cluster_03_pca_todos.png
    ├── cluster_metricas.csv
    ├── relabel_01_antes_despues.png
    ├── sup_01_feature_importance.png
    ├── sup_classification_reports.txt
    ├── sup_decision_tree_rules.txt
    ├── comparacion_01_corregidas_vs_originales.png
    ├── comparacion_02_confusion_matrix.png
    └── dataset_final_con_clusters.csv
```

---

## Requisitos

- Python **3.10 – 3.12**
- Las dependencias se listan en `requirements.txt`

---

## Instalación

```bash
# 1. Clonar o descomprimir el proyecto
cd ml_pipeline

# 2. Crear entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** `scikit-fuzzy` es opcional. Si no está instalado, el paso de
> Fuzzy C-Means se omite automáticamente y el pipeline continúa con los
> demás algoritmos.

---

## Uso

```bash
# Ajustar FILE_PATH dentro de ml_pipeline.py y ejecutar:
python ml_pipeline.py
```

El script imprime el progreso en consola y guarda todos los artefactos
en la carpeta `out/`.

---

## Variables de apalancamiento calculadas

| Variable | Fórmula |
|----------|---------|
| `deuda_activos` | Pasivos / Activos |
| `deuda_patrimonio` | Pasivos / \|Patrimonio\| |
| `multiplicador_cap` | Activos / \|Patrimonio\| |
| `cobertura_ingresos` | Ingresos Operacionales / Pasivos |
| `margen_neto` | Ganancia / \|Ingresos Operacionales\| |
| `roa` | Ganancia / Activos |
| `roe` | Ganancia / \|Patrimonio\| |

Todas se transforman con **log-modulus** `sign(x) · log(1 + |x|)` y se
estandarizan con **RobustScaler** antes del modelado.

---

## Algoritmos de clustering

### K-Means
Búsqueda del `k` óptimo (2–9) por **silhouette score**. Se usa `k` seleccionado
automáticamente para el resto del pipeline.

### Fuzzy C-Means
Implementado con `scikit-fuzzy`. Cada empresa recibe un grado de pertenencia
a cada cluster en lugar de una asignación dura.

### Subtractive Clustering
Implementación propia del algoritmo de Chiu (1994). Estima centros de forma
automática usando potencial de densidad radial. Se ejecuta sobre una muestra
de hasta 5,000 registros por eficiencia y luego se extiende al dataset completo
asignando cada punto a su centroide más cercano.

### DBSCAN
El parámetro `eps` se estima automáticamente usando el percentil 90 de las
distancias al 5° vecino más cercano (gráfica k-NN incluida).

### GMM – Gaussian Mixture Models
Se prueban las cuatro variantes de covarianza (`full`, `tied`, `diag`,
`spherical`). Se selecciona la de menor **BIC**.

---

## Re-evaluación de etiquetas

Se aplica una corrección por k-vecinos más cercanos (k=15):

1. Para cada empresa, se observa el cluster mayoritario entre sus 15 vecinos.
2. Si el cluster propio representa menos del 40 % del vecindario
   (**umbral de confianza**), la etiqueta se reemplaza por la mayoritaria.
3. Se reporta el porcentaje re-asignado (típicamente 20–35 %).

Este paso produce dos columnas en el dataset final:
- `label_original` — asignación directa del mejor método de clustering
- `label_corrected` — etiqueta revisada por k-NN

---

## Modelos supervisados

| Modelo | Detalles |
|--------|----------|
| Árbol de Decisión (depth=6) | Interpretable, reglas exportadas a `.txt` |
| Árbol de Decisión (depth=10) | Mayor capacidad expresiva |
| Regresión Logística | `lbfgs`, `max_iter=1000`, multiclase nativo |
| Regresión Lineal OvR | One-vs-Rest con `LinearRegression`, umbral por argmax |

La evaluación usa **5-fold Stratified CV** con métricas `accuracy` y `f1_weighted`.

---

## Comparación

`out/comparacion_resultados.csv` consolida para cada modelo:

| Columna | Descripción |
|---------|-------------|
| `ACC_mean_CORR` | Accuracy con etiquetas corregidas |
| `F1_mean_CORR` | F1 weighted con etiquetas corregidas |
| `ACC_mean_ORIG` | Accuracy con etiquetas originales |
| `F1_mean_ORIG` | F1 weighted con etiquetas originales |
| `Delta_ACC` | Diferencia (corregidas − originales) |
| `Delta_F1` | Diferencia (corregidas − originales) |

---

## Parámetros configurables (en `ml_pipeline.py`)

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `FILE_PATH` | ruta local | Ruta al CSV de entrada |
| `N_CLUSTERS` | automático | Se determina por silhouette de K-Means |
| `K_EVAL` | 15 | Vecinos para la re-evaluación de etiquetas |
| `THRESHOLD` | 0.40 | Umbral de confianza para reasignación |
| `factor` (IQR) | 5.0 | Factor de eliminación de outliers extremos |

---

## Salidas principales

| Archivo | Descripción |
|---------|-------------|
| `cluster_metricas.csv` | Silhouette, Davies-Bouldin y Calinski-Harabasz por método |
| `sup_classification_reports.txt` | Precision / Recall / F1 por clase en test set |
| `sup_decision_tree_rules.txt` | Reglas del árbol de decisión en texto plano |
| `comparacion_resultados.csv` | Tabla comparativa etiquetas corregidas vs. originales |
| `dataset_final_con_clusters.csv` | Dataset enriquecido con clusters y features escaladas |

---

## Licencia

Uso educativo / investigación. Los datos son de acceso público a través
de la Superintendencia de Sociedades de Colombia.