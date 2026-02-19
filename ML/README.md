# EDA y Preprocesamiento – 10,000 Empresas más Grandes del País
### Pipeline para Clustering por Apalancamiento Financiero

---

## Descripción General

Este script realiza el **análisis exploratorio (EDA)** y el **preprocesamiento** del dataset *10,000 Empresas más Grandes del País* (corte a febrero 2026), con el objetivo de construir un conjunto de features limpias y estandarizadas para un posterior modelo de clustering basado en indicadores de apalancamiento financiero.

---

## Dataset de Entrada

| Atributo | Detalle |
|---|---|
| Archivo | `10_000_Empresas_mas_Grandes_del_País_20260210.csv` |
| Filas | 40,000 |
| Columnas | 14 |
| Años disponibles | 2021, 2022, 2023, 2024 |
| Cobertura | Empresas supervisadas por Supersociedades y Superfinanciera |

### Columnas principales

- **Identificación:** `NIT`, `RAZÓN SOCIAL`, `SUPERVISOR`, `REGIÓN`, `DEPARTAMENTO DOMICILIO`, `CIUDAD DOMICILIO`, `CIIU`, `MACROSECTOR`, `Año de Corte`
- **Variables financieras:** `INGRESOS OPERACIONALES`, `GANANCIA (PÉRDIDA)`, `TOTAL ACTIVOS`, `TOTAL PASIVOS`, `TOTAL PATRIMONIO` *(en billones COP, formato `$XX.XX`)*

---

## Estructura del Script

```
eda_preprocesamiento_apalancamiento.py
│
├── Sección 1 – Carga de datos
├── Sección 2 – Limpieza y conversión de tipos
├── Sección 3 – Análisis exploratorio (EDA) + 5 gráficas
├── Sección 4 – Ingeniería de features de apalancamiento
├── Sección 5 – Preprocesamiento + 3 gráficas
└── Sección 6 – Exportación del dataset procesado
```

---

## Sección 1 — Carga de Datos

Se carga el CSV con `pandas`. Se imprime el shape inicial y la lista de columnas para verificar la integridad de la carga.

---

## Sección 2 — Limpieza y Conversión de Tipos

Las columnas financieras vienen como strings con formato monetario (`$216.85`). Se aplica la siguiente transformación:

1. Eliminar el símbolo `$`
2. Eliminar separadores de miles (comas)
3. Convertir a `float`

El campo `Año de Corte` viene con comas (`2,022`) y se convierte a entero. El `NIT` se limpia de comas y espacios.

---

## Sección 3 — Análisis Exploratorio (EDA)

Se generan 5 gráficas guardadas como PNG:

| Archivo | Contenido |
|---|---|
| `eda_01_distribucion_empresas.png` | Conteo de empresas por año de corte y por macrosector |
| `eda_02_distribuciones_financieras.png` | Histogramas de cada variable financiera en escala logarítmica (`log1p`) para visualizar la distribución sin que los outliers extremos dominen |
| `eda_03_boxplot_leverage_macrosector.png` | Boxplot del ratio **Pasivos/Activos** por macrosector, ordenado por mediana descendente — primer vistazo al apalancamiento sectorial |
| `eda_04_correlacion.png` | Heatmap de correlación (triángulo inferior) entre las 5 variables financieras brutas |
| `eda_05_evolucion_temporal.png` | Evolución de la mediana de Activos, Pasivos y Patrimonio por año, para detectar tendencias estructurales |

---

## Sección 4 — Ingeniería de Features de Apalancamiento

Se construyen **7 ratios financieros** clásicos orientados a medir el nivel de deuda y rentabilidad de cada empresa:

| Feature | Fórmula | Interpretación |
|---|---|---|
| `deuda_activos` | Pasivos / Activos | Proporción de activos financiada con deuda |
| `deuda_patrimonio` | Pasivos / \|Patrimonio\| | Nivel de apalancamiento financiero (D/E ratio) |
| `multiplicador_cap` | Activos / \|Patrimonio\| | Efecto multiplicador del apalancamiento sobre el capital |
| `cobertura_ingresos` | Ingresos Op. / Pasivos | Capacidad de ingresos para cubrir obligaciones |
| `margen_neto` | Ganancia / \|Ingresos Op.\| | Rentabilidad sobre ventas |
| `roa` | Ganancia / Activos | Retorno sobre activos |
| `roe` | Ganancia / \|Patrimonio\| | Retorno sobre capital propio |

> Se usa `eps = 1e-9` en todos los denominadores para evitar divisiones por cero.

---

## Sección 5 — Preprocesamiento

El preprocesamiento sigue un pipeline de 5 pasos secuenciales:

### Paso 1 — Filtrado de registros inválidos
Se eliminan filas con `TOTAL ACTIVOS <= 0`, ya que ratios como `deuda_activos` o `roa` no tienen sentido económico con activos negativos o nulos.

### Paso 2 — Eliminación de outliers extremos (IQR × 5)
Para cada feature de apalancamiento se calculan los percentiles 1 y 99, y se elimina cualquier fila que quede fuera del rango `[Q1 - 5·IQR, Q3 + 5·IQR]`. El factor 5 es conservador a propósito: solo remueve casos verdaderamente aberrantes sin truncar la variabilidad natural del dataset.

### Paso 3 — Imputación de NaN e Inf residuales
Valores `inf` y `-inf` se convierten a `NaN` y se rellenan con la **mediana** de cada feature, garantizando que no queden datos faltantes antes de la transformación.

### Paso 4 — Transformación Log-Modulus
Se aplica la función:

```
log_modulus(x) = sign(x) · log(1 + |x|)
```

Esta transformación reduce el sesgo de la distribución y maneja valores negativos (a diferencia de `log1p` puro). Es especialmente útil para ratios como `roe` o `margen_neto` que pueden ser fuertemente negativos en empresas con pérdidas.

### Paso 5 — Estandarización con RobustScaler
Se escala cada feature transformada usando `RobustScaler` de scikit-learn, que centra en la mediana y escala por el IQR. Es más robusto que `StandardScaler` ante outliers residuales.

### Gráficas del preprocesamiento

| Archivo | Contenido |
|---|---|
| `prepro_01_distribuciones_finales.png` | Histogramas de las 7 features tras log-modulus + RobustScaler |
| `prepro_02_pairplot_features.png` | Pairplot (muestra de 2,000 empresas) de las 4 features principales para visualizar separabilidad y estructura |
| `prepro_03_correlacion_post.png` | Heatmap de correlación entre las features ya procesadas, para detectar redundancia antes del clustering |

---

## Dataset de Salida

El script exporta `dataset_procesado_apalancamiento.csv` con la siguiente estructura:

- **Columnas de identificación:** `NIT`, `RAZÓN SOCIAL`, `MACROSECTOR`, `REGIÓN`, `Año de Corte`
- **Ratios originales (sin escalar):** `deuda_activos`, `deuda_patrimonio`, `multiplicador_cap`, `cobertura_ingresos`, `margen_neto`, `roa`, `roe`
- **Features procesadas (listas para clustering):** `deuda_activos_lm`, `deuda_patrimonio_lm`, `multiplicador_cap_lm`, `cobertura_ingresos_lm`, `margen_neto_lm`, `roa_lm`, `roe_lm`

---

## Dependencias

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

| Librería | Versión recomendada | Uso |
|---|---|---|
| pandas | ≥ 1.5 | Carga y manipulación de datos |
| numpy | ≥ 1.23 | Operaciones numéricas |
| matplotlib | ≥ 3.6 | Visualizaciones base |
| seaborn | ≥ 0.12 | Heatmaps, boxplots, pairplot |
| scipy | ≥ 1.9 | Estadística descriptiva auxiliar |
| scikit-learn | ≥ 1.1 | RobustScaler |

---
