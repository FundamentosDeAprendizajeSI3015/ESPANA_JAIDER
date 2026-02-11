import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Manejo de dependencia para Binary Encoding
try:
    from category_encoders import BinaryEncoder
except ImportError:
    print("[!] Instalando 'category_encoders'...")
    os.system('pip install category_encoders')
    from category_encoders import BinaryEncoder

def ejecutar_laboratorio():
    # 1. Crear carpetas necesarias
    if not os.path.exists('graficas'): os.makedirs('graficas')
    if not os.path.exists('salidas'): os.makedirs('salidas')

    # 2. Cargar datos
    df = pd.read_csv('Titanic-Dataset.csv')
    
    # 3. Estadísticas básicas
    mean_age = df['Age'].mean()
    median_age = df['Age'].median()
    mode_fare = df['Fare'].mode()[0]
    std_fare = df['Fare'].std()
    var_fare = df['Fare'].var()
    cuartiles_fare = df['Fare'].quantile([0.25, 0.5, 0.75])
    
    # 4. Gráficos y limpieza de datos
    # Boxplot con colores diferentes
    plt.figure(figsize=(8, 5))
    boxprops = dict(color='#E74C3C', linewidth=2)
    medianprops = dict(color='#2ECC71', linewidth=2)
    whiskerprops = dict(color='#3498DB', linewidth=1.5)
    capprops = dict(color='#3498DB', linewidth=1.5)
    flierprops = dict(markerfacecolor='#F1C40F', marker='o', markersize=6, alpha=0.6)
    
    plt.boxplot(df['Fare'].dropna(), 
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
                flierprops=flierprops,
                patch_artist=True)
    plt.title('Outliers en Tarifa (Fare)')
    plt.savefig('graficas/01_boxplot_fare.png')
    plt.close()

    # Eliminar outliers usando IQR
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    limite_sup = Q3 + 1.5 * IQR
    df_clean = df[df['Fare'] <= limite_sup].copy()
    outliers_eliminados = len(df) - len(df_clean)

    # Histogramas con colores diferentes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df_clean['Age'].dropna(), kde=True, ax=axes[0], 
                 color='#9B59B6', edgecolor='white', linewidth=1)
    axes[0].set_title('Distribución de Edad')
    
    sns.histplot(df_clean['Fare'], kde=True, ax=axes[1], 
                 color='#1ABC9C', edgecolor='white', linewidth=1)
    axes[1].set_title('Distribución de Tarifa')
    
    plt.savefig('graficas/02_histogramas.png')
    plt.close()

    # Gráfico de dispersión con colores diferentes
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_clean, x='Age', y='Fare', hue='Survived',
                   palette=['#E74C3C', '#2ECC71'], alpha=0.7)
    plt.title('Edad vs Tarifa')
    plt.savefig('graficas/03_dispersion_edad_fare.png')
    plt.close()

    # 5. Transformar columnas
    le = LabelEncoder()
    df_clean['Sex_Encoded'] = le.fit_transform(df_clean['Sex'])

    be = BinaryEncoder(cols=['Pclass'])
    df_clean = be.fit_transform(df_clean)

    df_clean = pd.get_dummies(df_clean, columns=['Embarked'], prefix='Port')

    scaler_std = StandardScaler()
    df_clean['Age_Scaled'] = scaler_std.fit_transform(df_clean[['Age']].fillna(df_clean['Age'].median()))

    scaler_mm = MinMaxScaler()
    df_clean['Fare_Scaled'] = scaler_mm.fit_transform(df_clean[['Fare']])

    df_clean['Fare_Log'] = np.log1p(df_clean['Fare'])

    # 6. Correlación
    plt.figure(figsize=(12, 10))
    matriz_corr = df_clean.corr(numeric_only=True)
    
    cmap_custom = sns.diverging_palette(20, 220, as_cmap=True)
    sns.heatmap(matriz_corr, annot=True, cmap=cmap_custom, fmt=".2f")
    plt.title('Matriz de Correlación')
    plt.savefig('graficas/04_matriz_correlacion.png')
    plt.close()

    # Quitar columnas no útiles
    df_final = df_clean.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Age', 'Fare'])

    # 7. Guardar resultados
    df_final.to_csv('salidas/titanic_limpio_2026.csv', index=False)

    with open('salidas/informe_EDA.md', 'w', encoding='utf-8') as m:
        m.write("# Informe de Análisis - Titanic\n\n")
        m.write("## 1. Estadísticas\n")
        m.write(f"* **Media de Edad:** {mean_age:.2f}\n")
        m.write(f"* **Mediana de Edad:** {median_age:.2f}\n")
        m.write(f"* **Desviación Estándar (Fare):** {std_fare:.2f}\n")
        m.write(f"* **Varianza (Fare):** {var_fare:.2f}\n\n")
        
        m.write("## 2. Outliers\n")
        m.write(f"* **Límite superior (IQR):** {limite_sup:.2f}\n")
        m.write(f"* **Registros eliminados:** {outliers_eliminados}\n\n")
        
        m.write("## 3. Conclusiones\n")
        m.write("1. La edad tiene distribución normal.\n")
        m.write("2. Hay correlación entre clase y supervivencia.\n")
        m.write("3. Se usó Binary Encoding para 'Pclass'.\n")
        m.write("4. Se escaló la edad para modelos de ML.")

    print("\n=== FIN ===")
    print(f"- Dataset: salidas/titanic_limpio_2026.csv")
    print(f"- Reporte: salidas/informe_EDA.md")
    print(f"- Gráficas: /graficas")

if __name__ == "__main__":
    ejecutar_laboratorio()