# NOMBRE: Jorge Luis Luis
# Informática Industrial Avanzada
# Máster en Ingeniería Industrial
# Universidad de La Laguna
# Curso 2025- 2026

# PROYECTO DE ANÁLISIS DE DATOS:

# => 1º PARTE: Preprocesamiento de Datos
# --------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

## Cargamos el dataset de "Wine Quality" y mostramos información basica de este
wine_dataset = fetch_ucirepo(id=186) 
df_vinos = wine_dataset.data.original
print("\n-> Primeras 5 filas del dataset:\n")
print(df_vinos.head())
print("\n-> Últimas 5 filas del dataset:\n")
print(df_vinos.tail())

## Con la siguiente información podemos observar que no hay valores nulos y que la mayoria de características son de tipo float64.
print("\n-> Información general del dataset:\n") 
df_vinos.info()
print("\n-> Estadísticas de las columnas numéricas:\n")
print(df_vinos.describe())

## Separamos las características del dataset y el objetivo de predicción:
caracteristicas = wine_dataset.data.features
objetivo = wine_dataset.data.targets['quality']

plt.figure(1)
sns.countplot(x=objetivo)
plt.title('Distribución de la calidad del vino')
plt.xlabel('Calidad del vino')
plt.ylabel('Cantidad de vinos')

## Como en la gráfica no se aprecia bien las cantidades exactas:
print("\n-> Cantidad de vinos segun su calidad:")
print(objetivo.value_counts().sort_index())

## Miramos si las variables están muy relacionadas entre sí
plt.figure(2)
matriz_corr = caracteristicas.corr()
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Correlación de Variables')

######## - CONCLUSIONES QUE SE REALIZAN DE LOS RESULTADOS OBTENIDOS - ########
"""Tras la exploración del dataset, se puede observar que se etsá ante un conjunto de datos donde la gran mayoria de las características son numéricas (tipo float64) y,
afortunadamente, no existen valores nulos o vacíos en ninguna de las muestras, lo que simplifica la fase de limpieza. Al analizar el diagrama de barras de la variable objetivo (calidad), se ve un 
desbalanceo en las clases: la mayoría de las muestras se concentran en calidades medias (notas 5 y 6), con muy pocos registros de vinos muy buenos o de muy baja calidad, un factor que dificultará 
el aprendizaje de los modelos en los casos extremos al no disponer de suficientes ejemplos de entrenamiento. Por último, la matriz de correlación nos revela que ciertas variables presentan altos niveles de 
dependencia entre sí (como ocurre con la densidad, el alcohol o los dióxidos de azufre), lo cual demuestra que existe redundancia en los datos y justifica la necesidad de aplicar la técnica de reducción de 
dimensionalidad: PCA, para simplificar el modelo sin perder información clave."""

# => 2º PARTE: Reducción de Dimensionalidad usando PCA
# -----------------------------------------------------

## Antes de aplicar PCA, es importante escalar las características para que todas tengan la misma importancia.
scaler = StandardScaler()
X_escaled = scaler.fit_transform(caracteristicas)
pca_analisis = PCA()
pca_analisis.fit(X_escaled)
varianza_acumulada = np.cumsum(pca_analisis.explained_variance_ratio_) # Calculamos la varianza acumulada para determinar cuántos componentes principales necesitamos (hasta llegar al 95% de esta)
# Gráficamos:
plt.figure(3)
plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title('Varianza Explicada por el PCA')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.show()

## Se decide emplear 9 componentes (con 8 nos quedamos un poco por debajo del 95%). Esto simplifica el modelo con respecto el original del comienzo.
pca_final = PCA(n_components=9)
X_pca = pca_final.fit_transform(X_escaled)

# => 3º PARTE: Clasificación
# ---------------------------

## Se usarán 3 modelos diferentes para comparar su rendimiento con los datos escalados sin PCA y con PCA:
## 1. LDA (Análisis Discriminante Lineal)
## 2. Bosque Aleatorio (Random Forest): Supuestamente es muy potente contra el sobreajuste (se prueba para ver el resultado que se obtiene y compararlo).
## 3. Naive Bayes

## Modelo 1: Análisis Discriminante Lineal (LDA)
modelo_lda = LinearDiscriminantAnalysis(solver='svd') ## Se usa este solveer porque es estable y porque maneja bastante bien datos con muchas características (como este caso de los vinos)
resultado_orig_lda = cross_val_score(modelo_lda, X_escaled, objetivo, cv=5).mean()
resultado_pca_lda = cross_val_score(modelo_lda, X_pca, objetivo, cv=5).mean()

print("1. Análisis Discriminante Lineal (LDA):")
print(f"Precisión sin PCA: {resultado_orig_lda:.4f}")
print(f"Precisión con PCA:  {resultado_pca_lda:.4f}\n")

# Modelo 2: Bosque Aleatorio
modelo_bosque = RandomForestClassifier(n_estimators=100, random_state=42) # Numero de estimadores empleados y aleatorieda en el proceso
resultado_orig_bosq = cross_val_score(modelo_bosque, X_escaled, objetivo, cv=5).mean()
resultado_pca_bosq = cross_val_score(modelo_bosque, X_pca, objetivo, cv=5).mean()

print("2. Bosque Aleatorio:")
print(f"Precisión sin PCA: {resultado_orig_bosq:.4f}")
print(f"Precisión con PCA:  {resultado_pca_bosq:.4f}\n")

# Modelo 3: Naive Bayes
modelo_bayes = GaussianNB()
resultado_orig_nb = cross_val_score(modelo_bayes, X_escaled, objetivo, cv=5).mean()
resultado_pca_nb = cross_val_score(modelo_bayes, X_pca, objetivo, cv=5).mean()

print("3. Naive Bayes:")
print(f"Precisión sin PCA: {resultado_orig_nb:.4f}")
print(f"Precisión con PCA:  {resultado_pca_nb:.4f}\n")

######## - CONCLUSIONES QUE SE REALIZAN DE LOS RESULTADOS OBTENIDOS - ########
"""Al comparar el rendimiento de los tres modelos evaluados, se observa comportamientos interesantes sobre el uso del PCA. Por un lado, tanto el Análisis Discriminante Lineal (LDA) 
como el Naive Bayes mejoran su precisión al aplicar reducción de dimensionalidad, destacando el salto del Naive Bayes (que pasa del 39.3% al 45.9%). Esto confirma que eliminar la redundancia y 
el ruido entre las variables favorece a estos clasificadores clásicos. Por otro lado, el Bosque Aleatorio sufre una leve caída en su rendimiento al usar PCA (del 48.1% al 46.6%); este comportamiento 
se debe a que los métodos basados en árboles gestionan de mejor forma la alta dimensionalidad y se resienten al aplicar reducción."""

# => 4º PARTE: Evaluación de los modelos y comparación
# ----------------------------------------------------

# Para el entrenamiento se empleara el 70% de los datos, y el 30% restante se usará para el test.
# Esta vez se emplea random_state, para así tener una aleatoriedad cuando se dividan los datos; y stratify, para que en dicho 30% de datos usados para el test haya la misma 
# proporción de vinos buenos, regulares y malos que en el dataset original. El PROBLEMA que se prevee que habrá es que como hay muy pocas muestras de vinos malos y buenos los 
# modelos fallarán principalmente al clasificar estos. Por otro lado, al haber más muestras de vinos regulares, los modelos seguramente tendrán un mejor rendimiento al clasificarlos. 
X_train_sin_pca, X_test_sin_pca, y_train, y_test = train_test_split(X_escaled, objetivo, test_size=0.3, random_state=42, stratify=objetivo)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, objetivo, test_size=0.3, random_state=42, stratify=objetivo)

print("\nMODELO 1: LDA")
# Entrenamos y predecimos SIN PCA
modelo_lda.fit(X_train_sin_pca, y_train)
pred_lda_sin_pca = modelo_lda.predict(X_test_sin_pca)
# Entrenamos y predecimos CON PCA
modelo_lda.fit(X_train_pca, y_train)
pred_lda_pca = modelo_lda.predict(X_test_pca)

print("\n-> Reporte SIN PCA:\n")
print(classification_report(y_test, pred_lda_sin_pca, zero_division=0))
print("\n-> Reporte CON PCA:\n")
print(classification_report(y_test, pred_lda_pca, zero_division=0))

cm_sin_pca = confusion_matrix(y_test, pred_lda_sin_pca)
cm_pca = confusion_matrix(y_test, pred_lda_pca)
# En la matriz de confusión cada fila representa las calificaciones reales, y cada columna representa las calificaciones predichas por el modelo.
# Comenzando desde la esquina  superior e inferior izq. con un valor de 3 y aumentando la clasificaión hasta un valor de 9 en la esquina inferior izq. y derecha respectivamnete.
print("\n-> Matriz de Confusión SIN PCA (LDA):\n")
print(cm_sin_pca)
print("\n-> Matriz de Confusión CON PCA (LDA):\n")
print(cm_pca)

######## - CONCLUSIONES QUE SE REALIZAN DE LOS RESULTADOS OBTENIDOS - ########
"""Al analizar el modelo LDA, se comprueba que su rendimiento es prácticamente idéntico tanto con los datos sin PCA como con los reducidos por PCA, manteniendo una exactitud global del 52%. 
Las matrices de confusión y los reportes muestran (como se predijo cuando se procesó y estudió el dataset) que el modelo hace un buen trabajo prediciendo los vinos de calidad media (5 y 6), 
pero falla al intentar identificar las calidades extremas (3, 4, 8 y 9) debido a la gran falta de muestras de estos tipos. La conclusión principal, a pesar de la exactitud del 52%, es positiva: 
el hecho de que la versión con PCA consiga prácticamente los mismos resultados demuestra que se ha logrado simplificar los datos y quitar variables redundantes sin sacrificar la capacidad de 
predicción del modelo."""

#------------------------------------------------------------------
print("\nMODELO 2: BOSQUE ALEATORIO")
modelo_bosque.fit(X_train_sin_pca, y_train)
pred_rf_sin_pca = modelo_bosque.predict(X_test_sin_pca)
modelo_bosque.fit(X_train_pca, y_train)
pred_rf_pca = modelo_bosque.predict(X_test_pca)

print("\n-> Reporte SIN PCA:\n")
print(classification_report(y_test, pred_rf_sin_pca, zero_division=0))
print("\n-> Reporte CON PCA:\n")
print(classification_report(y_test, pred_rf_pca, zero_division=0))

print("\n-> Matriz de Confusión SIN PCA (Bosque Aleatorio):\n")
cm_sin_pca = confusion_matrix(y_test, pred_rf_sin_pca)
print(cm_sin_pca)
print("\n-> Matriz de Confusión CON PCA (Bosque Aleatorio):\n")
cm_pca = confusion_matrix(y_test, pred_rf_pca)
print(cm_pca)

######## - CONCLUSIONES QUE SE REALIZAN DE LOS RESULTADOS OBTENIDOS - ########
"""En este caso, el Bosque Aleatorio supera al LDA, alcanzando un 68% de exactitud global. Al igual que observamos en el caso anterior, el rendimiento es exactamente el mismo (68%) 
usando los datos sin PCA que los reducidos por PCA. Aunque el desbalanceo de los datos sigue impidiendo que acierte las calidades extremas (3 y 9), este algoritmo demuestra ser mucho 
más robusto y capaz, logrando identificar correctamente varios vinos de notas 4 y 8 que el LDA fallaba por completo."""

#------------------------------------------------------------------
print("\nMODELO 3: NAIVE BAYES")
modelo_bayes.fit(X_train_sin_pca, y_train)
pred_nb_sin_pca = modelo_bayes.predict(X_test_sin_pca)
modelo_bayes.fit(X_train_pca, y_train)
pred_nb_pca = modelo_bayes.predict(X_test_pca)

print("\n-> Reporte SIN PCA:\n")
print(classification_report(y_test, pred_nb_sin_pca, zero_division=0))
print("\n-> Reporte CON PCA:\n")
print(classification_report(y_test, pred_nb_pca, zero_division=0))

print("\n-> Matriz de Confusión SIN PCA (Naive Bayes):\n")
cm_sin_pca = confusion_matrix(y_test, pred_nb_sin_pca)
print(cm_sin_pca)
print("\n-> Matriz de Confusión CON PCA (Naive Bayes):\n")
cm_pca = confusion_matrix(y_test, pred_nb_pca)
print(cm_pca)

######## - CONCLUSIONES QUE SE REALIZAN DE LOS RESULTADOS OBTENIDOS - ########
"""Con el modelo Naive Bayes se obtiene un resultado bastante significativo respecto al uso del PCA. Utilizando los datos originales, este modelo obtiene el peor rendimiento 
de todos (apenas un 42% de exactitud). Esto se debe a que Naive Bayes asume por regla matemática que todas las características son completamente independientes entre sí, algo que 
el mapa de correlacion demostró que era falso. Sin embargo, al entrenarlo con los datos reducidos por PCA, su exactitud da un salto considerable hasta el 51%. Esto se debe a que, al 
aplicar PCA, se eliminaron las mayores correlaciones, dandole al modelo unas variables un poco más independientes. Aunque sigue teniendo problemas para clasificar las calidades extremas,
el salto del 9% en su precisión es la prueba de que la reducción por PCA mejora drásticamente un modelo probabilístico."""

print("\nSE HA ALCANZADO EL FINAL DEL SCRIPT.")