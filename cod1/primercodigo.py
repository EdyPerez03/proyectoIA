import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from statistics import median

datos = pd.read_csv('exoplanets.csv')

print("Primeras filas del DataFrame:")
print(datos.head())
print("\nInformación del DataFrame:")
print(datos.info())

columnas_a_numerico = [
    "Mass (MJ)", "Radius (RJ)", "Period (days)",
    "Semi-major axis (AU)", "Temp. (K)", "Distance (ly)",
    "Host star mass (M☉)", "Host star temp. (K)"
]
for col in columnas_a_numerico:
    datos[col] = pd.to_numeric(datos[col], errors='coerce')

imputador = SimpleImputer(strategy="median")
datos[columnas_a_numerico] = imputador.fit_transform(datos[columnas_a_numerico])

datos["Mass (Tierra)"] = datos["Mass (MJ)"] * 317.8
datos["Radius (Tierra)"] = datos["Radius (RJ)"] * 11.2

columnas_a_escalar = ["Mass (Tierra)", "Radius (Tierra)", "Period (days)", "Semi-major axis (AU)", "Temp. (K)"]
escalador = MinMaxScaler()
datos[columnas_a_escalar] = escalador.fit_transform(datos[columnas_a_escalar])

clases_validas = datos['Discovery method'].value_counts()[datos['Discovery method'].value_counts() > 2].index
datos = datos[datos['Discovery method'].isin(clases_validas)]

X = datos[columnas_a_escalar]
y = datos['Discovery method']

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=45, stratify=y
)

smote = SMOTE(random_state=45, k_neighbors=1)
X_entrenamiento, y_entrenamiento = smote.fit_resample(X_entrenamiento, y_entrenamiento)

clasificador = RandomForestClassifier(random_state=45, class_weight='balanced')
parametros = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=clasificador, param_grid=parametros, cv=3, scoring='f1_weighted', n_jobs=-1
)
grid_search.fit(X_entrenamiento, y_entrenamiento)

print("Mejores parámetros encontrados:")
print(grid_search.best_params_)
print("\nReporte de clasificación:")
y_pred = grid_search.best_estimator_.predict(X_prueba)
print(classification_report(y_prueba, y_pred, zero_division=0))

pca = PCA(n_components=3)
X_entrenamiento_pca = pca.fit_transform(X_entrenamiento)
X_prueba_pca = pca.transform(X_prueba)

clasificador.fit(X_entrenamiento_pca, y_entrenamiento)
y_pred_pca = clasificador.predict(X_prueba_pca)

print("\nResultados con PCA:")
print(confusion_matrix(y_prueba, y_pred_pca))
print(classification_report(y_prueba, y_pred_pca))

kmeans = KMeans(n_clusters=3, random_state=45)
kmeans.fit(X)
datos['Cluster'] = kmeans.labels_

caracteristicas_tierra = {
    "Mass (Tierra)": 1,
    "Radius (Tierra)": 1,
    "Period (days)": 365.25,
    "Semi-major axis (AU)": 1,
    "Temp. (K)": 288
}

def calcular_similitud(fila):
    return np.sqrt(sum((fila[col] - caracteristicas_tierra[col])**2 for col in caracteristicas_tierra))

datos['Similitud'] = datos[columnas_a_escalar].apply(calcular_similitud, axis=1)
mas_similar = datos.loc[datos['Similitud'].idxmin()]
print("\nExoplaneta más similar a la Tierra:")
print(mas_similar)

scores = []
for _ in range(5):
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.2, random_state=45, stratify=y
    )
    clasificador.fit(X_entrenamiento, y_entrenamiento)
    y_pred = clasificador.predict(X_prueba)
    scores.append(f1_score(y_prueba, y_pred, average='weighted'))

print("\nMediana de la confiabilidad:", median(scores))
