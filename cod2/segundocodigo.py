import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
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

num_features = X_entrenamiento.shape[1]

componentes_optimos = [min(12, num_features), min(10, num_features), min(11, num_features),
                       min(9, num_features), min(5, num_features), min(3, num_features)]

f1_scores = {}

for n_componentes in componentes_optimos:
    print(f"\nEvaluando con {n_componentes} componentes principales:")
    pca = PCA(n_components=n_componentes)
    X_entrenamiento_pca = pca.fit_transform(X_entrenamiento)
    X_prueba_pca = pca.transform(X_prueba)

    grid_search = GridSearchCV(
        estimator=clasificador, param_grid=parametros, cv=3, scoring='f1_weighted', n_jobs=-1
    )
    grid_search.fit(X_entrenamiento_pca, y_entrenamiento)

    y_pred_pca = grid_search.best_estimator_.predict(X_prueba_pca)
    f1_score_pca = f1_score(y_prueba, y_pred_pca, average='weighted')
    f1_scores[n_componentes] = f1_score_pca

    print("Mejores parámetros encontrados:", grid_search.best_params_)
    print("\nReporte de clasificación:")
    print(classification_report(y_prueba, y_pred_pca, zero_division=0))

print("\nF1 scores por cantidad de componentes:")
for n_componentes, score in f1_scores.items():
    print(f"Componentes {n_componentes}: F1 score {score:.4f}")

mejor_numero_componentes = max(f1_scores, key=f1_scores.get)
print(f"\nNúmero óptimo de componentes principales: {mejor_numero_componentes}")

scores_50_50 = []

scores_20_80 = []

for _ in range(100):
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.5, random_state=np.random.randint(0, 1000), stratify=y
    )
    X_entrenamiento, y_entrenamiento = smote.fit_resample(X_entrenamiento, y_entrenamiento)
    clasificador.fit(X_entrenamiento, y_entrenamiento)
    y_pred = clasificador.predict(X_prueba)
    scores_50_50.append(f1_score(y_prueba, y_pred, average='weighted'))

for _ in range(100):
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.2, random_state=np.random.randint(0, 1000), stratify=y
    )
    X_entrenamiento, y_entrenamiento = smote.fit_resample(X_entrenamiento, y_entrenamiento)
    clasificador.fit(X_entrenamiento, y_entrenamiento)
    y_pred = clasificador.predict(X_prueba)
    scores_20_80.append(f1_score(y_prueba, y_pred, average='weighted'))

print("\nMediana de la confiabilidad con 100 ejecuciones (50/50 split):", median(scores_50_50))
print("\nMediana de la confiabilidad con 100 ejecuciones (20/80 split):", median(scores_20_80))
