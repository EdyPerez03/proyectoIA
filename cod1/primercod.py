import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from statistics import median

datos = pd.read_csv('exoplanets.csv')

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
clasificador.fit(X_entrenamiento, y_entrenamiento)

y_pred = clasificador.predict(X_prueba)

print("\nMatriz de Confusión:")
print(confusion_matrix(y_prueba, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_prueba, y_pred, zero_division=0))

scores = []
for _ in range(100):
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.5, random_state=np.random.randint(0, 1000), stratify=y
    )
    X_entrenamiento, y_entrenamiento = smote.fit_resample(X_entrenamiento, y_entrenamiento)
    clasificador.fit(X_entrenamiento, y_entrenamiento)
    y_pred = clasificador.predict(X_prueba)
    scores.append(f1_score(y_prueba, y_pred, average='weighted'))

print("\nMediana de la confiabilidad con 100 ejecuciones (50/50 split):", median(scores))
