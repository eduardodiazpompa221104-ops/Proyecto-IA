import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

RUTA_DATOS = "datos_senas"
ARCHIVO_DATASET = os.path.join(RUTA_DATOS, "dataset.npz")
ARCHIVO_MODELO = os.path.join(RUTA_DATOS, "modelo_lsm.joblib")
ARCHIVO_ENCODER = os.path.join(RUTA_DATOS, "encoder.joblib")

#  Cargar datos 
if not os.path.exists(ARCHIVO_DATASET):
    raise FileNotFoundError("No existe dataset.npz. Ejecuta primero capturar_datos.py")

data = np.load(ARCHIVO_DATASET, allow_pickle=False)
X = data["X"].astype(np.float32)
y = data["y"].astype(str)

print(f"Muestras: {len(X)} | Clases: {len(set(y))}")

#  Codificar etiquetas 
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
print(f"Clases: {list(encoder.classes_)}")

# Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Modelo 
modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)

# Validación cruzada 
scores = cross_val_score(modelo, X_train, y_train, cv=5, n_jobs=-1)
print(f"CV (5-fold): {scores.mean():.2%} +/- {scores.std():.2%}")

# Entrenamiento final 
modelo.fit(X_train, y_train)

# Evaluaciónm 
y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nPrecision en test: {acc:.2%}\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

# Guardar con joblib 
joblib.dump(modelo, ARCHIVO_MODELO, compress=3)
joblib.dump(encoder, ARCHIVO_ENCODER, compress=3)
print(f"Modelo guardado en: {ARCHIVO_MODELO}")
print(f"Encoder guardado en: {ARCHIVO_ENCODER}")