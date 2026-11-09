import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

RUTA_DATOS = Path(__file__).resolve().parent.parent / "datos_senas"
ARCHIVO = os.path.join(RUTA_DATOS, "dataset_din.npz")

if not os.path.exists(ARCHIVO):
    raise FileNotFoundError("No existe dataset_din.npz. Ejecuta primero capturar_dinamicas.py")

data = np.load(ARCHIVO, allow_pickle=False)
X = data["X"].astype(np.float32)          # (N, 30, 63)
y = data["y"].astype(str)

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
print(f"Secuencias: {len(X)} | Clases: {list(encoder.classes_)}")

if len(set(y)) < 2:
    raise ValueError("Necesitas al menos 2 clases (por ejemplo J y la clase 0 'sin sena').")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

modelo = keras.Sequential([
    layers.Input(shape=(30, 63)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(encoder.classes_), activation="softmax"),
])
modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

modelo.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=60, batch_size=16,
           callbacks=[keras.callbacks.EarlyStopping(patience=10,
                                                    restore_best_weights=True)])

loss, acc = modelo.evaluate(X_te, y_te, verbose=0)
print(f"\nPrecision en test: {acc:.2%}")

modelo.save(os.path.join(RUTA_DATOS, "modelo_din.keras"))
joblib.dump(encoder, os.path.join(RUTA_DATOS, "encoder_din.joblib"), compress=3)
print("Modelo dinámico guardado en:", RUTA_DATOS)