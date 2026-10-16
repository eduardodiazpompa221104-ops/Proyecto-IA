import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from collections import deque, Counter

# Rutas 
RUTA_DATOS = "datos_senas"
ARCHIVO_MODELO = os.path.join(RUTA_DATOS, "modelo_lsm.joblib")
ARCHIVO_ENCODER = os.path.join(RUTA_DATOS, "encoder.joblib")

if not (os.path.exists(ARCHIVO_MODELO) and os.path.exists(ARCHIVO_ENCODER)):
    raise FileNotFoundError("Faltan modelo_lsm.joblib o encoder.joblib. Entrena primero.")

modelo = joblib.load(ARCHIVO_MODELO)
encoder = joblib.load(ARCHIVO_ENCODER)

#  MediaPipe 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

#  Configuración de estabilidad 
BUFFER_SIZE = 12            # Frames suavizados
UMBRAL_CONFIRMACION = 18    # Frames seguidos con misma letra para agregar
FRAMES_SIN_MANO_RESET = 30  # Si no hay mano, reiniciar contador

historial = deque(maxlen=BUFFER_SIZE)
letra_actual = None
frames_confirmados = 0
frames_sin_mano = 0
ultima_agregada = None

frase = ""
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara.")

print("Presiona 'q' = salir | 'espacio' = espacio | 'b' = borrar letra | 'c' = limpiar frase")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    prediccion_frame = None

    if results.multi_hand_landmarks:
        frames_sin_mano = 0
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.asarray(landmarks, dtype=np.float32).reshape(1, -1)

            idx = modelo.predict(landmarks)[0]
            prediccion_frame = str(encoder.inverse_transform([idx])[0])

        historial.append(prediccion_frame)

        # Voto mayoritario del buffer
        if historial:
            letra_actual = Counter(historial).most_common(1)[0][0]

        cv2.putText(frame, f"Sena: {letra_actual}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    else:
        frames_sin_mano += 1
        if frames_sin_mano >= FRAMES_SIN_MANO_RESET:
            historial.clear()
            letra_actual = None
            frames_confirmados = 0
            ultima_agregada = None
        cv2.putText(frame, "Sin mano", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Confirmación para agregar letra 
    if letra_actual is not None:
        if letra_actual == ultima_agregada:
            frames_confirmados += 1
        else:
            frames_confirmados = 1
            ultima_agregada = letra_actual

        if frames_confirmados == UMBRAL_CONFIRMACION:
            frase += letra_actual
            print(f"+ '{letra_actual}' | Frase: {frase}")
            frames_confirmados = 0

    # Overlay 
    cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]),
                  (0, 0, 0), -1)
    cv2.putText(frame, f"Frase: {frase}", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("Reconocimiento LSM (Py 3.13)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        frase += " "
    elif key == ord('b'):
        frase = frase[:-1]
    elif key == ord('c'):
        frase = ""

cap.release()
cv2.destroyAllWindows()
print(f"\nFrase final: {frase}")