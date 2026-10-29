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
    raise FileNotFoundError("Faltan modelo_lsm.joblib o encoder.joblib. Entrena primero con entrenar_modelo.py.")

modelo = joblib.load(ARCHIVO_MODELO)
encoder = joblib.load(ARCHIVO_ENCODER)

# MediaPipe 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Configuración de estabilidad 
BUFFER_SIZE = 12 
UMBRAL_CONFIRMACION = 18 
FRAMES_SIN_MANO_RESET = 30 

historial = deque(maxlen=BUFFER_SIZE)
letra_actual = None
frames_confirmados = 0
frames_sin_mano = 0
ultima_agregada = None

frase = ""
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara.")

# Configurar resolución amplia a HD (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

NOMBRE_VENTANA = "Traductor de Lengua de Senas Mexicana"
cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
cv2.resizeWindow(NOMBRE_VENTANA, 1280, 720)

print("Presiona '1' = salir | 'espacio' = espacio | '2' = borrar letra | 'c' = limpiar frase")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
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
                mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2),
            )
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.asarray(landmarks, dtype=np.float32).reshape(1, -1)

            idx = modelo.predict(landmarks)[0]
            prediccion_frame = str(encoder.inverse_transform([idx])[0])

        historial.append(prediccion_frame)

        if historial:
            letra_actual = Counter(historial).most_common(1)[0][0]
    else:
        frames_sin_mano += 1
        if frames_sin_mano >= FRAMES_SIN_MANO_RESET:
            historial.clear()
            letra_actual = None
            frames_confirmados = 0
            ultima_agregada = None

    # Confirmación para agregar letra a la frase
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

    # PANEL SUPERIOR E INFERIOR ESTILIZADO CON OPACIDAD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)           # Panel superior
    cv2.rectangle(overlay, (0, h - 80), (w, h), (15, 15, 15), -1)        # Panel inferior
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Texto en el panel superior (Detección en vivo)
    if letra_actual is not None:
        # Barra de progreso visual para confirmar la letra
        progreso = min(frames_confirmados / UMBRAL_CONFIRMACION, 1.0)
        cv2.rectangle(frame, (20, 52), (20 + int(progreso * 200), 58), (0, 255, 0), -1)
        
        cv2.putText(frame, f"Sena detectada: {letra_actual}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2)
    else:
        cv2.putText(frame, "Esperando mano...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 255), 2)

    # Texto en el panel inferior (Frase construida)
    cv2.putText(frame, f"Frase: {frase}", (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Guía de atajos
    cv2.putText(frame, "[1] Salir  |  [2] Borrar letra  |  [Espacio] Espacio  |  [C] Limpiar",
                (w - 550, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow(NOMBRE_VENTANA, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        break
    elif key == ord(' '):
        frase += " "
    elif key == ord('2'):
        frase = frase[:-1]
    elif key == ord('c'):
        frase = ""

cap.release()
cv2.destroyAllWindows()
print(f"\nFrase final: {frase}")