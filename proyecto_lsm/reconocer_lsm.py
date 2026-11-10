import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # menos ruido de TensorFlow
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # solo CPU
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
from pathlib import Path
# Reloj para medir la estabilidad en segundos, independientemente de los FPS.
from time import monotonic
from tensorflow import keras
# Separa la confirmación de letras y la lectura en voz del procesamiento de cámara.
from estabilidad import ConfirmadorLetra
from voz import ReproductorVoz

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
RUTA_DATOS = Path(__file__).resolve().parent.parent / "datos_senas"
ARCHIVO_MODELO = os.path.join(RUTA_DATOS, "modelo_lsm.joblib")
ARCHIVO_ENCODER = os.path.join(RUTA_DATOS, "encoder.joblib")
ARCHIVO_MODELO_DIN = os.path.join(RUTA_DATOS, "modelo_din.keras")
ARCHIVO_ENCODER_DIN = os.path.join(RUTA_DATOS, "encoder_din.joblib")

if not (os.path.exists(ARCHIVO_MODELO) and os.path.exists(ARCHIVO_ENCODER)):
    raise FileNotFoundError(
        "Faltan modelo_lsm.joblib o encoder.joblib. Entrena primero con entrenar_modelo.py.")

# Modelo estático (Random Forest)
modelo = joblib.load(ARCHIVO_MODELO)
encoder = joblib.load(ARCHIVO_ENCODER)

# Modelo dinámico (LSTM): opcional. Sin él, solo se reconocen señas estáticas
# y el movimiento se usa para rechazar predicciones dudosas.
modelo_din = None
encoder_din = None
if os.path.exists(ARCHIVO_MODELO_DIN) and os.path.exists(ARCHIVO_ENCODER_DIN):
    modelo_din = keras.models.load_model(ARCHIVO_MODELO_DIN)
    encoder_din = joblib.load(ARCHIVO_ENCODER_DIN)
else:
    print("Aviso: no hay modelo dinamico (modelo_din.keras). "
          "Solo se reconocen senas estaticas.")

# ----------------------------------------------------------------------------
# MediaPipe
# ----------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ----------------------------------------------------------------------------
# Parámetros
# ----------------------------------------------------------------------------
# Umbrales iniciales; ajustar con pruebas de la cámara y el modelo.
# La mejor letra debe alcanzar 55 % y superar a la segunda por 8 puntos porcentuales.
CONFIANZA_MINIMA = 0.55
MARGEN_MINIMO = 0.08
# Debe mantenerse 0.6 s; después se exige retirar la mano durante 0.4 s.
confirmador = ConfirmadorLetra(segundos_estables=0.6, segundos_sin_mano=0.4)

# Señas dinámicas
T = 30                       # frames por secuencia (igual que en el entrenamiento)
UMBRAL_MOVIMIENTO = 0.02     # variación de x,y en la ventana; ajustar con pruebas
CONFIANZA_DINAMICA = 0.75    # confianza mínima del LSTM
PREDECIR_CADA = 6            # corre el LSTM cada N frames para no bajar los FPS
CLASE_NADA = "_"             # clase "movimiento sin seña"
SEGUNDOS_AVISO_DIN = 1.5     # tiempo que se muestra la última letra dinámica

buffer = deque(maxlen=T)
contador_frames = 0
ultima_dinamica = None       # (letra, confianza, momento)

letra_actual = None
frase = ""
# Este objeto lee una copia de la frase sin detener la captura de video.
reproductor = ReproductorVoz()

# ----------------------------------------------------------------------------
# Cámara
# ----------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara.")

# Configurar resolución HD (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

NOMBRE_VENTANA = "Traductor de Lengua de Senas Mexicana"
cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
cv2.resizeWindow(NOMBRE_VENTANA, 1280, 720)

print("Manten la sena estable y retira la mano entre letras.")
print("Para J, Z, N(Ñ) y Q haz el movimiento completo de la sena.")
print("Presiona '1' = salir | 'espacio' = espacio | '2' = borrar letra | "
      "'c' = limpiar frase | 'v' = escuchar frase")

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

    ahora = monotonic()
    contador_frames += 1

    # Se reinician en cada imagen para no reutilizar una letra si se pierde la mano.
    prediccion_frame = None
    candidata = None
    confianza = 0.0
    margen = 0.0
    movimiento = 0.0
    letra_dinamica = None
    mano_presente = bool(results.multi_hand_landmarks)

    # ------------------------------------------------------------------
    # 1) Landmarks + predicción estática (Random Forest)
    # ------------------------------------------------------------------
    if results.multi_hand_landmarks:
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

            # Acumula la pose en el buffer temporal para las señas dinámicas.
            buffer.append(landmarks[0])

            # Consulta el puntaje de cada clase en el Random Forest.
            probabilidades = modelo.predict_proba(landmarks)[0]
            orden = np.argsort(probabilidades)
            mejor = orden[-1]
            confianza = float(probabilidades[mejor])
            segunda = float(probabilidades[orden[-2]]) if len(orden) > 1 else 0.0
            # Un margen pequeño indica que el modelo duda entre dos letras.
            margen = confianza - segunda
            idx = modelo.classes_[mejor]
            # Convierte la clase numérica en una letra, incluso si luego se rechaza.
            candidata = str(encoder.inverse_transform([idx])[0])
            if confianza >= CONFIANZA_MINIMA and margen >= MARGEN_MINIMO:
                prediccion_frame = candidata
    else:
        # Sin mano no hay secuencia válida.
        buffer.clear()

    # ------------------------------------------------------------------
    # 2) Detector de movimiento + predicción dinámica (LSTM)
    # ------------------------------------------------------------------
    en_movimiento = False
    if mano_presente and len(buffer) == T:
        secuencia = np.stack(buffer).reshape(T, 21, 3)
        # Variación de (x, y) de los landmarks a lo largo de la ventana.
        movimiento = float(np.std(secuencia[:, :, :2], axis=0).mean())
        en_movimiento = movimiento > UMBRAL_MOVIMIENTO

        if en_movimiento:
            # Con movimiento no se confía en la predicción estática
            # (evita J->I, Z->D, N->N, etc.).
            prediccion_frame = None

            if (modelo_din is not None and not confirmador.esperando_retiro
                    and contador_frames % PREDECIR_CADA == 0):
                try:
                    p = modelo_din(secuencia[np.newaxis], training=False).numpy()[0]
                    k = int(np.argmax(p))
                    letra_din = str(encoder_din.inverse_transform([k])[0])
                    if p[k] >= CONFIANZA_DINAMICA and letra_din != CLASE_NADA:
                        letra_dinamica = letra_din
                        confianza = float(p[k])
                except Exception as error:
                    print(f"Error en el modelo dinamico: {error}")

    if letra_dinamica is not None:
        frase += letra_dinamica
        ultima_dinamica = (letra_dinamica, confianza, ahora)
        print(f"+ '{letra_dinamica}' (dinamica {confianza:.0%}) | Frase: {frase}")
        # Exige retirar la mano antes de la siguiente letra, igual que en las estáticas.
        confirmador.esperando_retiro = True
        buffer.clear()

    # ------------------------------------------------------------------
    # 3) Confirmación temporal de letras estáticas
    # ------------------------------------------------------------------
    letra_actual = prediccion_frame
    # Solo devuelve una letra cuando lleva suficiente tiempo siendo aceptada.
    # Devuelve None si hay incertidumbre, falta la mano o falta retirarla.
    nueva_letra = confirmador.actualizar(prediccion_frame, mano_presente, ahora)
    if nueva_letra is not None:
        frase += nueva_letra
        buffer.clear()
        print(f"+ '{nueva_letra}' | Frase: {frase}")

    # ------------------------------------------------------------------
    # 4) Interfaz
    # ------------------------------------------------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)           # Panel superior
    cv2.rectangle(overlay, (0, h - 80), (w, h), (15, 15, 15), -1)        # Panel inferior
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    mostrar_din = (ultima_dinamica is not None
                   and ahora - ultima_dinamica[2] <= SEGUNDOS_AVISO_DIN)

    if mostrar_din:
        letra_d, conf_d, _ = ultima_dinamica
        cv2.putText(frame, f"Sena dinamica: {letra_d} ({conf_d:.0%})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
    elif letra_actual is not None:
        # Barra de progreso visual para confirmar la letra
        progreso = confirmador.progreso
        cv2.rectangle(frame, (20, 52), (20 + int(progreso * 200), 58), (0, 255, 0), -1)
        cv2.putText(frame, f"Sena: {letra_actual} ({confianza:.0%})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2)
    else:
        if mano_presente and en_movimiento:
            estado = "Movimiento detectado: analizando sena dinamica..."
        elif mano_presente:
            # Mostrar la candidata rechazada ayuda a identificar confusiones del modelo.
            estado = f"Candidata: {candidata} ({confianza:.0%}) - incierta"
        else:
            estado = "Sin mano detectada: muestra la mano completa"
        cv2.putText(frame, estado, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 255), 2)
        if mano_presente and not confirmador.esperando_retiro:
            cv2.putText(frame,
                        f"Minimo: {CONFIANZA_MINIMA:.0%} | Margen: {margen:.0%}/{MARGEN_MINIMO:.0%}"
                        f" | Mov: {movimiento:.3f}/{UMBRAL_MOVIMIENTO:.3f}",
                        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)

    if confirmador.esperando_retiro:
        cv2.putText(frame, "Retira la mano antes de la siguiente letra", (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    # Frase construida
    cv2.putText(frame, f"Frase: {frase}", (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Guía de atajos
    cv2.putText(frame, "[1] Salir | [2] Borrar | [Espacio] Espacio | [C] Limpiar | [V] Escuchar frase",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow(NOMBRE_VENTANA, frame)

    # ------------------------------------------------------------------
    # 5) Teclado
    # ------------------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        break
    elif key == ord(' '):
        frase += " "
    elif key == ord('2'):
        frase = frase[:-1]
    elif key in (ord('c'), ord('C')):
        frase = ""
    elif key in (ord('v'), ord('V')):
        # La lectura comienza por petición del usuario, al terminar de formar la frase.
        reproductor.reproducir(frase)

cap.release()
cv2.destroyAllWindows()
print(f"\nFrase final: {frase}")