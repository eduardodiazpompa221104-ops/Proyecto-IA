import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from pathlib import Path
# Reloj para medir la estabilidad en segundos, independientemente de los FPS.
from time import monotonic
# Separa la confirmación de letras y la lectura en voz del procesamiento de cámara.
from estabilidad import ConfirmadorLetra
from voz import ReproductorVoz

# Rutas 
RUTA_DATOS = Path(__file__).resolve().parent.parent / "datos_senas"
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

# Umbrales iniciales; ajustar con pruebas de la cámara y el modelo.
# Son valores iniciales, no una medida de precisión del modelo.
CONFIANZA_MINIMA = 0.55
MARGEN_MINIMO = 0.08
# La mejor letra debe alcanzar 55 % y superar a la segunda por 8 puntos porcentuales.
# Debe mantenerse 0.6 s; después se exige retirar la mano durante 0.4 s.
confirmador = ConfirmadorLetra(segundos_estables=0.6, segundos_sin_mano=0.4)
letra_actual = None

frase = ""
# Este objeto lee una copia de la frase sin detener la captura de video.
reproductor = ReproductorVoz()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara.")

# Configurar resolución amplia a HD (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

NOMBRE_VENTANA = "Traductor de Lengua de Senas Mexicana"
cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
cv2.resizeWindow(NOMBRE_VENTANA, 1280, 720)

print("Manten la sena estable y retira la mano entre letras.")
print("Presiona '1' = salir | 'espacio' = espacio | '2' = borrar letra | 'c' = limpiar frase | 'v' = escuchar frase")

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

    # Se reinician en cada imagen para no reutilizar una letra si se pierde la mano.
    prediccion_frame = None
    candidata = None
    confianza = 0.0
    margen = 0.0
    mano_presente = bool(results.multi_hand_landmarks)

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

            # Consulta el puntaje de cada clase en el Random Forest.
            # Estos puntajes orientan el filtro; no garantizan que la letra sea correcta.
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

    letra_actual = prediccion_frame
    # Solo devuelve una letra cuando lleva suficiente tiempo siendo aceptada.
    # Devuelve None si hay incertidumbre, falta la mano o falta retirarla.
    nueva_letra = confirmador.actualizar(prediccion_frame, mano_presente, monotonic())
    if nueva_letra is not None:
        frase += nueva_letra
        print(f"+ '{nueva_letra}' | Frase: {frase}")

    # PANEL SUPERIOR E INFERIOR ESTILIZADO CON OPACIDAD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)           # Panel superior
    cv2.rectangle(overlay, (0, h - 80), (w, h), (15, 15, 15), -1)        # Panel inferior
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Texto en el panel superior (Detección en vivo)
    if letra_actual is not None:
        # Barra de progreso visual para confirmar la letra
        progreso = confirmador.progreso
        cv2.rectangle(frame, (20, 52), (20 + int(progreso * 200), 58), (0, 255, 0), -1)
        
        cv2.putText(frame, f"Sena: {letra_actual} ({confianza:.0%})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2)
    else:
        if mano_presente:
            # Mostrar la candidata rechazada ayuda a identificar confusiones del modelo.
            estado = f"Candidata: {candidata} ({confianza:.0%}) - incierta"
        else:
            estado = "Sin mano detectada: muestra la mano completa"
        cv2.putText(frame, estado, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 255), 2)
        if mano_presente and not confirmador.esperando_retiro:
            cv2.putText(frame,
                        f"Minimo: {CONFIANZA_MINIMA:.0%} | Margen: {margen:.0%}/{MARGEN_MINIMO:.0%}",
                        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)

    if confirmador.esperando_retiro:
        cv2.putText(frame, "Retira la mano antes de la siguiente letra", (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    # Texto en el panel inferior (Frase construida)
    cv2.putText(frame, f"Frase: {frase}", (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Guía de atajos
    cv2.putText(frame, "[1] Salir | [2] Borrar | [Espacio] Espacio | [C] Limpiar | [V] Escuchar frase",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

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
    elif key in (ord('v'), ord('V')):
        # La lectura comienza por petición del usuario, al terminar de formar la frase.
        reproductor.reproducir(frase)

cap.release()
cv2.destroyAllWindows()
print(f"\nFrase final: {frase}")
