import cv2
import mediapipe as mp
import numpy as np
import os

# Configuración 
RUTA_DATOS = "datos_senas"
ARCHIVO_DATASET = os.path.join(RUTA_DATOS, "dataset.npz")
os.makedirs(RUTA_DATOS, exist_ok=True)

#  MediaPipe 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Mapeo teclas -> letras LSM 
MAPEO_TECLAS = {ord(c): c.upper() for c in "abcdefghijklmnopqrstuvwxyz"}

# Cargar datos previos si existen 
if os.path.exists(ARCHIVO_DATASET):
    data = np.load(ARCHIVO_DATASET, allow_pickle=False)
    X = list(data["X"])
    y = list(data["y"].astype(str))
    print(f"Dataset previo cargado: {len(X)} muestras.")
else:
    X, y = [], []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara. Revisa permisos o conexion.")

print("Presiona la tecla de la letra que haras con la mano.")
print("'q' = salir y guardar | 'u' = deshacer ultima muestra")
print("Recomendado: ~40 muestras por letra en distintas posiciones e iluminacion.\n")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    landmarks_actuales = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )
            landmarks_actuales = []
            for lm in hand_landmarks.landmark:
                landmarks_actuales.extend([lm.x, lm.y, lm.z])
            landmarks_actuales = np.asarray(landmarks_actuales, dtype=np.float32)

        cv2.putText(frame, "Mano detectada", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Sin mano", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Contador por clase
    conteo = {}
    for etiqueta in y:
        conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
    texto_conteo = " ".join(f"{k}:{v}" for k, v in sorted(conteo.items()))
    cv2.putText(frame, f"Total: {len(X)} | {texto_conteo}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("Captura de Datos - LSM (Py 3.13)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('u') and X:
        X.pop(); y.pop()
        print(f"Deshecho. Total: {len(X)}")
    elif key in MAPEO_TECLAS and landmarks_actuales is not None:
        etiqueta = MAPEO_TECLAS[key]
        X.append(landmarks_actuales)
        y.append(etiqueta)
        print(f"Guardado '{etiqueta}' | Total: {len(X)}")

cap.release()
cv2.destroyAllWindows()

# Guardar en formato seguro .npz 
if X:
    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.asarray(y, dtype="U1")
    np.savez_compressed(ARCHIVO_DATASET, X=X_arr, y=y_arr)
    print(f"\nDataset guardado: {ARCHIVO_DATASET} ({len(X)} muestras)")
else:
    print("\nNo hay datos para guardar.")