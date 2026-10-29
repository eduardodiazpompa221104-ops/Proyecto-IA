import cv2
import mediapipe as mp
import numpy as np
import os

# Configuración 
RUTA_DATOS = "datos_senas"
ARCHIVO_DATASET = os.path.join(RUTA_DATOS, "dataset.npz")
os.makedirs(RUTA_DATOS, exist_ok=True)

# MediaPipe 
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

# Cargar datos previos de forma segura liberando el archivo
X, y = [], []
if os.path.exists(ARCHIVO_DATASET):
    with np.load(ARCHIVO_DATASET, allow_pickle=False) as data:
        X = list(data["X"])
        y = list(data["y"].astype(str))
    print(f"Dataset previo cargado: {len(X)} muestras.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara. Revisa permisos o conexion.")

# Configurar resolución de captura
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

NOMBRE_VENTANA = "Captura de Datos - LSM"
cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
cv2.resizeWindow(NOMBRE_VENTANA, 1280, 720)

print("Presiona la tecla de la letra que haras con la mano.")
print("'1' = salir y guardar | '2' = deshacer muestra | '3' = BORRAR TODO EL DATASET\n")

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

    # Panel superior
    overlay = frame.copy()
    panel_alto = 130
    cv2.rectangle(overlay, (0, 0), (w, panel_alto), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Estado de la mano
    if landmarks_actuales is not None:
        cv2.putText(frame, "Mano detectada", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Sin mano", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Total de muestras capturadas
    cv2.putText(frame, f"Total muestras: {len(X)}", (300, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Conteo por clase
    conteo = {}
    for etiqueta in y:
        conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
    
    items = [f"{k}:{v}" for k, v in sorted(conteo.items())]
    
    max_items_por_fila = 13
    y_offset = 70
    for i in range(0, len(items), max_items_por_fila):
        linea_texto = "  ".join(items[i:i + max_items_por_fila])
        cv2.putText(frame, linea_texto, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        y_offset += 25

    # Barra inferior con las instrucciones
    cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, " [1] Guardar y Salir  |  [2] Deshacer ultima  |  [3] BORRAR TODO  |  A-Z para capturar ",
                (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow(NOMBRE_VENTANA, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        break
    elif key == ord('2') and X:
        X.pop()
        y.pop()
        print(f"Última muestra eliminada. Total: {len(X)}")
    elif key == ord('3'):
        X.clear()
        y.clear()
        if os.path.exists(ARCHIVO_DATASET):
            try:
                os.remove(ARCHIVO_DATASET)
                print("¡TODO EL DATASET HA SIDO BORRADO DEL DISCO Y MEMORIA!")
            except PermissionError:
                print("Memoria limpiada. El archivo del disco se sobrescribirá al salir.")
        else:
            print("Memoria limpiada.")
    elif key in MAPEO_TECLAS and landmarks_actuales is not None:
        etiqueta = MAPEO_TECLAS[key]
        X.append(landmarks_actuales)
        y.append(etiqueta)
        print(f"Guardado '{etiqueta}' | Total: {len(X)}")

cap.release()
cv2.destroyAllWindows()

# Guardar dataset si hay muestras
if X:
    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.asarray(y, dtype="U1")
    np.savez_compressed(ARCHIVO_DATASET, X=X_arr, y=y_arr)
    print(f"\nDataset guardado: {ARCHIVO_DATASET} ({len(X)} muestras)")
else:
    print("\nNo hay datos para guardar.")