import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

RUTA_DATOS = Path(__file__).resolve().parent.parent / "datos_senas"
os.makedirs(RUTA_DATOS, exist_ok=True)
ARCHIVO = os.path.join(RUTA_DATOS, "dataset_din.npz")
T = 30  # frames por secuencia

# Teclas: j, z, n (= Ñ), q, 0 (= movimiento sin seña)
MAPEO = {ord('j'): 'J', ord('z'): 'Z', ord('n'): 'Ñ', ord('q'): 'Q',
         ord('k'): 'K', ord('x'): 'X', ord('0'): '_'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)

X, y = [], []
if os.path.exists(ARCHIVO):
    with np.load(ARCHIVO, allow_pickle=False) as d:
        X = list(d["X"])
        y = list(d["y"].astype(str))
    print(f"Dataset previo: {len(X)} secuencias.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

NOMBRE_VENTANA = "Captura dinamicas - LSM"
cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
cv2.resizeWindow(NOMBRE_VENTANA, 1280, 720)

grabando = None   # etiqueta que se está grabando
buffer = []
modo_borrado = False

print("J / Z / N(=Ñ) / Q = grabar 30 frames | 0 = movimiento sin seña")
print("'1' = guardar y salir | '2' = deshacer última | '3' = borrar todo")
print("'4' = borrar todos los registros de una etiqueta | 'Esc' = cancelar borrado")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    vec = None
    if res.multi_hand_landmarks:
        hl = res.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        vec = np.asarray([c for lm in hl.landmark for c in (lm.x, lm.y, lm.z)],
                         dtype=np.float32)

    if grabando is not None:
        if vec is None:
            # Si se pierde la mano, se descarta la secuencia completa.
            print("Mano perdida, secuencia descartada.")
            grabando, buffer = None, []
        else:
            buffer.append(vec)
            etiqueta_vis = "NADA" if grabando == "_" else grabando
            cv2.putText(frame, f"GRABANDO {etiqueta_vis}: {len(buffer)}/{T}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if len(buffer) == T:
                X.append(np.stack(buffer))
                y.append(grabando)
                print(f"Guardada '{grabando}' | Total: {len(X)}")
                grabando, buffer = None, []

    if grabando is None:
        conteo = {k: y.count(k) for k in sorted(set(y))}
        texto = "  ".join(f"{('NADA' if k == '_' else k)}:{v}" for k, v in conteo.items())
        cv2.putText(frame, f"Mano: {'SI' if vec is not None else 'NO'} | Secuencias: {texto}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if modo_borrado:
        cv2.putText(frame, "MODO BORRADO: J/Z/N/Q/K/X/0 | ESC para cancelar",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, " [J][Z][N=Ñ][Q] grabar | [0] sin sena | [1] Guardar | [2] Deshacer | [3] Todo | [4] Etiqueta",
                (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow(NOMBRE_VENTANA, frame)
    key = cv2.waitKey(1) & 0xFF

    if modo_borrado:
        if key == 27:
            modo_borrado = False
            print("Borrado por etiqueta cancelado.")
        elif key in MAPEO:
            etiqueta = MAPEO[key]
            indices_conservar = [i for i, actual in enumerate(y) if actual != etiqueta]
            eliminadas = len(y) - len(indices_conservar)
            X = [X[i] for i in indices_conservar]
            y = [y[i] for i in indices_conservar]
            modo_borrado = False
            print(f"Registros de '{etiqueta}' eliminados: {eliminadas} | Total: {len(X)}")
        continue

    if key == ord('1'):
        break
    elif key == ord('2') and X:
        X.pop()
        y.pop()
        print(f"Última secuencia eliminada. Total: {len(X)}")
    elif key == ord('3'):
        X.clear()
        y.clear()
        if os.path.exists(ARCHIVO):
            try:
                os.remove(ARCHIVO)
            except PermissionError:
                pass
        print("Dataset dinámico borrado.")
    elif key == ord('4') and grabando is None:
        modo_borrado = True
        print("Modo borrado: presiona J/Z/N/Q/K/X/0 para eliminar la etiqueta o Esc para cancelar.")
    elif key in MAPEO and grabando is None and vec is not None:
        grabando = MAPEO[key]
        buffer = []

cap.release()
cv2.destroyAllWindows()

if X:
    np.savez_compressed(ARCHIVO, X=np.stack(X).astype(np.float32),
                        y=np.asarray(y, dtype="U1"))
    print(f"Guardado: {ARCHIVO} ({len(X)} secuencias)")
else:
    if os.path.exists(ARCHIVO):
        os.remove(ARCHIVO)
        print("Dataset sin secuencias; archivo anterior eliminado.")
    print("No hay datos para guardar.")