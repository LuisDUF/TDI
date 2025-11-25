import cv2
import numpy as np
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import braycurtis  # O la métrica que estés usando
import onnxruntime as ort

from entrenarRedSiamesa import canonicalize_direction, normalize_sequence
from collections import deque

# Importamos la arquitectura de la red siamesa
from entrenarRedSiamesa import SiameseLSTM, EMBEDDING_DIM, LSTM_HIDDEN_DIM, NUM_LSTM_LAYERS, INPUT_DIM

# --- --- --- CONFIGURACIÓN Y CONSTANTES --- --- ---
ONNX_MODEL_PATH = './yolo11m-pose.onnx'
SIAMESE_MODEL_PATH = './best_siamese_model.pth'
DATABASE_PATH = './reference_embeddings.pkl'

# Parámetros
SEQUENCE_LENGTH = 60        # Ventana para las predicciones continuas
REID_THRESHOLD = 0.6        # Ajusta esto según tus pruebas
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

current_label = "Recopilando Frames: "

# NUEVO: keypoints de cuerpo (COCO: sin cara: 5–16)
BODY_KPT_INDICES = list(range(5, 17))   # 12 puntos
NUM_BODY_KPTS = len(BODY_KPT_INDICES)

# <<< NUEVO: conexiones del esqueleto en el orden COCO de 17 puntos
# Índices COCO: 
# 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
# 5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow,
# 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip,
# 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
BODY_SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),         # brazo izquierdo
    (6, 8), (8, 10),        # brazo derecho
    (5, 6),                 # hombros
    (5, 11), (6, 12),       # hombro-cadera
    (11, 12),               # caderas
    (11, 13), (13, 15),     # pierna izquierda
    (12, 14), (14, 16),     # pierna derecha
]

# --- --- CONFIGURACIÓN DE LA PRUEBA --- --- ---
GROUND_TRUTH_ID = 'Cesar'  # ¿Quién es la persona real en el video?
VIDEO_PATH = ".//CesarPrueba.mp4"
# ------------------------------------------

def normalize_frame_kpts(frame_kpts):
    """
    frame_kpts: (12,2) en píxeles.
    Devuelve (12,2) normalizado.
    """
    kpts = frame_kpts.copy().astype(np.float32)

    if np.all(kpts == 0):
        return kpts

    xs = kpts[:, 0]
    ys = kpts[:, 1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    if max_x == min_x and max_y == min_y:
        return kpts

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    scale = max(max_x - min_x, max_y - min_y)
    if scale < 1e-6:
        return kpts

    kpts[:, 0] = (kpts[:, 0] - center_x) / scale
    kpts[:, 1] = (kpts[:, 1] - center_y) / scale

    return kpts


# Configuración de Estilo para Gráficas
sns.set_theme(style="whitegrid")

# Cargar recursos (Modelo, DB, ONNX)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Cargando modelos...")
siamese_model = SiameseLSTM(
    input_dim=INPUT_DIM,              # <-- 12*2, consistente con el entrenamiento
    hidden_dim=LSTM_HIDDEN_DIM,
    embedding_dim=EMBEDDING_DIM,
    num_layers=NUM_LSTM_LAYERS
).to(device)
siamese_model.load_state_dict(torch.load(SIAMESE_MODEL_PATH, map_location=device))
siamese_model.eval()

with open(DATABASE_PATH, 'rb') as f:
    embedding_database = pickle.load(f)

reference_ids = list(embedding_database.keys())
reference_embeddings = np.array(list(embedding_database.values())).squeeze(axis=1)

ort_session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Variables para el reporte
history_data = []  # Predicciones por ventana (streaming)

# Buffer para ventanas + lista para video completo
keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)
video_keypoints = []   # (frame_idx, kpts_norm) para el embedding global

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

print(f"--- Iniciando Análisis de Video: {VIDEO_PATH} ---")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # --- Procesamiento YOLO (ONNX) ---
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
    img_tensor = img_tensor.transpose(0, 3, 1, 2)

    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    output = ort_outs[0][0].T

    person_detected = False
    if(frame_count<=60):
        current_label = "Recopilando frames: "+str(frame_count)



    # <<< NUEVO: variables para dibujar el esqueleto
    keypoints_rescaled_full = None
    keypoints_raw_full = None

    if len(output) > 0:
        best_detection_idx = np.argmax(output[:, 4])
        detection = output[best_detection_idx]
        if detection[4] > 0.5:
            person_detected = True

            # Caja (por si la quieres dibujar, no la usamos para normalizar)
            box = detection[:4]
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            # Extraer TODOS los 17 keypoints (COCO)
            keypoints_raw = detection[5:].reshape((17, 3))
            keypoints_raw_full = keypoints_raw  # <<< NUEVO: guardar completo para visibilidad

            # Re-escalar keypoints al tamaño original del frame
            frame_h, frame_w = frame.shape[:2]
            scale_x = frame_w / INPUT_WIDTH
            scale_y = frame_h / INPUT_HEIGHT

            # <<< NUEVO: re-escalar los 17 puntos para dibujar esqueleto
            keypoints_rescaled_full = np.zeros((17, 2), dtype=np.float32)
            keypoints_rescaled_full[:, 0] = keypoints_raw[:, 0] * scale_x
            keypoints_rescaled_full[:, 1] = keypoints_raw[:, 1] * scale_y

            # Quedarnos solo con keypoints de cuerpo (sin cara) para la red
            body_kpts_raw = keypoints_raw[BODY_KPT_INDICES]  # (12, 3)

            kpts_body_rescaled = np.zeros((NUM_BODY_KPTS, 2), dtype=np.float32)
            kpts_body_rescaled[:, 0] = body_kpts_raw[:, 0] * scale_x
            kpts_body_rescaled[:, 1] = body_kpts_raw[:, 1] * scale_y

            kpts_body_norm = normalize_frame_kpts(kpts_body_rescaled)

            # Guardamos para la ventana
            keypoints_buffer.append(kpts_body_norm)

            # Guardamos para el video completo
            video_keypoints.append((frame_count, kpts_body_norm))

    if not person_detected:
        keypoints_buffer.clear()
        current_label = "No Detectado"

    # --- Inferencia Red Siamesa (PREDICCIÓN CONTINUA POR VENTANAS) ---
    if len(keypoints_buffer) == SEQUENCE_LENGTH:
        sequence_np = np.array(keypoints_buffer)   # (T, 12, 2)
        sequence_np = canonicalize_direction(sequence_np)
        sequence_np = normalize_sequence(sequence_np)
        sequence_tensor = torch.from_numpy(sequence_np).float().unsqueeze(0).to(device)

        with torch.no_grad():
            live_embedding = siamese_model(sequence_tensor).cpu().numpy().squeeze()

        # Distancias contra la base de datos
        distances = [braycurtis(live_embedding, ref_emb) for ref_emb in reference_embeddings]
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        print("Ventana:", frame_count, "->", reference_ids[min_dist_idx], min_dist)
        print("Más lejano:", reference_ids[np.argmax(distances)], distances[np.argmax(distances)])

        # Lógica de Clasificación
        predicted_id = reference_ids[min_dist_idx]
        if min_dist < REID_THRESHOLD:
            final_label = predicted_id
        else:
            final_label = "Desconocido"

        current_label = final_label

        # REGISTRO DE DATOS PARA EL REPORTE (por ventana)
        is_correct = (final_label == GROUND_TRUTH_ID)

        history_data.append({
            'Frame': frame_count,   # Usamos el frame actual como referencia de la ventana
            'Distancia': min_dist,
            'Predicción': final_label,
            'Correcto': 'Si' if is_correct else 'No',
            'Ground_Truth': GROUND_TRUTH_ID
        })

        keypoints_buffer.clear()

    # <<< NUEVO: Dibujar esqueleto si tenemos keypoints
    # <<< NUEVO: Dibujar SOLO keypoints del cuerpo
    if keypoints_rescaled_full is not None and keypoints_raw_full is not None:

    # Líneas del cuerpo
        for i, j in BODY_SKELETON_CONNECTIONS:
            if keypoints_raw_full[i, 2] > 0.2 and keypoints_raw_full[j, 2] > 0.2:
                x1, y1 = keypoints_rescaled_full[i]
                x2, y2 = keypoints_rescaled_full[j]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 255, 255), 2)

    # Puntos del cuerpo
        for idx in BODY_KPT_INDICES:
            conf = keypoints_raw_full[idx, 2]
            if conf > 0.2:
                x, y = keypoints_rescaled_full[idx]
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Visualización en tiempo real
    color = (0, 255, 0) if current_label == GROUND_TRUTH_ID else (0, 0, 255)
    if(current_label!=""):
        cv2.putText(frame, f"{current_label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Analisis en Curso', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- --- PREDICCIÓN GLOBAL CON TODO EL VIDEO COMPLETO --- --- ---

if not video_keypoints:
    print("No se generaron suficientes datos (no se detectó persona) para el reporte.")
    exit()

# Construimos la secuencia larga con todos los frames donde hubo detección
frame_indices_global = [fk[0] for fk in video_keypoints]
kpts_list_global = [fk[1] for fk in video_keypoints]

sequence_np_global = np.stack(kpts_list_global, axis=0).astype(np.float32)
sequence_np_global = canonicalize_direction(sequence_np_global)
sequence_np_global = normalize_sequence(sequence_np_global)

sequence_tensor_global = torch.from_numpy(sequence_np_global).float().unsqueeze(0).to(device)

with torch.no_grad():
    live_embedding_global = siamese_model(sequence_tensor_global).cpu().numpy().squeeze()

distances_global = [braycurtis(live_embedding_global, ref_emb) for ref_emb in reference_embeddings]
min_dist_idx_global = np.argmin(distances_global)
min_dist_global = distances_global[min_dist_idx_global]

predicted_id_global = reference_ids[min_dist_idx_global]
if min_dist_global < REID_THRESHOLD:
    final_label_global = predicted_id_global
else:
    final_label_global = "Desconocido"

print("\n====== RESULTADO GLOBAL DEL VIDEO ======")
print("ID más cercano:", predicted_id_global)
print("Distancia mínima:", min_dist_global)
print("ID más lejano:", reference_ids[np.argmax(distances_global)],
      "Distancia máx:", distances_global[np.argmax(distances_global)])
print("Etiqueta final del video:", final_label_global)
print("========================================\n")

# --- --- --- GENERACIÓN DEL REPORTE GRÁFICO (VENTANAS) --- --- ---

if not history_data:
    print("No se generaron suficientes datos de ventanas para el reporte.")
    exit()

df = pd.DataFrame(history_data)
df = df.sort_values('Frame')

# Calcular métricas finales (por ventanas)
total_preds = len(df)
aciertos = len(df[df['Correcto'] == 'Si'])
precision = (aciertos / total_preds) * 100 if total_preds > 0 else 0

print(f"Generando gráficas... Precisión por ventanas: {precision:.2f}%")

# Dashboard
fig = plt.figure(figsize=(14, 8))
fig.suptitle(
    f'Reporte de resultados: {GROUND_TRUTH_ID}\n'
    f'Precisión: {precision:.2f}%',
    fontsize=16
)
gs = fig.add_gridspec(2, 2)

# 1. Distancia vs Tiempo (ventanas)
ax1 = fig.add_subplot(gs[0, :])
sns.lineplot(
    data=df, x='Frame', y='Distancia', marker='o',
    hue='Correcto', palette={'Si': 'green', 'No': 'red'}, ax=ax1
)
ax1.axhline(REID_THRESHOLD, color='blue', linestyle='--', label=f'Umbral ({REID_THRESHOLD})')
ax1.set_title('Evolución de la Distancia')
ax1.set_ylabel('Distancia')
ax1.legend()

# 2. Distribución de Predicciones (ventanas)
ax2 = fig.add_subplot(gs[1, 0])
conteo_preds = df['Predicción'].value_counts().reset_index()
conteo_preds.columns = ['Identidad', 'Cantidad']
sns.barplot(data=conteo_preds, x='Identidad', y='Cantidad', palette='viridis', ax=ax2)
ax2.set_title('Distribución de Identidades Predichas')
ax2.set_ylabel('Número de veces detectado')

# 3. Pastel de Aciertos vs Errores (ventanas)
ax3 = fig.add_subplot(gs[1, 1])
counts = df['Correcto'].value_counts()
ax3.pie(counts, labels=counts.index, autopct='%1.1f%%',
        colors=['#66b3ff', '#ff9999'], startangle=90)
ax3.set_title('Porcentaje de Aciertos')

plt.tight_layout()
filename = "reporte_validacion_ventanas_y_global.png"
plt.savefig(filename, dpi=300)
print(f"✅ Reporte guardado exitosamente como: {filename}")
plt.show()
