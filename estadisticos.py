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
# IMPORTANTE: asumo que en entrenarRedSiamesa ahora también definiste INPUT_DIM (= 12*2)
from entrenarRedSiamesa import SiameseLSTM, EMBEDDING_DIM, LSTM_HIDDEN_DIM, NUM_LSTM_LAYERS, INPUT_DIM

# --- --- --- CONFIGURACIÓN Y CONSTANTES --- --- ---
ONNX_MODEL_PATH = './yolo11m-pose.onnx'
SIAMESE_MODEL_PATH = './best_siamese_model.pth'
DATABASE_PATH = './reference_embeddings.pkl'

# Parámetros
SEQUENCE_LENGTH = 15
REID_THRESHOLD = 0.48  # Ajusta esto según tus pruebas
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# NUEVO: keypoints de cuerpo (COCO: sin cara: 5–16)
BODY_KPT_INDICES = list(range(5, 17))   # 12 puntos
NUM_BODY_KPTS = len(BODY_KPT_INDICES)

# --- --- CONFIGURACIÓN DE LA PRUEBA --- ---
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
history_data = []  # Aquí guardaremos cada predicción para el análisis final

keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)
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
    current_label = "..."

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

            # 🔹 NUEVO: quedarnos solo con keypoints de cuerpo (sin cara)
            body_kpts_raw = keypoints_raw[BODY_KPT_INDICES]  # (12, 3)

            # 🔹 IMPORTANTE: re-escalar igual que en el script de extracción
            # Coordenadas de YOLO están en el espacio 640x640 (INPUT_WIDTH/HEIGHT)
            frame_h, frame_w = frame.shape[:2]
            scale_x = frame_w / INPUT_WIDTH
            scale_y = frame_h / INPUT_HEIGHT

            kpts_body_rescaled = np.zeros((NUM_BODY_KPTS, 2), dtype=np.float32)
            kpts_body_rescaled[:, 0] = body_kpts_raw[:, 0] * scale_x
            kpts_body_rescaled[:, 1] = body_kpts_raw[:, 1] * scale_y

            kpts_body_norm = normalize_frame_kpts(kpts_body_rescaled)

            keypoints_buffer.append(kpts_body_norm)

    if not person_detected:
        keypoints_buffer.clear()
        current_label = "No Detectado"

    # --- Inferencia Red Siamesa ---
    if len(keypoints_buffer) == SEQUENCE_LENGTH:
        sequence_np = np.array(keypoints_buffer)   # (T, 12, 2)
        sequence_np = canonicalize_direction(sequence_np)
        sequence_np = normalize_sequence(sequence_np)
        sequence_tensor = torch.from_numpy(sequence_np).float().unsqueeze(0).to(device)

        # -> (1, seq_len, 12, 2)

        with torch.no_grad():
            live_embedding = siamese_model(sequence_tensor).cpu().numpy().squeeze()

        # Distancias contra la base de datos
        distances = [braycurtis(live_embedding, ref_emb) for ref_emb in reference_embeddings]
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        print(reference_ids[min_dist_idx], min_dist)
        print(reference_ids[np.argmax(distances)], distances[np.argmax(distances)] )

        # Lógica de Clasificación
        predicted_id = reference_ids[min_dist_idx]
        if min_dist < REID_THRESHOLD:
            final_label = predicted_id
        else:
            final_label = "Desconocido"

        current_label = final_label

        # --- REGISTRO DE DATOS PARA EL REPORTE ---
        is_correct = (final_label == GROUND_TRUTH_ID)

        history_data.append({
            'Frame': frame_count,
            'Distancia': min_dist,
            'Predicción': final_label,
            'Correcto': 'Si' if is_correct else 'No',
            'Ground_Truth': GROUND_TRUTH_ID
        })

        keypoints_buffer.clear()

    # Visualización en tiempo real (simple)
    color = (0, 255, 0) if current_label == GROUND_TRUTH_ID else (0, 0, 255)
    cv2.putText(frame, f"Pred: {current_label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Analisis en Curso', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- --- --- GENERACIÓN DEL REPORTE GRÁFICO --- --- ---

if not history_data:
    print("No se generaron suficientes datos para el reporte.")
    exit()

# Crear DataFrame
df = pd.DataFrame(history_data)

# Calcular métricas finales
total_preds = len(df)
aciertos = len(df[df['Correcto'] == 'Si'])
precision = (aciertos / total_preds) * 100 if total_preds > 0 else 0

print(f"\nGenerando gráficas... Precisión calculada: {precision:.2f}%")

# Configurar el Dashboard (3 gráficas en una imagen)
fig = plt.figure(figsize=(14, 8))
fig.suptitle(
    f'Reporte de Validación Re-ID\nGround Truth: {GROUND_TRUTH_ID} | Precisión Global: {precision:.2f}%',
    fontsize=16
)
gs = fig.add_gridspec(2, 2)

# 1. Distancia vs Tiempo
ax1 = fig.add_subplot(gs[0, :])  # Ocupa todo el ancho superior
sns.lineplot(
    data=df, x='Frame', y='Distancia', marker='o',
    hue='Correcto', palette={'Si': 'green', 'No': 'red'}, ax=ax1
)
ax1.axhline(REID_THRESHOLD, color='blue', linestyle='--', label=f'Umbral ({REID_THRESHOLD})')
ax1.set_title('Evolución de la Distancia (braycurtis)')
ax1.set_ylabel('Distancia (Menor es mejor)')
ax1.legend()

# 2. Distribución de Predicciones
ax2 = fig.add_subplot(gs[1, 0])
conteo_preds = df['Predicción'].value_counts().reset_index()
conteo_preds.columns = ['Identidad', 'Cantidad']
sns.barplot(data=conteo_preds, x='Identidad', y='Cantidad', palette='viridis', ax=ax2)
ax2.set_title('Distribución de Identidades Predichas')
ax2.set_ylabel('Número de veces detectado')

# 3. Pastel de Aciertos vs Errores
ax3 = fig.add_subplot(gs[1, 1])
counts = df['Correcto'].value_counts()
ax3.pie(counts, labels=counts.index, autopct='%1.1f%%',
        colors=['#66b3ff', '#ff9999'], startangle=90)
ax3.set_title('Porcentaje de Aciertos')

plt.tight_layout()
filename = "reporte_validacion_final.png"
plt.savefig(filename, dpi=300)
print(f"✅ Reporte guardado exitosamente como: {filename}")
plt.show()
