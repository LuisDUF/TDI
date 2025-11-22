import cv2
import numpy as np
import torch
import pickle
from scipy.spatial.distance import euclidean, cosine, minkowski, chebyshev, mahalanobis, braycurtis
import onnxruntime as ort
from collections import deque

# Importamos la arquitectura de la red siamesa
from entrenarRedSiamesa import SiameseLSTM, EMBEDDING_DIM, LSTM_HIDDEN_DIM, NUM_LSTM_LAYERS

# --- --- --- CONFIGURACIÓN Y CONSTANTES --- --- ---
# Modelo de Pose (ONNX)
ONNX_MODEL_PATH = './yolo11m-pose.onnx'
# Modelo de Re-ID (PyTorch)
SIAMESE_MODEL_PATH = './best_siamese_model.pth'
# Base de datos de embeddings
DATABASE_PATH = './reference_embeddings.pkl'

# Parámetros de la aplicación
SEQUENCE_LENGTH = 15  # Número de fotogramas para acumular antes de la inferencia. ¡Debe ser manejable!
REID_THRESHOLD = 10  # Umbral de distancia. ¡Necesitarás ajustarlo!

# Dimensiones de entrada para el modelo ONNX (de YOLO11)
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
# --- --- ------------------------------------ --- --- ---

# Cargar dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Cargar el modelo siamés entrenado
siamese_model = SiameseLSTM(input_dim=17*2, hidden_dim=LSTM_HIDDEN_DIM, embedding_dim=EMBEDDING_DIM, num_layers=NUM_LSTM_LAYERS).to(device)
siamese_model.load_state_dict(torch.load(SIAMESE_MODEL_PATH, map_location=device))
siamese_model.eval()

# 2. Cargar la base de datos de embeddings de referencia
with open(DATABASE_PATH, 'rb') as f:
    embedding_database = pickle.load(f)
reference_ids = list(embedding_database.keys())
reference_embeddings = np.array(list(embedding_database.values())).squeeze(axis=1)

# 3. Inicializar la sesión de inferencia de ONNX
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider']) # O ['CUDAExecutionProvider']

# 4. Inicializar buffer de secuencias
keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)
current_label = "Procesando..."

# Iniciar captura de video
cap = cv2.VideoCapture(".//dataset//train//cesar//cesar2.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # a. Pre-procesar el fotograma para el modelo ONNX
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
    img_tensor = img_tensor.transpose(0, 3, 1, 2)  # HWC a NCHW

    # b. Ejecutar el modelo de pose ONNX
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # c. Post-procesar la salida para obtener puntos clave
    # La salida de YOLO-Pose es [1, 56, N] donde 56 = 17 kpts * 3 (x,y,conf) + 5 (box, conf)
    output = ort_outs[0][0].T 
    
    person_detected = False
    if len(output) > 0:
        # Tomamos la detección con la confianza más alta
        best_detection_idx = np.argmax(output[:, 4])
        detection = output[best_detection_idx]
        
        box = detection[:4]
        conf = detection[4]
        
        if conf > 0.5:
            person_detected = True
            keypoints_raw = detection[5:].reshape((17, 3))
            
            # Normalizar los puntos clave por las dimensiones de la caja
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            keypoints_normalized = np.zeros((17, 2))
            keypoints_normalized[:, 0] = (keypoints_raw[:, 0] - x1) / w
            keypoints_normalized[:, 1] = (keypoints_raw[:, 1] - y1) / h
            
            # d. Añadir al búfer
            keypoints_buffer.append(keypoints_normalized)
            
            # Dibujar la caja y puntos clave en el frame original
            frame_h, frame_w, _ = frame.shape
            scale_x, scale_y = frame_w / INPUT_WIDTH, frame_h / INPUT_HEIGHT
            #cv2.rectangle(frame, (int(x1*scale_x), int(y1*scale_y)), (int(x2*scale_x), int(y2*scale_y)), (0, 255, 0), 2)

    if not person_detected:
        keypoints_buffer.clear() # Limpiar buffer si se pierde la persona
        current_label = "Desconocido"
        
    # e. Si el búfer está lleno, ejecutar Re-ID
    if len(keypoints_buffer) == SEQUENCE_LENGTH:
        sequence_tensor = torch.from_numpy(np.array(keypoints_buffer)).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            live_embedding = siamese_model(sequence_tensor).cpu().numpy().squeeze()

        # f. Calcular distancias y encontrar el más cercano
        distances = [braycurtis(live_embedding, ref_emb) for ref_emb in reference_embeddings]
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        # g. Asignar identidad o "Desconocido"
        if min_dist < REID_THRESHOLD:
            current_label = reference_ids[min_dist_idx]
        else:
            current_label = "Desconocido"
        # Limpiar el búfer para empezar a acumular una nueva secuencia
        keypoints_buffer.clear()

    # Mostrar la etiqueta en la pantalla
    cv2.putText(frame, f"ID: {current_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow('Real-Time Re-ID', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()