import os
import numpy as np
import cv2
import onnxruntime   as ort

# --- --- --- CONFIGURACIÓN --- --- ---
INPUT_DIR = "./dataset/train"
OUTPUT_DIR = "./keypoints/train"
ONNX_MODEL_PATH = './yolo11m-pose.onnx'

# Dimensiones de entrada para el modelo ONNX (de YOLOv8)
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
# --- --- ------------------- --- --- ---

def preprocess_frame(frame):
    """Prepara un fotograma para la inferencia con el modelo ONNX."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    
    # Normalizar y cambiar dimensiones HWC -> NCHW
    img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
    img_tensor = img_tensor.transpose(0, 3, 1, 2)
    return img_tensor

def postprocess_output(output, frame_shape):
    """Extrae los puntos clave de la salida del modelo ONNX."""
    # La salida de YOLO-Pose es [1, 56, N] -> la transponemos
    predictions = output[0][0].T
    
    if len(predictions) == 0:
        return None

    # Filtrar detecciones con baja confianza
    predictions = predictions[predictions[:, 4] > 0.5]
    if len(predictions) == 0:
        return None
        
    # Tomar la detección con la confianza más alta
    best_detection = predictions[np.argmax(predictions[:, 4])]
    
    # Extraer puntos clave (17 kpts * 3 (x, y, conf))
    keypoints_raw = best_detection[5:].reshape((17, 3))

    # Re-escalar los keypoints a las dimensiones originales del frame
    frame_h, frame_w = frame_shape
    scale_x, scale_y = frame_w / INPUT_WIDTH, frame_h / INPUT_HEIGHT
    
    keypoints_rescaled = np.zeros((17, 2))
    keypoints_rescaled[:, 0] = keypoints_raw[:, 0] * scale_x
    keypoints_rescaled[:, 1] = keypoints_raw[:, 1] * scale_y

    return keypoints_rescaled


def main():
    print("--- Extracción de Puntos Clave Optimizada con ONNX ---")

    # Cargar la sesión de inferencia de ONNX
    # Cargar la sesión de inferencia de ONNX
    print(f"Cargando modelo ONNX desde: {ONNX_MODEL_PATH}")

    # MODIFICACIÓN: Se especifica CUDA como el único proveedor
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider'])

    # VERIFICACIÓN: Imprimir el proveedor que se está utilizando
    print(f"✅ Usando el proveedor de ONNX: {ort_session.get_providers()[0]}")

    # Iterar sobre todos los videos del conjunto de datos
    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            if not filename.endswith(".mp4"):
                continue

            video_path = os.path.join(root, filename)
            print(f"\n🎥 Procesando video: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error al abrir el video: {video_path}")
                continue

            video_keypoints = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Pre-procesar el fotograma
                input_tensor = preprocess_frame(frame)
                
                # 2. Ejecutar inferencia
                ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
                ort_outs = ort_session.run(None, ort_inputs)

                # 3. Post-procesar la salida para obtener keypoints
                keypoints = postprocess_output(ort_outs, frame.shape[:2])

                if keypoints is not None:
                    # Aquí no normalizamos, guardamos las coordenadas absolutas
                    # ya que el modelo LSTM puede aprender de ellas.
                    # Si se quisiera normalizar, se haría aquí.
                    video_keypoints.append(keypoints)
                else:
                    # Añadir ceros si no se detecta persona
                    video_keypoints.append(np.zeros((17, 2)))

            cap.release()

            if not video_keypoints:
                print(f"⚠️ No se detectaron puntos clave en el video: {video_path}")
                continue

            sequence_data = np.array(video_keypoints)

            # Guardar en formato .npy
            relative_path = os.path.relpath(video_path, INPUT_DIR)
            output_path_without_ext = os.path.join(OUTPUT_DIR, os.path.splitext(relative_path)[0])
            os.makedirs(os.path.dirname(output_path_without_ext), exist_ok=True)
            output_npy_path = output_path_without_ext + ".npy"
            np.save(output_npy_path, sequence_data)
            
            print(f"✅ Puntos clave guardados en: {output_npy_path} con forma {sequence_data.shape}")

    print("\n--- Proceso finalizado ---")

if __name__ == "__main__":
    main()