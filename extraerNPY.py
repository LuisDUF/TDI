import os
import numpy as np
import cv2
import onnxruntime as ort

# --- --- --- CONFIGURACIÓN --- --- ---
INPUT_DIR = "./dataset/train"
OUTPUT_DIR = "./keypoints/train"
ONNX_MODEL_PATH = './yolo11m-pose.onnx'

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

NUM_FRAMES = 80  # <<--- Número de fotogramas a extraer

BODY_KPT_INDICES = list(range(5, 17))   # 12 keypoints de cuerpo
NUM_BODY_KPTS = len(BODY_KPT_INDICES)
# --- --- ------------------- --- --- ---

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
    img_tensor = img_tensor.transpose(0, 3, 1, 2)
    return img_tensor

def postprocess_output(output, frame_shape):
    predictions = output[0][0].T
    if len(predictions) == 0:
        return None

    predictions = predictions[predictions[:, 4] > 0.5]
    if len(predictions) == 0:
        return None
        
    best_detection = predictions[np.argmax(predictions[:, 4])]
    keypoints_raw = best_detection[5:].reshape((17, 3))

    body_kpts_raw = keypoints_raw[BODY_KPT_INDICES]

    frame_h, frame_w = frame_shape
    scale_x, scale_y = frame_w / INPUT_WIDTH, frame_h / INPUT_HEIGHT
    
    keypoints_rescaled = np.zeros((NUM_BODY_KPTS, 2), dtype=np.float32)
    keypoints_rescaled[:, 0] = body_kpts_raw[:, 0] * scale_x
    keypoints_rescaled[:, 1] = body_kpts_raw[:, 1] * scale_y

    return keypoints_rescaled

def get_frame_indices(total_frames, num_frames):
    if total_frames <= num_frames:
        return list(range(total_frames))
    step = total_frames / num_frames
    return [int(i * step) for i in range(num_frames)]

def main():
    print("--- Extracción de Puntos Clave con Frame Limitado ---")

    print(f"Cargando modelo ONNX desde: {ONNX_MODEL_PATH}")
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider'])
    print(f"✅ Usando el proveedor de ONNX: {ort_session.get_providers()[0]}")

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

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = get_frame_indices(total_frames, NUM_FRAMES)

            video_keypoints = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if not ret:
                    video_keypoints.append(np.zeros((NUM_BODY_KPTS, 2), dtype=np.float32))
                    continue

                input_tensor = preprocess_frame(frame)
                ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
                ort_outs = ort_session.run(None, ort_inputs)

                keypoints = postprocess_output(ort_outs, frame.shape[:2])
                if keypoints is None:
                    keypoints = np.zeros((NUM_BODY_KPTS, 2), dtype=np.float32)

                video_keypoints.append(keypoints)

            cap.release()

            sequence_data = np.array(video_keypoints)

            relative_path = os.path.relpath(video_path, INPUT_DIR)
            output_path_without_ext = os.path.join(OUTPUT_DIR, os.path.splitext(relative_path)[0])
            os.makedirs(os.path.dirname(output_path_without_ext), exist_ok=True)
            output_npy_path = output_path_without_ext + ".npy"
            np.save(output_npy_path, sequence_data)
            
            print(f"✅ Guardado: {output_npy_path} con forma {sequence_data.shape}")

    print("\n--- Proceso finalizado ---")

if __name__ == "__main__":
    main()
