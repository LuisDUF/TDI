import os
import numpy as np
from ultralytics import YOLO
import torch

# --- --- --- CONFIGURACIÓN --- --- ---
# Directorio donde están los videos grabados.
INPUT_DIR = "./dataset/train"
# Directorio donde se guardarán los puntos clave extraídos.
OUTPUT_DIR = "./keypoints/train"
# Modelo de YOLO a utilizar. 'yolov11n-pose.pt' es el más pequeño y rápido.
MODEL_NAME = 'yolov8n-pose.pt' 
# --- --- ------------------- --- --- ---

def main():
    """
    Función principal para procesar los videos y extraer los puntos clave de la pose.
    """
    print("--- Programa de Extracción de Puntos Clave con YOLOv8 ---")

    # 1. Cargar el modelo de pose pre-entrenado
    # Elige el dispositivo (GPU si está disponible, si no, CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando el dispositivo: {device}")
    model = YOLO(MODEL_NAME)
    model.to(device)

    # 2. Iterar sobre todos los videos del conjunto de datos
    print(f"Buscando videos en: {INPUT_DIR}")
    
    # os.walk es perfecto para recorrer estructuras de directorios anidadas
    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            if filename.endswith(".mp4"):
                video_path = os.path.join(root, filename)
                print(f"\n🎥 Procesando video: {video_path}")

                # 3. Ejecutar model.predict() en cada video
                # 'stream=True' es crucial para videos largos, ya que procesa fotograma por fotograma
                # sin agotar la memoria RAM.
                results = model(video_path, stream=True, verbose=False)

                video_keypoints = []
                
                # 4. Extraer, normalizar y guardar las secuencias de puntos clave
                for result in results:
                    # Los keypoints ya vienen en un formato accesible
                    keypoints = result.keypoints
                    
                    if keypoints is not None and len(keypoints.xyn) > 0:
                        # 'keypoints.xyn' contiene las coordenadas (x, y) ya NORMALIZADAS
                        # por el ancho y alto del fotograma. ¡Esto nos ahorra trabajo!
                        # Nos quedamos con la detección de la primera persona encontrada.
                        normalized_kpts = keypoints.xyn[0].cpu().numpy()
                        video_keypoints.append(normalized_kpts)
                    else:
                        # Si no se detecta ninguna persona en el fotograma,
                        # añadimos un array de ceros para mantener la consistencia de la secuencia.
                        # COCO tiene 17 puntos clave, cada uno con 2 coordenadas (x, y).
                        video_keypoints.append(np.zeros((17, 2)))

                if not video_keypoints:
                    print(f"⚠️ No se detectaron puntos clave en el video: {video_path}")
                    continue

                # Convierte la lista de arrays en un único array de NumPy
                # La forma final será (num_fotogramas, 17, 2)
                sequence_data = np.array(video_keypoints)

                # 5. Guardar en formato .npy
                # Creamos una estructura de directorios paralela para los archivos .npy
                relative_path = os.path.relpath(video_path, INPUT_DIR)
                output_path_without_ext = os.path.join(OUTPUT_DIR, os.path.splitext(relative_path)[0])
                
                # Asegurarse de que el directorio de salida exista
                os.makedirs(os.path.dirname(output_path_without_ext), exist_ok=True)
                
                output_npy_path = output_path_without_ext + ".npy"
                np.save(output_npy_path, sequence_data)
                
                print(f"✅ Puntos clave guardados en: {output_npy_path} con forma {sequence_data.shape}")

    print("\n--- Proceso finalizado ---")


if __name__ == "__main__":
    main()