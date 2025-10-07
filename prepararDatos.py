import os
import random
import shutil

# --- CONFIGURACIÓN ---
KEYPOINTS_DIR = "./keypoints/train"
VAL_DIR = "./keypoints/val"
VAL_SPLIT_RATIO = 0.2
# ---------------------

print("Dividiendo los datos en conjuntos de entrenamiento y validación...")

# Eliminar el directorio de validación antiguo si existe
if os.path.exists(VAL_DIR):
    shutil.rmtree(VAL_DIR)

# Recorrer cada directorio de persona en el conjunto de entrenamiento
for person_folder in os.listdir(KEYPOINTS_DIR):
    person_path = os.path.join(KEYPOINTS_DIR, person_folder)
    if not os.path.isdir(person_path):
        continue

    # Listar todos los clips de la persona
    clips = [f for f in os.listdir(person_path) if f.endswith('.npy')]
    random.shuffle(clips)

    # Determinar cuántos clips mover a validación
    num_val_clips = int(len(clips) * VAL_SPLIT_RATIO)
    if num_val_clips == 0 and len(clips) > 1:
        num_val_clips = 1 # Mover al menos un clip si hay más de uno

    val_clips = clips[:num_val_clips]

    # Crear el directorio de destino
    val_person_path = os.path.join(VAL_DIR, person_folder)
    os.makedirs(val_person_path, exist_ok=True)

    # Mover los archivos seleccionados
    for clip_to_move in val_clips:
        src_path = os.path.join(person_path, clip_to_move)
        dst_path = os.path.join(val_person_path, clip_to_move)
        shutil.move(src_path, dst_path)
        print(f"Movido '{src_path}' a '{dst_path}'")

print("\nDivisión completada.")
print(f"Datos de entrenamiento restantes en: {KEYPOINTS_DIR}")
print(f"Datos de validación creados en: {VAL_DIR}")