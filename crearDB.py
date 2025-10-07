import os
import torch
import numpy as np
import pickle
# Importamos la arquitectura de la red siamesa que definimos en el entrenamiento
from entrenarRedSiamesa import SiameseLSTM, EMBEDDING_DIM, LSTM_HIDDEN_DIM, NUM_LSTM_LAYERS

# --- CONFIGURACIÓN ---
KEYPOINTS_DIR = "./keypoints/train"  # Usamos los datos de entrenamiento como referencia
MODEL_PATH = "./best_siamese_model.pth"
DATABASE_PATH = "./reference_embeddings.pkl"
# ---------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando el dispositivo: {device}")

def create_reference_embeddings():
    # Cargar el modelo siamés entrenado
    print("Cargando el modelo siamés entrenado...")
    model = SiameseLSTM(input_dim=17*2, hidden_dim=LSTM_HIDDEN_DIM, embedding_dim=EMBEDDING_DIM, num_layers=NUM_LSTM_LAYERS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Poner el modelo en modo de evaluación

    embedding_database = {}

    print(f"Procesando clips del directorio: {KEYPOINTS_DIR}")
    for person_id in os.listdir(KEYPOINTS_DIR):
        person_path = os.path.join(KEYPOINTS_DIR, person_id)
        if not os.path.isdir(person_path):
            continue

        person_embeddings = []
        clip_files = [f for f in os.listdir(person_path) if f.endswith('.npy')]
        
        if not clip_files:
            continue

        print(f"  Generando embedding para: {person_id}")
        for clip_file in clip_files:
            # Cargar los puntos clave del clip
            sequence = np.load(os.path.join(person_path, clip_file))
            sequence_tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(device) # Añadir dimensión de batch

            with torch.no_grad():
                embedding = model(sequence_tensor).cpu().numpy()
                person_embeddings.append(embedding)

        # Calcular el embedding promedio para la persona para hacerlo más robusto
        if person_embeddings:
            mean_embedding = np.mean(person_embeddings, axis=0)
            embedding_database[person_id] = mean_embedding

    # Guardar la base de datos de embeddings en un archivo
    with open(DATABASE_PATH, 'wb') as f:
        pickle.dump(embedding_database, f)

    print(f"\n✅ Base de datos de embeddings creada y guardada en '{DATABASE_PATH}'")
    print(f"Total de identidades conocidas: {len(embedding_database)}")

if __name__ == "__main__":
    create_reference_embeddings()