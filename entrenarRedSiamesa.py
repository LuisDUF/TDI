import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- --- --- CONFIGURACIÓN --- --- ---
TRAIN_DIR = "./keypoints/train"
VAL_DIR = "./keypoints/val"
EMBEDDING_DIM = 128  # Dimensión del vector de características de salida
LSTM_HIDDEN_DIM = 256 # Dimensión oculta del LSTM
NUM_LSTM_LAYERS = 2   # Número de capas LSTM
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 150
# -----------------------------------

# Determinar el dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando el dispositivo: {device}")


# 1. Implementar la arquitectura de la Red Siamesa
class SiameseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers):
        super(SiameseLSTM, self).__init__()
        # La capa LSTM procesará la secuencia de puntos clave
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        # Una capa lineal para mapear la salida del LSTM al embedding final
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x tiene la forma (batch, seq_len, 17, 2)
        # Aplanamos los puntos clave para la entrada del LSTM: (batch, seq_len, 17*2)
        batch_size, seq_len, _, _ = x.shape
        x = x.view(batch_size, seq_len, -1)
        
        # El LSTM devuelve la salida para cada paso de tiempo y el estado final (h_n, c_n)
        _, (h_n, _) = self.lstm(x)
        
        # Usamos la salida del estado oculto de la última capa
        # h_n tiene la forma (num_layers, batch, hidden_dim) -> tomamos la última
        embedding = self.relu(h_n[-1])
        
        # Pasamos por la capa final para obtener el embedding
        embedding = self.fc(embedding)
        
        # Normalizamos el embedding (práctica común para Triplet Loss)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding

# 2. Crear una clase Dataset y DataLoader personalizada
class TripletKeypointDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.person_to_clips = self._get_person_clips()
        self.persons = list(self.person_to_clips.keys())
        self.all_clips = [os.path.join(self.data_dir, p, c) for p, clips in self.person_to_clips.items() for c in clips]


    def _get_person_clips(self):
        person_to_clips = {}
        for person_id in os.listdir(self.data_dir):
            person_path = os.path.join(self.data_dir, person_id)
            if os.path.isdir(person_path):
                clips = [f for f in os.listdir(person_path) if f.endswith('.npy')]
                if clips:
                    person_to_clips[person_id] = clips
        return person_to_clips

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, index):
        # 1. Seleccionar el Ancla
        anchor_path_full = self.all_clips[index]
        anchor_person_id = anchor_path_full.split(os.sep)[-2]
        
        # 2. Seleccionar el Positivo (mismo ID, clip diferente)
        positive_clips = self.person_to_clips[anchor_person_id]
        anchor_clip_name = os.path.basename(anchor_path_full)
        
        # Si la persona solo tiene un clip, usamos el mismo como positivo
        positive_clip_name = random.choice([c for c in positive_clips if c != anchor_clip_name] or positive_clips)
        positive_path_full = os.path.join(self.data_dir, anchor_person_id, positive_clip_name)

        # 3. Seleccionar el Negativo (ID diferente)
        negative_person_id = random.choice([p for p in self.persons if p != anchor_person_id])
        negative_clip_name = random.choice(self.person_to_clips[negative_person_id])
        negative_path_full = os.path.join(self.data_dir, negative_person_id, negative_clip_name)

        # Cargar los datos .npy
        anchor = np.load(anchor_path_full)
        positive = np.load(positive_path_full)
        negative = np.load(negative_path_full)
        
        # Convertir a tensores de PyTorch
        return (torch.from_numpy(anchor).float(),
                torch.from_numpy(positive).float(),
                torch.from_numpy(negative).float())


def main():
    def pad_collate(batch):
        anchors, positives, negatives = zip(*batch)

        # Rellenar las secuencias para que todas tengan el mismo largo
        anchors = pad_sequence(anchors, batch_first=True)
        positives = pad_sequence(positives, batch_first=True)
        negatives = pad_sequence(negatives, batch_first=True)

        return anchors, positives, negatives

    # Inicializar Datasets y DataLoaders
    train_dataset = TripletKeypointDataset(TRAIN_DIR)
    val_dataset = TripletKeypointDataset(VAL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=pad_collate)

    # Inicializar modelo, función de pérdida y optimizador
    # La entrada al LSTM es 17 puntos * 2 coordenadas (x,y) = 34
    model = SiameseLSTM(input_dim=17*2, hidden_dim=LSTM_HIDDEN_DIM, embedding_dim=EMBEDDING_DIM, num_layers=NUM_LSTM_LAYERS).to(device)
    
    # 3. Implementar la función de Triplet Loss
    criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')

    # 4. Escribir el bucle de entrenamiento
    print("\n--- Iniciando Entrenamiento ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)

        # Bucle de validación
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # 5. Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_siamese_model.pth')
            print(f"✨ Nuevo mejor modelo guardado con Val Loss: {best_val_loss:.4f}")

    print("\n--- Entrenamiento Finalizado ---")
    print(f"Mejor modelo guardado en 'best_siamese_model.pth'")

if __name__ == "__main__":
    main()