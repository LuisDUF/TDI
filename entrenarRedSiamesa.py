import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
import numpy as np


# --- --- --- CONFIGURACIÓN --- --- ---
TRAIN_DIR = "./keypoints/train"
VAL_DIR = "./keypoints/val"
EMBEDDING_DIM = 128   # Dimensión del vector de características de salida
LSTM_HIDDEN_DIM = 128 # Dimensión oculta del LSTM
NUM_LSTM_LAYERS = 2   # Número de capas LSTM
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 50

# IMPORTANTE: ahora usamos solo cuerpo -> 12 keypoints * 2 coords (x,y)
NUM_BODY_KPTS = 12
INPUT_DIM = NUM_BODY_KPTS * 2
# -----------------------------------

import torch
import torch.nn.functional as F

def pairwise_distance(embeddings):
    """
    embeddings: (B, D)
    devuelve matriz de distancias euclidianas (B, B)
    """
    # usaremos distancias al cuadrado y luego sqrt si hace falta
    dot_product = embeddings @ embeddings.t()              # (B, B)
    sq_norm = torch.diag(dot_product)                     # (B,)
    dist_sq = sq_norm.unsqueeze(0) - 2*dot_product + sq_norm.unsqueeze(1)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    dist = torch.sqrt(dist_sq + 1e-8)                     # para estabilidad
    return dist

def batch_hard_triplet_loss(embeddings, labels, margin=0.5):
    """
    embeddings: (B, D)
    labels: (B,)
    """
    device = embeddings.device
    B = embeddings.size(0)

    # matriz de distancias (B, B)
    dist_mat = pairwise_distance(embeddings)  # (B, B)

    # máscara de positivos y negativos
    labels = labels.view(-1, 1)              # (B, 1)
    same_label = (labels == labels.t())      # (B, B) bool
    diag = torch.eye(B, dtype=torch.bool, device=device)
    positive_mask = same_label & ~diag       # mismo label, no yo mismo
    negative_mask = ~same_label              # distinto label

    # para cada anchor, hardest positive: máximo dist_ap entre positivos
    dist_ap = torch.zeros(B, device=device)
    for i in range(B):
        pos_indices = positive_mask[i].nonzero(as_tuple=False).view(-1)
        if len(pos_indices) > 0:
            dist_ap[i] = dist_mat[i, pos_indices].max()
        else:
            dist_ap[i] = 0.0  # no tiene positivo en batch, no contribuye

    # para cada anchor, hardest negative: mínimo dist_an entre negativos
    dist_an = torch.zeros(B, device=device)
    for i in range(B):
        neg_indices = negative_mask[i].nonzero(as_tuple=False).view(-1)
        if len(neg_indices) > 0:
            dist_an[i] = dist_mat[i, neg_indices].min()
        else:
            dist_an[i] = 0.0  # raro, pero por seguridad

    # triplet loss por anchor
    losses = F.relu(dist_ap - dist_an + margin)  # (B,)
    # ignoramos anchors sin positivos válidos (dist_ap=0 y dist_an=0 → loss=margin, podrías ajustarlo si quieres)
    return losses.mean()


# Determinar el dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando el dispositivo: {device}")

class PKBatchSampler(Sampler):
    """
    Batch sampler tipo P×K:
      - En cada batch hay P identidades distintas
      - Para cada identidad hay K clips
    Resultado: batch_size = P * K
    """
    def __init__(self, labels, p, k):
        """
        labels: lista (o array) de etiquetas por sample (longitud = len(dataset))
        p: número de identidades por batch
        k: número de ejemplos por identidad
        """
        self.labels = np.array(labels)
        self.p = p
        self.k = k

        # Construimos un diccionario: label -> [indices]
        self.label_to_indices = {}
        for idx, lab in enumerate(self.labels):
            self.label_to_indices.setdefault(int(lab), []).append(idx)

        # Lista de identidades disponibles
        self.labels_set = list(self.label_to_indices.keys())

        # Si hay menos identidades que p, ajustamos p
        if len(self.labels_set) < self.p:
            print(f"⚠️ Solo hay {len(self.labels_set)} identidades, ajustando P de {self.p} a {len(self.labels_set)}")
            self.p = len(self.labels_set)

        # Tamaño de batch
        self.batch_size = self.p * self.k

    def __iter__(self):
        # En cada epoch:
        # barajamos identidades y vamos tomando grupos de P
        labels_shuffled = self.labels_set.copy()
        random.shuffle(labels_shuffled)

        batch = []
        i = 0
        while i + self.p <= len(labels_shuffled):
            selected_labels = labels_shuffled[i:i + self.p]
            i += self.p

            for lab in selected_labels:
                idxs = self.label_to_indices[lab]
                # si hay menos de k samples para esa persona, sampleamos con reemplazo
                if len(idxs) >= self.k:
                    chosen = random.sample(idxs, self.k)
                else:
                    chosen = np.random.choice(idxs, self.k, replace=True).tolist()
                batch.extend(chosen)

            # Devolvemos un batch completo
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # descartamos batch incompleto al final (si lo hubiera)

    def __len__(self):
        # número de batches por epoch según identidades y P
        return max(1, len(self.labels_set) // self.p)



# Índices de las caderas dentro de tus 12 keypoints de cuerpo
HIP_LEFT_IDX = 6
HIP_RIGHT_IDX = 7

def canonicalize_direction(sequence, motion_thresh=1e-3):
    """
    sequence: np.array (T, 12, 2) con coords en píxeles (o reescaladas), SIN normalizar aún.
    Devuelve una secuencia donde, si la persona iba mayormente hacia la izquierda,
    se espeja en X para que parezca que va hacia la derecha.
    """
    seq = sequence.copy().astype(np.float32)
    T, K, _ = seq.shape

    # Extraemos solo las caderas para estimar el movimiento global en X
    hips = seq[:, [HIP_LEFT_IDX, HIP_RIGHT_IDX], :]  # (T, 2, 2)

    # Si hay frames completamente en cero, los ignoramos
    valid_mask = ~(np.all(hips == 0, axis=(1, 2)))  # (T,)
    if valid_mask.sum() < 2:
        # No hay suficiente info para estimar dirección
        return seq

    hips_valid = hips[valid_mask]  # (T_valid, 2, 2)
    # Centro de caderas en X por frame
    center_x = hips_valid[:, :, 0].mean(axis=1)  # (T_valid,)

    # Desplazamiento total aproximado en X
    dx = center_x[-1] - center_x[0]

    # Si casi no se movió, no tocamos nada
    if abs(dx) < motion_thresh:
        return seq

    # Convención: queremos que "hacia la derecha" sea el estándar
    # Si dx < 0 => se movió hacia la izquierda → espejamos en X
    if dx < 0:
        seq[..., 0] *= -1.0  # reflejo horizontal simple

    return seq

def normalize_sequence(sequence):
    """
    sequence: (T, 12, 2) en píxeles (o ya canonicalizada).
    Normaliza frame a frame:
      - centra el cuerpo
      - escala por el tamaño del cuerpo
    """
    seq_norm = sequence.copy().astype(np.float32)
    T, K, _ = seq_norm.shape

    for t in range(T):
        frame_kpts = seq_norm[t]

        if np.all(frame_kpts == 0):
            continue

        xs = frame_kpts[:, 0]
        ys = frame_kpts[:, 1]

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        if max_x == min_x and max_y == min_y:
            continue

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        scale = max(max_x - min_x, max_y - min_y)
        if scale < 1e-6:
            continue

        frame_kpts[:, 0] = (frame_kpts[:, 0] - center_x) / scale
        frame_kpts[:, 1] = (frame_kpts[:, 1] - center_y) / scale

        seq_norm[t] = frame_kpts

    return seq_norm

# 1. Implementar la arquitectura de la Red Siamesa
class SiameseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers):
        super(SiameseLSTM, self).__init__()
        # La capa LSTM procesará la secuencia de puntos clave
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                    batch_first=True, dropout=0.0)

        # Una capa lineal para mapear la salida del LSTM al embedding final
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x tiene la forma (batch, seq_len, NUM_BODY_KPTS, 2)
        # Aplanamos los puntos clave para la entrada del LSTM: (batch, seq_len, NUM_BODY_KPTS*2)
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

class ClipDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []           # lista de (path, label_idx)
        self.label_to_person = []   # índice -> nombre de carpeta persona
        self.labels = []            # etiquetas paralelas a samples
        self._build_index()

    def _build_index(self):
        persons = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])
        self.person_to_label = {p: i for i, p in enumerate(persons)}
        self.label_to_person = persons

        for person in persons:
            person_path = os.path.join(self.data_dir, person)
            for fname in os.listdir(person_path):
                if fname.endswith(".npy"):
                    full_path = os.path.join(person_path, fname)
                    label = self.person_to_label[person]
                    self.samples.append((full_path, label))
                    self.labels.append(label)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path)  # (T, 12, 2)

        seq = canonicalize_direction(seq)
        seq = normalize_sequence(seq)

        seq_tensor = torch.from_numpy(seq).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return seq_tensor, label_tensor

def main():
    def pad_collate(batch):
        sequences, labels = zip(*batch)  # listas de tensores

        sequences_padded = pad_sequence(sequences, batch_first=True)  # (B, Tmax, 12, 2)
        labels_tensor = torch.stack(labels, dim=0)                    # (B,)

        return sequences_padded, labels_tensor

    # Inicializar Datasets
    train_dataset = ClipDataset(TRAIN_DIR)
    val_dataset   = ClipDataset(VAL_DIR)

    # Elegimos P y K de forma que P*K = BATCH_SIZE
    P = 4   # personas por batch
    K = BATCH_SIZE // P  # clips por persona

    print(f"P={P}, K={K}, batch_size={P*K}")

    train_batch_sampler = PKBatchSampler(train_dataset.labels, p=P, k=K)
    val_batch_sampler   = PKBatchSampler(val_dataset.labels,   p=P, k=K)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        collate_fn=pad_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=0,
        collate_fn=pad_collate
    )

    model = SiameseLSTM(
        input_dim=INPUT_DIM,              # 12*2
        hidden_dim=LSTM_HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LSTM_LAYERS
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    margin = 0.3

    best_val_loss = float('inf')

    print("\n--- Iniciando Entrenamiento (batch-hard Triplet) ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for sequences, labels in train_loader:
            sequences = sequences.to(device)      # (B, T, 12, 2)
            labels = labels.to(device)            # (B,)

            optimizer.zero_grad()
            embeddings = model(sequences)         # (B, 128)

            loss = batch_hard_triplet_loss(embeddings, labels, margin=margin)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # --- Validación ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                embeddings = model(sequences)
                loss = batch_hard_triplet_loss(embeddings, labels, margin=margin)
                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_siamese_model.pth')
            print(f"✨ Nuevo mejor modelo guardado con Val Loss: {best_val_loss:.4f}")

    print("\n--- Entrenamiento Finalizado ---")
if __name__ == "__main__":
    main()
