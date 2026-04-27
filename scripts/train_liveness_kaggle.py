import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from loguru import logger

# Configuración
DATA_PATH = r"C:\Users\sofia\.cache\kagglehub\datasets\trainingdatapro\real-vs-fake-anti-spoofing-video-classification\versions\1"
MODEL_PATH = "models/liveness_kaggle.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LivenessDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        return self.transform(frame), torch.tensor(label, dtype=torch.float32)

def extract_multiple_frames(video_path, num_frames=3):
    """Extrae múltiples frames para aumentar el dataset."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extraer frames en segundos 3, 5, 7
    target_seconds = [3, 5, 7][:num_frames]
    frames = []
    
    for second in target_seconds:
        frame_number = int(second * fps)
        if frame_number < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    
    cap.release()
    return frames

def prepare_data():
    """Extrae múltiples frames y prepara datos."""
    frames = []
    labels = []
    
    # Videos reales (label=1)
    real_path = os.path.join(DATA_PATH, "train", "real_video")
    real_videos = [v for v in os.listdir(real_path) if v.endswith('.mp4')]
    
    logger.info(f"Procesando {len(real_videos)} videos reales...")
    for video in real_videos:
        video_frames = extract_multiple_frames(os.path.join(real_path, video), num_frames=3)
        for frame in video_frames:
            frames.append(frame)
            labels.append(1)
    
    # Videos de ataque (label=0)
    attack_path = os.path.join(DATA_PATH, "train", "attack")
    attack_videos = [v for v in os.listdir(attack_path) if v.endswith('.mp4')]
    
    logger.info(f"Procesando {len(attack_videos)} videos de ataque...")
    for video in attack_videos:
        video_frames = extract_multiple_frames(os.path.join(attack_path, video), num_frames=3)
        for frame in video_frames:
            frames.append(frame)
            labels.append(0)
    
    logger.info(f"Frames extraídos: {len(frames)} (reales: {sum(labels)}, ataques: {len(labels)-sum(labels)})")
    
    # Dividir datos 80/20
    train_frames, test_frames, train_labels, test_labels = train_test_split(
        frames, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return train_frames, test_frames, train_labels, test_labels

def create_model():
    """Crear modelo DenseNet201."""
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    
    # Congelar capas base
    for param in model.parameters():
        param.requires_grad = False
    
    # Nueva capa de clasificación binaria
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, 2),   # clase 0 = spoof, clase 1 = live
    )
    
    # Solo entrenar la última capa
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model.to(DEVICE)

def train():
    """Entrenar el modelo."""
    logger.info("Iniciando entrenamiento de Liveness con dataset Kaggle...")
    
    # Preparar datos
    train_frames, test_frames, train_labels, test_labels = prepare_data()
    
    train_dataset = LivenessDataset(train_frames, train_labels)
    test_dataset = LivenessDataset(test_frames, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Crear modelo
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Learning rate más bajo
    
    # Entrenamiento
    for epoch in range(15): 
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for frames, labels in train_loader:
            frames = frames.to(DEVICE)
            labels = labels.long().to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # Validación
        model.eval()
        val_correct = 0
        val_total = 0
        all_train_preds = []
        all_train_labels = []
        all_val_preds = []
        all_val_labels = []
        
        # Recolectar predicciones de train para F1
        with torch.no_grad():
            for frames, labels in train_loader:
                frames = frames.to(DEVICE)
                labels = labels.long().to(DEVICE)
                outputs = model(frames)
                predicted = outputs.argmax(dim=1)
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
        
        # Recolectar predicciones de val para F1
        with torch.no_grad():
            for frames, labels in test_loader:
                frames = frames.to(DEVICE)
                labels = labels.long().to(DEVICE)
                outputs = model(frames)
                predicted = outputs.argmax(dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        train_acc = correct / total
        val_acc = val_correct / val_total
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        
        logger.info(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
    
    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Modelo guardado en {MODEL_PATH}")
    
    return model

if __name__ == "__main__":
    train()
