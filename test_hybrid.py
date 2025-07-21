import torch
from dataset import FERHybridDataset
from hybrid_classifier import HybridClassifier
import pandas as pd

embeddings_dir = "data/processed/combined_embeddings/"
labels_path = "data/processed/test_label.csv"
model_path = "models/hybrid_model.pth"

label2idx = {
    "happy": 0,
    "sad": 1,
    "angry": 2,
    "surprise": 3,
    "disgust": 4,
    "fear": 5,
    "neutral": 6
}

dataset = FERHybridDataset(embeddings_dir, labels_path, label2idx)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

model = HybridClassifier(input_dim=867, num_classes=7)
model.load_state_dict(torch.load(model_path))
model.eval()

with torch.no_grad():
    for idx, (x, y) in enumerate(loader):
        out = model(x)
        pred = torch.argmax(out, dim=1)
        print(f"Sample {idx}: Predicted = {pred.item()}, True = {y.item()}")
