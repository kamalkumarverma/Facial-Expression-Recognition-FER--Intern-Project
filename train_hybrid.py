import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ---------------------
# Paths
# ---------------------
combined_dir = "data/processed/combined_embeddings"
label_csv = "data/processed/train_label.csv"
os.makedirs("checkpoints", exist_ok=True)

# ---------------------
# Load labels
# ---------------------
labels_df = pd.read_csv(label_csv)

# Class counts (from your data)
class_counts = [1290, 281, 717, 4772, 1982, 705, 2524]
total_samples = sum(class_counts)
class_weights = [total_samples / c for c in class_counts]
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# ---------------------
# Dataset
# ---------------------
class CombinedDataset(Dataset):
    def __init__(self, combined_dir, labels_df):
        self.combined_dir = combined_dir
        self.files = sorted(os.listdir(combined_dir))
        self.labels_df = labels_df

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f_name = self.files[idx]
        emb_path = os.path.join(self.combined_dir, f_name)
        emb = torch.load(emb_path)

        img_name = f_name.replace(".pt", ".jpg")
        label_row = self.labels_df[self.labels_df["file_name"] == img_name]

        if label_row.empty:
            raise ValueError(f"No label found for {img_name}")

        label = int(label_row["label"].values[0]) - 1  # adjust to 0-index
        return emb, label

dataset = CombinedDataset(combined_dir, labels_df)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---------------------
# Classifier Model
# ---------------------
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=7):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Detect embedding dim from first file
sample_emb = torch.load(os.path.join(combined_dir, dataset.files[0]))
input_dim = sample_emb.shape[0]
print(f"Detected embedding dimension: {input_dim}")

model = SimpleClassifier(input_dim=input_dim, num_classes=7)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Use weighted CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# ---------------------
# Train Loop
# ---------------------
epochs = 20

for epoch in range(epochs):
    total_loss = 0
    model.train()
    for embs, labels in loader:
        outputs = model(embs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save final model
torch.save(model.state_dict(), "checkpoints/hybrid_classifier.pt")
print("✅ Training complete, model saved to checkpoints/hybrid_classifier.pt")
