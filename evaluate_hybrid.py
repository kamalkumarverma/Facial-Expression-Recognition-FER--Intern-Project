import os
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
embed_dir = "data/processed/combined_embeddings"
label_csv = "data/processed/train_label.csv"
model_path = "checkpoints/hybrid_classifier.pt"

# Load label CSV
df_labels = pd.read_csv(label_csv)

# Create mapping: filename → label
label_dict = dict(zip(df_labels['file_name'], df_labels['label']))

# Find all embedding files
embed_files = sorted([f for f in os.listdir(embed_dir) if f.endswith(".pt")])

# Detect embedding dimension
sample_embed = torch.load(os.path.join(embed_dir, embed_files[0]))
embed_dim = sample_embed.shape[0]
print(f"Detected embedding dimension: {embed_dim}")

# Define model using the same class as training
class HybridClassifier(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(embed_dim, 256)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 7)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = HybridClassifier(embed_dim)
model.load_state_dict(torch.load(model_path))
model.eval()
print("✅ Model loaded successfully.")

all_preds = []
all_true = []

for embed_file in embed_files:
    embed_path = os.path.join(embed_dir, embed_file)
    embed = torch.load(embed_path).unsqueeze(0)

    with torch.no_grad():
        logits = model(embed)
        pred = torch.argmax(logits, dim=1).item()

    img_name = embed_file.replace(".pt", ".jpg")

    if img_name in label_dict:
        true_label = int(label_dict[img_name]) - 1  # 🔥 Correctly shift to 0–6
    else:
        print(f"⚠️ No label found for: {img_name}, skipping...")
        continue

    all_preds.append(pred)
    all_true.append(true_label)

# Report
print("\nClassification Report:")
print(classification_report(all_true, all_preds, zero_division=0))

# Confusion matrix
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
