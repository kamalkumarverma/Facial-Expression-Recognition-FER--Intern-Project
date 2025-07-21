import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader, Data
import pandas as pd

# ----------------
# Paths
# ----------------
graph_dir = "data/processed/graphs"
# graph_dir = "data/processed/graphs_test"
label_csv = "data/processed/train_label.csv"
emb_dir = "data/processed/gcn_embeddings"
os.makedirs(emb_dir, exist_ok=True)

# ----------------
# Load labels
# ----------------
labels_df = pd.read_csv(label_csv)

# ----------------
# Dataset
# ----------------
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_dir, labels_df):
        self.graph_dir = graph_dir
        self.graph_files = sorted(os.listdir(graph_dir))
        self.labels_df = labels_df

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        g_path = os.path.join(self.graph_dir, self.graph_files[idx])
        # load with weights_only=False to avoid pickle errors
        data = torch.load(g_path, weights_only=False)

        # Make sure this is a Data object
        if not isinstance(data, Data):
            raise TypeError(f"Expected torch_geometric.data.Data but got {type(data)}")

        # Image file name to match CSV
        img_name = self.graph_files[idx].replace(".pt", ".jpg")

        # Find label
        label_row = self.labels_df[self.labels_df["file_name"] == img_name]
        if label_row.empty:
            raise ValueError(f"No label found for {img_name}")

        label = int(label_row["label"].values[0]) - 1  # Adjust if labels start at 1

        return data, torch.tensor(label, dtype=torch.long)

dataset = GraphDataset(graph_dir, labels_df)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ----------------
# GCN Model
# ----------------
class GCNModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=7):  # output_dim = num classes
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)  # Global mean pooling
        x = self.lin(x)
        return x

model = GCNModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ----------------
# Train loop
# ----------------
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    model.train()
    for data, label in loader:
        optimizer.zero_grad()
        logits = model(data[0])
        loss = criterion(logits.unsqueeze(0), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

print("✅ GCN training done. Saving embeddings...")

# ----------------
# Save embeddings per graph
# ----------------
model.eval()
with torch.no_grad():
    for i, (data, _) in enumerate(dataset):
        embedding = torch.mean(F.relu(model.conv2(F.relu(model.conv1(data.x, data.edge_index)), data.edge_index)), dim=0)
        emb_name = os.path.splitext(dataset.graph_files[i])[0] + ".pt"
        torch.save(embedding, os.path.join(emb_dir, emb_name))

print(f"✅ All GCN embeddings saved to: {emb_dir}")
