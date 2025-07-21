import os
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

# --- Paths ---
LANDMARKS_DIR = "data/processed/landmarks_csv"
OUTPUT_DIR = "data/processed/graphs"
LABELS_CSV = "data/processed/train_label.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_graph(landmark_path):
    # Load landmark CSV
    df = pd.read_csv(landmark_path)
    if df.empty:
        print(f"⚠️ Skipping empty file: {landmark_path}")
        return None

    # Flatten the row to one tensor
    landmark_tensor = torch.tensor(df.iloc[0].values, dtype=torch.float32)

    # Number of landmarks = total columns // 2 (assuming x,y only)
    num_points = landmark_tensor.shape[0] // 2

    # Node features: reshape to (num_points, 2)
    node_features = landmark_tensor.view(num_points, 2)

    # Fully connected adjacency matrix (example)
    adj = torch.ones((num_points, num_points)) - torch.eye(num_points)

    # Convert adjacency to edge_index
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()

    # Create PyG Data object
    data = Data(x=node_features, edge_index=edge_index)

    return data

# --- Read labels CSV ---
df_labels = pd.read_csv(LABELS_CSV)

# --- Build graphs for each image ---
for _, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Building graphs"):
    file_name = row["file_name"]
    base_name = os.path.splitext(file_name)[0]
    landmark_file = f"{base_name}.csv"
    landmark_path = os.path.join(LANDMARKS_DIR, landmark_file)

    if os.path.exists(landmark_path):
        data = build_graph(landmark_path)
        if data is not None:
            out_file = os.path.join(OUTPUT_DIR, f"{base_name}.pt")
            torch.save(data, out_file)
    else:
        print(f"❌ Landmark file not found: {landmark_file}")

print("✅ All graphs generated and saved!")
