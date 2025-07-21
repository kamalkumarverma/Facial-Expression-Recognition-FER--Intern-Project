import os
import torch
import pandas as pd
import numpy as np

# ---------------------
# Paths
# ---------------------
vit_dir = "data/processed/vit_embeddings"
au_dir = "data/processed/au_features"
gcn_dir = "data/processed/gcn_embeddings"
out_dir = "data/processed/combined_embeddings"
os.makedirs(out_dir, exist_ok=True)

# ---------------------
# List files
# ---------------------
vit_files = sorted([f for f in os.listdir(vit_dir) if f.endswith(".npy")])
gcn_files = sorted([f for f in os.listdir(gcn_dir) if f.endswith(".pt")])

# ---------------------
# Loop over ViT files
# ---------------------
for vit_file in vit_files:
    base_name = os.path.splitext(vit_file)[0]  # e.g., train_00001_aligned

    # ---------------------
    # Load ViT
    # ---------------------
    vit_path = os.path.join(vit_dir, vit_file)
    vit_emb = np.load(vit_path)
    vit_tensor = torch.tensor(vit_emb, dtype=torch.float32)

    # ---------------------
    # Load AU
    # ---------------------
    try:
        img_num = int(base_name.split("_")[1])
    except:
        print(f"⚠️ Could not parse number from {base_name}, skipping...")
        continue

    au_file = f"frame_{img_num}_features.csv"
    au_path = os.path.join(au_dir, au_file)
    if not os.path.exists(au_path):
        print(f"⚠️ AU file not found for: {base_name}")
        continue

    df_au = pd.read_csv(au_path)
    au_values = df_au.values.flatten()
    au_tensor = torch.tensor(au_values, dtype=torch.float32)

    # ---------------------
    # Load GCN
    # ---------------------
    gcn_file = f"{base_name}.pt"
    gcn_path = os.path.join(gcn_dir, gcn_file)
    if not os.path.exists(gcn_path):
        print(f"⚠️ GCN embedding not found for: {base_name}")
        continue

    gcn_tensor = torch.load(gcn_path)

    # ---------------------
    # Concatenate
    # ---------------------
    combined_tensor = torch.cat([vit_tensor, gcn_tensor, au_tensor], dim=0)

    # Save
    save_path = os.path.join(out_dir, f"{base_name}.pt")
    torch.save(combined_tensor, save_path)
    print(f"✅ Combined saved: {base_name}.pt")

print("🎉 All embeddings combined successfully (ViT + GCN + AU)!")
