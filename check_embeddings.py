import torch
import os

folder = "data/processed/combined_embeddings/"

for f in os.listdir(folder):
    if f.endswith(".pt"):
        path = os.path.join(folder, f)
        emb = torch.load(path)
        print(f"File: {f}")
        print(f"  Shape: {emb.shape}")
        print(f"  First 10 elements: {emb[:10]}\n")
        print(emb)