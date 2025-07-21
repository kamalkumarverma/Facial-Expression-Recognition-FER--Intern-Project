import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

au_dir = "data/processed/au_csv"
heatmap_dir = "data/processed/heatmaps"
os.makedirs(heatmap_dir, exist_ok=True)

files = sorted([f for f in os.listdir(au_dir) if f.endswith(".csv")])

for file in tqdm(files, desc="Generating heatmaps"):
    df = pd.read_csv(os.path.join(au_dir, file))
    
    # Only regression columns
    au_columns = [col for col in df.columns if col.endswith("_r")]
    au_values = df[au_columns].iloc[0].values

    heatmap = np.expand_dims(au_values, axis=0)

    plt.figure(figsize=(10, 1))
    plt.imshow(heatmap, cmap='viridis', aspect='auto')
    plt.axis('off')

    base_name = os.path.splitext(file)[0]
    plt.savefig(os.path.join(heatmap_dir, f"{base_name}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

print("✅ All AU heatmaps generated and saved!")
