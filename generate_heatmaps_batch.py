import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========== SETTINGS ==========
IMG_SIZE = 96

# AU center positions on 96x96 canvas
AU_CENTERS = {
    'AU01_r': (30, 20),
    'AU02_r': (50, 20),
    'AU04_r': (40, 30),
    'AU06_r': (35, 50),
    'AU12_r': (35, 65),
    'AU15_r': (40, 75),
    'AU17_r': (40, 80)
}

# ========== FUNCTIONS ==========

def generate_heatmap(au_values):
    """Generate single heatmap from AU values"""
    heatmap = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for au, intensity in au_values.items():
        if au in AU_CENTERS and intensity > 0:
            cx, cy = AU_CENTERS[au]
            cv2.circle(heatmap, (cx, cy), 5, float(intensity), -1)

    heatmap = cv2.GaussianBlur(heatmap, (15, 15), sigmaX=0)
    return np.clip(heatmap * 10, 0, 255).astype(np.uint8)

def process_csv(csv_path, output_dir, prefix):
    """Process a single CSV file into heatmap images"""
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for i, row in df.iterrows():
        au_values = {au: row[au] for au in AU_CENTERS if au in row}
        heatmap = generate_heatmap(au_values)
        filename = os.path.join(output_dir, f"{prefix}_heatmap_{i:04d}.png")
        cv2.imwrite(filename, heatmap)

# ========== MAIN LOOP ==========

def main():
    input_dir = "data/processed/au_csv"
    output_dir = "data/processed/heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for csv_file in tqdm(csv_files, desc="Generating Heatmaps"):
        name_prefix = os.path.splitext(csv_file)[0]
        csv_path = os.path.join(input_dir, csv_file)
        process_csv(csv_path, output_dir, name_prefix)

    print("✅ All heatmaps generated successfully!")

if __name__ == "__main__":
    main()
