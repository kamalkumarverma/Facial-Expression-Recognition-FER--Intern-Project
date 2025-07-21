import os
import pandas as pd
from tqdm import tqdm

# --- Paths ---
INPUT_DIR = "data/processed/au_csv/"
OUTPUT_DIR = "data/processed/au_features/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_features(df):
    features = {}
    for col in df.columns:
        features[f"{col}_mean"] = df[col].mean()
        features[f"{col}_max"] = df[col].max()
        features[f"{col}_min"] = df[col].min()
    return features

def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

    for file in tqdm(files, desc="Extracting AU features"):
        path = os.path.join(INPUT_DIR, file)
        df = pd.read_csv(path)

        if df.empty:
            continue

        features = compute_features(df)
        features_df = pd.DataFrame([features])
        save_path = os.path.join(OUTPUT_DIR, file.replace(".csv", "_features.csv"))
        features_df.to_csv(save_path, index=False)

    print(f"✅ AU feature files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
