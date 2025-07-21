import os
import pandas as pd
from tqdm import tqdm

# --- Paths ---
INPUT_CSV = "openface_output/training_img.csv"
OUTPUT_DIR_LANDMARKS = "data/processed/landmarks_csv/"
OUTPUT_DIR_AU = "data/processed/au_csv/"

# Create output directories if not exist
os.makedirs(OUTPUT_DIR_LANDMARKS, exist_ok=True)
os.makedirs(OUTPUT_DIR_AU, exist_ok=True)

def extract_landmarks(df):
    landmark_cols = [col for col in df.columns if col.startswith('x_') or col.startswith('y_')]
    return df[landmark_cols]

def extract_aus(df):
    au_cols = [col for col in df.columns if (' AU' in col and ('r' in col or 'c' in col))]
    return df[au_cols]

def main():
    df = pd.read_csv(INPUT_CSV)
    # Separate by frame or by image (usually OpenFace outputs frame column or image name)
    images = df['frame'].unique() if 'frame' in df.columns else [0]
    
    for img_name in tqdm(images, desc="Processing"):
        if 'frame' in df.columns:
            sub_df = df[df['frame'] == img_name]
            name = f"frame_{img_name}.csv"
        else:
            sub_df = df
            name = "training_img.csv"

        landmarks_df = extract_landmarks(sub_df)
        aus_df = extract_aus(sub_df)

        landmarks_df.to_csv(os.path.join(OUTPUT_DIR_LANDMARKS, name), index=False)
        aus_df.to_csv(os.path.join(OUTPUT_DIR_AU, name), index=False)

    print(f"✅ Landmark CSVs saved to: {OUTPUT_DIR_LANDMARKS}")
    print(f"✅ AU CSVs saved to: {OUTPUT_DIR_AU}")

if __name__ == "__main__":
    main()
