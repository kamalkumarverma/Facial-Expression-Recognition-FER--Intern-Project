import os
import subprocess
from tqdm import tqdm

# Path to OpenFace executable (bat file or binary)
OPENFACE_PATH = "run_openface.bat"  # Or "openface\\FeatureExtraction.exe"

INPUT_DIR = "data/raw"
OUTPUT_DIR = "openface_output"
FINAL_CSV_DIR = "data/processed/au_csv"

def run_openface_on_image(image_path, output_subdir):
    cmd = [
        OPENFACE_PATH,
        "-f", image_path,
        "-out_dir", output_subdir,
        "-aus", "-2Dfp", "-tracked"
    ]
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    os.makedirs(FINAL_CSV_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png"))]

    for file in tqdm(image_files, desc="🔍 Processing images"):
        input_path = os.path.join(INPUT_DIR, file)
        output_subdir = os.path.join(OUTPUT_DIR, file.split('.')[0])
        os.makedirs(output_subdir, exist_ok=True)

        success = run_openface_on_image(input_path, output_subdir)
        if success:
            # Find the generated CSV
            csv_path = os.path.join(output_subdir, file.replace(".jpg", ".csv"))
            if os.path.exists(csv_path):
                dst_csv = os.path.join(FINAL_CSV_DIR, file.replace(".jpg", ".csv"))
                os.rename(csv_path, dst_csv)
        else:
            print(f"❌ Failed: {file}")

if __name__ == "__main__":
    main()
