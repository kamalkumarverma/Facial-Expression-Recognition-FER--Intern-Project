import os
import subprocess
from tqdm import tqdm # type Ignore

# Path to OpenFace FeatureExtraction.exe
openface_exe = os.path.abspath("openface/OpenFace_2.2.0_win_x64/FeatureExtraction.exe")

# Folder with input images
input_folder = "data/raw"
output_folder = "openface_output"

def run_openface_on_folder():
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    print(f"🖼️ Found {len(image_files)} images in {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    for img in tqdm(image_files):
        img_path = os.path.join(input_folder, img)
        cmd = [
            openface_exe,
            "-f", img_path,
            "-out_dir", output_folder,
            "-aus",
            "-2Dfp",
            "-tracked"
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    run_openface_on_folder()
    print("✅ Finished OpenFace batch processing.")
