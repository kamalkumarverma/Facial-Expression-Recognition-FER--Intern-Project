import os
import pandas as pd

input_csv = "openface_output/training_img.csv"
output_dir = "data/processed/landmarks_csv"
os.makedirs(output_dir, exist_ok=True)

# Read CSV
df = pd.read_csv(input_csv)

# Remove possible spaces from columns
df.columns = df.columns.str.strip()

# Use 'frame' column if it exists, otherwise fallback to index
use_frame_column = "frame" in df.columns

for idx, row in df.iterrows():
    single_df = pd.DataFrame([row])

    if use_frame_column:
        frame_num = row["frame"]
        filename = f"frame_{int(frame_num)}.csv"
    else:
        filename = f"frame_{idx+1}.csv"

    single_df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"✅ Saved: {filename}")

print("🎉 All landmark CSVs created!")
