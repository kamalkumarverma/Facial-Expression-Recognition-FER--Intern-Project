import pandas as pd

# Paths
label_csv_path = "data/processed/train_label.csv"

# Read CSV
df = pd.read_csv(label_csv_path)

# Update file_name column
df["file_name"] = df["file_name"].str.replace(".jpg", "_aligned.csv", regex=False)

# Save back
df.to_csv(label_csv_path, index=False)

print("✅ Updated train_label.csv: filenames now end with _aligned.csv")
