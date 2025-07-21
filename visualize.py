# src/visualize.py
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV output
df = pd.read_csv("openface_output/test.csv")

# Filter only AU intensity columns
au_columns = [col for col in df.columns if 'AU' in col and '_r' in col]
au_data = df[au_columns].iloc[0]  # Get first frame (or image)

# Plot
plt.figure(figsize=(12, 5))
au_data.plot(kind='bar', color='skyblue')
plt.title("Action Unit Intensities for test.jpg")
plt.ylabel("Intensity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
