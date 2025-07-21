import os
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import ViTModel, ViTImageProcessor

# Directories
heatmaps_dir = "data/processed/heatmaps"
output_dir = "data/processed/vit_embeddings"
os.makedirs(output_dir, exist_ok=True)

# Load ViT model and processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
model.eval()

# Transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Process each heatmap
files = sorted([f for f in os.listdir(heatmaps_dir) if f.endswith(".png")])

print("Extracting ViT embeddings:")
for file in tqdm(files):
    img_path = os.path.join(heatmaps_dir, file)
    image = Image.open(img_path).convert("RGB")
    
    # Use processor to prepare
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # Use [CLS] token
    
    # Save embedding
    file_name = os.path.splitext(file)[0] + ".npy"
    np.save(os.path.join(output_dir, file_name), embedding)

print("✅ ViT embeddings extracted and saved!")
