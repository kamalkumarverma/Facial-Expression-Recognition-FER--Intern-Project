import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import ViTModel, ViTFeatureExtractor
import os
from tqdm import tqdm

# ViT extractor class
class ViTExtractor(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224'):
        super(ViTExtractor, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.vit = ViTModel.from_pretrained(model_name)

    def forward(self, img_path):
        image = Image.open(img_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.vit(**inputs)
        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0]
        return cls_embedding

if __name__ == "__main__":
    vit_model = ViTExtractor()

    # Directory where your heatmaps are stored
    heatmaps_dir = "data/processed/heatmaps/"
    output_dir = "data/processed/vit_embeddings/"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all PNG heatmaps
    for filename in tqdm(os.listdir(heatmaps_dir)):
        if filename.endswith(".png"):
            img_path = os.path.join(heatmaps_dir, filename)
            embedding = vit_model(img_path)

            # Save embedding as .pt file
            name = os.path.splitext(filename)[0]
            torch.save(embedding, os.path.join(output_dir, f"{name}_vit.pt"))

    print("✅ All embeddings saved!")
