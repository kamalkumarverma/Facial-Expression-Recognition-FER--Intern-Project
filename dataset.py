# src/dataset.py
import torch
import pandas as pd
import os

class FERHybridDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_dir, csv_file, label2idx):
        self.embeddings_dir = embeddings_dir
        self.df = pd.read_csv(csv_file)
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]["file_name"]
        label = self.df.iloc[idx]["label"]

        embedding = torch.load(os.path.join(self.embeddings_dir, file_name))
        y = torch.tensor(self.label2idx[label], dtype=torch.long)

        return embedding, y
