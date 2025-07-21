import torch
import os

graph_dir = "data/processed/graphs"
files = sorted(os.listdir(graph_dir))
sample_file = files[0]

data = torch.load(os.path.join(graph_dir, sample_file), weights_only=False)
print(type(data))
print(data)
