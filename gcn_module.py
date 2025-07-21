import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm

# Simple 2-layer GCN
class LandmarkGCN(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=64):
        super(LandmarkGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Global mean pooling
        out = torch.mean(x, dim=0)
        return out

def main():
    input_dir = "data/processed/graphs/"
    output_dir = "data/processed/gcn_embeddings/"
    os.makedirs(output_dir, exist_ok=True)

    model = LandmarkGCN()
    model.eval()

    for file in tqdm(os.listdir(input_dir)):
        if file.endswith(".pt"):
            graph_path = os.path.join(input_dir, file)
            graph = torch.load(graph_path, weights_only=False)  # <-- FIX here
            embedding = model(graph)
            save_path = os.path.join(output_dir, file.replace(".pt", "_emb.pt"))
            torch.save(embedding, save_path)

    print("✅ All GCN embeddings saved!")

if __name__ == "__main__":
    main()
