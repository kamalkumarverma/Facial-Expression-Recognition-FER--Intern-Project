# src/hybrid_classifier.py
import torch
import torch.nn as nn

class HybridClassifier(nn.Module):
    def __init__(self, input_dim=867, num_classes=7):  # adjust num_classes
        super(HybridClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
