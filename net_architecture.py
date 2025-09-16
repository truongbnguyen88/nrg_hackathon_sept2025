import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture for Phase 1 modeling
class Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        
        x = self.gelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x