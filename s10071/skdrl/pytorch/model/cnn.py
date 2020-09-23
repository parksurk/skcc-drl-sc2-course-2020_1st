import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFC(nn.Module):
    
    def __init__(self, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(2, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1173, 256)
        self.fc2 = nn.Linear(256, self.output_dim)
        self.relu = nn.ReLU()
        self.identity = nn.Identity()

    def forward(self, xs, state):
        xs = self.conv1(xs)
        xs = self.relu(xs)
        xs = self.pool(xs)
        xs = self.conv2(xs)
        xs = self.relu(xs)
        xs = self.pool(xs)
        xs = xs.view(-1, self.num_flat_features(xs))
        #print(xs.shape)
        #print(state.shape)
        xs = torch.cat((xs, state), -1)
        xs = self.fc1(xs)
        xs = self.relu(xs)
        xs = self.fc2(xs)
        xs = self.identity(xs)
        return xs
            
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features