import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, 
                 layers,
                 n_inputs,
                 n_classes,
                 device="cpu"):
        
        super().__init__()
        
        self.layers = []
        for nodes in layers:
            self.layers.append(nn.Linear(n_inputs, nodes))
            self.layers.append(nn.ReLU())
            n_inputs = nodes
        self.layers.append(nn.Linear(n_inputs, n_classes))
        self.model_stack = nn.Sequential(*self.layers)
        self.device = device

    def forward(self, x):
        
        return self.model_stack(x)
