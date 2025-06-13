import torch
import torch.nn as nn


class densenet1d(nn.Module):
    def __init__(self, 
                 layers=[2048, 4086, 2048, 1024, 512, 256, 128, 32],
                 n_inputs=512,
                 n_classes=10,
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
