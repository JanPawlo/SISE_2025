import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(2, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.hidden2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.output = nn.Linear(64, 2)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.output(x)
        x = self.act_output(x)
        return x
