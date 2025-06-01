import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(17, 128)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 256)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(256, 128)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(128, 2)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        return self.output(x)

# model = NeuralNetwork()
# print(model)