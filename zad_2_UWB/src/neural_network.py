import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(17, 24)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(24, 24)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(24, 2)
        # self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.output(x)
        return x

# model = NeuralNetwork()
# print(model)