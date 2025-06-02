import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(14, 32)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # 30% dropout

        self.hidden2 = nn.Linear(32, 64)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.output = nn.Linear(64, 2)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.act1(self.hidden1(x)))
        x = self.dropout2(self.act2(self.hidden2(x)))
        return self.act_output(self.output(x))

# model = NeuralNetwork()
# print(model)