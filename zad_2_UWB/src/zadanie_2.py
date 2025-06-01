import torch
import torch.optim as optim
import torch.nn as nn


from neural_network import NeuralNetwork
from load_data import load_all_training_data



def train_neural_network(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = NeuralNetwork()

    n_epochs = 100
    batch_size = 5

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i + batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i + batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

    return model

# UWAGA SA WCZYTYWANE PO 4 WPISY Z PLIKU W OBECNEJ IMPLEMENTACJI
X, y = load_all_training_data()
model = train_neural_network(X, y)
