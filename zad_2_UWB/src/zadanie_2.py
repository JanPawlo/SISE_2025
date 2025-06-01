import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from neural_network import NeuralNetwork
from load_data import load_all_training_data, load_all_test_data


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

def euclidean_error(a, b):
    return np.linalg.norm(a - b, axis=1)

def plot_ecdf(data, label):
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, ecdf, label=label)


def main():
    # UWAGA SA WCZYTYWANE PO 4 WPISY Z PLIKU W OBECNEJ IMPLEMENTACJI
    X_train, y_train, X_scaler, Y_scaler = load_all_training_data()
    X_test, y_test, coords = load_all_test_data(X_scaler, Y_scaler, Y_scaler)
    model = train_neural_network(X_train, y_train)

    y_pred_test = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

    # Błąd sieci neuronowej
    network_error = euclidean_error(y_pred_test, y_test)

    # Błąd danych testowych
    error_sensor = euclidean_error(coords, y_test)

    plt.figure(figsize=(8, 6))
    plot_ecdf(network_error, "Sieć neuronowa")
    plot_ecdf(error_sensor, "Błąd danych testowych")

    plt.xlabel("Błąd lokalizacji")
    plt.ylabel("Dystrybuanta empiryczna (ECDF)")
    plt.title("Porównanie błędu lokalizacji")
    plt.legend()
    plt.grid(True)
    plt.show()

main()
