import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from neural_network import NeuralNetwork
from load_data import load_all_training_data, load_all_test_data, load_data

def train_neural_network(X, y, n_steps=3):
    # Konwersja danych
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Podział na train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = NeuralNetwork(n_steps*2)

    n_epochs = 300
    batch_size = 32
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0.0

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            Xbatch = X_train[i:i + batch_size]
            ybatch = y_train[i:i + batch_size]

            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Średnia strata treningowa
        epoch_train_loss /= len(X_train) / batch_size
        train_losses.append(epoch_train_loss)

        # Walidacja
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = loss_fn(y_val_pred, y_val).item()
            val_losses.append(val_loss)

        print(f"Epoch {epoch}: train_loss = {epoch_train_loss:.6f}, val_loss = {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Wizualizacja strat
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    return model

def euclidean_error(a, b):
    return np.linalg.norm(a - b, axis=1)

def plot_ecdf(data, label):
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, ecdf, label=label)


def main():
    n_steps = 3 # liczba próbek z poprzednich chwil czasowych wykorzystywanych przez sieć neuronową - 1

    X_train, y_train, X_scaler, Y_scaler = load_all_training_data(n_steps)
    C_scaler = MinMaxScaler()
    X_test, y_test_scaled, coords = load_all_test_data(X_scaler, Y_scaler, C_scaler, n_steps)
    model = train_neural_network(X_train, y_train, n_steps)


    y_pred_test_scaled = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

    y_pred_test = Y_scaler.inverse_transform(y_pred_test_scaled)
    y_test = Y_scaler.inverse_transform(y_test_scaled)
    coords = C_scaler.inverse_transform(coords)

    # Błąd sieci neuronowej
    network_error = euclidean_error(y_pred_test, y_test)

    # Błąd danych testowych
    error_sensor = euclidean_error(coords, y_test)

    # Wykres pozycji
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c='green', label='Odczyty z czujników', alpha=0.3)
    plt.scatter(y_pred_test[:, 0], y_pred_test[:, 1], c='red', label='Przewidywania sieci neuronowej', alpha=0.3)
    plt.scatter(y_test[:, 0], y_test[:, 1], c='blue', label='Rzeczywiste pozycje', alpha=0.6)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Porównanie pozycji: rzeczywiste, przewidywane, czujniki")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # dystrybuanta bledu
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
