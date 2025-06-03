import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath: str):
    df = pd.read_excel(filepath)

    features = ["data__coordinates__x", "data__coordinates__y"]
    labels = ["reference__x", "reference__y"]
    coords = ["data__coordinates__x", "data__coordinates__y"]

    df = df.dropna(axis=1, how='all')
    df = df.dropna(subset=features + labels + ['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Plik {filepath} zawiera tylko puste dane po przefiltrowaniu.")

    X = df[features].values
    Y = df[labels].values
    C = df[coords].values

    return X, Y, C


def create_sequences(X, Y, C, n_steps):
    X_seq, Y_seq, C_seq = [], [], []
    for i in range(n_steps, len(X)):
        X_window = X[i - n_steps:i].flatten()
        X_seq.append(X_window)
        Y_seq.append(Y[i])
        C_seq.append(C[i])
    return np.array(X_seq), np.array(Y_seq), np.array(C_seq)


def load_all_training_data(n_steps=3):
    folders = ["../pomiary/F8", "../pomiary/F10"]
    pattern = "*_stat_*.xlsx"

    X_all, Y_all, C_all = [], [], []

    for folder in folders:
        filepaths = sorted(glob.glob(os.path.join(folder, pattern)))
        for path in filepaths:
            filename = os.path.basename(path)
            if "f8_stat" in filename.lower() or "f10_stat" in filename.lower():
                try:
                    X, Y, C = load_data(path)
                    X_seq, Y_seq, C_seq = create_sequences(X, Y, C, n_steps)
                    X_all.append(X_seq[:25])
                    Y_all.append(Y_seq[:25])
                    C_all.append(C_seq[:25])
                    print(f"Otwarto plik: {filename}")
                except Exception as e:
                    print(f"Błąd w pliku {filename}: {e}")

    X_combined = np.vstack(X_all)
    Y_combined = np.vstack(Y_all)
    C_combined = np.vstack(C_all)

    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X_combined)
    Y_scaled = Y_scaler.fit_transform(Y_combined)

    return X_scaled, Y_scaled, X_scaler, Y_scaler


def load_all_test_data(X_scaler, Y_scaler, C_scaler, n_steps=3):
    folders = ["../pomiary/F8", "../pomiary/F10"]
    pattern = "*_*.xlsx"

    X_all, Y_all, C_all = [], [], []

    for folder in folders:
        filepaths = sorted(glob.glob(os.path.join(folder, pattern)))
        for path in filepaths:
            filename = os.path.basename(path)
            if all(substr not in filename.lower() for substr in ["f8_stat", "f10_stat", "f8_random", "f10_random"]):
                try:
                    X, Y, C = load_data(path)
                    X_seq, Y_seq, C_seq = create_sequences(X, Y, C, n_steps)
                    X_all.append(X_seq)
                    Y_all.append(Y_seq)
                    C_all.append(C_seq)
                    print(f"Otwarto plik: {filename}")
                except Exception as e:
                    print(f"Błąd w pliku {filename}: {e}")

    X_combined = np.vstack(X_all)
    Y_combined = np.vstack(Y_all)
    C_combined = np.vstack(C_all)

    X_scaled = X_scaler.transform(X_combined)
    Y_scaled = Y_scaler.transform(Y_combined)
    C_scaled = C_scaler.fit_transform(C_combined)

    return X_scaled, Y_scaled, C_scaled

