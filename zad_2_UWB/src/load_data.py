import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data(filepath: str):
    df = pd.read_excel(filepath)

    features = [

        "data__coordinates__x", "data__coordinates__y"
    ]

    labels = ["reference__x", "reference__y"]
    coords = ["data__coordinates__x", "data__coordinates__y"]

    # Usuń kolumny, które są w całości puste
    df = df.dropna(axis=1, how='all')

    # Usuń wiersze z brakującymi danymi w istotnych kolumnach
    df = df.dropna(subset=features + labels)

    # Jeśli po czyszczeniu nie ma danych — rzuć wyjątek
    if df.empty:
        raise ValueError(f"Plik {filepath} zawiera tylko puste dane po przefiltrowaniu.")

    X = (df[features].values)
    Y = (df[labels].values)
    C = (df[coords].values)

    # 6sta kolumna - dane z yaw
    # yaw_rad = np.deg2rad(X[:5])

    return X, Y, C


def load_all_training_data():
    # Foldery do przeszukania
    folders = ["../pomiary/F8", "../pomiary/F10"]
    pattern = "*_stat_*.xlsx"

    X_all, Y_all = [], []

    for folder in folders:
        filepaths = glob.glob(os.path.join(folder, pattern))
        for path in filepaths:
            filename = os.path.basename(path)
            if "f8_stat" in filename.lower() or "f10_stat" in filename.lower():
                try:
                    X, Y, C = load_data(path)
                    X_all.append(X[:25])
                    Y_all.append(Y[:25])
                    print(f"Otwarto plik: {filename}")
                except Exception as e:
                    print(f"Błąd w pliku {filename}: {e}")

    # Sklej dane
    X_combined = np.vstack(X_all)
    Y_combined = np.vstack(Y_all)

    # Skalowanie
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X_combined)
    Y_scaled = Y_scaler.fit_transform(Y_combined)

    return X_scaled, Y_scaled, X_scaler, Y_scaler


def load_all_test_data(X_scaler, Y_scaler, C_scaler):
    # Foldery do przeszukania
    folders = ["../pomiary/F8", "../pomiary/F10"]
    pattern = "*_*.xlsx"

    X_all, Y_all, C_all = [], [], []

    for folder in folders:
        filepaths = glob.glob(os.path.join(folder, pattern))
        for path in filepaths:
            filename = os.path.basename(path)
            if "f8_stat" not in filename.lower() and "f10_stat" not in filename.lower() and "f8_random" not in filename.lower() and "f10_random" not in filename.lower()   :
                try:
                    X, Y, C = load_data(path)
                    X_all.append(X)
                    Y_all.append(Y)
                    C_all.append(C)
                    print(f"Otwarto plik: {filename}")
                except Exception as e:
                    print(f"Błąd w pliku {filename}: {e}")

    # Sklej dane
    X_combined = np.vstack(X_all)
    Y_combined = np.vstack(Y_all)
    C_combined = np.vstack(C_all)

    # Skalowanie
    X_scaled = X_scaler.transform(X_combined)
    Y_scaled = Y_scaler.transform(Y_combined)
    C_scaled = C_scaler.fit_transform(C_combined)

    return X_scaled, Y_scaled, C_scaled