import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str):
    df = pd.read_excel(filepath)

    features = [
        "data__tagData__gyro__x", "data__tagData__gyro__y", "data__tagData__gyro__z",
        "data__tagData__magnetic__x", "data__tagData__magnetic__y", "data__tagData__magnetic__z",
        "data__tagData__linearAcceleration__x", "data__tagData__linearAcceleration__y",
        "data__tagData__linearAcceleration__z",
        "data__acceleration__x", "data__acceleration__y", "data__acceleration__z",
        "data__orientation__yaw", "data__orientation__roll", "data__orientation__pitch",
        "data__coordinates__x", "data__coordinates__y"
    ]

    labels = ["reference__x", "reference__y"]

    X = (df[features].values)[:4]
    Y = (df[labels].values)[:4]

    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    Y_scaled = Y_scaler.fit_transform(Y)

    # 6sta kolumna - dane z yaw
    # yaw_rad = np.deg2rad(X[:5])

    return X_scaled, Y_scaled


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
                    X, Y = load_data(path)
                    X_all.append(X)
                    Y_all.append(Y)
                    print(f"Otwarto plik: {filename}")
                except Exception as e:
                    print(f"Błąd w pliku {filename}: {e}")

    # Sklej dane
    X_combined = np.vstack(X_all)
    Y_combined = np.vstack(Y_all)

    # Skalowanie
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_combined)
    Y_scaled = Y_scaler.fit_transform(Y_combined)

    return X_scaled, Y_scaled