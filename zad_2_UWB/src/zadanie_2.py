import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


""" Odczytujemy:
        pozycja: x, y
        poz. referencyjna x, y (prawdziwe pozycje)
        przyspieszenie (linear) x, y
        opoznienie 
"""
def load_data(filepath:str):
    df = pd.read_excel(filepath)
    
    features = ["data__coordinates__x", "data__coordinates__y",
                      "data__tagData__linearAcceleration__x", "data__tagData__linearAcceleration__y",
                      "data__metrics__latency", "data__orientation__yaw"]
    labels = ["reference__x", "reference__y"]
    
    X = df[features].values
    Y = df[labels].values
    
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    Y_scaled = Y_scaler.fit_transform(Y)
    
    # 6sta kolumna - dane z yaw
    yaw_rad = np.deg2rad(X[:5])



load_data("../pomiary/F8/f8_1p.xlsx")