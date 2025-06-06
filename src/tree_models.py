import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse


def load_data(csv_path, feature_cols, allowed_categories=["Optimal", "Quasi-optimal"]):
    df = pd.read_csv(csv_path)
    
    # filter the allowed categories
    df = df[df["solve_category"].isin(allowed_categories)].copy()

    # non-baseline params set (param_id ≠ 0)
    df = df[df["param_id"] != 0].copy()

    # One-hot encodding param_id
    df_onehot = pd.get_dummies(df["param_id"], prefix="param")

    # concat datasets and targets
    X = pd.concat([df[feature_cols], df_onehot], axis=1).astype(np.float32)
    y = np.log1p(df["runtime"].values.astype(np.float32))

    # Normalise scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def train_and_evaluate(X, y, model_type="rf"):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "xgb":
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    else:
        raise ValueError("Unsupported model type")
  
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_val)

    # evaluate 
    print(f"\n Model: {model_type.upper()}")
    print(f"  MAE : {mean_absolute_error(y_true, y_pred):.2f} seconds")
    print(f"  R²  : {r2_score(y_true, y_pred):.4f}")

    return model

