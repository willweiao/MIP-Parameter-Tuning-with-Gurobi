import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


class MIPDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# those hyperparameters need to be learned later   
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_data(csv_path, feature_cols):
    df = pd.read_csv(csv_path)

    # non-baseline params set (param_id ≠ 0)
    df = df[df["param_id"] != 0].copy()

    # One-hot encodding param_id
    df_onehot = pd.get_dummies(df["param_id"], prefix="param")
    X = pd.concat([df[feature_cols], df_onehot], axis=1).astype(np.float32)
    y = df["runtime"].values.astype(np.float32)

    # Normalise scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# === TRAIN+VAL LOOP===
def train_loop(model, train_loader, val_loader, optimizer, criterion,
          epochs=50, device='cpu'):

    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # === Training ===
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch+1:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model, train_losses, val_losses

# ===TEST LOOP===
def evaluate_loop(model, loader, device='cpu', label="Test"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            y_true.extend(y_batch.numpy().flatten())
            y_pred.extend(pred.flatten())

    print(f"\n {label} Set:")
    print("  MAE :", mean_absolute_error(y_true, y_pred))
    print("  R²  :", r2_score(y_true, y_pred))
