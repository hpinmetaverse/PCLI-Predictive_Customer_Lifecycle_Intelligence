# MLP Neural Network for Customer Churn Prediction
# Minor Project AK7 — JUET Guna (MP)
# Team: Harsh Vardhan Chauhan, Himanshu S. Patil, Rudransh Srivastava

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


class ChurnMLP(nn.Module):
    """
    Multi-Layer Perceptron for binary churn classification.

    Architecture:
        Input -> FC(128) -> BN -> ReLU -> Dropout(0.3)
              -> FC(64)  -> BN -> ReLU -> Dropout(0.3)
              -> FC(32)  -> ReLU
              -> FC(1)   -> Sigmoid
    """

    def __init__(self, input_dim: int):
        super(ChurnMLP, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              n_epochs: int = 50, batch_size: int = 64,
              learning_rate: float = 1e-3,
              random_state: int = 42) -> tuple:
    """
    Train the MLP model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Adam optimizer learning rate
        random_state: Random seed

    Returns:
        (trained_model, train_losses, val_losses)
    """
    torch.manual_seed(random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train.astype(float)).to(device)
    X_vl = torch.FloatTensor(X_val).to(device)
    y_vl = torch.FloatTensor(y_val.astype(float)).to(device)

    # Dataset & DataLoader
    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    input_dim = X_train.shape[1]
    model = ChurnMLP(input_dim).to(device)

    # Weighted BCE loss to handle any remaining imbalance
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    print(f"\nTraining MLP for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_vl)
            val_loss = criterion(val_preds, y_vl).item()

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1:3d}/{n_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Restore best weights
    model.load_state_dict(best_state)
    print(f"\n✓ MLP training complete. Best val loss: {best_val_loss:.4f}")

    return model, train_losses, val_losses


def evaluate_mlp(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluate MLP on test set.

    Args:
        model: Trained ChurnMLP
        X_test, y_test: Test data
        threshold: Decision threshold for binary classification

    Returns:
        dict with metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        proba = model(X_tensor).cpu().numpy()

    y_pred = (proba >= threshold).astype(int)

    metrics = {
        'model': 'MLP (PyTorch)',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, proba)
    }

    print(f"\nMLP Evaluation:")
    for k, v in metrics.items():
        if k != 'model':
            print(f"  {k.capitalize():12s}: {v:.4f}")

    return metrics


def plot_training_curves(train_losses: list, val_losses: list, save_path: str = None):
    """Plot train vs validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train Loss', color='#6366f1', linewidth=2)
    ax.plot(val_losses, label='Val Loss', color='#06b6d4', linewidth=2, linestyle='--')
    ax.set_title('MLP Training Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE Loss')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def save_mlp(model, filepath: str = './models/mlp_model.pkl'):
    """Save MLP model using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ MLP saved to: {filepath}")


def load_mlp(filepath: str = './models/mlp_model.pkl') -> ChurnMLP:
    """Load saved MLP model."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    model.eval()
    print(f"✓ MLP loaded from: {filepath}")
    return model
