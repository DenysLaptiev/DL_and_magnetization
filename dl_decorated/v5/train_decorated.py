import logging, random, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dl_molecule import math_utils

# ==== Параметры ====
H_values = np.concatenate([np.linspace(0, 2, 32), np.linspace(2, 10, 32)])
input_size = len(H_values)
hidden_size = 512
output_size = 4

T = 1.0
J_MIN, J_MAX = -1, 1
TRAIN_SAMPLES = 50000
VAL_SAMPLES = 5000
EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 5e-4
DROPOUT = 0.2
MODEL_PATH = 'model_decorated_v5.pth'

# ==== Утилиты ====
def normalize(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

def random_J():
    J1, J2, J3, J4 = np.random.uniform(J_MIN, J_MAX, 4)
    return J1, J1, J3, J4

def generate_sample(J_params):
    J1, J2, J3, J4 = J_params
    return np.array([math_utils.m(J3, J1, J4, H, T) + np.random.normal(0, 0.01) for H in H_values])

class MagnetizationDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

# ==== Модель ====
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        x = self.shared(x)
        return self.branch(x)

# ==== Основная функция ====
def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Generating datasets...")
    def make_set(n): return zip(*[ (x := random_J(), generate_sample(x)) for _ in range(n) ])
    train_Y, train_X = make_set(TRAIN_SAMPLES)
    val_Y, val_X = make_set(VAL_SAMPLES)
    train_X, val_X = map(lambda X: normalize(np.array(X)), (train_X, val_X))
    train_Y, val_Y = map(np.array, (train_Y, val_Y))

    train_loader = DataLoader(MagnetizationDataset(train_X, train_Y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MagnetizationDataset(val_X, val_Y), batch_size=BATCH_SIZE)

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    mse = nn.MSELoss()

    best_loss = float('inf')
    best_state = None
    train_losses, val_losses = [], []

    logging.info("Start training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X, Y in train_loader:
            pred = model(X)
            loss = mse(pred, Y) + 0.1 * mse(pred[:, 0], pred[:, 1])  # симметрия J1 ≈ J2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        train_losses.append(total_loss / len(train_loader.dataset))

        model.eval()
        with torch.no_grad():
            val_loss = sum((mse(model(X), Y) + 0.1 * mse(model(X)[:, 0], model(X)[:, 1])).item() * X.size(0)
                           for X, Y in val_loader)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

        logging.info(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")

    torch.save(best_state, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig('learning_curve_decorated_v5.png')
    plt.show()

if __name__ == "__main__":
    main()
