import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import logging

from dl_molecule import math_utils

# ==== Параметры ====
MODEL_PATH = 'model_decorated_v5.pth'
H_values = np.concatenate([np.linspace(0, 2, 32), np.linspace(2, 10, 32)])
input_size = len(H_values)

hidden_size = 512
output_size = 4
DROPOUT = 0.2
TEST_SAMPLES = 5000
BATCH_SIZE = 64
J_MIN, J_MAX = -1, 1
T = 1.0

# ==== Нормализация ====
def normalize(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

# ==== Генерация данных ====
def random_J():
    J1, J2, J3, J4 = np.random.uniform(J_MIN, J_MAX, 4)
    return J1, J1, J3, J4

def generate_sample(J_params):
    J1, J2, J3, J4 = J_params
    return np.array([math_utils.m(J3, J1, J4, H, T) for H in H_values])

class MagnetizationDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

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
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    logging.info(f"Model loaded from {MODEL_PATH}")

    # Генерация тестовых данных
    test_Y, test_X = zip(*[ (x := random_J(), generate_sample(x)) for _ in range(TEST_SAMPLES) ])
    test_X = normalize(np.array(test_X))
    test_Y = np.array(test_Y)

    test_loader = DataLoader(MagnetizationDataset(test_X, test_Y), batch_size=BATCH_SIZE)
    preds, targets = [], []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            out = model(X_batch)
            preds.append(out.numpy())
            targets.append(Y_batch.numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    abs_err = np.abs(targets - preds)
    mean_abs = np.mean(abs_err)
    median_abs = np.median(abs_err)

    eps = 0.1
    mask = np.abs(targets) > eps
    rel_err = np.zeros_like(abs_err)
    rel_err[mask] = abs_err[mask] / np.abs(targets[mask]) * 100

    mean_rel = np.mean(rel_err[mask])
    median_rel = np.median(rel_err[mask])

    print("\n=== Test Set Evaluation ===")
    print(f"Mean Absolute Error: {mean_abs:.6f}")
    print(f"Median Absolute Error: {median_abs:.6f}")
    print(f"Mean Relative Error (%): {mean_rel:.2f}")
    print(f"Median Relative Error (%): {median_rel:.2f}")
    print("============================\n")

    # Графики ошибок
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.linalg.norm(abs_err, axis=1), 'b.', markersize=2)
    plt.title("Absolute Error vs Test Sample")
    plt.xlabel("Test Sample")
    plt.ylabel("L2 Absolute Error")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.linalg.norm(rel_err, axis=1), 'r.', markersize=2)
    plt.title("Relative Error vs Test Sample")
    plt.xlabel("Test Sample")
    plt.ylabel("Relative Error (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('decorated_v5_errors.png')
    plt.show()

    # Истина vs Предсказание
    names = ['J1', 'J2', 'J3', 'J4']
    plt.figure(figsize=(16, 4))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.scatter(targets[:, i], preds[:, i], alpha=0.5)
        plt.plot([J_MIN, J_MAX], [J_MIN, J_MAX], 'r--')
        plt.xlabel(f'True {names[i]}')
        plt.ylabel(f'Predicted {names[i]}')
        plt.title(f'{names[i]} Prediction')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('decorated_v5_true_vs_pred.png')
    plt.show()

if __name__ == "__main__":
    main()
