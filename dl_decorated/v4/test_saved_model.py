# test_saved_model.py

import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dl_molecule import math_utils

# ============================
# Параметры
# ============================

MODEL_SAVE_PATH = 'model_decorated.pth'
H_values = np.concatenate([
    np.linspace(0, 2, 32),
    np.linspace(2, 10, 32)
])
input_size = len(H_values)

T = 1.0
J_MIN = -1
J_MAX = +1
TEST_SAMPLES_NUMBER = 5000
BATCH_SIZE_PARAMETER = 64
DROPOUT_PARAMETER = 0.1

hidden_size_1 = 512
hidden_size_2 = 256
output_size = 4

# ============================
# Нормализация данных
# ============================

def normalize(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

# ============================
# Генерация данных
# ============================

def random_J():
    J1, J2, J3, J4 = np.random.uniform(J_MIN, J_MAX, 4)
    J2 = J1
    return J1, J2, J3, J4

def generate_sample(J_params):
    J1, J2, J3, J4 = J_params
    Jd = J1
    J = J3
    Jt = J4
    mag_curve = [math_utils.m(J, Jd, Jt, H, T) for H in H_values]
    return np.array(mag_curve)

class MagnetizationDataset(Dataset):
    def __init__(self, samples, targets):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

# ============================
# Модель
# ============================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.fc_extra = nn.Linear(hidden_size_2, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.dropout = nn.Dropout(DROPOUT_PARAMETER)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc_extra(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ============================
# Основная функция
# ============================

def main():
    logging.basicConfig(level=logging.INFO)

    model = Net()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    logging.info(f"Loaded model from {MODEL_SAVE_PATH}")

    test_X, test_Y = [], []
    for _ in range(TEST_SAMPLES_NUMBER):
        J_params = random_J()
        mag_curve = generate_sample(J_params)
        test_X.append(mag_curve)
        test_Y.append(J_params)

    test_X = normalize(np.array(test_X))
    test_Y = np.array(test_Y)

    test_dataset = MagnetizationDataset(test_X, test_Y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PARAMETER, shuffle=False)

    predictions, targets = [], []

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            outputs = model(batch_X)
            predictions.append(outputs.numpy())
            targets.append(batch_Y.numpy())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets)

    absolute_errors = np.abs(targets - predictions)
    mean_absolute_error = np.mean(absolute_errors)
    median_absolute_error = np.median(absolute_errors)

    epsilon = 0.1
    mask = np.abs(targets) > epsilon
    relative_errors = np.zeros_like(absolute_errors)
    relative_errors[mask] = np.abs(targets[mask] - predictions[mask]) / np.abs(targets[mask]) * 100

    mean_relative_error = np.mean(relative_errors[mask])
    median_relative_error = np.median(relative_errors[mask])

    print("\n=== Test Set Evaluation ===")
    print(f"Mean Absolute Error: {mean_absolute_error:.6f}")
    print(f"Median Absolute Error: {median_absolute_error:.6f}")
    print(f"Mean Relative Error (%): {mean_relative_error:.2f}")
    print(f"Median Relative Error (%): {median_relative_error:.2f}")
    print("============================\n")

    # Графики
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.linalg.norm(absolute_errors, axis=1), 'bo', markersize=2)
    plt.title("Absolute Error vs Test Sample")
    plt.xlabel("Test Sample Number")
    plt.ylabel("Absolute Error (L2 norm)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.linalg.norm(relative_errors, axis=1), 'ro', markersize=2)
    plt.title("Relative Error vs Test Sample (Filtered)")
    plt.xlabel("Test Sample Number")
    plt.ylabel("Relative Error (%)")
    plt.ylim(-1, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dl_mk_decorated_errors_improved.png')
    plt.show()

    # Истинные vs Предсказанные параметры
    param_names = ['J1', 'J2', 'J3', 'J4']
    plt.figure(figsize=(16, 4))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
        plt.plot([J_MIN, J_MAX], [J_MIN, J_MAX], 'r--')
        plt.xlabel(f'True {param_names[i]}')
        plt.ylabel(f'Predicted {param_names[i]}')
        plt.title(f'{param_names[i]} Prediction')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('dl_mk_true_vs_pred_decorated.png')
    plt.show()

if __name__ == "__main__":
    main()
