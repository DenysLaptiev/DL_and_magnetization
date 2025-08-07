import logging
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dl_molecule import math_utils

# ============================
# Параметры
# ============================

COMPUTE_TYPE = 'analytical'
LATTICE_TYPE = 'decorated'

TRAIN_SAMPLES_NUMBER = 50000
VAL_SAMPLES_NUMBER = 5000
TEST_SAMPLES_NUMBER = 5000

H_values = np.concatenate([
    np.linspace(0, 2, 32),
    np.linspace(2, 10, 32)
])
input_size = len(H_values)

hidden_size_1 = 512
hidden_size_2 = 256
output_size = 4

DROPOUT_PARAMETER = 0.1
WEIGHT_DECAY_PARAMETER = 1e-4

num_epochs = 300
BATCH_SIZE_PARAMETER = 64
LEARNING_RATE_PARAMETER = 0.001

J_MIN = -1
J_MAX = +1

MODEL_SAVE_PATH = 'model_decorated.pth'

T = 1.0

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
    mag_curve = [math_utils.m(J, Jd, Jt, H, T) + np.random.normal(0, 0.01) for H in H_values]
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
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_extra(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ============================
# Основная функция
# ============================

def main():
    logging.basicConfig(level=logging.INFO)

    logging.info("Generating datasets...")
    train_X, train_Y = [], []
    val_X, val_Y = [], []
    test_X, test_Y = [], []

    for _ in range(TRAIN_SAMPLES_NUMBER):
        J_params = random_J()
        mag_curve = generate_sample(J_params)
        train_X.append(mag_curve)
        train_Y.append(J_params)

    for _ in range(VAL_SAMPLES_NUMBER):
        J_params = random_J()
        mag_curve = generate_sample(J_params)
        val_X.append(mag_curve)
        val_Y.append(J_params)

    for _ in range(TEST_SAMPLES_NUMBER):
        J_params = random_J()
        mag_curve = generate_sample(J_params)
        test_X.append(mag_curve)
        test_Y.append(J_params)

    train_X = normalize(np.array(train_X))
    val_X = normalize(np.array(val_X))
    test_X = normalize(np.array(test_X))

    train_dataset = MagnetizationDataset(train_X, train_Y)
    val_dataset = MagnetizationDataset(val_X, val_Y)
    test_dataset = MagnetizationDataset(test_X, test_Y)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PARAMETER, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PARAMETER, shuffle=False)

    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_PARAMETER, weight_decay=WEIGHT_DECAY_PARAMETER)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None

    logging.info("Start training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                running_val_loss += loss.item() * batch_X.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()

        scheduler.step()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

    torch.save(best_model_state, MODEL_SAVE_PATH)
    logging.info(f"Model saved to {MODEL_SAVE_PATH}")

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve_decorated.png')
    plt.show()

if __name__ == "__main__":
    main()
