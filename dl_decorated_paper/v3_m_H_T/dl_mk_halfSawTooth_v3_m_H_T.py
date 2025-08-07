# Улучшенный код обучения модели для задачи "decorated" решетки

import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from dl_decorated_paper import math_utils

# ============================
# Параметры
# ============================

#--------------File names
TRAINED_MODEL_FILE = 'model_decorated_mHT.pth'
LEARNING_CURVE_PLOT_FILE = 'learning_curve_decorated_mHT.png'

#--------------PhysicalSystem parameters
COMPUTE_TYPE = 'analytical'
LATTICE_TYPE = 'decorated'

H_MIN = 0
H_MAX = 10
H_POINTS_NUMBER = 64
H_values = np.linspace(H_MIN, H_MAX, H_POINTS_NUMBER)

#T = 1.0
T_MIN, T_MAX, T_POINTS_NUMBER = 0.1, 1.0, 8
T_values = np.linspace(T_MIN, T_MAX, T_POINTS_NUMBER)

J_MIN = -1
J_MAX = +1

# connection between code parameters and paper parameters
# J1=Jd
# J2=Jd
# J3=J
# J4=Jt

# lattice types
# 'general' -> J1, J2, J3, J4
# 'decorated' -> J1, J2=J1, J3, J4
# 'molecule' -> J1, J2=J1, J3=J1, J4=0
# 'simple' -> J1, J2=J1, J3=0, J4=J1

#--------------Training parameters
TRAIN_SAMPLES_NUMBER = 50000
VAL_SAMPLES_NUMBER = 5000
TEST_SAMPLES_NUMBER = 5000
num_epochs = 300
BATCH_SIZE_PARAMETER = 64
LEARNING_RATE_PARAMETER = 0.001

# TRAIN_SAMPLES_NUMBER = 1000  # увеличили
# VAL_SAMPLES_NUMBER = 200
# TEST_SAMPLES_NUMBER = 200
# num_epochs = 5  # больше эпох
# BATCH_SIZE_PARAMETER = 32  # больше батчей
# LEARNING_RATE_PARAMETER = 0.001

num_cells = 20
num_steps = 10000
equil_steps = 500


#--------------NN architecture parameters
input_size = len(H_values)
hidden_size_1 = 256
hidden_size_2 = 128
output_size = 4

DROPOUT_PARAMETER = 0.1
WEIGHT_DECAY_PARAMETER = 1e-4




# ============================
# Нормализация данных
# ============================

# def normalize(X):
#     mean = np.mean(X, axis=1, keepdims=True)
#     std = np.std(X, axis=1, keepdims=True) + 1e-8
#     return (X - mean) / std

def normalize_2d(X):
    # X shape: (N, T, H)
    mean = X.mean(axis=(1,2), keepdims=True)
    std  = X.std (axis=(1,2), keepdims=True) + 1e-8
    return (X - mean) / std


# ============================
# Генерация данных
# ============================

# Генерация случайных параметров J в диапазоне от J_MIN до J_MAX
def random_J(lattice_type):
    if lattice_type == 'general':
        J1, J2, J3, J4 = np.random.uniform(J_MIN, J_MAX, 4)
        return J1, J2, J3, J4
    elif lattice_type == 'decorated':
        J1, J2, J3, J4 = np.random.uniform(J_MIN, J_MAX, 4)
        J2 = J1
        return J1, J2, J3, J4
    elif lattice_type == 'molecule':
        J1, J2, J3, J4 = np.random.uniform(J_MIN, J_MAX, 4)
        J2 = J1
        J3 = J1
        J4 = 0
        return J1, J2, J3, J4
    elif lattice_type == 'simple':
        J1, J2, J3, J4 = np.random.uniform(J_MIN, J_MAX, 4)
        J2 = J1
        J3 = 0
        J4 = J1
        return J1, J2, J3, J4
    else:
        J1, J2, J3, J4 =0,0,0,0
        return J1, J2, J3, J4

# m(H) for fixed T = 1.0
# def generate_sample(J_params):
#     J1, J2, J3, J4 = J_params
#     Jd = J1
#     J = J3
#     Jt = J4
#     mag_curve = [math_utils.m(J, Jd, Jt, H, T) for H in H_values]
#     return np.array(mag_curve)


# m(H) for T_values (0.1,1.0)
def generate_sample_2d(J_params):
    J1, J2, J3, J4 = J_params
    Jd, J, Jt = J1, J3, J4

    # создаём карту shape=(T_points, H_points)
    mag_map = np.zeros((len(T_values), len(H_values)), dtype=np.float32)
    for ti, T in enumerate(T_values):
        mag_map[ti] = [math_utils.m(J, Jd, Jt, H, T) for H in H_values]
    return mag_map

class MagnetizationDataset(Dataset):
    def __init__(self, samples, targets):
        # self.samples = torch.tensor(samples, dtype=torch.float32)
        # self.targets = torch.tensor(targets, dtype=torch.float32)

        # samples: numpy array (N, T, H)
        self.samples = torch.tensor(samples, dtype=torch.float32).unsqueeze(1)
        # теперь shape будет (N, 1, T, H) — пригодится для 2D-CNN
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

# ============================
# Модель
# ============================

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size_1)
#         self.bn1 = nn.BatchNorm1d(hidden_size_1)
#         self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
#         self.bn2 = nn.BatchNorm1d(hidden_size_2)
#         self.fc3 = nn.Linear(hidden_size_2, output_size)
#         self.dropout = nn.Dropout(DROPOUT_PARAMETER)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

class Net2D(nn.Module):
    def __init__(self):
        super().__init__()
        # вход: (batch, 1, T, H)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2)
        # после двух пуллингов размер станет (32, T/2/2, H/2/2)
        flattened_size = 32 * (T_POINTS_NUMBER//4) * (H_POINTS_NUMBER//4)
        self.fc1   = nn.Linear(flattened_size, 128)
        self.fc2   = nn.Linear(128, 4)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)


# ============================
# Основная функция
# ============================

def main():
    logging.basicConfig(level=logging.INFO)

    # Генерация данных
    logging.info("Generating datasets...")
    train_X, train_Y = [], []
    val_X, val_Y = [], []
    test_X, test_Y = [], []

    for _ in range(TRAIN_SAMPLES_NUMBER):
        J_params = random_J(LATTICE_TYPE)
        # mag_curve = generate_sample(J_params)
        mag_curve = generate_sample_2d(J_params)
        train_X.append(mag_curve)
        train_Y.append(J_params)

    for _ in range(VAL_SAMPLES_NUMBER):
        J_params = random_J(LATTICE_TYPE)
        # mag_curve = generate_sample(J_params)
        mag_curve = generate_sample_2d(J_params)
        val_X.append(mag_curve)
        val_Y.append(J_params)

    for _ in range(TEST_SAMPLES_NUMBER):
        J_params = random_J(LATTICE_TYPE)
        # mag_curve = generate_sample(J_params)
        mag_curve = generate_sample_2d(J_params)
        test_X.append(mag_curve)
        test_Y.append(J_params)

    # train_X = normalize(np.array(train_X))
    # val_X = normalize(np.array(val_X))
    # test_X = normalize(np.array(test_X))
    train_X = normalize_2d(np.array(train_X))
    val_X = normalize_2d(np.array(val_X))
    test_X = normalize_2d(np.array(test_X))

    train_dataset = MagnetizationDataset(train_X, train_Y)
    val_dataset = MagnetizationDataset(val_X, val_Y)
    test_dataset = MagnetizationDataset(test_X, test_Y)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PARAMETER, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PARAMETER, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PARAMETER, shuffle=False)

    # ======= sanity checks ========
    batch_X, batch_Y = next(iter(train_loader))
    assert batch_X.shape == (BATCH_SIZE_PARAMETER, 1, T_POINTS_NUMBER, H_POINTS_NUMBER), \
        f"Expected input shape {(BATCH_SIZE_PARAMETER,1,T_POINTS_NUMBER,H_POINTS_NUMBER)}, got {batch_X.shape}"
    assert batch_Y.shape == (BATCH_SIZE_PARAMETER, 4), \
        f"Expected target shape {(BATCH_SIZE_PARAMETER,4)}, got {batch_Y.shape}"
    # ===============================

    #model = Net()
    model = Net2D()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_PARAMETER, weight_decay=WEIGHT_DECAY_PARAMETER)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    train_losses, val_losses = [], []

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

        scheduler.step()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

    torch.save(model.state_dict(), TRAINED_MODEL_FILE)
    logging.info(f"Model saved to {TRAINED_MODEL_FILE}")

    # Построение кривой обучения
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LEARNING_CURVE_PLOT_FILE)
    plt.show()

if __name__ == "__main__":
    main()
