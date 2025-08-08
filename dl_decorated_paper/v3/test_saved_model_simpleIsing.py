import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dl_decorated_paper import math_utils

# ============================
# Параметры
# ============================

#--------------File names
TRAINED_MODEL_FILE = 'model_decorated_mH.pth'
RESULT_EXPECTED_AND_ACTUAL_PLOT_FILE = 'dl_mk_true_vs_pred_decorated_mHT_simpleIsing.png'
RESIDUALS_HIST_FILE            = 'dl_residuals_hist_simple.png'

#--------------PhysicalSystem parameters
LATTICE_TYPE = 'simple'

H_MIN = 0
H_MAX = 10
H_POINTS_NUMBER = 64
H_values = np.linspace(H_MIN, H_MAX, H_POINTS_NUMBER)

T = 1.0

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


#--------------Training(Testing) parameters
TEST_SAMPLES_NUMBER = 5000
BATCH_SIZE_PARAMETER = 64

num_cells = 20


#--------------NN architecture parameters
input_size = len(H_values)
hidden_size_1 = 256
hidden_size_2 = 128
output_size = 4

DROPOUT_PARAMETER = 0.1


# ============================
# Нормализация данных
# ============================

def normalize(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

# ============================
# Генерация тестовых данных
# ============================

# def random_J():
#     J1, J2, J3, J4 = np.random.uniform(J_MIN, J_MAX, 4)
#     J2 = J1
#     return J1, J2, J3, J4

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
        x = self.fc3(x)
        return x

# ============================
# Основная функция
# ============================

def main():
    logging.basicConfig(level=logging.INFO)

    # Загрузка модели
    model = Net()
    model.load_state_dict(torch.load(TRAINED_MODEL_FILE))
    model.eval()
    logging.info(f"Loaded model from {TRAINED_MODEL_FILE}")

    # Генерация тестового датасета
    test_X = []
    test_Y = []
    for _ in range(TEST_SAMPLES_NUMBER):
        J_params = random_J(LATTICE_TYPE)
        mag_curve = generate_sample(J_params)
        test_X.append(mag_curve)
        test_Y.append(J_params)

    test_X = normalize(np.array(test_X))
    test_Y = np.array(test_Y)

    test_dataset = MagnetizationDataset(test_X, test_Y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PARAMETER, shuffle=False)

    # Оценка модели
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            outputs = model(batch_X)
            predictions.append(outputs.numpy())
            targets.append(batch_Y.numpy())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets)

    # Абсолютные ошибки
    absolute_errors = np.abs(targets - predictions)
    mean_absolute_error = np.mean(absolute_errors)
    median_absolute_error = np.median(absolute_errors)

    # Маска для относительных ошибок
    epsilon = 0.1
    mask = np.abs(targets) > epsilon

    relative_errors = np.zeros_like(absolute_errors)
    relative_errors[mask] = np.abs(targets[mask] - predictions[mask]) / np.abs(targets[mask]) * 100

    mean_relative_error = np.mean(relative_errors[mask])
    median_relative_error = np.median(relative_errors[mask])

    # Вывод результатов
    print("\n=== Test Set Evaluation ===")
    print(f"Mean Absolute Error: {mean_absolute_error:.6f}")
    print(f"Median Absolute Error: {median_absolute_error:.6f}")
    print(f"Mean Relative Error (%): {mean_relative_error:.2f}")
    print(f"Median Relative Error (%): {median_relative_error:.2f}")
    print("============================\n")

    # # Построение графиков ошибок
    # plt.figure(figsize=(12, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(range(len(targets)), np.linalg.norm(absolute_errors, axis=1), marker='o', linestyle='', color='blue')
    # plt.title("Absolute Error vs Test Sample")
    # plt.xlabel("Test Sample Number")
    # plt.ylabel("Absolute Error (L2 norm)")
    # plt.grid(True)
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(len(targets)), np.linalg.norm(relative_errors, axis=1), marker='o', linestyle='', color='red')
    # plt.title("Relative Error vs Test Sample (Filtered)")
    # plt.xlabel("Test Sample Number")
    # plt.ylabel("Relative Error (%)")
    # plt.ylim(-1, 100)
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.savefig('dl_mk_decorated_errors_improved.png')
    # plt.show()

    # Визуализация "Истина vs Предсказание"
    param_names = ['J1', 'J2', 'J3', 'J4']

    plt.figure(figsize=(16, 4))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
        plt.xlim(J_MIN, J_MAX)
        plt.ylim(J_MIN, J_MAX)
        plt.plot([J_MIN, J_MAX], [J_MIN, J_MAX], 'r--')
        plt.xlabel(f'True {param_names[i]}')
        plt.ylabel(f'Predicted {param_names[i]}')
        plt.title(f'{param_names[i]} Prediction')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULT_EXPECTED_AND_ACTUAL_PLOT_FILE)
    plt.show()

# 4) Абсолютные и квадратичные ошибки
    errors = predictions - targets
    abs_err = np.abs(errors)
    mse_per_j = np.mean(errors ** 2, axis=0)
    rmse_per_j = np.sqrt(mse_per_j)

    # 5) R² per parameter
    var_true = np.var(targets, axis=0)
    r2_per_j = 1.0 - mse_per_j / var_true

    # 6) Печать таблицы метрик
    '''
    MAE — MAE (Mean Absolute Error). Среднее абсолютное отклонение предсказания от истинного
    RMSE — RMSE (Root Mean Squared Error). Корень из среднего квадратичного отклонения. Чувствительнее к большим ошибкам (больше штрафует за крупные выбросы), чем MAE.
    R2 — коэффициент детерминации. Показывает, какую долю дисперсии (разброса) истинных значений параметра модель «объясняет»
        – R²≈1 означает почти идеальную подгонку;
        – R²≈0 — модель не лучше константы (среднего);
        – R²<0 — хуже, чем просто предсказывать среднее.
    '''
    param_names = ['J1', 'J2', 'J3', 'J4']
    print("\nParameter |   MAE    |   RMSE   |    R²   ")
    print("-----------------------------------------")
    for i, name in enumerate(param_names):
        mae = np.mean(abs_err[:, i])
        print(f"   {name:2s}     | {mae:7.4f} | {rmse_per_j[i]:7.4f} | {r2_per_j[i]:7.4f}")
    print()

    # 7) Parity-plots (истина vs предсказание)
    plt.figure(figsize=(16, 4))
    for i, name in enumerate(param_names):
        ax = plt.subplot(1, 4, i + 1)
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.4, s=10)
        ax.plot([J_MIN, J_MAX], [J_MIN, J_MAX], 'r--')
        ax.set_title(f"{name}: R²={r2_per_j[i]:.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
    plt.tight_layout()
    plt.savefig('R2' + RESULT_EXPECTED_AND_ACTUAL_PLOT_FILE)
    plt.show()

    # 8) Гистограммы остатков

    '''
    1) Остаток = Predicted − True.
    Справа от нуля — сеть завышает, слева — занижает.

    2) Среднее остатка (mean). 
    Оно близко к 0 → системного смещения почти нет.

    3) Разброс (std)
    Стандартное отклонение — «ширина» распределения ошибок. Чем меньше, тем точнее.

    4) Форма/симметрия/«хвосты»
    Узкий, симметричный «колокол» вокруг нуля — хорошо.
    Тяжёлые хвосты/асимметрия → есть редкие большие ошибки или систематическое смещение в отдельных зонах пространства параметров.
    У J3/J4 хвосты тяжелее и std больше → именно там модель ошибается сильнее (что согласуется с более низким R²).

    5)Зачем это нужно:
    Проверить смещение модели (bias): среднее далеко от 0 → модель систематически завышает/занижает.
    Понять устойчивость: тяжелые хвосты → есть области (часто около фазовых переходов/плато m(H), где обратная задача плохо обусловлена), где ошибка резко растёт.
    Связать с таблицей метрик: гистограммы объясняют, почему у J3/J4 MAE/RMSE хуже — распределения шире.
    '''
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(param_names):
        ax = plt.subplot(2, 2, i + 1)
        ax.hist(errors[:, i], bins=50, alpha=0.7)
        ax.set_title(f"Residuals of {name}\nmean={errors[:, i].mean():.3f}, std={errors[:, i].std():.3f}")
        ax.set_xlabel("Predicted − True")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESIDUALS_HIST_FILE)
    plt.show()


if __name__ == "__main__":
    main()