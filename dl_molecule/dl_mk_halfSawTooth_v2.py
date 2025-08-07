import logging
import random

import matplotlib.pyplot as plt
import numpy as np

from dl_molecule import math_utils

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# PyTorch импорты
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from scipy.stats import mode



# =============================
# Формирование датасетов для обучения, валидации и теста
# =============================

# Параметры симуляции
COMPUTE_TYPE = 'analytical' # 'analytical', 'montecarlo'
LATTICE_TYPE = 'molecule' # 'general', 'decorated', 'molecule', 'simple'
num_cells = 20           # число елементарных ячеек решётки
num_steps = 10000        # общее число шагов симуляции Монте-Карло
equil_steps = 500        # число шагов Монте-Карло для установления равновесия (не учитываются в измерениях).
T = 1.0                 # фиксированная температура

H_MIN = 0
H_MAX = 10
H_POINTS_NUMBER = 64
H_values = np.linspace(H_MIN, H_MAX, H_POINTS_NUMBER)  # H_POINTS_NUMBER точек для кривой M(H) от H_MIN до H_MAX

# Количество сэмплов в датасетах
TRAIN_SAMPLES_NUMBER = 10000
VAL_SAMPLES_NUMBER = 2000
TEST_SAMPLES_NUMBER = 2000

J_MIN = -1
J_MAX = +1



# =============================
# Определение нейронной сети для регрессии
# =============================

input_size = len(H_values)  # число точек M для кривой M(H)
hidden_size = 128 # число нейронов в скрытом слое
output_size = 4             # предсказываем четыре параметра J
DROPOUT_PARAMETER = 0.1 # 10% связей обрывается, чтобы избежать переобучение
WEIGHT_DECAY_PARAMETER=1e-4 # штраф на большие веса, поможет избежать переобучения


# =============================
# Обучение нейронной сети и построение кривой обучения
# =============================

num_epochs = 100

BATCH_SIZE_PARAMETER = 8
LEARNING_RATE_PARAMETER = 0.001















# =============================
# Функции для симуляции модели half‑saw‑tooth
# =============================

# Строим список bonds: какой тип связи между спинами определённых типов
def build_bonds(num_cells, J1, J2, J3, J4):
    """
    Формирует список связей для решётки half‑saw‑tooth.
    Для ячейки 0 (n=0) независимые спины с индексами:
      A0 = 0, B0 = 1, C0 = 2, D0 = 3.
    Для ячейки n (n>=1):
      A_n = 3*n, B_n = 3*n+1, C_n = 3*n+2, D_n = 3*n+3
      A_n = D_(n-1) (то есть уже существует)
    Связи:
      - (A, B) с J1,
      - (B, C) с J2,
      - (A, C) с J3,
      - (C, D) с J4.
    """
    bonds = []
    # Ячейка 0:
    bonds.append((0, 1, J1))  # (A0, B0)
    bonds.append((1, 2, J2))  # (B0, C0)
    bonds.append((0, 2, J3))  # (A0, C0)
    bonds.append((2, 3, J4))  # (C0, D0) -> D0 совпадает с A1
    # Ячейки n = 1, ..., num_cells-1:
    for n in range(1, num_cells):
        # В ячейке n спин A_n уже задан как D_(n-1)
        A = 3 * n         # A_n = D_(n-1)
        B = 3 * n + 1     # новый спин B_n
        C = 3 * n + 2     # новый спин C_n
        D = 3 * n + 3     # новый спин D_n (будет совпадать с A_(n+1))
        bonds.append((A, B, J1))  # (A_n, B_n)
        bonds.append((B, C, J2))  # (B_n, C_n)
        bonds.append((A, C, J3))  # (A_n, C_n)
        bonds.append((C, D, J4))  # (C_n, D_n)
    return bonds

# Строим список neighbors: для каждого спина i указываем его соседей и константы взаимодействия с соседями
def build_neighbors(num_spins, bonds):
    """
    Для каждого спина строит список соседей в формате: neighbors[i] = [(j, J), ...].
    """
    neighbors = {i: [] for i in range(num_spins)}
    for i, j, J in bonds:
        neighbors[i].append((j, J))
        neighbors[j].append((i, J))
    return neighbors

# Симуляция методом Монте‑Карло для системы Изинга на решётке half‑saw‑tooth.
# Аргументы:
#   num_spins  -- число независимых спинов (для num_cells ячеек: 3*num_cells + 1).
#   neighbors  -- словарь соседей.
#   T          -- температура.
#   H          -- внешнее магнитное поле.
#   num_steps  -- общее число шагов симуляции.
#   equil_steps-- число шагов для установления равновесия (не учитываются в измерениях).
#   verbose    -- если True, выводить промежуточное логирование.
# Возвращает нормированную намагниченность <M> (сумма спинов, делённая на число спинов).
def monte_carlo_half_sawtooth(num_spins, neighbors, T, H, num_steps, equil_steps=500, verbose=False):
    """
    Симуляция методом Монте‑Карло для системы Изинга на решётке half‑saw‑tooth.
    Аргументы:
      num_spins  -- число независимых спинов (для num_cells ячеек: 3*num_cells + 1).
      neighbors  -- словарь соседей.
      T          -- температура.
      H          -- внешнее магнитное поле.
      num_steps  -- общее число шагов симуляции.
      equil_steps-- число шагов для установления равновесия (не учитываются в измерениях).
      verbose    -- если True, выводить промежуточное логирование.
    Возвращает нормированную намагниченность <M> (сумма спинов, делённая на число спинов).
    """
    spins = np.random.choice([-1, 1], size=num_spins)
    mag_list = []
    progress_interval = max(num_steps // 10, 1)
    for step in range(num_steps):
        # Выбираем случайный спин для попытки переворота
        i = random.randint(0, num_spins - 1)
        local_field = 0.0
        for j, J in neighbors[i]:
            local_field += J * spins[j]
        # Изменение энергии при перевороте спина i
        delta_E = 2 * spins[i] * (local_field + H)
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            spins[i] = -spins[i]
        if step >= equil_steps:
            mag_list.append(np.sum(spins))
        # Логирование прогресса симуляции
        if verbose and step % progress_interval == 0:
            logging.info(f"MC simulation (H={H:.2f}): step {step}/{num_steps}")
    avg_m = np.mean(mag_list) / num_spins
    return avg_m


# Генерируем датасет. J_params = [J1, J2, J3, J4] и кривая намагниченности M(H) (H=H_values)
def generate_sample(compute_type, J_params, T, num_cells, num_steps, equil_steps, H_values, verbose=False):
    """
    Для заданного набора параметров J_params = [J1, J2, J3, J4] генерирует
    кривую намагниченности M(H) по значениям внешнего поля из H_values.
    """
    if compute_type == 'montecarlo':
        J1, J2, J3, J4 = J_params
        bonds = build_bonds(num_cells, J1, J2, J3, J4)
        num_spins = 3 * num_cells + 1
        neighbors = build_neighbors(num_spins, bonds)
        mag_curve = []
        for idx, H in enumerate(H_values):
            m = monte_carlo_half_sawtooth(num_spins, neighbors, T, H, num_steps, equil_steps, verbose=verbose)
            mag_curve.append(m)
            if verbose:
                logging.info(f"Generated sample: H[{idx}]={H:.2f}, M={m:.4f}")
        return np.array(mag_curve)
    elif compute_type == 'analytical':

        #!!!!! there is analytical solution only for 'decorated', 'simple' and 'molecule' lattice_type. No analytical solution for 'general' lattice_type (use 'montecarlo' for 'general')
        J1, J2, J3, J4 = J_params
        Jd = J1 #(J2=J1)
        J = J3
        Jt = J4
        mag_curve = []
        for idx, H in enumerate(H_values):
            m = math_utils.m(J, Jd, Jt, H, T)
            mag_curve.append(m)
            if verbose:
                logging.info(f"Generated sample: H[{idx}]={H:.2f}, M={m:.4f}")
        return np.array(mag_curve)
    else:
        mag_curve = []
        return np.array(mag_curve)





# =============================
# Формирование датасетов для обучения, валидации и теста
# =============================

# Генерация случайных параметров J в диапазоне от J_MIN до J_MAX
def random_J(lattice_type):
    # Генерация случайных параметров J в диапазоне от -2 до +2

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



# Определяем класс Dataset для PyTorch
class MagnetizationDataset(Dataset):
    def __init__(self, samples, targets):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]



# =============================
# Определение нейронной сети для регрессии (улучшенная версия)
# =============================

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=DROPOUT_PARAMETER)  # Dropout с заданным параметром
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Добавляем dropout только после активации скрытого слоя
        x = self.fc2(x)
        return x





def generate_train_dataset():
    logging.info("Starting generation of training dataset...")
    # Генерация обучающего датасета
    train_X = []
    train_Y = []
    for i in range(TRAIN_SAMPLES_NUMBER):
        J_params = random_J(LATTICE_TYPE)
        print('J_params=', J_params)
        mag_curve = generate_sample(COMPUTE_TYPE, J_params, T, num_cells, num_steps, equil_steps, H_values, verbose=False)
        train_X.append(mag_curve)
        train_Y.append(J_params)
        logging.info(f"Training sample {i + 1}/{TRAIN_SAMPLES_NUMBER} generated.")
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    return train_X, train_Y

def generate_validation_dataset():
    logging.info("Generating validation dataset...")
    # Генерация датасета для валидации
    val_X = []
    val_Y = []
    for i in range(VAL_SAMPLES_NUMBER):
        J_params = random_J(LATTICE_TYPE)
        mag_curve = generate_sample(COMPUTE_TYPE, J_params, T, num_cells, num_steps, equil_steps, H_values, verbose=False)
        val_X.append(mag_curve)
        val_Y.append(J_params)
        logging.info(f"Validation sample {i + 1}/{VAL_SAMPLES_NUMBER} generated.")
    val_X = np.array(val_X)
    val_Y = np.array(val_Y)
    return val_X, val_Y

def generate_test_dataset():
    logging.info("Generating test dataset...")
    # Генерация тестового датасета
    test_X = []
    test_Y = []
    for i in range(TEST_SAMPLES_NUMBER):
        J_params = random_J(LATTICE_TYPE)
        mag_curve = generate_sample(COMPUTE_TYPE, J_params, T, num_cells, num_steps, equil_steps, H_values, verbose=False)
        test_X.append(mag_curve)
        test_Y.append(J_params)
        logging.info(f"Test sample {i + 1}/{TEST_SAMPLES_NUMBER} generated.")
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    return test_X, test_Y

def main():
    train_X, train_Y = generate_train_dataset()

    val_X, val_Y = generate_validation_dataset()

    test_X, test_Y = generate_test_dataset()




    # Создаём объекты Dataset и DataLoader
    batch_size = BATCH_SIZE_PARAMETER
    train_dataset = MagnetizationDataset(train_X, train_Y)
    val_dataset = MagnetizationDataset(val_X, val_Y)
    test_dataset = MagnetizationDataset(test_X, test_Y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Net(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_PARAMETER,
                           weight_decay=WEIGHT_DECAY_PARAMETER)  # штраф на большие веса, поможет избежать переобучения

    # =============================
    # Обучение нейронной сети и построение кривой обучения
    # =============================

    train_losses = []
    val_losses = []

    logging.info("Starting training of the neural network...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)

        # Расчёт ошибки на валидационном датасете
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                running_val_loss += loss.item() * batch_X.size(0)
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    # Построение кривой обучения
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Кривая обучения")
    plt.legend()
    plt.grid(True)
    plt.savefig('dl_mk_learningCurve.png')
    plt.show()

    # =============================
    # Оценка на тестовом датасете с улучшенной обработкой ошибок
    # =============================

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            test_loss += loss.item() * batch_X.size(0)
    test_loss /= len(test_dataset)
    logging.info(f"Test Loss (MSE): {test_loss:.6f}")

    # Предсказания на всём тестовом наборе
    with torch.no_grad():
        sample_input = torch.tensor(test_X, dtype=torch.float32)
        predictions = model(sample_input).numpy()

    targets = np.array(test_Y)

    # Абсолютные ошибки
    absolute_errors = np.abs(targets - predictions)
    mean_absolute_error = np.mean(absolute_errors)
    median_absolute_error = np.median(absolute_errors)

    # Маска: только там, где истинные значения "достаточно далеко от нуля"
    epsilon = 0.1
    mask = np.abs(targets) > epsilon

    # Относительные ошибки (с маской)
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

    # Построение графиков ошибок
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(targets)), np.linalg.norm(absolute_errors, axis=1), marker='o', linestyle='', color='blue')
    plt.title("Absolute Error vs Test Sample")
    plt.xlabel("Test Sample Number")
    plt.ylabel("Absolute Error (L2 norm)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(targets)), np.linalg.norm(relative_errors, axis=1), marker='o', linestyle='', color='red')
    plt.title("Relative Error vs Test Sample (Filtered)")
    plt.xlabel("Test Sample Number")
    plt.ylabel("Relative Error (%)")
    plt.ylim(-1, 100)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('dl_mk_molecule_errors_improved.png')
    plt.show()

    # =============================
    # Визуализация "Истина vs Предсказание" для каждого параметра J
    # =============================

    param_names = ['J1', 'J2', 'J3', 'J4']

    plt.figure(figsize=(16, 4))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
        plt.xlim(J_MIN, J_MAX)
        plt.ylim(J_MIN, J_MAX)
        plt.plot([J_MIN, J_MAX], [J_MIN, J_MAX], 'r--')  # Диагональ "идеальных предсказаний"
        plt.xlabel(f'True {param_names[i]}')
        plt.ylabel(f'Predicted {param_names[i]}')
        plt.title(f'{param_names[i]} Prediction')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('dl_mk_true_vs_pred.png')
    plt.show()


if __name__ == '__main__':
    main()