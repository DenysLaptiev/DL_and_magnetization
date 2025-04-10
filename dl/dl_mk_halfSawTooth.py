import numpy as np
import random
import matplotlib.pyplot as plt
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# PyTorch импорты
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



# =============================
# Формирование датасетов для обучения, валидации и теста
# =============================

# Параметры симуляции
num_cells = 20           # для демонстрации – небольшая решётка
num_steps = 10000        # число шагов Монте-Карло
equil_steps = 500
T = 1.0                 # фиксированная температура
H_values = np.linspace(0, 10, 20)  # 50 точек для кривой M(H) от -5 до +5

# Количество сэмплов в датасетах
TRAIN_SAMPLES_NUMBER = 500
VAL_SAMPLES_NUMBER = 100
TEST_SAMPLES_NUMBER = 100

J_MIN = -1
J_MAX = +1



# =============================
# Определение нейронной сети для регрессии
# =============================

input_size = len(H_values)  # 50 значений кривой M(H)
hidden_size = 64 # упростил сеть, чтоб избежать переобучения 64 -> 32
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
def generate_sample(J_params, T, num_cells, num_steps, equil_steps, H_values, verbose=False):
    """
    Для заданного набора параметров J_params = [J1, J2, J3, J4] генерирует
    кривую намагниченности M(H) по значениям внешнего поля из H_values.
    """
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


# =============================
# Формирование датасетов для обучения, валидации и теста
# =============================

# Генерация случайных параметров J в диапазоне от J_MIN до J_MAX
def random_J():
    # Генерация случайных параметров J в диапазоне от -2 до +2
    return np.random.uniform(J_MIN, J_MAX, 4)



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
# Определение нейронной сети для регрессии
# =============================

# Определение класса для нейронной сети
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def generate_train_dataset():
    logging.info("Starting generation of training dataset...")
    # Генерация обучающего датасета
    train_X = []
    train_Y = []
    for i in range(TRAIN_SAMPLES_NUMBER):
        J_params = random_J()
        print('J_params=', J_params)
        mag_curve = generate_sample(J_params, T, num_cells, num_steps, equil_steps, H_values, verbose=False)
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
        J_params = random_J()
        mag_curve = generate_sample(J_params, T, num_cells, num_steps, equil_steps, H_values, verbose=False)
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
        J_params = random_J()
        mag_curve = generate_sample(J_params, T, num_cells, num_steps, equil_steps, H_values, verbose=False)
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

    # Оценка на тестовом датасете
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            test_loss += loss.item() * batch_X.size(0)
    test_loss /= len(test_dataset)
    logging.info(f"Test Loss: {test_loss:.4f}")

    # Сравнение предсказаний и истинных значений для тестовых сэмплов
    with torch.no_grad():
        sample_input = torch.tensor(test_X, dtype=torch.float32)
        predictions = model(sample_input).numpy()

    print("Test Targets vs Predictions:")
    for target, pred in zip(test_Y, predictions):
        abs_error = np.abs(target - pred)
        # Для относительной ошибки избегаем деления на 0
        rel_error = np.where(np.abs(target) > 1e-6, abs_error / np.abs(target) * 100, 0)
        print("Target:", target)
        print("Prediction:", pred)
        print("Absolute Error:", abs_error)
        print("Relative Error (%):", rel_error)
        print("----------")


if __name__ == '__main__':
    main()