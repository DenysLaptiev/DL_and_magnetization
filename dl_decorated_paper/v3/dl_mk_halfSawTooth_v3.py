# Улучшенный код обучения модели для задачи "decorated" решетки

import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# используем torch
# 1. Динамический граф вычислений
# PyTorch строит вычислительный граф «на ходу» (eager execution).
#
# 2. Простота и «питоничность»
# API PyTorch максимально близок к чистому Python/Numpy-коду.
#
# 3. Большое сообщество и экосистема
# PyTorch на сегодня — один из самых популярных фреймворков в академической среде (более 60 % публикаций ML/статей используют его для репроducibility), а также активно набирает долю в индустрии.

from dl_decorated_paper import math_utils

# ============================
# Параметры
# ============================

#--------------File names
TRAINED_MODEL_FILE = 'model_decorated_mH.pth'
LEARNING_CURVE_PLOT_FILE = 'learning_curve_decorated_mH.png'

#--------------PhysicalSystem parameters
COMPUTE_TYPE = 'analytical'
LATTICE_TYPE = 'decorated'

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

#--------------Training parameters
#Сколько случайных примеров (пар «кривая m(H) → J-параметры») генерируется для обучения.
TRAIN_SAMPLES_NUMBER = 50000

#Размер валидационного подмножества, на котором отслеживается «общая» ошибка во время тренировки (не участвует в градиентном спуске).
VAL_SAMPLES_NUMBER = 5000

#Размер тестового подмножества, которым вы оцениваете финальную модель после окончания обучения.
TEST_SAMPLES_NUMBER = 5000

#Сколько целых проходов по всему тренировочному набору данных будет выполнено.
num_epochs = 300

#Размер мини-батча — сколько образцов одновременно прогоняется через сеть перед обновлением весов.
BATCH_SIZE_PARAMETER = 64

#Начальная скорость обучения (шаг градиентного спуска) для оптимизатора Adam.
LEARNING_RATE_PARAMETER = 0.001
'''
                    Процесс обучения
1. У нас есть TRAIN_SAMPLES_NUMBER образцов.
2. Мы разбиваем их на мини-батчи по BATCH_SIZE_PARAMETER штук.
3. Для каждого батча мы:

делаем forward (прогон через сеть),
вычисляем loss,
backward (градиенты) и
optimizer.step() (обновляем веса).

4. Когда мы перебрали все батчи (то есть прошли по всем TRAIN_SAMPLES_NUMBER примерам),
это завершает одну эпоху.
5. Затем мы снова начинаем перебор батчей с первого, и так повторяем num_epochs раз.

Если последний батч размером меньше BATCH_SIZE_PARAMETER, он тоже обрабатывается точно так же.


В данном коде при создании train_loader мы передаём shuffle=True:

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE_PARAMETER,
                          shuffle=True)
Это значит, что при каждой эпохе PyTorch будет перемешивать весь тренировочный датасет перед тем, как разбить его на батчи. В результате:

Батчи на разных эпохах будут состоять из случайного набора образцов.
В одном и том же батче в двух разных эпохах могут оказаться разные примеры.

Для валидации и теста мы обычно используем shuffle=False, чтобы оценки были воспроизводимы и последовательны:

val_loader   = DataLoader(val_dataset, batch_size=…, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=…, shuffle=False)
'''

#--------------Monte-Carlo parameters (for generation of Dataset)
#В текущем скрипте эти 3 параметра не используются, но пригодятся, если будем генерировать кривые методом МК вместо аналитической функции.
#число спинов (ячееек) в цепочке
num_cells = 20

#количество шагов «раскрутки» системы до равновесия
num_steps = 10000

#число шагов измерения после эквилибровки
equil_steps = 500


'''
                    Monte-Carlo генерация датасетапа 
                    (альтернатива датасетапу из точной аналитической формулы)
                    
Поясним значения параметров.
1) equil_steps (эквилибровка, warm-up)

Представляет число MC-шагов (итераций), которые мы выполняем до того, 
как начинать собирать измерения.

Цель — «раскрутить» систему из случайного начального состояния, 
чтобы она пришла в статистическое равновесие при заданных 𝐻,𝑇,𝐽.

Без достаточной эквилибровки вы рискуете измерять моментально сгенерированные, 
ещё не «устоявшиеся» конфигурации, что даст искажённые результаты.

2) num_steps (сборка измерений, sampling)

Это число MC-шагов после эквилибровки, в ходе которых мы фикcируем текущую намагниченность 
(и/или другие наблюдаемые) для усреднения.

Чем больше num_steps, тем точнее оценка среднего 𝑚(𝐻,𝑇), 
потому что усреднение идёт по большему числу статистически независимых конфигураций.

Обычно между измерениями делают несколько дополнительных MC-шагов, 
чтобы снизить автокорреляцию, но в простом варианте можно измерять каждый шаг.
'''



#--------------NN architecture parameters
# Размер входного вектора. В нашем случае это число точек на кривой 𝑚(𝐻), то есть 64.
input_size = len(H_values)

# Число нейронов в первом скрытом (fully-connected) слое.
# Чем больше — тем выше ёмкость сети, но и тем проще переобучиться.
hidden_size_1 = 256

# Число нейронов во втором скрытом слое.
# Обычно это меньше, чем в первом, чтобы «сжать» представление.
hidden_size_2 = 128

# Размер выходного вектора сети. У нас четыре параметра 𝐽1,𝐽2,𝐽3,𝐽4,
# которые сеть предсказывает одновременно.
output_size = 4

# Вероятность «выключения» (drop) нейрона на каждом слое при обучении.
# Dropout помогает регуляризовать модель и снижает переобучение.
DROPOUT_PARAMETER = 0.1

# Коэффициент L2-регуляризации весов (параметр weight_decay в Adam).
# Он добавляет штраф за большие значения весов и дополнительно предотвращает переобучение.
WEIGHT_DECAY_PARAMETER = 1e-4




# ============================
# Нормализация данных
# ============================

def normalize(X):
    # X.shape = (N, H) — N (например N=TRAIN_SAMPLES_NUMBER) образцов, каждый длины H_POINTS_NUMBER (число точек по H_values)
    # считаем среднее для каждой кривой
    mean = np.mean(X, axis=1, keepdims=True)
    # mean.shape = (N, 1)

    # считаем стандартное отклонение для каждой кривой
    std = np.std(X, axis=1, keepdims=True) + 1e-8
    # std.shape = (N, 1); +1e-8 — чтобы не делить на ноль

    # таким образом каждая кривая после нормализации имеет среднее 0 и дисперсию 1
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
def generate_sample(J_params):
    J1, J2, J3, J4 = J_params
    Jd = J1
    J = J3
    Jt = J4
    mag_curve = [math_utils.m(J, Jd, Jt, H, T) for H in H_values]
    # На выходе имеем 1D массив длины H_POINTS_NUMBER - точки на кривой намагниченности.
    # Элементы массива - значения намагниченности соответствующие полям из H_POINTS_NUMBER при данных J,Jd,Jt и T
    return np.array(mag_curve)



'''
                            MagnetizationDataset
Это специальный класс, который делает наши данные совместимыми с PyTorch-DataLoader’ом.

1. Наследование от torch.utils.data.Dataset
Позволяет нашему классу работать с любыми инструментами PyTorch для загрузки и перебора данных.

2.__init__(self, samples, targets)

samples — массив входных данных (матрица формы (N, H_POINTS_NUMBER) после нормализации), 
где N — число примеров (например N=TRAIN_SAMPLES_NUMBER), H_POINTS_NUMBER — длина кривой.

targets — массив целевых значений (N, 4) (четыре параметра 𝐽).
Внутри мы превращаем их в тензоры torch.Tensor с типом float32, чтобы дальше подавать в нейросеть.

3.__len__(self)
Должен возвращать число примеров в датасете. 
PyTorch вызывает его, чтобы понять, сколько всего образцов и когда остановиться.

4.__getitem__(self, idx)
По индексу idx возвращает кортеж (sample, target).

self.samples[idx] выдаёт одномерный тензор длины H_POINTS_NUMBER.

self.targets[idx] — тензор длины 4 с параметрами J.
DataLoader берёт этот кортеж и группирует их в батчи заданного размера.

Таким образом, MagnetizationDataset — это просто «обёртка» над нашими массивами, 
превращающая их в источник данных для обучения и валидации, понятный для PyTorch.
'''
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


'''
                            Net
Это простая полностью-связная (feed-forward) сеть с двумя скрытыми слоями и регуляризацией..

1. Наследование от nn.Module
Позволяет определить архитектуру и автоматически управлять параметрами (весами, градиентами).

2. Инициализация слоёв

первый линейный (fully-connected) слой: берёт вектор длины input_size (64 точек кривой 𝑚(𝐻)) 
и проецирует в пространство размерности hidden_size_1 (256 нейронов). 
    self.fc1 = nn.Linear(input_size, hidden_size_1)


батч-нормализация: стабилизирует распределение активаций после fc1, ускоряет обучение и даёт лёгкую регуляризацию.
    self.bn1 = nn.BatchNorm1d(hidden_size_1)

второй линейный слой: из 256 нейронов в 128, плюс нормализация.
    self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
    self.bn2 = nn.BatchNorm1d(hidden_size_2)

выходной слой: из 128 нейронов в четырёхмерный вектор {J1,J2,J3,J4}.
    self.fc3 = nn.Linear(hidden_size_2, output_size)
    
    
при обучении случайно «выключает» 10 % нейронов, чтобы снизить переобучение.
    Dropout(0.1)
    
нелинейная функция активации, обнуляет отрицательные значения в скрытых слоях.
Это нужно чтобы увеличить скорость обучения.
    ReLU

3.Метод forward(self, x)
Определяет, как данные проходят через сеть 

x = self.fc1(x)        # линейное преобразование
x = self.bn1(x)        # нормализация по батчу
x = self.relu(x)       # нелинейность
x = self.dropout(x)    # регуляризация
x = self.fc2(x)        # второй линейный слой
x = self.bn2(x)        # батч-нормализация
x = self.relu(x)       # ReLU
x = self.dropout(x)    # Dropout
x = self.fc3(x)        # линейный выход
return x               # без активации – регрессия


Два скрытых слоя позволяют сети выучить сложные зависимоcти между формой кривой m(H) и параметрами J.

BatchNorm + Dropout в каждом блоке улучшают стабильность обучения и уменьшают переобучение.

Отсутствие нелинейности после последнего слоя (fc3) — потому что это задача регрессии (а не классификации 0 или 1), 
и мы хотим получить любые вещественные значения на выходе. 
А ReLU бы "обрезала" отрицательные числа, а сигмоид выдала бы числа в диапазоне только [0,1] 
'''
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

    # Генерация данных
    logging.info("Generating datasets...")
    train_X, train_Y = [], []
    val_X, val_Y = [], []
    test_X, test_Y = [], []

    for _ in range(TRAIN_SAMPLES_NUMBER):
        J_params = random_J(LATTICE_TYPE)
        mag_curve = generate_sample(J_params)
        train_X.append(mag_curve)
        train_Y.append(J_params)

    for _ in range(VAL_SAMPLES_NUMBER):
        J_params = random_J(LATTICE_TYPE)
        mag_curve = generate_sample(J_params)
        val_X.append(mag_curve)
        val_Y.append(J_params)

    for _ in range(TEST_SAMPLES_NUMBER):
        J_params = random_J(LATTICE_TYPE)
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
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PARAMETER, shuffle=False)

    model = Net()
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
