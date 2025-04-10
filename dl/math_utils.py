import numpy as np
import matplotlib.pyplot as plt


T=0.25


# J1=Jd
# J2=Jd
# J3=J
# J4=Jt


J1=-1
J3=-1
J4=-1

Jd=J1
J=J3
Jt=J4

H_MIN = 0
H_MAX = 6
H_POINTS_NUMBER = 20
Hs = np.linspace(H_MIN, H_MAX, H_POINTS_NUMBER)  # H_POINTS_NUMBER точек для кривой M(H) от H_MIN до H_MAX





def R11(x, y, z, hp):
    return 2 * np.exp(x + 2 * hp) * np.cosh(z + hp) + 2 * np.exp(-x) * np.cosh(hp)


def R12(x, y, z, hp):
    return 2 * np.exp(y + hp) * np.cosh(z + hp) + 2 * np.exp(-y - hp) * np.cosh(hp)


def R21(x, y, z, hp):
    return 2 * np.exp(y - hp) * np.cosh(z - hp) + 2 * np.exp(-y + hp) * np.cosh(hp)


def R22(x, y, z, hp):
    return 2 * np.exp(x - 2 * hp) * np.cosh(z - hp) + 2 * np.exp(-x) * np.cosh(hp)


def x(J, Jd, Jt, T):
    return (J + Jt) / T


def y(J, Jd, Jt, T):
    return (J - Jt) / T


def z(J, Jd, Jt, T):
    return 2 * Jd / T


def hp(h, T):
    return h / T


def create_matrix(R11, R12, R21, R22):
    # Создаем матрицу 2x2 из 4 элементов
    matrixR = np.array([[R11, R12], [R21, R22]])
    return matrixR


def matrix_trace_el(R11, R12, R21, R22):
    # Возвращаем след матрицы
    return R11 + R22


def matrix_trace(matrix):
    # Возвращаем след матрицы
    return np.trace(matrix)


def matrix_det_el(R11, R12, R21, R22):
    # Возвращаем след матрицы
    return R11 * R22 - R12 * R21


def matrix_det(matrix):
    # Возвращаем след матрицы
    return np.linalg.det(matrix)


def lam(J, Jd, Jt, h, T):
    var_x = x(J, Jd, Jt, T)
    var_y = y(J, Jd, Jt, T)
    var_z = z(J, Jd, Jt, T)
    var_hp = hp(h, T)

    elementR11 = R11(var_x, var_y, var_z, var_hp)
    elementR12 = R12(var_x, var_y, var_z, var_hp)
    elementR21 = R21(var_x, var_y, var_z, var_hp)
    elementR22 = R22(var_x, var_y, var_z, var_hp)

    matrixR = create_matrix(elementR11, elementR12, elementR21, elementR22)

    trace = matrix_trace(matrixR)
    det = matrix_det(matrixR)

    return 1 / 2 * (trace + np.sqrt((trace) ** 2 - 4 * det))


def m(J, Jd, Jt, h, T):
    delta = 1e-5
    dlam_dh = (lam(J, Jd, Jt, h + delta, T) - lam(J, Jd, Jt, h - delta, T)) / (2 * delta)
    return T / (3 * lam(J, Jd, Jt, h, T)) * dlam_dh



def main():
    ms = []

    for H in Hs:
        var_m = m(J, Jd, Jt, H, T)
        ms.append(var_m)

    plt.plot(Hs,ms)

    plt.xlabel('h')
    plt.ylabel('m')
    plt.legend(facecolor='white', framealpha=1)
    plt.savefig("m_h_teor.png")
    plt.show()


if __name__ == '__main__':
    main()
