import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# 1. Функція
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


# Межі інтегрування
a, b = 0, 24

# 2. Знаходимо точне значення інтегралу I0 через вбудовану функцію
I0, _ = quad(f, a, b)
print(f"Точне значення інтегралу I0: {I0:.15f}\n")


# 3. Функція для обчислення інтегралу складовою формулою Сімпсона
def simpson_composite(f, a, b, N):
    if N % 2 != 0: N += 1  # Кількість розбиттів має бути парною
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    # Формула Сімпсона: h/3 * (f0 + 4*сума_непарних + 2*сума_парних + fN)
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3) * S


# 4. Дослідження залежності точності від N та пошук N_opt
target_eps = 1e-12
N_range = np.arange(10, 1001, 10)
simpson_errors = [abs(simpson_composite(f, a, b, n) - I0) for n in N_range]

N_opt = 10
while abs(simpson_composite(f, a, b, N_opt) - I0) > target_eps:
    N_opt += 2
    if N_opt > 5000: break  # Запобіжник

print(f"Оптимальне N_opt для точності 1e-12: {N_opt}")
print(f"Отримана точність eps_opt: {abs(simpson_composite(f, a, b, N_opt) - I0):.2e}\n")

# 5. Обчислення N0 (N_opt/10, кратне 8)
N0 = (N_opt // 10)
N0 = (N0 // 8 + 1) * 8 if N0 % 8 != 0 else max(8, N0)
eps0 = abs(simpson_composite(f, a, b, N0) - I0)
print(f"Вибране N0 = {N0}, Похибка eps0 = {eps0:.2e}")

# 6. Метод Рунге-Ромберга для N0
I_N0 = simpson_composite(f, a, b, N0)
I_N0_half = simpson_composite(f, a, b, N0 // 2)

I_R = I_N0 + (I_N0 - I_N0_half) / 15
epsR = abs(I_R - I0)
print(f"Метод Рунге-Ромберга (I_R): {I_R:.15f}, Похибка epsR: {epsR:.2e}")

# 7. Метод Ейткена для N0
I1 = I_N0  # для N0
I2 = I_N0_half  # для N0/2
I3 = simpson_composite(f, a, b, N0 // 4)  # для N0/4
I_E = (I2 ** 2 - I1 * I3) / (2 * I2 - (I1 + I3))
p_aitken = (1 / np.log(2)) * np.log(abs((I3 - I2) / (I2 - I1)))
epsE = abs(I_E - I0)
print(f"Метод Ейткена (I_E): {I_E:.15f}, Похибка epsE: {epsE:.2e}, Порядок p: {p_aitken:.2f}\n")


# 9. Адаптивний алгоритм із підрахунком викликів функції
def adaptive_simpson(f, a, b, eps, func_calls):
    def step(a, b, eps, whole):
        mid = (a + b) / 2
        h = b - a
        # Обчислюємо ліву та праву частини
        left_val = (h / 12) * (f(a) + 4 * f((a + mid) / 2) + f(mid))
        right_val = (h / 12) * (f(mid) + 4 * f((mid + b) / 2) + f(b))
        func_calls[0] += 4  # Додаємо нові точки

        if abs(left_val + right_val - whole) <= 15 * eps:
            return left_val + right_val + (left_val + right_val - whole) / 15
        return step(a, mid, eps / 2, left_val) + step(mid, b, eps / 2, right_val)

    func_calls[0] = 3  # a, b, mid
    initial_whole = (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))
    return step(a, b, eps, initial_whole)


# Дослідження адаптивного алгоритму для різних толерантностей
tols = np.logspace(-2, -10, 9)
adapt_calls = []
adapt_errors = []

for t in tols:
    calls = [0]
    res = adaptive_simpson(f, a, b, t, calls)
    adapt_calls.append(calls[0])
    adapt_errors.append(abs(res - I0))

# --- ПОБУДОВА ГРАФІКІВ ---

plt.figure(figsize=(12, 10))

# Графік 1: Похибка складової формули Сімпсона (Завдання 4)
plt.subplot(2, 1, 1)
plt.semilogy(N_range, simpson_errors, label='Метод Сімпсона |I(N)-I0|')
plt.axhline(target_eps, color='r', linestyle='--', label='Задана точність 1e-12')
plt.scatter([N0], [eps0], color='orange', zorder=5, label=f'N0={N0}')
plt.title('Залежність похибки від числа розбиттів (Метод Сімпсона)')
plt.xlabel('N')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True, which="both", ls="--")

# Графік 2: Адаптивний алгоритм (Завдання 9)
plt.subplot(2, 1, 2)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.loglog(tols, adapt_errors, 'g-o', label='Фактична похибка')
ax2.semilogx(tols, adapt_calls, 'b-s', label='Кількість обчислень f(x)')
ax1.set_xlabel('Задана точність (Tolerance)')
ax1.set_ylabel('Похибка', color='g')
ax2.set_ylabel('Кількість викликів f(x)', color='b')
plt.title('Ефективність адаптивного алгоритму')
ax1.grid(True, which="both")

plt.tight_layout()
plt.show()