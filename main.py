import numpy as np

# 1. Визначаємо функцію вологості M(t)
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

# Точна аналітична похідна для перевірки
def M_prime_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# Функція для чисельного диференціювання (центральна різниця)
def central_diff(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)

# --- ПАРАМЕТРИ ЗАВДАННЯ ---
t0 = 1.0  # Точка, в якій рахуємо
exact_val = M_prime_exact(t0)

print(f"--- Крок 1: Точне значення ---")
print(f"M'({t0}) точно = {exact_val:.10f}\n")

# --- Крок 2: Пошук оптимального h ---
print(f"--- Крок 2: Пошук оптимального h ---")
h_values = [10**i for i in range(-20, 4)] # від 10^-20 до 10^3
best_h = h_values[0]
min_error = float('inf')

for h in h_values:
    try:
        approx = central_diff(t0, h)
        error = abs(approx - exact_val)
        if error < min_error:
            min_error = error
            best_h = h
    except ZeroDivisionError:
        continue

print(f"Найкращий крок h0 = {best_h:.1e}")
print(f"Мінімальна похибка R0 = {min_error:.2e}\n")

# --- Крок 4-6: Метод Рунге-Ромберга ---
print(f"--- Крок 4-6: Метод Рунге-Ромберга ---")
h_lab = 1e-3 # Крок за умовою
D_h = central_diff(t0, h_lab)      #
D_2h = central_diff(t0, 2 * h_lab)  #

R1 = abs(D_h - exact_val) # Похибка при h

# Формула Рунге-Ромберга [cite: 196]
D_RR = D_h + (D_h - D_2h) / 3
R2 = abs(D_RR - exact_val)

print(f"D(h) = {D_h:.10f}, Похибка R1 = {R1:.2e}")
print(f"D_RR = {D_RR:.10f}, Похибка R2 = {R2:.2e}")
print(f"Точність покращилась у {R1/R2:.1f} разів\n")

# --- Крок 7: Метод Ейткена ---
print(f"--- Крок 7: Метод Ейткена ---")
D_4h = central_diff(t0, 4 * h_lab)

# Уточнене значення за Ейткеном
numerator = (D_2h**2) - (D_4h * D_h)
denominator = 2 * D_2h - (D_4h + D_h)
D_E = numerator / denominator

# Порядок точності p
p = np.log(abs((D_4h - D_2h) / (D_2h - D_h))) / np.log(2)
R3 = abs(D_E - exact_val)

print(f"D_E (Ейткен) = {D_E:.10f}, Похибка R3 = {R3:.2e}")
print(f"Визначений порядок точності p = {p:.2f}")