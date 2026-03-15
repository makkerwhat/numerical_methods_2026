import numpy as np
import matplotlib.pyplot as plt
import csv


data = [
    (1,  -2),(2,  0),(3, 5),(4, 10),(5, 15),(6, 20),(7, 23),(8, 22),(9, 17),(10, 10),(11, 15),(12, 0),(13,-10),(14,3),(15, 7),
    (16, 13),(17, 19),(18, 20),(19, 22),(20, 21),(21, 18),(22,15),(23, 10),(24, 3)
]

csv_filename = "data.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "t_ms"])
    writer.writerows(data)

print(f"Дані збережено у {csv_filename}")

# зчитування з CSV
def read_csv(filename):
    x_vals, y_vals = [], []
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_vals.append(float(row["n"]))
            y_vals.append(float(row["t_ms"]))
    return np.array(x_vals), np.array(y_vals)

x_all, y_all = read_csv(csv_filename)
print(f"\nЗчитано {len(x_all)} вузлів з файлу:")
for xi, yi in zip(x_all, y_all):
    print(f"month:{int(xi)}  temperature:{yi}°C ")


def form_matrix(x, m):
     A = np.zeros((m + 1, m + 1))
     for i in range(m + 1):
        for j in range(m + 1):
         A[i,j] = np.sum(x ** (i + j))
     return A


def form_vector(x, y, m):
    b = np.zeros(m + 1)
    for i in range(m + 1):
        b[i] = np.sum(y * x ** i)
    return b


m = 3  # ступінь полінома
A = form_matrix(x_all, m)
b = form_vector(x_all, y_all, m)

print("Матриця A:")
print(A)
print("\nВектор b:")
print(b)

# Розв'язання системи для знаходження коефіцієнтів полінома
coeffs = np.linalg.solve(A, b)
print("\nКоефіцієнти полінома:", coeffs)


def gauss_solve(A, b):
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = len(b)

    # Прямий хід з вибором головного елемента
    for k in range(n):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        # Елімінація
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]

    # Зворотній хід
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i+1:] * x_sol[i+1:])) / A[i, i]

    return x_sol


def polynomial(x, coef):
    y_poly = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coef):
        y_poly += c * x ** i
    return y_poly


def variance(y_true, y_approx):
    return np.mean((y_true - y_approx) ** 2)


# --- Тригонометричний прогноз (реалістичний) ---
def form_trig_matrix(x, n_harmonics, T=12):
    cols = [np.ones_like(x)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * x / T))
        cols.append(np.sin(2 * np.pi * k * x / T))
    return np.column_stack(cols)

def trig_solve(x, y, n_harmonics, T=12):
    A = form_trig_matrix(x, n_harmonics, T)
    return gauss_solve(A.T @ A, A.T @ y)

def trig_polynomial(x, coef, T=12):
    return form_trig_matrix(x, (len(coef)-1)//2, T) @ coef


max_degree = 4
variances = []

for m in range(1, max_degree + 1):
    A = form_matrix(x_all, m)
    b_vec = form_vector(x_all, y_all, m)
    coef = gauss_solve(A, b_vec)
    y_approx = polynomial(x_all, coef)
    var = variance(y_all, y_approx)
    variances.append(var)

optimal_m = np.argmin(variances) + 1

# -------------------------------
# 4. Побудова апроксимації
# -------------------------------
A = form_matrix(x_all, optimal_m)
b_vec = form_vector(x_all, y_all, optimal_m)
coef = gauss_solve(A, b_vec)
y_approx = polynomial(x_all, coef)

# -------------------------------
# 5. Прогноз на наступні 3 місяці
# -------------------------------
coef_trig = trig_solve(x_all, y_all, 3)
x_future = np.array([25, 26, 27])
y_future_trig = trig_polynomial(x_future, coef_trig)
y_future = polynomial(x_future, coef)
# -------------------------------
# 6. Похибка апроксимації
# -------------------------------
error = y_all - y_approx
# -------------------------------
# 7. Вивід результатів
# -------------------------------
print("Дисперсії для різних ступенів полінома:")
for m, var in enumerate(variances, start=1):
    marker = " <-- оптимальний" if m == optimal_m else ""
    print(f"  Ступінь {m}: MSE = {var:.4f}{marker}")

print(f"\nОптимальний ступінь полінома: {optimal_m}")
print(f"Коефіцієнти полінома: {coef}")



print("\nПрогноз на наступні 3 місяці:")
print(f"{'Місяць':>8} {'Поліном':>12} {'Тригонометр':>14}")
print("-" * 36)
for xi, yp, yt in zip(x_future, y_future, y_future_trig):
    print(f"{int(xi):>8} {yp:>11.2f}°C {yt:>12.2f}°C")



fig, axes = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle(f"Апроксимація температури (поліном m={optimal_m})", fontsize=14)


# Графік 1: Фактичні дані + апроксимація + прогноз
ax1 = axes[0]
ax1.plot(x_all, y_all, 'bo-', label='Фактичні дані', markersize=5)
ax1.plot(x_all, y_approx, 'r-', label=f'Поліном (m={optimal_m})', linewidth=2)
ax1.plot(x_future, y_future, 'g^--', label='Прогноз', markersize=8)
ax1.set_xlabel("Місяць")
ax1.set_ylabel("Температура (°C)")
ax1.set_title("Фактичні дані та апроксимація")
ax1.legend()
ax1.grid(True)

# Графік 2: Похибка апроксимації
ax2 = axes[1]
ax2.bar(x_all, error, color='orange', alpha=0.7, label='Похибка')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_xlabel("Місяць")
ax2.set_ylabel("Похибка (°C)")
ax2.set_title("Похибка апроксимації (фактичне − апроксимація)")
ax2.legend()
ax2.grid(True, axis='y')

# Графік 3: MSE для різних ступенів
ax3 = axes[2]
degrees = list(range(1, max_degree + 1))
bars = ax3.bar(degrees, variances, color='steelblue', alpha=0.8)
bars[optimal_m - 1].set_color('crimson')
ax3.set_xlabel("Ступінь полінома")
ax3.set_ylabel("MSE")
ax3.set_title("Дисперсія для різних ступенів (червоний = оптимальний)")
ax3.set_xticks(degrees)
ax3.grid(True, axis='y')

plt.tight_layout()
plt.show()
