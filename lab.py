import requests
import numpy as np
import matplotlib.pyplot as plt

url = (
    "https://api.open-elevation.com/api/v1/lookup?locations="
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)

response = requests.get(url)
data = response.json()

results = data["results"]

print("№ | Latitude | Longitude | Elevation (m)")
print("----------------------------------------")

n = len(results)
print("Кількість вузлів:", n)
print("\nТабуляція вузлів:")
print("№ | Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
 print(f"{i:2d} | {point['latitude']:.6f} | "
 f"{point['longitude']:.6f} | "
 f"{point['elevation']:.2f}")


# --- КРОК 4: Розрахунок відстані ---
# Земля кругла, тому просто віднімати координати не можна.

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# Складаємо списки координат та висот
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

# Рахуємо кумулятивну (накопичену) відстань
distances = [0]  # Стартуємо з 0 метрів
for i in range(1, n):
    # Рахуємо шлях від попередньої точки до поточної
    d = haversine(*coords[i - 1], *coords[i])
    # Додаємо цей шлях до загальної суми
    distances.append(distances[-1] + d)

print("\nГотові дані для графіка (Відстань | Висота):")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} m | {elevations[i]:8.2f} m")

# --- КРОК 3.1: Запис результатів табуляції у файл ---

with open("tabulation_results.txt", "w", encoding="utf-8") as file:
    file.write("Результати табуляції (вузли маршруту)\n")
    file.write("-" * 60 + "\n")
    file.write(f"{'№':<3} | {'Latitude':<10} | {'Longitude':<10} | {'Elevation':<10} | {'Distance':<10}\n")
    file.write("-" * 60 + "\n")

    for i in range(n):
        line = f"{i:<3} | {coords[i][0]:.6f} | {coords[i][1]:.6f} | {elevations[i]:10.2f} | {distances[i]:10.2f}\n"
        file.write(line)

print("\nРезультати табуляції збережено у файл 'tabulation_results.txt'")

# --- КРОК 6, 7, 8: Коефіцієнти та метод прогонки ---

def get_spline_coefficients(x, y):
    n_points = len(x)
    h = np.diff(x)  # Кроки hi = xi - xi-1 [cite: 16]

    # Готуємо систему рівнянь для Ci (трьохдіагональна матриця) [cite: 47, 52]
    # alpha[i]*c[i-1] + beta[i]*c[i] + gamma[i]*c[i+1] = delta[i]
    alpha = np.zeros(n_points)
    beta = np.ones(n_points)
    gamma = np.zeros(n_points)
    delta = np.zeros(n_points)

    # Задаємо умови для вільного сплайна: c1 = 0, cn = 0 [cite: 31, 45, 48]
    for i in range(1, n_points - 1):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])  # [cite: 43]

    # Пряма прогонка [cite: 53, 63, 64]
    A = np.zeros(n_points)
    B = np.zeros(n_points)
    for i in range(1, n_points):
        m = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / m
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / m

    # Зворотна прогонка [cite: 66, 71]
    c = np.zeros(n_points)
    c[n_points - 1] = B[n_points - 1]
    for i in range(n_points - 2, -1, -1):
        c[i] = A[i] * c[i + 1] + B[i]

    # Тепер рахуємо інші коефіцієнти: a, b, d [cite: 143]
    a = y[:-1]  # ai = yi-1 [cite: 36]
    d = np.zeros(n_points - 1)
    b = np.zeros(n_points - 1)

    for i in range(n_points - 1):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])  # [cite: 37]
        b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 3) * (c[i + 1] + 2 * c[i])  # [cite: 38]

    return a, b, c[:-1], d


# Виконуємо розрахунок
a, b, c, d = get_spline_coefficients(distances, elevations)

# --- ВИВІД КОЕФІЦІЄНТІВ У КОНСОЛЬ (згідно з п. 7, 8, 9) ---

print("\n" + "="*50)
print("КОЕФІЦІЄНТИ КУБІЧНИХ СПЛАЙНІВ")
print("="*50)
print(f"{'№':<3} | {'a_i':<10} | {'b_i':<10} | {'c_i':<10} | {'d_i':<10}")
print("-" * 50)

# Проходимо по кожному інтервалу (їх на 1 менше, ніж точок)
for i in range(len(a)):
    print(f"{i+1:<3} | {a[i]:10.2f} | {b[i]:10.2f} | {c[i]:10.2f} | {d[i]:10.6f}")

print("="*50)

print("\nКоефіцієнти сплайна (a, b, c, d) успішно обчислені.")

# --- КРОК 10: Побудова графіка ---

# Створюємо масив точок для плавної лінії (від 0 до кінцевої відстані)
x_smooth = np.linspace(distances[0], distances[-1], 300)
y_smooth = []

for x_val in x_smooth:
    # Визначаємо, в який інтервал [x_i, x_i+1] потрапляє поточна точка
    idx = 0
    for i in range(len(distances) - 1):
        if distances[i] <= x_val <= distances[i + 1]:
            idx = i
            break

    # Формула кубічного сплайна: S = a + b*dx + c*dx^2 + d*dx^3
    dx = x_val - distances[idx]
    val = a[idx] + b[idx] * dx + c[idx] * (dx ** 2) + d[idx] * (dx ** 3)
    y_smooth.append(val)

# Малюємо
plt.figure(figsize=(10, 6))
plt.plot(distances, elevations, 'ro', label='Вузли (дані)')
plt.plot(x_smooth, y_smooth, 'b-', label='Кубічний сплайн')
plt.title('Профіль висоти маршруту (Кубічний сплайн)')
plt.xlabel('Відстань (м)')
plt.ylabel('Висота (м)')
plt.grid(True)
plt.legend()
plt.show()

total_ascent = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n))
print(f"Загальна довжина маршруту: {distances[-1]:.2f} м")
print(f"Сумарний набір висоти: {total_ascent:.2f} м")

# --- КРОК 12: Окремий графік похибки (згідно з методичкою) ---

# 1. Рахуємо значення сплайна точно у вузлах
y_approx_at_nodes = []
for x_val in distances:
    idx = 0
    for i in range(len(distances) - 1):
        if distances[i] <= x_val <= distances[i+1]:
            idx = i
            break
    dx = x_val - distances[idx]
    val = a[idx] + b[idx]*dx + c[idx]*(dx**2) + d[idx]*(dx**3)
    y_approx_at_nodes.append(val)

# 2. Обчислюємо похибку epsilon = |y - y_approx|
errors = np.abs(np.array(elevations) - np.array(y_approx_at_nodes))

# 3. Будуємо графік похибки
plt.figure(figsize=(10, 4))
plt.plot(distances, errors, 'g-o', label='Похибка ε = |y - y_набл|')
plt.title('Графік похибки наближення (п. 12)')
plt.xlabel('Відстань (м)')
plt.ylabel('Абсолютна похибка (м)')
plt.grid(True)
plt.legend()
plt.show()

print(f"\nМаксимальна похибка у вузлах: {np.max(errors):.2e} м")