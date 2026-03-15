import csv
import numpy as np
import matplotlib.pyplot as plt


# --- 1. ЗЧИТУВАННЯ ДАНИХ  ---
def read_data(filename):
    x, y = [], []
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x.append(float(row['Tasks']))
                y.append(float(row['Cost']))
        return np.array(x), np.array(y)
    except FileNotFoundError:
        print(f"Помилка: Файл {filename} не знайдено!")
        return None, None


# --- 2. РОЗДІЛЕНІ РІЗНИЦІ ---
def get_divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef[0, :]


# --- 3. МНОГОЧЛЕН НЬЮТОНА---
def newton_poly(coef, x_nodes, x_val):
    n = len(x_nodes)
    res = coef[0]
    product = 1.0
    for i in range(1, n):
        product *= (x_val - x_nodes[i - 1])
        res += coef[i] * product
    return res


# --- ГОЛОВНИЙ БЛОК ВИКОНАННЯ ---
if __name__ == "__main__":
    # Завантажуємо базові дані (5 точок)
    x_nodes, y_nodes = read_data("data.csv")

    if x_nodes is not None:
        target_x = 15000
        # Створюємо еталонну функцію на основі 5 точок для розрахунку похибки [cite: 159]
        coeffs_ref = get_divided_diff(x_nodes, y_nodes)

        # Список кількостей вузлів для дослідження
        n_values = [5, 10, 20]

        # Налаштування трьох окремих графіків похибок
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        plt.subplots_adjust(hspace=0.4)

        print(f"{'n вузлів':<10} | {'Прогноз для ' + str(target_x):<15}")
        print("-" * 30)

        x_range = np.linspace(min(x_nodes), max(x_nodes), 500)
        y_ref = [newton_poly(coeffs_ref, x_nodes, xi) for xi in x_range]

        for i, n in enumerate(n_values):
            # Генеруємо n вузлів для дослідження
            x_study = np.linspace(min(x_nodes), max(x_nodes), n)
            y_study = [newton_poly(coeffs_ref, x_nodes, xi) for xi in x_study]

            # Будуємо модель для поточної кількості вузлів
            coeffs_n = get_divided_diff(x_study, y_study)
            prediction = newton_poly(coeffs_n, x_study, target_x)
            print(f"{n:<10} | {prediction:<15.4f} $")

            # Обчислюємо похибку epsilon(x) = |f(x) - Nn(x)|
            y_interp = [newton_poly(coeffs_n, x_study, xi) for xi in x_range]
            error = np.abs(np.array(y_ref) - np.array(y_interp))

            # Малюємо графік похибки для кожного n # Малюємо графік похибки для кожного n # Малюємо графік похибки для кожного n
            axes[i].plot(x_range, error, color='red', label=f'Похибка ε(x) при n={n}')
            axes[i].fill_between(x_range, error, color='red', alpha=0.1)
            axes[i].set_title(f'Графік похибки інтерполяції (n = {n} вузлів)')
            axes[i].set_xlabel('Кількість задач (Tasks)')
            axes[i].set_ylabel('Абсолютна похибка')
            axes[i].grid(True)
            axes[i].legend()

        plt.show()

        # Окремий графік самої моделі (для звіту)
        plt.figure(figsize=(10, 6))
        plt.plot(x_range, y_ref, 'b-', label='Інтерполяційна крива (n=5)')
        plt.scatter(x_nodes, y_nodes, color='black', label='Вузли з таблиці')
        plt.axvline(x=target_x, color='green', linestyle='--', label=f'Прогноз x={target_x}')
        plt.title('Модель прогнозування вартості (Варіант 4)')
        plt.legend()
        plt.grid(True)
        plt.show()