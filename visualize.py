import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(filename="heatmap.txt"):
    activations = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Пропускаем текстовые заголовки и пустые строки
                if "Activations:" in line or line.strip() == "":
                    continue
                # Считываем числа из строки
                values = [float(x) for x in line.split()]
                activations.extend(values)
    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден. Сначала запусте программу на C!")
        return

    # Преобразуем список в массив NumPy
    activations = np.array(activations)

    # Проверяем, что у нас ровно 256 значений (как мы настроили в config.txt)
    if len(activations) == 256:
        # Сворачиваем плоский массив 256 в матрицу 16x16
        matrix = activations.reshape((16, 16))

        # Настраиваем внешний вид графика
        plt.figure(figsize=(8, 6))

        # Строим теплокарту (цветовая схема 'magma' отлично смотрится для нейросетей)
        plt.imshow(matrix, cmap='magma', interpolation='nearest')

        plt.colorbar(label='Уровень активации нейрона после ReLU')
        plt.title('Теплокарта скрытого слоя 16x16 нейронов')
        plt.xlabel('Координата X')
        plt.ylabel('Координата Y')

        # Показываем график
        plt.tight_layout()
        plt.savefig("heatmap_result.png") # для visualize.py
    else:
        print(f"Ошибка: ожидалось 256 значений для матрицы 16x16, но получено {len(activations)}.")

if __name__ == "__main__":
    plot_heatmap()