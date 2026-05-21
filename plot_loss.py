import matplotlib.pyplot as plt
import csv

def plot_loss(filename="loss_history.csv"):
    epochs = []
    losses = []

    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            # Пропускаем первую строку с заголовками "epoch,loss"
            next(reader)

            for row in reader:
                if len(row) == 2:
                    epochs.append(int(row[0]))
                    losses.append(float(row[1]))
    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден. Убедись, что программа на C отработала успешно!")
        return

    # Настраиваем размер и внешний вид графика
    plt.figure(figsize=(8, 5))

    # Строим линию (маркер 'o' добавит точки на каждый шаг, color='b' сделает линию синей)
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', markersize=4, label='Cross-Entropy Loss')

    # Добавляем подписи
    plt.title('График падения ошибки при обучении нейросети')
    plt.xlabel('Эпоха (Epoch)')
    plt.ylabel('Значение ошибки (Loss)')

    # Включаем сетку для удобства чтения
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Показываем результат
    plt.tight_layout()
    plt.savefig("loss_graph.png")     # для plot_loss.py

if __name__ == "__main__":
    plot_loss()