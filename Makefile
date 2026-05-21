# Компилятор C и флаги
CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LIBS = -lm -pthread

# Интерпретатор Python (если не сработает, поменяй слово python на py)
PYTHON = python

# Файлы проекта на C
TARGET = neural_net_app.exe
SRCS = main.c config_parser.c data_loader.c neural_net.c utils.c
OBJS = $(SRCS:.c=.o)

# Основная сборка C-кода
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Запуск нейросети
run: all
	.\$(TARGET)

# --- НОВЫЕ КОМАНДЫ ДЛЯ PYTHON ---

# Визуализация теплокарты (матрица 16x16)
visualize:
	$(PYTHON) visualize.py

# Отрисовка графика падения ошибки
plot:
	$(PYTHON) plot_loss.py

run_all: run visualize plot

# Очистка мусора
clean:
	del /Q /F *.o $(TARGET) loss_history.csv heatmap.txt