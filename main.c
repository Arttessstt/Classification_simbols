#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neural_net.h"
#include "config_parser.h"
#include "data_loader.h"
#include "utils.h"

/**
 * @brief Генерация синтетических данных на случай, если файл с MNIST не найден.
 * Помогает проверить работоспособность сети без больших внешних файлов.
 */
Sample* generate_synthetic_data(int *num_samples) {
    *num_samples = 20;
    Sample *samples = (Sample*)malloc(*num_samples * sizeof(Sample));
    if (!samples) return NULL;

    for (int i = 0; i < *num_samples; i++) {
        samples[i].pixels = (double*)malloc(784 * sizeof(double));
        samples[i].label = i % 10;

        for (int j = 0; j < 784; j++) {
            // Создаем простые паттерны для синтетических данных
            if (j % 10 == samples[i].label) {
                samples[i].pixels[j] = 0.9;
            } else {
                samples[i].pixels[j] = 0.1;
            }
        }
    }
    return samples;
}

int main() {
    printf("--- Инициализация нейронной сети ---\n");

    // Загрузка архитектуры из файла конфигурации
    Network *net = load_network_from_config("config.txt");
    if (!net) {
        printf("Ошибка: Не удалось загрузить config.txt. Проверьте формат файла.\n");
        return 1;
    }

    int num_samples = 0;
    // Попытка загрузки реального датасета MNIST
    Sample *train_data = load_csv("mnist_train.csv", &num_samples);

    if (!train_data) {
        printf("Файл mnist_train.csv не найден. Используются СИНТЕТИЧЕСКИЕ данные.\n");
        train_data = generate_synthetic_data(&num_samples);
    } else {
        printf("Датасет загружен успешно. Количество примеров: %d\n", num_samples);
    }

    // Открытие файла для записи истории обучения
    FILE *loss_file = fopen("loss_history.csv", "w");
    if (loss_file) fprintf(loss_file, "epoch,loss\n");

    printf("Начало процесса обучения (%d эпох)...\n\n", 100);

    int epochs = 100;
    double augmented_buffer[784];

    for (int epoch = 1; epoch <= epochs; epoch++) {
        double total_loss = 0;

        for (int i = 0; i < num_samples; i++) {
            // Копируем данные во временный буфер для преобразований
            memcpy(augmented_buffer, train_data[i].pixels, 784 * sizeof(double));

            // Аугментация (повороты) и добавление шума согласно ТЗ
            augment_image(augmented_buffer);
            add_noise(augmented_buffer, 784);

            // Прямой проход (вычисление выхода)
            double *output = forward_propagation(net, augmented_buffer);
            
            // Вычисление функции потерь
            total_loss += cross_entropy_loss(output, train_data[i].label);

            // Обратный проход и обновление весов
            backward_propagation(net, train_data[i].label);
            update_weights(net);
        }

        double avg_loss = total_loss / num_samples;
        if (loss_file) fprintf(loss_file, "%d,%f\n", epoch, avg_loss);

        // Вывод прогресса каждые 10 эпох
        if (epoch % 10 == 0) {
            printf("Эпоха %d | Средняя ошибка (Loss): %.4f\n", epoch, avg_loss);
        }
    }

    if (loss_file) {
        fclose(loss_file);
        printf("\nИстория ошибок сохранена в loss_history.csv!\n");
    }

    // Демонстрация предсказаний для первого примера
    printf("\n--- Тестовый запуск на первом примере (Ожидалось: %d) ---\n", train_data[0].label);
    double *probs = forward_propagation(net, train_data[0].pixels);
    for(int i = 0; i < net->layers[net->num_layers - 1].out_nodes; i++) {
        printf("Класс %d: %6.2f%%\n", i, probs[i] * 100.0);
    }

    // Сохранение визуализации скрытого слоя
    save_heatmap(net, "heatmap.txt");
    printf("\nАктивации скрытого слоя сохранены в heatmap.txt!\n");

    // Освобождение ресурсов
    free_samples(train_data, num_samples);
    free_network(net);
    
    printf("\nРабота успешно завершена.\n");
    return 0;
}
