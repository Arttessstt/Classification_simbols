#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Загружает данные из CSV-файла.
 * Формат файла: label, pixel1, pixel2, ..., pixel784
 * 
 * @param filename Путь к файлу
 * @param num_samples Указатель для возврата количества считанных строк
 * @return Массив структур Sample или NULL при ошибке
 */
Sample* load_csv(const char *filename, int *num_samples) {
    FILE *file = fopen(filename, "r");
    if (!file) return NULL;

    // Начальная емкость массива
    int capacity = 10000;
    Sample *samples = (Sample*)malloc(capacity * sizeof(Sample));
    if (!samples) {
        fclose(file);
        return NULL;
    }

    char line[10000];
    int count = 0;

    // Чтение файла построчно
    while (fgets(line, sizeof(line), file)) {
        // Динамическое расширение массива при необходимости
        if (count >= capacity) {
            capacity *= 2;
            Sample *temp = realloc(samples, capacity * sizeof(Sample));
            if (!temp) {
                // В случае ошибки возвращаем то, что успели прочитать
                break;
            }
            samples = temp;
        }

        samples[count].pixels = (double*)malloc(784 * sizeof(double));
        if (!samples[count].pixels) break;

        // Парсинг метки класса
        char *token = strtok(line, ",");
        if (!token) {
            free(samples[count].pixels);
            continue;
        }
        samples[count].label = atoi(token);

        // Парсинг пикселей (интенсивность 0-255)
        for (int i = 0; i < 784; i++) {
            token = strtok(NULL, ",");
            if (token != NULL) {
                // Нормализация в диапазон [0, 1] для лучшей сходимости
                samples[count].pixels[i] = atof(token) / 255.0;
            } else {
                samples[count].pixels[i] = 0.0;
            }
        }
        count++;
    }
    fclose(file);
    *num_samples = count;
    return samples;
}

/**
 * @brief Добавляет случайный шум во входные данные для повышения устойчивости сети.
 * Согласно ТЗ: изменяется 5-10% пикселей на ±0.2.
 */
void add_noise(double *pixels, int size) {
    // Выбираем процент зашумления от 5% до 10%
    double noise_ratio = 0.05 + ((double)rand() / RAND_MAX) * 0.05;
    
    for (int i = 0; i < size; i++) {
        if (((double)rand() / RAND_MAX) < noise_ratio) {
            double noise = (((double)rand() / RAND_MAX) > 0.5) ? 0.2 : -0.2;
            pixels[i] += noise;
            
            // Ограничение значений диапазоном [0, 1]
            if (pixels[i] < 0) pixels[i] = 0.0;
            if (pixels[i] > 1) pixels[i] = 1.0;
        }
    }
}

/**
 * @brief Получает значение пикселя с проверкой границ (используется для аугментации).
 */
double get_pixel(double *pixels, int x, int y) {
    if (x < 0 || x >= 28 || y < 0 || y >= 28) return 0.0;
    return pixels[y * 28 + x];
}

/**
 * @brief Поворачивает изображение на случайный угол для аугментации данных.
 * Использует билинейную интерполяцию для сглаживания.
 */
void augment_image(double *pixels) {
    // Применяем аугментацию только в 50% случаев
    if (((double)rand() / RAND_MAX) > 0.5) return;

    double temp[784] = {0.0};
    double max_angle = 12.0; // Максимальный угол поворота в градусах
    double angle_deg = ((double)rand() / RAND_MAX) * (2.0 * max_angle) - max_angle;
    double angle_rad = angle_deg * M_PI / 180.0;

    double cos_a = cos(angle_rad);
    double sin_a = sin(angle_rad);

    double cx = 13.5; // Центр изображения 28x28
    double cy = 13.5;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            double dx = x - cx;
            double dy = y - cy;

            // Обратное преобразование координат
            double source_x = cx + (dx * cos_a + dy * sin_a);
            double source_y = cy + (-dx * sin_a + dy * cos_a);

            int x1 = (int)floor(source_x);
            int y1 = (int)floor(source_y);
            int x2 = x1 + 1;
            int y2 = y1 + 1;

            if (x1 >= 0 && x2 < 28 && y1 >= 0 && y2 < 28) {
                double x_diff = source_x - x1;
                double y_diff = source_y - y1;

                // Билинейная интерполяция
                double p11 = get_pixel(pixels, x1, y1);
                double p12 = get_pixel(pixels, x2, y1);
                double p21 = get_pixel(pixels, x1, y2);
                double p22 = get_pixel(pixels, x2, y2);

                double top_row = p11 * (1.0 - x_diff) + p12 * x_diff;
                double bottom_row = p21 * (1.0 - x_diff) + p22 * x_diff;
                double final_value = top_row * (1.0 - y_diff) + bottom_row * y_diff;

                temp[y * 28 + x] = final_value;
            }
        }
    }

    // Копируем результат обратно
    memcpy(pixels, temp, 784 * sizeof(double));
}

void free_samples(Sample *samples, int num_samples) {
    if (!samples) return;
    for (int i = 0; i < num_samples; i++) {
        free(samples[i].pixels);
    }
    free(samples);
}
