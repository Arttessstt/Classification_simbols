#ifndef DATA_LOADER_H
#define DATA_LOADER_H

/**
 * @struct Sample
 * @brief Структура для хранения одного обучающего примера.
 */
typedef struct {
    double *pixels; ///< Массив значений пикселей (нормализованный 0.0-1.0)
    int label;      ///< Правильная метка класса (0-9 для цифр)
} Sample;

/**
 * @brief Загрузка данных из CSV файла.
 */
Sample* load_csv(const char *filename, int *num_samples);

/**
 * @brief Добавление шума к изображению.
 */
void add_noise(double *pixels, int size);

/**
 * @brief Освобождение памяти массива примеров.
 */
void free_samples(Sample *samples, int num_samples);

/**
 * @brief Аугментация изображения (поворот).
 */
void augment_image(double *pixels);

#endif