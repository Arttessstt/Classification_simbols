#include "utils.h"
#include <math.h>

/**
 * @brief Функция активации ReLU (Rectified Linear Unit).
 * Возвращает x, если x > 0, иначе 0.
 */
double relu(double x) {
    return x > 0 ? x : 0.0;
}

/**
 * @brief Производная функции активации ReLU.
 * Используется при обратном распространении ошибки.
 */
double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

/**
 * @brief Функция Softmax для выходного слоя.
 * Преобразует выходные значения в вероятности, сумма которых равна 1.
 * @param input Массив значений выходного слоя
 * @param size Размер массива
 */
void softmax(double *input, int size) {
    // Находим максимум для численной стабильности (избегаем переполнения exp)
    double max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) max = input[i];
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i] - max);
        sum += input[i];
    }
    
    // Делим на сумму всех экспонент
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

/**
 * @brief Вычисление функции потерь Cross-Entropy (Кросс-энтропия).
 * Измеряет разницу между предсказанным распределением и реальной меткой.
 * @param predicted Вероятности от Softmax
 * @param target_label Индекс правильного класса
 * @return Значение потерь
 */
double cross_entropy_loss(double *predicted, int target_label) {
    // Добавляем маленькое значение 1e-15 для избежания log(0)
    return -log(predicted[target_label] + 1e-15);
}
