#ifndef UTILS_H
#define UTILS_H

/**
 * @brief Функция активации ReLU.
 */
double relu(double x);

/**
 * @brief Производная ReLU.
 */
double relu_derivative(double x);

/**
 * @brief Функция Softmax для нормализации выхода.
 */
void softmax(double *input, int size);

/**
 * @brief Функция потерь Cross-Entropy.
 */
double cross_entropy_loss(double *predicted, int target_label);

#endif