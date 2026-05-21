#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

/**
 * @struct Layer
 * @brief Структура, представляющая один слой нейронной сети.
 */
typedef struct {
    int in_nodes;       ///< Количество входных узлов
    int out_nodes;      ///< Количество выходных узлов
    double *weights;    ///< Матрица весов (размер: in_nodes * out_nodes)
    double *biases;     ///< Вектор смещений (размер: out_nodes)
    double *inputs;     ///< Входные значения для данного слоя
    double *outputs;    ///< Выходные значения (после активации)
    double *grad_weights; ///< Градиенты для весов
    double *grad_biases;  ///< Градиенты для смещений
    double *delta;      ///< Ошибка (delta) слоя для обратного распространения
} Layer;

/**
 * @struct Network
 * @brief Структура, представляющая всю нейронную сеть.
 */
typedef struct {
    Layer *layers;      ///< Массив слоев
    int num_layers;     ///< Общее количество слоев (без учета входного)
    double learning_rate; ///< Скорость обучения
    double lambda;      ///< Коэффициент L2-регуляризации
} Network;

/**
 * @struct ThreadData
 * @brief Данные для передачи в потоки при параллельном вычислении слоя.
 */
typedef struct {
    Layer *layer;       ///< Слой, который обрабатывается
    int start_row;      ///< Начальный индекс выходного нейрона
    int end_row;        ///< Конечный индекс выходного нейрона
} ThreadData;

/* Основные функции управления сетью */

/**
 * @brief Инициализация нейронной сети с заданной архитектурой.
 * @param layer_sizes Массив размеров слоев
 * @param num_layers Количество слоев
 * @param lr Скорость обучения
 * @param reg Коэффициент регуляризации
 * @return Указатель на созданную сеть или NULL при ошибке
 */
Network* init_network(int *layer_sizes, int num_layers, double lr, double reg);

/**
 * @brief Освобождение памяти, занятой сетью.
 * @param net Указатель на сеть
 */
void free_network(Network *net);

/**
 * @brief Прямое распространение сигнала через сеть.
 * @param net Указатель на сеть
 * @param input_data Входной вектор (784 для MNIST)
 * @return Указатель на выходной слой (вероятности классов)
 */
double* forward_propagation(Network *net, double *input_data);

/**
 * @brief Обратное распространение ошибки.
 * @param net Указатель на сеть
 * @param target_label Индекс правильного класса
 */
void backward_propagation(Network *net, int target_label);

/**
 * @brief Обновление весов и смещений на основе градиентов (SGD + L2).
 * @param net Указатель на сеть
 */
void update_weights(Network *net);

/**
 * @brief Сохранение активаций скрытых слоев для визуализации.
 * @param net Указатель на сеть
 * @param filename Имя выходного файла
 */
void save_heatmap(Network *net, const char *filename);

#endif