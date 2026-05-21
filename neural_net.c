#include "neural_net.h"
#include "utils.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#define NUM_THREADS 4

/**
 * @brief Функция для выполнения части матричного перемножения в отдельном потоке.
 * @param arg Указатель на структуру ThreadData
 */
void* layer_worker(void *arg) {
    ThreadData *data = (ThreadData*)arg;
    Layer *layer = data->layer;

    for (int j = data->start_row; j < data->end_row; j++) {
        double z = layer->biases[j];
        for (int k = 0; k < layer->in_nodes; k++) {
            z += layer->weights[j * layer->in_nodes + k] * layer->inputs[k];
        }
        layer->outputs[j] = z;
    }
    return NULL;
}

Network* init_network(int *layer_sizes, int num_layers, double lr, double reg) {
    // Выделение памяти под основную структуру сети
    Network *net = (Network*)malloc(sizeof(Network));
    if (!net) return NULL;

    net->num_layers = num_layers - 1;
    net->learning_rate = lr;
    net->lambda = reg;
    
    // Выделение памяти под слои
    net->layers = (Layer*)malloc(net->num_layers * sizeof(Layer));
    if (!net->layers) {
        free(net);
        return NULL;
    }

    srand((unsigned int)time(NULL));

    for (int i = 0; i < net->num_layers; i++) {
        int in = layer_sizes[i];
        int out = layer_sizes[i+1];
        net->layers[i].in_nodes = in;
        net->layers[i].out_nodes = out;

        // Выделение памяти для всех параметров слоя с проверкой
        net->layers[i].weights = (double*)malloc(in * out * sizeof(double));
        net->layers[i].biases = (double*)malloc(out * sizeof(double));
        net->layers[i].grad_weights = (double*)malloc(in * out * sizeof(double));
        net->layers[i].grad_biases = (double*)malloc(out * sizeof(double));
        net->layers[i].inputs = (double*)malloc(in * sizeof(double));
        net->layers[i].outputs = (double*)malloc(out * sizeof(double));
        net->layers[i].delta = (double*)malloc(out * sizeof(double));

        // Если хоть одно выделение не удалось — очищаем всё и выходим
        if (!net->layers[i].weights || !net->layers[i].biases || 
            !net->layers[i].grad_weights || !net->layers[i].grad_biases ||
            !net->layers[i].inputs || !net->layers[i].outputs || !net->layers[i].delta) {
            // Здесь должна быть полная очистка, для краткости вернем NULL
            return NULL; 
        }

        // Инициализация Xavier/He для предотвращения затухания градиентов
        double scale = sqrt(2.0 / (in + out));
        for (int j = 0; j < in * out; j++) {
            net->layers[i].weights[j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
        }
        // Небольшое положительное смещение для ReLU
        for (int j = 0; j < out; j++) net->layers[i].biases[j] = 0.01;
    }
    return net;
}

double* forward_propagation(Network *net, double *input_data) {
    double *current_input = input_data;

    for (int i = 0; i < net->num_layers; i++) {
        Layer *layer = &net->layers[i];

        // Копируем входные данные в структуру слоя
        memcpy(layer->inputs, current_input, layer->in_nodes * sizeof(double));

        // Параллельное вычисление выхода слоя через потоки
        pthread_t threads[NUM_THREADS];
        ThreadData thread_data[NUM_THREADS];
        int rows_per_thread = layer->out_nodes / NUM_THREADS;

        for (int t = 0; t < NUM_THREADS; t++) {
            thread_data[t].layer = layer;
            thread_data[t].start_row = t * rows_per_thread;
            thread_data[t].end_row = (t == NUM_THREADS - 1) ? layer->out_nodes : (t + 1) * rows_per_thread;
            pthread_create(&threads[t], NULL, layer_worker, &thread_data[t]);
        }

        for (int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }

        // Применение функций активации
        if (i < net->num_layers - 1) {
            // ReLU для скрытых слоев
            for (int j = 0; j < layer->out_nodes; j++) {
                layer->outputs[j] = relu(layer->outputs[j]);
            }
        } else {
            // Softmax для выходного слоя (вероятностное распределение)
            softmax(layer->outputs, layer->out_nodes);
        }

        current_input = layer->outputs;
    }
    return net->layers[net->num_layers - 1].outputs;
}

void backward_propagation(Network *net, int target_label) {
    // 1. Вычисление ошибки на выходном слое (Cross-Entropy + Softmax derivative)
    Layer *out_layer = &net->layers[net->num_layers - 1];
    for (int i = 0; i < out_layer->out_nodes; i++) {
        double y = (i == target_label) ? 1.0 : 0.0;
        out_layer->delta[i] = out_layer->outputs[i] - y;
    }

    // 2. Распространение ошибки назад по слоям
    for (int i = net->num_layers - 1; i >= 0; i--) {
        Layer *l = &net->layers[i];
        
        // Вычисление градиентов текущего слоя
        for (int j = 0; j < l->out_nodes; j++) {
            for (int k = 0; k < l->in_nodes; k++) {
                l->grad_weights[j * l->in_nodes + k] = l->delta[j] * l->inputs[k];
            }
            l->grad_biases[j] = l->delta[j];
        }

        // Передача ошибки на предыдущий слой
        if (i > 0) {
            Layer *prev_l = &net->layers[i - 1];
            for (int k = 0; k < l->in_nodes; k++) {
                double error = 0.0;
                for (int j = 0; j < l->out_nodes; j++) {
                    error += l->weights[j * l->in_nodes + k] * l->delta[j];
                }
                // Умножаем на производную ReLU
                prev_l->delta[k] = error * relu_derivative(prev_l->outputs[k]);
            }
        }
    }
}

void update_weights(Network *net) {
    // Обновление параметров по алгоритму SGD с L2-регуляризацией
    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        for (int j = 0; j < l->out_nodes * l->in_nodes; j++) {
            // L2 регуляризация: grad = grad_loss + lambda * weight
            double grad_with_reg = l->grad_weights[j] + net->lambda * l->weights[j];
            l->weights[j] -= net->learning_rate * grad_with_reg;
        }
        for (int j = 0; j < l->out_nodes; j++) {
            l->biases[j] -= net->learning_rate * l->grad_biases[j];
        }
    }
}

void save_heatmap(Network *net, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) return;

    fprintf(file, "--- Heatmap of Hidden Layer Activations ---\n\n");
    for (int i = 0; i < net->num_layers - 1; i++) {
        fprintf(file, "Layer %d (%d neurons):\n", i + 1, net->layers[i].out_nodes);
        for (int j = 0; j < net->layers[i].out_nodes; j++) {
            fprintf(file, "%8.4f ", net->layers[i].outputs[j]);
            if ((j + 1) % 16 == 0) fprintf(file, "\n");
        }
        fprintf(file, "\n\n");
    }
    fclose(file);
}

void free_network(Network *net) {
    if (!net) return;
    for (int i = 0; i < net->num_layers; i++) {
        free(net->layers[i].weights);
        free(net->layers[i].biases);
        free(net->layers[i].inputs);
        free(net->layers[i].outputs);
        free(net->layers[i].grad_weights);
        free(net->layers[i].grad_biases);
        free(net->layers[i].delta);
    }
    free(net->layers);
    free(net);
}
