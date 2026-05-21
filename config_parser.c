#include "config_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Загружает архитектуру и гиперпараметры сети из текстового файла.
 * @param filename Путь к файлу конфигурации (например, config.txt)
 * @return Указатель на инициализированную сеть или NULL при ошибке
 */
Network* load_network_from_config(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Ошибка: Не удалось открыть файл конфигурации");
        return NULL;
    }

    char line[256];
    int layers_config[16]; // Максимум 16 слоев
    int num_layers = 0;
    double lr = 0.001;
    double reg = 0.01;

    while (fgets(line, sizeof(line), file)) {
        // Пропуск пустых строк и комментариев
        if (line[0] == '\n' || line[0] == '#') continue;

        // Парсинг структуры слоев: "neurons: 784, 256, 10"
        if (strncmp(line, "neurons:", 8) == 0) {
            char *token = strtok(line + 8, ", \t\n");
            while (token != NULL && num_layers < 16) {
                layers_config[num_layers++] = atoi(token);
                token = strtok(NULL, ", \t\n");
            }
        } 
        // Парсинг скорости обучения: "learning_rate: 0.001"
        else if (strncmp(line, "learning_rate:", 14) == 0) {
            sscanf(line + 14, "%lf", &lr);
        } 
        // Парсинг коэффициента регуляризации: "regularization: 0.01"
        else if (strncmp(line, "regularization:", 15) == 0) {
            sscanf(line + 15, "%lf", &reg);
        }
    }
    fclose(file);

    // Проверка, что структура слоев была считана корректно
    if (num_layers < 2) {
        fprintf(stderr, "Ошибка: Некорректная архитектура сети в config.txt\n");
        return NULL;
    }

    // Создание и инициализация сети
    return init_network(layers_config, num_layers, lr, reg);
}
