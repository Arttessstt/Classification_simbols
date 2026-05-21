#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include "neural_net.h"

/**
 * @brief Функция для загрузки конфигурации нейронной сети из файла.
 * 
 * Читает количество нейронов в каждом слое, скорость обучения и коэффициент L2-регуляризации.
 * 
 * @param filename Путь к файлу конфигурации.
 * @return Network* Указатель на созданную структуру нейронной сети.
 */
Network* load_network_from_config(const char *filename);

#endif