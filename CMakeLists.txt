cmake_minimum_required(VERSION 4.0)
project(Project C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math")

set(CMAKE_C_STANDARD 11)

add_executable(Project main.c
        neural_net.h
        neural_net.c
        config_parser.c
        config_parser.h
        utils.h
        utils.c
        data_loader.c
        data_loader.h)

target_link_libraries(Project pthread)