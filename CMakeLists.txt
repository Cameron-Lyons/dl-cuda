cmake_minimum_required(VERSION 3.18)
project(dl-cuda LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(dl-cuda src/main.cu)

set_target_properties(dl-cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(dl-cuda PRIVATE CUDA::curand)
