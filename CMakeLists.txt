cmake_minimum_required(VERSION 3.18)
project(dl-cuda LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75;80;86)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(dl-cuda src/main.cu)

set_target_properties(dl-cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(dl-cuda PRIVATE CUDA::curand)
target_compile_options(dl-cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

enable_testing()
add_executable(dl-cuda-host-tests tests/host_tests.cu)
target_include_directories(dl-cuda-host-tests PRIVATE src)
target_compile_options(dl-cuda-host-tests PRIVATE -Xcompiler=-Wall,-Wextra)
target_link_libraries(dl-cuda-host-tests PRIVATE CUDA::cudart)
add_test(NAME host_tests COMMAND dl-cuda-host-tests)
