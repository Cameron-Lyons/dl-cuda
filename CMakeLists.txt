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

add_library(dl-cuda-core STATIC src/examples_api.cu)
target_include_directories(dl-cuda-core PUBLIC include src)
target_link_libraries(dl-cuda-core PUBLIC CUDA::curand)
set_target_properties(dl-cuda-core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(dl-cuda-core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

add_executable(dl-cuda-char-lm examples/char_lm.cu)
target_include_directories(dl-cuda-char-lm PRIVATE include src)
target_link_libraries(dl-cuda-char-lm PRIVATE dl-cuda-core CUDA::curand)
set_target_properties(dl-cuda-char-lm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(dl-cuda-char-lm PRIVATE -Xcompiler=-Wall,-Wextra)

add_executable(dl-cuda-xor examples/xor.cu)
target_include_directories(dl-cuda-xor PRIVATE include src)
target_link_libraries(dl-cuda-xor PRIVATE dl-cuda-core CUDA::curand)
set_target_properties(dl-cuda-xor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(dl-cuda-xor PRIVATE -Xcompiler=-Wall,-Wextra)

# Backward-compatible binary name.
add_executable(dl-cuda examples/char_lm.cu)
target_include_directories(dl-cuda PRIVATE include src)
target_link_libraries(dl-cuda PRIVATE dl-cuda-core CUDA::curand)
set_target_properties(dl-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(dl-cuda PRIVATE -Xcompiler=-Wall,-Wextra)

enable_testing()
add_executable(dl-cuda-host-tests tests/host_tests.cu)
target_include_directories(dl-cuda-host-tests PRIVATE src)
target_compile_options(dl-cuda-host-tests PRIVATE -Xcompiler=-Wall,-Wextra)
target_link_libraries(dl-cuda-host-tests PRIVATE CUDA::cudart)
add_test(NAME host_tests COMMAND dl-cuda-host-tests)

add_executable(dl-cuda-gpu-tests tests/gpu_correctness_tests.cu)
target_include_directories(dl-cuda-gpu-tests PRIVATE src include)
target_compile_options(dl-cuda-gpu-tests PRIVATE -Xcompiler=-Wall,-Wextra)
target_link_libraries(dl-cuda-gpu-tests PRIVATE CUDA::cudart CUDA::curand)
add_test(NAME gpu_correctness_tests COMMAND dl-cuda-gpu-tests)
