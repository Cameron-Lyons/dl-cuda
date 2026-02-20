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
  set(_dl_cuda_default_archs 75;80;86)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    list(APPEND _dl_cuda_default_archs 89)
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    list(APPEND _dl_cuda_default_archs 90)
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${_dl_cuda_default_archs})
  message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "Override with -DCMAKE_CUDA_ARCHITECTURES=<arch list> if needed.")
endif()

find_package(CUDAToolkit REQUIRED)

option(DL_CUDA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

function(dl_cuda_enable_warnings target_name)
  if(DL_CUDA_WARNINGS_AS_ERRORS)
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
    )
  else()
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
  endif()
endfunction()

add_library(dl-cuda-core STATIC src/examples_api.cu)
target_include_directories(dl-cuda-core PUBLIC include src)
target_link_libraries(dl-cuda-core PUBLIC CUDA::curand)
set_target_properties(dl-cuda-core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
dl_cuda_enable_warnings(dl-cuda-core)

add_executable(dl-cuda-char-lm examples/char_lm.cu)
target_include_directories(dl-cuda-char-lm PRIVATE include src)
target_link_libraries(dl-cuda-char-lm PRIVATE dl-cuda-core CUDA::curand)
set_target_properties(dl-cuda-char-lm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
dl_cuda_enable_warnings(dl-cuda-char-lm)

add_executable(dl-cuda-xor examples/xor.cu)
target_include_directories(dl-cuda-xor PRIVATE include src)
target_link_libraries(dl-cuda-xor PRIVATE dl-cuda-core CUDA::curand)
set_target_properties(dl-cuda-xor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
dl_cuda_enable_warnings(dl-cuda-xor)

# Backward-compatible binary name.
add_executable(dl-cuda examples/char_lm.cu)
target_include_directories(dl-cuda PRIVATE include src)
target_link_libraries(dl-cuda PRIVATE dl-cuda-core CUDA::curand)
set_target_properties(dl-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
dl_cuda_enable_warnings(dl-cuda)

enable_testing()
add_executable(dl-cuda-host-tests tests/host_tests.cu)
target_include_directories(dl-cuda-host-tests PRIVATE src)
target_link_libraries(dl-cuda-host-tests PRIVATE CUDA::cudart)
dl_cuda_enable_warnings(dl-cuda-host-tests)
add_test(NAME host_tests COMMAND dl-cuda-host-tests)

add_executable(dl-cuda-gpu-tests tests/gpu_correctness_tests.cu)
target_include_directories(dl-cuda-gpu-tests PRIVATE src include)
target_link_libraries(dl-cuda-gpu-tests PRIVATE CUDA::cudart CUDA::curand)
dl_cuda_enable_warnings(dl-cuda-gpu-tests)
add_test(NAME gpu_correctness_tests COMMAND dl-cuda-gpu-tests)

add_custom_target(lint
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lint.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(format
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/format.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
