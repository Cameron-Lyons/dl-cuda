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
option(DL_CUDA_SUPPRESS_DEPRECATED_GPU_TARGETS_WARNING
  "Suppress nvcc deprecated GPU targets warning"
  ON
)
option(DL_CUDA_PER_THREAD_DEFAULT_STREAM
  "Compile CUDA code with --default-stream=per-thread"
  ON
)

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
  if(DL_CUDA_SUPPRESS_DEPRECATED_GPU_TARGETS_WARNING
     AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    )
  endif()
  if(DL_CUDA_PER_THREAD_DEFAULT_STREAM
     AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
    )
  endif()
endfunction()

add_library(dl-cuda-core STATIC
  src/v2/nn.cu
  src/v2/optim.cu
  src/v2/loss.cu
  src/v2/checkpoint.cu
  src/v2/examples_api.cu
)
target_include_directories(dl-cuda-core PUBLIC include)
target_link_libraries(dl-cuda-core PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(dl-cuda-core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
dl_cuda_enable_warnings(dl-cuda-core)

add_executable(dl-cuda examples/cli.cu)
target_include_directories(dl-cuda PRIVATE include)
target_link_libraries(dl-cuda PRIVATE dl-cuda-core)
set_target_properties(dl-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
dl_cuda_enable_warnings(dl-cuda)

enable_testing()
add_executable(dl-cuda-host-tests tests/host_tests.cu)
target_include_directories(dl-cuda-host-tests PRIVATE src)
target_link_libraries(dl-cuda-host-tests PRIVATE CUDA::cudart CUDA::cublas)
dl_cuda_enable_warnings(dl-cuda-host-tests)
add_test(NAME host_tests COMMAND dl-cuda-host-tests)

add_custom_target(lint
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lint.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(format
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/format.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
