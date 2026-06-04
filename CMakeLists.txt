cmake_minimum_required(VERSION 3.18)
project(dl-cuda LANGUAGES CXX)

include(CTest)
include(CheckLanguage)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DL_CUDA_ENABLE_CUDA "Build the CUDA-backed core library and CLI" ON)
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
    )
  else()
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
    )
  endif()

  if(DL_CUDA_ENABLE_CUDA AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    if(DL_CUDA_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
      )
    else()
      target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
      )
    endif()

    if(DL_CUDA_SUPPRESS_DEPRECATED_GPU_TARGETS_WARNING)
      target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
      )
    endif()
    if(DL_CUDA_PER_THREAD_DEFAULT_STREAM)
      target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
      )
    endif()
  endif()
endfunction()

if(DL_CUDA_ENABLE_CUDA)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "DL_CUDA_ENABLE_CUDA=ON but no CUDA compiler was found. "
      "Reconfigure with -DDL_CUDA_ENABLE_CUDA=OFF to build host-only tests."
    )
  endif()

  enable_language(CUDA)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(_dl_cuda_default_archs 75;80;86)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND _dl_cuda_default_archs 89)
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
      list(APPEND _dl_cuda_default_archs 90)
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
      list(APPEND _dl_cuda_default_archs 100 120)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${_dl_cuda_default_archs})
    message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "Override with -DCMAKE_CUDA_ARCHITECTURES=<arch list> if needed.")
  endif()

  find_package(CUDAToolkit REQUIRED)

  set(DL_CUDA_CORE_SOURCES
    src/autograd.cu
    src/checkpoint.cu
    src/loss.cu
    src/nn/activations.cu
    src/nn/conv2d.cu
    src/nn/dropout.cu
    src/nn/embedding.cu
    src/nn/linear.cu
    src/nn/module.cu
    src/nn/normalization.cu
    src/nn/pooling.cu
    src/optim/adam.cu
    src/optim/clip_grad.cu
    src/optim/detail/common.cu
    src/optim/optimizer.cu
    src/optim/rmsprop.cu
    src/optim/schedulers.cu
    src/optim/sgd.cu
    src/runtime.cu
    src/tensor.cu
    src/tensor_ops.cu
    src/trainer.cu
  )

  set(DL_CUDA_EXAMPLE_SOURCES
    src/examples/char_sample.cu
    src/examples/char_train.cu
    src/examples/detail/char_metadata.cu
    src/examples/detail/common.cu
    src/examples/xor.cu
  )

  add_library(dl-cuda-core STATIC
    ${DL_CUDA_CORE_SOURCES}
  )
  target_include_directories(dl-cuda-core PUBLIC include)
  target_link_libraries(dl-cuda-core PUBLIC CUDA::cudart CUDA::cublas)
  if(TARGET CUDA::cublasLt AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.0)
    target_link_libraries(dl-cuda-core PUBLIC CUDA::cublasLt)
    target_compile_definitions(dl-cuda-core PUBLIC DLCUDA_HAS_CUBLASLT=1)
  endif()
  set_target_properties(dl-cuda-core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  dl_cuda_enable_warnings(dl-cuda-core)

  add_library(dl-cuda-examples STATIC
    ${DL_CUDA_EXAMPLE_SOURCES}
  )
  target_include_directories(dl-cuda-examples PUBLIC include)
  target_link_libraries(dl-cuda-examples PUBLIC dl-cuda-core)
  set_target_properties(dl-cuda-examples PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  dl_cuda_enable_warnings(dl-cuda-examples)

  add_executable(dl-cuda examples/cli.cu)
  target_include_directories(dl-cuda PRIVATE include examples)
  target_link_libraries(dl-cuda PRIVATE dl-cuda-examples)
  set_target_properties(dl-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  dl_cuda_enable_warnings(dl-cuda)

  if(BUILD_TESTING)
    add_executable(dl-cuda-gpu-correctness-tests
      tests/gpu_correctness_tests.cu
      tests/gpu/autograd_tests.cu
      tests/gpu/layer_tests.cu
      tests/gpu/mixed_precision_tests.cu
      tests/gpu/optimizer_tests.cu
    )
    target_include_directories(dl-cuda-gpu-correctness-tests PRIVATE include)
    target_link_libraries(dl-cuda-gpu-correctness-tests PRIVATE dl-cuda-core)
    dl_cuda_enable_warnings(dl-cuda-gpu-correctness-tests)
    add_test(NAME gpu_correctness_tests COMMAND dl-cuda-gpu-correctness-tests)
    set_tests_properties(gpu_correctness_tests PROPERTIES LABELS gpu)

    add_test(NAME char_lm_smoke
      COMMAND dl-cuda train-char --seq-len 8 --d-model 8 --epochs 2 --print-every 1
              --val-every 1 --val-windows 1 --test-windows 1 --gen-len 16 --no-save
    )
    set_tests_properties(char_lm_smoke PROPERTIES LABELS gpu)
  endif()
endif()

if(BUILD_TESTING)
  add_executable(dl-cuda-host-tests tests/host_api_tests.cpp)
  target_include_directories(dl-cuda-host-tests PRIVATE include)
  dl_cuda_enable_warnings(dl-cuda-host-tests)
  add_test(NAME host_tests COMMAND dl-cuda-host-tests)

  add_executable(dl-cuda-cli-tests tests/cli_parse_tests.cpp)
  target_include_directories(dl-cuda-cli-tests PRIVATE examples include)
  dl_cuda_enable_warnings(dl-cuda-cli-tests)
  add_test(NAME cli_tests COMMAND dl-cuda-cli-tests)
endif()

add_custom_target(lint
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lint.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(format
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/format.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
