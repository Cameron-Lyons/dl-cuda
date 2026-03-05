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
option(DL_CUDA_BUILD_LEGACY_TESTS
  "Build legacy v1 tests that exercise src/*.cuh directly"
  OFF
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
    set(CMAKE_CUDA_ARCHITECTURES ${_dl_cuda_default_archs})
    message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "Override with -DCMAKE_CUDA_ARCHITECTURES=<arch list> if needed.")
  endif()

  find_package(CUDAToolkit REQUIRED)

  add_library(dl-cuda-core STATIC
    src/v2/nn.cu
    src/v2/optim.cu
    src/v2/loss.cu
    src/v2/checkpoint.cu
  )
  target_include_directories(dl-cuda-core PUBLIC include)
  target_link_libraries(dl-cuda-core PUBLIC CUDA::cudart CUDA::cublas)
  set_target_properties(dl-cuda-core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  dl_cuda_enable_warnings(dl-cuda-core)

  add_library(dl-cuda-examples STATIC
    src/v2/examples_api.cu
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
    add_executable(dl-cuda-v2-cuda-smoke-tests tests/v2_cuda_smoke_tests.cu)
    target_include_directories(dl-cuda-v2-cuda-smoke-tests PRIVATE include)
    target_link_libraries(dl-cuda-v2-cuda-smoke-tests PRIVATE dl-cuda-core)
    dl_cuda_enable_warnings(dl-cuda-v2-cuda-smoke-tests)
    add_test(NAME v2_cuda_smoke_tests COMMAND dl-cuda-v2-cuda-smoke-tests)

    if(DL_CUDA_BUILD_LEGACY_TESTS)
      add_executable(dl-cuda-legacy-host-tests tests/host_tests.cu)
      target_include_directories(dl-cuda-legacy-host-tests PRIVATE src)
      target_link_libraries(dl-cuda-legacy-host-tests PRIVATE CUDA::cudart CUDA::cublas)
      dl_cuda_enable_warnings(dl-cuda-legacy-host-tests)
      add_test(NAME legacy_host_tests COMMAND dl-cuda-legacy-host-tests)

      add_executable(dl-cuda-legacy-gpu-tests tests/gpu_correctness_tests.cu)
      target_include_directories(dl-cuda-legacy-gpu-tests PRIVATE src)
      target_link_libraries(dl-cuda-legacy-gpu-tests PRIVATE CUDA::cudart CUDA::cublas)
      dl_cuda_enable_warnings(dl-cuda-legacy-gpu-tests)
      add_test(NAME legacy_gpu_tests COMMAND dl-cuda-legacy-gpu-tests)
    endif()
  endif()
endif()

if(BUILD_TESTING)
  add_executable(dl-cuda-v2-host-tests tests/v2_host_api_tests.cpp)
  target_include_directories(dl-cuda-v2-host-tests PRIVATE include)
  dl_cuda_enable_warnings(dl-cuda-v2-host-tests)
  add_test(NAME v2_host_tests COMMAND dl-cuda-v2-host-tests)

  add_executable(dl-cuda-v2-cli-tests tests/v2_cli_parse_tests.cpp)
  target_include_directories(dl-cuda-v2-cli-tests PRIVATE examples include)
  dl_cuda_enable_warnings(dl-cuda-v2-cli-tests)
  add_test(NAME v2_cli_tests COMMAND dl-cuda-v2-cli-tests)
endif()

add_custom_target(lint
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lint.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(format
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/format.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
