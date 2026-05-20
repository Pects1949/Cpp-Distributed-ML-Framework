cmake_minimum_required(VERSION 3.20)
project(cppdist-train VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_MPI  "Enable MPI distributed backend"  OFF)
option(ENABLE_CUDA "Enable CUDA GPU acceleration"     OFF)

if(ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    message(STATUS "MPI found: ${MPI_CXX_LIBRARIES}")
    add_compile_definitions(CPPDIST_USE_MPI)
else()
    message(STATUS "MPI disabled -- single-process mode")
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(CPPDIST_USE_CUDA)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
else()
    message(STATUS "CUDA disabled -- CPU-only build")
endif()

add_library(cppdist STATIC
    src/tensor.cpp
    src/layer.cpp
    src/loss.cpp
    src/optimizer.cpp
    src/model.cpp
    src/distributed.cpp
    src/dataloader.cpp
)

target_include_directories(cppdist PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(cppdist PRIVATE
    -O2 -Wall -Wextra -Wno-unused-parameter
    $<$<CONFIG:Debug>:-g -O0>
)

if(ENABLE_MPI)
    target_link_libraries(cppdist PUBLIC MPI::MPI_CXX)
    target_include_directories(cppdist PUBLIC ${MPI_CXX_INCLUDE_DIRS})
endif()

add_subdirectory(examples/mlp_classify)
