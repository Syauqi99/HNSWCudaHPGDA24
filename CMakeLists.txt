cmake_minimum_required(VERSION 3.14)

# Initialize CUDA project
project(hnsw1 CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include_1)

# Add the executables
add_executable(hnsw1 main_1.cu include_1/cuda_kernels.cu)
add_executable(main_search_layer main_search_layer.cu include_1/cuda_kernels.cu)

# Set CUDA specific flags for all targets
set(CUDA_TARGETS hnsw1 main_search_layer)

foreach(TARGET ${CUDA_TARGETS})
    set_target_properties(${TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75"
    )
endforeach()

# Add optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")


