cmake_minimum_required(VERSION 3.14)

# Initialize CUDA project
project(hnsw_search CUDA CXX)

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

# Set CUDA specific properties
set_target_properties(main_search_layer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

# Add optimization flags
set_target_properties(main_search_layer PROPERTIES
    CUDA_FLAGS "-O3"
)


