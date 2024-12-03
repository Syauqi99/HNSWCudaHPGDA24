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
add_executable(test_euclidean main_euclidean.cu include_1/cuda_kernels.cu)

# Set CUDA specific flags
set_target_properties(hnsw1 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

set_target_properties(test_euclidean PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

# Add optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

