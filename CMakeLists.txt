cmake_minimum_required(VERSION 3.14)

# Initialize CUDA project
project(hnsw_search CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75)  # For newer GPUs
# Alternative: Support multiple architectures
# set(CMAKE_CUDA_ARCHITECTURES 60 70 75)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include_1)

# Add the executables
add_executable(euclidean_cuda euclidean_distance.cu)          # CUDA version
add_executable(euclidean_cpu euclidean_cpu.cpp)          # CPU version

# Add optimization flags for CUDA targets
set_target_properties(
    euclidean_cuda 
    euclidean_cpu
    PROPERTIES
        CUDA_FLAGS "-O3"
)

# Add optimization flags for CPU target
target_compile_options(euclidean_cpu PRIVATE -O3)
target_compile_options(vector_diff PRIVATE -O3)


