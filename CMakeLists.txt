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
add_executable(euclidean_cuda main_cu.cu)          # CUDA version
add_executable(euclidean_cpu main_cu.cpp)          # CPU version
# Add optimization flags for CUDA targets
set_target_properties(
    euclidean_cuda
    PROPERTIES
        CUDA_FLAGS "-O3"
)

# Add optimization flags for CPU target
target_compile_options(euclidean_cpu PRIVATE -O3)


