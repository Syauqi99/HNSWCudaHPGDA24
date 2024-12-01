cmake_minimum_required(VERSION 3.14)

# Initialize CUDA project
project(hnsw1 CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include_1)

# Add the executable and specify the source file
add_executable(hnsw1 main_1.cu)

# Set CUDA specific flags
set_target_properties(hnsw1 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"  # Adjust based on your GPU architecture
)

# Add optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

