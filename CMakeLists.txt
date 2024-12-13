cmake_minimum_required(VERSION 3.18)

# Set the project name and enable CUDA
project(HNSWCudaProject LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directories
include_directories(include include_cuda)

# Enable CUDA language
enable_language(CUDA)

# Specify CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80)  # Adjust based on your target GPU architectures

# Add the executable for the CUDA version
add_executable(main_cuda_test main_cuda_test.cu)

# Link CUDA libraries
target_link_libraries(main_cuda_test ${CUDA_LIBRARIES})

# Set properties for CUDA compilation
set_target_properties(main_cuda_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)

# Add the executable for the non-CUDA version if needed
# add_executable(main main.cpp)
# target_link_libraries(main <other_libraries>)

# Add any additional libraries or dependencies here
# target_link_libraries(main_cuda_test <other_libraries>)


