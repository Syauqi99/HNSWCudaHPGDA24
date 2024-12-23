cmake_minimum_required(VERSION 3.18)

# Set the project name and enable CUDA
project(HNSWCudaProject LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directories
include_directories(include include_cuda include_batch include_stream include_vectors include_allocation include_pinned)

# Enable CUDA language
enable_language(CUDA)

# Specify CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80)  # Adjust based on your target GPU architectures

# Add the executables for each CUDA version
add_executable(main_cuda_test main_cuda_test.cu)
add_executable(main_allocation_test main_allocation_test.cu)
add_executable(main_batch_test main_batch_test.cu)
add_executable(main_pinned_test main_pinned_test.cu)
add_executable(main_stream_test main_stream_test.cu)
add_executable(main_vectors_test main_vectors_test.cu)

# Add the executable for the CPU version
add_executable(main main.cpp)

# Link CUDA libraries to all CUDA executables
target_link_libraries(main_cuda_test ${CUDA_LIBRARIES})
target_link_libraries(main_allocation_test ${CUDA_LIBRARIES})
target_link_libraries(main_batch_test ${CUDA_LIBRARIES})
target_link_libraries(main_pinned_test ${CUDA_LIBRARIES})
target_link_libraries(main_stream_test ${CUDA_LIBRARIES})
target_link_libraries(main_vectors_test ${CUDA_LIBRARIES})

# Set properties for CUDA compilation for all CUDA executables
foreach(target main_cuda_test main_allocation_test main_batch_test main_pinned_test main_stream_test main_vectors_test)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 14
    )
endforeach()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Add any additional libraries or dependencies here
# target_link_libraries(<target_name> <other_libraries>)


