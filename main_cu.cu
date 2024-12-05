#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>  // Added this header for std::cout

__global__ void cuda_euclidean_distance(float *a, float *b, float *result, int N) {
    // Declare shared memory array - visible to all threads in the same block
    // Size must be specified when kernel is launched
    extern __shared__ float shared_data[];

    // Calculate global thread ID and stride
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread ID
    int gridStride = blockDim.x * gridDim.x;               // Total number of threads
    int localId = threadIdx.x;                             // Local thread ID within the block

    // Initialize local sum for this thread
    float sum = 0.0f;

    // Each thread processes multiple elements with grid-stride loop
    // This allows handling arrays larger than total number of threads
    for (int i = threadId; i < N; i += gridStride) {
        float diff = a[i] - b[i];        // Calculate difference
        sum += diff * diff;              // Add squared difference to local sum
    }

    // Store this thread's sum in shared memory
    shared_data[localId] = sum;
    
    // Ensure all threads in block have written to shared memory
    __syncthreads();

    // Parallel reduction in shared memory
    // This loop reduces the partial sums in shared memory to a single sum per block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localId < stride) {
            // Each thread adds a value from the second half to the first half
            shared_data[localId] += shared_data[localId + stride];
        }
        // Ensure all threads have finished reading shared memory before next iteration
        __syncthreads();
    }

    // Only thread 0 in each block writes the final result
    if (localId == 0) {
        // Atomically add this block's sum to the global result
        // atomicAdd is necessary because multiple blocks may write simultaneously
        atomicAdd(result, shared_data[0]);
    }
}

// Define a simple Data structure to hold the data
template <typename T = float>
struct Data {
    std::vector<T> x;

    Data(std::vector<T> v) : x(v) {}

    size_t size() const { return x.size(); }

    const T& operator[](size_t i) const { return x[i]; }
};

// Euclidean distance function
template <typename T = float>
float euclidean_distance(const Data<T>& p1, const Data<T>& p2) {
    float result = 0;
    for (size_t i = 0; i < p1.size(); i++) {
        result += std::pow(p1[i] - p2[i], 2);
    }
    result = std::sqrt(result);
    return result;
}

int main() {
    int N = 1000000;  // Size of arrays
  
    // Initialize vectors with some values
    std::vector<float> a(N);
    std::vector<float> b(N);
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    float *d_a, *d_b, *d_result;
    size_t size = N * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, sizeof(float));

    // Copy data from vectors to device
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    // Initialize result to 0
    cudaMemset(d_result, 0, sizeof(float));

    // Get device properties
    int deviceId;
    cudaDeviceProp props;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);

    int threadsPerBlock = props.maxThreadsPerBlock;
    int numberOfBlocks = props.multiProcessorCount * 4;

    // Launch kernel with shared memory size = threadsPerBlock * sizeof(float)
    cuda_euclidean_distance<<<numberOfBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_a, d_b, d_result, N
    );

    // Check for errors
    cudaError_t addVectorsErr = cudaGetLastError();
    if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

    cudaError_t asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    // Copy result back to host
    float host_result;
    cudaMemcpy(&host_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    host_result = sqrt(host_result);  // Take square root of final sum

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Time taken: " << duration << " microseconds" << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    cudaMemPrefetchAsync(d_result, sizeof(float), cudaCpuDeviceId); // Prefetch c to CPU


    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Create Data objects
    Data<float> data_a(a);
    Data<float> data_b(b);
  
    // Calculate Euclidean distance
    float result = euclidean_distance(data_a, data_b);
  
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Result: " << result << std::endl;
    std::cout << "Time taken CPU: " << duration << " microseconds" << std::endl;


    return 0;
}
