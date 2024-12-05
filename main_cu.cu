#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>  // Added this header for std::cout

__global__ void cuda_euclidean_distance(float *a, float *b, float *result, int N) {
    extern __shared__ float shared_data[];
    
    // Use multiple elements per thread to reduce memory latency
    const int elementsPerThread = 4;  // Process 4 elements per thread
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = blockDim.x * gridDim.x * elementsPerThread;
    int localId = threadIdx.x;

    // Initialize local sum using register
    float local_sum = 0.0f;

    // Process multiple elements per thread
    #pragma unroll
    for (int i = threadId * elementsPerThread; i < N; i += gridStride) {
        float sum_chunk = 0.0f;
        
        #pragma unroll
        for (int j = 0; j < elementsPerThread && (i + j) < N; j++) {
            float diff = a[i + j] - b[i + j];
            sum_chunk += diff * diff;
        }
        local_sum += sum_chunk;
    }

    // Store in shared memory
    shared_data[localId] = local_sum;
    __syncthreads();

    // Optimized reduction using warp-level primitives
    if (blockDim.x >= 64) {
        if (localId < 32) {
            // Warp-level reduction (no sync needed within a warp)
            volatile float* smem = shared_data;
            if (blockDim.x >= 64) smem[localId] += smem[localId + 32];
            if (blockDim.x >= 32) smem[localId] += smem[localId + 16];
            if (blockDim.x >= 16) smem[localId] += smem[localId + 8];
            if (blockDim.x >= 8)  smem[localId] += smem[localId + 4];
            if (blockDim.x >= 4)  smem[localId] += smem[localId + 2];
            if (blockDim.x >= 2)  smem[localId] += smem[localId + 1];
        }
    } else {
        // Original reduction for smaller block sizes
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (localId < stride) {
                shared_data[localId] += shared_data[localId + stride];
            }
            __syncthreads();
        }
    }

    if (localId == 0) {
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

    int threadsPerBlock = 256;  // Use power of 2, typically 256 works well
    int blocksPerSM = 2048 / threadsPerBlock;  // Maximize occupancy
    int numberOfBlocks = props.multiProcessorCount * blocksPerSM;
    
    // Use cudaMemcpyAsync for better performance
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(d_a, a.data(), size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), size, cudaMemcpyHostToDevice, stream);
    
    // Launch kernel
    cuda_euclidean_distance<<<numberOfBlocks, threadsPerBlock, threadsPerBlock * sizeof(float), stream>>>(
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

    // Clean up stream
    cudaStreamDestroy(stream);

    return 0;
}
