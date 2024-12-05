#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>

__global__ void cuda_euclidean_distance(float *a, float *b, float *result, int N) {
    extern __shared__ float shared_data[];

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = blockDim.x * gridDim.x;
    int localId = threadIdx.x;

    float sum = 0.0f;

    for (int i = threadId; i < N; i += gridStride) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    shared_data[localId] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localId < stride) {
            shared_data[localId] += shared_data[localId + stride];
        }
        __syncthreads();
    }

    if (localId == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

int main() {
    int N = 1000000;  // Size of arrays
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_result = (float*)malloc(sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i + 1);
    }
    *h_result = 0.0f;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Allocate device memory
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    // Get device properties
    int deviceId;
    cudaDeviceProp props;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);

    // Calculate optimal launch parameters
    int threadsPerBlock = 256;  // Or use props.maxThreadsPerBlock
    int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    numberOfBlocks = min(numberOfBlocks, props.maxGridSize[0]);

    // Launch kernel
    cuda_euclidean_distance<<<numberOfBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_a, d_b, d_result, N
    );

    // Copy result back to host
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Result: " << sqrt(*h_result) << std::endl;
    std::cout << "Time taken: " << duration << " microseconds" << std::endl;

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);
    free(h_result);

    return 0;
}
