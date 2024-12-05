#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <vector>

__global__ void cuda_euclidean_distance(float *a, float *b, float *result, int N, int dim) {
    extern __shared__ float shared_data[];

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = blockDim.x * gridDim.x;
    int localId = threadIdx.x;

    float sum = 0.0f;

    // Each thread processes one vector (all dimensions)
    if (threadId < N) {
        for (int d = 0; d < dim; d++) {
            float diff = a[threadId * dim + d] - b[threadId * dim + d];
            sum += diff * diff;
        }
    }

    shared_data[localId] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
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
    const int N = 1000000;  // Number of vectors
    const int DIM = 128;    // Dimension of each vector
    size_t size = N * DIM * sizeof(float);  // Total size for each array
    
    // Initialize vectors with some values
    std::vector<float> a(N * DIM);
    std::vector<float> b(N * DIM);
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < DIM; d++) {
            a[i * DIM + d] = static_cast<float>(i);
            b[i * DIM + d] = static_cast<float>(i + 1);
        }
    }

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Allocate device memory
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    // Get device properties
    int deviceId;
    cudaDeviceProp props;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);

    // Calculate optimal launch parameters
    int threadsPerBlock = 256;
    int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    numberOfBlocks = min(numberOfBlocks, props.maxGridSize[0]);

    // Launch kernel
    cuda_euclidean_distance<<<numberOfBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_a, d_b, d_result, N, DIM
    );

    // Create host result variable and copy result back
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Result: " << sqrt(h_result) << std::endl;
    std::cout << "Time taken: " << duration << " microseconds" << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}
