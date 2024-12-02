#include <cuda_runtime.h>
#include "cuda_kernels.cuh"
#include <vector>
using namespace std;

namespace hnsw {

// CUDA kernel declarations
__device__ float euclidean_distance_cuda(const float* a, const float* b, int dim) {
    // Use shared memory for frequently accessed data
    __shared__ float cache[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    // Process multiple elements per thread using loop unrolling
    #pragma unroll 4
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    // Store partial sum in shared memory
    cache[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }
    
    return (tid == 0) ? sqrt(cache[0]) : 0.0f;
}

float cuda_euclidean_distance(const std::vector<float>& p1, const std::vector<float>& p2) {
    int dim = p1.size();
    float result;
    
    // Use pinned memory for faster transfers
    float *h_result;
    cudaMallocHost(&h_result, sizeof(float));
    
    // Allocate device memory
    float *d_vec1, *d_vec2;
    cudaMalloc(&d_vec1, dim * sizeof(float));
    cudaMalloc(&d_vec2, dim * sizeof(float));
    
    // Use asynchronous memory transfers
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaMemcpyAsync(d_vec1, p1.data(), dim * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_vec2, p2.data(), dim * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // Launch kernel with optimal thread configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    batch_distance_calculation<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_vec1, d_vec2, h_result, dim);
    
    // Synchronize and get result
    cudaStreamSynchronize(stream);
    result = *h_result;
    
    // Cleanup
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFreeHost(h_result);
    cudaStreamDestroy(stream);
    
    return result;
}

// Optimized batch kernel for multiple distance calculations
__global__ void batch_distance_calculation(const float* queries,
                                         const float* dataset,
                                         float* distances,
                                         int dim) {
    extern __shared__ float shared_mem[];
    float* query_shared = shared_mem;
    float* data_shared = &shared_mem[blockDim.x];
    
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    // Load data into shared memory
    if (tid < dim) {
        query_shared[tid] = queries[tid];
        data_shared[tid] = dataset[tid];
    }
    __syncthreads();
    
    // Compute distance using loop unrolling
    #pragma unroll 4
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = query_shared[i] - data_shared[i];
        sum += diff * diff;
    }
    
    // Parallel reduction
    __shared__ float partial_sums[256];
    partial_sums[tid] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        distances[0] = sqrt(partial_sums[0]);
    }
}

} // namespace hnsw