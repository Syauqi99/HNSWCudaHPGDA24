#include <cuda_runtime.h>
#include "cuda_kernels.cuh"
#include <vector>
using namespace std;

namespace hnsw {

__device__ float euclidean_distance_cuda(const float* a, const float* b, int dim) {
    __shared__ float shared_sum[256];  // Using shared memory for reduction
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    // Each thread processes multiple elements with stride
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    return (tid == 0) ? sqrt(shared_sum[0]) : 0.0f;
}

class AsyncDistanceCalculator {
private:
    float *d_vec1 = nullptr, *d_vec2 = nullptr;
    float *d_result = nullptr;
    float *h_result = nullptr;  // Pinned memory for async transfers
    cudaStream_t stream;
    int current_dim = 0;

public:
    AsyncDistanceCalculator() {
        cudaStreamCreate(&stream);
        cudaMallocHost(&h_result, sizeof(float));  // Allocate pinned memory
    }

    ~AsyncDistanceCalculator() {
        if (d_vec1) cudaFree(d_vec1);
        if (d_vec2) cudaFree(d_vec2);
        if (d_result) cudaFree(d_result);
        if (h_result) cudaFreeHost(h_result);
        cudaStreamDestroy(stream);
    }

    void ensure_device_memory(int dim) {
        if (dim != current_dim) {
            // Free old memory if exists
            if (d_vec1) cudaFree(d_vec1);
            if (d_vec2) cudaFree(d_vec2);
            if (d_result) cudaFree(d_result);

            // Allocate new memory
            cudaMalloc(&d_vec1, dim * sizeof(float));
            cudaMalloc(&d_vec2, dim * sizeof(float));
            cudaMalloc(&d_result, sizeof(float));
            current_dim = dim;
        }
    }

    float calculate_distance(const vector<float>& p1, const vector<float>& p2) {
        int dim = p1.size();
        ensure_device_memory(dim);

        // Async memory transfers to device
        cudaMemcpyAsync(d_vec1, p1.data(), dim * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_vec2, p2.data(), dim * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = 1;
        batch_distance_calculation<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_vec1, d_vec2, d_result, 1, 1, dim);

        // Async copy result back to pinned memory
        cudaMemcpyAsync(h_result, d_result, sizeof(float), 
                       cudaMemcpyDeviceToHost, stream);
        
        // Wait for all operations in stream to complete
        cudaStreamSynchronize(stream);
        
        return *h_result;
    }
};

// Static instance for reuse
static AsyncDistanceCalculator calculator;

float cuda_euclidean_distance(const vector<float>& p1, const vector<float>& p2) {
    return calculator.calculate_distance(p1, p2);
}

} // namespace hnsw 