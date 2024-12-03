#include <cuda_runtime.h>
#include "cuda_kernels.cuh"
#include <vector>
using namespace std;

namespace hnsw {

// CUDA kernel declarations
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
    float *d_vectors = nullptr;  // Combined buffer for all vectors
    float *d_distances = nullptr;
    float *h_distances = nullptr;  // Pinned memory
    cudaStream_t stream;
    int current_capacity = 0;
    int vector_dim = 0;

public:
    AsyncDistanceCalculator() {
        cudaStreamCreate(&stream);
        cudaMallocHost(&h_distances, 1024 * sizeof(float));  // Pre-allocate for batch processing
    }

    ~AsyncDistanceCalculator() {
        if (d_vectors) cudaFree(d_vectors);
        if (d_distances) cudaFree(d_distances);
        if (h_distances) cudaFreeHost(h_distances);
        cudaStreamDestroy(stream);
    }

    void ensure_device_memory(int batch_size, int dim) {
        int required_capacity = batch_size * 2 * dim;  // Space for pairs of vectors
        if (required_capacity > current_capacity || dim != vector_dim) {
            if (d_vectors) cudaFree(d_vectors);
            if (d_distances) cudaFree(d_distances);
            
            cudaMalloc(&d_vectors, required_capacity * sizeof(float));
            cudaMalloc(&d_distances, batch_size * sizeof(float));
            
            current_capacity = required_capacity;
            vector_dim = dim;
        }
    }

    float calculate_distance(const vector<float>& p1, const vector<float>& p2) {
        int dim = p1.size();
        ensure_device_memory(1, dim);

        // Copy both vectors in one transfer
        cudaMemcpyAsync(d_vectors, p1.data(), dim * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_vectors + dim, p2.data(), dim * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);

        // Launch kernel with optimal configuration
        int threadsPerBlock = 256;
        batch_distance_calculation<<<1, threadsPerBlock, 0, stream>>>(
            d_vectors, d_vectors + dim, d_distances, 1, 1, dim);

        // Async copy result
        cudaMemcpyAsync(h_distances, d_distances, sizeof(float), 
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        return h_distances[0];
    }
};

// Static instance for reuse
static AsyncDistanceCalculator calculator;

float cuda_euclidean_distance(const vector<float>& p1, const vector<float>& p2) {
    return calculator.calculate_distance(p1, p2);
}

vector<float> batch_cuda_euclidean_distance(const vector<vector<float>>& vectors1, const vector<vector<float>>& vectors2, int batch_size);

} // namespace hnsw 