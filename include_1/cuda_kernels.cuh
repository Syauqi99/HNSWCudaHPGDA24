#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>

namespace hnsw {

struct SearchLayerCUDAContext {
    thrust::device_vector<float> d_query_vector;
    thrust::device_vector<float> d_candidate_vectors;
    thrust::device_vector<float> d_distances;
    thrust::device_vector<int> d_candidate_ids;
    
    std::vector<float> h_distances;
    std::vector<int> h_candidate_ids;
    
    explicit SearchLayerCUDAContext(size_t vector_dim, size_t batch_size = 1024) {
        d_query_vector.resize(vector_dim);
        d_candidate_vectors.resize(vector_dim * batch_size);
        d_distances.resize(batch_size);
        d_candidate_ids.resize(batch_size);
        h_distances.resize(batch_size);
        h_candidate_ids.resize(batch_size);
    }
};

// CUDA kernel declarations
__global__ void compute_distances_kernel(
    const float* query_vector,
    const float* candidate_vectors,
    float* distances,
    const int vector_dim,
    const int n_candidates
);

// Helper functions
void batch_compute_distances(
    SearchLayerCUDAContext& ctx,
    const std::vector<float>& query_vector,
    const std::vector<const std::vector<float>*>& candidate_vectors,
    std::vector<float>& distances,
    int batch_size
);

// Existing declarations
__device__ float euclidean_distance_cuda(const float* a, const float* b, int dim);
__global__ void batch_distance_calculation(const float* queries, const float* dataset,
                                         float* distances, int n_queries, int n_points, int dim);

} // namespace hnsw

#endif // CUDA_KERNELS_CUH