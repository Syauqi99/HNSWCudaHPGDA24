#include "hnsw1.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <random>

using namespace std;
using namespace hnsw;

// CUDA kernel for parallel distance computation
__global__ void compute_distances_kernel(float* query, float* points, float* distances, 
                                       int num_points, int num_dimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        float dist = 0.0f;
        for (int d = 0; d < num_dimensions; d++) {
            float diff = query[d] - points[idx * num_dimensions + d];
            dist += diff * diff;
        }
        distances[idx] = sqrt(dist);
    }
}

int main() {
    // Parameters
    const int NUM_POINTS = 10000;
    const int NUM_DIMENSIONS = 128;
    const int K = 10;
    const int EF = 64;
    const int NUM_RUNS = 100;
    const int BLOCK_SIZE = 256;
    
    // Generate random dataset
    Dataset<> dataset;
    mt19937 rng(42);
    uniform_real_distribution<float> dist(-1.0, 1.0);
    
    cout << "Generating random dataset..." << endl;
    for (int i = 0; i < NUM_POINTS; i++) {
        vector<float> features(NUM_DIMENSIONS);
        for (float& f : features) {
            f = dist(rng);
        }
        dataset.emplace_back(Data<>(i, features));
    }

    // Build HNSW index
    HNSW hnsw(16, 64);  // M=16, ef_construction=64
    cout << "Building index..." << endl;
    hnsw.build(dataset);

    // Generate random query
    vector<float> query_features(NUM_DIMENSIONS);
    for (float& f : query_features) {
        f = dist(rng);
    }
    Data<> query(-1, query_features);

    // Prepare GPU memory
    float *d_query, *d_points, *d_distances;
    cudaMalloc(&d_query, NUM_DIMENSIONS * sizeof(float));
    cudaMalloc(&d_points, NUM_POINTS * NUM_DIMENSIONS * sizeof(float));
    cudaMalloc(&d_distances, NUM_POINTS * sizeof(float));

    // Copy query to GPU
    cudaMemcpy(d_query, query_features.data(), NUM_DIMENSIONS * sizeof(float), 
               cudaMemcpyHostToDevice);

    // Copy dataset points to GPU
    vector<float> points_data;
    points_data.reserve(NUM_POINTS * NUM_DIMENSIONS);
    for (const auto& data : dataset) {
        points_data.insert(points_data.end(), data.features.begin(), data.features.end());
    }
    cudaMemcpy(d_points, points_data.data(), 
               NUM_POINTS * NUM_DIMENSIONS * sizeof(float), 
               cudaMemcpyHostToDevice);

    // Benchmark CPU version
    cout << "\nRunning CPU benchmark..." << endl;
    auto start_cpu = chrono::high_resolution_clock::now();
    for(int i = 0; i < NUM_RUNS; i++) {
        auto result_cpu = hnsw.search_layer(query, hnsw.enter_node_id, EF, 0);
    }
    auto end_cpu = chrono::high_resolution_clock::now();
    auto cpu_time = chrono::duration_cast<chrono::microseconds>(end_cpu - start_cpu).count() / NUM_RUNS;

    // Benchmark GPU version
    cout << "Running GPU benchmark..." << endl;
    auto start_gpu = chrono::high_resolution_clock::now();
    
    for(int i = 0; i < NUM_RUNS; i++) {
        // Launch kernel
        int num_blocks = (NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_distances_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_query, d_points, d_distances, NUM_POINTS, NUM_DIMENSIONS
        );
        cudaDeviceSynchronize();
    }
    
    auto end_gpu = chrono::high_resolution_clock::now();
    auto gpu_time = chrono::duration_cast<chrono::microseconds>(end_gpu - start_gpu).count() / NUM_RUNS;

    // Print results
    cout << "\nBenchmark Results:" << endl;
    cout << "Dataset size: " << NUM_POINTS << " points" << endl;
    cout << "Dimensions: " << NUM_DIMENSIONS << endl;
    cout << "Number of runs: " << NUM_RUNS << endl;
    cout << "Average CPU time: " << cpu_time << " microseconds" << endl;
    cout << "Average GPU time: " << gpu_time << " microseconds" << endl;
    cout << "Speedup: " << static_cast<float>(cpu_time) / gpu_time << "x" << endl;

    // Cleanup
    cudaFree(d_query);
    cudaFree(d_points);
    cudaFree(d_distances);

    return 0;
} 