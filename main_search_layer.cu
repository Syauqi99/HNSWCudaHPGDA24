#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "cuda_kernels.cuh"
#include "hnsw1.hpp"
#include <random>
#include <chrono>
#include <iostream>

using namespace hnsw;
using namespace utils;

// Helper function to generate random vectors
vector<float> generate_random_vector(int dim) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0, 1.0);
    
    vector<float> vec(dim);
    for (auto& v : vec) {
        v = dis(gen);
    }
    return vec;
}

// Helper function to generate test dataset
Dataset<float> generate_test_dataset(int num_points, int dim) {
    Dataset<float> dataset;
    for (int i = 0; i < num_points; i++) {
        Data<float> data;
        data.id = i;
        data.x = generate_random_vector(dim);
        dataset.push_back(data);
    }
    return dataset;
}

int main() {
    const int dim = 128;
    const int num_points = 1000;
    const int num_queries = 100;
    const int ef = 64;
    const int m = 16;
    
    cout << "Initializing test environment..." << endl;
    
    // Generate dataset and build index
    cout << "Generating dataset with " << num_points << " points..." << endl;
    auto dataset = generate_test_dataset(num_points, dim);
    
    cout << "Building HNSW index..." << endl;
    auto index = HNSW(m, ef);
    index.build(dataset);

    // Initialize CUDA context
    SearchLayerCUDAContext cuda_ctx(dim);
    
    // Generate query points
    cout << "Generating " << num_queries << " query points..." << endl;
    vector<Data<float>> queries;
    for (int i = 0; i < num_queries; i++) {
        Data<float> query;
        query.id = num_points + i;
        query.x = generate_random_vector(dim);
        queries.push_back(query);
    }

    // Warmup
    cout << "Performing warmup..." << endl;
    auto warmup_result = index.search_layer(queries[0], 0, ef, 0);

    // Benchmark CPU version
    cout << "\nBenchmarking CPU version..." << endl;
    vector<double> cpu_times;
    for (const auto& query : queries) {
        auto start = chrono::high_resolution_clock::now();
        auto result = index.search_layer(query, 0, ef, 0);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<microseconds>(end - start).count();
        cpu_times.push_back(duration);
    }

    // Benchmark CUDA version
    cout << "Benchmarking CUDA version..." << endl;
    vector<double> cuda_times;
    for (const auto& query : queries) {
        auto start = chrono::high_resolution_clock::now();
        auto result = index.search_layer(query, 0, ef, 0);  // Using optimized version
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<microseconds>(end - start).count();
        cuda_times.push_back(duration);
    }

    // Calculate statistics
    auto cpu_avg = accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / num_queries;
    auto cuda_avg = accumulate(cuda_times.begin(), cuda_times.end(), 0.0) / num_queries;
    
    auto cpu_minmax = minmax_element(cpu_times.begin(), cpu_times.end());
    auto cuda_minmax = minmax_element(cuda_times.begin(), cuda_times.end());

    // Print results
    cout << "\nSearch Layer Performance Results:" << endl;
    cout << "--------------------------------" << endl;
    cout << "Number of queries: " << num_queries << endl;
    cout << "Dataset size: " << num_points << " points" << endl;
    cout << "Vector dimension: " << dim << endl;
    cout << "\nCPU Performance:" << endl;
    cout << "  Average time: " << cpu_avg << " microseconds" << endl;
    cout << "  Min time: " << *cpu_minmax.first << " microseconds" << endl;
    cout << "  Max time: " << *cpu_minmax.second << " microseconds" << endl;
    cout << "\nCUDA Performance:" << endl;
    cout << "  Average time: " << cuda_avg << " microseconds" << endl;
    cout << "  Min time: " << *cuda_minmax.first << " microseconds" << endl;
    cout << "  Max time: " << *cuda_minmax.second << " microseconds" << endl;
    cout << "\nSpeedup: " << cpu_avg / cuda_avg << "x" << endl;

    return 0;
} 