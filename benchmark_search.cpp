#include <iostream>
#include <chrono>
#include "hnsw1.hpp"
#include "utils1.hpp"

using namespace std;
using namespace hnsw;
using namespace utils;

int main() {
    // Test parameters
    const int n_vectors = 1000;  // Number of vectors in dataset
    const int dim = 128;         // Dimension of vectors
    const int m = 16;            // HNSW parameter
    const int ef = 64;           // HNSW parameter
    const int k = 10;            // Number of nearest neighbors to find
    const int n_queries = 100;   // Number of queries to test

    // Generate random dataset
    Dataset<float> dataset;
    for(int i = 0; i < n_vectors; i++) {
        vector<float> vec(dim);
        for(int j = 0; j < dim; j++) {
            vec[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        dataset.emplace_back(Data<float>(vec, i));
    }

    // Build index
    cout << "Building index..." << endl;
    auto index = HNSW(m, ef);
    index.build(dataset);

    // Generate random queries
    Dataset<float> queries;
    for(int i = 0; i < n_queries; i++) {
        vector<float> vec(dim);
        for(int j = 0; j < dim; j++) {
            vec[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        queries.emplace_back(Data<float>(vec, i));
    }

    // Test CPU version
    cout << "\nTesting CPU version..." << endl;
    auto cpu_start = get_now();
    for(const auto& query : queries) {
        auto result = index.search_layer(query, 0, ef, 0);
    }
    auto cpu_end = get_now();
    auto cpu_duration = get_duration(cpu_start, cpu_end);

    // Test GPU version
    cout << "Testing GPU version..." << endl;
    auto gpu_start = get_now();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for(const auto& query : queries) {
        auto result = index.hibrid_search_layer(query, 0, ef, 0);
    }
    auto gpu_end = get_now();
    auto gpu_duration = get_duration(gpu_start, gpu_end);

    cudaStreamDestroy(stream);

    // Print results
    cout << "\nResults:" << endl;
    cout << "CPU time: " << cpu_duration / 1000.0 << " ms" << endl;
    cout << "GPU time: " << gpu_duration / 1000.0 << " ms" << endl;
    cout << "Speedup: " << static_cast<float>(cpu_duration) / gpu_duration << "x" << endl;

    return 0;
} 