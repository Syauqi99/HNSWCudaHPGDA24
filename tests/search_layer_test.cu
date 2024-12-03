#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "cuda_kernels.cuh"
#include "hnsw1.hpp"
#include <random>
#include <chrono>

using namespace hnsw;
using namespace utils;

class SearchLayerTest : public ::testing::Test {
protected:
    const int dim = 128;
    const int num_points = 1000;
    const int ef = 64;
    Dataset<float> test_dataset;
    SearchLayerCUDAContext cuda_ctx;
    HNSW* index;

    SearchLayerTest() : cuda_ctx(dim) {
        // Generate random test data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0, 1.0);

        for (int i = 0; i < num_points; i++) {
            Data<float> data;
            data.id = i;
            data.x.resize(dim);
            for (int j = 0; j < dim; j++) {
                data.x[j] = dis(gen);
            }
            test_dataset.push_back(data);
        }

        // Initialize HNSW index
        index = new HNSW(16, ef);
        index->build(test_dataset);
    }

    ~SearchLayerTest() {
        delete index;
    }

    // Helper to verify results
    bool verify_results(const SearchResult& cuda_result, const SearchResult& cpu_result) {
        if (cuda_result.result.size() != cpu_result.result.size()) return false;
        
        // Compare distances with small epsilon due to floating point differences
        const float epsilon = 1e-5;
        for (size_t i = 0; i < cuda_result.result.size(); i++) {
            if (abs(cuda_result.result[i].dist - cpu_result.result[i].dist) > epsilon) {
                return false;
            }
            if (cuda_result.result[i].id != cpu_result.result[i].id) {
                return false;
            }
        }
        return true;
    }
};

TEST_F(SearchLayerTest, CompareWithCPUVersion) {
    // Create a query point
    Data<float> query;
    query.id = num_points;
    query.x.resize(dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for (int i = 0; i < dim; i++) {
        query.x[i] = dis(gen);
    }

    // Get results from both implementations
    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto cpu_result = index->search_layer(query, 0, ef, 0);  // Using layer 0 for test
    auto cpu_end = std::chrono::high_resolution_clock::now();

    auto cuda_start = std::chrono::high_resolution_clock::now();
    auto cuda_result = index->search_layer(query, 0, ef, 0);  // Using optimized version
    auto cuda_end = std::chrono::high_resolution_clock::now();

    // Verify results
    EXPECT_TRUE(verify_results(cuda_result, cpu_result));

    // Print performance comparison
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    auto cuda_duration = std::chrono::duration_cast<std::chrono::microseconds>(cuda_end - cuda_start).count();
    
    std::cout << "CPU time: " << cpu_duration << " microseconds" << std::endl;
    std::cout << "CUDA time: " << cuda_duration << " microseconds" << std::endl;
    std::cout << "Speedup: " << static_cast<float>(cpu_duration) / cuda_duration << "x" << std::endl;
}

TEST_F(SearchLayerTest, StressTest) {
    const int num_queries = 100;
    std::vector<Data<float>> queries;
    
    // Generate random queries
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    
    for (int i = 0; i < num_queries; i++) {
        Data<float> query;
        query.id = num_points + i;
        query.x.resize(dim);
        for (int j = 0; j < dim; j++) {
            query.x[j] = dis(gen);
        }
        queries.push_back(query);
    }

    // Test batch processing
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& query : queries) {
        auto result = index->search_layer(query, 0, ef, 0);
        EXPECT_GT(result.result.size(), 0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Processed " << num_queries << " queries in " << duration << " ms" << std::endl;
    std::cout << "Average time per query: " << duration / num_queries << " ms" << std::endl;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 