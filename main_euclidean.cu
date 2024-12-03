#include <cuda_runtime.h>
#include <utils1.hpp>
#include "cuda_kernels.cuh"
#include <chrono>
#include <random>
#include <iostream>

using namespace std;
using namespace utils;

// Function to generate random vector data
vector<float> generate_random_vector(int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0, 1.0);
    
    vector<float> vec(size);
    for (auto& v : vec) {
        v = dis(gen);
    }
    return vec;
}

int main() {
    const int dim = 128;  // Dimension of vectors (same as SIFT)
    const int num_tests = 10000;  // Number of test cases
    
    // Generate test data
    vector<Data<float>> test_data;
    for (int i = 0; i < num_tests; i++) {
        Data<float> data;
        data.x = generate_random_vector(dim);
        data.id = i;
        test_data.push_back(data);
    }
    
    // Test CPU version
    auto cpu_start = chrono::high_resolution_clock::now();
    for (int i = 0; i < num_tests - 1; i++) {
        float cpu_dist = euclidean_distance(test_data[i], test_data[i + 1]);
    }
    auto cpu_end = chrono::high_resolution_clock::now();
    auto cpu_duration = chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start);
    
    // Test CUDA version
    auto cuda_start = chrono::high_resolution_clock::now();
    for (int i = 0; i < num_tests - 1; i++) {
        float cuda_dist = to_cuda_euclidean_distance(test_data[i], test_data[i + 1]);
    }
    auto cuda_end = chrono::high_resolution_clock::now();
    auto cuda_duration = chrono::duration_cast<chrono::microseconds>(cuda_end - cuda_start);
    
    // Verify results match
    cout << "\nTesting correctness with first pair:" << endl;
    float cpu_result = euclidean_distance(test_data[0], test_data[1]);
    float cuda_result = to_cuda_euclidean_distance(test_data[0], test_data[1]);
    cout << "CPU result:  " << cpu_result << endl;
    cout << "CUDA result: " << cuda_result << endl;
    cout << "Difference:  " << abs(cpu_result - cuda_result) << endl;
    
    // Print performance comparison
    cout << "\nPerformance comparison for " << num_tests << " distance calculations:" << endl;
    cout << "CPU time:  " << cpu_duration.count() / 1000.0 << " ms" << endl;
    cout << "CUDA time: " << cuda_duration.count() / 1000.0 << " ms" << endl;
    cout << "Speedup:   " << (float)cpu_duration.count() / cuda_duration.count() << "x" << endl;
    
    return 0;
} 