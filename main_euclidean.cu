#include <cuda_runtime.h>
#include <utils1.hpp>
#include "cuda_kernels.cuh"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

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
    const int num_tests = 100000;  // Increased number of tests
    const int warmup_runs = 1000;  // Number of warmup runs
    
    cout << "Generating test data..." << endl;
    // Generate test data
    vector<Data<float>> test_data;
    for (int i = 0; i < num_tests; i++) {
        Data<float> data;
        data.x = generate_random_vector(dim);
        data.id = i;
        test_data.push_back(data);
    }
    
    // Warmup runs for CUDA
    cout << "Performing warmup runs..." << endl;
    for (int i = 0; i < warmup_runs; i++) {
        float warmup = to_cuda_euclidean_distance(test_data[0], test_data[1]);
    }
    
    cout << "\nStarting performance tests..." << endl;
    
    // Test CPU version
    auto cpu_start = chrono::high_resolution_clock::now();
    vector<float> cpu_results(num_tests - 1);
    for (int i = 0; i < num_tests - 1; i++) {
        cpu_results[i] = euclidean_distance(test_data[i], test_data[i + 1]);
    }
    auto cpu_end = chrono::high_resolution_clock::now();
    auto cpu_duration = chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start);
    
    // Test CUDA version
    auto cuda_start = chrono::high_resolution_clock::now();
    vector<float> cuda_results(num_tests - 1);
    for (int i = 0; i < num_tests - 1; i++) {
        cuda_results[i] = to_cuda_euclidean_distance(test_data[i], test_data[i + 1]);
    }
    auto cuda_end = chrono::high_resolution_clock::now();
    auto cuda_duration = chrono::duration_cast<chrono::microseconds>(cuda_end - cuda_start);
    
    // Verify results match
    cout << "\nVerifying results..." << endl;
    double max_diff = 0.0;
    double avg_diff = 0.0;
    for (int i = 0; i < num_tests - 1; i++) {
        double diff = abs(cpu_results[i] - cuda_results[i]);
        max_diff = max(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= (num_tests - 1);
    
    // Print detailed results
    cout << fixed << setprecision(6);
    cout << "\nAccuracy Comparison:" << endl;
    cout << "Maximum difference: " << max_diff << endl;
    cout << "Average difference: " << avg_diff << endl;
    
    cout << "\nPerformance Results:" << endl;
    cout << "Number of distance calculations: " << num_tests << endl;
    cout << "CPU time:  " << cpu_duration.count() / 1000.0 << " ms" << endl;
    cout << "CUDA time: " << cuda_duration.count() / 1000.0 << " ms" << endl;
    cout << "Speedup:   " << (float)cpu_duration.count() / cuda_duration.count() << "x" << endl;
    
    cout << "\nPer-Operation Time:" << endl;
    cout << "CPU:  " << cpu_duration.count() / (double)(num_tests - 1) << " us/op" << endl;
    cout << "CUDA: " << cuda_duration.count() / (double)(num_tests - 1) << " us/op" << endl;
    
    return 0;
} 