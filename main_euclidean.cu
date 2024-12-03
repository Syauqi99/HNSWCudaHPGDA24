#include <cuda_runtime.h>
#include <utils1.hpp>
#include "cuda_kernels.cuh"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace utils;
using namespace hnsw;

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
    const int dim = 128;
    const int num_tests = 100000;
    const int batch_size = 1024;  // Process in batches
    const int warmup_runs = 1000;
    
    cout << "Generating test data..." << endl;
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
    #pragma omp parallel for
    for (int i = 0; i < num_tests - 1; i++) {
        cpu_results[i] = euclidean_distance(test_data[i], test_data[i + 1]);
    }
    auto cpu_end = chrono::high_resolution_clock::now();
    auto cpu_duration = chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start);
    
    // Test CUDA version with batching
    auto cuda_start = chrono::high_resolution_clock::now();
    vector<float> cuda_results(num_tests - 1);
    
    for (int batch = 0; batch < (num_tests - 1); batch += batch_size) {
        int current_batch_size = min(batch_size, num_tests - 1 - batch);
        vector<vector<float>> batch_vectors1, batch_vectors2;
        
        // Prepare batch data
        for (int i = 0; i < current_batch_size; i++) {
            batch_vectors1.push_back(test_data[batch + i].x);
            batch_vectors2.push_back(test_data[batch + i + 1].x);
        }
        
        // Process batch
        vector<float> batch_results = batch_cuda_euclidean_distance(
            batch_vectors1, batch_vectors2, current_batch_size);
            
        // Store results
        copy(batch_results.begin(), batch_results.end(), 
             cuda_results.begin() + batch);
    }
    
    auto cuda_end = chrono::high_resolution_clock::now();
    auto cuda_duration = chrono::duration_cast<chrono::microseconds>(cuda_end - cuda_start);
    
    // Print results and verification
    cout << "\nVerifying results..." << endl;
    double max_diff = 0.0;
    double avg_diff = 0.0;
    for (int i = 0; i < num_tests - 1; i++) {
        double diff = abs(cpu_results[i] - cuda_results[i]);
        max_diff = max(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= (num_tests - 1);
    
    cout << fixed << setprecision(6);
    cout << "\nAccuracy Comparison:" << endl;
    cout << "Maximum difference: " << max_diff << endl;
    cout << "Average difference: " << avg_diff << endl;
    
    cout << "\nPerformance Results:" << endl;
    cout << "Number of distance calculations: " << num_tests << endl;
    cout << "CPU time:  " << cpu_duration.count() / 1000.0 << " ms" << endl;
    cout << "CUDA time: " << cuda_duration.count() / 1000.0 << " ms" << endl;
    cout << "Speedup:   " << (float)cpu_duration.count() / cuda_duration.count() << "x" << endl;
    
    return 0;
} 