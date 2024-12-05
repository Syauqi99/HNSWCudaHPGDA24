#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>

// Define a simple Data structure to hold the data
template <typename T = float>
struct Data {
    std::vector<T> x;

    Data(std::vector<T> v) : x(v) {}

    size_t size() const { return x.size(); }

    const T& operator[](size_t i) const { return x[i]; }
};

// Euclidean distance function
template <typename T = float>
float euclidean_distance(const Data<T>& p1, const Data<T>& p2) {
    float result = 0;
    for (size_t i = 0; i < p1.size(); i++) {
        result += std::pow(p1[i] - p2[i], 2);
    }
    result = std::sqrt(result);
    return result;
}

int main() {
    const int N = 1000000;  // Size of arrays

    // Initialize vectors with some values
    std::vector<float> a(N);
    std::vector<float> b(N);
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Create Data objects
    Data<float> data_a(a);
    Data<float> data_b(b);
    
    // Calculate Euclidean distance
    float result = euclidean_distance(data_a, data_b);
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Result: " << result << std::endl;
    std::cout << "Time taken: " << duration << " microseconds" << std::endl;

    return 0;
} 