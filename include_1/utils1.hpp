#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cassert>
#include <functional>
#include <random>
#include <map>
#include "cuda_kernels.cuh"  // Include CUDA functions

using namespace std;

namespace utils {

    template <typename T = float>
    // data constructor
    struct Data {
        // id of the data (in this case this is like the row)
        size_t id;
        // the set of data in float
        std::vector<T> x;

        // if nothing id =0, vector is {0}
        Data() : id(0), x({0}) {}

        Data(size_t i, std::vector<T> v) {
            id = i; // set id equal to the input
            // copy the vectors v to x
            std::copy(v.begin(), v.end(), std::back_inserter(x));
        }

        Data(std::vector<T> v) {
            // if no id then set id = 0
            id = 0;
            // copy the vector only
            std::copy(v.begin(), v.end(), std::back_inserter(x));
        }

        // return the value without copy
        auto& operator [] (size_t i) { return x[i];}

        // return a copy
        const auto& operator [] (size_t i) const { return x[i]; }

        // define the comparison between data
        bool operator==(const Data &o) const {
            // they only compare the id
            if (id == o.id) return true;
            return false;
        }
        // same as before but the opposite
        bool operator!=(const Data &o) const {
            if (id != o.id) return true;
            return false;
        }

        // return size
        size_t size() const { return x.size(); }

        // iterator
        auto begin() const { return x.begin(); }
        auto end() const { return x.end(); }

        // print the data
        void show() const {
            std::cout << id << ": ";
            for (const auto &xi : x) {
                std::cout << xi << ' ';
            }
            std::cout << std::endl;
        }
    };
    
    // template is an alias so if the type is not specified then it will float. Otherwise, it follows the one that they specify
    template <typename T = float>
    // Dataset is the type alias so we can use dataset right away
    using Dataset = vector<Data<T>>;
    template <typename T = float>
    using DistanceFunction = function<float(Data<T>, Data<T>)>;
    // this function wrapper that return float, with two parameters
    // abstract functions

    // Original CPU version of euclidean distance
    template <typename T = float>
    auto euclidean_distance(const Data<T>& p1, const Data<T>& p2) {
        float result = 0;
        for (size_t i = 0; i < p1.size(); i++) {
            result += std::pow(p1[i] - p2[i], 2);
        }
        result = std::sqrt(result);
        return result;
    }

    // CUDA version of euclidean distance
    template <typename T = float>
    float cuda_euclidean_distance(const Data<T>& p1, const Data<T>& p2) {
        return hnsw::cuda_euclidean_distance(p1, p2);
    }

    // Function to get the appropriate distance function based on use_cuda flag
    template <typename T = float>
    DistanceFunction<T> get_distance_function(bool use_cuda = false) {
        if (use_cuda) {
            return DistanceFunction<T>(cuda_euclidean_distance<T>);
        }
        return DistanceFunction<T>(euclidean_distance<T>);
    }

    // Helper function to create a distance function directly
    template <typename T = float>
    DistanceFunction<T> make_cuda_distance() {
        return DistanceFunction<T>(cuda_euclidean_distance<T>);
    }

    // reading the data from files
    Dataset<float> fvecs_read(const std::string& filename, int size) {
        // filename input constructor to read the data
        std::ifstream file(filename, std::ios::binary);
        // there file not found or the data is corrupted return an error
        if (!file.is_open()) {
            throw std::runtime_error("I/O error: Unable to open the file " + filename);
        }

        // Read the vector size
        int d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int));
        int vecsizeof = sizeof(int) + d * sizeof(float);

        // Get the number of vectors
        file.seekg(0, std::ios::end);
        int a = 1;
        int bmax = file.tellg() / vecsizeof;
        int b = bmax;

        b=size;

        assert(a >= 1);
        if (b > bmax) {
            b = bmax;
        }

        if (b == 0 || b < a) {
            return {};
        }

        // Compute the number of vectors that are really read and go to starting positions
        int n = b - a + 1;
        file.seekg((a - 1) * vecsizeof, std::ios::beg);

        // Read n vectors
        std::vector<float> buffer((d + 1) * n);
        file.read(reinterpret_cast<char*>(buffer.data()), (d + 1) * n * sizeof(float));

        // Reshape the vectors
        Dataset<float> dataset;
        for (int i = 0; i < n; ++i) {
            std::vector<float> values;
            for (int j = 0; j < d; ++j) {
                values.push_back(static_cast<float>(buffer[i * (d + 1) + j + 1])); // Start from 1 to skip the dimension value
            }
            dataset.push_back(Data<float>(a + i - 1, values)); // Index starts from 0
        }

        file.close();

        return dataset;
    }

    // count the running time of the code 
    auto get_now() { return chrono::system_clock::now(); }

    auto get_duration(chrono::system_clock::time_point start,
                      chrono::system_clock::time_point end) {
        return chrono::duration_cast<chrono::microseconds>(end - start).count();
    }

    // numeric limit
    constexpr auto double_max = numeric_limits<double>::max();
    constexpr auto double_min = numeric_limits<double>::min();

    constexpr auto float_max = numeric_limits<float>::max();
    constexpr auto float_min = numeric_limits<float>::min();
    // constructor of the neighbor of each nodes
    struct Neighbor {
        float dist;
        int id;

        // optimizes by cuda
        Neighbor() : dist(float_max), id(-1) {}
        Neighbor(float dist, int id) : dist(dist), id(id) {}
    };

    using Neighbors = vector<Neighbor>;

    // for making comparison between two neighbor by distance gives true if less
    struct CompLess {
        constexpr bool operator()(const Neighbor& n1, const Neighbor& n2) const noexcept {
            return n1.dist < n2.dist;
        }
    };

    // for making comparison between two neighbor by distance gives true if more
    struct CompGreater {
        constexpr bool operator()(const Neighbor& n1, const Neighbor& n2) const noexcept {
            return n1.dist > n2.dist;
        }
    };

    // for making comparison between two neighbor by distance gives true if less
    template <typename T>

    // knn linear search 
    // optimizes with CUDA
    auto scan_knn_search(const Data<T>& query, int k, const Dataset<T>& dataset) {
        const auto df = euclidean_distance<float>;
        auto threshold = float_max;

        multimap<float, int> result_map;
        for (const auto& data : dataset) {
            const auto dist = df(query, data);

            if (result_map.size() < k || dist < threshold) {
                result_map.emplace(dist, data.id);
                threshold = (--result_map.cend())->first;
                if (result_map.size() > k) result_map.erase(--result_map.cend());
            }
        }

        vector<Neighbor> result;
        for (const auto& result_pair : result_map) {
            result.emplace_back(result_pair.first, result_pair.second);
        }

        return result;
    }

    auto calc_recall(const Neighbors& actual, const Neighbors& expect, int k) {
        float recall = 0;

        for (int i = 0; i < k; ++i) {
            const auto n1 = actual[i];
            int match = 0;
            for (int j = 0; j < k; ++j) {
                const auto n2 = expect[j];
                if (n1.id != n2.id) continue;
                match = 1;
                break;
            }
            recall += match;
        }

        recall /= actual.size();
        return recall;
    }

    /*
        the .ivec is a raw binary file (little endian)
        where you have consecutive values in the following format
        vector0_size/vec0(0)/vec0(1)/vec0(2)/.../vec0(vec_size-1)/vector1_size/vec1(0) ...
        where each value is an int (4 bytes)
        in the case of ground truth, the vectorX_size is the K of the ANNS procedure
        and each value is the index inside the original dataset of the KNN of the queryX
    */
    vector<Neighbors> load_ivec(const string& neighbor_path, int n, int K) {
        ifstream ifs(neighbor_path, ios::binary); // Open in binary mode
        if (!ifs) throw runtime_error("Can't open file: " + neighbor_path);

        vector<Neighbors> neighbors_list(n); // Vector of Neighbors for each head_id

        // Read data
        for (int i = 0; i < n; ++i) {
            int head_id = i;
            // cout << head_id << ": ";
            // Read K elements for each head_id
            for (int j = 0; j < K; ++j) {
                int neigh;
                ifs.read(reinterpret_cast<char*>(&neigh), sizeof(int)); // Read tail_id

                // cout << neigh << " / ";
                neighbors_list[head_id].emplace_back(42, neigh); // Add Neighbor to corresponding head_id
            }
            // cout << endl;
        }

        return neighbors_list;
    }
}

#endif //UTILS_HPP