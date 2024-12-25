#include <hnsw_cuda.hpp>
#include <hnsw_pinned.hpp>
#include <hnsw_stream.hpp>
#include <hnsw_vectors.hpp>
#include <hnsw_batch.hpp>
#include <hnsw_allocation.hpp>
#include <utils_cuda.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#define REPETITIONS 1

using namespace utils;
using namespace hnsw;

template<typename T>
void run_benchmark(const string& name, T& index, 
                  const Dataset<>& queries, 
                  const vector<vector<int>>& ground_truth,
                  const string& result_base_dir,
                  int k, int m, int ef, int n, int n_query) {
    
    long total_queries = 0;
    SearchResults results(n_query);
    vector<long> repetition_times;

    for (int rep = 0; rep < REPETITIONS; rep++) {
        long rep_time = 0;
        for (int i = 0; i < n_query; i++) {
            const auto& query = queries[i];

            auto q_start = get_now();
            auto result = index.search_layer_cuda(query, k, ef, 0);
            auto q_end = get_now();
            long query_time = get_duration(q_start, q_end);
            rep_time += query_time;
            total_queries += query_time;

            result.recall = calc_recall(result.result, ground_truth[query.id], k);
            results[i] = result;
        }
        repetition_times.push_back(rep_time);
    }
    
    cout << name << " time for " << REPETITIONS * n_query 
         << " queries: " << total_queries / 1000 << " [ms]" << endl;

    const string save_name = "n" + to_string(n) + "-nq" + to_string(n_query) + 
                           "-k" + to_string(k) + "-m" + to_string(m) + 
                           "-ef" + to_string(ef) + ".csv";
    
    // Save timing results
    const string times_path = result_base_dir + "times_" + name + "-" + save_name;
    ofstream times_ofs(times_path);
    times_ofs << "repetition,time_ns,time_ms" << endl;
    for (int rep = 0; rep < REPETITIONS; rep++) {
        times_ofs << rep << "," << repetition_times[rep] << "," << repetition_times[rep]/1000.0 << endl;
    }
    times_ofs.close();

    // Save search results
    const string log_path = result_base_dir + "log_" + name + "-" + save_name;
    const string result_path = result_base_dir + "result_" + name + "-" + save_name;
    results.save(log_path, result_path);
}

void run_all_implementations(const string& base_dir, int k, int m, int ef_construction, 
                           int ef, int n, int n_query) {
    cout << "\nRunning benchmarks for n=" << n << ", n_query=" << n_query << endl;
    cout << "----------------------------------------" << endl;

    const string data_path = base_dir + "datasets/siftsmall/siftsmall_base.fvecs";
    const string query_path = base_dir + "datasets/siftsmall/siftsmall_query.fvecs";
    const string ground_truth_path = base_dir + "datasets/siftsmall/siftsmall_groundtruth.ivecs";

    // // SIFT1M (normal) - 1,000,000	base / 10,000 query / 128 dim
    // const string data_path = base_dir + "datasets/sift/sift_base.fvecs";
    // const string query_path = base_dir + "datasets/sift/sift_query.fvecs";
    // const string ground_truth_path = base_dir +
    // "datasets/sift/sift_groundtruth.ivecs"; 

    const string result_base_dir = base_dir + "results/";

    // Load dataset
    cout << "Loading dataset..." << endl;
    const auto dataset = fvecs_read(data_path, n);
    const auto queries = fvecs_read(query_path, n_query);
    const auto ground_truth = load_ivec(ground_truth_path, n_query, k);

    // Run CUDA implementation
    cout << "\nRunning CUDA implementation..." << endl;
    {
        auto start = get_now();
        auto index = HNSWCuda(m, ef_construction);
        index.build(dataset);
        auto end = get_now();
        cout << "CUDA index construction: " << get_duration(start, end) / 1000 << " [ms]" << endl;
        run_benchmark("cuda", index, queries, ground_truth, result_base_dir, k, m, ef, n, n_query);
    }

    // Run Pinned implementation
    cout << "\nRunning Pinned implementation..." << endl;
    {
        auto start = get_now();
        auto index = HNSWPinned(m, ef_construction);
        index.build(dataset);
        auto end = get_now();
        cout << "Pinned index construction: " << get_duration(start, end) / 1000 << " [ms]" << endl;
        run_benchmark("pinned", index, queries, ground_truth, result_base_dir, k, m, ef, n, n_query);
    }

    // Run Stream implementation
    cout << "\nRunning Stream implementation..." << endl;
    {
        auto start = get_now();
        auto index = HNSWStream(m, ef_construction);
        index.build(dataset);
        auto end = get_now();
        cout << "Stream index construction: " << get_duration(start, end) / 1000 << " [ms]" << endl;
        run_benchmark("stream", index, queries, ground_truth, result_base_dir, k, m, ef, n, n_query);
    }

    // Run Vectors implementation
    cout << "\nRunning Vectors implementation..." << endl;
    {
        auto start = get_now();
        auto index = HNSWVectors(m, ef_construction);
        index.build(dataset);
        auto end = get_now();
        cout << "Vectors index construction: " << get_duration(start, end) / 1000 << " [ms]" << endl;
        run_benchmark("vectors", index, queries, ground_truth, result_base_dir, k, m, ef, n, n_query);
    }

    // Run Batch implementation
    cout << "\nRunning Batch implementation..." << endl;
    {
        auto start = get_now();
        auto index = HSNWBatch(m, ef_construction);
        index.build(dataset);
        auto end = get_now();
        cout << "Batch index construction: " << get_duration(start, end) / 1000 << " [ms]" << endl;
        run_benchmark("batch", index, queries, ground_truth, result_base_dir, k, m, ef, n, n_query);
    }

    // Run Allocation implementation
    cout << "\nRunning Allocation implementation..." << endl;
    {
        auto start = get_now();
        auto index = HSNWAllocation(m, ef_construction);
        index.build(dataset);
        auto end = get_now();
        cout << "Allocation index construction: " << get_duration(start, end) / 1000 << " [ms]" << endl;
        run_benchmark("allocation", index, queries, ground_truth, result_base_dir, k, m, ef, n, n_query);
    }
}

int main() {
    const string base_dir = "/content/hpgda_contest_MM/";
    
    int k = 100;
    int m = 16;
    int ef_construction = 100;
    int ef = 100;

    // Define different dataset sizes and query counts to test
    vector<pair<int, int>> configurations = {
        {1000, 1},     
        {1000, 10},    
        {10000, 100},    
    };

    for (const auto& config : configurations) {
        int n = config.first;
        int n_query = config.second;
        run_all_implementations(base_dir, k, m, ef_construction, ef, n, n_query);
    }

    cout << "\nAll benchmarks completed!" << endl;
    return 0;
} 