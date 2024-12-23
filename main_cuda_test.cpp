#include <hnsw.hpp>
#include <utils.hpp>
#include <iostream>
#include <chrono>

#define REPETITIONS 10  // Reduced for testing

using namespace utils;
using namespace hnsw;
using namespace std::chrono;

void test_performance(const Dataset<>& dataset, const Dataset<>& queries, 
                     const vector<vector<int>>& ground_truth,
                     int k, int m, int ef_construction, int ef) {
    
    // Test original version
    auto start = high_resolution_clock::now();
    auto index_original = HNSW(m, ef_construction);
    index_original.build(dataset);
    auto end = high_resolution_clock::now();
    auto build_time_original = duration_cast<milliseconds>(end - start).count();
    cout << "Original Build Time: " << build_time_original << " ms" << endl;

    // Test CUDA version
    start = high_resolution_clock::now();
    auto index_cuda = HNSW(m, ef_construction);
    index_cuda.build(dataset);
    end = high_resolution_clock::now();
    auto build_time_cuda = duration_cast<milliseconds>(end - start).count();
    cout << "CUDA Build Time: " << build_time_cuda << " ms" << endl;

    // Query testing
    long query_time_original = 0;
    long query_time_cuda = 0;
    double avg_recall_original = 0;
    double avg_recall_cuda = 0;

    for (int rep = 0; rep < REPETITIONS; rep++) {
        for (size_t i = 0; i < queries.size(); i++) {
            const auto& query = queries[i];

            // Test original
            start = high_resolution_clock::now();
            auto result_original = index_original.knn_search(query, k, ef);
            end = high_resolution_clock::now();
            query_time_original += duration_cast<microseconds>(end - start).count();
            avg_recall_original += calc_recall(result_original.result, ground_truth[query.id], k);

            // Test CUDA
            start = high_resolution_clock::now();
            auto result_cuda = index_cuda.knn_search(query, k, ef);
            end = high_resolution_clock::now();
            query_time_cuda += duration_cast<microseconds>(end - start).count();
            avg_recall_cuda += calc_recall(result_cuda.result, ground_truth[query.id], k);
        }
    }

    // Calculate averages
    int total_queries = REPETITIONS * queries.size();
    avg_recall_original /= total_queries;
    avg_recall_cuda /= total_queries;
    
    // Print results
    cout << "\nPerformance Comparison (" << total_queries << " queries):" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Original Version:" << endl;
    cout << "  Avg Query Time: " << query_time_original / total_queries << " us" << endl;
    cout << "  Avg Recall: " << avg_recall_original << endl;
    cout << "CUDA Version:" << endl;
    cout << "  Avg Query Time: " << query_time_cuda / total_queries << " us" << endl;
    cout << "  Avg Recall: " << avg_recall_cuda << endl;
    cout << "\nSpeedup:" << endl;
    cout << "  Build: " << (float)build_time_original / build_time_cuda << "x" << endl;
    cout << "  Query: " << (float)query_time_original / query_time_cuda << "x" << endl;
}

int main() {
    const string base_dir = "C:/Users/Recup/OneDrive/Documentos/Contest/hpgda_contest_MM/";
    
    // Parameters
    int k = 100;
    int m = 16;
    int ef_construction = 100;
    int ef = 100;

    // Load SIFT10k dataset
    const string data_path = base_dir + "datasets/siftsmall/siftsmall_base.fvecs";
    const string query_path = base_dir + "datasets/siftsmall/siftsmall_query.fvecs";
    const string ground_truth_path = base_dir + "datasets/siftsmall/siftsmall_groundtruth.ivecs";
    const int n = 10000, n_query = 100;

    cout << "Loading dataset..." << endl;
    const auto dataset = fvecs_read(data_path, n);
    const auto queries = fvecs_read(query_path, n_query);
    const auto ground_truth = load_ivec(ground_truth_path, n_query, k);
    cout << "Dataset loaded." << endl;

    // Run performance test
    test_performance(dataset, queries, ground_truth, k, m, ef_construction, ef);

    return 0;
}