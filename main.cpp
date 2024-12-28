#include <hnsw.hpp>
#include <utils.hpp>
#include <fstream>
#include <vector>
#include <iostream>

using namespace utils;
using namespace hnsw;

// Function to save total query times
void save_total_query_times(const std::string& file_path, const std::vector<long>& total_query_times) {
    std::ofstream ofs(file_path);
    ofs << "repetition,total_time_ns,total_time_ms" << std::endl;
    for (size_t rep = 0; rep < total_query_times.size(); ++rep) {
        ofs << rep << "," << total_query_times[rep] << "," << total_query_times[rep] / 1000.0 << std::endl;
    }
    ofs.close();
}

int main(int argc, char* argv[]) {
    const std::string base_dir = "/content/hpgda_contest_MM/";

    // Default values
    int k = 100;
    int m = 16;
    int ef_construction = 100;
    int ef = 100;
    int n = 1000;
    int n_query = 1;
    int repetitions = 1;

    // Parse command-line arguments
    if (argc > 1) k = std::stoi(argv[1]);
    if (argc > 2) m = std::stoi(argv[2]);
    if (argc > 3) ef_construction = std::stoi(argv[3]);
    if (argc > 4) ef = std::stoi(argv[4]);
    if (argc > 5) n = std::stoi(argv[5]);
    if (argc > 6) n_query = std::stoi(argv[6]);
    if (argc > 7) repetitions = std::stoi(argv[7]);

    const std::string data_path = base_dir + "datasets/siftsmall/siftsmall_base.fvecs";
    const std::string query_path = base_dir + "datasets/siftsmall/siftsmall_query.fvecs";
    const std::string ground_truth_path = base_dir + "datasets/siftsmall/siftsmall_groundtruth.ivecs";

    const auto dataset = fvecs_read(data_path, n);
    const auto queries = fvecs_read(query_path, n_query);
    const auto ground_truth = load_ivec(ground_truth_path, n_query, k);

    const auto start = get_now();
    auto index = HNSW(m, ef_construction);
    index.build(dataset);
    const auto end = get_now();
    const auto build_time = get_duration(start, end);
    std::cout << "index_construction: " << build_time / 1000 << " [ms]" << std::endl;

    std::vector<long> total_query_times;
    SearchResults results(n_query);
    for (int rep = 0; rep < repetitions; rep++) {
        long total_queries = 0;
        for (int i = 0; i < n_query; i++) {
            const auto& query = queries[i];

            auto q_start = get_now();
            auto result = index.knn_search(query, k, ef);
            auto q_end = get_now();
            total_queries += get_duration(q_start, q_end);

            result.recall = calc_recall(result.result, ground_truth[query.id], k);
            results[i] = result;
        }
        total_query_times.push_back(total_queries);
    }
    long total_queries = std::accumulate(total_query_times.begin(), total_query_times.end(), 0);
    std::cout << "time for " << repetitions * n_query << " queries: " << total_queries / 1000 << " [ms]" << std::endl;

    const std::string save_name = "k" + std::to_string(k) + "-m" + std::to_string(m) + "-ef" + std::to_string(ef) +
                                  "-n" + std::to_string(n) + "-nq" + std::to_string(n_query) + "-rep" + std::to_string(repetitions) + ".csv";
    const std::string result_base_dir = base_dir + "results/";
    const std::string log_path = result_base_dir + "log_cpu-" + save_name;
    const std::string result_path = result_base_dir + "result_cpu-" + save_name;
    results.save(log_path, result_path);

    const std::string times_path = result_base_dir + "times_cpu-" + save_name;
    save_total_query_times(times_path, total_query_times);

    return 0;
}