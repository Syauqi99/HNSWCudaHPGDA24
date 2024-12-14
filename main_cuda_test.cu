#include <hnsw_cuda.hpp>
#include <utils_cuda.hpp>

#define REPETITIONS 1

using namespace utils;
using namespace hnsw;

int main() {
    const string base_dir = "/content/hpgda_contest_MM/";

  int k = 100;
  int m = 16;
  int ef_construction = 100;
  int ef = 100;

  // SIFT10k (small) - 10,000	base / 100 query / 128 dim
  const string data_path = base_dir + "datasets/siftsmall/siftsmall_base.fvecs";
  const string query_path =
      base_dir + "datasets/siftsmall/siftsmall_query.fvecs";
  const string ground_truth_path =
      base_dir + "datasets/siftsmall/siftsmall_groundtruth.ivecs";
  const int n = 1000, n_query = 1;

  // // SIFT1M (normal) - 1,000,000	base / 10,000 query / 128 dim
  // const string data_path = base_dir + "datasets/sift/sift_base.fvecs";
  // const string query_path = base_dir + "datasets/sift/sift_query.fvecs";
  // const string ground_truth_path = base_dir +
  // "datasets/sift/sift_groundtruth.ivecs"; const int n = 1000000, n_query =
  // 10000;

  const auto dataset = fvecs_read(data_path, n);
  const auto queries = fvecs_read(query_path, n_query);
  const auto ground_truth = load_ivec(ground_truth_path, n_query, k);

  const auto start = get_now();
  auto index = HNSWCuda(m, ef_construction);
  index.build(dataset);
  const auto end = get_now();
  const auto build_time = get_duration(start, end);
  cout << "index_construction: " << build_time / 1000 << " [ms]" << endl;

  long total_queries = 0;
  SearchResults results(n_query);
  // Simulating REPETITIONS * n_query search procedure (during dev you can
  // reduce REPETITIONS = 1 for fast testing)
  for (int rep = 0; rep < REPETITIONS; rep++) {
    for (int i = 0; i < n_query; i++) {
      const auto& query = queries[i];

      auto q_start = get_now();
      auto result = index.knn_search_cuda(query, k, ef);
      auto q_end = get_now();
      total_queries += get_duration(q_start, q_end);

      result.recall = calc_recall(result.result, ground_truth[query.id], k);
      results[i] = result;
    }
  }
  cout << "results: " << results << endl;
  
  cout << "time for " << REPETITIONS * n_query
       << " queries: " << total_queries / 1000 << " [ms]" << endl;

  const string save_name =
      "k" + to_string(k) + "-m" + to_string(m) + "-ef" + to_string(ef) + ".csv";
  const string result_base_dir = base_dir + "results/";
  const string log_path = result_base_dir + "log-" + save_name;
  const string result_path = result_base_dir + "result-" + save_name;
  results.save(log_path, result_path);
} 