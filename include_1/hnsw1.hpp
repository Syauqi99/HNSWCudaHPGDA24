#ifndef HNSW_HNSW_HPP
#define HNSW_HNSW_HPP

#include <queue>
#include <utils1.hpp>
#include <random>
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

using namespace std::chrono;

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void restVectorsInto(float *result, float *a, float *b, int N)
{
  // Process multiple elements per thread
  const int elementsPerThread = 4;
  int index = (threadIdx.x + blockIdx.x * blockDim.x) * elementsPerThread;
  
  #pragma unroll
  for(int i = 0; i < elementsPerThread && index + i < N; i++)
  {
    int idx = index + i;
    if (idx < N) {
      float diff = a[idx] - b[idx];
      result[idx] = diff * diff;
    }
  }
}

float process_distance_vector(float *distances, int N) {
    float sum = 0.0f;  // Initialize sum
    for(int i = 0; i < N; i++) {
        sum += distances[i];
    }
    sum = sqrt(sum);
    return sum;
}

using namespace std;
using namespace utils;

// Host-side function to process neighbors
void processNeighborsCuda(
    const Data<>& query,
    const vector<int>& neighbor_ids,
    vector<bool>& visited,
    float current_top_dist,
    int ef,
    cudaStream_t& stream,
    priority_queue<Neighbor, vector<Neighbor>, CompGreater>& candidates,
    priority_queue<Neighbor, vector<Neighbor>, CompLess>& top_candidates
) {
    const int num_neighbors = neighbor_ids.size();
    const int dim = query.size();
    
    // Allocate device memory
    float *d_query, *d_layer_data, *d_distances;
    int *d_neighbor_ids, *d_valid_neighbors;
    bool *d_visited;
    
    cudaMalloc(&d_query, dim * sizeof(float));
    cudaMalloc(&d_layer_data, this->dataset.size() * dim * sizeof(float));
    cudaMalloc(&d_distances, num_neighbors * sizeof(float));
    cudaMalloc(&d_neighbor_ids, num_neighbors * sizeof(int));
    cudaMalloc(&d_valid_neighbors, num_neighbors * sizeof(int));
    cudaMalloc(&d_visited, this->dataset.size() * sizeof(bool));
    
    // Copy data to device
    cudaMemcpyAsync(d_query, query.x.data(), dim * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_layer_data, this->dataset.data(), this->dataset.size() * dim * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_neighbor_ids, neighbor_ids.data(), num_neighbors * sizeof(int),
                   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_visited, visited.data(), visited.size() * sizeof(bool),
                   cudaMemcpyHostToDevice, stream);
    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (num_neighbors + threadsPerBlock - 1) / threadsPerBlock;
    
    processNeighborsKernel<<<numBlocks, threadsPerBlock, dim * sizeof(float), stream>>>(
        d_query, d_layer_data, d_neighbor_ids, num_neighbors,
        d_visited, current_top_dist, ef, d_distances, d_valid_neighbors, dim
    );
    
    // Allocate host memory for results
    vector<float> distances(num_neighbors);
    vector<int> valid_neighbors(num_neighbors);
    
    // Copy results back to host
    cudaMemcpyAsync(distances.data(), d_distances, num_neighbors * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(valid_neighbors.data(), d_valid_neighbors, num_neighbors * sizeof(int),
                   cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(visited.data(), d_visited, visited.size() * sizeof(bool),
                   cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // Process results
    for (int i = 0; i < num_neighbors; i++) {
        if (valid_neighbors[i]) {
            candidates.emplace(distances[i], neighbor_ids[i]);
            top_candidates.emplace(distances[i], neighbor_ids[i]);
            
            if (top_candidates.size() > ef) {
                top_candidates.pop();
            }
        }
    }
        
    // Cleanup
    cudaFree(d_query);
    cudaFree(d_layer_data);
    cudaFree(d_distances);
    cudaFree(d_neighbor_ids);
    cudaFree(d_valid_neighbors);
    cudaFree(d_visited);
}

__global__ void processNeighborsKernel(
    const float* query_data,
    const float* layer_data,
    const int* neighbor_ids,
    const int num_neighbors,
    bool* visited,
    float current_top_dist,
    int ef,
    float* distances,
    int* valid_neighbors,
    const int dim
) {
    extern __shared__ float shared_query[];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < dim) {
        shared_query[tid] = query_data[tid];
    }
    __syncthreads();
    
    if (tid < num_neighbors) {
        int neighbor_id = neighbor_ids[tid];
        
        unsigned int* visited_uint = reinterpret_cast<unsigned int*>(&visited[neighbor_id]);
        if (atomicCAS(visited_uint, 0U, 1U) == 0U) {
            float dist = 0.0f;
            const float* neighbor_data = layer_data + neighbor_id * dim;
            
            #pragma unroll 4
            for (int d = 0; d < dim; d++) {
                float diff = shared_query[d] - neighbor_data[d];
                dist += diff * diff;
            }
            dist = sqrtf(dist);
            
            distances[tid] = dist;
            valid_neighbors[tid] = (dist < current_top_dist || ef > 0) ? 1 : 0;
        } else {
            valid_neighbors[tid] = 0;
            distances[tid] = INFINITY;
        }
    }
}

vector<int> neighbors_to_ids(const Neighbors& neighbors) {
    vector<int> ids;
    ids.reserve(neighbors.size());
    for (const auto& n : neighbors) {
        ids.push_back(n.id);
    }
    return ids;
}

namespace hnsw {
    struct Node {
        const Data<>& data;
        Neighbors neighbors;

        explicit Node(const Data<>& data_) : data(data_) {}
    };

    using Layer = vector<Node>;

    struct SearchResult {
        Neighbors result;
        double recall = 0;
    };

    struct SearchResults {
        vector<SearchResult> results;

        SearchResults(size_t size) : results(size) {}
        void push_back(const SearchResult& result) { results.emplace_back(result); }
        void push_back(SearchResult&& result) { results.emplace_back(move(result)); }
        decltype(auto) operator [] (int i) { return results[i]; }

        void save(const string& log_path, const string& result_path) {
            ofstream log_ofs(log_path);
            string line = "query_id,recall";
            log_ofs << line << endl;

            ofstream result_ofs(result_path);
            line = "query_id,data_id,dist";
            result_ofs << line << endl;

            int query_id = 0;
            for (const auto& result : results) {
                log_ofs << query_id << ","<< result.recall << endl;

                for (const auto& neighbor : result.result) {
                    result_ofs << query_id << ","
                               << neighbor.id << ","
                               << neighbor.dist << endl;
                }

                query_id++;
            }
        }
    };

    struct HNSW {
        const int m, m_max_0, ef_construction;
        const double m_l;
        const bool extend_candidates, keep_pruned_connections;

        int enter_node_id;
        int enter_node_level;
        vector<Layer> layers;
        map<int, vector<int>> layer_map;
        Dataset<> dataset;
        DistanceFunction<> calc_dist;

        mt19937 engine;
        uniform_real_distribution<double> unif_dist;

        // CUDA variables
        float *d_x, *d_y; // Independent and dependent values on device

        HNSW(int m, int ef_construction = 64, bool extend_candidates = false, bool keep_pruned_connections = true) :
                m(m), m_max_0(m * 2), m_l(1 / log(1.0 * m)),
                enter_node_id(-1), enter_node_level(-1),
                ef_construction(ef_construction),
                extend_candidates(extend_candidates),
                keep_pruned_connections(keep_pruned_connections),
                calc_dist(euclidean_distance<float>),
                engine(42), unif_dist(0.0, 1.0) {}

        const Node& get_enter_node() const { return layers.back()[enter_node_id]; }

        int get_new_node_level() {
            return static_cast<int>(-log(unif_dist(engine)) * m_l);
        }

        auto hibrid_search_layer(const Data<>& query, int start_node_id, int ef, int l_c) {
            auto result = SearchResult();

            vector<bool> visited(dataset.size());
            visited[start_node_id] = true;

            priority_queue<Neighbor, vector<Neighbor>, CompGreater> candidates;
            priority_queue<Neighbor, vector<Neighbor>, CompLess> top_candidates;

            // Create CUDA stream
            cudaStream_t stream;
            cudaStreamCreate(&stream);

            const auto& start_node = layers[l_c][start_node_id];
            const auto dist_from_en = calc_dist(query, start_node.data);

            candidates.emplace(dist_from_en, start_node_id);
            top_candidates.emplace(dist_from_en, start_node_id);

            while (!candidates.empty()) {
                const auto nearest_candidate = candidates.top();
                const auto& nearest_candidate_node = layers[l_c][nearest_candidate.id];
                candidates.pop();

                if (nearest_candidate.dist > top_candidates.top().dist) break;

                // Process neighbors using CUDA
                auto neighbor_ids = neighbors_to_ids(nearest_candidate_node.neighbors);
                processNeighborsCuda(
                    query,
                    neighbor_ids,
                    visited,
                    top_candidates.top().dist,
                    ef,
                    stream,
                    candidates,     
                    top_candidates   
                );
            }

            // Build result from top candidates
            while (!top_candidates.empty()) {
                result.result.emplace_back(top_candidates.top());
                top_candidates.pop();
            }

            // Cleanup CUDA resources
            cudaStreamDestroy(stream);
            return result;
        }

        auto search_layer(const Data<>& query, int start_node_id, int ef, int l_c) {
            auto result = SearchResult();

            vector<bool> visited(dataset.size());
            visited[start_node_id] = true;

            priority_queue<Neighbor, vector<Neighbor>, CompGreater> candidates;
            priority_queue<Neighbor, vector<Neighbor>, CompLess> top_candidates;

            const auto& start_node = layers[l_c][start_node_id];
            const auto dist_from_en = calc_dist(query, start_node.data);

            candidates.emplace(dist_from_en, start_node_id);
            top_candidates.emplace(dist_from_en, start_node_id);

            while (!candidates.empty()) {
                const auto nearest_candidate = candidates.top();
                const auto& nearest_candidate_node = layers[l_c][nearest_candidate.id];
                candidates.pop();
                // i changed someting
                if (nearest_candidate.dist > top_candidates.top().dist) break;

                for (const auto neighbor : nearest_candidate_node.neighbors) {
                    if (visited[neighbor.id]) continue;
                    visited[neighbor.id] = true;

                    const auto& neighbor_node = layers[l_c][neighbor.id];
                    const auto dist_from_neighbor = calc_dist(query, neighbor_node.data);

                    if (dist_from_neighbor < top_candidates.top().dist ||
                        top_candidates.size() < ef) {
                        candidates.emplace(dist_from_neighbor, neighbor.id);
                        top_candidates.emplace(dist_from_neighbor, neighbor.id);

                        if (top_candidates.size() > ef) top_candidates.pop();
                    }
                }
            }

            while (!top_candidates.empty()) {
                result.result.emplace_back(top_candidates.top());
                top_candidates.pop();
            }

            reverse(result.result.begin(), result.result.end());

            return result;
        }

        auto select_neighbors_heuristic(const Data<>& query, vector<Neighbor> initial_candidates, int n_neighbors, int l_c) {
            const auto& layer = layers[l_c];
            priority_queue<Neighbor, vector<Neighbor>, CompGreater>
                    candidates, discarded_candidates;

            vector<bool> added(dataset.size());
            added[query.id] = true;

            // init candidates
            for (const auto& candidate : initial_candidates) {
                if (added[candidate.id]) continue;
                added[candidate.id] = true;
                candidates.emplace(candidate);
            }

            if (extend_candidates) {
                for (const auto& candidate : initial_candidates) {
                    if (added[candidate.id]) continue;
                    added[candidate.id] = true;

                    const auto& candidate_node = layer[candidate.id];
                    for (const auto& neighbor : candidate_node.neighbors) {
                        const auto& neighbor_node = layer[neighbor.id];
                        const auto dist_from_neighbor =
                                calc_dist(query, neighbor_node.data);
                        candidates.emplace(dist_from_neighbor, neighbor.id);
                    }
                }
            }

            // init neighbors
            vector<Neighbor> neighbors;
            neighbors.emplace_back(candidates.top());
            candidates.pop();

            // select edge
            while (!candidates.empty() && neighbors.size() < n_neighbors) {
                const auto candidate = candidates.top();
                candidates.pop();
                const auto& candidate_node = layer[candidate.id];

                bool good = true;
                for (const auto& neighbor : neighbors) {
                    const auto& neighbor_node = layer[neighbor.id];
                    const auto dist = calc_dist(candidate_node.data, neighbor_node.data);

                    if (dist < candidate.dist) {
                        good = false;
                        break;
                    }
                }

                if (good) neighbors.emplace_back(candidate);
                else discarded_candidates.emplace(candidate);
            }

            if (keep_pruned_connections) {
                while (!discarded_candidates.empty() && neighbors.size() < n_neighbors) {
                    neighbors.emplace_back(discarded_candidates.top());
                    discarded_candidates.pop();
                }
            }

            return neighbors;
        }

        void insert(const Data<>& new_data) {
            auto l_new_node = get_new_node_level(); // all levels below that will have the node in it
            for (int l_c = l_new_node; l_c >= 0; --l_c)
                layer_map[l_c].emplace_back(new_data.id); // register the ID of that node in the map for all the lower layers

            // navigate the layers from the start_node_layer down to the layer where the new node needs to be added
            auto start_node_id = enter_node_id;
            for (int l_c = enter_node_level; l_c > l_new_node; --l_c) {
                // get the nearest neighbour node to the new node in the current layer 
                const auto nn_layer = search_layer(new_data, start_node_id, 1, l_c).result[0];
                start_node_id = nn_layer.id;
            }
            // when the previous cycle ends, we have that start_node_id == entry point node in the layer where we want to add the new node
            // i.e. we have the entry point to the layer of the new node

            // now we iterate from this layer down to layer zero 
            for (int l_c = min(enter_node_level, l_new_node); l_c >= 0; --l_c) {
                // compute the NN nodes in the current layer with respect to the node that i want to insert
                // neighbors will be populated with the nearest neighbors nodes (and corresponding distances) to the new_data node
                // the entry_point for computing the NN nodes is start_node_id which is the result of the 
                // previous greedy search from the top layer to the current one
                auto neighbors = search_layer(new_data, start_node_id, ef_construction, l_c).result;

                // if the NN list is greater than M, filter them out with an heuristic
                if (neighbors.size() > m)
                    neighbors = select_neighbors_heuristic(new_data, neighbors, m, l_c);

                auto& layer = layers[l_c];
                // for each neighbor
                for (const auto neighbor : neighbors) {
                    if (neighbor.id == new_data.id)
                        continue;
                    // get the Node class (the list of neighbors contains only IDs)
                    auto& neighbor_node = layer[neighbor.id];

                    // add a bidirectional edge
                    layer[new_data.id].neighbors.emplace_back(neighbor); // link the new node to its NN
                    neighbor_node.neighbors.emplace_back(neighbor.dist, new_data.id); // add the link from the neighbour to the new node

                    const auto m_max = l_c ? m : m_max_0; // if we are in layer 0 m_max should be equal to m_max of layer 0 (which is different from the normal M)

                    // if the neighbor now has more than the allowed maximum of neighbours (we have added new node to its list)
                    if (neighbor_node.neighbors.size() > m_max) {
                        // compute a restricted set and update the list
                        const auto new_neighbor_neighbors = select_neighbors_heuristic(neighbor_node.data, neighbor_node.neighbors, m_max, l_c);
                        neighbor_node.neighbors = new_neighbor_neighbors;
                    }
                }

                if (l_c == 0) // if we arrived at layer 0 we can skip the following step
                    break;
                
                // repeat the operation taking into consideration the nearest neighbour of new_data  
                start_node_id = neighbors[0].id;
            }

            // when we finish the previous loop, we have added the new Node to the graph

            // if new node is top (this can either happen at the first element added or if we are adding a new layer)
            if (layers.empty() || l_new_node > enter_node_level) {
                // change enter node
                enter_node_id = new_data.id;

                // add new layer
                layers.resize(l_new_node + 1);
                for (int l_c = max(enter_node_level, 0); l_c <= l_new_node; ++l_c) {
                    for (const auto& data : dataset) {
                        layers[l_c].emplace_back(data);
                    }
                }
                enter_node_level = l_new_node;
            }
        }

        void build(const Dataset<>& dataset_) {
            dataset = dataset_;
            for (const auto& data : dataset){
                insert(data);
            }
            cout << "Index construction completed." << endl;
        }

        auto knn_search(const Data<>& query, int k, int ef) {
            SearchResult result;
            // search in upper layers
            auto start_id_layer = enter_node_id;
            for (int l_c = enter_node_level; l_c >= 1; --l_c) {
                const auto result_layer = search_layer(query, start_id_layer, 1, l_c);
                const auto& nn_id_layer = result_layer.result[0].id;
                start_id_layer = nn_id_layer;
            }

            const auto& nn_upper_layer = layers[1][start_id_layer];

            // search in base layer
            const auto result_layer = search_layer(query, start_id_layer, ef, 0);
            const auto candidates = result_layer.result;
            for (const auto& candidate : candidates) {
                result.result.emplace_back(candidate);
                if (result.result.size() >= k) break;
            }

            return result;
        }
    };
}

#endif //HNSW_HNSW_HPP
