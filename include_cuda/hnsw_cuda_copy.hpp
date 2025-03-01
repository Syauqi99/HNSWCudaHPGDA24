#ifndef HNSW_CUDA_HPP
#define HNSW_CUDA_HPP

#include <queue>
#include <utils_cuda.hpp>
#include <random>
#include <cuda_runtime.h>
#include <cstdio>  // For fprintf
#include <cstdlib> // For exit
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(err); \
    } \
} while(0)

using namespace std;
using namespace utils;

namespace hnsw {
    // Add GPU-optimized structures
    struct GpuNode {
        float* data;  // Raw pointer instead of Data<> reference
        int* neighbor_ids;
        float* neighbor_distances;
        int num_neighbors;
    };

    struct GpuData {
        float* vectors;  // Contiguous array of vectors
        int dimensions;
        int num_vectors;
    };

    __global__ void calculateDistances(
        const float* query,
        const float* all_vectors,
        const int* indices,
        float* distances,
        int dim,
        int num_neighbors
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_neighbors) return;
        
        int vector_idx = indices[idx];
        float distance = 0.0f;
        
        // Access vectors using correct stride
        const float* vector = all_vectors + (vector_idx * dim);
        for (int i = 0; i < dim; i++) {
            float diff = vector[i] - query[i];
            distance += diff * diff;
        }
        
        distances[idx] = sqrtf(distance);
    }

    __global__ void printVectors(const float* vectors, const int* neighbor_indices, int dim, int num_neighbors) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Boundary check for thread index
        if (idx >= num_neighbors) return;
        
        int neighbor_idx = neighbor_indices[idx];  // Get the index of the current neighbor
        
        // Print each element of the vector
        for (int i = 0; i < dim; i++) {
            printf("Thread %d: Neighbor Index %d, Element Index %d, Value %f\n", 
                idx, neighbor_idx, i, vectors[neighbor_idx * dim + i]);
        }
    }
    
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

    struct HNSWCuda {
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

        // Add GPU memory management
        GpuData d_dataset;  // Device dataset
        vector<GpuNode> d_nodes;  // Device nodes
        
        // Add CUDA stream for async operations
        cudaStream_t stream;
        
        // Add GPU buffer members
        float* d_query_buffer;
        float* d_neighbor_buffer;
        float* d_distances_buffer;
        size_t buffer_size;
        const int BATCH_SIZE = 1024;  // Move BATCH_SIZE as class member
        #define MAX_DIM 128  // Adjust based on your maximum dimension

        // Add new member for storing all vectors
        float* d_all_vectors;
        int vector_dim;
        int total_vectors;

        // Add new member for pinned host memory
        float* h_pinned_vectors;

        // Modify these constants
        const int MAX_VECTORS_PER_BATCH = 2048;  // Increased from 1024
        const int MAX_NEIGHBORS_PER_NODE = 256;  // New constant

        // New members for dynamic buffer management
        size_t total_buffer_size;
        size_t vector_buffer_size;

        HNSWCuda(int m, int ef_construction = 64, bool extend_candidates = false, bool keep_pruned_connections = true) :
                m(m), m_max_0(m * 2), m_l(1 / log(1.0 * m)),
                enter_node_id(-1), enter_node_level(-1),
                ef_construction(ef_construction),
                extend_candidates(extend_candidates),
                keep_pruned_connections(keep_pruned_connections),
                calc_dist(euclidean_distance<float>),
                engine(42), unif_dist(0.0, 1.0) {
            // Initialize CUDA stream
            cudaStreamCreate(&stream);

            // Calculate buffer sizes dynamically
            vector_buffer_size = MAX_VECTORS_PER_BATCH * MAX_DIM * sizeof(float);
            total_buffer_size = vector_buffer_size * 2; // Double for safety

            // Allocate larger buffers
            CUDA_CHECK(cudaMalloc(&d_query_buffer, MAX_DIM * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_distances_buffer, MAX_VECTORS_PER_BATCH * sizeof(float)));
            CUDA_CHECK(cudaMallocHost(&h_pinned_vectors, vector_buffer_size));

            // Add error checking
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA allocation error: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("CUDA allocation failed");
            }
        }

        ~HNSWCuda() {
            if (d_all_vectors) {
                CUDA_CHECK(cudaFree(d_all_vectors));
            }
            CUDA_CHECK(cudaFree(d_query_buffer));
            CUDA_CHECK(cudaFree(d_distances_buffer));

            // Add pinned memory cleanup
            CUDA_CHECK(cudaFreeHost(h_pinned_vectors));
            cudaStreamDestroy(stream);
        }

        const Node& get_enter_node() const { return layers.back()[enter_node_id]; }

        int get_new_node_level() {
            return static_cast<int>(-log(unif_dist(engine)) * m_l);
        }

        auto search_layer_cuda(const Data<>& query, int start_node_id, int ef, int l_c) {
            auto result = SearchResult();
            vector<bool> visited(dataset.size(), false);
            visited[start_node_id] = true;

            priority_queue<Neighbor, vector<Neighbor>, CompGreater> candidates;
            priority_queue<Neighbor, vector<Neighbor>, CompLess> top_candidates;

            // Use pinned memory for the query
            float* h_pinned_query;
            CUDA_CHECK(cudaMallocHost(&h_pinned_query, query.x.size() * sizeof(float)));
            memcpy(h_pinned_query, query.x.data(), query.x.size() * sizeof(float));

            // Copy query once using pinned memory
            CUDA_CHECK(cudaMemcpyAsync(d_query_buffer, h_pinned_query, 
                                       query.x.size() * sizeof(float), 
                                       cudaMemcpyHostToDevice, stream));

            const auto& start_node = layers[l_c][start_node_id];
            const auto dist_from_en = calc_dist(query, start_node.data);

            candidates.emplace(dist_from_en, start_node_id);
            top_candidates.emplace(dist_from_en, start_node_id);

            while (!candidates.empty()) {
                vector<int> batch_indices;
                
                // Collect batch of indices
                while (!candidates.empty() && batch_indices.size() < BATCH_SIZE) {
                    const auto nearest = candidates.top();
                    candidates.pop();

                    if (nearest.dist > top_candidates.top().dist) break;

                    for (const auto& neighbor : layers[l_c][nearest.id].neighbors) {
                        if (!visited[neighbor.id]) {
                            batch_indices.push_back(neighbor.id);
                            visited[neighbor.id] = true;
                        }
                    }
                }

                if (!batch_indices.empty()) {
                    int numNeighbors = batch_indices.size();
                    
                    // Copy indices to GPU
                    int* d_batch_indices;
                    CUDA_CHECK(cudaMalloc(&d_batch_indices, numNeighbors * sizeof(int)));
                    CUDA_CHECK(cudaMemcpy(d_batch_indices, batch_indices.data(),
                                          numNeighbors * sizeof(int),
                                          cudaMemcpyHostToDevice));

                    int blockSize = 256;
                    int numBlocks = (numNeighbors + blockSize - 1) / blockSize;
                    if (numBlocks == 0) numBlocks = 1;  // Ensure at least one block

                    // Calculate distances using pre-stored vectors
                    calculateDistances<<<numBlocks, blockSize, 0, stream>>>(
                        d_query_buffer,
                        d_all_vectors,
                        d_batch_indices,
                        d_distances_buffer,
                        vector_dim,
                        numNeighbors
                    );

                    // Check for kernel launch errors
                    CUDA_CHECK(cudaGetLastError());

                    // Use pinned memory for distances
                    float* h_pinned_distances;
                    CUDA_CHECK(cudaMallocHost(&h_pinned_distances, numNeighbors * sizeof(float)));

                    CUDA_CHECK(cudaMemcpyAsync(h_pinned_distances, d_distances_buffer,
                                               numNeighbors * sizeof(float),
                                               cudaMemcpyDeviceToHost, stream));
                    
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaFree(d_batch_indices));

                    // Process results
                    for (size_t i = 0; i < batch_indices.size(); i++) {
                        float dist = h_pinned_distances[i];
                        int id = batch_indices[i];

                        if (dist < top_candidates.top().dist || top_candidates.size() < ef) {
                            candidates.emplace(dist, id);
                            top_candidates.emplace(dist, id);

                            if (top_candidates.size() > ef) top_candidates.pop();
                        }
                    }

                    // Free pinned memory for distances
                    CUDA_CHECK(cudaFreeHost(h_pinned_distances));
                }
            }

            while (!top_candidates.empty()) {
                result.result.emplace_back(top_candidates.top());
                top_candidates.pop();
            }

            reverse(result.result.begin(), result.result.end());

            // Free pinned memory for query
            CUDA_CHECK(cudaFreeHost(h_pinned_query));

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

        void batch_insert(const vector<Data<>>& batch) {
            // Pre-compute levels for all nodes in batch
            vector<int> node_levels(batch.size());
            for (size_t i = 0; i < batch.size(); i++) {
                node_levels[i] = get_new_node_level();
            }
            
            int max_level = *max_element(node_levels.begin(), node_levels.end());
            
            // Process each node in batch
            for (size_t i = 0; i < batch.size(); i++) {
                const auto& new_data = batch[i];
                int l_new_node = node_levels[i];
                
                // Register node in layer maps
                for (int l_c = l_new_node; l_c >= 0; --l_c) {
                    layer_map[l_c].emplace_back(new_data.id);
                }
                
                // Navigate down through layers to find entry point
                auto start_node_id = enter_node_id;
                for (int l_c = enter_node_level; l_c > l_new_node; --l_c) {
                    const auto nn_layer = search_layer_cuda(new_data, start_node_id, 1, l_c).result[0];
                    start_node_id = nn_layer.id;
                }
                
                // Process each layer
                for (int l_c = min(enter_node_level, l_new_node); l_c >= 0; --l_c) {
                    auto neighbors = search_layer_cuda(new_data, start_node_id, ef_construction, l_c).result;
                    
                    if (neighbors.size() > m) {
                        neighbors = select_neighbors_heuristic(new_data, neighbors, m, l_c);
                    }
                    
                    auto& layer = layers[l_c];
                    for (const auto neighbor : neighbors) {
                        if (neighbor.id == new_data.id) continue;
                        
                        auto& neighbor_node = layer[neighbor.id];
                        layer[new_data.id].neighbors.emplace_back(neighbor);
                        neighbor_node.neighbors.emplace_back(neighbor.dist, new_data.id);
                        
                        const auto m_max = l_c ? m : m_max_0;
                        if (neighbor_node.neighbors.size() > m_max) {
                            neighbor_node.neighbors = select_neighbors_heuristic(
                                neighbor_node.data,
                                neighbor_node.neighbors,
                                m_max,
                                l_c
                            );
                        }
                    }
                    
                    if (l_c == 0) break;
                    start_node_id = neighbors[0].id;
                }
                
                // Update entry point if needed
                if (l_new_node > enter_node_level) {
                    enter_node_id = new_data.id;
                    layers.resize(l_new_node + 1);
                    for (int l_c = max(enter_node_level, 0); l_c <= l_new_node; ++l_c) {
                        for (const auto& data : dataset) {
                            layers[l_c].emplace_back(data);
                        }
                    }
                    enter_node_level = l_new_node;
                }
            }
        }

        void build(const Dataset<>& dataset_) {
            dataset = dataset_;
            vector_dim = dataset_[0].x.size();
            total_vectors = dataset_.size();
            
            if (vector_dim > MAX_DIM) {
                throw std::runtime_error("Vector dimension exceeds MAX_DIM");
            }

            // Allocate GPU memory in chunks
            size_t total_size = static_cast<size_t>(total_vectors) * vector_dim * sizeof(float);
            CUDA_CHECK(cudaMalloc(&d_all_vectors, total_size));

            // Process in smaller chunks
            const size_t vectors_per_chunk = MAX_VECTORS_PER_BATCH;
            const size_t num_chunks = (total_vectors + vectors_per_chunk - 1) / vectors_per_chunk;

            for (size_t chunk = 0; chunk < num_chunks; chunk++) {
                const size_t start_idx = chunk * vectors_per_chunk;
                const size_t end_idx = std::min(start_idx + vectors_per_chunk, static_cast<size_t>(total_vectors));
                const size_t chunk_size = end_idx - start_idx;

                // Copy chunk to pinned memory
                for (size_t i = 0; i < chunk_size; i++) {
                    const auto& data = dataset_[start_idx + i];
                    memcpy(h_pinned_vectors + i * vector_dim,
                          data.x.data(),
                          vector_dim * sizeof(float));
                }

                // Async copy to GPU
                CUDA_CHECK(cudaMemcpyAsync(d_all_vectors + start_idx * vector_dim,
                                         h_pinned_vectors,
                                         chunk_size * vector_dim * sizeof(float),
                                         cudaMemcpyHostToDevice,
                                         stream));
                
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // Process insertions in batches
            vector<Data<>> batch;
            batch.reserve(MAX_VECTORS_PER_BATCH);

            for (const auto& data : dataset) {
                batch.push_back(data);
                if (batch.size() >= MAX_VECTORS_PER_BATCH) {
                    batch_insert(batch);
                    batch.clear();
                }
            }

            if (!batch.empty()) {
                batch_insert(batch);
            }

            cout << "Index construction completed." << endl;
        }

        auto knn_search_cuda(const Data<>& query, int k, int ef) {
            //std::cout << "running" << std::endl;

            SearchResult result;
            //std::cout << "running result " << std::endl;
            // search in upper layers
            auto start_id_layer = enter_node_id;
            //std::cout << "start id layer" << std::endl;

            for (int l_c = enter_node_level; l_c >= 1; --l_c) {
                //std::cout << "Before calling search_layer" << std::endl;
                const auto result_layer = search_layer_cuda(query, start_id_layer, 1, l_c);
                const auto& nn_id_layer = result_layer.result[0].id;
                start_id_layer = nn_id_layer;
            }

            const auto& nn_upper_layer = layers[1][start_id_layer];

            // search in base layer
            //std::cout << "Before calling search_layer 2" << std::endl;
            const auto result_layer = search_layer_cuda(query, start_id_layer, ef, 0);
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