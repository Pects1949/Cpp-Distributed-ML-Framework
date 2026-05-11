#pragma once
#include "tensor.hpp"
#include <vector>
#include <utility>

namespace cppdist {

class DataLoader {
public:
    // full_X: [total_samples, features], full_y: [total_samples]
    // Automatically shards by rank, shuffles each epoch
    DataLoader(const Tensor& full_X, const Tensor& full_y,
               int batch_size, int rank, int world_size,
               bool shuffle = true, unsigned int seed = 42);

    // Fill batch_X [batch, features] and batch_y [batch]; return false at epoch end
    bool next_batch(Tensor& batch_X, Tensor& batch_y);

    void reset_epoch();   // re-shuffle and reset cursor

    int num_samples() const { return shard_n_; }
    int num_batches()  const;

private:
    Tensor X_;         // this rank's shard [shard_n_, features]
    Tensor y_;         // this rank's shard [shard_n_]
    int batch_size_;
    bool shuffle_;
    unsigned int seed_;
    int shard_n_;
    int features_;

    std::vector<int> indices_;
    int cursor_{0};

    void shard(const Tensor& full_X, const Tensor& full_y,
               int rank, int world_size);
    void shuffle_indices();
};

// Generate synthetic linearly-separable classification dataset.
// Returns (X [n_samples, n_features], y [n_samples]) with class labels 0..n_classes-1.
std::pair<Tensor, Tensor> make_classification_dataset(
    int n_samples, int n_features, int n_classes = 2,
    float noise = 0.1f, unsigned int seed = 0);

} // namespace cppdist
