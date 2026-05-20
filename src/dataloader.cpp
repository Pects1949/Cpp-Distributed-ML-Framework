#include "cppdist/dataloader.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <stdexcept>
#include <cstring>

namespace cppdist {

// ---------------------------------------------------------------------------
// DataLoader
// ---------------------------------------------------------------------------

DataLoader::DataLoader(const Tensor& full_X, const Tensor& full_y,
                       int batch_size, int rank, int world_size,
                       bool shuffle, unsigned int seed)
    : batch_size_(batch_size), shuffle_(shuffle), seed_(seed) {
    shard(full_X, full_y, rank, world_size);
    reset_epoch();
}

void DataLoader::shard(const Tensor& full_X, const Tensor& full_y,
                       int rank, int world_size) {
    int total = full_X.size(0);
    features_ = full_X.size(1);

    // Evenly distribute samples; last rank gets the remainder
    int base  = total / world_size;
    int start = rank * base;
    int end   = (rank == world_size - 1) ? total : start + base;
    shard_n_  = end - start;

    X_ = Tensor({shard_n_, features_}, false);
    y_ = Tensor({shard_n_}, false);

    const float* sx = full_X.data() + start * features_;
    const float* sy = full_y.data() + start;
    std::copy(sx, sx + shard_n_ * features_, X_.data());
    std::copy(sy, sy + shard_n_, y_.data());
}

void DataLoader::shuffle_indices() {
    std::mt19937 rng(seed_);
    std::shuffle(indices_.begin(), indices_.end(), rng);
    ++seed_;  // different shuffle each epoch
}

void DataLoader::reset_epoch() {
    indices_.resize(shard_n_);
    std::iota(indices_.begin(), indices_.end(), 0);
    if (shuffle_) shuffle_indices();
    cursor_ = 0;
}

int DataLoader::num_batches() const {
    return (shard_n_ + batch_size_ - 1) / batch_size_;
}

bool DataLoader::next_batch(Tensor& batch_X, Tensor& batch_y) {
    if (cursor_ >= shard_n_) return false;
    int end = std::min(cursor_ + batch_size_, shard_n_);
    int bs  = end - cursor_;

    batch_X = Tensor({bs, features_}, false);
    batch_y = Tensor({bs}, false);
    float* bx = batch_X.data();
    float* by = batch_y.data();
    const float* xs = X_.data();
    const float* ys = y_.data();

    for (int i = 0; i < bs; ++i) {
        int idx = indices_[cursor_ + i];
        std::copy(xs + idx * features_, xs + (idx + 1) * features_, bx + i * features_);
        by[i] = ys[idx];
    }
    cursor_ = end;
    return true;
}

// ---------------------------------------------------------------------------
// Synthetic dataset generator
// ---------------------------------------------------------------------------

std::pair<Tensor, Tensor> make_classification_dataset(
    int n_samples, int n_features, int n_classes, float noise, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::uniform_real_distribution<float> uniform(-1.f, 1.f);

    Tensor X({n_samples, n_features}, false);
    Tensor y({n_samples}, false);
    float* xd = X.data();
    float* yd = y.data();

    // One random centroid per class
    std::vector<std::vector<float>> centroids(n_classes, std::vector<float>(n_features));
    for (int c = 0; c < n_classes; ++c)
        for (int f = 0; f < n_features; ++f)
            centroids[c][f] = uniform(rng) * 3.f;

    int per_class = n_samples / n_classes;
    for (int i = 0; i < n_samples; ++i) {
        int cls = std::min(i / per_class, n_classes - 1);
        yd[i] = static_cast<float>(cls);
        for (int f = 0; f < n_features; ++f)
            xd[i * n_features + f] = centroids[cls][f] + normal(rng) * noise;
    }

    // Shuffle samples
    std::vector<int> perm(n_samples);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    Tensor Xs({n_samples, n_features}, false);
    Tensor ys({n_samples}, false);
    for (int i = 0; i < n_samples; ++i) {
        int p = perm[i];
        std::copy(xd + p * n_features, xd + (p + 1) * n_features, Xs.data() + i * n_features);
        ys.data()[i] = yd[p];
    }
    return {Xs, ys};
}

} // namespace cppdist
