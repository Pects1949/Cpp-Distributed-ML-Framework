#include "cppdist/tensor.hpp"
#include <cassert>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <random>

namespace cppdist {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int compute_size_impl(const std::vector<int>& s) {
    if (s.empty()) return 0;
    int n = 1;
    for (int d : s) {
        if (d <= 0) throw std::invalid_argument("shape dimensions must be positive");
        n *= d;
    }
    return n;
}

int Tensor::compute_size(const std::vector<int>& s) { return compute_size_impl(s); }

void Tensor::alloc(int n) {
    data_ = std::shared_ptr<float[]>(new float[n]());  // zero-initialized
    size_ = n;
}

void Tensor::ensure_grad() {
    if (!grad_data_)
        grad_data_ = std::shared_ptr<float[]>(new float[size_]());
}

void Tensor::accumulate_grad(const Tensor& delta) {
    ensure_grad();
    assert(delta.numel() == size_);
    float* g = grad_data_.get();
    const float* d = delta.data();
    for (int i = 0; i < size_; ++i) g[i] += d[i];
}

Tensor Tensor::make_output(std::vector<int> shape, bool rg) {
    return Tensor(std::move(shape), rg);
}

// ---------------------------------------------------------------------------
// Constructors and factories
// ---------------------------------------------------------------------------

Tensor::Tensor(std::vector<int> shape, bool requires_grad)
    : shape_(std::move(shape)), size_(compute_size(shape_)), requires_grad_(requires_grad) {
    if (size_ > 0) {
        alloc(size_);
        if (requires_grad_) ensure_grad();
    }
}

Tensor::Tensor(std::vector<int> shape, float fill_value, bool requires_grad)
    : Tensor(std::move(shape), requires_grad) {
    std::fill(data_.get(), data_.get() + size_, fill_value);
}

Tensor Tensor::zeros(std::vector<int> shape, bool requires_grad) {
    return Tensor(std::move(shape), 0.f, requires_grad);
}

Tensor Tensor::ones(std::vector<int> shape, bool requires_grad) {
    return Tensor(std::move(shape), 1.f, requires_grad);
}

Tensor Tensor::randn(std::vector<int> shape, float mean, float stddev,
                     unsigned int seed, bool requires_grad) {
    Tensor t(std::move(shape), requires_grad);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    for (int i = 0; i < t.size_; ++i) t.data_.get()[i] = dist(rng);
    return t;
}

Tensor Tensor::uniform(std::vector<int> shape, float low, float high,
                       unsigned int seed, bool requires_grad) {
    Tensor t(std::move(shape), requires_grad);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    for (int i = 0; i < t.size_; ++i) t.data_.get()[i] = dist(rng);
    return t;
}

Tensor Tensor::from_data(const std::vector<float>& data, std::vector<int> shape,
                         bool requires_grad) {
    int n = compute_size(shape);
    if (static_cast<int>(data.size()) != n)
        throw std::invalid_argument("data size mismatch");
    Tensor t(std::move(shape), requires_grad);
    std::copy(data.begin(), data.end(), t.data_.get());
    return t;
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

int Tensor::size(int dim) const {
    if (dim < 0 || dim >= ndim())
        throw std::out_of_range("dimension out of range");
    return shape_[dim];
}

// ---------------------------------------------------------------------------
// Gradient
// ---------------------------------------------------------------------------

void Tensor::set_requires_grad(bool v) {
    requires_grad_ = v;
    if (v) ensure_grad();
}

Tensor Tensor::grad() const {
    Tensor g;
    g.data_          = grad_data_;
    g.shape_         = shape_;
    g.size_          = size_;
    g.requires_grad_ = false;
    return g;
}

void Tensor::zero_grad() {
    if (grad_data_)
        std::fill(grad_data_.get(), grad_data_.get() + size_, 0.f);
}

// ---------------------------------------------------------------------------
// Backward (autograd entry point)
// ---------------------------------------------------------------------------

void Tensor::backward() {
    Tensor ones_upstream = Tensor::ones(shape_);
    backward(ones_upstream);
}

void Tensor::backward(const Tensor& upstream_grad) {
    if (requires_grad_) accumulate_grad(upstream_grad);
    if (grad_fn_) grad_fn_(upstream_grad);
}

// ---------------------------------------------------------------------------
// Clone / detach
// ---------------------------------------------------------------------------

Tensor Tensor::clone() const {
    Tensor t(shape_, requires_grad_);
    std::copy(data_.get(), data_.get() + size_, t.data_.get());
    if (grad_data_) {
        t.ensure_grad();
        std::copy(grad_data_.get(), grad_data_.get() + size_, t.grad_data_.get());
    }
    return t;
}

Tensor Tensor::detach() const {
    Tensor t(shape_, false);
    std::copy(data_.get(), data_.get() + size_, t.data_.get());
    return t;
}

} // namespace cppdist
