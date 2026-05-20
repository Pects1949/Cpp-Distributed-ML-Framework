#include "cppdist/tensor.hpp"
#include <cassert>
#include <algorithm>
#include <numeric>
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
// Arithmetic — each op captures inputs by shallow copy and attaches grad_fn_
// ---------------------------------------------------------------------------

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_)
        throw std::invalid_argument("operator+ shape mismatch; use add_bias for broadcast");
    bool rg = requires_grad_ || other.requires_grad_;
    Tensor out = make_output(shape_, rg);
    const float* a = data_.get(); const float* b = other.data_.get(); float* c = out.data_.get();
    for (int i = 0; i < size_; ++i) c[i] = a[i] + b[i];

    if (rg) {
        Tensor sa = *this, sb = other;
        out.grad_fn_ = [sa, sb](const Tensor& dc) mutable {
            if (sa.requires_grad_) sa.backward(dc);
            if (sb.requires_grad_) sb.backward(dc);
        };
    }
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_)
        throw std::invalid_argument("operator- shape mismatch");
    bool rg = requires_grad_ || other.requires_grad_;
    Tensor out = make_output(shape_, rg);
    const float* a = data_.get(); const float* b = other.data_.get(); float* c = out.data_.get();
    for (int i = 0; i < size_; ++i) c[i] = a[i] - b[i];

    if (rg) {
        Tensor sa = *this, sb = other;
        out.grad_fn_ = [sa, sb](const Tensor& dc) mutable {
            if (sa.requires_grad_) sa.backward(dc);
            if (sb.requires_grad_) {
                Tensor neg = make_output(dc.shape_, false);
                const float* g = dc.data(); float* d = neg.data();
                for (int i = 0; i < dc.numel(); ++i) d[i] = -g[i];
                sb.backward(neg);
            }
        };
    }
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_)
        throw std::invalid_argument("operator* shape mismatch");
    bool rg = requires_grad_ || other.requires_grad_;
    Tensor out = make_output(shape_, rg);
    const float* a = data_.get(); const float* b = other.data_.get(); float* c = out.data_.get();
    for (int i = 0; i < size_; ++i) c[i] = a[i] * b[i];

    if (rg) {
        Tensor sa = *this, sb = other;
        out.grad_fn_ = [sa, sb](const Tensor& dc) mutable {
            if (sa.requires_grad_) {
                Tensor d = make_output(dc.shape_, false);
                const float* g = dc.data(); const float* bv = sb.data(); float* dv = d.data();
                for (int i = 0; i < dc.numel(); ++i) dv[i] = g[i] * bv[i];
                sa.backward(d);
            }
            if (sb.requires_grad_) {
                Tensor d = make_output(dc.shape_, false);
                const float* g = dc.data(); const float* av = sa.data(); float* dv = d.data();
                for (int i = 0; i < dc.numel(); ++i) dv[i] = g[i] * av[i];
                sb.backward(d);
            }
        };
    }
    return out;
}

Tensor Tensor::operator*(float s) const {
    bool rg = requires_grad_;
    Tensor out = make_output(shape_, rg);
    const float* a = data_.get(); float* c = out.data_.get();
    for (int i = 0; i < size_; ++i) c[i] = a[i] * s;

    if (rg) {
        Tensor sa = *this;
        out.grad_fn_ = [sa, s](const Tensor& dc) mutable {
            Tensor d = make_output(dc.shape_, false);
            const float* g = dc.data(); float* dv = d.data();
            for (int i = 0; i < dc.numel(); ++i) dv[i] = g[i] * s;
            sa.backward(d);
        };
    }
    return out;
}

Tensor operator*(float s, const Tensor& t) { return t * s; }

Tensor Tensor::operator/(float s) const { return (*this) * (1.f / s); }

// ---------------------------------------------------------------------------
// Shape manipulation: reshape, sum, mean
// ---------------------------------------------------------------------------

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    int new_n = compute_size(new_shape);
    if (new_n != size_) throw std::invalid_argument("reshape: total size mismatch");
    bool rg = requires_grad_;
    Tensor out;
    out.data_ = data_; out.grad_data_ = grad_data_;
    out.shape_ = std::move(new_shape); out.size_ = size_; out.requires_grad_ = rg;
    if (rg) {
        Tensor sa = *this;
        out.grad_fn_ = [sa](const Tensor& dc) mutable {
            Tensor dself;
            dself.data_ = dc.data_; dself.grad_data_ = nullptr;
            dself.shape_ = sa.shape_; dself.size_ = sa.size_; dself.requires_grad_ = false;
            sa.backward(dself);
        };
    }
    return out;
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    if (dim == -1) {
        bool rg = requires_grad_;
        Tensor out = make_output({1}, rg);
        float total = 0.f;
        const float* a = data_.get();
        for (int i = 0; i < size_; ++i) total += a[i];
        out.data_.get()[0] = total;
        if (rg) {
            Tensor sa = *this;
            out.grad_fn_ = [sa](const Tensor& dc) mutable {
                float g = dc.data()[0];
                Tensor d = Tensor::ones(sa.shape_) * g;
                sa.backward(d);
            };
        }
        return out;
    }
    if (dim < 0 || dim >= ndim()) throw std::out_of_range("sum: dim out of range");
    std::vector<int> out_shape;
    for (int i = 0; i < ndim(); ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape_[i]);
    }
    if (out_shape.empty()) out_shape = {1};
    bool rg = requires_grad_;
    Tensor out = make_output(out_shape, rg);
    float* o = out.data_.get(); const float* a = data_.get();
    if (ndim() == 2) {
        int m = shape_[0], n = shape_[1];
        if (dim == 0) {
            for (int j = 0; j < n; ++j) { float s = 0.f; for (int i = 0; i < m; ++i) s += a[i*n+j]; o[j] = s; }
        } else {
            for (int i = 0; i < m; ++i) { float s = 0.f; for (int j = 0; j < n; ++j) s += a[i*n+j]; o[i] = s; }
        }
    } else if (ndim() == 1) {
        float s = 0.f; for (int i = 0; i < size_; ++i) s += a[i]; o[0] = s;
    } else {
        throw std::runtime_error("sum: only 1D/2D supported");
    }
    if (rg) {
        Tensor sa = *this; int sdim = dim;
        out.grad_fn_ = [sa, sdim](const Tensor& dc) mutable {
            Tensor d = make_output(sa.shape_, false);
            float* dv = d.data(); const float* g = dc.data();
            if (sa.ndim() == 2) {
                int m = sa.shape_[0], n = sa.shape_[1];
                if (sdim == 0) { for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) dv[i*n+j] = g[j]; }
                else           { for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) dv[i*n+j] = g[i]; }
            } else { for (int i = 0; i < sa.numel(); ++i) dv[i] = g[0]; }
            sa.backward(d);
        };
    }
    return out;
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    if (dim == -1) return sum(-1, keepdim) * (1.f / static_cast<float>(size_));
    return sum(dim, keepdim) * (1.f / static_cast<float>(shape_[dim]));
}

// ---------------------------------------------------------------------------
// Activation ops and bias add
// ---------------------------------------------------------------------------

Tensor Tensor::relu() const {
    bool rg = requires_grad_;
    Tensor out = make_output(shape_, rg);
    const float* a = data_.get(); float* b = out.data_.get();
    for (int i = 0; i < size_; ++i) b[i] = a[i] > 0.f ? a[i] : 0.f;
    if (rg) {
        Tensor sa = *this;
        out.grad_fn_ = [sa](const Tensor& dc) mutable {
            Tensor d = make_output(sa.shape_, false);
            const float* inp = sa.data(); const float* g = dc.data(); float* dv = d.data();
            for (int i = 0; i < sa.numel(); ++i) dv[i] = (inp[i] > 0.f) ? g[i] : 0.f;
            sa.backward(d);
        };
    }
    return out;
}

Tensor Tensor::sigmoid() const {
    bool rg = requires_grad_;
    Tensor out = make_output(shape_, rg);
    const float* a = data_.get(); float* b = out.data_.get();
    for (int i = 0; i < size_; ++i) b[i] = 1.f / (1.f + std::exp(-a[i]));
    if (rg) {
        Tensor so = out, sa = *this;
        out.grad_fn_ = [so, sa](const Tensor& dc) mutable {
            Tensor d = make_output(sa.shape_, false);
            const float* s = so.data(); const float* g = dc.data(); float* dv = d.data();
            for (int i = 0; i < sa.numel(); ++i) dv[i] = g[i] * s[i] * (1.f - s[i]);
            sa.backward(d);
        };
    }
    return out;
}

Tensor Tensor::tanh_act() const {
    bool rg = requires_grad_;
    Tensor out = make_output(shape_, rg);
    const float* a = data_.get(); float* b = out.data_.get();
    for (int i = 0; i < size_; ++i) b[i] = std::tanh(a[i]);
    if (rg) {
        Tensor so = out, sa = *this;
        out.grad_fn_ = [so, sa](const Tensor& dc) mutable {
            Tensor d = make_output(sa.shape_, false);
            const float* t = so.data(); const float* g = dc.data(); float* dv = d.data();
            for (int i = 0; i < sa.numel(); ++i) dv[i] = g[i] * (1.f - t[i] * t[i]);
            sa.backward(d);
        };
    }
    return out;
}

Tensor Tensor::exp_t() const {
    bool rg = requires_grad_;
    Tensor out = make_output(shape_, rg);
    const float* a = data_.get(); float* b = out.data_.get();
    for (int i = 0; i < size_; ++i) b[i] = std::exp(a[i]);
    if (rg) {
        Tensor so = out, sa = *this;
        out.grad_fn_ = [so, sa](const Tensor& dc) mutable {
            Tensor d = make_output(sa.shape_, false);
            const float* e = so.data(); const float* g = dc.data(); float* dv = d.data();
            for (int i = 0; i < sa.numel(); ++i) dv[i] = g[i] * e[i];
            sa.backward(d);
        };
    }
    return out;
}

Tensor Tensor::log_t() const {
    bool rg = requires_grad_;
    Tensor out = make_output(shape_, rg);
    const float* a = data_.get(); float* b = out.data_.get();
    for (int i = 0; i < size_; ++i) b[i] = std::log(a[i] + 1e-12f);
    if (rg) {
        Tensor sa = *this;
        out.grad_fn_ = [sa](const Tensor& dc) mutable {
            Tensor d = make_output(sa.shape_, false);
            const float* inp = sa.data(); const float* g = dc.data(); float* dv = d.data();
            for (int i = 0; i < sa.numel(); ++i) dv[i] = g[i] / (inp[i] + 1e-12f);
            sa.backward(d);
        };
    }
    return out;
}

Tensor Tensor::softmax(int dim) const {
    if (ndim() != 2 || dim != 1)
        throw std::runtime_error("softmax: only 2D tensors with dim=1 supported");
    int m = shape_[0], n = shape_[1];
    bool rg = requires_grad_;
    Tensor out = make_output(shape_, rg);
    const float* a = data_.get(); float* b = out.data_.get();
    for (int i = 0; i < m; ++i) {
        const float* row = a + i * n; float* orow = b + i * n;
        float maxv = *std::max_element(row, row + n);
        float sum = 0.f;
        for (int j = 0; j < n; ++j) { orow[j] = std::exp(row[j] - maxv); sum += orow[j]; }
        for (int j = 0; j < n; ++j) orow[j] /= sum;
    }
    if (rg) {
        Tensor so = out, sa = *this;
        out.grad_fn_ = [so, sa, m, n](const Tensor& dc) mutable {
            Tensor d = make_output(sa.shape_, false);
            const float* s = so.data(); const float* g = dc.data(); float* dv = d.data();
            for (int i = 0; i < m; ++i) {
                const float* si = s + i*n; const float* gi = g + i*n; float* di = dv + i*n;
                float dot = 0.f; for (int j = 0; j < n; ++j) dot += si[j] * gi[j];
                for (int j = 0; j < n; ++j) di[j] = si[j] * (gi[j] - dot);
            }
            sa.backward(d);
        };
    }
    return out;
}

Tensor Tensor::add_bias(const Tensor& bias) const {
    if (ndim() != 2 || bias.ndim() != 1 || shape_[1] != bias.size_)
        throw std::invalid_argument("add_bias: shape mismatch");
    int m = shape_[0], n = shape_[1];
    bool rg = requires_grad_ || bias.requires_grad_;
    Tensor out = make_output(shape_, rg);
    const float* a = data_.get(); const float* b = bias.data_.get(); float* c = out.data_.get();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            c[i * n + j] = a[i * n + j] + b[j];
    if (rg) {
        Tensor sa = *this, sb = bias;
        out.grad_fn_ = [sa, sb, m, n](const Tensor& dc) mutable {
            if (sa.requires_grad_) sa.backward(dc);
            if (sb.requires_grad_) {
                Tensor db = make_output({n}, false);
                const float* g = dc.data(); float* dv = db.data();
                for (int j = 0; j < n; ++j) {
                    float s = 0.f; for (int i = 0; i < m; ++i) s += g[i*n+j]; dv[j] = s;
                }
                sb.backward(db);
            }
        };
    }
    return out;
}

// ---------------------------------------------------------------------------
// Linear algebra: matmul and transpose
// ---------------------------------------------------------------------------

Tensor Tensor::transpose() const {
    if (ndim() != 2) throw std::invalid_argument("transpose only supported for 2D tensors");
    int m = shape_[0], n = shape_[1];
    bool rg = requires_grad_;
    Tensor out = make_output({n, m}, rg);
    const float* a = data_.get(); float* b = out.data_.get();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            b[j * m + i] = a[i * n + j];
    if (rg) {
        Tensor sa = *this;
        out.grad_fn_ = [sa](const Tensor& dc) mutable {
            Tensor dt = dc.transpose(); dt.requires_grad_ = false;
            sa.backward(dt);
        };
    }
    return out;
}

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::invalid_argument("matmul: only 2D tensors supported");
    int m = a.shape_[0], k = a.shape_[1], n = b.shape_[1];
    if (k != b.shape_[0]) throw std::invalid_argument("matmul: inner dimensions must match");
    bool rg = a.requires_grad_ || b.requires_grad_;
    Tensor out = make_output({m, n}, rg);
    const float* A = a.data_.get(); const float* B = b.data_.get(); float* C = out.data_.get();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0.f;
            for (int l = 0; l < k; ++l) s += A[i * k + l] * B[l * n + j];
            C[i * n + j] = s;
        }
    if (rg) {
        Tensor sa = a, sb = b;
        out.grad_fn_ = [sa, sb](const Tensor& dc) mutable {
            // dA = dC @ B^T,  dB = A^T @ dC
            if (sa.requires_grad_) {
                Tensor dA = matmul(dc, sb.transpose()); dA.requires_grad_ = false;
                sa.backward(dA);
            }
            if (sb.requires_grad_) {
                Tensor dB = matmul(sa.transpose(), dc); dB.requires_grad_ = false;
                sb.backward(dB);
            }
        };
    }
    return out;
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

// ---------------------------------------------------------------------------
// In-place ops (optimizer use only — no graph nodes created)
// ---------------------------------------------------------------------------

void Tensor::add_inplace(const Tensor& other, float scale) {
    assert(size_ == other.size_);
    float* a = data_.get(); const float* b = other.data_.get();
    for (int i = 0; i < size_; ++i) a[i] += scale * b[i];
}

void Tensor::scale_inplace(float s) {
    float* a = data_.get();
    for (int i = 0; i < size_; ++i) a[i] *= s;
}

void Tensor::copy_from(const Tensor& other) {
    assert(size_ == other.size_);
    std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
}

void Tensor::fill(float v) {
    std::fill(data_.get(), data_.get() + size_, v);
}

} // namespace cppdist
