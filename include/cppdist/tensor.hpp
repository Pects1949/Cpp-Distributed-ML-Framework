#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <random>

namespace cppdist {

// Dense float32 tensor with eager autograd via closure-based computation graph.
// Copies are shallow (data shared via shared_ptr); use clone() for deep copies.
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<int> shape, bool requires_grad = false);
    Tensor(std::vector<int> shape, float fill_value, bool requires_grad = false);

    static Tensor zeros(std::vector<int> shape, bool requires_grad = false);
    static Tensor ones(std::vector<int> shape, bool requires_grad = false);
    static Tensor randn(std::vector<int> shape, float mean = 0.f, float stddev = 1.f,
                        unsigned int seed = 0, bool requires_grad = false);
    static Tensor uniform(std::vector<int> shape, float low, float high,
                          unsigned int seed = 0, bool requires_grad = false);
    static Tensor from_data(const std::vector<float>& data, std::vector<int> shape,
                            bool requires_grad = false);

    // Shape
    const std::vector<int>& shape() const { return shape_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    int numel() const { return size_; }
    int size(int dim) const;

    // Data access
    float*       data()       { return data_.get(); }
    const float* data() const { return data_.get(); }
    float&       operator[](int i)       { return data_.get()[i]; }
    float        operator[](int i) const { return data_.get()[i]; }

    // Gradient
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool v);
    float*       grad_data()       { return grad_data_.get(); }
    const float* grad_data() const { return grad_data_.get(); }
    Tensor grad() const;   // view of grad buffer as a detached Tensor
    void zero_grad();      // zero gradient buffer in-place

    // Autograd entry points
    void backward();                            // upstream = all-ones (scalar loss)
    void backward(const Tensor& upstream_grad); // explicit upstream gradient

    // Arithmetic — autograd-tracked, return new Tensors
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // elementwise
    Tensor operator*(float s) const;
    Tensor operator/(float s) const;
    friend Tensor operator*(float s, const Tensor& t);

    // Shape operations
    Tensor reshape(std::vector<int> new_shape) const;
    Tensor transpose() const;  // 2D only: [m,n] -> [n,m]
    Tensor sum(int dim, bool keepdim = false) const;  // dim=-1: sum all elements
    Tensor mean(int dim = -1, bool keepdim = false) const;

    // Activations (autograd-tracked)
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh_act() const;
    Tensor exp_t() const;
    Tensor log_t() const;
    Tensor softmax(int dim = 1) const;

    // Bias add: self [batch,n] + bias [n], broadcasts over dim 0
    Tensor add_bias(const Tensor& bias) const;

    // Matrix multiply: A[m,k] @ B[k,n] -> [m,n]
    static Tensor matmul(const Tensor& a, const Tensor& b);

    Tensor clone()  const;  // deep copy, same requires_grad
    Tensor detach() const;  // deep copy, no grad tracking

    // In-place ops for optimizer (do NOT create graph nodes)
    void add_inplace(const Tensor& other, float scale = 1.f);
    void scale_inplace(float s);
    void copy_from(const Tensor& other);
    void fill(float v);

    bool empty() const { return size_ == 0; }

    // Allow external grad_fn injection (used by loss implementations)
    void set_grad_fn(std::function<void(const Tensor&)> fn) {
        grad_fn_ = std::move(fn);
    }

private:
    std::shared_ptr<float[]> data_;
    std::shared_ptr<float[]> grad_data_;
    std::vector<int>         shape_;
    int                      size_{0};
    bool                     requires_grad_{false};

    // Backward closure — captures input Tensors by value (shallow, shares buffers)
    std::function<void(const Tensor&)> grad_fn_;

    void alloc(int n);
    void ensure_grad();
    void accumulate_grad(const Tensor& delta);

    static int compute_size(const std::vector<int>& s);
    static Tensor make_output(std::vector<int> shape, bool rg);
};

} // namespace cppdist
