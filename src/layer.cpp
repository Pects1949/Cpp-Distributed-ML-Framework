#include "cppdist/layer.hpp"
#include <cmath>

namespace cppdist {

int Layer::num_parameters() const {
    int total = 0;
    for (auto* p : const_cast<Layer*>(this)->parameters())
        total += p->numel();
    return total;
}

void Layer::zero_grad() {
    for (auto* p : parameters()) p->zero_grad();
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

Linear::Linear(int in_features, int out_features, bool use_bias)
    : in_features_(in_features), use_bias_(use_bias) {
    weight_ = Tensor({out_features, in_features}, true);
    if (use_bias_) bias_ = Tensor({out_features}, true);
    init_weights();
}

void Linear::init_weights() {
    // He uniform: limit = sqrt(6 / in_features)
    float limit = std::sqrt(6.f / static_cast<float>(in_features_));
    float* w = weight_.data();
    int n = weight_.numel();
    // Simple LCG for reproducible init without a seed parameter
    uint32_t state = 12345;
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        float u = static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu);
        w[i] = u * 2.f * limit - limit;
    }
    if (use_bias_) bias_.fill(0.f);
}

Tensor Linear::forward(const Tensor& input) {
    // input: [batch, in_features]
    // out:   [batch, out_features]
    Tensor out = Tensor::matmul(input, weight_.transpose());
    if (use_bias_) out = out.add_bias(bias_);
    return out;
}

std::vector<Tensor*> Linear::parameters() {
    if (use_bias_) return {&weight_, &bias_};
    return {&weight_};
}

} // namespace cppdist
