#include "cppdist/optimizer.hpp"
#include <cmath>
#include <cassert>

namespace cppdist {

// ---------------------------------------------------------------------------
// Optimizer base
// ---------------------------------------------------------------------------

Optimizer::Optimizer(std::vector<Tensor*> params, float lr)
    : params_(std::move(params)), lr_(lr) {}

void Optimizer::zero_grad() {
    for (auto* p : params_) p->zero_grad();
}

// ---------------------------------------------------------------------------
// SGD with momentum
// ---------------------------------------------------------------------------

SGD::SGD(std::vector<Tensor*> params, float lr, float momentum, float weight_decay)
    : Optimizer(std::move(params), lr), momentum_(momentum), weight_decay_(weight_decay) {}

void SGD::init() {
    velocity_.clear();
    for (auto* p : params_) velocity_.emplace_back(Tensor::zeros(p->shape()));
    initialized_ = true;
}

void SGD::step() {
    if (!initialized_) init();
    for (int i = 0; i < static_cast<int>(params_.size()); ++i) {
        Tensor* p   = params_[i];
        float*  p_d = p->data();
        float*  v_d = velocity_[i].data();
        const float* g_d = p->grad_data();
        if (!g_d) continue;
        int n = p->numel();
        for (int j = 0; j < n; ++j) {
            float g = g_d[j] + weight_decay_ * p_d[j];
            v_d[j]  = momentum_ * v_d[j] + g;
            p_d[j] -= lr_ * v_d[j];
        }
    }
}

} // namespace cppdist
