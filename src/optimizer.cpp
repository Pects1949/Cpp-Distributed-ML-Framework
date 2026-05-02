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


// ---------------------------------------------------------------------------
// Adam with bias correction
// ---------------------------------------------------------------------------

Adam::Adam(std::vector<Tensor*> params, float lr, float beta1, float beta2,
           float eps, float weight_decay)
    : Optimizer(std::move(params), lr),
      beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {}

void Adam::init() {
    m_.clear(); v_.clear();
    for (auto* p : params_) {
        m_.emplace_back(Tensor::zeros(p->shape()));
        v_.emplace_back(Tensor::zeros(p->shape()));
    }
    initialized_ = true;
}

void Adam::step() {
    if (!initialized_) init();
    ++step_count_;
    float bc1 = 1.f - std::pow(beta1_, static_cast<float>(step_count_));
    float bc2 = 1.f - std::pow(beta2_, static_cast<float>(step_count_));
    for (int i = 0; i < static_cast<int>(params_.size()); ++i) {
        Tensor* p   = params_[i];
        float*  p_d = p->data();
        float*  m_d = m_[i].data();
        float*  v_d = v_[i].data();
        const float* g_d = p->grad_data();
        if (!g_d) continue;
        int n = p->numel();
        for (int j = 0; j < n; ++j) {
            float g  = g_d[j] + weight_decay_ * p_d[j];
            m_d[j]   = beta1_ * m_d[j] + (1.f - beta1_) * g;
            v_d[j]   = beta2_ * v_d[j] + (1.f - beta2_) * g * g;
            float mh = m_d[j] / bc1;
            float vh = v_d[j] / bc2;
            p_d[j]  -= lr_ * mh / (std::sqrt(vh) + eps_);
        }
    }
}

} // namespace cppdist
