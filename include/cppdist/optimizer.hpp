#pragma once
#include "tensor.hpp"
#include <vector>

namespace cppdist {

class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor*> params, float lr);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    void zero_grad();

    float lr() const { return lr_; }
    void  set_lr(float lr) { lr_ = lr; }

protected:
    std::vector<Tensor*> params_;
    float                lr_;
};

// SGD with momentum and weight decay.
// v = momentum * v + (grad + weight_decay * param)
// param -= lr * v
class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> params, float lr,
        float momentum = 0.9f, float weight_decay = 0.f);

    void step() override;

private:
    float momentum_;
    float weight_decay_;
    std::vector<Tensor> velocity_;
    bool initialized_{false};
    void init();
};

// Adam: adaptive moment estimation.
// m = beta1*m + (1-beta1)*g
// v = beta2*v + (1-beta2)*g^2
// param -= lr * (m/(1-beta1^t)) / (sqrt(v/(1-beta2^t)) + eps)
class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> params, float lr = 1e-3f,
         float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.f);

    void step() override;

private:
    float beta1_, beta2_, eps_, weight_decay_;
    int   step_count_{0};
    std::vector<Tensor> m_, v_;
    bool initialized_{false};
    void init();
};

} // namespace cppdist
