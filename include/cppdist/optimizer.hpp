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

} // namespace cppdist
