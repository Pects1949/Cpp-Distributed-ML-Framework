#pragma once
#include "tensor.hpp"
#include <string>

namespace cppdist {

class Loss {
public:
    virtual ~Loss() = default;
    virtual Tensor forward(const Tensor& predictions, const Tensor& targets) = 0;
    virtual std::string name() const = 0;
};

// mean((preds - targets)^2) — autograd flows through Tensor ops
class MSELoss : public Loss {
public:
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;
    std::string name() const override { return "MSELoss"; }
};

} // namespace cppdist
