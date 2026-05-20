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

// Cross-entropy with log-sum-exp trick.
// logits: [batch, num_classes] (raw, before softmax)
// targets: [batch] integer class indices
class CrossEntropyLoss : public Loss {
public:
    Tensor forward(const Tensor& logits, const Tensor& targets) override;
    std::string name() const override { return "CrossEntropyLoss"; }
};

} // namespace cppdist
