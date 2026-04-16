#pragma once
#include "tensor.hpp"
#include <vector>
#include <string>
#include <memory>

namespace cppdist {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual std::vector<Tensor*> parameters() = 0;
    virtual std::string name() const = 0;
    virtual int num_parameters() const;

    virtual void zero_grad();
    virtual void train(bool mode = true) { training_ = mode; }
    bool is_training() const { return training_; }

protected:
    bool training_{true};
};

// Fully-connected layer: out = input @ weight.T + bias
// weight shape: [out_features, in_features]
// bias shape:   [out_features]
class Linear : public Layer {
public:
    Linear(int in_features, int out_features, bool use_bias = true);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "Linear"; }

    const Tensor& weight() const { return weight_; }
    const Tensor& bias()   const { return bias_;   }

private:
    int    in_features_;
    bool   use_bias_;
    Tensor weight_;
    Tensor bias_;

    void init_weights();
};

class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override { return input.relu(); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::string name() const override { return "ReLU"; }
};

class Sigmoid : public Layer {
public:
    Tensor forward(const Tensor& input) override { return input.sigmoid(); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::string name() const override { return "Sigmoid"; }
};

class TanhActivation : public Layer {
public:
    Tensor forward(const Tensor& input) override { return input.tanh_act(); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::string name() const override { return "Tanh"; }
};

} // namespace cppdist
