#pragma once
#include "layer.hpp"
#include <memory>
#include <vector>

namespace cppdist {

class Sequential : public Layer {
public:
    Sequential() = default;

    void add(std::shared_ptr<Layer> layer);

    template<typename LayerType, typename... Args>
    void add(Args&&... args) {
        layers_.push_back(std::make_shared<LayerType>(std::forward<Args>(args)...));
    }

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "Sequential"; }

    void zero_grad() override;
    void train(bool mode = true) override;

    void print_summary() const;
    int total_parameters() const;

private:
    std::vector<std::shared_ptr<Layer>> layers_;
};

} // namespace cppdist
