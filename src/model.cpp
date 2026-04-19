#include "cppdist/model.hpp"
#include <cstdio>
#include <numeric>

namespace cppdist {

void Sequential::add(std::shared_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& layer : layers_) x = layer->forward(x);
    return x;
}

std::vector<Tensor*> Sequential::parameters() {
    std::vector<Tensor*> all;
    for (auto& layer : layers_) {
        auto ps = layer->parameters();
        all.insert(all.end(), ps.begin(), ps.end());
    }
    return all;
}

void Sequential::zero_grad() {
    for (auto& layer : layers_) layer->zero_grad();
}

void Sequential::train(bool mode) {
    training_ = mode;
    for (auto& layer : layers_) layer->train(mode);
}

void Sequential::print_summary() const {
    std::printf("Sequential(\n");
    int total = 0;
    for (auto& layer : layers_) {
        int np = layer->num_parameters();
        std::printf("  %-20s  params: %d\n", layer->name().c_str(), np);
        total += np;
    }
    std::printf(") total params: %d\n", total);
}

int Sequential::total_parameters() const {
    int total = 0;
    for (auto& layer : layers_) total += layer->num_parameters();
    return total;
}

} // namespace cppdist
