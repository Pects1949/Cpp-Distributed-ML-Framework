#include "cppdist/loss.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace cppdist {

// ---------------------------------------------------------------------------
// MSELoss
// ---------------------------------------------------------------------------

Tensor MSELoss::forward(const Tensor& predictions, const Tensor& targets) {
    Tensor diff = predictions - targets;
    Tensor sq   = diff * diff;
    return sq.mean(-1);
}

} // namespace cppdist
