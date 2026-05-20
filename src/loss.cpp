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


// ---------------------------------------------------------------------------
// CrossEntropyLoss
// ---------------------------------------------------------------------------

// Uses log-sum-exp for numerical stability. The backward is implemented
// analytically rather than through autograd ops for the same reason: computing
// softmax(logits) - one_hot(y) avoids catastrophic cancellation in log(softmax).
Tensor CrossEntropyLoss::forward(const Tensor& logits, const Tensor& targets) {
    if (logits.ndim() != 2)
        throw std::invalid_argument("CrossEntropyLoss: logits must be 2D [batch, classes]");
    if (targets.ndim() != 1 || targets.size(0) != logits.size(0))
        throw std::invalid_argument("CrossEntropyLoss: targets must be 1D [batch]");

    int batch = logits.size(0), nc = logits.size(1);
    const float* ld = logits.data();
    const float* td = targets.data();

    float total = 0.f;
    for (int i = 0; i < batch; ++i) {
        const float* row = ld + i * nc;
        float maxv = *std::max_element(row, row + nc);
        float sumexp = 0.f;
        for (int j = 0; j < nc; ++j) sumexp += std::exp(row[j] - maxv);
        total += maxv + std::log(sumexp) - row[static_cast<int>(td[i])];
    }

    bool rg = logits.requires_grad();
    Tensor out({1}, rg);
    out.data()[0] = total / static_cast<float>(batch);

    if (rg) {
        Tensor lc = logits, tc = targets;
        int b = batch, c = nc;
        out.set_grad_fn([lc, tc, b, c](const Tensor& dc) mutable {
            float upstream = dc.data()[0];
            Tensor dlogits({b, c}, false);
            const float* ld2 = lc.data(); const float* td2 = tc.data(); float* d = dlogits.data();
            for (int i = 0; i < b; ++i) {
                const float* row = ld2 + i * c;
                float maxv = *std::max_element(row, row + c);
                float sumexp = 0.f;
                for (int j = 0; j < c; ++j) sumexp += std::exp(row[j] - maxv);
                int cls = static_cast<int>(td2[i]);
                for (int j = 0; j < c; ++j) {
                    float sm = std::exp(row[j] - maxv) / sumexp;
                    d[i*c+j] = (sm - (j == cls ? 1.f : 0.f)) / static_cast<float>(b) * upstream;
                }
            }
            lc.backward(dlogits);
        });
    }
    return out;
}

} // namespace cppdist
