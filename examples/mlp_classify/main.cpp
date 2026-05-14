#include "cppdist/tensor.hpp"
#include "cppdist/layer.hpp"
#include "cppdist/model.hpp"
#include "cppdist/loss.hpp"
#include "cppdist/optimizer.hpp"
#include "cppdist/distributed.hpp"
#include "cppdist/dataloader.hpp"

#include <cstdio>
#include <algorithm>

// Quick autograd sanity check: d/dx(x^2) at x=3 should equal 6
static void autograd_sanity_check() {
    cppdist::Tensor x = cppdist::Tensor::from_data({3.f}, {1}, /*requires_grad=*/true);
    cppdist::Tensor y = x * x;
    y.backward();
    float grad = x.grad_data()[0];
    std::printf("[sanity] d/dx(x^2) at x=3: got %.4f, expected 6.0000 -- %s\n",
                grad, std::abs(grad - 6.f) < 1e-5f ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
    // --- Distributed backend (MPI if compiled with ENABLE_MPI, else no-op) ---
    auto dist = cppdist::make_backend(&argc, &argv);
    int rank       = dist->rank();
    int world_size = dist->world_size();

    if (dist->is_root()) autograd_sanity_check();

    // --- Synthetic dataset: 1000 samples, 20 features, 2 classes ---
    cppdist::Tensor full_X, full_y;
    if (dist->is_root()) {
        auto [X, y] = cppdist::make_classification_dataset(
            /*n_samples=*/1000, /*n_features=*/20, /*n_classes=*/2,
            /*noise=*/0.3f, /*seed=*/42);
        full_X = X;
        full_y = y;
    } else {
        // Allocate empty tensors so broadcast can fill them
        full_X = cppdist::Tensor({1000, 20}, false);
        full_y = cppdist::Tensor({1000},    false);
    }
    dist->broadcast(full_X, 0);
    dist->broadcast(full_y, 0);

    // --- DataLoader shards the dataset by rank ---
    cppdist::DataLoader loader(full_X, full_y,
                               /*batch_size=*/32,
                               rank, world_size,
                               /*shuffle=*/true,
                               /*seed=*/42 + static_cast<unsigned>(rank));

    // --- Model: Linear(20->64)->ReLU->Linear(64->32)->ReLU->Linear(32->2) ---
    cppdist::Sequential model;
    model.add<cppdist::Linear>(20, 64);
    model.add<cppdist::ReLU>();
    model.add<cppdist::Linear>(64, 32);
    model.add<cppdist::ReLU>();
    model.add<cppdist::Linear>(32, 2);

    if (dist->is_root()) model.print_summary();

    // Sync initial weights so all ranks start identically
    {
        auto params = model.parameters();
        for (cppdist::Tensor* p : params) dist->broadcast(*p, 0);
    }

    // --- Loss and optimizer ---
    cppdist::CrossEntropyLoss criterion;
    cppdist::Adam optimizer(model.parameters(), /*lr=*/1e-3f);

    // --- Training loop ---
    const int num_epochs = 20;
    model.train(true);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        loader.reset_epoch();
        float epoch_loss = 0.f;
        int   n_batches  = 0;

        cppdist::Tensor batch_X, batch_y;
        while (loader.next_batch(batch_X, batch_y)) {
            cppdist::Tensor logits = model.forward(batch_X);
            cppdist::Tensor loss   = criterion.forward(logits, batch_y);

            optimizer.zero_grad();
            loss.backward();

            // Average gradients across all MPI ranks
            {
                auto params = model.parameters();
                dist->allreduce_gradients(params);
            }
            optimizer.step();

            epoch_loss += loss.data()[0];
            ++n_batches;
        }

        if (dist->is_root()) {
            std::printf("[Epoch %2d/%d] loss: %.4f\n",
                        epoch + 1, num_epochs, epoch_loss / n_batches);
        }
        dist->barrier();
    }

    // --- Final accuracy on full dataset (rank 0 only) ---
    if (dist->is_root()) {
        model.train(false);
        cppdist::Tensor logits = model.forward(full_X);
        int n       = full_X.size(0);
        int nc      = 2;
        int correct = 0;
        const float* ld = logits.data();
        const float* td = full_y.data();
        for (int i = 0; i < n; ++i) {
            int pred  = (ld[i * nc] > ld[i * nc + 1]) ? 0 : 1;
            int truth = static_cast<int>(td[i]);
            if (pred == truth) ++correct;
        }
        std::printf("\nFinal accuracy (full dataset): %.2f%%\n",
                    100.f * correct / static_cast<float>(n));
    }

    dist->finalize();
    return 0;
}
