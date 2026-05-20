#include "cppdist/distributed.hpp"
#include <cstdio>

namespace cppdist {

#ifdef CPPDIST_USE_MPI

void MPIBackend::init(int* argc, char*** argv) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
}

void MPIBackend::finalize() { MPI_Finalize(); }

void MPIBackend::allreduce_gradients(std::vector<Tensor*>& params) {
    if (world_size_ == 1) return;
    float inv = 1.f / static_cast<float>(world_size_);
    for (Tensor* p : params) {
        if (!p->requires_grad() || !p->grad_data()) continue;
        float* g   = p->grad_data();
        int    cnt = p->numel();
        // MPI_IN_PLACE avoids a temporary buffer: each rank sums in-place
        MPI_Allreduce(MPI_IN_PLACE, g, cnt, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        for (int i = 0; i < cnt; ++i) g[i] *= inv;
    }
}

void MPIBackend::broadcast(Tensor& tensor, int root) {
    MPI_Bcast(tensor.data(), tensor.numel(), MPI_FLOAT, root, MPI_COMM_WORLD);
}

void MPIBackend::barrier() { MPI_Barrier(MPI_COMM_WORLD); }

#endif  // CPPDIST_USE_MPI

std::unique_ptr<DistributedBackend> make_backend(int* argc, char*** argv) {
#ifdef CPPDIST_USE_MPI
    auto b = std::make_unique<MPIBackend>();
    b->init(argc, argv);
    std::printf("[cppdist] MPI backend: rank %d / %d\n", b->rank(), b->world_size());
    return b;
#else
    auto b = std::make_unique<NoOpBackend>();
    b->init(argc, argv);
    std::printf("[cppdist] No-op backend: single-process mode\n");
    return b;
#endif
}

} // namespace cppdist
