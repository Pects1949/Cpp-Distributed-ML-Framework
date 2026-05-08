#pragma once
#include "tensor.hpp"
#include <vector>
#include <memory>

#ifdef CPPDIST_USE_MPI
#include <mpi.h>
#endif

namespace cppdist {

class DistributedBackend {
public:
    virtual ~DistributedBackend() = default;

    virtual void init(int* argc, char*** argv) = 0;
    virtual void finalize() = 0;
    virtual int  rank()       const = 0;
    virtual int  world_size() const = 0;
    bool is_root() const { return rank() == 0; }

    // Average gradients across all ranks via AllReduce
    virtual void allreduce_gradients(std::vector<Tensor*>& params) = 0;

    // Broadcast tensor data from root to all ranks
    virtual void broadcast(Tensor& tensor, int root = 0) = 0;

    virtual void barrier() = 0;
};

// Single-process fallback — all operations are no-ops
class NoOpBackend : public DistributedBackend {
public:
    void init(int* argc, char*** argv) override {}
    void finalize() override {}
    int  rank()       const override { return 0; }
    int  world_size() const override { return 1; }
    void allreduce_gradients(std::vector<Tensor*>&) override {}
    void broadcast(Tensor&, int) override {}
    void barrier() override {}
};

#ifdef CPPDIST_USE_MPI
class MPIBackend : public DistributedBackend {
public:
    void init(int* argc, char*** argv) override;
    void finalize() override;
    int  rank()       const override { return rank_; }
    int  world_size() const override { return world_size_; }
    void allreduce_gradients(std::vector<Tensor*>& params) override;
    void broadcast(Tensor& tensor, int root = 0) override;
    void barrier() override;

private:
    int rank_{0};
    int world_size_{1};
};
#endif

// Returns MPIBackend if compiled with CPPDIST_USE_MPI, otherwise NoOpBackend
std::unique_ptr<DistributedBackend> make_backend(int* argc, char*** argv);

} // namespace cppdist
