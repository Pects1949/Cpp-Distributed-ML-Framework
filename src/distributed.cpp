#include "cppdist/distributed.hpp"
#include <cstdio>

namespace cppdist {

std::unique_ptr<DistributedBackend> make_backend(int* argc, char*** argv) {
    auto b = std::make_unique<NoOpBackend>();
    b->init(argc, argv);
    std::printf("[cppdist] No-op backend: single-process mode\n");
    return b;
}

} // namespace cppdist
