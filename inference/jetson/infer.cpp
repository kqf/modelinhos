#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#define CHECK(call)                                                  \
    do {                                                             \
        cudaError_t err = (call);                                    \
        if (err != cudaSuccess) {                                    \
            std::cerr << #call << ": " << cudaGetErrorString(err)    \
                      << "\n";                                       \
            std::exit(1);                                            \
        }                                                            \
    } while (0)

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << msg << "\n";
    }
};

struct Destroy {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) obj->destroy();
    }
};
template <typename T>
using Ptr = std::unique_ptr<T, Destroy>;

static size_t element_size(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kBOOL: return 1;
    }
    return 4;
}

static size_t volume(const nvinfer1::Dims& dims) {
    size_t n = 1;
    for (int i = 0; i < dims.nbDims; i++) n *= dims.d[i];
    return n;
}

static std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(1);
    }
    std::vector<char> data(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    file.read(data.data(), data.size());
    return data;
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "usage: " << argv[0]
                  << " model.engine height width [fp32|fp16|int8]\n";
        return 1;
    }
    const std::string path = argv[1];
    const int h = std::atoi(argv[2]);
    const int w = std::atoi(argv[3]);
    const std::string mode = argc == 5 ? argv[4] : "fp32";
    const int warmup = 10;
    const int iters = 100;

    CHECK(cudaFree(nullptr));
    size_t free0 = 0, total = 0;
    CHECK(cudaMemGetInfo(&free0, &total));

    Logger logger;
    std::vector<char> blob = read_file(path);
    Ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    Ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine) {
        std::cerr << "cannot deserialize " << path << "\n";
        return 1;
    }
    Ptr<nvinfer1::IExecutionContext> context(
        engine->createExecutionContext());

    const int n = engine->getNbBindings();
    std::vector<void*> bindings(n, nullptr);
    std::vector<size_t> bytes(n, 0);
    std::vector<std::vector<char>> host(n);
    for (int i = 0; i < n; i++) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        if (engine->bindingIsInput(i)) {
            // Exports are static, but tolerate a dynamic batch/size.
            bool dynamic = false;
            for (int d = 0; d < dims.nbDims; d++) dynamic |= dims.d[d] < 0;
            if (dynamic) {
                dims = nvinfer1::Dims4{1, 3, h, w};
                context->setBindingDimensions(i, dims);
            }
            if (dims.nbDims == 4 && (dims.d[2] != h || dims.d[3] != w)) {
                std::cerr << path << ": engine input is " << dims.d[2]
                          << "x" << dims.d[3] << ", expected " << h << "x"
                          << w << "\n";
                return 1;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        bytes[i] = volume(dims) * element_size(engine->getBindingDataType(i));
        host[i].resize(bytes[i]);
        CHECK(cudaMalloc(&bindings[i], bytes[i]));
        if (engine->bindingIsInput(i)) {
            std::mt19937 rng(0);
            std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
            if (engine->getBindingDataType(i) == nvinfer1::DataType::kFLOAT) {
                float* data = reinterpret_cast<float*>(host[i].data());
                for (size_t k = 0; k < bytes[i] / 4; k++) data[k] = uniform(rng);
            }
        }
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // One frame = upload input, run the full graph, download every
    // output, then wait. Copies are included so the numbers compare
    // with the CPU runner, where setInput/forward cover the same work.
    auto run = [&]() {
        for (int i = 0; i < n; i++)
            if (engine->bindingIsInput(i))
                CHECK(cudaMemcpyAsync(bindings[i], host[i].data(), bytes[i],
                                      cudaMemcpyHostToDevice, stream));
        if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
            std::cerr << path << ": enqueue failed\n";
            std::exit(1);
        }
        for (int i = 0; i < n; i++)
            if (!engine->bindingIsInput(i))
                CHECK(cudaMemcpyAsync(host[i].data(), bindings[i], bytes[i],
                                      cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
    };

    // Warmup: the first runs include lazy allocations and clock ramp-up;
    // keep them out of the measurement.
    for (int i = 0; i < warmup; i++) run();

    std::vector<double> ms(iters);
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        run();
        auto t1 = std::chrono::steady_clock::now();
        ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double mean = 0.0;
    for (double m : ms) mean += m;
    mean /= iters;
    double var = 0.0;
    for (double m : ms) var += (m - mean) * (m - mean);
    double std = std::sqrt(var / (iters - 1));

    // The Nano's GPU shares system RAM, but cudaMalloc'd memory (weights,
    // activations, workspace) does not show up in the process RSS. Report
    // peak RSS (kilobytes on Linux) plus the device memory taken since the
    // CUDA context came up. cudaMemGetInfo is system-wide, so other GPU
    // users on the board would add noise.
    size_t free1 = 0;
    CHECK(cudaMemGetInfo(&free1, &total));
    rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double device = free0 > free1 ? (free0 - free1) / 1024.0 / 1024.0 : 0.0;
    double ram = usage.ru_maxrss / 1024.0 + device;

    CHECK(cudaStreamDestroy(stream));
    for (void* b : bindings) CHECK(cudaFree(b));

    // Same fixed-width columns as the OpenCV runner, so the tables line up.
    std::cout << std::fixed << std::setprecision(2)
              << std::left << std::setw(48) << path
              << "  " << std::setw(4) << mode
              << std::right
              << "  w=" << std::setw(5) << w
              << "  h=" << std::setw(5) << h
              << "  mean=" << std::setw(8) << mean << " ms"
              << "  std=" << std::setw(7) << std << " ms"
              << "  fps=" << std::setw(8) << 1000.0 / mean
              << "  ram=" << std::setw(8) << ram << " MB" << std::endl;
    return 0;
}
