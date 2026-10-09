#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "usage: " << argv[0]
                  << " model.onnx height width [fp32|fp16|int8]\n";
        return 1;
    }
    const std::string path = argv[1];
    const int h = std::atoi(argv[2]);
    const int w = std::atoi(argv[3]);
    const std::string mode = argc == 5 ? argv[4] : "fp32";
    const int warmup = 10;
    const int iters = 100;

    cv::Mat image(h, w, CV_32FC3);
    cv::randu(image, 0.0f, 1.0f);
    cv::Mat blob = cv::dnn::blobFromImage(image);  // HWC -> NCHW

    cv::dnn::Net net = cv::dnn::readNetFromONNX(path);
    // int8: quantize the whole net (per-channel, the random blob as
    // calibration -- fine for timing, redo with real frames before
    // trusting accuracy), float kept at the input/output boundary.
    // fp16: halves memory bandwidth, needs ARMv8.2 fp16 SIMD (RPi 5;
    // falls back to fp32 math elsewhere). Fusion and Winograd are on by
    // default; stated here so turning them off for A/B runs is a
    // one-line change.
    if (mode == "int8") net = net.quantize({blob}, CV_32F, CV_32F);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(mode == "fp16" ? cv::dnn::DNN_TARGET_CPU_FP16
                                           : cv::dnn::DNN_TARGET_CPU);
    net.enableFusion(true);
    net.enableWinograd(true);
    std::vector<cv::String> names = net.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outputs;

    // Warmup: the first forwards include graph compilation and buffer
    // allocation; keep them out of the measurement. Every forward runs
    // the full graph (all unconnected outputs), never a truncated one.
    for (int i = 0; i < warmup; i++) {
        net.setInput(blob);
        net.forward(outputs, names);
    }

    std::vector<double> ms(iters);
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        net.setInput(blob);
        net.forward(outputs, names);
        auto t1 = std::chrono::steady_clock::now();
        ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double mean = 0.0;
    for (double m : ms) mean += m;
    mean /= iters;
    double var = 0.0;
    for (double m : ms) var += (m - mean) * (m - mean);
    double std = std::sqrt(var / (iters - 1));

    // Peak RSS of the whole process: ru_maxrss is bytes on macOS,
    // kilobytes on Linux.
    rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    double ram = usage.ru_maxrss / 1024.0 / 1024.0;
#else
    double ram = usage.ru_maxrss / 1024.0;
#endif

    std::cout << std::fixed << std::setprecision(2) << path << "  " << mode
              << "  w=" << w << " h=" << h << "  mean=" << mean
              << " ms  std=" << std << " ms  fps=" << 1000.0 / mean
              << "  ram=" << ram << " MB" << std::endl;
    return 0;
}
