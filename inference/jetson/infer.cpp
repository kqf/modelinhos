#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "usage: " << argv[0]
                  << " model.onnx height width [fp32|fp16|int8]\n";
        return 1;
    }
    const std::string path = argv[1];
    const std::string mode = argc == 5 ? argv[4] : "fp32";
    const int h = 0;
    const int w = 0;
    const double mean = 0.0;
    const double std = 0.0;
    const double fps = 0.0;
    const double ram = 0.0;

    std::cout << std::fixed << std::setprecision(2)
              << std::left << std::setw(48) << path
              << "  " << std::setw(4) << mode
              << std::right
              << "  w=" << std::setw(5) << w
              << "  h=" << std::setw(5) << h
              << "  mean=" << std::setw(8) << mean << " ms"
              << "  std=" << std::setw(7) << std << " ms"
              << "  fps=" << std::setw(8) << fps
              << "  ram=" << std::setw(8) << ram << " MB" << std::endl;
    return 0;
}
