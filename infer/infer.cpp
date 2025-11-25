#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <chrono>

int main() {
    const std::string onnx_path = "blaze_opencv_compatible.onnx";
    const int H = 256;
    const int W = 256;
    const int NUM_ITERS = 100;

    cv::Mat resized(H, W, CV_32FC3);
    cv::randu(resized, 0.0f, 1.0f);

    // HWC -> NCHW
    cv::Mat blob = cv::dnn::blobFromImage(resized);
    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnx_path);

    net.setInput(blob, "image");
    net.forward();   // warmup

    double total_ms = 0.0;

    for (int i = 0; i < NUM_ITERS; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        net.setInput(blob, "image");
        net.forward();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
    }

    double avg_ms = total_ms / NUM_ITERS;
    std::cout << "Average inference time over " << NUM_ITERS
              << " runs: " << avg_ms << " ms\n";

    std::vector<cv::Mat> outputs;
    net.setInput(blob, "image");
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    cv::Mat vis;
    resized.copyTo(vis);

    cv::Mat boxes = outputs[0].reshape(1, outputs[0].total() / 16);
    cv::Mat scores;

    if (outputs.size() >= 2)
        scores = outputs[1];
    else
        scores = cv::Mat::ones(boxes.rows, 1, CV_32F);

    for (int i = 0; i < boxes.rows; i++) {
        float score = scores.at<float>(i, 0);
        const float* det = boxes.ptr<float>(i);

        int x1 = int(det[1] * W);
        int y1 = int(det[0] * H);
        int x2 = int(det[3] * W);
        int y2 = int(det[2] * H);

        if (x2 - x1 < 5 || y2 - y1 < 5) continue;

        cv::rectangle(vis, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)),
                      cv::Scalar(0, 255, 0), 3);

        cv::putText(vis, cv::format("%.2f", score),
                    cv::Point(x1, std::max(0, y1 - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("result_blazeface.jpg", vis);

    return 0;
}
