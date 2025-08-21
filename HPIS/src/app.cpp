#include "ImageLoader.hpp"
#include "ImageStitcher.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Profiler.hpp"
#include "utils.hpp"

int main() {
    std::string folder = "../../inputs/A";
    auto files = listJPGFiles(folder);
    for (const auto& f : files) {
        std::cout << f << std::endl;
    }

    ImageStitcher stitcher;
    Profiler profiler;
    profiler.start(); // Start measuring
    stitcher.addImage(files[0]);
    stitcher.addImage(files[1]);
    profiler.end("Image Stitching Resource Usage"); // End measuring

    cv::Mat vis = Utils::draw(stitcher.img1);
    cv::imwrite("../../outputs/stitched_image.jpg", vis);
    cv::Mat vis2 = Utils::draw(stitcher.img2);
    cv::imwrite("../../outputs/stitched_image2.jpg", vis2);
    cv::Mat visMatches = Utils::drawMatchesROIs(stitcher.img1, stitcher.img2);
    cv::imwrite("../../outputs/matched_rois.jpg", visMatches);
    return 0;
}
