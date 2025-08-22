#include "ImageLoader.hpp"
#include "ImageStitcher.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Profiler.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_folder>" << std::endl;
        return -1;
    }

    std::string folder = argv[1];
    auto files = listJPGFiles(folder);
    if (files.size() < 2) {
        std::cerr << "Need at least 2 images in " << folder << std::endl;
        return -1;
    }

    for (const auto& f : files) std::cout << f << std::endl;

    ImageStitcher stitcher;
    stitcher.nmax=files.size();
    Profiler profiler;
    profiler.start();
    stitcher.addImage(files[0]);
    stitcher.addImage(files[1]);
    profiler.end("Image Stitching Resource Usage");

    cv::imwrite("../../outputs/stitched_image.jpg", Utils::draw(stitcher.img1));
    cv::imwrite("../../outputs/stitched_image2.jpg", Utils::draw(stitcher.img2));
    cv::imwrite("../../outputs/matched_rois.jpg", Utils::drawMatchesROIs(stitcher.img1, stitcher.img2));

    return 0;
}

