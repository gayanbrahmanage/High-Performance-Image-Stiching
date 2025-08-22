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

    ImageStitcher stitcher;
    stitcher.nmax=files.size();
    Profiler profiler;
    profiler.start();
    for (const auto& f : files){
        stitcher.addImage(f);
        std::cout << "Added " <<f<< std::endl;
    }

    profiler.end("Image Stitching Resource Usage");

    cv::imwrite("../../outputs/panorama.jpg", stitcher.Panorama);
    
    return 0;
}

