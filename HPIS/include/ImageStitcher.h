#ifndef IMAGE_STITCHER_H
#define IMAGE_STITCHER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <thread> 
#include <unordered_set>
#include "Profiler.hpp"

class roi{
    public:
        cv::Rect region;
        cv::KeyPoint keypoint;
        cv::KeyPoint keypoint_match;
        cv::Mat descriptor;
        bool valid=false;
        roi();
        ~roi();
        
};

class image{
    public:
        cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 5, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
        // 500 → nfeatures
        // 1.2f → scaleFactor
        // 8 → nlevels
        // 5 → edgeThreshold (small for small ROIs)
        // 0 → firstLevel
        // 2 → WTA_K
        // cv::ORB::HARRIS_SCORE → scoreType
        // 31 → patchSize
        // 5 → fastThreshold

        Profiler profiler;

        std::vector<std::pair<int,int>> roi_matches;
        cv::Mat H;
        int id=-1;
        int width=2268;
        int height=4032;
        int roi_width=100;
        int roi_height=100;
        int roi_row_count;
        int roi_col_count;
        std::vector<std::vector<roi>> ROIs;
        cv::Mat img;
        image();
        ~image();
        void set(const std::string& image_name);
        void reset();

};

class ImageStitcher {


public:
    int n_images=0;
    int nmax=0;
    image img1;
    image img2;
    cv::Mat Panorama;  // global canvas
    cv::Mat GHomography = cv::Mat::eye(3, 3, CV_64F); // global accumulated H

    Profiler profiler;

    // Constructor
    ImageStitcher();

    // Destructor
    ~ImageStitcher();

    // Add image to stitcher
    void addImage(const std::string& image_name);

};

// static void detectorb(const cv::Mat& img, roi& r, cv::Ptr<cv::ORB> orb);
// static void matchorb(image& img1, image& img2);

#endif // IMAGE_STITCHER_H
