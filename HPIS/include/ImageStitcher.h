#ifndef IMAGE_STITCHER_H
#define IMAGE_STITCHER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <thread> 
#include <unordered_set>


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
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        std::vector<std::pair<int,int>> roi_matches;
        cv::Mat H;
        int id=-1;
        int width;
        int height;
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
    image img1;
    image img2;

    // Constructor
    ImageStitcher();

    // Destructor
    ~ImageStitcher();

    // Add image to stitcher
    void addImage(const std::string& image_name);

    // Perform stitching and return panorama
    cv::Mat stitch();

};

// static void detectorb(const cv::Mat& img, roi& r, cv::Ptr<cv::ORB> orb);
// static void matchorb(image& img1, image& img2);

#endif // IMAGE_STITCHER_H
