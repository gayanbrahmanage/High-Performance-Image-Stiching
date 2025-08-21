#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include "ImageStitcher.h" 

namespace Utils {

    // Draw all ROIs and their keypoints on a copy of the image
    inline cv::Mat draw(const image& imgObj) {
        cv::Mat output;
        imgObj.img.copyTo(output);

        // Loop through all ROIs
        for (int r = 0; r < imgObj.roi_row_count; ++r) {
            for (int c = 0; c < imgObj.roi_col_count; ++c) {
                const roi& rObj = imgObj.ROIs[r][c];

                // Draw rectangle
                cv::rectangle(output, rObj.region, cv::Scalar(0, 255, 0), 1);

                // Draw keypoint if it exists (non-empty)
                if(!rObj.valid)
                    continue;

                if (rObj.keypoint.pt.x >= 0 && rObj.keypoint.pt.y >= 0) {

                    //std::cout<<rObj.keypoint.pt.x<<", "<<rObj.keypoint.pt.y<<std::endl;
                    cv::circle(output, rObj.keypoint.pt, 20, cv::Scalar(0,0,255), -1);
                }
            }
        }

        return output;
    }

    inline cv::Mat drawMatchesROIs(const image& img1, const image& img2)
    {
        // 1. Create a combined image (side by side)
        int rows = std::max(img1.img.rows, img2.img.rows);
        int cols = img1.img.cols + img2.img.cols;
        cv::Mat output(rows, cols, CV_8UC3, cv::Scalar::all(0));

        // Copy img1 and img2 into output
        img1.img.copyTo(output(cv::Rect(0, 0, img1.img.cols, img1.img.rows)));
        img2.img.copyTo(output(cv::Rect(img1.img.cols, 0, img2.img.cols, img2.img.rows)));

        // 2. Draw lines for each match
        for (const auto& match : img1.roi_matches) {
            int r = match.first;
            int c = match.second;

            const roi& roi1 = img1.ROIs[r][c];

            cv::Point2f pt1 = roi1.keypoint.pt;
            cv::Point2f pt2 = roi1.keypoint_match.pt;

            // Adjust pt2 x-coordinate because img2 is drawn to the right of img1
            pt2.x += static_cast<float>(img1.img.cols);

            // Draw keypoints
            cv::circle(output, pt1, 10, cv::Scalar(0, 0, 255), -1);
            cv::circle(output, pt2, 10, cv::Scalar(0, 255, 0), -1);

            // Draw line connecting keypoints
            cv::line(output, pt1, pt2, cv::Scalar(255, 0, 0), 1);
        }

        return output;
    }

} // namespace Utils


#endif // UTILS_HPP
