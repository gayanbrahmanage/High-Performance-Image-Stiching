#include "ImageStitcher.h"

// Custom distance function (you already provided)
inline int DescriptorDistance(const cv::Mat &a, const cv::Mat &b){
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;
    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    }
    return dist;
}

inline void detectorb(const cv::Mat& img, roi& r, cv::Ptr<cv::ORB> orb)
{
    // Detect keypoints and compute descriptors in the ROI patch
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    //std::cout << "Detecting keypoints in the region: " << region << std::endl;
    orb->detectAndCompute(img(r.region), cv::noArray(), keypoints, descriptors);

    if (keypoints.empty()) {
        r.keypoint = cv::KeyPoint(); // empty keypoint
        r.descriptor.release();
        //std::cerr << "No keypoints found in the region: " << region << std::endl;
        return;
    }

    r.valid=true;
    // Find the keypoint with the maximum response
    int bestIdx = 0;
    float maxResponse = keypoints[0].response;
    for (size_t i = 1; i < keypoints.size(); ++i) {
        if (keypoints[i].response > maxResponse) {
            maxResponse = keypoints[i].response;
            bestIdx = static_cast<int>(i);
            
        }
    }

    // Adjust coordinates to full image
    r.keypoint = keypoints[bestIdx];
    r.keypoint.pt.x += r.region.x;
    r.keypoint.pt.y += r.region.y;

    //std::cerr << "keypoints found in the region: " << bestKeypoint.pt.x << "/ "<<bestKeypoint.pt.y<< std::endl;

    // Extract the corresponding descriptor
    r.descriptor = descriptors.row(bestIdx).clone();
}

inline void matchorb(image& img1, image& img2, std::vector<std::pair<int,int>>& roi_matches)
{
    for (int i = 0; i < img1.ROIs.size(); i++) // row in img1
    {
        for (int j = 0; j < img1.ROIs[i].size(); j++) // col in img1
        {
            const roi& r1 = img1.ROIs[i][j];
            if (!r1.valid) continue;

            int bestDist = INT_MAX;
            cv::Point bestMatch(-1, -1);
            int d=1;
            // Search in row i-1, i, i+1 of img2
            for (int di = -d; di <= d; di++)
            {
                int r2row = i + di;
                if (r2row < 0 || r2row >= img2.ROIs.size()) continue;

                for (int jj = 0; jj < j; jj++)
                {
                    const roi& r2 = img2.ROIs[r2row][jj];
                    if (!r2.valid) continue;

                    int dist = DescriptorDistance(r1.descriptor, r2.descriptor);
                    int ORB_HAMMING_DISTANCE_TH=64;

                    if(dist>ORB_HAMMING_DISTANCE_TH)
                    continue;

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestMatch = cv::Point(jj, r2row);
                    }
                }
            }

            if (bestMatch.x >= 0)
            {
                // std::cout << "ROI (" << i << "," << j << ") best match -> ("
                //           << bestMatch.y << "," << bestMatch.x 
                //           << ") with dist=" << bestDist << std::endl;
                img1.ROIs[i][j].keypoint_match = img2.ROIs[bestMatch.y][bestMatch.x].keypoint;
                roi_matches.emplace_back(i, j);
            }
        }
    }
    std::cout << "Found " << roi_matches.size() << " matches." << std::endl;
}

// Compute homography from exactly 4 point correspondences
inline cv::Mat computeHomography4Points(const std::vector<cv::Point2f>& srcPoints,
                                 const std::vector<cv::Point2f>& dstPoints) {
    if (srcPoints.size() != 4 || dstPoints.size() != 4) {
        throw std::runtime_error("Need exactly 4 point correspondences");
    }

    // Construct the linear system A*h = 0
    cv::Mat A(8, 9, CV_64F, cv::Scalar(0));
    for (int i = 0; i < 4; i++) {
        double x = srcPoints[i].x;
        double y = srcPoints[i].y;
        double u = dstPoints[i].x;
        double v = dstPoints[i].y;

        A.at<double>(2*i, 0) = -x;
        A.at<double>(2*i, 1) = -y;
        A.at<double>(2*i, 2) = -1;
        A.at<double>(2*i, 6) = x*u;
        A.at<double>(2*i, 7) = y*u;
        A.at<double>(2*i, 8) = u;

        A.at<double>(2*i+1, 3) = -x;
        A.at<double>(2*i+1, 4) = -y;
        A.at<double>(2*i+1, 5) = -1;
        A.at<double>(2*i+1, 6) = x*v;
        A.at<double>(2*i+1, 7) = y*v;
        A.at<double>(2*i+1, 8) = v;
    }

    // Solve using SVD
    cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat h = svd.vt.row(8).reshape(0, 3);  // last row of V^T

    // Normalize so that H(2,2) = 1
    h /= h.at<double>(2, 2);

    return h;
}

// Compute mean reprojection error between two sets of points given a homography
inline double computeReprojectionError(const image& img1, const cv::Mat& H)
{
    CV_Assert(H.size() == cv::Size(3,3));

    double totalError = 0.0;
    int count = 0;

    for (const auto& match : img1.roi_matches) {
        int r = match.first;
        int c = match.second;

        const roi& roiObj = img1.ROIs[r][c];

        // Project src point using H
        const cv::Point2f& pt1 = roiObj.keypoint.pt;
        const cv::Point2f& pt2 = roiObj.keypoint_match.pt;

        cv::Mat pt1_h = (cv::Mat_<double>(3,1) << pt1.x, pt1.y, 1.0);
        cv::Mat pt2_h = H * pt1_h;

        double x_proj = pt2_h.at<double>(0,0) / pt2_h.at<double>(2,0);
        double y_proj = pt2_h.at<double>(1,0) / pt2_h.at<double>(2,0);

        double dx = x_proj - pt2.x;
        double dy = y_proj - pt2.y;

        totalError += std::sqrt(dx*dx + dy*dy);
        count++;
    }

    if (count == 0) return 0.0;
    return totalError / count; // mean reprojection error
}

inline void ransac(image& img1) {
    cv::RNG rng(cv::getTickCount());

    double bestError = 1e6;
    cv::Mat bestH;

    for (int iter = 0; iter < 100; iter++) {
        // Pick 4 random matches directly from roi_matches
        std::vector<cv::Point2f> srcPoints, dstPoints;
        std::unordered_set<int> usedIdx;

        while ((int)srcPoints.size() < 4) {
            int idx = rng.uniform(0, (int)img1.roi_matches.size());
            if (usedIdx.count(idx)) continue; // avoid duplicates
            usedIdx.insert(idx);

            const auto& match = img1.roi_matches[idx];
            int r = match.first;
            int c = match.second;

            srcPoints.push_back(img1.ROIs[r][c].keypoint.pt);
            dstPoints.push_back(img1.ROIs[r][c].keypoint_match.pt);
        }

        
        // Compute H and reprojection error
        cv::Mat H = computeHomography4Points(srcPoints, dstPoints);
        if (H.empty()) continue;

        // Compute reprojection error
        double error = computeReprojectionError(img1, H);
        // Keep the best H
        if (error < bestError) {
            bestError = error;
            bestH = H.clone();
        }
    }

    std::cout << "Best reprojection error: " << bestError << std::endl;
    if (!bestH.empty()) {
        std::cout << "Best Homography:\n" << bestH << std::endl;
    }

    img1.H = bestH;

}

cv::Mat generatePanorama(const image& img1, const image& img2, cv::Mat& Panorama, cv::Mat& GHomography, int nmax)
{
    CV_Assert(!img1.img.empty() && !img2.img.empty());
    CV_Assert(!img1.H.empty());

    // First image: initialize Panorama and GHomography
    if (img1.id == 1) {
        Panorama = cv::Mat::zeros(img1.img.rows, img1.img.cols * nmax, img1.img.type());
        img1.img.copyTo(Panorama(cv::Rect(0, 0, img1.img.cols, img1.img.rows)));
        GHomography = img1.H.clone();
    } else {
        // Accumulate homography
        GHomography = GHomography * img1.H.inv();
    }

    // Warp img2 corners in Panorama space
    std::vector<cv::Point2f> corners = {
        {0.f, 0.f},
        {static_cast<float>(img2.img.cols), 0.f},
        {static_cast<float>(img2.img.cols), static_cast<float>(img2.img.rows)},
        {0.f, static_cast<float>(img2.img.rows)}
    };
    std::vector<cv::Point2f> warpedCorners(4);
    cv::perspectiveTransform(corners, warpedCorners, GHomography);

    // Compute bounding box
    float minX = 0.f, minY = 0.f;
    float maxX = static_cast<float>(Panorama.cols);
    float maxY = static_cast<float>(Panorama.rows);
    for (auto& pt : warpedCorners) {
        minX = std::min(minX, pt.x);
        minY = std::min(minY, pt.y);
        maxX = std::max(maxX, pt.x);
        maxY = std::max(maxY, pt.y);
    }

    int xOffset = static_cast<int>(-std::floor(minX));
    int yOffset = static_cast<int>(-std::floor(minY));
    int newWidth = static_cast<int>(std::ceil(maxX)) + xOffset;
    int newHeight = static_cast<int>(std::ceil(maxY)) + yOffset;

    // Expand Panorama if needed
    if (newWidth > Panorama.cols || newHeight > Panorama.rows) {
        cv::Mat newPanorama = cv::Mat::zeros(newHeight, newWidth, Panorama.type());
        Panorama.copyTo(newPanorama(cv::Rect(xOffset, yOffset, Panorama.cols, Panorama.rows)));
        Panorama = newPanorama;
    }

    // Translate homography to fit new panorama
    cv::Mat translation = (cv::Mat_<double>(3,3) << 1,0,xOffset, 0,1,yOffset, 0,0,1);
    cv::Mat H_translated = translation * GHomography;

    // Warp img2 safely
    cv::Mat warped;
    cv::warpPerspective(img2.img, warped, H_translated, Panorama.size());

    // Copy warped img2 into Panorama safely
    cv::Mat roi = Panorama(cv::Rect(0, 0, std::min(warped.cols, Panorama.cols), std::min(warped.rows, Panorama.rows)));
    cv::Mat warpedROI = warped(cv::Rect(0, 0, roi.cols, roi.rows));
    //warpedROI.copyTo(roi);

    return Panorama;
}




// cv::Mat generatePanorama(const image& img1, const image& img2, cv::Mat& Panorama, cv::Mat& GHomography)
// {
//     CV_Assert(!img1.img.empty() && !img2.img.empty());
//     CV_Assert(!img1.H.empty());

//     // Inverse homography maps img2 → img1
//     cv::Mat H_inv = img1.H.inv();
//     GHomography = GHomography * H_inv;

//     // Warp corners of img2 to img1 coordinate space
//     std::vector<cv::Point2f> cornersImg2 = {
//         {0,0},
//         {static_cast<float>(img2.img.cols),0},
//         {static_cast<float>(img2.img.cols),static_cast<float>(img2.img.rows)},
//         {0, static_cast<float>(img2.img.rows)}
//     };
//     std::vector<cv::Point2f> warpedCorners(4);
//     cv::perspectiveTransform(cornersImg2, warpedCorners, H_inv);

//     // Compute bounding box of panorama
//     float minX = 0, minY = 0;
//     float maxX = static_cast<float>(img1.img.cols);
//     float maxY = static_cast<float>(img1.img.rows);

//     for (const auto& pt : warpedCorners) {
//         minX = std::min(minX, pt.x);
//         minY = std::min(minY, pt.y);
//         maxX = std::max(maxX, pt.x);
//         maxY = std::max(maxY, pt.y);
//     }

//     // Translation to handle negative coordinates
//     cv::Mat translation = (cv::Mat_<double>(3,3) << 1,0,-minX, 0,1,-minY, 0,0,1);
//     cv::Mat H_translated = translation * H_inv;

//     int panoWidth = static_cast<int>(maxX - minX);
//     int panoHeight = static_cast<int>(maxY - minY);

//     // Warp img2 into panorama
//     cv::Mat panorama;
//     cv::warpPerspective(img2.img, panorama, H_translated, cv::Size(panoWidth, panoHeight));

//     // Copy img1 into panorama
//     cv::Mat roi = panorama(cv::Rect(static_cast<int>(-minX), static_cast<int>(-minY), img1.img.cols, img1.img.rows));
//     img1.img.copyTo(roi);

//     return panorama;
// }



roi::roi(){

}

roi::~roi(){

}

image::image(){

}

image::~image(){

}

void image::set(const std::string& image_name) {
    img = cv::imread(image_name);
    if (img.empty()) {
        throw std::runtime_error("Could not open or find the image");
    }

    // initialization
    width = img.cols;
    height = img.rows;
    roi_col_count = (width + roi_width - 1) / roi_width;
    roi_row_count = (height + roi_height - 1) / roi_height;

    std::cout << "Image loaded: " << image_name << std::endl;
    std::cout << "Width: " << width << ", Height: " << height << std::endl;
    std::cout << "ROI Count (Cols x Rows): " << roi_col_count << " x " << roi_row_count << std::endl;

            // Allocate the grid
    ROIs.resize(roi_row_count, std::vector<roi>(roi_col_count));

    for (int r = 0; r < roi_row_count; ++r) {
        for (int c = 0; c < roi_col_count; ++c) {
            int x = c * roi_width;
            int y = r * roi_height;

            // Ensure ROI stays inside image
            int w = std::min(roi_width, width - x);
            int h = std::min(roi_height, height - y);
            roi new_roi;
            new_roi.region = cv::Rect(x, y, w, h);
            ROIs[r][c] = new_roi;
        }
    }

    // on-the-fly process
    float percentatge=0.5;
    for(int i=ROIs.size()*(0.5-percentatge); i<ROIs.size()*(0.5+percentatge); i++){
        for(int j=0; j<ROIs[i].size(); j++){
            detectorb(img, ROIs[i][j], orb);
        }
    }

    //
}

void image::reset(){
    roi_matches.clear();
    id=-1;
}

// Constructor
ImageStitcher::ImageStitcher() {
    // Empty for now
}

// Destructor
ImageStitcher::~ImageStitcher() {
    // Empty for now
}

// Add an image to the stitcher
void ImageStitcher::addImage(const std::string& image_name) {
    if(img1.id==-1){
        img1.set(image_name);
        n_images++;
        img1.id = n_images;

    }else if(img2.id==-1){
        img2.set(image_name);
        n_images++;
        img2.id = n_images;
    }

    if(n_images >= 2) {
        if(n_images%2==0){
            matchorb(img1, img2, img1.roi_matches);
            ransac(img1);
            cv::Mat panorama = generatePanorama(img1, img2, Panorama, GHomography, nmax);
            img1.id=-1;
            cv::imwrite("../../outputs/panorama.jpg", panorama);
        }else{

        }

    }
}

